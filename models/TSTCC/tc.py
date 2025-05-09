import os
import json
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer

from .attention import Seq_Transformer
from .loss import NTXentLoss
from .model import BaseModel



class TC(nn.Module):
    def __init__(self, out_dim, hidden_dim, timesteps=1, device="cuda"):
        '''
        
        Args:
            in_dims (int): The input dimension. For a univariate time series, this should be set to 1.
            out_dim (int): The representation dimension.
            hidden_dim (int): Dimension of the hidden layers.
            timesteps (int): Number of timesteps to be predicted using the context vector. Default = 1.
            device (str): Device to run the model. Default = 'cuda'.
        '''
        super(TC, self).__init__()
        self.num_channels = out_dim
        self.timestep = timesteps
        self.Wk = nn.ModuleList([nn.Linear(hidden_dim, self.num_channels) for _ in range(self.timestep)])
        self.lsoftmax = nn.LogSoftmax(dim=1) # fix: required the 'dim' param
        self.device = device
        
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dim, out_dim // 2),
            nn.BatchNorm1d(out_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim // 2, out_dim // 4),
        )

        self.seq_transformer = Seq_Transformer(patch_size=self.num_channels, dim=hidden_dim, depth=4, heads=4, mlp_dim=64)


    def forward(self, features_aug1, features_aug2):
        '''Compute the temporal contrastive loss and the projection of `features_aug1`'''
        z_aug1 = features_aug1  # features are (batch_size, #channels, seq_len)
        seq_len = z_aug1.shape[2]
        z_aug1 = z_aug1.transpose(1, 2)

        z_aug2 = features_aug2
        z_aug2 = z_aug2.transpose(1, 2)

        batch = z_aug1.shape[0]
        t_samples = torch.randint(seq_len - self.timestep, size=(1,)).long().to(self.device)  # randomly pick time stamps

        nce = 0  # average over timestep and batch
        encode_samples = torch.empty((self.timestep, batch, self.num_channels)).float().to(self.device)

        for i in np.arange(1, self.timestep + 1):
            encode_samples[i - 1] = z_aug2[:, t_samples + i, :].view(batch, self.num_channels)
        forward_seq = z_aug1[:, :t_samples + 1, :]

        c_t = self.seq_transformer(forward_seq)

        pred = torch.empty((self.timestep, batch, self.num_channels)).float().to(self.device)
        for i in np.arange(0, self.timestep):
            linear = self.Wk[i]
            pred[i] = linear(c_t)
        for i in np.arange(0, self.timestep):
            total = torch.mm(encode_samples[i], torch.transpose(pred[i], 0, 1))
            nce += torch.sum(torch.diag(self.lsoftmax(total)))
        nce /= -1. * batch * self.timestep
        return nce, self.projection_head(c_t)
    

    def fit_ssl(self, 
                train_loader: DataLoader, 
                val_loader: DataLoader, 
                tc_optim: Optimizer, 
                encoder: BaseModel, 
                encoder_optim: Optimizer, 
                config: dict, 
                cc_temp: float = 0.2, 
                use_cosine: bool = True,
                resume_train: bool = False
                ) -> tuple[dict, dict]:
        '''Fit the TS-TCC model in the pretext task.
        
        Args:
            encoder: Feature Extractor (Encoder) model.
            encoder_optim (Optimizer): Optimizer of the ``encoder``.
            tc_optim (Optimizer): Optimizer of the Temporal Contrast (TC) model.
            train_dataset (Load_Dataset): Augmented version of the train dataset.
            val_dataset (Load_Dataset): Augmented version of the validation dataset.
            n_epochs (int): Number of epochs to train the model.
            batch_size (int): Batch's size.
            cc_temp (float): Temperature of the Contextual Constrasting (CC) loss function. Default = 0.2
            use_cosine (bool): If True use the cosine similarity to compute the CC loss function. Default = True.
            save_best (bool): If True is returned the best model (based on validation loss) istead of the last epoch model. Default = False. 
            save_temp (bool): Save the partially trained model after each epoch. Default = True.

        Returns:
            pretrained model params (dict), losses (dict)
        '''
        # create the dir to save the checkpoints
        checkpoint_path = f"{config['save_dir']}/{config['problem']}_pretrained_TSTCC_last.pth"
        if not os.path.isdir(config['save_dir']):
            os.makedirs(config['save_dir'])

        # Resume the training stoped
        if resume_train:
            checkpoint = torch.load(checkpoint_path, weights_only=True)
            encoder.load_state_dict(checkpoint['encoder'])
            self.load_state_dict(checkpoint['tc_model']) 

        # send model to device
        encoder.to(self.device)
        self.to(self.device)

        cc_criterion = NTXentLoss(self.device, config['batch_size'], 
                                  temperature=cc_temp, 
                                  use_cosine_similarity=use_cosine)

        train_losses = []
        val_losses = []
        repr_shape = None

        best_encode_params = None
        best_epoch = None
        best_loss = torch.inf

        if resume_train: # start from the next epoch
            start_epoch = checkpoint['epoch'] + 1
        else: # start from the first epoch
            start_epoch = 0 
        
        for i in range(start_epoch, config['epochs']):

            # -------------- Trainig phase -------------- #
            encoder.train()
            self.train()

            running_loss = []

            batch_iter = tqdm(train_loader, desc=f"{i+1:2.0f}/{config['epochs']}", total=len(train_loader))
            for (data, labels, aug1, aug2) in batch_iter:
                # send to device
                data, labels = data.float().to(self.device), labels.long().to(self.device)
                aug1, aug2 = aug1.float().to(self.device), aug2.float().to(self.device)

                # optimizer
                encoder_optim.zero_grad()
                tc_optim.zero_grad()

                features1 = encoder(aug1) # weak augmented
                features2 = encoder(aug2) # strong augmented

                # normalize projection feature vectors
                features1 = F.normalize(features1, dim=1)
                features2 = F.normalize(features2, dim=1)

                # compute loss [temporal contrasting]
                tc_loss1, tc_feat1 = self.forward(features1, features2)
                tc_loss2, tc_feat2 = self.forward(features2, features1)

                # compute loss [contextual contrasting]
                cc_loss = cc_criterion(tc_feat1, tc_feat2)
                
                # compute overall loss [temporal + contextual]
                lambda1 = 1
                lambda2 = 0.7
                loss = lambda1 * (tc_loss1 + tc_loss2) + cc_loss * lambda2
                loss.backward()

                encoder_optim.step()
                tc_optim.step()
                
                running_loss.append(loss.item())

            train_loss = sum(running_loss) / len(running_loss)
            train_losses.append(train_loss)


            # -------------- Validation phase -------------- #
            encoder.eval()
            self.eval()

            running_loss = []

            with torch.no_grad():
                for (data, labels, aug1, aug2) in val_loader:
                    # send to device
                    data, labels = data.float().to(self.device), labels.long().to(self.device)
                    aug1, aug2 = aug1.float().to(self.device), aug2.float().to(self.device)

                    features1 = encoder(aug1) # weak augmented
                    features2 = encoder(aug2) # strong augmented

                    # shape of the representation without batch dimension
                    if repr_shape is None:
                        repr_shape = features1.shape[1:]

                    # normalize projection feature vectors
                    features1 = F.normalize(features1, dim=1)
                    features2 = F.normalize(features2, dim=1)

                    # compute loss [temporal contrasting]
                    tc_loss1, tc_feat1 = self.forward(features1, features2)
                    tc_loss2, tc_feat2 = self.forward(features2, features1)

                    # compute loss [contextual contrasting]
                    cc_loss = cc_criterion(tc_feat1, tc_feat2)
                    
                    # compute overall loss [temporal + contextual]
                    lambda1 = 1
                    lambda2 = 0.7
                    loss = lambda1 * (tc_loss1 + tc_loss2) + cc_loss * lambda2

                    running_loss.append(loss.item())

            val_loss = sum(running_loss) / len(running_loss)
            val_losses.append(val_loss)


            # -------------- Save models parameters -------------- #
            # TODO: Save the optimizer states
            checkpoint = {
                'encoder': encoder.state_dict(),
                'encoder_optim': encoder_optim.state_dict(),
                'tc_model': self.state_dict(),
                'tc_optim': tc_optim.state_dict(),
                'epoch': i
            }
            torch.save(checkpoint, checkpoint_path)

            # save the best model
            if val_loss < best_loss:
                best_loss = val_loss
                best_epoch = i
                
                torch.save(checkpoint, f"{config['save_dir']}/{config['problem']}_pretrained_TSTCC_best.pth")
            

            # -------------- Print and save the losses -------------- #
            print(f"[Train {train_loss:.4f} | Val {val_loss:.4f}]")
            json.dump(obj={'train_losses': train_losses, 'val_losses': val_losses},
                      fp=open(f"{config['save_dir']}/{config['problem']}_TSTCC_losses.json", 'w'))


        # return models to cpu
        encoder.cpu()
        self.cpu()

        logs = {
            'train_losses': train_losses, 
            'val_losses': val_losses,
            'repr_shape': repr_shape,
            'best_epoch': best_epoch,
        }

        return best_encode_params, logs


