import os
import json
import numpy as np
from tqdm import tqdm
from typing import Optional

import torch
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.utils.data import TensorDataset, DataLoader

from .encoder import TSEncoder
from .losses import hierarchical_contrastive_loss
from .utils import take_per_row, torch_pad_nan


class TS2Vec:
    '''The TS2Vec model'''
    
    def __init__(
        self,
        in_dim,
        out_dim=320,
        hidden_dim=64,
        depth=10,
        device='cuda',
        mask_mode='binomial',
        kernel_size=3,
        dropout=0.1,
        after_iter_callback=None,
        after_epoch_callback=None
    ):
        ''' Initialize a TS2Vec model.
        
        Args:
            in_dim (int): The input dimension. For a univariate time series, this should be set to 1.
            out_dim (int): The representation dimension.
            hidden_dim (int): The hidden dimension of the encoder.
            depth (int): The number of hidden residual blocks in the encoder.
            device (int): The gpu used for training and inference.
            mask_mode (str): Type of mask used by the encoder ['binomial', 'continuous', 'all_true', 'all_false', 'mask_last']. Default = 'binomial'.
            after_iter_callback (Union[Callable, NoneType]): A callback function that would be called after each iteration.
            after_epoch_callback (Union[Callable, NoneType]): A callback function that would be called after each epoch.
        '''
        
        super().__init__()
        self.device = device
        
        self._net = TSEncoder(input_dims=in_dim, output_dims=out_dim, hidden_dims=hidden_dim, 
                              depth=depth, kernel_size=kernel_size, dropout=dropout, 
                              mask_mode=mask_mode)
        self.net = torch.optim.swa_utils.AveragedModel(self._net)
        self.net.update_parameters(self._net)
        
        self.after_iter_callback = after_iter_callback
        self.after_epoch_callback = after_epoch_callback
        
        self.n_epochs = 0
        self.n_iters = 0
    

    def to(self, device):
        '''Move both model and averaged model to `device`.'''
        self.net.to(device)
        self._net.to(device)

    def cpu(self):
        '''Move both model and averaged model to `cpu`.'''
        self.net.cpu()
        self._net.cpu()

    
    def train(self, mode: bool=True):
        '''Set both model and averaged model in training mode.'''
        self.net.train(mode)
        self._net.train(mode)


    def eval(self):
        '''Set both model and averaged model in evaluation mode.'''
        self.net.eval()
        self._net.eval()


    def tuning_mode(self, mode: str):
        '''Set the tuning mode.
        
        Args:
            mode (str): Set the tuning mode `linear-probing` | `fine-tuning`.
        '''
        if mode == 'linear-probing': # freeze model
            self._net.requires_grad_(False)
            self.net.requires_grad_(False)


    def swa_update_params(self):
        '''Update the averaged model parameters.'''
        self.net.update_parameters(self._net)

    def state_dict(self):
        return self.net.state_dict()
    
    def parameters(self):
        return self.net.parameters()


    def fit_ssl(self, 
                train_loader: DataLoader, 
                val_loader: DataLoader, 
                optimizer: Optimizer, 
                config: dict,
                temporal_unit: Optional[int] = 0, 
                resume_train: Optional[bool] = False,
                ):
        ''' Training the TS2Vec model.
        
        Args:
            train_loader (DataLoader): The training data. It should have a shape of (n_instance, n_timestamps, n_features). All missing data should be set to NaN.
            val_loader (DataLoader): The validation data. It should have a shape of (n_instance, n_timestamps, n_features).
            optimizer (Optimizer): Model optimizer.
            config (dict): The training configuration. It should contain the following keys: 'save_dir', 'problem', 'epochs'.
            temporal_unit (Optional, int): The minimum unit to perform temporal contrast. When training on a very long sequence, this param helps to reduce the cost of time and memory.
            resume_train (Optional, bool): Resume the training from the last checkpoint. Default = False.

        Returns:
            best_model_param (dict): a dict containing the best model parameters.
            losses (dict): a dict containing the losses on each epoch.
        '''
        self.to(self.device)

        # create the dir to save the checkpoints
        checkpoint_path = f"{config['save_dir']}/{config['problem']}_pretrained_TS2Vec_last.pth"
        if not os.path.isdir(config['save_dir']):
            os.makedirs(config['save_dir'])

        # Resume the training stoped
        if resume_train:
            checkpoint = torch.load(checkpoint_path, weights_only=True)
            self.load_state_dict(checkpoint['state_dict']) 

        train_losses = []
        val_losses = []

        best_loss = torch.inf
        best_epoch = None
        best_model_params = None
        
        start_epoch = 0 if resume_train is False else checkpoint['epoch']-1
        for i in range(start_epoch, config['epochs']):

            # ---------------- Training phase ---------------- #
            self.train()
            running_loss = 0

            batch_iter = tqdm(train_loader, desc=f"{i+1:2.0f}/{config['epochs']}", total=len(train_loader))
            for batch in batch_iter:
                x = batch[0].to(self.device)
                # if max_train_length is not None and x.size(1) > max_train_length:
                #     window_offset = np.random.randint(x.size(1) - max_train_length + 1)
                #     x = x[:, window_offset : window_offset + max_train_length]
                
                optimizer.zero_grad()

                crop_l = np.random.randint(low=2 ** (temporal_unit + 1), high=x.size(1)+1)
                view_a, view_b = self._extract_context_view(x, crop_l)

                # encode the views
                view_a = self._net(view_a)
                view_a = view_a[:, -crop_l:]
                
                view_b = self._net(view_b)
                view_b = view_b[:, :crop_l]
                
                # compute loss and update the model params
                loss = hierarchical_contrastive_loss(view_a, view_b, temporal_unit=temporal_unit)
                loss.backward()
                optimizer.step()
                self.swa_update_params()
                    
                running_loss += loss.item()
                
                if self.after_iter_callback is not None:
                    self.after_iter_callback(self, loss.item())
            

            running_loss /= len(train_loader)
            train_losses.append(running_loss)

            # ---------------- Evaluation phase ---------------- #
            self.eval()
            running_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    x = batch[0].to(self.device)

                    crop_l = np.random.randint(low=2 ** (temporal_unit + 1), high=x.size(1)+1)
                    view_a, view_b = self._extract_context_view(x, crop_l)

                    # encode the views
                    view_a = self.net(view_a)
                    view_a = view_a[:, -crop_l:]

                    view_b = self.net(view_b)
                    view_b = view_b[:, :crop_l]

                    loss = hierarchical_contrastive_loss(view_a, view_b, temporal_unit=temporal_unit)
                    running_loss += loss.item()


            # -------------- Save checkpoint -------------- #
            checkpoint = {'state_dict': self.net.state_dict(), 'epoch': i+1}
            torch.save(checkpoint, checkpoint_path)

            running_loss /= len(val_loader)
            if running_loss < best_loss:
                best_loss = running_loss
                best_model_params = self.net.state_dict()
                best_epoch = i+1

                checkpoint = {'state_dict': best_model_params, 'epoch': best_epoch}
                torch.save(checkpoint, f"{config['save_dir']}/{config['problem']}_pretrained_TS2Vec_best.pth")

            val_losses.append(running_loss)

            print(f"[Train: {train_losses[-1]:.4f} | Val: {val_losses[-1]:.4f}]")
            json.dump(obj={'train_losses': train_losses, 'val_losses': val_losses},
                      fp=open(f"{config['save_dir']}/{config['problem']}_TS2Ve_losses.json", 'w'))

            if self.after_epoch_callback is not None:
                self.after_epoch_callback(self, running_loss)
        
        self.cpu()
        logs = {
            'train_losses': train_losses, 
            'val_losses': val_losses,
            'best_epoch': best_epoch
        }

        return best_model_params, logs


    def _extract_context_view(self, x, crop_len):
        ts_l = x.size(1)
        
        crop_left = np.random.randint(ts_l - crop_len + 1)
        crop_right = crop_left + crop_len
        crop_eleft = np.random.randint(crop_left + 1)
        crop_eright = np.random.randint(low=crop_right, high=ts_l + 1)
        crop_offset = np.random.randint(low=-crop_eleft, high=ts_l - crop_eright + 1, size=x.size(0))

        view_a = take_per_row(x, crop_offset + crop_eleft, crop_right - crop_eleft)
        view_b = take_per_row(x, crop_offset + crop_left, crop_eright - crop_left)

        return view_a, view_b


    def _eval_with_pooling(self, x: torch.Tensor, mask=None, slicing=None, encoding_window=None):
        ''' Compute representations using the model.
        Args:
            x (torch.Tensor): The input data. It should have a shape of (n_instance, n_timestamps, n_features). No missing data should be passed.
            mask (str): The mask used by encoder can be specified with this parameter. This can be set to 'binomial', 'continuous', 'all_true', 'all_false' or 'mask_last'.
            slicing (slice): The slice used to slice the output.
            encoding_window (Union[str, int]): When this param is specified, the computed representation would the max pooling over this window. This can be set to 'full_series', 'multiscale' or an integer specifying the pooling kernel size.

        Returns:
            out: The representations for data.
        '''
        z1 = self.net(x, mask)
        # max pooling over the termporal dimension
        embedding = F.adaptive_max_pool1d(z1.transpose(1, 2), 1).transpose(1, 2)

        return embedding
    

    def linear_prob(self, data, rearrange=True):
        '''Compute representations using the model

        Args:
            data (numpy.ndarray): Data to be encoded.
            rearrange (bool): Set this to True if the input data is in (n_instance, n_features, n_timestamps) format. Otherwise, set this to False.

        Returns:
            repr: The representations for data with the `data` original dimension order.
        '''

        if rearrange: # (B, C, T) -> (B, T, C)
            data = torch.permute(data, (0, 2, 1)) 
            
        z1 = self.net(data)
        # max pooling over the termporal dimension
        repr = F.adaptive_max_pool1d(z1.transpose(1, 2), 1).transpose(1, 2)

        if rearrange: # (B, T, C) -> (B, C, T) 
            repr = torch.permute(repr, (0, 2, 1)) 
        
        return repr.squeeze()
    

    def _encode(self, data, mask=None, encoding_window=None, causal=False, sliding_length=None, sliding_padding=0, batch_size=None):
        ''' Compute representations using the model.
        
        Args:
            data (numpy.ndarray): This should have a shape of (n_instance, n_timestamps, n_features). All missing data should be set to NaN.
            mask (str): The mask used by encoder can be specified with this parameter. This can be set to 'binomial', 'continuous', 'all_true', 'all_false' or 'mask_last'.
            encoding_window (Union[str, int]): When this param is specified, the computed representation would the max pooling over this window. This can be set to 'full_series', 'multiscale' or an integer specifying the pooling kernel size.
            causal (bool): When this param is set to True, the future informations would not be encoded into representation of each timestamp.
            sliding_length (Union[int, NoneType]): The length of sliding window. When this param is specified, a sliding inference would be applied on the time series.
            sliding_padding (int): This param specifies the contextual data length used for inference every sliding windows.
            batch_size (Union[int, NoneType]): The batch size used for inference. If not specified, this would be the data length.
            
        Returns:
            repr: The representations for data.
        '''
        self.to(self.device)

        assert self.net is not None, 'please train or load a net first'
        assert data.ndim == 3
        if batch_size is None:
            batch_size = len(data)
        n_samples, ts_l, _ = data.shape

        org_training = self.net.training
        self.net.eval()
        
        dataset = TensorDataset(torch.from_numpy(data).to(torch.float))
        loader = DataLoader(dataset, batch_size=batch_size)
        
        with torch.no_grad():
            output = []
            for batch in loader:
                x = batch[0]
                if sliding_length is not None:
                    reprs = []
                    if n_samples < batch_size:
                        calc_buffer = []
                        calc_buffer_l = 0
                    for i in range(0, ts_l, sliding_length):
                        l = i - sliding_padding
                        r = i + sliding_length + (sliding_padding if not causal else 0)
                        x_sliding = torch_pad_nan(
                            x[:, max(l, 0) : min(r, ts_l)],
                            left=-l if l<0 else 0,
                            right=r-ts_l if r>ts_l else 0,
                            dim=1
                        )
                        if n_samples < batch_size:
                            if calc_buffer_l + n_samples > batch_size:
                                out = self._eval_with_pooling(
                                    torch.cat(calc_buffer, dim=0),
                                    mask,
                                    slicing=slice(sliding_padding, sliding_padding+sliding_length),
                                    encoding_window=encoding_window
                                )
                                reprs += torch.split(out, n_samples)
                                calc_buffer = []
                                calc_buffer_l = 0
                            calc_buffer.append(x_sliding)
                            calc_buffer_l += n_samples
                        else:
                            out = self._eval_with_pooling(
                                x_sliding,
                                mask,
                                slicing=slice(sliding_padding, sliding_padding+sliding_length),
                                encoding_window=encoding_window
                            )
                            reprs.append(out)

                    if n_samples < batch_size:
                        if calc_buffer_l > 0:
                            out = self._eval_with_pooling(
                                torch.cat(calc_buffer, dim=0),
                                mask,
                                slicing=slice(sliding_padding, sliding_padding+sliding_length),
                                encoding_window=encoding_window
                            )
                            reprs += torch.split(out, n_samples)
                            calc_buffer = []
                            calc_buffer_l = 0
                    
                    out = torch.cat(reprs, dim=1)
                    if encoding_window == 'full_series':
                        out = F.max_pool1d(
                            out.transpose(1, 2).contiguous(),
                            kernel_size = out.size(1),
                        ).squeeze(1)
                else:
                    out = self._eval_with_pooling(x, mask, encoding_window=encoding_window)
                    if encoding_window == 'full_series':
                        out = out.squeeze(1)
                        
                output.append(out)
                
            output = torch.cat(output, dim=0)
            

        self.cpu()
        self.net.train(org_training)
        return output.numpy()
    

    def save(self, fn):
        ''' Save the model to a file.
        
        Args:
            fn (str): filename.
        '''
        torch.save(self.net.state_dict(), fn)
    
    
    def load(self, fn):
        ''' Load the model from a file.
        
        Args:
            fn (str): filename.
        '''
        state_dict = torch.load(fn, map_location=self.device, weights_only=True)
        self.net.load_state_dict(state_dict)
    

    def load_state_dict(self, state_dict, **kwargs):
        self.net.load_state_dict(state_dict, **kwargs)