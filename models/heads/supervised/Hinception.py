import torch
from torch import nn
from typing import Tuple
import numpy as np
from torch.nn import functional as F
import wandb
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from mine_utils import set_seed

def noop(x: torch.Tensor) -> torch.Tensor:
    return x

class InceptionModule(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Tuple[int, int] = [40, 20, 10],
                 bottleneck: bool = True) -> None:
        super().__init__()
        
        self.kernel_sizes = kernel_size
        bottleneck = bottleneck if in_channels > 1 else False
        self.bottleneck = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False, padding='same') if bottleneck else noop
        
        self.convolutions = nn.ModuleList([
            nn.Conv1d(out_channels if bottleneck else in_channels,
                      out_channels,
                      kernel_size=k,
                      padding='same',
                      bias=False) for k in self.kernel_sizes
        ])
        self.maxconv = nn.Sequential(*[nn.MaxPool1d(3, stride=1, padding=1),
                                       nn.Conv1d(in_channels, out_channels, kernel_size=1, padding='same', bias=False)])
        self.batchnorm = nn.BatchNorm1d(out_channels * 4)
        self.activation = nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_ = x
        x = self.bottleneck(x)
        x = torch.cat([conv(x) for conv in self.convolutions] + [self.maxconv(x_)], dim=1)
        return self.activation(x)


class Hinception(nn.Module):

    def __init__(self, sequence_length, in_channels, num_classes, hname = 'H1') -> None:
        super().__init__()
        self.sequence_length = sequence_length
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.hname = hname

        custom_kernels_sizes = [2, 4, 8, 16, 32, 64]
        custom_convolutions = []
        
        for ks in custom_kernels_sizes:
            filter_ = np.ones(shape=(1, self.in_channels, ks))
            indices_ = np.arange(ks)

            filter_[:, :, indices_ % 2 == 0] *= -1 # increasing detection filter

            custom_conv = nn.Conv1d(in_channels=self.in_channels, out_channels=1,
                                    kernel_size=ks, bias=False, padding='same')
            custom_conv.weight = nn.Parameter(torch.from_numpy(filter_).float())
            
            for param in custom_conv.parameters():
                param.requires_grad = False
            custom_convolutions.append(custom_conv)

        for ks in custom_kernels_sizes:
            filter_ = np.ones(shape=(1, self.in_channels, ks))
            indices_ = np.arange(ks)
            
            filter_[:,:, indices_ % 2 > 0] *= -1 # decreasing detection filter
            
            custom_conv = nn.Conv1d(in_channels=self.in_channels, out_channels=1,
                                    kernel_size=ks, bias=False, padding='same')
            
            custom_conv.weight = nn.Parameter(torch.from_numpy(filter_).float())
            for param in custom_conv.parameters():
                param.requires_grad = False
            custom_convolutions.append(custom_conv)

        for ks in [6, 12, 24, 48, 96]:
            filter_ = np.zeros(shape=(1, self.in_channels, ks + ks // 2))
            x_mesh = np.linspace(start=0, stop=1, num=ks//4 + 1)[1:].reshape((-1, 1, 1))
            
            filter_left = x_mesh ** 2
            filter_right = filter_left[::-1]
            
            filter_left = np.transpose(filter_left, (1, 2, 0))
            filter_right = np.transpose(filter_right, (1, 2, 0))
            
            filter_[:, :, 0:ks//4] = -filter_left
            filter_[:, :, ks//4:ks//2] = -filter_right
            filter_[:, :, ks//2:3*ks//4] = 2 * filter_left
            filter_[:, :, 3*ks//4:ks] = 2 * filter_right
            filter_[:, :, ks:5*ks//4] = -filter_left
            filter_[:, :, 5*ks//4:] = -filter_right
            
            custom_conv = nn.Conv1d(in_channels=self.in_channels, out_channels=1,
                                    kernel_size=ks, bias=False, padding='same')
            custom_conv.weight = nn.Parameter(torch.from_numpy(filter_).float())

            for param in custom_conv.parameters():
                param.requires_grad = False
            custom_convolutions.append(custom_conv)

        self.custom_convolutions = nn.ModuleList(custom_convolutions)
        self.custom_activation = nn.ReLU()

        self.inception_module_1 = InceptionModule(in_channels=1, out_channels=32)
        self.inception_module_2 = InceptionModule(in_channels=145, out_channels=32)
        self.inception_module_3 = InceptionModule(in_channels=32 * 4, out_channels=32)

        self.inception_module_4 = InceptionModule(in_channels=32 * 4, out_channels=32)
        self.inception_module_5 = InceptionModule(in_channels=32 * 4, out_channels=32)
        self.inception_module_6 = InceptionModule(in_channels=32 * 4, out_channels=32)

        self.linear = nn.Linear(in_features=32 * 4, out_features=num_classes)
        self.criteria = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_ = x
        custom_feature_maps = torch.concat([custom_conv(x) for custom_conv in self.custom_convolutions], dim=1) # Concatenate channel-wise
        custom_feature_maps = self.custom_activation(custom_feature_maps)

        feature_maps_1 = torch.cat([self.inception_module_1(x), custom_feature_maps], dim=1)
        feature_maps_2 = self.inception_module_2(feature_maps_1)
        feature_maps_3 = self.inception_module_3(feature_maps_2)

        feature_maps_3_ = feature_maps_3 + x_ # First Residual

        feature_maps_4 = self.inception_module_4(feature_maps_3_)
        feature_maps_5 = self.inception_module_5(feature_maps_4)
        feature_maps_6 = self.inception_module_6(feature_maps_5)

        feature_maps_6_ = feature_maps_6 + feature_maps_3_ # Second Residual
        feature_maps = torch.mean(feature_maps_6_, dim=-1)
        return self.linear(feature_maps)
    

    def to_train(self, train_loader, loss, opt, sched, device):
        self.to(device)
        self.train() 
        running_loss = 0.0
        total_correct = 0
        total_samples = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            opt.zero_grad()
            logits = self.forward(x)

            loss_value = loss(logits, y)
            loss_value.backward()
            opt.step()

            running_loss += loss_value.item() 
            probs = F.softmax(logits, dim=1) 
            argmax = torch.argmax(probs, dim=1)
            total_correct += (argmax == y).sum().item()
            total_samples += x.size(0)

        avg_loss = running_loss / total_samples
        accuracy = total_correct / total_samples
        sched.step(running_loss / len(train_loader))
        return avg_loss, accuracy

    @torch.no_grad()
    def evaluate(self, val_loader, loss, device):
        self.to(device)
        self.eval() 
        total_loss = 0.0
        total_samples = 0
        all_preds = []
        all_labels = []

        for batch in val_loader:
            x, y = batch
            x, y = x.to(device), y.to(device)

            logits = self.forward(x)
            loss_value = loss(logits, y)

            total_loss += loss_value.item() * x.size(0)
            probs = F.softmax(logits, dim=-1) 
            argmax = torch.argmax(probs, dim=1)
            total_samples += x.size(0)

            all_preds.extend(argmax.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

        avg_val_loss = total_loss / total_samples
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        return  avg_val_loss, acc, f1
    
    

class HinceptionTime(nn.Module):
    def __init__(self,  configs: dict, sequence_length: int, in_channels: int, num_classes: int, num_models: int = 5, seed = 42) -> None:
        super().__init__()
        self.num_models = num_models
        self.seed = seed
        self.models = nn.ModuleList([
            Hinception(sequence_length=sequence_length, in_channels=in_channels, num_classes=num_classes, hname=f'H{idx+1}')
            for idx in range(self.num_models)
        ])

        self.setup(configs)

    def setup(self, configs: dict) -> None:
        self.losses = [configs['loss'] for _ in range(self.num_models)]
        self.opts   = [configs['optimizer'] for _ in range(self.num_models)]
        self.schedulers = [configs['scheduler'] for _ in range(self.num_models)]
        
        self.set_optimizers(configs['lr'])
        self.set_schedulers(configs['sched_params'])

    def set_optimizers(self, lr) -> None:
        self.opts = [opt(model.parameters(), lr, weight_decay=0) for model, opt in zip(self.models, self.opts)]

    def set_schedulers(self, params) -> None:
        self.schedulers = [sched(opt, **params) for opt, sched in zip(self.opts, self.schedulers)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = [model(x) for model in self.models]  # list of tensors [B, C]
        logits = torch.stack(outputs, dim=1)         # [B, N, C]
        probs  = F.softmax(logits, dim=-1)           # (B, N, C)    [32, 5, 6]
        prob_mean = torch.mean(probs, dim=1)         # (B, 1, C)    [32, 1, 6]
        argmax = torch.argmax(prob_mean, dim=1)  
        return argmax
    
    def run(self, train_loader, val_loader, epochs, device, run_idx=0, path=''):
            set_seed(self.seed + 1)
            for idx, (inception, loss, opt, sched) in enumerate(zip(self.models, self.losses, self.opts, self.schedulers)):
                min_val_loss = float('inf')
                print(inception.__class__.__name__)
                print(f'\ncurrent Hinception model: {inception.hname}\n')
                for epoch in range(epochs):
                    train_loss, train_acc = inception.to_train(train_loader, loss, opt, sched, device)
                    val_loss, val_acc, val_f1 = inception.evaluate(val_loader, loss, device)
                    if val_loss < min_val_loss:
                        min_val_loss = val_loss
                        best_model_path = f'{path}/run_{run_idx+1}_{inception.__class__.__name__}{idx+1}_best_model.pth'
                        torch.save(inception.state_dict(), best_model_path)
                    print(f'Epoch {epoch+1}/{epochs}, Train Loss {train_loss:.4f} Train Acc {train_acc:.4f} Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}')
                
                    wandb.log({
                        f"train_loss_inception/run_{run_idx+1}": train_loss,
                        f"val_loss_inception/run_{run_idx+1}": val_loss,
                        f"val_acc_inception/run_{run_idx+1}": val_acc,
                        f"val_f1_inception/run_{run_idx+1}": val_f1,
                        f"step_inception/run_{run_idx+1}": epoch
                    })
                
                inception.load_state_dict(torch.load(best_model_path))
                torch.save(self.state_dict(), f'{path}/run_{run_idx+1}_HinceptionTime_merged_best_individuals_inceptions.pth')