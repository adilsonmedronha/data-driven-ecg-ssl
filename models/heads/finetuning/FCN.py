import torch
import torch.nn as nn

import torch.nn.functional as F

class FCN(nn.Module):

    def __init__(self, in_dim: int, num_classes: int) -> None:
        super().__init__()
        
        self.num_classes = num_classes

        self.conv_layers = nn.Sequential(*[
            nn.Conv1d(in_channels=in_dim, out_channels=128, kernel_size=8, padding='same'),
            nn.BatchNorm1d(num_features=128),
            nn.ReLU(),
            
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, padding='same'),
            nn.BatchNorm1d(num_features=256),
            nn.ReLU(),

            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, padding='same'),
            nn.BatchNorm1d(num_features=128),
            nn.ReLU(),
        ])
        
        self.linear = nn.Linear(in_features=128, out_features=1 if num_classes == 2 else num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1) if x.ndim < 3 else x
        x = self.conv_layers(x)
        self.embedding = torch.mean(x, dim=-1)
        return self.linear(self.embedding)