import torch
from torch import nn
from torch.nn import functional as F



class BaseModel(nn.Module):
    '''Encode model'''

    def __init__(self, in_dim, out_dim, kernel_size=8, stride=1, dropout=0.35):
        '''
        
        Args:
            in_dim (int): The input dimension. For a univariate time series, this should be set to 1.
            out_dim (int): The representation dimension.
            kernel_size (int): Size of the first convolutional kernel. Default = 25.
            stride (int): Stride of the first convolutional layer. Default = 3.
            dropout (float): Dropout probability. Default = 0.35.
        '''
        super(BaseModel, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(in_dim, 32, kernel_size=kernel_size, stride=stride, bias=False, padding=(kernel_size//2)),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(dropout)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(64, out_dim, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        # self.logits = nn.Linear(feat_len * out_dim, num_classes)


    def forward(self, x_in):
        x = self.conv_block1(x_in)
        x = self.conv_block2(x)
        x = self.conv_block3(x)

        return x
    

    def encode(self, x_in) -> torch.Tensor: 
        '''Encode the input time series.'''

        repr = self.forward(x_in)
        repr = F.adaptive_avg_pool1d(repr, 1)

        return repr


    def tuning_mode(self, mode: str):
        '''Set the tuning mode.
        
        Args:
            mode (str): Set the tuning mode `linear-probing` | `fine-tuning`.
        '''

        # freeze model
        if mode == 'linear-probing':
            self.requires_grad_(False)