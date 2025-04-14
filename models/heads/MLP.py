import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, output_size):
        super(MLP, self).__init__()
        self.logits = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(in_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_size),
            nn.LeakyReLU(),
            nn.Dropout(0.3)
        )

    def forward(self, x):
        return self.logits(x)