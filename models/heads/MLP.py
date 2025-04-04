import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, output_size):
        super(MLP, self).__init__()
        self.z1 = nn.Linear(in_dim, hidden_dim)
        self.a1 = nn.ReLU()
        self.z2 = nn.Linear(hidden_dim, output_size)
        self.a2 = nn.ReLU()
    
    def forward(self, x):
        a1 = self.a1(self.z1(x))
        self.embedding = self.z2(a1)
        a2 = self.a2(self.embedding)
        return a2