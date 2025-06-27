import torch.nn as nn


class MLP(nn.Module):
     def __init__(self, in_dim, hidden_dim, num_classes):
         super(MLP, self).__init__()
     
         self.layers = nn.Sequential(
             nn.Dropout(0.1),
             nn.Linear(in_dim, hidden_dim),
             nn.LeakyReLU(),
             nn.Dropout(0.2),
             nn.Linear(hidden_dim, hidden_dim),
             nn.LeakyReLU(),
             nn.Dropout(0.2),
             nn.Linear(hidden_dim, hidden_dim),
             nn.LeakyReLU(),
             nn.Dropout(0.3),
             nn.Linear(hidden_dim, hidden_dim)
         )

         self.logits = nn.Linear(in_features=hidden_dim, out_features=num_classes if num_classes > 2 else 1)
 
     def forward(self, x):
         _x = self.layers(x)
         return self.logits(_x)