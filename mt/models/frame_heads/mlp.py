import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, in_dim: int, dim: int, out_dim: int, **kwargs):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.fc3 = nn.Linear(dim, out_dim)

    def forward(self, x):
        x = F.silu(self.fc1(x))
        x = F.silu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x