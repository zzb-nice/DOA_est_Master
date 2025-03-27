import torch
import torch.nn as nn


class scm_embeding(nn.Module):
    def __init__(self, M, ebedding_dim):
        super().__init__()
        self.M = M
        self.linear = nn.Linear(2*M, ebedding_dim)

    def forward(self, x: torch.Tensor):
        x = x.transpose(-1, -3)
        x = torch.flatten(x, -2)
        x = self.linear(x)

        return x
