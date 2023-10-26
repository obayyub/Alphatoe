import torch
import torch.nn as nn
from torch import Tensor


class SparseAutoEncoder(nn.Module):
    def __init__(self, mundane_dim, hidden_dim, nonlinearity=nn.ReLU()):
        super().__init__()
        self.W_in = nn.Parameter(
            nn.init.kaiming_uniform_(
                torch.empty(mundane_dim, hidden_dim), nonlinearity="relu"
            )
        )
        self.b_in = nn.Parameter(torch.zeros(hidden_dim))
        self.W_out = nn.Parameter(
            nn.init.kaiming_uniform_(
                torch.empty(hidden_dim, mundane_dim), nonlinearity="relu"
            )
        )
        self.b_out = nn.Parameter(torch.zeros(mundane_dim))
        self.nonlinearity = nonlinearity

    def forward(self, input: Tensor, pt=False):
        input = input - self.b_out
        acts = self.nonlinearity(input @ self.W_in + self.b_in)
        l1_regularization = acts.abs().sum()  # / self.hidden_dim
        if pt:
            print(acts)
        return l1_regularization, acts @ self.W_out + self.b_out
