import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional


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

    def normalize_weights(self):
        with torch.no_grad():
            norms = self.W_out.norm(p=2, dim=0, keepdim=True)
            self.W_out.div_(norms)


    def forward(
        self,
        input: Tensor,
        pt=False,
        feature_modulations: Optional[list[tuple[int, float]]] = None,
        modulation_type: str = "*",
    ):
        input = input - self.b_out
        acts = self.nonlinearity(input @ self.W_in + self.b_in)
        l1_regularization = acts.abs().sum()  # / self.hidden_dim
        l0 = (acts > 0).sum(dim=1).float().mean()
        if modulation_type = "*":
            modulator = lambda l,r: l * r
        elif modulation_type = "+":
            modulator = lambda l,r: l + r
        else:
            assert False, ("unknown modulator type")
        if feature_modulations:
            for feature, multiplier in feature_modulations:
                print(f"feature {feature} before: {acts[:,:,feature]}")
                acts[:, :, feature] = modulator(acts[:, :, feature], multiplier)
                print(f"feature {feature} after: {acts[:,:,feature]}")
                print()
        if pt:
            print(acts)
        self.normalize_weights()
        return l0, l1_regularization, acts @ self.W_out + self.b_out

    def get_act_density(self, input: Tensor):
        input = input - self.b_out
        acts = self.nonlinearity(input @ self.W_in + self.b_in)
        return (acts > 0).sum(0)

    def get_activations(self, input: Tensor):
        input = input - self.b_out
        return self.nonlinearity(input @ self.W_in + self.b_in)
