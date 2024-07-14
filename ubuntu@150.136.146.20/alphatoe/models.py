import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional
import torch.nn.functional as F


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
        if modulation_type == "*":
            modulator = lambda l, r: l * r
        elif modulation_type == "+":
            modulator = lambda l, r: l + r
        else:
            assert False, "unknown modulator type"
        if feature_modulations:
            for feature, multiplier in feature_modulations:
                print(f"feature {feature} before: {acts[:,:,feature]}")
                acts[:, :, feature] = modulator(acts[:, :, feature], multiplier)
                print(f"feature {feature} after: {acts[:,:,feature]}")
                print()
        if pt:
            print(acts)
        self.normalize_weights()

        # Directly use normalized W_out to compute cosine similarity matrix
        cosine_similarity_matrix = self.W_out.T @ self.W_out

        # Exclude self-similarities
        eye = torch.eye(cosine_similarity_matrix.size(0), device=cosine_similarity_matrix.device)
        cosine_similarity_l1 = cosine_similarity_matrix.abs().masked_select(~eye.bool()).sum() / (cosine_similarity_matrix.numel() - cosine_similarity_matrix.size(0))


        
        return l0, l1_regularization, cosine_similarity_l1, acts @ self.W_out + self.b_out


    def get_act_density(self, input: Tensor):
        input = input - self.b_out
        acts = self.nonlinearity(input @ self.W_in + self.b_in)
        return (acts > 0).sum(0)

    def get_activations(self, input: Tensor):
        input = input - self.b_out
        return self.nonlinearity(input @ self.W_in + self.b_in)
    
    def calculate_cosine_similarity(self):
        # Normalize W_in rows
        win_row_norm = F.normalize(self.W_in, p=2, dim=1)
        win_row_cosine = torch.mm(win_row_norm, win_row_norm.t())
        win_row_cosine.fill_diagonal_(0)  # Remove self-similarities
        win_row_l1 = win_row_cosine.abs().sum().item() / (win_row_cosine.numel() - win_row_cosine.size(0))

        # Normalize W_in columns
        win_col_norm = F.normalize(self.W_in, p=2, dim=0)
        win_col_cosine = torch.mm(win_col_norm.t(), win_col_norm)
        win_col_cosine.fill_diagonal_(0)  # Remove self-similarities
        win_col_l1 = win_col_cosine.abs().sum().item() / (win_col_cosine.numel() - win_col_cosine.size(0))

        # Normalize W_out rows
        wout_row_norm = F.normalize(self.W_out, p=2, dim=1)
        wout_row_cosine = torch.mm(wout_row_norm, wout_row_norm.t())
        wout_row_cosine.fill_diagonal_(0)  # Remove self-similarities
        wout_row_l1 = wout_row_cosine.abs().sum().item() / (wout_row_cosine.numel() - wout_row_cosine.size(0))

        # Normalize W_out columns
        wout_col_norm = F.normalize(self.W_out, p=2, dim=0)
        wout_col_cosine = torch.mm(wout_col_norm.t(), wout_col_norm)
        wout_col_cosine.fill_diagonal_(0)  # Remove self-similarities
        wout_col_l1 = wout_col_cosine.abs().sum().item() / (wout_col_cosine.numel() - wout_col_cosine.size(0))

        return win_row_l1, win_col_l1, wout_row_l1, wout_col_l1