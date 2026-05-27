"""Multi-target attention for LaMGen-3D."""
import torch
import torch.nn as nn


class MultiTargetAttention(nn.Module):
    def __init__(self, dim=128, n_targets=4):
        super().__init__()
        self.target_queries = nn.Parameter(torch.randn(n_targets, dim))
        self.cross_attn = nn.MultiheadAttention(dim, num_heads=8, batch_first=True)

    def forward(self, x, target_features):
        queries = self.target_queries.unsqueeze(0).expand(x.shape[0], -1, -1)
        attn_out, _ = self.cross_attn(queries, target_features, target_features)
        return attn_out
