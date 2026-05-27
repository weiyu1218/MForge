"""Lorentz-equivariant Transformer layer."""
import torch
import torch.nn as nn
from mf_humu.encoders.lorentz_attention import LorentzAttention


class LorentzEquivariantLayer(nn.Module):
    def __init__(self, dim=128, heads=8):
        super().__init__()
        self.attention = LorentzAttention(dim=dim, heads=heads)
        self.norm1 = nn.LayerNorm(dim + 1)
        self.norm2 = nn.LayerNorm(dim + 1)
        self.ffn = nn.Sequential(nn.Linear(dim + 1, 4 * (dim + 1)), nn.GELU(), nn.Linear(4 * (dim + 1), dim + 1))

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x
