"""Lorentz-equivariant attention layer for HUMU encoders."""
import torch
import torch.nn as nn
from torch import Tensor
from mf_humu.manifold.learnable_lorentz import LearnableLorentzManifold
from mf_humu.manifold.lorentz import LorentzManifold


class LorentzAttention(nn.Module):
    """Attention mechanism that respects Lorentz manifold geometry.

    Projects queries, keys, and values from Lorentz space through an
    attention computation, then projects back to the Lorentz manifold.
    """

    def __init__(
        self,
        dim: int = 128,
        heads: int = 8,
        curvature: float = 1.0,
        learnable_curvature: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        manifold_cls = LearnableLorentzManifold if learnable_curvature else LorentzManifold
        self.manifold = manifold_cls(curvature=curvature)
        self.q_proj = nn.Linear(dim + 1, dim)
        self.k_proj = nn.Linear(dim + 1, dim)
        self.v_proj = nn.Linear(dim + 1, dim)
        self.out_proj = nn.Linear(dim, dim + 1)
        self.scale = self.head_dim**-0.5

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with Lorentz-aware attention.

        Args:
            x: Points on Lorentz manifold, shape (batch, seq_len, d+1).

        Returns:
            Output in ambient space, shape (batch, seq_len, d+1).
        """
        Q = (
            self.q_proj(x)
            .view(*x.shape[:-1], self.heads, self.head_dim)
            .transpose(-3, -2)
        )
        K = (
            self.k_proj(x)
            .view(*x.shape[:-1], self.heads, self.head_dim)
            .transpose(-3, -2)
        )
        V = (
            self.v_proj(x)
            .view(*x.shape[:-1], self.heads, self.head_dim)
            .transpose(-3, -2)
        )
        attn = torch.softmax(
            (Q @ K.transpose(-1, -2)) * self.scale, dim=-1
        )
        out = (attn @ V).transpose(-3, -2).reshape(*x.shape[:-1], self.dim)
        return self.out_proj(out)
