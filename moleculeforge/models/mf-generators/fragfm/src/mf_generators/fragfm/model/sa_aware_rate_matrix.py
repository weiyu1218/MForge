"""SA score-aware rate matrix for discrete diffusion."""
import torch
import torch.nn as nn


class SAAwareRateMatrix(nn.Module):
    def __init__(self, vocab_size=10000):
        super().__init__()
        self.base_rate = nn.Parameter(torch.ones(vocab_size, vocab_size) * 0.01)
        self.sa_score_embedding = nn.Embedding(10, vocab_size * vocab_size)

    def forward(self, sa_score_bin: torch.Tensor):
        """Modulate rate matrix based on SA score (0-9 bins)."""
        sa_modulation = self.sa_score_embedding(sa_score_bin).view(-1, self.base_rate.shape[0], self.base_rate.shape[1])
        return self.base_rate * (1 + torch.tanh(sa_modulation))
