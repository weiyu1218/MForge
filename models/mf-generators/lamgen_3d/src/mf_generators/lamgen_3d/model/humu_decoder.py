"""HUMU decoder that outputs Lorentz coordinates directly."""
import torch
import torch.nn as nn
from mf_humu.manifold.lorentz import LorentzManifold


class HUMUDecoder(nn.Module):
    def __init__(self, dim=128, curvature=1.0):
        super().__init__()
        self.manifold = LorentzManifold(curvature=curvature)
        self.decoder = nn.Sequential(
            nn.Linear(dim, 256), nn.GELU(),
            nn.Linear(256, 256), nn.GELU(),
            nn.Linear(256, dim + 1),
        )

    def forward(self, latent):
        raw = self.decoder(latent)
        # Project onto Lorentz hyperboloid
        hciv = self.manifold._project(raw)
        return hciv
