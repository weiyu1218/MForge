"""Molecule autoencoder for UAS (Unfamiliarity-Aware Sampling)."""
import torch
import torch.nn as nn


class MoleculeAutoencoder(nn.Module):
    def __init__(self, input_dim=128, latent_dim=64, dim=None):
        super().__init__()
        if dim is not None:
            input_dim = dim
            latent_dim = max(1, input_dim // 2)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z

    def reconstruction_loss(self, x):
        recon, _ = self.forward(x)
        return ((recon - x) ** 2).mean(dim=tuple(range(1, x.ndim)))
