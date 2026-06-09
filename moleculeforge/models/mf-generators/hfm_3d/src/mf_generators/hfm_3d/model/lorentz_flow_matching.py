"""Lorentz Flow Matching on the hyperboloid tangent bundle."""
import torch
import torch.nn as nn
from mf_humu.manifold.lorentz import LorentzManifold


class LorentzFlowMatching(nn.Module):
    def __init__(self, dim=128, curvature=1.0, n_steps=20):
        super().__init__()
        self.manifold = LorentzManifold(curvature=curvature)
        self.dim = dim
        self.n_steps = n_steps
        self.velocity_net = nn.Sequential(
            nn.Linear(dim + 2, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, dim + 1),
        )

    def compute_vector_field(self, x_t, t):
        v = self.velocity_net(torch.cat([x_t, t.expand(x_t.shape[0], 1)], dim=-1))
        return self.manifold.project_tangent(x_t, v)

    def forward(self, x0, x1):
        t = torch.rand(x0.shape[0], 1, device=x0.device) * 0.98
        log_x0_x1 = self.manifold.logmap(x0, x1)
        x_t = self.manifold.expmap(x0, t * log_x0_x1)
        v_target = self.manifold.logmap(x_t, x1) / (1.0 - t).clamp_min(1e-3)
        v_pred = self.compute_vector_field(x_t, t)
        return ((v_pred - v_target) ** 2).mean()
