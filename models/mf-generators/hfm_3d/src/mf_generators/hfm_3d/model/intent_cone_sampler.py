"""Intent cone guided sampling for HFM-3D."""
import torch
from mf_humu.manifold.lorentz import LorentzManifold
from mf_humu.operations.intent_cone import sample_within_cone
from mf_core.types.humu import IntentCone
from mf_generators.hfm_3d.model.ode_solver import MidpointODESolver


class IntentConeSampler:
    def __init__(self, manifold, velocity_fn, n_steps=20):
        self.manifold = manifold
        self.solver = MidpointODESolver(velocity_fn, n_steps=n_steps)

    def sample(self, cone: IntentCone, n_samples: int):
        z_cone = sample_within_cone(cone, n_samples, self.manifold)
        return self.solver.integrate(z_cone)
