"""Conditional sampler for HFM-3D with property-guided generation."""
import torch
from mf_humu.manifold.lorentz import LorentzManifold
from mf_generators.hfm_3d.model.ode_solver import MidpointODESolver


class ConditionalSampler:
    def __init__(self, flow_model, manifold, property_predictor=None, n_steps=20):
        self.flow_model = flow_model
        self.manifold = manifold
        self.property_predictor = property_predictor
        self.solver = MidpointODESolver(self._guided_velocity, n_steps=n_steps)

    def _guided_velocity(self, z, t):
        v = self.flow_model.compute_vector_field(z, t)
        if self.property_predictor is not None:
            with torch.enable_grad():
                z.requires_grad_(True)
                props = self.property_predictor(z)
                grad = torch.autograd.grad(props.sum(), z)[0]
                v = v + 0.1 * grad
        return v

    def sample(self, z0):
        return self.solver.integrate(z0)
