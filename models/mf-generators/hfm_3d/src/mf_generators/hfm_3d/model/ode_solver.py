"""Midpoint ODE solver for Lorentz flow integration."""
import torch


class MidpointODESolver:
    def __init__(self, velocity_fn, n_steps=20):
        self.velocity_fn = velocity_fn
        self.n_steps = n_steps

    def integrate(self, z0, t_span=(0.0, 1.0)):
        """Integrate ODE using midpoint method (order 2)."""
        dt = (t_span[1] - t_span[0]) / self.n_steps
        z = z0
        for step in range(self.n_steps):
            t = t_span[0] + step * dt
            k1 = self.velocity_fn(z, t)
            z_mid = z + 0.5 * dt * k1
            k2 = self.velocity_fn(z_mid, t + 0.5 * dt)
            z = z + dt * k2
        return z
