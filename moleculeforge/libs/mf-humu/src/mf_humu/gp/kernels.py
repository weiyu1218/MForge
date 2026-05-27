"""Gaussian Process kernels for Lorentz manifold."""
import torch
from torch import Tensor
from mf_humu.manifold.lorentz import LorentzManifold


class LorentzMaternKernel:
    """Matern kernel adapted for the Lorentz hyperboloid manifold.

    Uses the Lorentz geodesic distance as input to the Matern covariance
    function. Supports nu = 1/2, 3/2, and 5/2.

    Reference:
        Jaquier et al., "Geometry-aware Bayesian Optimization in Robotics
        using Riemannian Matern Kernels", CoRL 2021.
    """

    def __init__(
        self,
        length_scale: float = 1.0,
        nu: float = 2.5,
        manifold: LorentzManifold | None = None,
    ):
        self.length_scale = length_scale
        self.nu = nu
        self.manifold = manifold or LorentzManifold()

    def __call__(self, x1: Tensor, x2: Tensor) -> Tensor:
        """Compute kernel matrix between x1 and x2.

        Args:
            x1: Points on Lorentz manifold, shape (n1, d+1).
            x2: Points on Lorentz manifold, shape (n2, d+1).

        Returns:
            Kernel matrix of shape (n1, n2).
        """
        # Compute pairwise Lorentz geodesic distances
        dists = self._pairwise_distance(x1, x2)
        return self._matern_covariance(dists)

    def _pairwise_distance(self, x1: Tensor, x2: Tensor) -> Tensor:
        """Compute pairwise Lorentz geodesic distance matrix.

        Args:
            x1: shape (n1, d+1)
            x2: shape (n2, d+1)

        Returns:
            Distance matrix of shape (n1, n2).
        """
        n1, n2 = x1.shape[0], x2.shape[0]
        dists = torch.zeros(n1, n2, device=x1.device)
        for i in range(n1):
            # Broadcast x1[i] against all of x2
            xi = x1[i : i + 1].expand(n2, -1)
            dists[i] = self.manifold.distance(xi, x2).squeeze(-1)
        return dists

    def _matern_covariance(self, dists: Tensor) -> Tensor:
        """Compute Matern covariance given geodesic distances.

        Args:
            dists: Distance matrix scaled by length_scale.

        Returns:
            Covariance matrix.
        """
        d = dists / self.length_scale

        if self.nu == 0.5:
            # Exponential kernel
            return torch.exp(-d)
        elif self.nu == 1.5:
            sqrt3 = torch.tensor(3.0, device=dists.device)
            return (1.0 + sqrt3.sqrt() * d) * torch.exp(-sqrt3.sqrt() * d)
        elif self.nu == 2.5:
            sqrt5 = torch.tensor(5.0, device=dists.device)
            d_sq = d**2
            return (
                1.0 + sqrt5.sqrt() * d + (5.0 / 3.0) * d_sq
            ) * torch.exp(-sqrt5.sqrt() * d)
        else:
            # General approximation for arbitrary nu
            from scipy.special import gamma, kv

            eps = torch.tensor(1e-8, device=dists.device)
            d_safe = torch.clamp(d, min=eps)
            sqrt_2nu = torch.sqrt(torch.tensor(2.0 * self.nu, device=dists.device))
            const = 2.0 ** (1.0 - self.nu) / gamma(self.nu)
            return const * (sqrt_2nu * d_safe) ** self.nu * kv(self.nu, sqrt_2nu * d_safe)
