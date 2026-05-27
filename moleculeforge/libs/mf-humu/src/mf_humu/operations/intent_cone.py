"""Intent Cone sampling on the Lorentz manifold."""
import torch
from torch import Tensor
from mf_core.types.humu import IntentCone
from mf_humu.manifold.lorentz import LorentzManifold


def sample_within_cone(
    cone: IntentCone, n_samples: int, manifold: LorentzManifold
) -> Tensor:
    """Sample points from within the intent cone on the Lorentz manifold.

    Generates random points within the half-angle cone around the specified
    axis direction on the Lorentz hyperboloid model.

    Args:
        cone: IntentCone defining the sampling region (axis, half_angle, etc.).
        n_samples: Number of sample points to generate.
        manifold: LorentzManifold instance for geometric operations.

    Returns:
        Tensor of shape (n_samples, d+1) of points on the Lorentz manifold.
    """
    axis = torch.tensor(cone.axis, dtype=torch.float32).unsqueeze(0)
    half_angle = cone.half_angle

    samples = []
    for _ in range(n_samples):
        # Generate random direction in tangent space at axis
        v = torch.randn(1, 129)
        # Tangent space at origin: time component must be zero
        v[..., 0] = 0
        # Scale by random angle within half_angle
        angle = torch.rand(1) * half_angle
        v_norm = torch.sqrt(manifold.inner(v, v) + 1e-8)
        v = v / v_norm * angle
        # Exponential map from axis
        sample = manifold.expmap(axis, v)
        samples.append(sample)
    return torch.cat(samples, dim=0)
