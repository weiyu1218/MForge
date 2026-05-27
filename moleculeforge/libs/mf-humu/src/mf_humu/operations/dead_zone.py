"""Patent Dead Zone potential for avoiding patented chemical space."""
import torch
from torch import Tensor


def patent_dead_zone_potential(
    z: Tensor,
    dead_zones: list[Tensor],
    radius: float = 0.5,
    manifold=None,
) -> Tensor:
    """Compute repulsive potential from known patent dead zones.

    Returns a scalar potential that increases as z approaches any dead zone.
    Uses a soft repulsive potential: sum_i exp(-||z - dz_i||^2 / (2*radius^2))

    Args:
        z: Latent points to evaluate, shape (batch, d+1).
        dead_zones: List of dead zone point tensors, each shape (n_i, d+1).
        radius: Characteristic radius of the repulsive potential.

    Returns:
        Scalar potential for each point in z, shape (batch,).
    """
    if radius <= 0:
        raise ValueError("radius must be positive")
    if not dead_zones:
        return torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)
    points = []
    for dead_zone in dead_zones:
        dz = dead_zone.to(device=z.device, dtype=z.dtype)
        if dz.ndim == 1:
            dz = dz.unsqueeze(0)
        points.append(dz)
    dead_zone_matrix = torch.cat(points, dim=0)
    eps = getattr(manifold, "eps", 1e-8)
    inner = (
        -z[:, 0:1] * dead_zone_matrix[:, 0].unsqueeze(0)
        + z[:, 1:] @ dead_zone_matrix[:, 1:].T
    )
    distance = torch.arccosh(torch.clamp(-inner, min=1.0 + eps))
    min_distance = distance.min(dim=1).values
    return torch.exp(-(min_distance**2) / (2 * radius**2))
