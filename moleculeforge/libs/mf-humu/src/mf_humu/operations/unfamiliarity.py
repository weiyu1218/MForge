"""Unfamiliarity estimation using autoencoder reconstruction error."""
import torch
from torch import Tensor


def compute_unfamiliarity(
    z: Tensor, reference: Tensor, n_neighbors: int = 10
) -> Tensor:
    """Compute unfamiliarity score based on distance to nearest reference points.

    Lower values = more familiar (closer to reference distribution).
    Higher values = more unfamiliar (farther from reference).

    Args:
        z: Latent points to evaluate, shape (batch, d+1).
        reference: Reference distribution points, shape (n_ref, d+1).
        n_neighbors: Number of nearest neighbors to consider.

    Returns:
        Unfamiliarity score for each point in z, shape (batch,).
    """
    unfamiliarity = torch.zeros(z.shape[0], device=z.device)
    ref = reference.to(z.device)
    for i in range(z.shape[0]):
        dists = torch.norm(z[i : i + 1] - ref, dim=-1)
        nearest_dists = dists.topk(
            min(n_neighbors, len(ref)), largest=False
        ).values
        unfamiliarity[i] = nearest_dists.mean()
    return unfamiliarity
