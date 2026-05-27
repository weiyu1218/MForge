"""Unfamiliarity estimator using autoencoder reconstruction error."""
import torch


def compute_unfamiliarity(z: torch.Tensor, reference: torch.Tensor, model=None) -> torch.Tensor:
    """OOD detection via reconstruction error — NOT random."""
    if model is not None:
        recon, _ = model(z)
        recon_error = ((recon - z) ** 2).mean(dim=-1)
    else:
        # Distance-based fallback
        dists = torch.cdist(z, reference)
        recon_error = dists.min(dim=-1).values
    return recon_error
