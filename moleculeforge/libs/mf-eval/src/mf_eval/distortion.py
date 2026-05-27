"""Distance distortion metrics for embeddings."""
from __future__ import annotations

import torch


def pairwise_distance_distortion(source_distances, embedding_distances) -> dict[str, float | int]:
    source = _as_square_tensor(source_distances, "source_distances")
    embedding = _as_square_tensor(embedding_distances, "embedding_distances")
    if source.shape != embedding.shape:
        raise ValueError("source_distances and embedding_distances must have the same shape")
    if source.shape[0] < 2:
        return {
            "mean_absolute_error": 0.0,
            "mean_relative_error": 0.0,
            "spearman_r": 1.0,
            "n_pairs": 0,
        }
    mask = torch.triu(torch.ones_like(source, dtype=torch.bool), diagonal=1)
    source_values = source[mask]
    embedding_values = embedding[mask]
    abs_error = (source_values - embedding_values).abs()
    relative_error = abs_error / source_values.abs().clamp_min(1e-8)
    return {
        "mean_absolute_error": float(abs_error.mean().item()),
        "mean_relative_error": float(relative_error.mean().item()),
        "spearman_r": _spearman_r(source_values, embedding_values),
        "n_pairs": int(source_values.numel()),
    }


def _as_square_tensor(value, name: str) -> torch.Tensor:
    tensor = value if isinstance(value, torch.Tensor) else torch.tensor(value, dtype=torch.float32)
    tensor = tensor.float()
    if tensor.ndim != 2 or tensor.shape[0] != tensor.shape[1]:
        raise ValueError(f"{name} must be a square distance matrix")
    return tensor


def _spearman_r(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() < 2:
        return 1.0
    a_rank = _rank(a)
    b_rank = _rank(b)
    a_centered = a_rank - a_rank.mean()
    b_centered = b_rank - b_rank.mean()
    denom = torch.sqrt((a_centered**2).sum() * (b_centered**2).sum())
    if float(denom.item()) == 0.0:
        return 1.0 if torch.allclose(a_rank, b_rank) else 0.0
    return float(((a_centered * b_centered).sum() / denom).item())


def _rank(values: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(values)
    ranks = torch.empty_like(values, dtype=torch.float32)
    ranks[order] = torch.arange(values.numel(), dtype=torch.float32, device=values.device)
    return ranks
