"""HCIV and IntentCone generation (deterministic sampling)."""

import torch
from mf_core.types.humu import HCIV, IntentCone


def generate_random_hciv(dim: int, seed: int | None = None) -> HCIV:
    offset = float(seed or 0)
    basis = torch.arange(1, dim + 1, dtype=torch.float32)
    spatial = torch.sin(basis * 12.9898 + offset * 78.233) * 0.5
    time = torch.sqrt(1.0 + (spatial ** 2).sum())
    coords = torch.cat([time.unsqueeze(0), spatial], dim=0)

    return HCIV(
        coordinates=coords.tolist(),
        dim=dim,
        curvature=1.0,
    )


def generate_intent_cone(
    apex: HCIV,
    dim: int,
    seed: int | None = None,
    half_angle: float = 0.5,
) -> IntentCone:
    return IntentCone(
        apex=apex,
        axis_direction=apex,
        axis=list(apex.coordinates),
        half_angle=float(half_angle),
        angle_radians=float(half_angle),
        length=1.0,
        curvature=1.0,
    )
