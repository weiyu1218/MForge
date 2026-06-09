"""Lorentz hyperboloid payload validation helpers."""

from __future__ import annotations

import math
from collections.abc import Sequence


def normalize_lorentz_embedding(
    value: object,
    *,
    expected_dim: int,
    curvature: float = 1.0,
    tolerance: float = 1e-4,
) -> list[float] | None:
    """Return finite Lorentz full coordinates, or None for non-steering input.

    The current HUMU/HFM contract uses full hyperboloid coordinates in R^{d+1}.
    A valid point must satisfy x0 > 0 and -x0^2 + ||x_spatial||^2 ~= -1/k.
    """

    if expected_dim <= 1:
        return None
    if not math.isfinite(curvature) or curvature <= 0.0:
        return None
    if isinstance(value, str | bytes | bytearray):
        return None
    if not isinstance(value, Sequence):
        return None
    if len(value) != expected_dim:
        return None
    try:
        embedding = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(item) for item in embedding):
        return None
    time_coord = embedding[0]
    if time_coord <= 0.0:
        return None
    spatial_norm_sq = math.fsum(item * item for item in embedding[1:])
    expected_time_sq = spatial_norm_sq + 1.0 / curvature
    actual_time_sq = time_coord * time_coord
    allowed_error = max(
        float(tolerance),
        float(tolerance) * max(expected_time_sq, actual_time_sq, 1.0),
    )
    if abs(actual_time_sq - expected_time_sq) > allowed_error:
        return None
    return embedding
