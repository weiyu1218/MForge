"""Hyperbolic geometry functions for Lorentz model distance computation."""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def project_to_lorentz(v: NDArray[np.float64]) -> NDArray[np.float64]:
    """Project Euclidean vector v to Lorentz hyperboloid (K=1).

    Returns x = [sqrt(1 + ||v||^2), v] such that ⟨x,x⟩_L = -1.
    """
    v = np.asarray(v, dtype=np.float64)
    norm_sq = np.sum(v**2)
    x0 = np.sqrt(1.0 + norm_sq)
    return np.concatenate(([x0], v))


def lorentz_inner_product(x: NDArray[np.float64], y: NDArray[np.float64]) -> float:
    """Lorentz inner product: -x0*y0 + sum(xi*yi) for i=1..d."""
    return float(-x[0] * y[0] + np.dot(x[1:], y[1:]))


def hyperbolic_distance(x: NDArray[np.float64], y: NDArray[np.float64]) -> float:
    """Geodesic distance on Lorentz hyperboloid (K=1).

    d(x,y) = arccosh(-⟨x,y⟩_L)
    """
    inner = lorentz_inner_product(x, y)
    arg = -inner
    if arg <= 1.0 + 1e-12:
        return 0.0
    return float(np.arccosh(arg))


def batch_project(
    v_batch: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Project a batch of Euclidean vectors to Lorentz hyperboloid.

    Args:
        v_batch: shape (batch_size, d)

    Returns:
        shape (batch_size, d+1)
    """
    v_batch = np.asarray(v_batch, dtype=np.float64)
    norm_sq = np.sum(v_batch**2, axis=1, keepdims=True)
    x0 = np.sqrt(1.0 + norm_sq)
    return np.concatenate((x0, v_batch), axis=1)
