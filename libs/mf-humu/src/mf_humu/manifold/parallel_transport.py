"""Parallel transport on the Lorentz manifold.

Implements parallel transport of a tangent vector v from x to y along the
geodesic connecting them on the Lorentz hyperboloid model.

Reference:
    Chami et al., "Hyperbolic Graph Neural Networks", NeurIPS 2019.
    Nickel & Kiela, "Learning Continuous Hierarchies in the Lorentz Model",
    ICML 2018.
"""
import torch
from torch import Tensor
from mf_humu.manifold.lorentz import LorentzManifold


def parallel_transport(
    x: Tensor, y: Tensor, v: Tensor, manifold: LorentzManifold | None = None
) -> Tensor:
    """Parallel transport tangent vector v from point x to point y.

    Args:
        x: Source point on the Lorentz manifold, shape (..., d+1).
        y: Target point on the Lorentz manifold, shape (..., d+1).
        v: Tangent vector at x, shape (..., d+1).
        manifold: Lorentz manifold instance. Created with default curvature
            if None.

    Returns:
        Transported tangent vector at y, shape (..., d+1).

    The parallel transport formula on the Lorentz model:
        PT_{x->y}(v) = v - <log_x(y), v>_L / <x, y>_L * (x + y)
    """
    if manifold is None:
        manifold = LorentzManifold()

    k = manifold.k

    # Lorentz inner product between x and y
    alpha = manifold.inner(x, y)  # shape (..., 1)

    # Logarithmic map from x to y (tangent vector at x)
    log_xy = manifold.logmap(x, y)  # shape (..., d+1)

    # Inner product <log_x(y), v>_L
    inner_log_v = manifold.inner(log_xy, v)  # shape (..., 1)

    # alpha is negative, with -1/k <= alpha < -eps
    # Use a numerically safe division
    denominator = torch.clamp(-alpha, min=manifold.eps)

    # Transport formula
    # PT(v) = v - <log_x(y), v>_L / <x, y>_L * (x + y)
    # where <x, y>_L = alpha is negative, so we use -alpha in denominator
    coeff = inner_log_v / (alpha + k * alpha * alpha + 1e-8)

    # The direction is x + y (in ambient space, related to the geodesic)
    direction = x + y

    v_transported = v - coeff * direction

    # Ensure result is tangent to the manifold at y:
    # <v_transported, y>_L should be 0
    inner_correction = manifold.inner(v_transported, y)
    v_transported = v_transported - inner_correction * k * y

    return v_transported


def parallel_transport_matrix(
    x: Tensor, y: Tensor, manifold: LorentzManifold | None = None
) -> Tensor:
    """Compute the parallel transport matrix from x to y.

    This is useful when the same transport needs to be applied to multiple
    tangent vectors.

    Args:
        x: Source point, shape (..., d+1).
        y: Target point, shape (..., d+1).
        manifold: Lorentz manifold instance.

    Returns:
        Transport matrix of shape (..., d+1, d+1) such that
        v_transported = (PT_matrix @ v.unsqueeze(-1)).squeeze(-1).
    """
    if manifold is None:
        manifold = LorentzManifold()

    d_plus_1 = x.shape[-1]
    device = x.device

    alpha = manifold.inner(x, y)

    # Build identity-like transport matrix
    I = torch.eye(d_plus_1, device=device).expand(*x.shape[:-1], d_plus_1, d_plus_1)

    # Direction term (x + y) as an outer product with log_x(y)^T
    log_xy = manifold.logmap(x, y)
    direction = x + y

    # The log_xy coordinate is in the tangent space at x; we need the metric
    # dual to get covector components. In Lorentz metric, the dual of vector
    # u is g * u where g = diag(-1, 1, ..., 1).
    g = torch.ones(d_plus_1, device=device)
    g[0] = -1.0
    metric = torch.diag(g).expand(*x.shape[:-1], d_plus_1, d_plus_1)

    # covector = g @ log_xy
    log_xy_covector = (metric @ log_xy.unsqueeze(-1)).squeeze(-1)

    # <log_x(y), u>_L = g_ij * log^i * u^j = covector^T @ u
    # The transport correction is: outer(direction, covector) / alpha
    # where alpha = <x, y>_L
    correction_scalar = 1.0 / torch.clamp(alpha, max=-manifold.eps)
    correction_matrix = correction_scalar.unsqueeze(-1).unsqueeze(-1) * torch.einsum(
        "...i,...j->...ij", direction, log_xy_covector
    )

    pt_matrix = I - correction_matrix

    return pt_matrix
