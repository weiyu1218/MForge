"""Expected Hypervolume Improvement (EHVI) acquisition function.

Computes the expected hypervolume improvement for multi-objective Bayesian
optimization on the Lorentz manifold. Used to select candidate molecules that
maximally improve the Pareto front.

Reference:
    Emmerich et al., "Hypervolume-based Expected Improvement: Monotonicity
    Properties and Exact Computation", CEC 2011.
    Daulton et al., "Differentiable Expected Hypervolume Improvement for
    Parallel Multi-Objective Bayesian Optimization", NeurIPS 2020.
"""
import torch
from torch import Tensor
from math import pi


def _erf(x: Tensor) -> Tensor:
    """Error function (vectorized)."""
    return torch.erf(x)


def _normal_pdf(x: Tensor) -> Tensor:
    """Standard normal PDF."""
    return torch.exp(-0.5 * x**2) / torch.sqrt(
        torch.tensor(2.0 * pi, device=x.device)
    )


def _normal_cdf(x: Tensor) -> Tensor:
    """Standard normal CDF."""
    return 0.5 * (1.0 + _erf(x / torch.sqrt(torch.tensor(2.0, device=x.device))))


def ehvi(
    mu: Tensor,
    sigma: Tensor,
    ref_point: Tensor,
    pareto_front: Tensor | None = None,
) -> Tensor:
    """Compute Expected Hypervolume Improvement.

    Args:
        mu: Predicted mean objectives, shape (n_candidates, n_objectives).
        sigma: Predicted standard deviations, shape (n_candidates, n_objectives).
        ref_point: Reference point for hypervolume computation,
            shape (n_objectives,).
        pareto_front: Current Pareto front points, shape (n_pareto, n_objectives).
            If None, ref_point is used as the only reference.

    Returns:
        EHVI values for each candidate, shape (n_candidates,).
    """
    n_candidates, n_obj = mu.shape

    if pareto_front is None:
        pareto_front = ref_point.unsqueeze(0)

    # For each candidate, compute the probability-weighted improvement
    # over each cell of the Pareto front partition
    ehvi_values = torch.zeros(n_candidates, device=mu.device)

    for i in range(n_candidates):
        # Standardized improvement
        z = (mu[i] - pareto_front) / torch.clamp(
            sigma[i].unsqueeze(0), min=1e-8
        )

        # Expected improvement per objective (assuming minimization)
        ei_per_obj = (
            (pareto_front - mu[i].unsqueeze(0))
            * _normal_cdf(z)
            + sigma[i].unsqueeze(0) * _normal_pdf(z)
        )
        # Negative because we assume minimization (improvement = front - candidate)
        # Actually for minimization: improvement = max(front - candidate, 0)
        ei_per_obj = torch.clamp(ei_per_obj, min=0.0)

        # Hypervolume contribution: product of per-objective EIs
        hv_contrib = ei_per_obj.prod(dim=-1)
        ehvi_values[i] = hv_contrib.sum()

    return ehvi_values


def ehvi_monte_carlo(
    mu: Tensor,
    sigma: Tensor,
    ref_point: Tensor,
    pareto_front: Tensor | None = None,
    n_samples: int = 100,
) -> Tensor:
    """Monte Carlo approximation of EHVI.

    More accurate than the analytical decomposition for n_obj > 2.

    Args:
        mu: Predicted means, shape (n_candidates, n_objectives).
        sigma: Predicted stds, shape (n_candidates, n_objectives).
        ref_point: Reference point, shape (n_objectives,).
        pareto_front: Current Pareto front, shape (n_pareto, n_objectives).
        n_samples: Number of Monte Carlo samples.

    Returns:
        Approximate EHVI values, shape (n_candidates,).
    """
    n_candidates, n_obj = mu.shape
    device = mu.device

    if pareto_front is None:
        pareto_front = ref_point.unsqueeze(0)

    # Sample from the posterior
    samples = mu.unsqueeze(1) + sigma.unsqueeze(1) * torch.randn(
        n_candidates, n_samples, n_obj, device=device
    )

    # For each sample, compute hypervolume improvement
    hv_improvements = torch.zeros(n_candidates, n_samples, device=device)

    for c in range(n_candidates):
        for s in range(n_samples):
            candidate_obj = samples[c, s]
            # Check if the sample dominates any Pareto front point
            # (assuming minimization: lower is better)
            dominated = (candidate_obj.unsqueeze(0) <= pareto_front).all(dim=-1)
            dominates = (candidate_obj.unsqueeze(0) < pareto_front).any(dim=-1)

            # Improvement: distance from ref_point to candidate, bounded by front
            improvement = torch.clamp(
                ref_point - candidate_obj, min=0.0
            ).prod()
            hv_improvements[c, s] = improvement if dominated.any() or dominates.all() else 0.0

    return hv_improvements.mean(dim=-1)
