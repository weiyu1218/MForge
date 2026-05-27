"""Unit tests for hyperbolic geometry functions (numerical correctness)."""

from __future__ import annotations

import numpy as np
import pytest

from mf_core.db.vector.hyperbolic_metric import (
    batch_project,
    hyperbolic_distance,
    lorentz_inner_product,
    project_to_lorentz,
)


@pytest.mark.unit
def test_project_to_lorentz_shape() -> None:
    v = np.array([0.5, 0.3, 0.1])
    projected = project_to_lorentz(v)
    assert projected.shape == (4,)  # dim + 1
    assert projected[0] > 1.0  # x0 = sqrt(1 + ||v||^2) > 1


@pytest.mark.unit
def test_project_to_lorentz_on_manifold() -> None:
    """The projected point must satisfy x0^2 - sum(xi^2) = 1."""
    v = np.array([1.0, 2.0, 3.0])
    projected = project_to_lorentz(v)
    lorentz_norm = -projected[0] ** 2 + np.sum(projected[1:] ** 2)
    assert np.isclose(lorentz_norm, -1.0, atol=1e-10)


@pytest.mark.unit
def test_lorentz_inner_product_identity() -> None:
    """Two identical Lorentz points should have ⟨x,x⟩_L = -1 (on manifold)."""
    v = np.array([0.5, 0.5])
    x = project_to_lorentz(v)
    inner = lorentz_inner_product(x, x)
    assert np.isclose(inner, -1.0, atol=1e-10)


@pytest.mark.unit
def test_hyperbolic_distance_zero() -> None:
    """Distance from a point to itself should be 0."""
    v = np.array([1.0, 0.0, 0.0])
    x = project_to_lorentz(v)
    dist = hyperbolic_distance(x, x)
    assert np.isclose(dist, 0.0, atol=1e-10)


@pytest.mark.unit
def test_hyperbolic_distance_symmetry() -> None:
    """d(x, y) == d(y, x)."""
    x = project_to_lorentz(np.array([1.0, 0.0]))
    y = project_to_lorentz(np.array([0.0, 1.0]))
    assert np.isclose(hyperbolic_distance(x, y), hyperbolic_distance(y, x))


@pytest.mark.unit
def test_hyperbolic_distance_positive() -> None:
    """Different points should have positive distance."""
    x = project_to_lorentz(np.array([0.1, 0.2]))
    y = project_to_lorentz(np.array([0.5, 0.3]))
    dist = hyperbolic_distance(x, y)
    assert dist > 0.0


@pytest.mark.unit
def test_batch_project() -> None:
    vectors = np.random.randn(5, 128)
    projected = batch_project(vectors)
    assert projected.shape == (5, 129)
    # Each row should be on the manifold
    for i in range(5):
        lorentz_norm = -projected[i, 0] ** 2 + np.sum(projected[i, 1:] ** 2)
        assert np.isclose(lorentz_norm, -1.0, atol=1e-10)
