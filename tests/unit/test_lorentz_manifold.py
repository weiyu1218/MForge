"""Unit tests for LorentzManifold (Layer 2 — HUMU manifold operations)."""

from __future__ import annotations

import pytest
import torch


class TestLorentzManifold:
    def _get_manifold(self, curvature: float = 1.0):
        from mf_humu.manifold.lorentz import LorentzManifold
        return LorentzManifold(curvature=curvature)

    def test_expmap_logmap_roundtrip(self) -> None:
        """expmap -> logmap should recover the original tangent vector."""
        manifold = self._get_manifold()
        dim = 8
        origin = manifold.origin(dim)

        # Random tangent
        tangent = torch.zeros(dim + 1)
        tangent[1:] = torch.randn(dim) * 0.5

        # expmap -> logmap roundtrip
        point = manifold.expmap(origin, tangent)
        recovered = manifold.logmap(origin, point)

        # Check roundtrip error < 1e-5
        error = (tangent[1:] - recovered[1:]).abs().max().item()
        assert error < 1e-4, f"Roundtrip error {error} too large"

    def test_point_on_manifold(self) -> None:
        """Points produced by expmap should satisfy the manifold constraint."""
        manifold = self._get_manifold()
        dim = 8
        origin = manifold.origin(dim)

        tangent = torch.zeros(dim + 1)
        tangent[1:] = torch.randn(dim) * 0.5

        point = manifold.expmap(origin, tangent)

        # Check <x,x>_L = -1/c
        inner = manifold.inner(point, point, keepdim=False)
        expected = -1.0 / manifold.c
        assert torch.allclose(inner, torch.tensor(expected), atol=1e-5)

    def test_distance_nonnegative(self) -> None:
        """Distance should always be non-negative."""
        manifold = self._get_manifold()
        dim = 8
        origin = manifold.origin(dim)

        t1 = torch.zeros(dim + 1)
        t1[1:] = torch.randn(dim) * 0.5
        t2 = torch.zeros(dim + 1)
        t2[1:] = torch.randn(dim) * 0.5

        p1 = manifold.expmap(origin, t1)
        p2 = manifold.expmap(origin, t2)

        d = manifold.distance(p1, p2)
        assert (d >= 0).all()

    def test_project_tangent(self) -> None:
        """project_tangent should produce a vector in the tangent space."""
        manifold = self._get_manifold()
        dim = 8
        origin = manifold.origin(dim)

        # Random vector (not necessarily tangent)
        v = torch.randn(dim + 1)
        projected = manifold.project_tangent(origin, v)

        # Check <base, projected>_L ≈ 0
        inner = manifold.inner(origin, projected, keepdim=False)
        assert inner.abs().item() < 1e-5

    def test_origin_coordinates(self) -> None:
        """Origin should be (1/sqrt(c), 0, ..., 0)."""
        manifold = self._get_manifold(curvature=2.0)
        origin = manifold.origin(4)
        assert origin.shape == (5,)
        assert torch.allclose(origin[0], torch.tensor(1.0 / (2.0 ** 0.5)), atol=1e-6)
        assert (origin[1:] == 0).all()

    def test_different_curvatures(self) -> None:
        """Manifold should work with different curvature values."""
        for c in [0.5, 1.0, 2.0]:
            manifold = self._get_manifold(curvature=c)
            origin = manifold.origin(4)
            tangent = torch.zeros(5)
            tangent[1:] = torch.randn(4) * 0.3
            point = manifold.expmap(origin, tangent)

            # Verify constraint
            inner = manifold.inner(point, point, keepdim=False)
            expected = -1.0 / c
            assert torch.allclose(inner, torch.tensor(expected), atol=1e-5)
