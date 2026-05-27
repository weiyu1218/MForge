"""Unit tests for TaskAwareRouter (Layer 3 — TAR)."""

from __future__ import annotations

import pytest
import torch

from mf_core.routing.task_router import GENERATOR_NAMES, TaskAwareRouter, TaskProfile


class TestTaskAwareRouter:
    def _make_router(self) -> TaskAwareRouter:
        return TaskAwareRouter(hciv_dim=16, task_dim=8, hidden_dim=32, n_generators=8)

    def test_forward_returns_probability_distribution(self) -> None:
        router = self._make_router()
        hciv = torch.randn(16)
        profile = TaskProfile()

        weights = router.forward(hciv, profile)

        # Check all 8 generators present
        assert len(weights) == 8

        # Check weights sum to 1.0
        total = sum(weights.values())
        assert abs(total - 1.0) < 1e-5, f"Weights sum to {total}, expected 1.0"

        # Check all weights >= 0
        for name, w in weights.items():
            assert w >= 0.0, f"{name} weight {w} is negative"

    def test_hard_rules_scaffold_hop(self) -> None:
        router = self._make_router()
        hciv = torch.randn(16)
        profile = TaskProfile(stage="scaffold_hop")

        weights = router.forward(hciv, profile)

        # CReM should have very low weight in scaffold_hop mode
        assert weights["crem_3d"] < 0.01

    def test_hard_rules_low_data(self) -> None:
        router = self._make_router()
        hciv = torch.randn(16)
        profile_low = TaskProfile(data_richness=10.0)
        profile_high = TaskProfile(data_richness=200.0)

        weights_low = router.forward(hciv, profile_low)
        weights_high = router.forward(hciv, profile_high)

        # ICLM should have higher weight with low data
        assert weights_low["iclm"] >= weights_high["iclm"]

    def test_hard_rules_high_fto(self) -> None:
        router = self._make_router()
        hciv = torch.randn(16)
        profile_low = TaskProfile(fto_risk=0.1)
        profile_high = TaskProfile(fto_risk=0.9)

        weights_low = router.forward(hciv, profile_low)
        weights_high = router.forward(hciv, profile_high)

        # MMPT-RAG should have higher weight with high FTO risk
        assert weights_high["mmpt_rag"] >= weights_low["mmpt_rag"]

    def test_route_with_samples(self) -> None:
        router = self._make_router()
        hciv = torch.randn(16)
        profile = TaskProfile()

        allocation = router.route_with_samples(hciv, profile, total_samples=100)

        assert len(allocation) == 8
        total = sum(allocation.values())
        assert total == 100

        # All generators should get at least 1 sample
        for name, n in allocation.items():
            assert n >= 1

    def test_update_with_feedback(self) -> None:
        router = self._make_router()

        # Initial oracle history should be zeros
        assert router.oracle_history["hfm_3d"]["avg_hvi"] == 0.0

        # Update feedback
        router.update_with_feedback("hfm_3d", hvi_reward=0.5)
        router.update_with_feedback("hfm_3d", hvi_reward=0.3)

        # Check oracle history updated
        assert router.oracle_history["hfm_3d"]["n_calls"] == 2.0
        assert abs(router.oracle_history["hfm_3d"]["avg_hvi"] - 0.4) < 1e-6

    def test_oracle_history_in_forward(self) -> None:
        router = self._make_router()
        hciv = torch.randn(16)
        profile = TaskProfile()

        # Feed back some rewards
        for _ in range(5):
            router.update_with_feedback("hfm_3d", hvi_reward=0.8)
            router.update_with_feedback("iclm", hvi_reward=0.2)

        # Now forward should use history
        weights = router.forward(hciv, profile)

        # hfm_3d should have higher weight due to better history
        assert weights["hfm_3d"] > weights["iclm"]


class TestTaskProfile:
    def test_feature_vector_length(self) -> None:
        profile = TaskProfile()
        vec = profile.to_feature_vector()
        assert len(vec) == 8

    def test_feature_vector_values(self) -> None:
        profile = TaskProfile(
            target_family="GPCR",
            stage="lead_opt",
            fto_risk=0.5,
        )
        vec = profile.to_feature_vector()
        assert vec[0] == 0.2  # GPCR
        assert vec[5] == 0.5  # lead_opt
        assert vec[6] == 0.5  # fto_risk


@pytest.mark.asyncio
async def test_generator_router_service_returns_real_generator_names() -> None:
    from generator_router_svc.main import GeneratorRouterServicer

    request = type("RouteRequest", (), {"n_select": 4})()
    response = await GeneratorRouterServicer().Route(request, None)

    assert len(response.selected_generators) == 4
    assert set(response.selected_generators).issubset(GENERATOR_NAMES)
    assert not any(name.startswith("gen-") for name in response.selected_generators)
    assert len(response.selection_weights) == 4
