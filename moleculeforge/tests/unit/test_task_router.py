"""Unit tests for TaskAwareRouter (Layer 3 — TAR)."""

from __future__ import annotations

import asyncio
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Never

import grpc
import pytest
import torch
from mf_core.routing.task_router import (
    GENERATOR_NAMES,
    ProxylessSearchScheduler,
    TaskAwareRouter,
    TaskProfile,
)

ROOT = Path(__file__).resolve().parents[2]


def _install_recording_docker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    docker_log = tmp_path / "docker.log"
    fake_docker = fake_bin / "docker"
    fake_docker.write_text(
        '#!/bin/sh\nprintf "%s\\n" "$*" >> "$DOCKER_CALL_LOG"\n',
        encoding="utf-8",
    )
    fake_docker.chmod(0o755)
    monkeypatch.setenv("PATH", f"{fake_bin}:/usr/bin:/bin")
    monkeypatch.setenv("DOCKER_CALL_LOG", str(docker_log))
    return docker_log


def _make_router_service(
    tmp_path: Path,
    *,
    state_name: str = "router-state.json",
) -> object:
    from generator_router_svc.main import GeneratorRouterServicer

    return GeneratorRouterServicer(
        state_path=tmp_path / state_name,
        bootstrap=True,
    )


def _valid_cig_bytes(project_id: str = "project-router") -> bytes:
    from mf_core.proto_gen.moleculeforge.v1.core import cig_pb2

    return cig_pb2.CIG(
        project_id=project_id,
        objectives=[
            cig_pb2.ObjectiveNode(
                id="qed",
                name="QED",
                type=cig_pb2.MAXIMIZE,
                property="qed",
                weight=1.0,
                pareto_tier=1,
            )
        ],
        created_by="test",
    ).SerializeToString()


def _cig_without_objectives_bytes(project_id: str = "project-router") -> bytes:
    from mf_core.proto_gen.moleculeforge.v1.core import cig_pb2

    return cig_pb2.CIG(project_id=project_id).SerializeToString()


def _valid_hciv(spatial: list[float] | None = None) -> list[float]:
    spatial = list(spatial or [0.0] * 128)
    return [math.sqrt(1.0 + sum(value * value for value in spatial)), *spatial]


def _valid_route_request(
    router_pb2: object,
    **overrides: object,
) -> object:
    values = {
        "project_id": "project-router",
        "run_id": "run-feedback",
        "cig": _valid_cig_bytes(),
        "hciv": _valid_hciv(),
        "request_id": "request-router",
        "n_select": 1,
        "n_samples": 1,
    }
    values.update(overrides)
    return router_pb2.RouterRequest(**values)


class TestTaskAwareRouter:
    def _make_router(self) -> TaskAwareRouter:
        return TaskAwareRouter(
            hciv_dim=16,
            task_dim=8,
            hidden_dim=32,
            n_generators=len(GENERATOR_NAMES),
        )

    def test_forward_returns_probability_distribution(self) -> None:
        router = self._make_router()
        hciv = torch.randn(16)
        profile = TaskProfile()

        weights = router.forward(hciv, profile)

        assert len(weights) == len(GENERATOR_NAMES)

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
        torch.manual_seed(0)
        router = self._make_router()
        hciv = torch.randn(16)
        profile_low = TaskProfile(data_richness=10.0)
        profile_high = TaskProfile(data_richness=200.0)

        weights_low = router.forward(hciv, profile_low)
        weights_high = router.forward(hciv, profile_high)

        # ICLM should have higher weight with low data
        assert weights_low["iclm"] >= weights_high["iclm"]

    def test_route_with_samples(self) -> None:
        router = self._make_router()
        hciv = torch.randn(16)
        profile = TaskProfile()

        allocation = router.route_with_samples(hciv, profile, total_samples=100)

        assert len(allocation) == len(GENERATOR_NAMES)
        total = sum(allocation.values())
        assert total == 100

        # All generators should get at least 1 sample
        for _name, n in allocation.items():
            assert n >= 1

    def test_route_with_samples_uses_minimum_one_largest_remainder(self) -> None:
        router = TaskAwareRouter(
            hciv_dim=2,
            task_dim=8,
            hidden_dim=2,
            n_generators=3,
        )
        router.forward = lambda _hciv, _profile: {
            "hfm_3d": 0.5,
            "fragfm": 0.3,
            "crem_3d": 0.2,
        }

        allocation = router.route_with_samples(
            torch.zeros(2),
            TaskProfile(),
            total_samples=11,
        )

        assert allocation == {
            "hfm_3d": 5,
            "fragfm": 3,
            "crem_3d": 3,
        }

    @pytest.mark.parametrize("total_samples", [-1, 0, 1, 2])
    def test_route_with_samples_rejects_too_few_samples(
        self,
        total_samples: int,
    ) -> None:
        router = TaskAwareRouter(
            hciv_dim=2,
            task_dim=8,
            hidden_dim=2,
            n_generators=3,
        )

        with pytest.raises(
            ValueError,
            match="total_samples must be at least the number of generators",
        ):
            router.route_with_samples(
                torch.zeros(2),
                TaskProfile(),
                total_samples=total_samples,
            )

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

    def test_update_with_feedback_applies_reinforce_policy_update(self) -> None:
        router = self._make_router()
        before = router.policy_logits.detach().clone()

        router.update_with_feedback(
            "hfm_3d",
            hvi_reward=0.8,
            baseline=0.2,
            learning_rate=0.5,
        )

        hfm_idx = GENERATOR_NAMES.index("hfm_3d")
        assert router.policy_logits[hfm_idx] > before[hfm_idx]
        assert torch.sum(router.policy_logits).item() == pytest.approx(0.0, abs=1e-7)

    def test_reinforce_policy_logits_affect_forward_weights(self) -> None:
        router = TaskAwareRouter(hciv_dim=2, task_dim=8, hidden_dim=2, n_generators=2)
        with torch.no_grad():
            router.projection.weight.zero_()
            router.projection.bias.zero_()
            router.task_projection.weight.zero_()
            router.task_projection.bias.zero_()
            router.gen_embeddings.zero_()
            router.policy_logits.zero_()

        before = router.forward(torch.zeros(2), TaskProfile())
        router.update_with_feedback(
            "hfm_3d",
            hvi_reward=1.0,
            baseline=0.0,
            learning_rate=1.0,
        )
        after = router.forward(torch.zeros(2), TaskProfile())

        assert before["hfm_3d"] == pytest.approx(before["fragfm"])
        assert after["hfm_3d"] > after["fragfm"]

    def test_proxyless_architecture_logits_gate_forward_weights(self) -> None:
        router = TaskAwareRouter(hciv_dim=2, task_dim=8, hidden_dim=2, n_generators=2)
        with torch.no_grad():
            router.projection.weight.zero_()
            router.projection.bias.zero_()
            router.task_projection.weight.zero_()
            router.task_projection.bias.zero_()
            router.gen_embeddings.zero_()
            router.policy_logits.zero_()
            router.architecture_logits.zero_()

        before = router.forward(torch.zeros(2), TaskProfile())
        with torch.no_grad():
            router.architecture_logits[1] = 2.0
        after = router.forward(torch.zeros(2), TaskProfile())

        assert before["hfm_3d"] == pytest.approx(before["fragfm"])
        assert after["fragfm"] > after["hfm_3d"]

    def test_proxyless_expected_cost_uses_architecture_probabilities(self) -> None:
        router = TaskAwareRouter(hciv_dim=2, task_dim=8, hidden_dim=2, n_generators=2)
        with torch.no_grad():
            router.architecture_logits[:] = torch.tensor([0.0, 2.0])

        expected_cost = router.proxyless_expected_cost({"hfm_3d": 10.0, "fragfm": 1.0})

        probabilities = torch.softmax(torch.tensor([0.0, 2.0]), dim=0)
        assert expected_cost.item() == pytest.approx(
            float(probabilities[0] * 10.0 + probabilities[1] * 1.0)
        )

    def test_proxyless_architecture_optimizer_step_updates_reward_cost_tradeoff(self) -> None:
        router = TaskAwareRouter(hciv_dim=2, task_dim=8, hidden_dim=2, n_generators=2)
        with torch.no_grad():
            router.architecture_logits.zero_()

        result = router.proxyless_architecture_optimizer_step(
            generator_rewards={"hfm_3d": 0.2, "fragfm": 0.8},
            generator_costs={"hfm_3d": 5.0, "fragfm": 1.0},
            cost_weight=0.1,
            learning_rate=1.0,
        )

        assert result["objective"] > 0.0
        assert router.architecture_logits[1] > router.architecture_logits[0]
        probabilities = router.proxyless_architecture_probabilities()
        assert probabilities["fragfm"] > probabilities["hfm_3d"]

    def test_proxyless_search_scheduler_runs_multi_dataset_rounds(self) -> None:
        router = TaskAwareRouter(hciv_dim=2, task_dim=8, hidden_dim=2, n_generators=2)
        with torch.no_grad():
            router.architecture_logits.zero_()

        scheduler = ProxylessSearchScheduler(
            router=router,
            generator_costs={"hfm_3d": 5.0, "fragfm": 1.0},
            cost_weight=0.1,
            learning_rate=1.0,
        )

        result = scheduler.run(
            {
                "kras": [
                    {"hfm_3d": 0.2, "fragfm": 0.8},
                    {"hfm_3d": 0.1, "fragfm": 0.9},
                ],
                "egfr": [
                    {"hfm_3d": 0.3, "fragfm": 0.7},
                ],
            }
        )

        assert [item["dataset"] for item in result["rounds"]] == ["kras", "kras", "egfr"]
        assert (
            result["architecture_probabilities"]["fragfm"]
            > (result["architecture_probabilities"]["hfm_3d"])
        )
        assert router.oracle_history["fragfm"]["n_calls"] == 3.0
        assert router.oracle_history["hfm_3d"]["n_calls"] == 3.0

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

    def test_forward_uses_task_profile_feature_vector(self) -> None:
        router = TaskAwareRouter(hciv_dim=2, task_dim=8, hidden_dim=2, n_generators=2)
        with torch.no_grad():
            router.projection.weight.zero_()
            router.projection.bias.zero_()
            router.task_projection.weight.zero_()
            router.task_projection.bias.zero_()
            router.task_projection.weight[0, 0] = 1.0
            router.gen_embeddings.zero_()
            router.gen_embeddings[0, 0] = 1.0
            router.gen_embeddings[1, 0] = -1.0

        unknown = router.forward(torch.zeros(2), TaskProfile(target_family=""))
        gpcr = router.forward(torch.zeros(2), TaskProfile(target_family="GPCR"))

        assert unknown["hfm_3d"] == pytest.approx(unknown["fragfm"])
        assert gpcr["hfm_3d"] > gpcr["fragfm"]


class TestTaskProfile:
    def test_feature_vector_length(self) -> None:
        profile = TaskProfile()
        vec = profile.to_feature_vector()
        assert len(vec) == 8

    def test_feature_vector_values(self) -> None:
        profile = TaskProfile(
            target_family="GPCR",
            stage="lead_opt",
        )
        vec = profile.to_feature_vector()
        assert vec[0] == 0.2  # GPCR
        assert vec[5] == 0.5  # lead_opt


@pytest.mark.asyncio
async def test_generator_router_service_returns_real_generator_names(
    tmp_path: Path,
) -> None:
    from mf_core.proto_gen.moleculeforge.v1.generator import router_pb2

    request = _valid_route_request(
        router_pb2,
        request_id="route-real-generators",
        n_select=4,
        n_samples=8,
    )
    response = await _make_router_service(tmp_path).Route(request, None)

    assert len(response.selected_generators) == 4
    assert set(response.selected_generators).issubset(GENERATOR_NAMES)
    assert not any(name.startswith("gen-") for name in response.selected_generators)
    assert len(response.selection_weights) == 4
    assert isinstance(response, router_pb2.RouterResponse)
    assert sum(allocation.n_samples for allocation in response.allocations) == 8


@pytest.mark.asyncio
async def test_generator_router_service_uses_request_hciv_and_profile(
    tmp_path: Path,
) -> None:
    from mf_core.proto_gen.moleculeforge.v1.generator import router_pb2

    service = _make_router_service(tmp_path)
    seen: dict[str, object] = {}

    def recording_forward(
        hciv: torch.Tensor,
        profile: TaskProfile,
        **_kwargs: object,
    ) -> dict[str, float]:
        seen["hciv"] = hciv
        seen["profile"] = profile
        return {name: 1.0 / len(GENERATOR_NAMES) for name in GENERATOR_NAMES}

    service.router.forward = recording_forward

    spatial = [1.0, 2.0, 3.0, *([0.0] * 125)]
    request = _valid_route_request(
        router_pb2,
        request_id="route-profile",
        n_select=2,
        hciv=_valid_hciv(spatial),
        target_family="kinase",
        stage="lead_opt",
        data_richness=25.0,
        novelty_demand=0.8,
        multi_target=True,
        sa_constraint=3.0,
        n_samples=32,
    )

    await service.Route(request, None)

    assert torch.equal(
        seen["hciv"],
        torch.tensor(spatial),
    )
    assert seen["profile"] == TaskProfile(
        target_family="kinase",
        stage="lead_opt",
        data_richness=25.0,
        novelty_demand=0.8,
        multi_target=True,
        sa_constraint=3.0,
        n_samples=32,
    )


@pytest.mark.asyncio
async def test_generator_router_service_deprecates_request_generator_performance(
    tmp_path: Path,
) -> None:
    from mf_core.proto_gen.moleculeforge.v1.generator import router_pb2

    service = _make_router_service(tmp_path)
    request_without_legacy = _valid_route_request(
        router_pb2,
        request_id="legacy-performance",
        n_select=6,
        n_samples=12,
    )
    request_with_legacy = _valid_route_request(
        router_pb2,
        request_id="legacy-performance",
        n_select=6,
        n_samples=12,
        generator_performance=[0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    )

    baseline = await service.Route(request_without_legacy, None)
    history_before = json.loads(json.dumps(service.router.oracle_history))
    response = await service.Route(request_with_legacy, None)
    weights = await service.GetWeights(request_with_legacy, None)

    assert list(response.selected_generators) == list(baseline.selected_generators)
    assert list(response.selection_weights) == pytest.approx(list(baseline.selection_weights))
    assert list(response.allocations) == list(baseline.allocations)
    assert response.state_version == baseline.state_version == weights.state_version
    assert service.router.oracle_history == history_before
    assert list(response.warnings) == ["generator_performance is deprecated and ignored"]


@pytest.mark.asyncio
async def test_generator_router_feedback_updates_kd_teacher_scores(
    tmp_path: Path,
) -> None:
    from mf_core.proto_gen.moleculeforge.v1.generator import router_pb2

    service = _make_router_service(tmp_path)
    await service.Route(
        _valid_route_request(
            router_pb2,
            run_id="run-feedback-kd",
            request_id="feedback-kd",
            n_select=len(GENERATOR_NAMES),
            n_samples=len(GENERATOR_NAMES),
        ),
        None,
    )
    request = router_pb2.RouterFeedbackRequest(
        feedback_id="feedback-kd-validation",
        run_id="run-feedback-kd",
        request_id="feedback-kd",
        iteration=1,
        phase=router_pb2.ROUTER_FEEDBACK_PHASE_VALIDATION,
        generator_name="hfm_3d",
        candidate_ids=["candidate-1"],
        canonical_smiles="CCO",
        evidence_ids=["evidence-1"],
        teacher_score=0.7,
        teacher_source="validation-adapter",
        teacher_version="1",
    )

    response = await service.SubmitFeedback(request, None)

    assert response.acknowledged is True
    assert response.duplicate is False
    assert service.kd_layer.running_counts[0].item() == 1.0
    assert service.router.oracle_history["hfm_3d"]["avg_hvi"] == pytest.approx(0.7)


@pytest.mark.asyncio
async def test_generator_router_feedback_does_not_invoke_hypseek_teacher_command(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from mf_core.proto_gen.moleculeforge.v1.generator import router_pb2

    command = f'{sys.executable} -c "raise SystemExit(17)"'
    monkeypatch.setenv("HYPSEEK_TEACHER_COMMAND", command)
    service = _make_router_service(tmp_path)
    await service.Route(
        _valid_route_request(
            router_pb2,
            run_id="run-feedback-command",
            request_id="feedback-command",
            n_select=len(GENERATOR_NAMES),
            n_samples=len(GENERATOR_NAMES),
        ),
        None,
    )
    request = router_pb2.RouterFeedbackRequest(
        feedback_id="feedback-command-validation",
        run_id="run-feedback-command",
        request_id="feedback-command",
        phase=router_pb2.ROUTER_FEEDBACK_PHASE_VALIDATION,
        generator_name="fragfm",
        candidate_ids=["candidate-1"],
        canonical_smiles="CCN",
        evidence_ids=["evidence-1"],
        teacher_score=0.25,
        teacher_source="coordinator",
        teacher_version="1",
    )

    response = await service.SubmitFeedback(request, None)

    assert response.acknowledged is True
    assert service.kd_layer.running_counts[1].item() == 1.0
    assert service.router.oracle_history["fragfm"]["avg_hvi"] == pytest.approx(0.25)


@pytest.mark.asyncio
async def test_generator_router_feedback_does_not_invoke_hypseek_teacher_url(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import generator_router_svc.main as router_main
    from mf_core.proto_gen.moleculeforge.v1.generator import router_pb2

    monkeypatch.delenv("HYPSEEK_TEACHER_COMMAND", raising=False)
    monkeypatch.setenv("HYPSEEK_TEACHER_URL", "https://hypseek.example/teacher")
    calls = []

    def post_json(url: str, payload: dict, timeout_seconds: float) -> dict:
        calls.append(
            {
                "url": url,
                "payload": payload,
                "timeout_seconds": timeout_seconds,
            }
        )
        return {"oracle_name": "hypseek", "teacher_distribution": [0.1, 0.9]}

    monkeypatch.setattr(router_main, "_post_json", post_json, raising=False)
    service = router_main.GeneratorRouterServicer(
        state_path=tmp_path / "router-state.json",
        bootstrap=True,
    )
    await service.Route(
        _valid_route_request(
            router_pb2,
            run_id="run-feedback-url",
            request_id="feedback-url",
            n_select=len(GENERATOR_NAMES),
            n_samples=len(GENERATOR_NAMES),
        ),
        None,
    )
    request = router_pb2.RouterFeedbackRequest(
        feedback_id="feedback-url-validation",
        run_id="run-feedback-url",
        request_id="feedback-url",
        phase=router_pb2.ROUTER_FEEDBACK_PHASE_VALIDATION,
        generator_name="fragfm",
        candidate_ids=["candidate-1"],
        canonical_smiles="CCN",
        evidence_ids=["evidence-1"],
        teacher_score=0.9,
        teacher_source="coordinator",
        teacher_version="1",
    )

    response = await service.SubmitFeedback(request, None)

    assert calls == []
    assert response.acknowledged is True
    assert service.kd_layer.running_counts[1].item() == 1.0
    assert service.router.oracle_history["fragfm"]["avg_hvi"] == pytest.approx(0.9)


@pytest.mark.asyncio
async def test_generator_router_service_runs_proxyless_search_request(
    tmp_path: Path,
) -> None:
    from mf_core.proto_gen.moleculeforge.v1.generator import router_pb2

    service = _make_router_service(tmp_path)
    request = router_pb2.RouterProxylessSearchRequest(
        reward_batches_json=json.dumps(
            {
                "kras": [
                    _full_proxyless_rewards(),
                    {**_full_proxyless_rewards(), "fragfm": 1.0},
                ]
            }
        ),
        generator_costs_json=json.dumps(_full_proxyless_costs()),
        cost_weight=0.1,
        learning_rate=1.0,
        temperature=1.0,
    )

    response = await service.RunProxylessSearch(request, None)
    result = json.loads(response.result_json)

    assert response.acknowledged is True
    assert response.round_count == 2
    assert list(response.generator_names) == list(GENERATOR_NAMES)
    assert len(response.architecture_probabilities) == len(GENERATOR_NAMES)
    assert result["rounds"][0]["dataset"] == "kras"
    assert (
        result["architecture_probabilities"]["fragfm"]
        > (result["architecture_probabilities"]["hfm_3d"])
    )
    assert service.router.oracle_history["fragfm"]["n_calls"] == 2.0


@pytest.mark.asyncio
async def test_generator_router_service_rejects_stateless_proxyless_search_command(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from mf_core.proto_gen.moleculeforge.v1.generator import router_pb2

    command = (
        f"{sys.executable} -c "
        '"import json,sys; '
        "payload=json.load(sys.stdin); "
        "assert payload['reward_batches_by_dataset']['kras'][0]['fragfm'] == 0.8; "
        "assert payload['generator_costs']['fragfm'] == 1.0; "
        "print(json.dumps({'rounds':[{'dataset':'kras','round_index':0}],"
        "'architecture_probabilities':{'hfm_3d':0.25,'fragfm':0.75}}))\""
    )
    monkeypatch.setenv("TAR_PROXYLESS_SEARCH_COMMAND", command)
    monkeypatch.setenv("TAR_PROXYLESS_SEARCH_TIMEOUT_SECONDS", "10")
    service = _make_router_service(tmp_path)
    request = router_pb2.RouterProxylessSearchRequest(
        reward_batches_json=json.dumps({"kras": [_full_proxyless_rewards()]}),
        generator_costs_json=json.dumps(_full_proxyless_costs()),
        cost_weight=0.1,
        learning_rate=1.0,
        temperature=1.0,
    )

    with pytest.raises(
        RuntimeError,
        match="external Proxyless search cannot atomically update Router state",
    ):
        await service.RunProxylessSearch(request, None)


def _proxyless_runner_payload() -> dict[str, object]:
    return {
        "reward_batches_by_dataset": {
            "kras": [
                _full_proxyless_rewards(),
                {**_full_proxyless_rewards(), "fragfm": 1.0},
            ]
        },
        "generator_costs": _full_proxyless_costs(),
        "cost_weight": 0.1,
        "learning_rate": 1.0,
        "temperature": 1.0,
    }


def test_tar_proxyless_runner_executes_shared_scheduler() -> None:
    from generator_router_svc.tar_proxyless_runner import run_proxyless_search

    result = run_proxyless_search(_proxyless_runner_payload())

    assert len(result["rounds"]) == 2
    assert (
        result["architecture_probabilities"]["fragfm"]
        > (result["architecture_probabilities"]["hfm_3d"])
    )
    assert set(result["architecture_logits"]) == set(GENERATOR_NAMES)
    assert result["generator_names"] == list(GENERATOR_NAMES)


def test_tar_proxyless_runner_cli_reads_stdin_and_writes_json() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "generator_router_svc.tar_proxyless_runner"],
        input=json.dumps(_proxyless_runner_payload()),
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0
    assert completed.stderr == ""
    result = json.loads(completed.stdout)
    assert len(result["rounds"]) == 2
    assert (
        result["architecture_probabilities"]["fragfm"]
        > (result["architecture_probabilities"]["hfm_3d"])
    )


@pytest.mark.asyncio
async def test_generator_router_service_rejects_builtin_stateless_runner_command(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from mf_core.proto_gen.moleculeforge.v1.generator import router_pb2

    monkeypatch.setenv(
        "TAR_PROXYLESS_SEARCH_COMMAND",
        f"{sys.executable} -m generator_router_svc.tar_proxyless_runner",
    )
    monkeypatch.setenv("TAR_PROXYLESS_SEARCH_TIMEOUT_SECONDS", "120")
    service = _make_router_service(tmp_path)
    request = router_pb2.RouterProxylessSearchRequest(
        reward_batches_json=json.dumps(_proxyless_runner_payload()["reward_batches_by_dataset"]),
        generator_costs_json=json.dumps(_proxyless_runner_payload()["generator_costs"]),
        cost_weight=0.1,
        learning_rate=1.0,
        temperature=1.0,
    )

    with pytest.raises(
        RuntimeError,
        match="external Proxyless search cannot atomically update Router state",
    ):
        await service.RunProxylessSearch(request, None)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("n_select", "n_samples", "message"),
    [
        (0, 1, "n_select must be positive"),
        (-1, 1, "n_select must be positive"),
        (7, 7, "n_select exceeds eligible generators"),
        (2, 0, "n_samples must be positive"),
        (2, -1, "n_samples must be positive"),
        (3, 2, "n_samples must be at least n_select"),
    ],
)
async def test_generator_router_rejects_invalid_counts_before_state_change(
    tmp_path: Path,
    n_select: int,
    n_samples: int,
    message: str,
) -> None:
    from mf_core.proto_gen.moleculeforge.v1.generator import router_pb2

    state_path = tmp_path / "router-state.json"
    service = _make_router_service(tmp_path)
    state_before = state_path.read_bytes()

    with pytest.raises(ValueError, match=message):
        await service.Route(
            _valid_route_request(
                router_pb2,
                request_id="invalid-counts",
                n_select=n_select,
                n_samples=n_samples,
            ),
            None,
        )

    assert state_path.read_bytes() == state_before


def test_router_profile_preserves_explicit_zero_values() -> None:
    from generator_router_svc.main import _profile_from_request
    from mf_core.proto_gen.moleculeforge.v1.generator import router_pb2

    profile = _profile_from_request(
        router_pb2.RouterRequest(
            data_richness=0.0,
            novelty_demand=0.0,
            sa_constraint=0.0,
        )
    )

    assert profile.data_richness == 0.0
    assert profile.novelty_demand == 0.0
    assert profile.sa_constraint == 0.0


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("data_richness", -1.0, "data_richness"),
        ("novelty_demand", -0.1, "novelty_demand"),
        ("novelty_demand", 1.1, "novelty_demand"),
        ("sa_constraint", -1.0, "sa_constraint"),
    ],
)
def test_router_profile_rejects_out_of_range_values(
    field: str,
    value: float,
    message: str,
) -> None:
    from generator_router_svc.main import _profile_from_request
    from mf_core.proto_gen.moleculeforge.v1.generator import router_pb2

    request = router_pb2.RouterRequest()
    setattr(request, field, value)

    with pytest.raises(ValueError, match=message):
        _profile_from_request(request)


@pytest.mark.parametrize("length", [1, 3, 127, 130])
def test_router_rejects_noncanonical_hciv_lengths(length: int) -> None:
    from generator_router_svc.main import _hciv_from_request
    from mf_core.proto_gen.moleculeforge.v1.generator import router_pb2

    with pytest.raises(ValueError, match="hciv"):
        _hciv_from_request(
            router_pb2.RouterRequest(hciv=[0.0] * length),
            128,
        )


@pytest.mark.asyncio
async def test_router_requires_request_id_before_state_change(tmp_path: Path) -> None:
    from mf_core.proto_gen.moleculeforge.v1.generator import router_pb2

    state_path = tmp_path / "router-state.json"
    service = _make_router_service(tmp_path)
    state_before = state_path.read_bytes()

    with pytest.raises(ValueError, match="request_id is required"):
        await service.Route(
            _valid_route_request(router_pb2, request_id="", n_select=1, n_samples=1),
            None,
        )

    assert state_path.read_bytes() == state_before


@pytest.mark.asyncio
async def test_router_rejects_request_id_reuse_with_different_n_select(
    tmp_path: Path,
) -> None:
    from mf_core.proto_gen.moleculeforge.v1.generator import router_pb2

    service = _make_router_service(tmp_path)
    await service.Route(
        _valid_route_request(
            router_pb2,
            request_id="stable-request",
            n_select=1,
            n_samples=2,
        ),
        None,
    )

    with pytest.raises(
        ValueError,
        match="request_id is already bound to a different routing context",
    ):
        await service.Route(
            _valid_route_request(
                router_pb2,
                request_id="stable-request",
                n_select=2,
                n_samples=2,
            ),
            None,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("complexity", "expected"),
    [
        (1, ["hfm_3d"]),
        (2, ["hfm_3d", "fragfm"]),
        (3, ["fragfm", "mmpt_rag"]),
        (0, list(GENERATOR_NAMES)),
    ],
)
async def test_generator_router_filters_by_task_complexity_in_canonical_order(
    tmp_path: Path,
    complexity: int,
    expected: list[str],
) -> None:
    from mf_core.proto_gen.moleculeforge.v1.generator import router_pb2

    service = _make_router_service(tmp_path, state_name=f"complexity-{complexity}.json")
    service.router.forward = lambda *_args, **_kwargs: {name: 1.0 for name in GENERATOR_NAMES}
    response = await service.Route(
        _valid_route_request(
            router_pb2,
            request_id=f"complexity-{complexity}",
            n_select=len(expected),
            n_samples=len(expected),
            available_generator_names=list(GENERATOR_NAMES),
            task_complexity=complexity,
        ),
        None,
    )

    assert list(response.selected_generators) == expected
    assert [item.generator_name for item in response.allocations] == expected


@pytest.mark.asyncio
async def test_generator_router_intersects_complexity_with_available_generators(
    tmp_path: Path,
) -> None:
    from mf_core.proto_gen.moleculeforge.v1.generator import router_pb2

    service = _make_router_service(tmp_path)
    response = await service.Route(
        _valid_route_request(
            router_pb2,
            request_id="available-intersection",
            n_select=1,
            n_samples=2,
            available_generator_names=["crem_3d", "fragfm"],
            task_complexity=router_pb2.TASK_COMPLEXITY_MEDIUM,
        ),
        None,
    )

    assert list(response.selected_generators) == ["fragfm"]
    assert [item.generator_name for item in response.allocations] == ["fragfm"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("available", "message"),
    [
        (["unknown"], "unknown available generator"),
        (["hfm_3d", "hfm_3d"], "available generator names must be unique"),
        (["crem_3d"], "no eligible generators"),
    ],
)
async def test_generator_router_rejects_invalid_available_generators(
    tmp_path: Path,
    available: list[str],
    message: str,
) -> None:
    from mf_core.proto_gen.moleculeforge.v1.generator import router_pb2

    service = _make_router_service(tmp_path, state_name=f"available-{len(available)}.json")

    with pytest.raises(ValueError, match=message):
        await service.Route(
            _valid_route_request(
                router_pb2,
                request_id="invalid-available",
                n_select=1,
                n_samples=1,
                available_generator_names=available,
                task_complexity=router_pb2.TASK_COMPLEXITY_MEDIUM,
            ),
            None,
        )


@pytest.mark.asyncio
async def test_generator_router_all_zero_eligible_weights_use_uniform_distribution(
    tmp_path: Path,
) -> None:
    from mf_core.proto_gen.moleculeforge.v1.generator import router_pb2

    service = _make_router_service(tmp_path)
    service.router.forward = lambda *_args, **_kwargs: {name: 0.0 for name in GENERATOR_NAMES}
    request = _valid_route_request(
        router_pb2,
        request_id="zero-weights",
        n_select=3,
        n_samples=5,
        available_generator_names=list(GENERATOR_NAMES[:3]),
    )

    route = await service.Route(request, None)
    weights = await service.GetWeights(request, None)

    assert list(route.selected_generators) == list(GENERATOR_NAMES[:3])
    assert list(route.selection_weights) == pytest.approx([1.0 / 3.0] * 3)
    assert [item.n_samples for item in route.allocations] == [2, 2, 1]
    assert list(weights.generator_names) == list(GENERATOR_NAMES[:3])
    assert list(weights.weights) == pytest.approx([1.0 / 3.0] * 3)


@pytest.mark.asyncio
async def test_route_and_get_weights_share_context_snapshot_and_state_version(
    tmp_path: Path,
) -> None:
    from mf_core.proto_gen.moleculeforge.v1.generator import router_pb2

    service = _make_router_service(tmp_path)
    request = _valid_route_request(
        router_pb2,
        request_id="same-context",
        n_select=6,
        n_samples=12,
        hciv=_valid_hciv([0.25] * 128),
        target_family="GPCR",
        stage="lead_opt",
        data_richness=80.0,
        novelty_demand=0.9,
        multi_target=True,
        sa_constraint=3.0,
        generator_weights=[0.0] * len(GENERATOR_NAMES),
        available_generator_names=list(reversed(GENERATOR_NAMES)),
    )

    route = await service.Route(request, None)
    weights = await service.GetWeights(request, None)

    route_map = dict(zip(route.selected_generators, route.selection_weights, strict=True))
    weight_map = dict(zip(weights.generator_names, weights.weights, strict=True))
    assert list(weights.generator_names) == list(GENERATOR_NAMES)
    assert route_map == pytest.approx(weight_map)
    assert route.state_version == weights.state_version


@pytest.mark.asyncio
async def test_get_weights_first_call_persists_and_replays_full_deterministic_snapshot(
    tmp_path: Path,
) -> None:
    from generator_router_svc.main import GeneratorRouterServicer
    from mf_core.proto_gen.moleculeforge.v1.generator import router_pb2

    state_path = tmp_path / "router-state.json"
    service = _make_router_service(tmp_path)
    request = _valid_route_request(
        router_pb2,
        request_id="weights-first-snapshot",
        n_select=2,
        n_samples=5,
    )
    service.router.forward = lambda *_args, **_kwargs: {
        name: (6.0 if name == "hfm_3d" else 1.0) for name in GENERATOR_NAMES
    }

    first = await service.GetWeights(request, None)
    persisted = json.loads(state_path.read_text())
    snapshot = persisted["request_route_snapshots"]["weights-first-snapshot"]
    service.router.forward = lambda *_args, **_kwargs: {
        name: (6.0 if name == "fragfm" else 1.0) for name in GENERATOR_NAMES
    }
    retry = await service.GetWeights(request, None)
    route = await service.Route(request, None)
    restored = GeneratorRouterServicer(state_path=state_path, bootstrap=False)
    restored_weights = await restored.GetWeights(request, None)

    assert retry == first
    assert restored_weights == first
    assert route.state_version == first.state_version
    assert list(route.selected_generators) == ["hfm_3d", "fragfm"]
    assert snapshot["n_samples"] == 5
    assert snapshot["n_select"] == 2
    assert snapshot["run_id"] == "run-feedback"
    assert len(snapshot["context_key"]) == 64
    assert sum(snapshot["allocations"].values()) == 5


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "performance",
    [
        [0.5],
        [0.0] * 7,
        [0.0, 0.0, 0.0, 0.0, 0.0, float("nan")],
        [0.0, 0.0, 0.0, 0.0, 0.0, float("inf")],
    ],
)
async def test_generator_router_rejects_invalid_deprecated_performance_without_mutation(
    tmp_path: Path,
    performance: list[float],
) -> None:
    from mf_core.proto_gen.moleculeforge.v1.generator import router_pb2

    state_path = tmp_path / "router-state.json"
    service = _make_router_service(tmp_path)
    state_before = state_path.read_bytes()
    history_before = json.loads(json.dumps(service.router.oracle_history))

    with pytest.raises(ValueError, match="generator_performance"):
        await service.Route(
            _valid_route_request(
                router_pb2,
                request_id="invalid-performance",
                n_select=1,
                n_samples=1,
                generator_performance=performance,
            ),
            None,
        )

    assert state_path.read_bytes() == state_before
    assert service.router.oracle_history == history_before


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "request_fields",
    [
        pytest.param(
            {"hciv": [float("nan")]},
            id="hciv-nan",
        ),
        pytest.param(
            {"generator_weights": [0.0] * 5 + [float("inf")]},
            id="generator-weight-inf",
        ),
    ],
)
async def test_generator_router_rejects_nonfinite_request_values(
    tmp_path: Path,
    request_fields: dict[str, list[float]],
) -> None:
    from mf_core.proto_gen.moleculeforge.v1.generator import router_pb2

    service = _make_router_service(tmp_path)

    with pytest.raises(ValueError, match="finite"):
        await service.Route(
            _valid_route_request(
                router_pb2,
                request_id="nonfinite",
                n_select=1,
                n_samples=1,
                **request_fields,
            ),
            None,
        )


def _typed_feedback(
    router_pb2: object,
    *,
    feedback_id: str,
    request_id: str,
    generator_name: str,
    score: float,
    run_id: str = "run-feedback",
) -> object:
    return router_pb2.RouterFeedbackRequest(
        feedback_id=feedback_id,
        run_id=run_id,
        request_id=request_id,
        iteration=1,
        phase=router_pb2.ROUTER_FEEDBACK_PHASE_VALIDATION,
        generator_name=generator_name,
        candidate_ids=[f"candidate-{feedback_id}"],
        canonical_smiles="CCO",
        evidence_ids=[f"evidence-{feedback_id}"],
        teacher_score=score,
        teacher_source="coordinator",
        teacher_version="1",
    )


@pytest.mark.asyncio
async def test_generator_router_feedback_run_id_must_match_routed_request(
    tmp_path: Path,
) -> None:
    from mf_core.proto_gen.moleculeforge.v1.generator import router_pb2

    state_path = tmp_path / "router-state.json"
    service = _make_router_service(tmp_path)
    await service.Route(
        _valid_route_request(
            router_pb2,
            run_id="run-routed",
            request_id="run-bound-feedback",
            n_select=len(GENERATOR_NAMES),
            n_samples=len(GENERATOR_NAMES),
        ),
        None,
    )
    state_before = state_path.read_bytes()
    history_before = json.loads(json.dumps(service.router.oracle_history))

    with pytest.raises(ValueError, match="run_id"):
        await service.SubmitFeedback(
            _typed_feedback(
                router_pb2,
                feedback_id="wrong-run-feedback",
                run_id="other-run",
                request_id="run-bound-feedback",
                generator_name="hfm_3d",
                score=0.5,
            ),
            None,
        )

    assert state_path.read_bytes() == state_before
    assert service.router.oracle_history == history_before


@pytest.mark.asyncio
async def test_generator_router_request_id_cannot_be_reused_across_runs(
    tmp_path: Path,
) -> None:
    from mf_core.proto_gen.moleculeforge.v1.generator import router_pb2

    state_path = tmp_path / "router-state.json"
    service = _make_router_service(tmp_path)
    first_request = _valid_route_request(
        router_pb2,
        run_id="run-a",
        request_id="run-bound-route",
    )
    await service.Route(first_request, None)
    state_before = state_path.read_bytes()

    with pytest.raises(ValueError, match="run_id"):
        await service.Route(
            _valid_route_request(
                router_pb2,
                run_id="run-b",
                request_id="run-bound-route",
            ),
            None,
        )

    assert state_path.read_bytes() == state_before


@pytest.mark.asyncio
async def test_generator_router_legacy_route_without_run_id_requires_binding_before_feedback(
    tmp_path: Path,
) -> None:
    from mf_core.proto_gen.moleculeforge.v1.generator import router_pb2

    service = _make_router_service(tmp_path)
    route_request = _valid_route_request(
        router_pb2,
        run_id="",
        request_id="legacy-route",
        n_select=len(GENERATOR_NAMES),
        n_samples=len(GENERATOR_NAMES),
    )

    response = await service.Route(route_request, None)

    assert response.selected_generators
    assert service.request_route_snapshots["legacy-route"]["run_id"] is None
    feedback = _typed_feedback(
        router_pb2,
        feedback_id="legacy-feedback",
        run_id="bound-run",
        request_id="legacy-route",
        generator_name=response.selected_generators[0],
        score=0.5,
    )
    with pytest.raises(ValueError, match="routed again"):
        await service.SubmitFeedback(feedback, None)

    route_request.run_id = "bound-run"
    await service.Route(route_request, None)
    accepted = await service.SubmitFeedback(feedback, None)

    assert accepted.acknowledged is True
    assert service.request_route_snapshots["legacy-route"]["run_id"] == "bound-run"


@pytest.mark.asyncio
@pytest.mark.parametrize("method_name", ["Route", "GetWeights"])
async def test_generator_router_bound_request_accepts_legacy_read_without_run_id(
    tmp_path: Path,
    method_name: str,
) -> None:
    from mf_core.proto_gen.moleculeforge.v1.generator import router_pb2

    state_path = tmp_path / "router-state.json"
    service = _make_router_service(tmp_path)
    request = _valid_route_request(
        router_pb2,
        run_id="bound-run",
        request_id="bound-legacy-read",
    )
    await service.Route(request, None)
    state_before = state_path.read_bytes()
    version_before = service.state_version
    request.run_id = ""

    await getattr(service, method_name)(request, None)

    assert state_path.read_bytes() == state_before
    assert service.state_version == version_before
    assert service.request_route_snapshots["bound-legacy-read"]["run_id"] == "bound-run"


@pytest.mark.asyncio
async def test_generator_router_run_binding_survives_restart(
    tmp_path: Path,
) -> None:
    from generator_router_svc.main import GeneratorRouterServicer
    from mf_core.proto_gen.moleculeforge.v1.generator import router_pb2

    state_path = tmp_path / "router-state.json"
    service = _make_router_service(tmp_path)
    await service.Route(
        _valid_route_request(
            router_pb2,
            run_id="run-persisted",
            request_id="persisted-run-binding",
            n_select=len(GENERATOR_NAMES),
            n_samples=len(GENERATOR_NAMES),
        ),
        None,
    )
    restored = GeneratorRouterServicer(state_path=state_path, bootstrap=False)
    state_before = state_path.read_bytes()

    with pytest.raises(ValueError, match="run_id"):
        await restored.SubmitFeedback(
            _typed_feedback(
                router_pb2,
                feedback_id="restart-wrong-run",
                run_id="other-run",
                request_id="persisted-run-binding",
                generator_name="hfm_3d",
                score=0.5,
            ),
            None,
        )

    assert state_path.read_bytes() == state_before
    assert restored.router.oracle_history["hfm_3d"]["n_calls"] == 0.0


@pytest.mark.asyncio
async def test_generator_router_feedback_is_idempotent(
    tmp_path: Path,
) -> None:
    from mf_core.proto_gen.moleculeforge.v1.generator import router_pb2

    state_path = tmp_path / "router-state.json"
    service = _make_router_service(tmp_path)
    await service.Route(
        _valid_route_request(
            router_pb2,
            request_id="feedback-idempotent",
            n_select=len(GENERATOR_NAMES),
            n_samples=len(GENERATOR_NAMES),
        ),
        None,
    )
    request = _typed_feedback(
        router_pb2,
        feedback_id="feedback-1",
        request_id="feedback-idempotent",
        generator_name="hfm_3d",
        score=0.0,
    )

    first = await service.SubmitFeedback(request, None)
    state_after_first = state_path.read_bytes()
    second = await service.SubmitFeedback(request, None)

    assert first.acknowledged is True
    assert first.duplicate is False
    assert second.acknowledged is True
    assert second.duplicate is True
    assert second.state_version == first.state_version
    assert state_path.read_bytes() == state_after_first
    assert service.router.oracle_history["hfm_3d"]["n_calls"] == 1.0
    assert service.router.oracle_history["hfm_3d"]["avg_hvi"] == 0.0


@pytest.mark.asyncio
async def test_generator_router_semantic_feedback_idempotency_survives_restart(
    tmp_path: Path,
) -> None:
    from generator_router_svc.main import GeneratorRouterServicer
    from mf_core.proto_gen.moleculeforge.v1.generator import router_pb2

    state_path = tmp_path / "router-state.json"
    service = _make_router_service(tmp_path)
    await service.Route(
        _valid_route_request(
            router_pb2,
            request_id="feedback-semantic-retry",
            n_select=len(GENERATOR_NAMES),
            n_samples=len(GENERATOR_NAMES),
        ),
        None,
    )
    original = _typed_feedback(
        router_pb2,
        feedback_id="caller-attempt-1",
        request_id="feedback-semantic-retry",
        generator_name="hfm_3d",
        score=0.4,
    )
    first = await service.SubmitFeedback(original, None)
    restored = GeneratorRouterServicer(state_path=state_path, bootstrap=False)
    retry = router_pb2.RouterFeedbackRequest()
    retry.CopyFrom(original)
    retry.feedback_id = "caller-attempt-2"

    second = await restored.SubmitFeedback(retry, None)

    assert second.acknowledged is True
    assert second.duplicate is True
    assert second.state_version == first.state_version
    assert restored.router.oracle_history["hfm_3d"]["n_calls"] == 1.0


@pytest.mark.asyncio
async def test_generator_router_rejects_changed_content_for_semantic_feedback_identity(
    tmp_path: Path,
) -> None:
    from generator_router_svc.main import GeneratorRouterServicer
    from mf_core.proto_gen.moleculeforge.v1.generator import router_pb2

    state_path = tmp_path / "router-state.json"
    service = _make_router_service(tmp_path)
    await service.Route(
        _valid_route_request(
            router_pb2,
            request_id="feedback-semantic-conflict",
            n_select=len(GENERATOR_NAMES),
            n_samples=len(GENERATOR_NAMES),
        ),
        None,
    )
    original = _typed_feedback(
        router_pb2,
        feedback_id="caller-attempt-1",
        request_id="feedback-semantic-conflict",
        generator_name="hfm_3d",
        score=0.4,
    )
    await service.SubmitFeedback(original, None)
    restored = GeneratorRouterServicer(state_path=state_path, bootstrap=False)
    changed = router_pb2.RouterFeedbackRequest()
    changed.CopyFrom(original)
    changed.feedback_id = "caller-attempt-2"
    changed.teacher_score = 0.9

    with pytest.raises(ValueError, match="semantic identity"):
        await restored.SubmitFeedback(changed, None)

    assert restored.router.oracle_history["hfm_3d"]["n_calls"] == 1.0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mutation",
    ["request_id", "generator_name", "score", "candidate_ids", "phase"],
)
async def test_generator_router_rejects_feedback_id_reuse_with_changed_payload(
    tmp_path: Path,
    mutation: str,
) -> None:
    from mf_core.proto_gen.moleculeforge.v1.generator import router_pb2

    service = _make_router_service(tmp_path)
    for request_id in ("feedback-original", "feedback-other"):
        await service.Route(
            _valid_route_request(
                router_pb2,
                request_id=request_id,
                n_select=len(GENERATOR_NAMES),
                n_samples=len(GENERATOR_NAMES),
            ),
            None,
        )
    original = _typed_feedback(
        router_pb2,
        feedback_id="feedback-bound",
        request_id="feedback-original",
        generator_name="hfm_3d",
        score=0.4,
    )
    await service.SubmitFeedback(original, None)
    changed = router_pb2.RouterFeedbackRequest()
    changed.CopyFrom(original)
    if mutation == "request_id":
        changed.request_id = "feedback-other"
    elif mutation == "generator_name":
        changed.generator_name = "fragfm"
    elif mutation == "score":
        changed.teacher_score = 0.9
    elif mutation == "candidate_ids":
        changed.candidate_ids[:] = ["different-candidate"]
    else:
        changed.phase = router_pb2.ROUTER_FEEDBACK_PHASE_CRITIC

    with pytest.raises(ValueError, match="different payload"):
        await service.SubmitFeedback(changed, None)

    assert service.router.oracle_history["hfm_3d"]["n_calls"] == 1.0
    assert service.router.oracle_history["fragfm"]["n_calls"] == 0.0


@pytest.mark.asyncio
async def test_generator_router_feedback_payload_binding_survives_restart(
    tmp_path: Path,
) -> None:
    from generator_router_svc.main import GeneratorRouterServicer
    from mf_core.proto_gen.moleculeforge.v1.generator import router_pb2

    state_path = tmp_path / "router-state.json"
    service = _make_router_service(tmp_path)
    await service.Route(
        _valid_route_request(
            router_pb2,
            request_id="feedback-restart",
            n_select=len(GENERATOR_NAMES),
            n_samples=len(GENERATOR_NAMES),
        ),
        None,
    )
    original = _typed_feedback(
        router_pb2,
        feedback_id="feedback-restart-bound",
        request_id="feedback-restart",
        generator_name="hfm_3d",
        score=0.4,
    )
    await service.SubmitFeedback(original, None)
    restored = GeneratorRouterServicer(state_path=state_path, bootstrap=False)
    changed = router_pb2.RouterFeedbackRequest()
    changed.CopyFrom(original)
    changed.teacher_score = 0.8

    with pytest.raises(ValueError, match="different payload"):
        await restored.SubmitFeedback(changed, None)


@pytest.mark.asyncio
async def test_generator_router_concurrent_feedback_versions_are_contiguous(
    tmp_path: Path,
) -> None:
    from mf_core.proto_gen.moleculeforge.v1.generator import router_pb2

    service = _make_router_service(tmp_path)
    route = await service.Route(
        _valid_route_request(
            router_pb2,
            request_id="feedback-concurrent",
            n_select=len(GENERATOR_NAMES),
            n_samples=len(GENERATOR_NAMES),
        ),
        None,
    )
    requests = [
        _typed_feedback(
            router_pb2,
            feedback_id="feedback-concurrent-1",
            request_id="feedback-concurrent",
            generator_name="hfm_3d",
            score=0.2,
        ),
        _typed_feedback(
            router_pb2,
            feedback_id="feedback-concurrent-2",
            request_id="feedback-concurrent",
            generator_name="fragfm",
            score=0.8,
        ),
    ]

    responses = await asyncio.gather(
        *(service.SubmitFeedback(request, None) for request in requests)
    )
    versions = sorted(response.state_version for response in responses)

    assert versions == [route.state_version + 1, route.state_version + 2]
    assert service.state_version == versions[-1]
    persisted = json.loads((tmp_path / "router-state.json").read_text())
    assert persisted["state_version"] == versions[-1]
    assert persisted["feedback_ids"] == [
        "feedback-concurrent-1",
        "feedback-concurrent-2",
    ]


@pytest.mark.asyncio
async def test_generator_router_state_round_trips_all_runtime_state(
    tmp_path: Path,
) -> None:
    from generator_router_svc.main import GeneratorRouterServicer
    from mf_core.proto_gen.moleculeforge.v1.generator import router_pb2

    state_path = tmp_path / "router-state.json"
    service = _make_router_service(tmp_path)
    route_request = _valid_route_request(
        router_pb2,
        request_id="round-trip",
        n_select=2,
        n_samples=4,
        hciv=_valid_hciv([0.1] * 128),
        target_family="kinase",
        available_generator_names=list(GENERATOR_NAMES),
        task_complexity=router_pb2.TASK_COMPLEXITY_MEDIUM,
    )
    await service.Route(route_request, None)
    service.kd_layer.update_teacher_embedding_targets(0, [[1.0, 2.0, 3.0]])
    await service.SubmitFeedback(
        _typed_feedback(
            router_pb2,
            feedback_id="round-trip-feedback",
            request_id="round-trip",
            generator_name="hfm_3d",
            score=0.6,
        ),
        None,
    )
    await service.RunProxylessSearch(
        router_pb2.RouterProxylessSearchRequest(
            reward_batches_json=json.dumps({"dataset": [_full_proxyless_rewards()]}),
            generator_costs_json=json.dumps(_full_proxyless_costs()),
            cost_weight=0.1,
            learning_rate=0.5,
            temperature=1.0,
        ),
        None,
    )
    raw_before_restart = state_path.read_bytes()
    payload = json.loads(raw_before_restart)
    router_tensors_before = {
        key: value.detach().clone() for key, value in service.router.state_dict().items()
    }
    kd_tensors_before = {
        key: value.detach().clone() for key, value in service.kd_layer.state_dict().items()
    }

    restored = GeneratorRouterServicer(state_path=state_path, bootstrap=False)

    assert state_path.read_bytes() == raw_before_restart
    assert payload["schema_version"] == 3
    assert payload["generator_names"] == list(GENERATOR_NAMES)
    assert payload["router"]["dimensions"] == {
        "hciv_dim": 128,
        "hidden_dim": 32,
        "n_generators": 6,
        "task_dim": 8,
    }
    assert payload["kd"]["dimensions"] == {
        "mode": "production_real",
        "n_generators": 6,
    }
    assert payload["bootstrap_metadata"]["bootstrapped"] is True
    assert payload["context_state"]
    assert payload["request_context_map"] == {"round-trip": next(iter(payload["context_state"]))}
    assert payload["request_route_snapshots"]["round-trip"]["run_id"] == "run-feedback"
    assert payload["feedback_ids"] == ["round-trip-feedback"]
    assert len(payload["feedback_semantic_payloads"]) == 1
    assert restored.state_version == service.state_version
    assert restored.router.oracle_history == service.router.oracle_history
    assert restored.context_state == service.context_state
    assert restored.request_context_map == service.request_context_map
    assert restored.feedback_ids == service.feedback_ids
    assert restored.kd_layer._quality_scores == service.kd_layer._quality_scores
    assert torch.equal(
        restored.kd_layer._teacher_embedding_targets[0],
        service.kd_layer._teacher_embedding_targets[0],
    )
    for key, expected in router_tensors_before.items():
        assert torch.equal(restored.router.state_dict()[key], expected)
    for key, expected in kd_tensors_before.items():
        assert torch.equal(restored.kd_layer.state_dict()[key], expected)

    weights = await restored.GetWeights(route_request, None)
    assert weights.state_version == restored.state_version
    restored_version = restored.state_version
    restored_logits = restored.router.architecture_logits.detach().clone()
    await restored.RunProxylessSearch(
        router_pb2.RouterProxylessSearchRequest(
            reward_batches_json=json.dumps(
                {"dataset": [{**_full_proxyless_rewards(), "fragfm": 1.0}]}
            ),
            generator_costs_json=json.dumps(_full_proxyless_costs()),
            cost_weight=0.1,
            learning_rate=0.5,
            temperature=1.0,
        ),
        None,
    )
    assert restored.state_version == restored_version + 1
    assert not torch.equal(restored.router.architecture_logits, restored_logits)
    restarted_again = GeneratorRouterServicer(
        state_path=state_path,
        bootstrap=False,
    )
    assert torch.equal(
        restarted_again.router.architecture_logits,
        restored.router.architecture_logits,
    )


@pytest.mark.asyncio
async def test_generator_router_rejects_lossy_schema_v2_feedback_state(
    tmp_path: Path,
) -> None:
    from generator_router_svc.main import GeneratorRouterServicer
    from mf_core.proto_gen.moleculeforge.v1.generator import router_pb2

    state_path = tmp_path / "router-state.json"
    service = _make_router_service(tmp_path)
    await service.Route(
        _valid_route_request(
            router_pb2,
            request_id="schema-v2",
            n_select=len(GENERATOR_NAMES),
            n_samples=len(GENERATOR_NAMES),
        ),
        None,
    )
    feedback = _typed_feedback(
        router_pb2,
        feedback_id="schema-v2-feedback",
        request_id="schema-v2",
        generator_name="hfm_3d",
        score=0.6,
    )
    await service.SubmitFeedback(feedback, None)
    legacy_payload = json.loads(state_path.read_text())
    legacy_payload["schema_version"] = 2
    legacy_payload.pop("feedback_semantic_payloads")
    legacy_payload["request_route_snapshots"]["schema-v2"].pop("run_id")
    state_path.write_text(json.dumps(legacy_payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="schema_version 2.*feedback.*re-bootstrap"):
        GeneratorRouterServicer(state_path=state_path, bootstrap=False)


@pytest.mark.asyncio
async def test_generator_router_migrates_schema_v2_state_without_feedback(
    tmp_path: Path,
) -> None:
    from generator_router_svc.main import GeneratorRouterServicer
    from mf_core.proto_gen.moleculeforge.v1.generator import router_pb2

    state_path = tmp_path / "router-state.json"
    service = _make_router_service(tmp_path)
    await service.Route(
        _valid_route_request(
            router_pb2,
            request_id="schema-v2-empty",
        ),
        None,
    )
    legacy_payload = json.loads(state_path.read_text())
    legacy_payload["schema_version"] = 2
    legacy_payload.pop("feedback_semantic_payloads")
    legacy_payload["request_route_snapshots"]["schema-v2-empty"].pop("run_id")
    state_path.write_text(json.dumps(legacy_payload), encoding="utf-8")

    restored = GeneratorRouterServicer(state_path=state_path, bootstrap=False)
    assert restored.request_route_snapshots["schema-v2-empty"]["run_id"] is None
    await restored.Route(
        _valid_route_request(
            router_pb2,
            run_id="run-migrated",
            request_id="schema-v2-empty",
        ),
        None,
    )
    persisted = json.loads(state_path.read_text())

    assert persisted["schema_version"] == 3
    assert persisted["feedback_semantic_payloads"] == {}
    assert persisted["request_route_snapshots"]["schema-v2-empty"]["run_id"] == "run-migrated"


def test_generator_router_missing_state_requires_explicit_bootstrap(
    tmp_path: Path,
) -> None:
    from generator_router_svc.main import GeneratorRouterServicer

    with pytest.raises(RuntimeError, match="TAR_BOOTSTRAP=true"):
        GeneratorRouterServicer(
            state_path=tmp_path / "missing-state.json",
            bootstrap=False,
        )


@pytest.mark.parametrize(
    "corruption",
    [
        "schema-v1",
        "missing-route-snapshots",
        "missing-feedback-payloads",
        "missing-feedback-semantic-payloads",
        "missing-bound-snapshot",
        "allocation-total",
        "snapshot-context",
    ],
)
def test_generator_router_rejects_state_without_complete_replay_contract(
    tmp_path: Path,
    corruption: str,
) -> None:
    from generator_router_svc.main import GeneratorRouterServicer
    from mf_core.proto_gen.moleculeforge.v1.generator import router_pb2

    state_path = tmp_path / f"incomplete-{corruption}.json"
    service = GeneratorRouterServicer(state_path=state_path, bootstrap=True)
    if corruption in {
        "missing-bound-snapshot",
        "allocation-total",
        "snapshot-context",
    }:
        asyncio.run(
            service.Route(
                _valid_route_request(
                    router_pb2,
                    request_id="state-contract",
                    n_select=2,
                    n_samples=4,
                ),
                None,
            )
        )
    payload = json.loads(state_path.read_text())
    if corruption == "schema-v1":
        payload["schema_version"] = 1
    elif corruption == "missing-route-snapshots":
        payload.pop("request_route_snapshots")
    elif corruption == "missing-feedback-payloads":
        payload.pop("feedback_payloads")
    elif corruption == "missing-feedback-semantic-payloads":
        payload.pop("feedback_semantic_payloads")
    elif corruption == "missing-bound-snapshot":
        payload["request_route_snapshots"].clear()
    elif corruption == "allocation-total":
        snapshot = payload["request_route_snapshots"]["state-contract"]
        first_name = snapshot["selected_generators"][0]
        snapshot["allocations"][first_name] -= 1
    else:
        payload["request_route_snapshots"]["state-contract"]["context_key"] = "0" * 64
    state_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="re-bootstrap or migrate|snapshot"):
        GeneratorRouterServicer(state_path=state_path, bootstrap=False)


@pytest.mark.parametrize(
    "corruption",
    [
        "generator-order",
        "schema-bool",
        "bootstrap-time-bool",
        "tensor-shape",
        "tensor-shape-bool",
    ],
)
def test_generator_router_rejects_corrupt_state(
    tmp_path: Path,
    corruption: str,
) -> None:
    from generator_router_svc.main import GeneratorRouterServicer

    state_path = tmp_path / f"corrupt-{corruption}.json"
    GeneratorRouterServicer(state_path=state_path, bootstrap=True)
    payload = json.loads(state_path.read_text())
    if corruption == "generator-order":
        payload["generator_names"].reverse()
    elif corruption == "schema-bool":
        payload["schema_version"] = True
    elif corruption == "bootstrap-time-bool":
        payload["bootstrap_metadata"]["created_at_ns"] = True
    elif corruption == "tensor-shape-bool":
        payload["router"]["tensors"]["projection.weight"]["shape"][0] = True
    else:
        payload["router"]["tensors"]["projection.weight"]["shape"] = [1, 1]
    state_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="Router state"):
        GeneratorRouterServicer(state_path=state_path, bootstrap=False)


def test_generator_router_production_factory_requires_state_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from generator_router_svc.main import create_generator_router_servicer_from_env

    monkeypatch.delenv("TAR_STATE_PATH", raising=False)
    monkeypatch.delenv("TAR_BOOTSTRAP", raising=False)
    with pytest.raises(RuntimeError, match="TAR_STATE_PATH is required"):
        create_generator_router_servicer_from_env()

    state_path = tmp_path / "factory-state.json"
    monkeypatch.setenv("TAR_STATE_PATH", str(state_path))
    monkeypatch.setenv("TAR_BOOTSTRAP", "true")
    service = create_generator_router_servicer_from_env()

    assert service.state_path == state_path
    assert state_path.is_file()


@pytest.mark.asyncio
async def test_generator_router_persistence_failure_rolls_back_memory_and_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import generator_router_svc.main as router_main
    from mf_core.proto_gen.moleculeforge.v1.generator import router_pb2

    state_path = tmp_path / "router-state.json"
    service = _make_router_service(tmp_path)
    route = await service.Route(
        _valid_route_request(
            router_pb2,
            request_id="rollback",
            n_select=len(GENERATOR_NAMES),
            n_samples=len(GENERATOR_NAMES),
        ),
        None,
    )
    state_before = state_path.read_bytes()
    history_before = json.loads(json.dumps(service.router.oracle_history))

    def fail_replace(_source: object, _destination: object) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(router_main.os, "replace", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        await service.SubmitFeedback(
            _typed_feedback(
                router_pb2,
                feedback_id="rollback-feedback",
                request_id="rollback",
                generator_name="hfm_3d",
                score=0.9,
            ),
            None,
        )

    assert service.state_version == route.state_version
    assert service.router.oracle_history == history_before
    assert state_path.read_bytes() == state_before
    assert list(tmp_path.iterdir()) == [state_path]


@pytest.mark.asyncio
async def test_generator_router_real_grpc_returns_generated_messages(
    tmp_path: Path,
) -> None:
    from generator_router_svc.main import GeneratorRouterServicer
    from mf_core.proto_gen.moleculeforge.v1.generator import router_pb2, router_pb2_grpc

    service = GeneratorRouterServicer(
        state_path=tmp_path / "grpc-state.json",
        bootstrap=True,
    )
    server = grpc.aio.server()
    router_pb2_grpc.add_GeneratorRouterServiceServicer_to_server(service, server)
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    channel = grpc.aio.insecure_channel(f"127.0.0.1:{port}")
    stub = router_pb2_grpc.GeneratorRouterServiceStub(channel)
    request = _valid_route_request(
        router_pb2,
        request_id="grpc",
        n_select=2,
        n_samples=4,
    )
    try:
        route = await stub.Route(request)
        weights = await stub.GetWeights(request)
        feedback = await stub.SubmitFeedback(
            _typed_feedback(
                router_pb2,
                feedback_id="grpc-feedback",
                request_id="grpc",
                generator_name=route.selected_generators[0],
                score=0.5,
            )
        )

        assert isinstance(route, router_pb2.RouterResponse)
        assert isinstance(weights, router_pb2.RouterWeightsResponse)
        assert isinstance(feedback, router_pb2.RouterFeedbackResponse)

        with pytest.raises(grpc.aio.AioRpcError) as error:
            await stub.Route(
                _valid_route_request(
                    router_pb2,
                    request_id="grpc-invalid",
                    n_select=0,
                    n_samples=1,
                )
            )
        assert error.value.code() == grpc.StatusCode.INVALID_ARGUMENT
    finally:
        await channel.close()
        await server.stop(None)


def _set_hypseek_server_identity(
    monkeypatch: pytest.MonkeyPatch,
    *,
    version: str,
) -> None:
    monkeypatch.setenv("HYPSEEK_TEACHER_SOURCE", "hypseek")
    monkeypatch.setenv("HYPSEEK_TEACHER_VERSION", version)


def test_hypseek_teacher_response_requires_server_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from generator_router_svc.main import (
        HypSeekTeacherUnavailableError,
        hypseek_teacher_response,
    )

    monkeypatch.delenv("HYPSEEK_TEACHER_COMMAND", raising=False)
    monkeypatch.delenv("HYPSEEK_TEACHER_SOURCE", raising=False)
    monkeypatch.delenv("HYPSEEK_TEACHER_VERSION", raising=False)

    with pytest.raises(HypSeekTeacherUnavailableError, match="HYPSEEK_TEACHER_SOURCE"):
        hypseek_teacher_response(
            {
                "records": [{"candidate_id": "candidate-1", "outcome": "PASS"}],
                "teacher_policy": {
                    "teacher_source": "hypseek",
                    "teacher_version": "teacher-v1",
                    "allow_synthetic": True,
                },
            }
        )


def test_hypseek_teacher_response_rejects_caller_version_impersonation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from generator_router_svc.main import hypseek_teacher_response

    monkeypatch.delenv("HYPSEEK_TEACHER_COMMAND", raising=False)
    monkeypatch.setenv("HYPSEEK_TEACHER_SOURCE", "hypseek")
    monkeypatch.setenv("HYPSEEK_TEACHER_VERSION", "server-v1")

    with pytest.raises(ValueError, match="teacher_version"):
        hypseek_teacher_response(
            {
                "records": [{"candidate_id": "candidate-1", "outcome": "PASS"}],
                "teacher_policy": {
                    "teacher_source": "hypseek",
                    "teacher_version": "caller-v999",
                    "allow_synthetic": True,
                },
            }
        )


def test_hypseek_teacher_health_reports_configured_server_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fastapi.testclient import TestClient
    from generator_router_svc.main import hypseek_app

    _set_hypseek_server_identity(monkeypatch, version="server-v1")

    response = TestClient(hypseek_app).get("/healthz")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "hypseek_teacher",
        "teacher_source": "hypseek",
        "teacher_version": "server-v1",
    }


def test_hypseek_teacher_health_rejects_missing_server_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fastapi.testclient import TestClient
    from generator_router_svc.main import hypseek_app

    monkeypatch.delenv("HYPSEEK_TEACHER_SOURCE", raising=False)
    monkeypatch.delenv("HYPSEEK_TEACHER_VERSION", raising=False)

    response = TestClient(hypseek_app).get("/healthz")

    assert response.status_code == 503
    assert "HYPSEEK_TEACHER_SOURCE" in response.json()["detail"]


def test_hypseek_teacher_response_runs_command_with_original_records_and_policy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from generator_router_svc.main import hypseek_teacher_response

    command_script = tmp_path / "hypseek_teacher.py"
    command_script.write_text(
        "\n".join(
            [
                "import json",
                "import sys",
                "payload = json.load(sys.stdin)",
                "expected = {",
                '    "records": [{"candidate_id": "candidate-1", "outcome": "PASS"}],',
                '    "teacher_policy": {',
                '        "teacher_source": "hypseek",',
                '        "teacher_version": "teacher-v1",',
                '        "allow_synthetic": False,',
                "    },",
                "}",
                "if payload != expected:",
                "    raise SystemExit(17)",
                'print(json.dumps({"teacher_score": 0.625}))',
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv(
        "HYPSEEK_TEACHER_COMMAND",
        f"{sys.executable} {command_script}",
    )
    monkeypatch.setenv("HYPSEEK_TEACHER_TIMEOUT_SECONDS", "2")
    _set_hypseek_server_identity(monkeypatch, version="teacher-v1")
    response = hypseek_teacher_response(
        {
            "records": [{"candidate_id": "candidate-1", "outcome": "PASS"}],
            "teacher_policy": {
                "teacher_source": "hypseek",
                "teacher_version": "teacher-v1",
                "allow_synthetic": False,
            },
        }
    )

    assert response == {
        "teacher_score": 0.625,
        "teacher_source": "hypseek",
        "teacher_version": "teacher-v1",
        "synthetic": False,
    }


def test_hypseek_teacher_endpoint_reduces_real_command_distribution() -> None:
    from fastapi.testclient import TestClient
    from generator_router_svc.main import hypseek_app

    command = (
        f"{sys.executable} -c "
        '"import json,sys; json.load(sys.stdin); '
        "print(json.dumps({'distribution':[0.25,0.75]}))\""
    )
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setenv("HYPSEEK_TEACHER_COMMAND", command)
        monkeypatch.setenv("HYPSEEK_TEACHER_TIMEOUT_SECONDS", "2")
        _set_hypseek_server_identity(monkeypatch, version="teacher-v2")
        response = TestClient(hypseek_app).post(
            "/teacher",
            json={
                "records": [
                    {"candidate_id": "candidate-1", "outcome": "PASS"},
                    {"candidate_id": "candidate-2", "outcome": "FAIL"},
                ],
                "teacher_policy": {
                    "teacher_source": "hypseek",
                    "teacher_version": "teacher-v2",
                    "allow_synthetic": False,
                },
            },
        )

    assert response.status_code == 200
    assert response.json() == {
        "teacher_score": 0.5,
        "teacher_source": "hypseek",
        "teacher_version": "teacher-v2",
        "synthetic": False,
    }


def test_hypseek_teacher_endpoint_uses_explicit_synthetic_adapter_only_when_allowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fastapi.testclient import TestClient
    from generator_router_svc.main import hypseek_app

    monkeypatch.delenv("HYPSEEK_TEACHER_COMMAND", raising=False)
    monkeypatch.delenv("HYPSEEK_TEACHER_TIMEOUT_SECONDS", raising=False)
    _set_hypseek_server_identity(monkeypatch, version="builtin-v1")
    payload = {
        "records": [
            {"candidate_id": "candidate-1", "outcome": "PASS"},
            {"candidate_id": "candidate-2", "outcome": "FAIL"},
        ],
        "teacher_policy": {
            "teacher_source": "hypseek",
            "teacher_version": "builtin-v1",
            "allow_synthetic": True,
        },
    }

    response = TestClient(hypseek_app).post("/teacher", json=payload)

    assert response.status_code == 200
    assert response.json() == {
        "teacher_score": 0.5,
        "teacher_source": "hypseek",
        "teacher_version": "builtin-v1",
        "synthetic": True,
    }


def test_hypseek_teacher_endpoint_rejects_missing_command_for_real_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fastapi.testclient import TestClient
    from generator_router_svc.main import hypseek_app

    monkeypatch.delenv("HYPSEEK_TEACHER_COMMAND", raising=False)
    _set_hypseek_server_identity(monkeypatch, version="teacher-v1")
    response = TestClient(hypseek_app).post(
        "/teacher",
        json={
            "records": [{"candidate_id": "candidate-1", "outcome": "PASS"}],
            "teacher_policy": {
                "teacher_source": "hypseek",
                "teacher_version": "teacher-v1",
                "allow_synthetic": False,
            },
        },
    )

    assert response.status_code == 503
    assert "HYPSEEK_TEACHER_COMMAND" in response.json()["detail"]


def test_hypseek_teacher_endpoint_enforces_command_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fastapi.testclient import TestClient
    from generator_router_svc.main import hypseek_app

    command = (
        f"{sys.executable} -c "
        '"import json,sys,time; json.load(sys.stdin); time.sleep(1); '
        "print(json.dumps({'teacher_score':0.5}))\""
    )
    monkeypatch.setenv("HYPSEEK_TEACHER_COMMAND", command)
    monkeypatch.setenv("HYPSEEK_TEACHER_TIMEOUT_SECONDS", "0.01")
    _set_hypseek_server_identity(monkeypatch, version="teacher-v1")

    response = TestClient(hypseek_app).post(
        "/teacher",
        json={
            "records": [{"candidate_id": "candidate-1", "outcome": "PASS"}],
            "teacher_policy": {
                "teacher_source": "hypseek",
                "teacher_version": "teacher-v1",
                "allow_synthetic": False,
            },
        },
    )

    assert response.status_code == 502
    assert "timed out" in response.json()["detail"]


@pytest.mark.parametrize(
    "command_output",
    [
        "not-json",
        '{"teacher_score":NaN}',
        '{"teacher_score":1.1}',
        '{"teacher_distribution":[0.5]}',
    ],
)
def test_hypseek_teacher_endpoint_rejects_invalid_command_output(
    monkeypatch: pytest.MonkeyPatch,
    command_output: str,
) -> None:
    from fastapi.testclient import TestClient
    from generator_router_svc.main import hypseek_app

    command = (
        f'{sys.executable} -c "import json,sys; json.load(sys.stdin); print({command_output!r})"'
    )
    monkeypatch.setenv("HYPSEEK_TEACHER_COMMAND", command)
    monkeypatch.setenv("HYPSEEK_TEACHER_TIMEOUT_SECONDS", "2")
    _set_hypseek_server_identity(monkeypatch, version="teacher-v1")
    response = TestClient(hypseek_app).post(
        "/teacher",
        json={
            "records": [
                {"candidate_id": "candidate-1", "outcome": "PASS"},
                {"candidate_id": "candidate-2", "outcome": "FAIL"},
            ],
            "teacher_policy": {
                "teacher_source": "hypseek",
                "teacher_version": "teacher-v1",
                "allow_synthetic": False,
            },
        },
    )

    assert response.status_code == 502


def test_review_contract_zero_data_richness_has_zero_feature() -> None:
    assert TaskProfile(data_richness=0.0).to_feature_vector()[1] == 0.0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"project_id": ""}, "project_id is required"),
        ({"cig": b""}, "cig is required"),
        (
            {"cig": _valid_cig_bytes("different-project")},
            "CIG project_id must match",
        ),
        (
            {"cig": _cig_without_objectives_bytes()},
            "CIG objectives must be non-empty",
        ),
        ({"hciv": []}, "hciv must contain exactly 129"),
        ({"hciv": [0.0] * 128}, "hciv must contain exactly 129"),
        ({"hciv": [2.0, *([0.0] * 128)]}, "valid Lorentz"),
        ({"hciv": [float("nan"), *([0.0] * 128)]}, "finite"),
    ],
)
async def test_review_contract_route_requires_canonical_context(
    tmp_path: Path,
    overrides: dict[str, object],
    message: str,
) -> None:
    from mf_core.proto_gen.moleculeforge.v1.generator import router_pb2

    state_path = tmp_path / "router-state.json"
    service = _make_router_service(tmp_path)
    state_before = state_path.read_bytes()

    with pytest.raises(ValueError, match=message):
        await service.Route(_valid_route_request(router_pb2, **overrides), None)

    assert state_path.read_bytes() == state_before


@pytest.mark.asyncio
async def test_review_contract_route_retry_returns_persisted_snapshot(
    tmp_path: Path,
) -> None:
    from mf_core.proto_gen.moleculeforge.v1.generator import router_pb2

    service = _make_router_service(tmp_path)
    request = _valid_route_request(
        router_pb2,
        request_id="snapshot",
        n_select=1,
        n_samples=3,
    )
    service.router.forward = lambda *_args, **_kwargs: {
        name: (6.0 if name == "hfm_3d" else 1.0) for name in GENERATOR_NAMES
    }
    first = await service.Route(request, None)
    service.router.forward = lambda *_args, **_kwargs: {
        name: (6.0 if name == "fragfm" else 1.0) for name in GENERATOR_NAMES
    }

    retry = await service.Route(request, None)
    persisted = json.loads((tmp_path / "router-state.json").read_text())

    assert retry == first
    assert retry.state_version == first.state_version
    snapshot = persisted["request_route_snapshots"]["snapshot"]
    assert snapshot["allocations"] == {"hfm_3d": 3}
    assert snapshot["eligible_generator_names"] == list(GENERATOR_NAMES)
    assert snapshot["eligible_weights"] == pytest.approx(
        {
            "hfm_3d": 6.0 / 11.0,
            "fragfm": 1.0 / 11.0,
            "crem_3d": 1.0 / 11.0,
            "mmpt_rag": 1.0 / 11.0,
            "iclm": 1.0 / 11.0,
            "uas": 1.0 / 11.0,
        }
    )
    assert snapshot["expected_rewards"] == pytest.approx({"hfm_3d": 6.0 / 11.0})
    assert snapshot["normalized_weights"] == {"hfm_3d": 1.0}
    assert snapshot["selected_generators"] == ["hfm_3d"]


@pytest.mark.asyncio
async def test_review_contract_feedback_rejects_generator_not_selected_for_request(
    tmp_path: Path,
) -> None:
    from mf_core.proto_gen.moleculeforge.v1.generator import router_pb2

    service = _make_router_service(tmp_path)
    service.router.forward = lambda *_args, **_kwargs: {
        name: (1.0 if name == "hfm_3d" else 0.0) for name in GENERATOR_NAMES
    }
    await service.Route(
        _valid_route_request(router_pb2, request_id="selected-only"),
        None,
    )

    with pytest.raises(ValueError, match="was not selected"):
        await service.SubmitFeedback(
            _typed_feedback(
                router_pb2,
                feedback_id="unselected-feedback",
                request_id="selected-only",
                generator_name="fragfm",
                score=0.8,
            ),
            None,
        )


@pytest.mark.asyncio
async def test_review_contract_duplicate_feedback_still_checks_selected_generator(
    tmp_path: Path,
) -> None:
    from mf_core.proto_gen.moleculeforge.v1.generator import router_pb2

    service = _make_router_service(tmp_path)
    service.router.forward = lambda *_args, **_kwargs: {
        name: (1.0 if name == "hfm_3d" else 0.0) for name in GENERATOR_NAMES
    }
    await service.Route(
        _valid_route_request(router_pb2, request_id="duplicate-selected-only"),
        None,
    )
    accepted = _typed_feedback(
        router_pb2,
        feedback_id="duplicate-selection-feedback",
        request_id="duplicate-selected-only",
        generator_name="hfm_3d",
        score=0.8,
    )
    await service.SubmitFeedback(accepted, None)

    with pytest.raises(ValueError, match="was not selected"):
        await service.SubmitFeedback(
            _typed_feedback(
                router_pb2,
                feedback_id="duplicate-selection-feedback",
                request_id="duplicate-selected-only",
                generator_name="fragfm",
                score=0.8,
            ),
            None,
        )


@pytest.mark.asyncio
async def test_review_contract_get_weights_binds_request_context(
    tmp_path: Path,
) -> None:
    from mf_core.proto_gen.moleculeforge.v1.generator import router_pb2

    service = _make_router_service(tmp_path)
    await service.GetWeights(
        _valid_route_request(router_pb2, request_id="weights-bind"),
        None,
    )

    with pytest.raises(
        ValueError,
        match="request_id is already bound to a different routing context",
    ):
        await service.Route(
            _valid_route_request(
                router_pb2,
                project_id="other-project",
                cig=_valid_cig_bytes("other-project"),
                request_id="weights-bind",
            ),
            None,
        )


@pytest.mark.asyncio
async def test_review_contract_directory_fsync_failure_keeps_committed_state_aligned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    import generator_router_svc.main as router_main
    from mf_core.proto_gen.moleculeforge.v1.generator import router_pb2

    service = _make_router_service(tmp_path)
    route = await service.Route(
        _valid_route_request(
            router_pb2,
            request_id="post-replace-fsync",
            n_select=len(GENERATOR_NAMES),
            n_samples=len(GENERATOR_NAMES),
        ),
        None,
    )
    real_fsync = router_main.os.fsync
    calls = 0

    def fail_directory_fsync(descriptor: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("directory fsync failed")
        real_fsync(descriptor)

    monkeypatch.setattr(router_main.os, "fsync", fail_directory_fsync)
    with caplog.at_level("WARNING"):
        response = await service.SubmitFeedback(
            _typed_feedback(
                router_pb2,
                feedback_id="post-replace-feedback",
                request_id="post-replace-fsync",
                generator_name=route.selected_generators[0],
                score=0.9,
            ),
            None,
        )
    persisted = json.loads((tmp_path / "router-state.json").read_text())

    assert response.acknowledged is True
    assert response.state_version == service.state_version
    assert persisted["state_version"] == service.state_version
    assert persisted["feedback_ids"] == ["post-replace-feedback"]
    assert "directory fsync failed after atomic replace" in caplog.text


def test_review_contract_router_runtime_does_not_claim_teacher_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from generator_router_svc.main import runtime_status

    monkeypatch.setenv("HYPSEEK_TEACHER_COMMAND", "teacher --json")

    assert all(item["name"] != "hypseek_teacher_command" for item in runtime_status())


def _full_proxyless_costs() -> dict[str, float]:
    return {name: float(index + 1) for index, name in enumerate(GENERATOR_NAMES)}


def _full_proxyless_rewards() -> dict[str, float]:
    return {name: float(index) / 10.0 for index, name in enumerate(GENERATOR_NAMES)}


class _GrpcAbortError(Exception):
    pass


class _RecordingGrpcContext:
    def __init__(self) -> None:
        self.code: grpc.StatusCode | None = None
        self.details = ""

    async def abort(self, code: grpc.StatusCode, details: str) -> Never:
        self.code = code
        self.details = details
        raise _GrpcAbortError(details)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("reward_mutation", "cost_mutation", "temperature", "message"),
    [
        (("remove", "uas"), None, 1.0, "missing generator reward"),
        (None, ("remove", "uas"), 1.0, "missing generator cost"),
        (("add", "unknown"), None, 1.0, "unknown generator"),
        (None, ("bool", "uas"), 1.0, "finite"),
        (None, None, 0.0, "temperature"),
    ],
)
async def test_review_contract_proxyless_validation_aborts_invalid_argument(
    tmp_path: Path,
    reward_mutation: tuple[str, str] | None,
    cost_mutation: tuple[str, str] | None,
    temperature: float,
    message: str,
) -> None:
    from mf_core.proto_gen.moleculeforge.v1.generator import router_pb2

    rewards = _full_proxyless_rewards()
    costs = _full_proxyless_costs()
    if reward_mutation == ("remove", "uas"):
        rewards.pop("uas")
    elif reward_mutation == ("add", "unknown"):
        rewards["unknown"] = 1.0
    if cost_mutation == ("remove", "uas"):
        costs.pop("uas")
    elif cost_mutation == ("bool", "uas"):
        costs["uas"] = True
    context = _RecordingGrpcContext()
    request = router_pb2.RouterProxylessSearchRequest(
        reward_batches_json=json.dumps({"dataset": [rewards]}),
        generator_costs_json=json.dumps(costs),
        cost_weight=0.1,
        learning_rate=0.1,
        temperature=temperature,
    )

    with pytest.raises(_GrpcAbortError, match=message):
        await _make_router_service(tmp_path).RunProxylessSearch(request, context)

    assert context.code == grpc.StatusCode.INVALID_ARGUMENT


@pytest.mark.parametrize(
    "corruption",
    [
        "router-dimension-bool",
        "router-tensor-value-bool",
        "history-average-bool",
        "quality-score-bool",
    ],
)
def test_review_contract_state_rejects_bool_numeric_fields(
    tmp_path: Path,
    corruption: str,
) -> None:
    from generator_router_svc.main import GeneratorRouterServicer

    state_path = tmp_path / f"numeric-bool-{corruption}.json"
    GeneratorRouterServicer(state_path=state_path, bootstrap=True)
    payload = json.loads(state_path.read_text())
    if corruption == "router-dimension-bool":
        payload["router"]["dimensions"]["hciv_dim"] = True
    elif corruption == "router-tensor-value-bool":
        payload["router"]["tensors"]["architecture_logits"]["values"][0] = True
    elif corruption == "history-average-bool":
        payload["router"]["oracle_history"]["hfm_3d"]["avg_hvi"] = True
    else:
        payload["kd"]["quality_scores"][0] = True
    state_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="Router state"):
        GeneratorRouterServicer(state_path=state_path, bootstrap=False)


def test_hypseek_teacher_and_router_state_deployment_wiring() -> None:
    import yaml

    expected_url = "http://hypseek-teacher-svc:8012/teacher"
    state_path = "/var/lib/moleculeforge/router/state.json"
    compose = yaml.safe_load(
        (ROOT / "infra/docker/docker-compose.dev.yml").read_text(encoding="utf-8")
    )
    k8s = list(
        yaml.safe_load_all(
            (ROOT / "infra/kubernetes/deployments/moleculeforge-services.yaml").read_text(
                encoding="utf-8"
            )
        )
    )
    helm_values = yaml.safe_load(
        (ROOT / "infra/helm/moleculeforge/values.yaml").read_text(encoding="utf-8")
    )

    compose_services = compose["services"]
    compose_router = compose_services["generator-router-svc"]
    compose_teacher = compose_services["hypseek-teacher-svc"]
    compose_generator_agent = compose_services["generator-coord-agent"]
    compose_orchestrator = compose_services["orchestrator-svc"]
    assert compose_generator_agent["environment"]["HYPSEEK_TEACHER_URL"] == expected_url
    assert compose_generator_agent["environment"]["HYPSEEK_TEACHER_TIMEOUT_SECONDS"] == (
        "${HYPSEEK_TEACHER_CLIENT_TIMEOUT_SECONDS:-60}"
    )
    assert "HYPSEEK_TEACHER_URL" not in compose_orchestrator["environment"]
    assert "HYPSEEK_TEACHER_TIMEOUT_SECONDS" not in compose_orchestrator["environment"]
    assert "HYPSEEK_TEACHER_URL" not in compose_router["environment"]
    assert "HYPSEEK_TEACHER_COMMAND" not in compose_router["environment"]
    assert compose_router["environment"]["TAR_STATE_PATH"] == state_path
    assert compose_router["environment"]["TAR_BOOTSTRAP"] == "true"
    assert "router_state_data:/var/lib/moleculeforge/router" in compose_router["volumes"]
    assert "router_state_data" in compose["volumes"]
    assert {
        "HYPSEEK_TEACHER_SOURCE",
        "HYPSEEK_TEACHER_VERSION",
        "HYPSEEK_TEACHER_COMMAND",
        "HYPSEEK_TEACHER_TIMEOUT_SECONDS",
    } <= set(compose_teacher["environment"])
    assert compose_teacher["environment"]["HYPSEEK_TEACHER_TIMEOUT_SECONDS"] == (
        "${HYPSEEK_TEACHER_SERVER_TIMEOUT_SECONDS:-60}"
    )

    deployments = {
        document["metadata"]["name"]: document
        for document in k8s
        if document and document.get("kind") == "Deployment"
    }

    def deployment_env(name: str) -> dict[str, dict]:
        container = deployments[name]["spec"]["template"]["spec"]["containers"][0]
        return {item["name"]: item for item in container.get("env", [])}

    router_deployment = deployments["generator-router-svc"]
    router_container = router_deployment["spec"]["template"]["spec"]["containers"][0]
    router_env = deployment_env("generator-router-svc")
    teacher_env = deployment_env("hypseek-teacher-svc")
    generator_agent_env = deployment_env("generator-coord-agent")
    orchestrator_env = deployment_env("orchestrator-svc")
    assert generator_agent_env["HYPSEEK_TEACHER_URL"]["value"] == expected_url
    assert "HYPSEEK_TEACHER_TIMEOUT_SECONDS" in generator_agent_env
    assert "HYPSEEK_TEACHER_URL" not in orchestrator_env
    assert "HYPSEEK_TEACHER_TIMEOUT_SECONDS" not in orchestrator_env
    assert "HYPSEEK_TEACHER_URL" not in router_env
    assert "HYPSEEK_TEACHER_COMMAND" not in router_env
    assert router_env["TAR_STATE_PATH"]["value"] == state_path
    assert router_env["TAR_BOOTSTRAP"]["value"] == "true"
    assert router_deployment["spec"]["strategy"] == {"type": "Recreate"}
    assert router_container["volumeMounts"] == [
        {"name": "router-state", "mountPath": "/var/lib/moleculeforge/router"}
    ]
    assert router_deployment["spec"]["template"]["spec"]["volumes"] == [
        {
            "name": "router-state",
            "persistentVolumeClaim": {"claimName": "generator-router-state"},
        }
    ]
    assert {
        "HYPSEEK_TEACHER_SOURCE",
        "HYPSEEK_TEACHER_VERSION",
        "HYPSEEK_TEACHER_COMMAND",
        "HYPSEEK_TEACHER_TIMEOUT_SECONDS",
    } <= set(teacher_env)
    claims = {
        document["metadata"]["name"]: document
        for document in k8s
        if document and document.get("kind") == "PersistentVolumeClaim"
    }
    assert claims["generator-router-state"]["metadata"]["namespace"] == "mf-agents"

    helm_services = helm_values["services"]
    helm_router = helm_services["generator-router-svc"]
    helm_teacher = helm_services["hypseek-teacher-svc"]
    helm_generator_agent = helm_services["generator-coord-agent"]
    helm_orchestrator = helm_services["orchestrator-svc"]
    assert helm_generator_agent["env"]["HYPSEEK_TEACHER_URL"] == expected_url
    assert "HYPSEEK_TEACHER_TIMEOUT_SECONDS" in helm_generator_agent["env"]
    assert "HYPSEEK_TEACHER_URL" not in helm_orchestrator.get("env", {})
    assert "HYPSEEK_TEACHER_TIMEOUT_SECONDS" not in helm_orchestrator.get("env", {})
    assert "HYPSEEK_TEACHER_URL" not in helm_router["env"]
    assert "HYPSEEK_TEACHER_COMMAND" not in helm_router.get("envValueFrom", {})
    assert helm_router["env"]["TAR_STATE_PATH"] == state_path
    assert helm_router["env"]["TAR_BOOTSTRAP"] == "true"
    assert helm_router["strategy"] == {"type": "Recreate"}
    assert helm_router["volumeMounts"] == [
        {"name": "router-state", "mountPath": "/var/lib/moleculeforge/router"}
    ]
    assert helm_router["volumes"] == [
        {
            "name": "router-state",
            "persistentVolumeClaim": {"claimName": "generator-router-state"},
        }
    ]
    assert {
        "HYPSEEK_TEACHER_SOURCE",
        "HYPSEEK_TEACHER_VERSION",
        "HYPSEEK_TEACHER_COMMAND",
        "HYPSEEK_TEACHER_TIMEOUT_SECONDS",
    } <= set(helm_teacher["envValueFrom"])
    assert helm_values["persistentVolumeClaims"]["generator-router-state"]["namespace"] == (
        "mf-agents"
    )


def test_full_workflow_agent_runtime_deployments_are_executable_and_dependency_aware() -> None:
    import yaml

    compose = yaml.safe_load(
        (ROOT / "infra/docker/docker-compose.dev.yml").read_text(encoding="utf-8")
    )
    k8s_documents = list(
        yaml.safe_load_all(
            (ROOT / "infra/kubernetes/deployments/moleculeforge-services.yaml").read_text(
                encoding="utf-8"
            )
        )
    )
    helm_values = yaml.safe_load(
        (ROOT / "infra/helm/moleculeforge/values.yaml").read_text(encoding="utf-8")
    )
    workers = {
        "nl2obj-agent": (
            "nl2obj",
            {"CIG_COMPILER_TARGET"},
            {"redis", "neo4j", "cig-compiler-svc"},
        ),
        "generator-coord-agent": (
            "generator_coord",
            {
                "GENERATOR_ROUTER_TARGET",
                "HFM_3D_GENERATOR_TARGET",
                "FRAGFM_GENERATOR_TARGET",
                "CREM_3D_GENERATOR_TARGET",
                "MMPT_RAG_GENERATOR_TARGET",
                "ICLM_GENERATOR_TARGET",
                "HYPSEEK_TEACHER_URL",
                "HYPSEEK_TEACHER_TIMEOUT_SECONDS",
            },
            {
                "redis",
                "neo4j",
                "generator-router-svc",
                "hfm-generator-svc",
                "fragfm-generator-svc",
                "crem-generator-svc",
                "mmpt-generator-svc",
                "iclm-svc",
                "hypseek-teacher-svc",
            },
        ),
        "validation-agent": (
            "validation",
            {
                "L1_ADMET_ORACLE_TARGET",
                "L1_BOLTZ2_ORACLE_TARGET",
                "L2_DOCK_ORACLE_TARGET",
                "L3_FEP_ORACLE_TARGET",
            },
            {
                "redis",
                "neo4j",
                "admet-svc",
                "boltz2-svc",
                "dock-svc",
                "fep-svc",
            },
        ),
        "retrosyn-agent": (
            "retrosyn",
            {
                "RETROSYN_PLANNER_COMMAND",
                "RETROSYN_PLANNER_COMMANDS_JSON",
                "AIZYNTH_CONFIG_PATH",
                "HUMU_ENCODER_TARGET",
            },
            {"redis", "neo4j", "humu-encoder-svc"},
        ),
        "supply-agent": (
            "supply",
            {"SUPPLY_ORACLE_TARGET"},
            {"redis", "neo4j", "supply-oracle-svc"},
        ),
        "srb-agent": (
            "srb",
            {"SILA2_PLAN_COMMAND", "SILA2_PLAN_TIMEOUT_SECONDS"},
            {"redis", "neo4j"},
        ),
        "critic-agent": (
            "critic",
            set(),
            {"redis", "neo4j"},
        ),
    }
    shared_env = {
        "REDIS_URL",
        "AGENT_MESSAGE_HMAC_SECRET",
        "NEO4J_URI",
        "NEO4J_USER",
        "NEO4J_PASSWORD",
    }
    probe = {
        "exec": {
            "command": [
                "python",
                "-c",
                "import os; os.kill(1, 0)",
            ]
        },
        "initialDelaySeconds": 10,
        "periodSeconds": 10,
        "timeoutSeconds": 5,
        "failureThreshold": 3,
    }

    compose_services = compose["services"]
    k8s_deployments = {
        item["metadata"]["name"]: item
        for item in k8s_documents
        if item and item.get("kind") == "Deployment"
    }
    k8s_services = {
        item["metadata"]["name"] for item in k8s_documents if item and item.get("kind") == "Service"
    }
    k8s_secrets = {
        item["metadata"]["name"]: item
        for item in k8s_documents
        if item and item.get("kind") == "Secret"
    }
    helm_services = helm_values["services"]
    assert set(k8s_secrets["agent-runtime-secrets"]["stringData"]) == shared_env

    for worker_name, (agent_name, required_env, dependencies) in workers.items():
        expected_command = ["python", "-m", "mf_agents.runtime", "--agent", agent_name]

        compose_worker = compose_services[worker_name]
        assert compose_worker["command"] == expected_command
        assert "ports" not in compose_worker
        assert shared_env | required_env <= set(compose_worker["environment"])
        assert dependencies <= set(compose_worker["depends_on"])
        assert compose_worker["healthcheck"]["test"] == [
            "CMD",
            "python",
            "-c",
            "import os; os.kill(1, 0)",
        ]
        assert compose_worker["healthcheck"]["start_period"] == "10s"
        assert compose_worker["restart"] == "on-failure"

        deployment = k8s_deployments[worker_name]
        container = deployment["spec"]["template"]["spec"]["containers"][0]
        container_env = {item["name"]: item for item in container.get("env", [])}
        assert container["image"] == "moleculeforge/agent-runtime:latest"
        assert container["command"] == expected_command
        assert "ports" not in container
        assert required_env <= set(container_env)
        assert container["envFrom"] == [{"secretRef": {"name": "agent-runtime-secrets"}}]
        assert container["readinessProbe"] == probe
        assert container["livenessProbe"] == probe

        helm_worker = helm_services[worker_name]
        helm_env = set(helm_worker.get("env", {})) | set(helm_worker.get("envValueFrom", {}))
        assert helm_worker["image"]["repository"] == "agent-runtime"
        assert helm_worker["command"] == expected_command
        assert helm_worker["ports"] == []
        assert shared_env | required_env <= helm_env
        assert helm_worker["readinessProbe"] == probe
        assert helm_worker["livenessProbe"] == probe

    assert set(workers).isdisjoint(k8s_services)
    assert {
        "generator_coord",
        "validation",
        "retrosyn",
        "supply",
        "srb",
        "critic",
        "nl2obj",
    } == {definition[0] for definition in workers.values()}


def test_image_build_script_produces_the_deployment_image_dependency_chain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    docker_log = _install_recording_docker(tmp_path, monkeypatch)

    completed = subprocess.run(
        ["./infra/scripts/build_all_images.sh"],
        cwd=ROOT,
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    build_calls = docker_log.read_text(encoding="utf-8").splitlines()
    assert build_calls == [
        "build -f infra/docker/base/Dockerfile.base -t moleculeforge/base:latest .",
        "build -f infra/docker/base/Dockerfile.chem -t moleculeforge/chem:latest .",
        "build -f infra/docker/base/Dockerfile.generator -t moleculeforge/generator:latest .",
        "build -f infra/docker/base/Dockerfile.oracle -t moleculeforge/oracle:latest .",
        "build -f infra/docker/base/Dockerfile.agent -t moleculeforge/agent-runtime:latest .",
    ]

    built_images: set[str] = set()
    for call in build_calls:
        parts = call.split()
        dockerfile = ROOT / parts[parts.index("-f") + 1]
        image = parts[parts.index("-t") + 1]
        assert dockerfile.is_file()
        parent = next(
            line.split(maxsplit=1)[1]
            for line in dockerfile.read_text(encoding="utf-8").splitlines()
            if line.startswith("FROM ")
        )
        if parent.startswith("moleculeforge/"):
            assert parent in built_images
        built_images.add(image)

    assert "moleculeforge/agent-runtime:latest" in built_images


def test_image_build_ci_publishes_the_same_complete_image_chain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import yaml

    repository_root = ROOT.parent
    workflow_path = repository_root / ".github/workflows/ci-build-images.yml"
    workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
    build_steps = [
        step
        for step in workflow["jobs"]["build"]["steps"]
        if "build_all_images.sh" in step.get("run", "")
    ]
    assert len(build_steps) == 1
    build_step = build_steps[0]
    assert build_step["env"] == {
        "PUBLISH_REGISTRY": "ghcr.io/${{ github.repository }}",
        "PUBLISH_TAG": "${{ github.sha }}",
    }
    assert build_step["working-directory"] == "moleculeforge"

    docker_log = _install_recording_docker(tmp_path, monkeypatch)
    monkeypatch.setenv("PUBLISH_REGISTRY", "ghcr.io/Example/MoleculeForge")
    monkeypatch.setenv("PUBLISH_TAG", "commit-sha")

    completed = subprocess.run(
        build_step["run"].split(),
        cwd=ROOT,
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    calls = docker_log.read_text(encoding="utf-8").splitlines()
    published_images = [
        f"ghcr.io/example/moleculeforge/{name}:commit-sha"
        for name in ("base", "chem", "generator", "oracle", "agent-runtime")
    ]
    for image in published_images:
        local_image = image.replace(
            "ghcr.io/example/moleculeforge/",
            "moleculeforge/",
        ).replace(":commit-sha", ":latest")
        assert f"tag {local_image} {image}" in calls
        assert f"push {image}" in calls


def test_generator_router_deployment_wires_proxyless_search_env() -> None:
    compose = (ROOT / "infra/docker/docker-compose.dev.yml").read_text(encoding="utf-8")
    k8s = (ROOT / "infra/kubernetes/deployments/moleculeforge-services.yaml").read_text(
        encoding="utf-8"
    )
    helm_values = (ROOT / "infra/helm/moleculeforge/values.yaml").read_text(encoding="utf-8")

    assert "TAR_PROXYLESS_SEARCH_COMMAND: ${TAR_PROXYLESS_SEARCH_COMMAND:-}" in compose
    assert (
        "TAR_PROXYLESS_SEARCH_TIMEOUT_SECONDS: ${TAR_PROXYLESS_SEARCH_TIMEOUT_SECONDS:-300}"
    ) in compose

    assert "name: tar-proxyless-search-config" in k8s
    assert "name: TAR_PROXYLESS_SEARCH_COMMAND" in k8s
    assert "key: proxyless-search-command" in k8s
    assert "name: TAR_PROXYLESS_SEARCH_TIMEOUT_SECONDS" in k8s
    assert "key: proxyless-search-timeout-seconds" in k8s

    assert 'TAR_PROXYLESS_SEARCH_TIMEOUT_SECONDS: "300"' in helm_values
    assert "TAR_PROXYLESS_SEARCH_COMMAND:" in helm_values
    assert "key: proxyless-search-command" in helm_values
