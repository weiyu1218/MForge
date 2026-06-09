"""Unit tests for TaskAwareRouter (Layer 3 — TAR)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from mf_core.routing.task_router import (
    GENERATOR_NAMES,
    ProxylessSearchScheduler,
    TaskAwareRouter,
    TaskProfile,
)

ROOT = Path(__file__).resolve().parents[2]


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
        assert result["architecture_probabilities"]["fragfm"] > (
            result["architecture_probabilities"]["hfm_3d"]
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
async def test_generator_router_service_returns_real_generator_names() -> None:
    from generator_router_svc.main import GeneratorRouterServicer

    request = type("RouteRequest", (), {"n_select": 4})()
    response = await GeneratorRouterServicer().Route(request, None)

    assert len(response.selected_generators) == 4
    assert set(response.selected_generators).issubset(GENERATOR_NAMES)
    assert not any(name.startswith("gen-") for name in response.selected_generators)
    assert len(response.selection_weights) == 4


@pytest.mark.asyncio
async def test_generator_router_service_uses_request_hciv_and_profile() -> None:
    from generator_router_svc.main import GeneratorRouterServicer
    from mf_core.proto_gen.moleculeforge.v1.generator import router_pb2

    service = GeneratorRouterServicer()
    seen: dict[str, object] = {}

    class RecordingRouter:
        hciv_dim = 3
        oracle_history = {
            name: {"avg_hvi": 0.0, "n_calls": 0.0}
            for name in GENERATOR_NAMES
        }

        def forward(self, hciv: torch.Tensor, profile: TaskProfile) -> dict[str, float]:
            seen["hciv"] = hciv
            seen["profile"] = profile
            return {name: 1.0 / len(GENERATOR_NAMES) for name in GENERATOR_NAMES}

    service.router = RecordingRouter()

    request = router_pb2.RouterRequest(
        n_select=2,
        hciv=[1.0, 2.0, 3.0],
        target_family="kinase",
        stage="lead_opt",
        data_richness=25.0,
        novelty_demand=0.8,
        multi_target=True,
        sa_constraint=3.0,
        n_samples=32,
    )

    await service.Route(request, None)

    assert torch.equal(seen["hciv"], torch.tensor([1.0, 2.0, 3.0]))
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
async def test_generator_router_service_uses_request_generator_performance() -> None:
    from generator_router_svc.main import GeneratorRouterServicer
    from mf_core.proto_gen.moleculeforge.v1.generator import router_pb2

    service = GeneratorRouterServicer()
    request = router_pb2.RouterRequest(
        n_select=1,
        generator_performance=[0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    )

    response = await service.Route(request, None)

    assert response.selected_generators == ["fragfm"]


@pytest.mark.asyncio
async def test_generator_router_feedback_updates_kd_teacher_scores() -> None:
    from generator_router_svc.main import GeneratorRouterServicer

    service = GeneratorRouterServicer()
    request = type(
        "FeedbackRequest",
        (),
        {
            "generator_name": "hfm_3d",
            "reward": 0.1,
            "oracle_feedback": [
                {"oracle_name": "boltz2", "normalized_score": 0.8},
                {"oracle_name": "gnina", "normalized_score": 0.6},
            ],
        },
    )()

    response = await service.SubmitFeedback(request, None)

    assert response.acknowledged is True
    assert response.generator_name == "hfm_3d"
    assert response.teacher_score == pytest.approx(0.7)
    assert service.kd_layer.running_counts[0].item() == 2.0
    assert service.router.oracle_history["hfm_3d"]["avg_hvi"] == pytest.approx(0.7)


@pytest.mark.asyncio
async def test_generator_router_feedback_uses_hypseek_teacher_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from generator_router_svc.main import GeneratorRouterServicer

    command = (
        f"{sys.executable} -c "
        "\"import json,sys; "
        "payload=json.load(sys.stdin); "
        "assert payload['generator_name'] == 'fragfm'; "
        "print(json.dumps({'oracle_name':'hypseek',"
        "'teacher_distribution':[0.25, 0.75]}))\""
    )
    monkeypatch.setenv("HYPSEEK_TEACHER_COMMAND", command)
    service = GeneratorRouterServicer()
    request = type(
        "FeedbackRequest",
        (),
        {
            "generator_name": "fragfm",
            "reward": 0.1,
            "oracle_feedback": [],
        },
    )()

    response = await service.SubmitFeedback(request, None)

    assert response.teacher_score == pytest.approx(0.5)
    assert service.kd_layer.running_counts[1].item() == 2.0
    assert service.router.oracle_history["fragfm"]["avg_hvi"] == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_generator_router_feedback_uses_hypseek_teacher_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import generator_router_svc.main as router_main

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
    service = router_main.GeneratorRouterServicer()
    request = type(
        "FeedbackRequest",
        (),
        {
            "generator_name": "fragfm",
            "reward": 0.1,
            "oracle_feedback": [],
        },
    )()

    response = await service.SubmitFeedback(request, None)

    assert calls == [
        {
            "url": "https://hypseek.example/teacher",
            "payload": {
                "generator_name": "fragfm",
                "reward": 0.1,
                "oracle_feedback": [],
            },
            "timeout_seconds": 60.0,
        }
    ]
    assert response.teacher_score == pytest.approx(0.5)
    assert service.kd_layer.running_counts[1].item() == 2.0
    assert service.router.oracle_history["fragfm"]["avg_hvi"] == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_generator_router_service_runs_proxyless_search_request() -> None:
    from generator_router_svc.main import GeneratorRouterServicer
    from mf_core.proto_gen.moleculeforge.v1.generator import router_pb2

    service = GeneratorRouterServicer()
    request = router_pb2.RouterProxylessSearchRequest(
        reward_batches_json=json.dumps(
            {
                "kras": [
                    {"hfm_3d": 0.2, "fragfm": 0.8},
                    {"hfm_3d": 0.1, "fragfm": 0.9},
                ]
            }
        ),
        generator_costs_json=json.dumps({"hfm_3d": 5.0, "fragfm": 1.0}),
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
    assert result["architecture_probabilities"]["fragfm"] > (
        result["architecture_probabilities"]["hfm_3d"]
    )
    assert service.router.oracle_history["fragfm"]["n_calls"] == 2.0


@pytest.mark.asyncio
async def test_generator_router_service_uses_proxyless_search_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from generator_router_svc.main import GeneratorRouterServicer
    from mf_core.proto_gen.moleculeforge.v1.generator import router_pb2

    command = (
        f"{sys.executable} -c "
        "\"import json,sys; "
        "payload=json.load(sys.stdin); "
        "assert payload['reward_batches_by_dataset']['kras'][0]['fragfm'] == 0.8; "
        "assert payload['generator_costs']['fragfm'] == 1.0; "
        "print(json.dumps({'rounds':[{'dataset':'kras','round_index':0}],"
        "'architecture_probabilities':{'hfm_3d':0.25,'fragfm':0.75}}))\""
    )
    monkeypatch.setenv("TAR_PROXYLESS_SEARCH_COMMAND", command)
    monkeypatch.setenv("TAR_PROXYLESS_SEARCH_TIMEOUT_SECONDS", "10")
    service = GeneratorRouterServicer()
    request = router_pb2.RouterProxylessSearchRequest(
        reward_batches_json=json.dumps({"kras": [{"hfm_3d": 0.2, "fragfm": 0.8}]}),
        generator_costs_json=json.dumps({"hfm_3d": 5.0, "fragfm": 1.0}),
        cost_weight=0.1,
        learning_rate=1.0,
    )

    response = await service.RunProxylessSearch(request, None)
    result = json.loads(response.result_json)

    assert response.acknowledged is True
    assert response.round_count == 1
    assert result["architecture_probabilities"] == {"hfm_3d": 0.25, "fragfm": 0.75}


def _proxyless_runner_payload() -> dict[str, object]:
    return {
        "reward_batches_by_dataset": {
            "kras": [
                {"hfm_3d": 0.2, "fragfm": 0.8},
                {"hfm_3d": 0.1, "fragfm": 0.9},
            ]
        },
        "generator_costs": {"hfm_3d": 5.0, "fragfm": 1.0},
        "cost_weight": 0.1,
        "learning_rate": 1.0,
        "temperature": 1.0,
    }


def test_tar_proxyless_runner_executes_shared_scheduler() -> None:
    from generator_router_svc.tar_proxyless_runner import run_proxyless_search

    result = run_proxyless_search(_proxyless_runner_payload())

    assert len(result["rounds"]) == 2
    assert result["architecture_probabilities"]["fragfm"] > (
        result["architecture_probabilities"]["hfm_3d"]
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
    assert result["architecture_probabilities"]["fragfm"] > (
        result["architecture_probabilities"]["hfm_3d"]
    )


@pytest.mark.asyncio
async def test_generator_router_service_uses_builtin_proxyless_runner_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from generator_router_svc.main import GeneratorRouterServicer
    from mf_core.proto_gen.moleculeforge.v1.generator import router_pb2

    monkeypatch.setenv(
        "TAR_PROXYLESS_SEARCH_COMMAND",
        f"{sys.executable} -m generator_router_svc.tar_proxyless_runner",
    )
    monkeypatch.setenv("TAR_PROXYLESS_SEARCH_TIMEOUT_SECONDS", "120")
    service = GeneratorRouterServicer()
    request = router_pb2.RouterProxylessSearchRequest(
        reward_batches_json=json.dumps(
            _proxyless_runner_payload()["reward_batches_by_dataset"]
        ),
        generator_costs_json=json.dumps(_proxyless_runner_payload()["generator_costs"]),
        cost_weight=0.1,
        learning_rate=1.0,
        temperature=1.0,
    )

    response = await service.RunProxylessSearch(request, None)
    result = json.loads(response.result_json)

    assert response.acknowledged is True
    assert response.round_count == 2
    assert result["generator_names"] == list(GENERATOR_NAMES)
    assert result["architecture_probabilities"]["fragfm"] > (
        result["architecture_probabilities"]["hfm_3d"]
    )


def test_hypseek_teacher_response_builds_distribution_from_score_records() -> None:
    from generator_router_svc.main import hypseek_teacher_response

    response = hypseek_teacher_response(
        {
            "oracle_feedback": [
                {"hypseek_score": 2.0},
                {"hypseek_score": 4.0},
            ],
            "score_field": "hypseek_score",
            "min_score": 0.0,
            "max_score": 4.0,
        }
    )

    assert response == {
        "oracle_name": "hypseek",
        "teacher_distribution": [0.5, 1.0],
    }


def test_hypseek_teacher_endpoint_returns_teacher_distribution() -> None:
    from fastapi.testclient import TestClient
    from generator_router_svc.main import hypseek_app

    client = TestClient(hypseek_app)

    response = client.post(
        "/teacher",
        json={
            "oracle_feedback": [
                {"normalized_score": 0.25},
                {"normalized_score": 0.75},
            ],
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "oracle_name": "hypseek",
        "teacher_distribution": [0.25, 0.75],
    }


def test_hypseek_teacher_deployment_wires_router_url() -> None:
    expected_url = "http://hypseek-teacher-svc:8012/teacher"
    compose = (ROOT / "infra/docker/docker-compose.dev.yml").read_text(encoding="utf-8")
    k8s = (ROOT / "infra/kubernetes/deployments/moleculeforge-services.yaml").read_text(
        encoding="utf-8"
    )
    helm_values = (ROOT / "infra/helm/moleculeforge/values.yaml").read_text(
        encoding="utf-8"
    )

    assert "hypseek-teacher-svc:" in compose
    assert "generator_router_svc.main:hypseek_app" in compose
    assert f"HYPSEEK_TEACHER_URL: {expected_url}" in compose

    assert "name: hypseek-teacher-svc" in k8s
    assert "containerPort: 8012" in k8s
    assert f'value: "{expected_url}"' in k8s

    assert "hypseek-teacher-svc:" in helm_values
    assert "repository: generator-router-svc" in helm_values
    assert f"HYPSEEK_TEACHER_URL: {expected_url}" in helm_values


def test_generator_router_deployment_wires_proxyless_search_env() -> None:
    compose = (ROOT / "infra/docker/docker-compose.dev.yml").read_text(encoding="utf-8")
    k8s = (ROOT / "infra/kubernetes/deployments/moleculeforge-services.yaml").read_text(
        encoding="utf-8"
    )
    helm_values = (ROOT / "infra/helm/moleculeforge/values.yaml").read_text(
        encoding="utf-8"
    )

    assert "TAR_PROXYLESS_SEARCH_COMMAND: ${TAR_PROXYLESS_SEARCH_COMMAND:-}" in compose
    assert (
        "TAR_PROXYLESS_SEARCH_TIMEOUT_SECONDS: "
        "${TAR_PROXYLESS_SEARCH_TIMEOUT_SECONDS:-300}"
    ) in compose

    assert "name: tar-proxyless-search-config" in k8s
    assert "name: TAR_PROXYLESS_SEARCH_COMMAND" in k8s
    assert "key: proxyless-search-command" in k8s
    assert "name: TAR_PROXYLESS_SEARCH_TIMEOUT_SECONDS" in k8s
    assert "key: proxyless-search-timeout-seconds" in k8s

    assert "TAR_PROXYLESS_SEARCH_TIMEOUT_SECONDS: \"300\"" in helm_values
    assert "TAR_PROXYLESS_SEARCH_COMMAND:" in helm_values
    assert "key: proxyless-search-command" in helm_values
