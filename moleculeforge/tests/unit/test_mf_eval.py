from __future__ import annotations

import builtins
import sys
from pathlib import Path

import pytest
import torch


def test_pairwise_distance_distortion_metrics() -> None:
    from mf_eval.distortion import pairwise_distance_distortion

    metrics = pairwise_distance_distortion(
        torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
        torch.tensor([[0.0, 2.0], [2.0, 0.0]]),
    )

    assert metrics["n_pairs"] == 1
    assert metrics["mean_absolute_error"] == pytest.approx(1.0)
    assert metrics["mean_relative_error"] == pytest.approx(1.0)
    assert metrics["spearman_r"] == 1.0


def test_pairwise_distance_distortion_rejects_shape_mismatch() -> None:
    from mf_eval.distortion import pairwise_distance_distortion

    with pytest.raises(ValueError, match="shape"):
        pairwise_distance_distortion(torch.zeros(2, 2), torch.zeros(3, 3))


def test_evaluate_moses_rejects_missing_rdkit(monkeypatch: pytest.MonkeyPatch) -> None:
    from mf_eval.molecule.moses import evaluate_moses

    real_import = builtins.__import__

    def block_rdkit_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "rdkit" or name.startswith("rdkit."):
            raise ImportError("blocked rdkit")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", block_rdkit_import)

    with pytest.raises(RuntimeError, match="RDKit is required"):
        evaluate_moses(["CCO"], ["CCN"])


def test_find_activity_cliffs_and_separation_metric() -> None:
    from mf_eval.cliff_analysis import cliff_separation_auroc, find_activity_cliffs

    cliffs = find_activity_cliffs(
        ["CCO", "CCN", "c1ccccc1"],
        [1.0, 3.0, 1.1],
        similarity_threshold=0.2,
        activity_delta_threshold=1.0,
    )
    assert cliffs
    assert {"i", "j", "similarity", "activity_delta"} <= set(cliffs[0])

    score = cliff_separation_auroc(
        torch.tensor([[0.0, 0.0], [0.1, 0.0], [5.0, 5.0]], dtype=torch.float32),
        [True, False, False],
    )
    assert score is not None
    assert 0.0 <= score <= 1.0

    assert cliff_separation_auroc(torch.zeros(2, 2), [True, True]) is None


def test_hypervolume_2d_and_improvement() -> None:
    from mf_eval.hv_evaluator import (
        filter_non_dominated,
        hypervolume_2d,
        hypervolume_improvement,
    )

    front = filter_non_dominated([[1.0, 1.0], [2.0, 2.0], [3.0, 1.0]])

    assert front.tolist() == [[2.0, 2.0], [3.0, 1.0]]
    assert hypervolume_2d(front, reference=[0.0, 0.0]) == pytest.approx(5.0)
    assert hypervolume_improvement(
        [3.0, 2.0],
        front,
        reference=[0.0, 0.0],
    ) == pytest.approx(1.0)


def test_probability_of_feasibility_scores_constraint_satisfaction() -> None:
    from mf_eval.hv_evaluator import probability_of_feasibility

    values = probability_of_feasibility(
        mu=torch.tensor([[0.5, 2.0], [1.5, 4.0]]),
        sigma=torch.tensor([[0.1, 0.1], [0.1, 0.1]]),
        lower_bounds=torch.tensor([0.0, 1.0]),
        upper_bounds=torch.tensor([1.0, 3.0]),
    )

    assert values.shape == (2,)
    assert values[0] > 0.99
    assert values[1] < 1e-4


def test_constrained_hypervolume_improvement_multiplies_hvi_by_pof() -> None:
    from mf_eval.hv_evaluator import constrained_hypervolume_improvement

    front = [[2.0, 2.0], [3.0, 1.0]]
    unconstrained = constrained_hypervolume_improvement(
        candidate=[3.0, 2.0],
        front=front,
        reference=[0.0, 0.0],
        constraint_mu=[0.5],
        constraint_sigma=[0.01],
        lower_bounds=[0.0],
        upper_bounds=[1.0],
    )
    infeasible = constrained_hypervolume_improvement(
        candidate=[3.0, 2.0],
        front=front,
        reference=[0.0, 0.0],
        constraint_mu=[2.0],
        constraint_sigma=[0.01],
        lower_bounds=[0.0],
        upper_bounds=[1.0],
    )

    assert unconstrained == pytest.approx(1.0)
    assert infeasible < 1e-6


def test_rank_constrained_hvi_candidates_orders_by_feasible_improvement() -> None:
    from mf_eval.hv_evaluator import rank_constrained_hvi_candidates

    ranked = rank_constrained_hvi_candidates(
        candidates=[[3.0, 2.0], [2.5, 2.5], [4.0, 3.0]],
        front=[[2.0, 2.0], [3.0, 1.0]],
        reference=[0.0, 0.0],
        constraint_mu=[[0.5], [2.0], [0.5]],
        constraint_sigma=[[0.01], [0.01], [0.01]],
        lower_bounds=[0.0],
        upper_bounds=[1.0],
    )

    assert [item["candidate_index"] for item in ranked] == [2, 0, 1]
    assert ranked[0]["candidate"] == [4.0, 3.0]
    assert ranked[0]["score"] > ranked[1]["score"] > ranked[2]["score"]
    assert ranked[2]["probability_of_feasibility"] < 1e-6


def test_rank_tangent_gp_constrained_hvi_candidates_uses_embedding_gp_loop() -> None:
    from mf_eval.hv_evaluator import rank_tangent_gp_constrained_hvi_candidates

    ranked = rank_tangent_gp_constrained_hvi_candidates(
        candidate_embeddings=[[0.05], [0.95]],
        observed_embeddings=[[0.0], [1.0]],
        observed_objectives=[[1.0, 1.0], [4.0, 3.0]],
        observed_constraints=[[0.5], [0.5]],
        front=[[1.0, 1.0]],
        reference=[0.0, 0.0],
        lower_bounds=[0.0],
        upper_bounds=[1.0],
        lengthscale=0.3,
    )

    assert [item["candidate_index"] for item in ranked] == [1, 0]
    assert ranked[0]["predicted_objective"][0] > ranked[1]["predicted_objective"][0]
    assert ranked[0]["probability_of_feasibility"] > 0.9
    assert ranked[0]["score"] > ranked[1]["score"]


def test_expected_hypervolume_improvement_uses_gp_uncertainty() -> None:
    from mf_eval.hv_evaluator import expected_hypervolume_improvement

    values = expected_hypervolume_improvement(
        mu=torch.tensor([[2.0, 2.0], [2.0, 2.0]]),
        sigma=torch.tensor([[0.01, 0.01], [1.0, 1.0]]),
        front=torch.tensor([[2.0, 2.0]]),
        reference=torch.tensor([0.0, 0.0]),
    )

    assert values.shape == (2,)
    assert values[1] > values[0]


def test_rank_tangent_gp_constrained_ehvi_candidates_scores_ehvi_by_pof() -> None:
    from mf_eval.hv_evaluator import rank_tangent_gp_constrained_ehvi_candidates

    ranked = rank_tangent_gp_constrained_ehvi_candidates(
        candidate_embeddings=[[0.05], [0.95]],
        observed_embeddings=[[0.0], [1.0]],
        observed_objectives=[[1.0, 1.0], [4.0, 3.0]],
        observed_constraints=[[0.5], [0.5]],
        front=[[1.0, 1.0]],
        reference=[0.0, 0.0],
        lower_bounds=[0.0],
        upper_bounds=[1.0],
        lengthscale=0.3,
    )

    assert ranked[0]["expected_hypervolume_improvement"] > 0.0
    assert ranked[0]["probability_of_feasibility"] > 0.9
    assert ranked[0]["score"] == pytest.approx(
        ranked[0]["expected_hypervolume_improvement"]
        * ranked[0]["probability_of_feasibility"]
    )
    assert ranked[0]["score"] >= ranked[1]["score"]
    assert "predicted_objective_sigma" in ranked[0]


def test_humu_logmap_tangent_features_maps_lorentz_points_to_base_tangent_space() -> None:
    from mf_eval.hv_evaluator import humu_logmap_tangent_features
    from mf_humu.manifold.lorentz import LorentzManifold

    manifold = LorentzManifold(curvature=1.0)
    base = manifold.origin(dim=1)
    tangent = torch.tensor([[0.0, 0.2]], dtype=torch.float32)
    point = manifold.expmap(base.unsqueeze(0), tangent)

    features = humu_logmap_tangent_features(point, base_embedding=base)

    assert torch.allclose(features, tangent, atol=1e-4)
    assert torch.allclose(
        manifold.inner(base.unsqueeze(0), features),
        torch.zeros(1, 1),
        atol=1e-5,
    )


def test_rank_humu_logmap_gp_constrained_ehvi_candidates_uses_logmap_features() -> None:
    from mf_eval.hv_evaluator import (
        rank_humu_logmap_gp_constrained_ehvi_candidates,
        rank_tangent_gp_constrained_ehvi_candidates,
    )
    from mf_humu.manifold.lorentz import LorentzManifold

    manifold = LorentzManifold(curvature=1.0)
    base = manifold.origin(dim=1)
    observed_tangent = torch.tensor([[0.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    candidate_tangent = torch.tensor([[0.0, 0.05], [0.0, 0.95]], dtype=torch.float32)
    observed_humu = manifold.expmap(base.unsqueeze(0), observed_tangent)
    candidate_humu = manifold.expmap(base.unsqueeze(0), candidate_tangent)
    expected = rank_tangent_gp_constrained_ehvi_candidates(
        candidate_embeddings=candidate_tangent,
        observed_embeddings=observed_tangent,
        observed_objectives=[[1.0, 1.0], [4.0, 3.0]],
        observed_constraints=[[0.5], [0.5]],
        front=[[1.0, 1.0]],
        reference=[0.0, 0.0],
        lower_bounds=[0.0],
        upper_bounds=[1.0],
        lengthscale=0.3,
    )

    ranked = rank_humu_logmap_gp_constrained_ehvi_candidates(
        candidate_humu_embeddings=candidate_humu,
        observed_humu_embeddings=observed_humu,
        observed_objectives=[[1.0, 1.0], [4.0, 3.0]],
        observed_constraints=[[0.5], [0.5]],
        base_embedding=base,
        front=[[1.0, 1.0]],
        reference=[0.0, 0.0],
        lower_bounds=[0.0],
        upper_bounds=[1.0],
        lengthscale=0.3,
    )

    assert [item["candidate_index"] for item in ranked] == [
        item["candidate_index"]
        for item in expected
    ]
    assert ranked[0]["tangent_embedding"] == pytest.approx(
        candidate_tangent[ranked[0]["candidate_index"]].tolist(),
        abs=1e-4,
    )


@pytest.mark.asyncio
async def test_async_pcbo_oracle_loop_samples_ranked_candidates_and_updates_observations() -> None:
    from mf_eval.hv_evaluator import (
        async_pcbo_oracle_loop,
        filter_non_dominated,
        rank_tangent_gp_constrained_hvi_candidates,
    )

    calls = []
    candidate_embeddings = [[0.05], [0.95], [0.75]]
    observed_embeddings = [[0.0], [1.0]]
    observed_objectives = [[1.0, 1.0], [4.0, 3.0]]
    observed_constraints = [[0.5], [0.5]]
    expected_ranked = rank_tangent_gp_constrained_hvi_candidates(
        candidate_embeddings=candidate_embeddings,
        observed_embeddings=observed_embeddings,
        observed_objectives=observed_objectives,
        observed_constraints=observed_constraints,
        front=filter_non_dominated(observed_objectives),
        reference=[0.0, 0.0],
        lower_bounds=[0.0],
        upper_bounds=[1.0],
        lengthscale=0.3,
    )
    expected_selected = [
        int(item["candidate_index"])
        for item in expected_ranked[:2]
    ]

    async def oracle_evaluate(request: dict) -> dict:
        calls.append(request)
        candidate_index = int(request["candidate_index"])
        return {
            "objectives": [5.0 + candidate_index, 4.0 + candidate_index],
            "constraints": [0.5],
            "source": "unit-oracle",
        }

    result = await async_pcbo_oracle_loop(
        candidate_embeddings=candidate_embeddings,
        observed_embeddings=observed_embeddings,
        observed_objectives=observed_objectives,
        observed_constraints=observed_constraints,
        oracle_evaluate=oracle_evaluate,
        reference=[0.0, 0.0],
        lower_bounds=[0.0],
        upper_bounds=[1.0],
        batch_size=2,
        n_iterations=1,
        lengthscale=0.3,
    )

    assert result["selected_indices"] == expected_selected
    assert [call["candidate_index"] for call in calls] == expected_selected
    assert calls[0]["candidate_embedding"] == pytest.approx(
        candidate_embeddings[expected_selected[0]]
    )
    assert calls[0]["acquisition"]["candidate_index"] == expected_selected[0]
    assert result["oracle_results"][0]["source"] == "unit-oracle"
    assert torch.allclose(
        result["observed_embeddings"],
        torch.tensor([
            *observed_embeddings,
            *[candidate_embeddings[index] for index in expected_selected],
        ]),
    )
    assert torch.allclose(
        result["observed_objectives"],
        torch.tensor([
            *observed_objectives,
            *[[5.0 + index, 4.0 + index] for index in expected_selected],
        ]),
    )
    assert torch.allclose(
        result["observed_constraints"],
        torch.tensor([[0.5], [0.5], [0.5], [0.5]]),
    )


@pytest.mark.asyncio
async def test_pcbo_optimization_scheduler_runs_candidate_oracle_update_rounds() -> None:
    from mf_eval.hv_evaluator import PCBOOptimizationScheduler

    provider_calls = []
    oracle_calls = []

    async def candidate_provider(state: dict) -> list[list[float]]:
        provider_calls.append(state)
        round_index = int(state["round_index"])
        return [[0.25 + round_index * 0.5], [0.45 + round_index * 0.5]]

    async def oracle_evaluate(request: dict) -> dict:
        oracle_calls.append(request)
        round_index = int(request["round_index"])
        return {
            "objectives": [5.0 + round_index, 4.0 + round_index],
            "constraints": [0.5],
        }

    scheduler = PCBOOptimizationScheduler(
        candidate_provider=candidate_provider,
        oracle_evaluate=oracle_evaluate,
        reference=[0.0, 0.0],
        lower_bounds=[0.0],
        upper_bounds=[1.0],
        batch_size=1,
        n_rounds=2,
        lengthscale=0.3,
    )

    result = await scheduler.run(
        observed_embeddings=[[0.0], [1.0]],
        observed_objectives=[[1.0, 1.0], [4.0, 3.0]],
        observed_constraints=[[0.5], [0.5]],
    )

    assert [call["round_index"] for call in provider_calls] == [0, 1]
    assert provider_calls[1]["observed_embeddings"].shape[0] == 3
    assert [call["round_index"] for call in oracle_calls] == [0, 1]
    assert len(result["rounds"]) == 2
    assert result["rounds"][0]["selected_indices"]
    assert result["observed_embeddings"].shape[0] == 4
    assert result["observed_objectives"].tolist()[-2:] == [[5.0, 4.0], [6.0, 5.0]]
    assert result["observed_constraints"].tolist()[-2:] == [[0.5], [0.5]]


@pytest.mark.asyncio
async def test_pareto_bo_service_runs_pcbo_scheduler_with_json_result() -> None:
    from pareto_bo.service import ParetoBOService

    async def candidate_provider(state: dict) -> list[list[float]]:
        round_index = int(state["round_index"])
        return [[0.25 + round_index * 0.5], [0.45 + round_index * 0.5]]

    async def oracle_evaluate(request: dict) -> dict:
        round_index = int(request["round_index"])
        return {
            "objectives": [5.0 + round_index, 4.0 + round_index],
            "constraints": [0.5],
        }

    service = ParetoBOService(
        candidate_provider=candidate_provider,
        oracle_evaluate=oracle_evaluate,
    )

    result = await service.optimize(
        {
            "observed_embeddings": [[0.0], [1.0]],
            "observed_objectives": [[1.0, 1.0], [4.0, 3.0]],
            "observed_constraints": [[0.5], [0.5]],
            "reference": [0.0, 0.0],
            "lower_bounds": [0.0],
            "upper_bounds": [1.0],
            "batch_size": 1,
            "n_rounds": 2,
            "lengthscale": 0.3,
        }
    )

    assert len(result["rounds"]) == 2
    assert result["rounds"][0]["selected_indices"]
    assert result["observed_embeddings"][-1] == pytest.approx([0.75])
    assert result["observed_objectives"][-2:] == [[5.0, 4.0], [6.0, 5.0]]
    assert result["observed_constraints"][-2:] == [[0.5], [0.5]]


@pytest.mark.asyncio
async def test_pareto_bo_service_uses_configured_json_commands(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    from pareto_bo.service import ParetoBOService

    provider = tmp_path / "candidate_provider.py"
    provider.write_text(
        "import json, sys\n"
        "payload = json.load(sys.stdin)\n"
        "assert payload['round_index'] == 0\n"
        "assert payload['observed_embeddings'] == [[0.0], [1.0]]\n"
        "print(json.dumps({'candidate_embeddings': [[0.25], [0.75]]}))\n",
        encoding="utf-8",
    )
    oracle = tmp_path / "oracle_evaluate.py"
    oracle.write_text(
        "import json, sys\n"
        "payload = json.load(sys.stdin)\n"
        "assert payload['round_index'] == 0\n"
        "assert 'acquisition' in payload\n"
        "candidate_index = int(payload['candidate_index'])\n"
        "print(json.dumps({"
        "'objectives': [5.0 + candidate_index, 4.0 + candidate_index], "
        "'constraints': [0.5], "
        "'source': 'pcbo-command'"
        "}))\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("PARETO_BO_CANDIDATE_PROVIDER", raising=False)
    monkeypatch.delenv("PARETO_BO_ORACLE_EVALUATE", raising=False)
    monkeypatch.setenv("PARETO_BO_CANDIDATE_PROVIDER_COMMAND", f"{sys.executable} {provider}")
    monkeypatch.setenv("PARETO_BO_ORACLE_EVALUATE_COMMAND", f"{sys.executable} {oracle}")

    service = ParetoBOService.from_env()
    result = await service.optimize(
        {
            "observed_embeddings": [[0.0], [1.0]],
            "observed_objectives": [[1.0, 1.0], [4.0, 3.0]],
            "observed_constraints": [[0.5], [0.5]],
            "reference": [0.0, 0.0],
            "lower_bounds": [0.0],
            "upper_bounds": [1.0],
            "batch_size": 1,
            "n_rounds": 1,
            "lengthscale": 0.3,
        }
    )

    assert result["rounds"][0]["oracle_results"][0]["source"] == "pcbo-command"
    selected_index = result["rounds"][0]["selected_indices"][0]
    assert result["observed_embeddings"][-1] == pytest.approx(
        result["rounds"][0]["candidate_embeddings"][selected_index]
    )
    assert result["observed_objectives"][-1] == pytest.approx(
        [5.0 + selected_index, 4.0 + selected_index]
    )


def test_pareto_bo_command_preflight_rejects_missing_executable() -> None:
    from pareto_bo import service as module

    with pytest.raises(RuntimeError, match="not found"):
        module._run_json_command(
            "missing-pareto-provider --json",
            {"round_index": 0},
            source_env="PARETO_BO_CANDIDATE_PROVIDER_COMMAND",
        )
    with pytest.raises(RuntimeError, match="not found"):
        module._run_json_command(
            "missing-pareto-oracle --json",
            {"candidate_index": 0},
            source_env="PARETO_BO_ORACLE_EVALUATE_COMMAND",
        )


@pytest.mark.asyncio
async def test_pareto_bo_fastapi_endpoint_uses_env_configured_service(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    from pareto_bo import service as module

    provider = tmp_path / "candidate_provider.py"
    provider.write_text(
        "import json, sys\n"
        "payload = json.load(sys.stdin)\n"
        "assert payload['round_index'] == 0\n"
        "print(json.dumps({'candidate_embeddings': [[0.25], [0.75]]}))\n",
        encoding="utf-8",
    )
    oracle = tmp_path / "oracle_evaluate.py"
    oracle.write_text(
        "import json, sys\n"
        "payload = json.load(sys.stdin)\n"
        "candidate_index = int(payload['candidate_index'])\n"
        "print(json.dumps({"
        "'objectives': [5.0 + candidate_index, 4.0 + candidate_index], "
        "'constraints': [0.5]"
        "}))\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("PARETO_BO_CANDIDATE_PROVIDER", raising=False)
    monkeypatch.delenv("PARETO_BO_ORACLE_EVALUATE", raising=False)
    monkeypatch.setenv("PARETO_BO_CANDIDATE_PROVIDER_COMMAND", f"{sys.executable} {provider}")
    monkeypatch.setenv("PARETO_BO_ORACLE_EVALUATE_COMMAND", f"{sys.executable} {oracle}")

    route_paths = {getattr(route, "path", "") for route in module.rest_app.routes}
    result = await module.optimize_endpoint(
        {
            "observed_embeddings": [[0.0], [1.0]],
            "observed_objectives": [[1.0, 1.0], [4.0, 3.0]],
            "observed_constraints": [[0.5], [0.5]],
            "reference": [0.0, 0.0],
            "lower_bounds": [0.0],
            "upper_bounds": [1.0],
            "batch_size": 1,
            "n_rounds": 1,
            "lengthscale": 0.3,
        }
    )

    assert "/v1/pareto-bo/optimize" in route_paths
    assert result["rounds"][0]["selected_indices"]
    assert result["observed_constraints"][-1] == [0.5]


def test_pareto_bo_deployment_wires_service_env() -> None:
    root = Path(__file__).resolve().parents[2]
    compose = (root / "infra/docker/docker-compose.dev.yml").read_text(encoding="utf-8")
    k8s = (root / "infra/kubernetes/deployments/moleculeforge-services.yaml").read_text(
        encoding="utf-8"
    )
    helm_values = (root / "infra/helm/moleculeforge/values.yaml").read_text(
        encoding="utf-8"
    )

    assert "pareto-bo-svc" in compose
    assert "pareto-bo-svc" in k8s
    assert "pareto-bo-svc" in helm_values
    assert "/workspace/pipelines/pareto_bo/src" in compose
    for env_name in (
        "PARETO_BO_CANDIDATE_PROVIDER",
        "PARETO_BO_CANDIDATE_PROVIDER_COMMAND",
        "PARETO_BO_ORACLE_EVALUATE",
        "PARETO_BO_ORACLE_EVALUATE_COMMAND",
        "PARETO_BO_COMMAND_TIMEOUT_SECONDS",
    ):
        assert env_name in compose
        assert env_name in k8s
        assert env_name in helm_values

    assert (
        "PARETO_BO_COMMAND_TIMEOUT_SECONDS: "
        "${PARETO_BO_COMMAND_TIMEOUT_SECONDS:-300}"
    ) in compose
    assert "name: pareto-bo-config" in k8s
    assert "configMapKeyRef:" in k8s
    assert "envValueFrom:" in helm_values


def test_hypervolume_rejects_non_2d_contract() -> None:
    from mf_eval.hv_evaluator import hypervolume_2d

    with pytest.raises(ValueError, match="2D"):
        hypervolume_2d([[1.0, 1.0, 1.0]], reference=[0.0, 0.0, 0.0])


# W3: PCBO reference provider / evaluator tests


def test_tangent_space_noise_provider_generates_candidates_from_observed() -> None:
    from pareto_bo.providers import TangentSpaceNoiseCandidateProvider

    provider = TangentSpaceNoiseCandidateProvider(noise_scale=0.1, n_candidates=4)
    state = {
        "round_index": 0,
        "observed_embeddings": [[0.0, 0.0], [1.0, 1.0]],
        "observed_objectives": [[1.0, 1.0], [2.0, 2.0]],
        "observed_constraints": [[0.5], [0.5]],
    }
    candidates = provider.propose(state)
    import torch

    assert isinstance(candidates, torch.Tensor)
    assert candidates.shape == (4, 2)


def test_tangent_space_noise_provider_respects_env_vars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pareto_bo.providers import TangentSpaceNoiseCandidateProvider

    monkeypatch.setenv("PARETO_BO_CANDIDATE_NOISE_SCALE", "0.5")
    monkeypatch.setenv("PARETO_BO_CANDIDATE_COUNT", "8")
    provider = TangentSpaceNoiseCandidateProvider.from_env()
    state = {
        "round_index": 0,
        "observed_embeddings": [[0.5]],
        "observed_objectives": [[1.0, 1.0]],
        "observed_constraints": [[0.5]],
    }
    candidates = provider.propose(state)
    assert candidates.shape[0] == 8


@pytest.mark.asyncio
async def test_local_oracle_evaluator_falls_back_to_embedding_proxy_without_oracle() -> None:
    from pareto_bo.providers import LocalOracleEvaluator

    evaluator = LocalOracleEvaluator()
    result = await evaluator(
        {
            "candidate_index": 0,
            "candidate_embedding": [0.5, 0.3],
            "acquisition": {},
        }
    )
    assert "objectives" in result
    assert "constraints" in result
    assert len(result["objectives"]) == 2
    assert len(result["constraints"]) == 1
    assert result.get("source") == "embedding_proxy"


@pytest.mark.asyncio
async def test_pareto_bo_service_from_env_uses_default_providers_when_no_env_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pareto_bo.service import ParetoBOService

    monkeypatch.delenv("PARETO_BO_CANDIDATE_PROVIDER", raising=False)
    monkeypatch.delenv("PARETO_BO_CANDIDATE_PROVIDER_COMMAND", raising=False)
    monkeypatch.delenv("PARETO_BO_ORACLE_EVALUATE", raising=False)
    monkeypatch.delenv("PARETO_BO_ORACLE_EVALUATE_COMMAND", raising=False)

    service = ParetoBOService.from_env()
    result = await service.optimize(
        {
            "observed_embeddings": [[0.0, 0.0], [1.0, 1.0]],
            "observed_objectives": [[1.0, 1.0], [2.0, 2.0]],
            "observed_constraints": [[0.5], [0.5]],
            "reference": [0.0, 0.0],
            "lower_bounds": [0.0],
            "upper_bounds": [1.0],
            "batch_size": 1,
            "n_rounds": 1,
            "lengthscale": 0.3,
        }
    )
    assert len(result["rounds"]) == 1
    assert result["rounds"][0]["selected_indices"]
