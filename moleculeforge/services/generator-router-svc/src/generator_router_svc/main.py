"""Generator Router Service - gRPC server for task-aware generator routing."""
import asyncio
import json
import logging
import os
import shlex
import subprocess
import urllib.error
import urllib.request
from concurrent import futures

import grpc
import torch
from fastapi import FastAPI, HTTPException
from mf_core.artifacts import (
    CommandRequirement,
    RequirementStatus,
    check_command,
    require_available,
)
from mf_core.proto_gen.moleculeforge.v1.generator import router_pb2, router_pb2_grpc
from mf_core.routing.cross_paradigm_kd import (
    CrossParadigmKDLayer,
    hypseek_teacher_feedback,
)
from mf_core.routing.task_router import (
    GENERATOR_NAMES,
    ProxylessSearchScheduler,
    TaskAwareRouter,
    TaskProfile,
)

logger = logging.getLogger(__name__)
hypseek_app = FastAPI(title="HypSeek Teacher Service", version="0.1.0")
_HYPSEEK_TEACHER_COMMAND = CommandRequirement(
    "hypseek_teacher_command",
    "HYPSEEK_TEACHER_COMMAND",
)
_TAR_PROXYLESS_SEARCH_COMMAND = CommandRequirement(
    "tar_proxyless_search_command",
    "TAR_PROXYLESS_SEARCH_COMMAND",
)
_COMMAND_REQUIREMENTS = (
    _HYPSEEK_TEACHER_COMMAND,
    _TAR_PROXYLESS_SEARCH_COMMAND,
)


def runtime_status() -> list[dict]:
    return [status.to_dict() for status in _runtime_statuses()]


def _runtime_statuses() -> list[RequirementStatus]:
    statuses: list[RequirementStatus] = []
    for requirement in _COMMAND_REQUIREMENTS:
        if os.environ.get(requirement.env_var, "").strip():
            statuses.append(check_command(requirement))
    return statuses


def _require_command_available(
    requirement: CommandRequirement,
    command: str,
) -> None:
    env = {**os.environ, requirement.env_var: command}
    require_available([check_command(requirement, env=env)])


def hypseek_teacher_response(payload: dict) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise ValueError("HypSeek teacher payload must be a JSON object")
    records = payload.get("records") or payload.get("oracle_feedback")
    if not isinstance(records, list) or not records:
        raise ValueError("HypSeek teacher payload requires non-empty oracle_feedback")
    score_field = str(payload.get("score_field") or "normalized_score")
    if score_field == "normalized_score":
        min_score = float(payload.get("min_score", 0.0))
        max_score = float(payload.get("max_score", 1.0))
    else:
        if "min_score" not in payload or "max_score" not in payload:
            raise ValueError("custom HypSeek score_field requires min_score and max_score")
        min_score = float(payload["min_score"])
        max_score = float(payload["max_score"])
    return hypseek_teacher_feedback(
        records,
        score_field=score_field,
        min_score=min_score,
        max_score=max_score,
        higher_is_better=bool(payload.get("higher_is_better", True)),
    )


@hypseek_app.post("/teacher")
async def hypseek_teacher_endpoint(payload: dict) -> dict[str, object]:
    try:
        return hypseek_teacher_response(payload)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@hypseek_app.get("/healthz")
async def hypseek_healthz() -> dict[str, str]:
    return {"status": "ok", "service": "hypseek_teacher"}


class GeneratorRouterServicer:
    def __init__(self):
        self.router = TaskAwareRouter(n_generators=len(GENERATOR_NAMES))
        self.kd_layer = CrossParadigmKDLayer(n_generators=len(GENERATOR_NAMES))
        self.hypseek_teacher_command = os.getenv("HYPSEEK_TEACHER_COMMAND", "").strip()
        self.hypseek_teacher_url = os.getenv("HYPSEEK_TEACHER_URL", "").strip()
        self.proxyless_search_command = os.getenv("TAR_PROXYLESS_SEARCH_COMMAND", "").strip()
        self.proxyless_search_timeout_seconds = float(
            os.getenv("TAR_PROXYLESS_SEARCH_TIMEOUT_SECONDS", "300")
        )

    async def Route(self, request, context):
        """Route a generation request to generators using the shared TAR."""
        n_select = max(1, min(int(getattr(request, "n_select", 2) or 2), len(GENERATOR_NAMES)))
        profile = _profile_from_request(request)
        _apply_generator_performance(self.router, request)
        hciv = _hciv_from_request(request, self.router.hciv_dim)
        weights = self.router.forward(hciv, profile)
        selected = sorted(weights.items(), key=lambda item: item[1], reverse=True)[:n_select]
        selected_generators = [name for name, _ in selected]
        selected_weights = [weight for _, weight in selected]
        total = sum(selected_weights)
        if total > 0:
            selected_weights = [weight / total for weight in selected_weights]

        return type(
            "RouteResponse",
            (),
            {
                "request_id": getattr(request, "request_id", ""),
                "selected_generators": selected_generators,
                "selection_weights": selected_weights,
                "strategy": "task_aware_router",
                "expected_rewards": [weights[name] for name in selected_generators],
                "targets": [
                    {
                        "generator_id": name,
                        "weight": weight,
                        "endpoint": f"{name}:50051",
                    }
                    for name, weight in zip(selected_generators, selected_weights, strict=True)
                ],
            },
        )()

    async def SubmitFeedback(self, request, context):
        """Submit reward feedback to online learner."""
        generator_name = getattr(request, "generator_name", "")
        generator_idx = getattr(request, "generator_idx", None)
        if not generator_name and generator_idx is not None:
            generator_name = GENERATOR_NAMES[int(generator_idx)]
        reward = getattr(request, "reward", 0.0)
        oracle_feedback = _oracle_feedback_from_request(request)
        hypseek_feedback = _hypseek_feedback_from_command(
            self.hypseek_teacher_command,
            generator_name=generator_name,
            reward=float(reward),
            oracle_feedback=oracle_feedback,
        )
        if not hypseek_feedback:
            hypseek_feedback = _hypseek_feedback_from_url(
                self.hypseek_teacher_url,
                generator_name=generator_name,
                reward=float(reward),
                oracle_feedback=oracle_feedback,
            )
        if hypseek_feedback:
            oracle_feedback.append(hypseek_feedback)
        teacher_score = None
        if oracle_feedback:
            generator_idx = _generator_index(generator_name, generator_idx)
            teacher_score = self.kd_layer.update_teacher_scores(
                generator_name,
                generator_idx,
                oracle_feedback,
            )
            reward = teacher_score
        self.router.update_with_feedback(generator_name, reward)
        return type(
            "FeedbackResponse",
            (),
            {
                "acknowledged": True,
                "generator_name": generator_name,
                "teacher_score": teacher_score,
            },
        )()

    async def RunProxylessSearch(self, request, context):
        """Run TAR ProxylessNAS-style architecture search from reward batches."""
        payload = _proxyless_search_payload_from_request(request)
        if self.proxyless_search_command:
            result = _proxyless_search_from_command(
                self.proxyless_search_command,
                payload,
                timeout_seconds=self.proxyless_search_timeout_seconds,
            )
        else:
            scheduler = ProxylessSearchScheduler(
                router=self.router,
                generator_costs=payload["generator_costs"],
                cost_weight=payload["cost_weight"],
                learning_rate=payload["learning_rate"],
                temperature=payload["temperature"],
            )
            result = scheduler.run(payload["reward_batches_by_dataset"])
        return _proxyless_search_response(result)

    async def GetWeights(self, request, context):
        """Get current generator selection weights."""
        weights = self.router.forward(torch.zeros(self.router.hciv_dim), TaskProfile())
        return type(
            "WeightsResponse",
            (),
            {
                "generator_names": list(GENERATOR_NAMES),
                "weights": [weights[name] for name in GENERATOR_NAMES],
                "counts": [
                    self.router.oracle_history[name]["n_calls"]
                    for name in GENERATOR_NAMES
                ],
                "rewards": [
                    self.router.oracle_history[name]["avg_hvi"]
                    for name in GENERATOR_NAMES
                ],
            },
        )()


def _profile_from_request(request) -> TaskProfile:
    prior_weights = {}
    request_weights = list(getattr(request, "generator_weights", []) or [])
    for name, weight in zip(GENERATOR_NAMES, request_weights, strict=False):
        prior_weights[name] = float(weight)
    return TaskProfile(
        target_family=str(getattr(request, "target_family", "")),
        stage=str(getattr(request, "stage", "hit_finding") or "hit_finding"),
        data_richness=float(getattr(request, "data_richness", 100.0) or 100.0),
        novelty_demand=float(getattr(request, "novelty_demand", 0.5) or 0.5),
        multi_target=bool(getattr(request, "multi_target", False)),
        sa_constraint=float(getattr(request, "sa_constraint", 4.0) or 4.0),
        n_samples=int(getattr(request, "n_samples", 100) or 100),
        prior_weights=prior_weights,
    )


def _hciv_from_request(request, hciv_dim: int) -> torch.Tensor:
    values = [float(item) for item in getattr(request, "hciv", []) or []]
    if len(values) == hciv_dim + 1:
        values = values[1:]
    if len(values) < hciv_dim:
        values = values + [0.0] * (hciv_dim - len(values))
    return torch.tensor(values[:hciv_dim], dtype=torch.float32)


def _apply_generator_performance(router, request) -> None:
    performance = list(getattr(request, "generator_performance", []) or [])
    if not performance:
        return
    for name, reward in zip(GENERATOR_NAMES, performance, strict=False):
        if name not in router.oracle_history:
            continue
        router.oracle_history[name]["avg_hvi"] = float(reward)
        router.oracle_history[name]["n_calls"] = max(
            float(router.oracle_history[name].get("n_calls", 0.0)),
            1.0,
        )


def _generator_index(generator_name: str, generator_idx) -> int:
    if generator_idx is not None:
        return int(generator_idx)
    if generator_name not in GENERATOR_NAMES:
        raise ValueError(f"Unknown generator_name: {generator_name}")
    return GENERATOR_NAMES.index(generator_name)


def _oracle_feedback_from_request(request) -> list:
    feedback = getattr(request, "oracle_feedback", None)
    if feedback is None:
        return []
    if isinstance(feedback, list):
        return feedback
    return list(feedback)


def _proxyless_search_payload_from_request(request) -> dict:
    return {
        "reward_batches_by_dataset": _parse_json_object(
            getattr(request, "reward_batches_json", ""),
            "reward_batches_json",
        ),
        "generator_costs": _parse_json_object(
            getattr(request, "generator_costs_json", ""),
            "generator_costs_json",
        ),
        "cost_weight": float(getattr(request, "cost_weight", 0.0)),
        "learning_rate": float(getattr(request, "learning_rate", 0.0)),
        "temperature": float(getattr(request, "temperature", 1.0) or 1.0),
    }


def _parse_json_object(raw_json: str, field_name: str) -> dict:
    if not raw_json:
        raise ValueError(f"{field_name} must be a JSON object")
    try:
        value = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{field_name} must be valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a JSON object")
    return value


def _proxyless_search_from_command(
    command: str,
    payload: dict,
    *,
    timeout_seconds: float,
) -> dict:
    _require_command_available(_TAR_PROXYLESS_SEARCH_COMMAND, command)
    completed = subprocess.run(
        shlex.split(command),
        input=json.dumps(payload, sort_keys=True),
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        raise RuntimeError(f"TAR_PROXYLESS_SEARCH_COMMAND failed: {stderr}")
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("TAR_PROXYLESS_SEARCH_COMMAND returned invalid JSON") from exc
    if not isinstance(result, dict):
        raise RuntimeError("TAR_PROXYLESS_SEARCH_COMMAND must return a JSON object")
    _validate_proxyless_search_result(result, "TAR_PROXYLESS_SEARCH_COMMAND")
    return result


def _validate_proxyless_search_result(result: dict, source: str) -> None:
    if not isinstance(result.get("rounds"), list):
        raise RuntimeError(f"{source} result must contain rounds list")
    if not isinstance(result.get("architecture_probabilities"), dict):
        raise RuntimeError(f"{source} result must contain architecture_probabilities object")


def _proxyless_search_response(result: dict) -> router_pb2.RouterProxylessSearchResponse:
    _validate_proxyless_search_result(result, "Proxyless search")
    architecture_probabilities = result["architecture_probabilities"]
    generator_names = [
        name for name in GENERATOR_NAMES if name in architecture_probabilities
    ]
    generator_names.extend(
        str(name)
        for name in architecture_probabilities
        if str(name) not in set(generator_names)
    )
    return router_pb2.RouterProxylessSearchResponse(
        acknowledged=True,
        result_json=json.dumps(result, sort_keys=True),
        generator_names=generator_names,
        architecture_probabilities=[
            float(architecture_probabilities[name]) for name in generator_names
        ],
        round_count=len(result["rounds"]),
    )


def _hypseek_feedback_from_command(
    command: str,
    *,
    generator_name: str,
    reward: float,
    oracle_feedback: list,
) -> dict | None:
    if not command:
        return None
    _require_command_available(_HYPSEEK_TEACHER_COMMAND, command)
    payload = {
        "generator_name": generator_name,
        "reward": reward,
        "oracle_feedback": list(oracle_feedback),
    }
    completed = subprocess.run(
        shlex.split(command),
        input=json.dumps(payload, sort_keys=True),
        capture_output=True,
        text=True,
        timeout=float(os.getenv("HYPSEEK_TEACHER_TIMEOUT_SECONDS", "60")),
        check=False,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        raise RuntimeError(f"HYPSEEK_TEACHER_COMMAND failed: {stderr}")
    try:
        response = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("HYPSEEK_TEACHER_COMMAND returned invalid JSON") from exc
    if not isinstance(response, dict):
        raise RuntimeError("HYPSEEK_TEACHER_COMMAND must return a JSON object")
    if "oracle_name" not in response:
        response["oracle_name"] = "hypseek"
    if "teacher_distribution" not in response and "normalized_score" not in response:
        raise RuntimeError(
            "HYPSEEK_TEACHER_COMMAND must return teacher_distribution or normalized_score"
        )
    return response


def _hypseek_feedback_from_url(
    url: str,
    *,
    generator_name: str,
    reward: float,
    oracle_feedback: list,
) -> dict | None:
    if not url:
        return None
    payload = {
        "generator_name": generator_name,
        "reward": reward,
        "oracle_feedback": list(oracle_feedback),
    }
    response = _post_json(
        url,
        payload,
        float(os.getenv("HYPSEEK_TEACHER_TIMEOUT_SECONDS", "60")),
    )
    if not isinstance(response, dict):
        raise RuntimeError("HYPSEEK_TEACHER_URL must return a JSON object")
    if "oracle_name" not in response:
        response["oracle_name"] = "hypseek"
    if "teacher_distribution" not in response and "normalized_score" not in response:
        raise RuntimeError(
            "HYPSEEK_TEACHER_URL must return teacher_distribution or normalized_score"
        )
    return response


def _post_json(url: str, payload: dict, timeout_seconds: float) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, sort_keys=True).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(f"HYPSEEK_TEACHER_URL request failed: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError("HYPSEEK_TEACHER_URL returned invalid JSON") from exc


async def serve():
    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=10))
    router_pb2_grpc.add_GeneratorRouterServiceServicer_to_server(
        GeneratorRouterServicer(),
        server,
    )
    server.add_insecure_port("[::]:50052")
    await server.start()
    logger.info("Generator Router Service running on :50052")
    await server.wait_for_termination()


if __name__ == "__main__":
    asyncio.run(serve())
