"""Generator Router Service - gRPC server for task-aware generator routing."""

import asyncio
import hashlib
import json
import logging
import math
import os
import shlex
import subprocess
import tempfile
import time
from collections.abc import Mapping
from concurrent import futures
from pathlib import Path
from typing import Never

import grpc
import torch
from fastapi import FastAPI, HTTPException
from google.protobuf.message import DecodeError
from mf_core.artifacts import (
    CommandRequirement,
    RequirementStatus,
    check_command,
)
from mf_core.geometry import normalize_lorentz_embedding
from mf_core.proto_gen.moleculeforge.v1.core import cig_pb2
from mf_core.proto_gen.moleculeforge.v1.generator import router_pb2, router_pb2_grpc
from mf_core.routing.cross_paradigm_kd import (
    CrossParadigmKDLayer,
    OracleFeedback,
)
from mf_core.routing.task_router import (
    GENERATOR_NAMES,
    ProxylessSearchScheduler,
    TaskAwareRouter,
    TaskProfile,
    minimum_one_largest_remainder,
)

logger = logging.getLogger(__name__)
hypseek_app = FastAPI(title="HypSeek Teacher Service", version="0.1.0")
_TAR_PROXYLESS_SEARCH_COMMAND = CommandRequirement(
    "tar_proxyless_search_command",
    "TAR_PROXYLESS_SEARCH_COMMAND",
)
_HYPSEEK_POLICY_FIELDS = {
    "teacher_source",
    "teacher_version",
    "allow_synthetic",
}


class HypSeekTeacherUnavailableError(RuntimeError):
    pass


class HypSeekTeacherExecutionError(RuntimeError):
    pass


def runtime_status() -> list[dict]:
    return [status.to_dict() for status in _runtime_statuses()]


def _runtime_statuses() -> list[RequirementStatus]:
    statuses: list[RequirementStatus] = []
    if os.environ.get(_TAR_PROXYLESS_SEARCH_COMMAND.env_var, "").strip():
        statuses.append(check_command(_TAR_PROXYLESS_SEARCH_COMMAND))
    return statuses


def hypseek_teacher_response(payload: dict) -> dict[str, object]:
    records, policy = _validated_hypseek_request(payload)
    teacher_source, teacher_version = _configured_hypseek_identity()
    if policy["teacher_source"] != teacher_source:
        raise ValueError(
            "teacher_policy.teacher_source does not match the configured teacher source"
        )
    if policy["teacher_version"] != teacher_version:
        raise ValueError(
            "teacher_policy.teacher_version does not match the configured teacher version"
        )
    command = os.environ.get("HYPSEEK_TEACHER_COMMAND", "").strip()
    if command:
        score = _run_hypseek_teacher_command(
            command,
            records,
            policy,
            _positive_environment_float("HYPSEEK_TEACHER_TIMEOUT_SECONDS"),
        )
        synthetic = False
    else:
        if not policy["allow_synthetic"]:
            raise HypSeekTeacherUnavailableError(
                "HYPSEEK_TEACHER_COMMAND is required when synthetic output is disallowed"
            )
        score = _synthetic_teacher_score(records)
        synthetic = True
    return {
        "teacher_score": score,
        "teacher_source": teacher_source,
        "teacher_version": teacher_version,
        "synthetic": synthetic,
    }


@hypseek_app.post("/teacher")
async def hypseek_teacher_endpoint(payload: dict) -> dict[str, object]:
    try:
        return await asyncio.to_thread(hypseek_teacher_response, payload)
    except HypSeekTeacherUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except HypSeekTeacherExecutionError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@hypseek_app.get("/healthz")
async def hypseek_healthz() -> dict[str, str]:
    try:
        teacher_source, teacher_version = _configured_hypseek_identity()
    except HypSeekTeacherUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {
        "status": "ok",
        "service": "hypseek_teacher",
        "teacher_source": teacher_source,
        "teacher_version": teacher_version,
    }


def _validated_hypseek_request(
    payload: object,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    if not isinstance(payload, Mapping) or set(payload) != {"records", "teacher_policy"}:
        raise ValueError("HypSeek teacher payload must contain exactly records and teacher_policy")
    raw_records = payload["records"]
    if (
        not isinstance(raw_records, list)
        or not raw_records
        or not all(isinstance(record, Mapping) for record in raw_records)
    ):
        raise ValueError("HypSeek teacher records must be a non-empty list of objects")
    records = [dict(record) for record in raw_records]
    try:
        json.dumps(records, allow_nan=False, separators=(",", ":"), sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("HypSeek teacher records must be canonical JSON") from exc
    policy = _validated_hypseek_policy(payload["teacher_policy"])
    return records, policy


def _validated_hypseek_policy(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != _HYPSEEK_POLICY_FIELDS:
        raise ValueError(
            "HypSeek teacher_policy must contain exactly teacher_source, "
            "teacher_version, and allow_synthetic"
        )
    source = value["teacher_source"]
    version = value["teacher_version"]
    allow_synthetic = value["allow_synthetic"]
    if source != "hypseek":
        raise ValueError("HypSeek teacher_policy.teacher_source must be hypseek")
    if not isinstance(version, str) or not version.strip():
        raise ValueError("HypSeek teacher_policy.teacher_version must be a non-empty string")
    if not isinstance(allow_synthetic, bool):
        raise ValueError("HypSeek teacher_policy.allow_synthetic must be a boolean")
    return {
        "teacher_source": source,
        "teacher_version": version.strip(),
        "allow_synthetic": allow_synthetic,
    }


def _configured_hypseek_identity() -> tuple[str, str]:
    source = os.environ.get("HYPSEEK_TEACHER_SOURCE", "").strip()
    if not source:
        raise HypSeekTeacherUnavailableError("HYPSEEK_TEACHER_SOURCE is required")
    if source != "hypseek":
        raise HypSeekTeacherUnavailableError("HYPSEEK_TEACHER_SOURCE must be hypseek")
    version = os.environ.get("HYPSEEK_TEACHER_VERSION", "").strip()
    if not version:
        raise HypSeekTeacherUnavailableError("HYPSEEK_TEACHER_VERSION is required")
    return source, version


def _positive_environment_float(name: str) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        raise HypSeekTeacherUnavailableError(f"{name} is required")
    try:
        value = float(raw)
    except ValueError as exc:
        raise HypSeekTeacherUnavailableError(f"{name} must be a finite positive number") from exc
    if not math.isfinite(value) or value <= 0.0:
        raise HypSeekTeacherUnavailableError(f"{name} must be a finite positive number")
    return value


def _run_hypseek_teacher_command(
    command: str,
    records: list[dict[str, object]],
    policy: dict[str, object],
    timeout_seconds: float,
) -> float:
    try:
        arguments = shlex.split(command)
    except ValueError as exc:
        raise HypSeekTeacherUnavailableError("HYPSEEK_TEACHER_COMMAND is invalid") from exc
    if not arguments:
        raise HypSeekTeacherUnavailableError("HYPSEEK_TEACHER_COMMAND is empty")
    command_payload = {
        "records": records,
        "teacher_policy": policy,
    }
    try:
        encoded_payload = json.dumps(
            command_payload,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        result = subprocess.run(  # noqa: S603
            arguments,
            input=encoded_payload,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
    except FileNotFoundError as exc:
        raise HypSeekTeacherUnavailableError(
            "HYPSEEK_TEACHER_COMMAND executable is unavailable"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise HypSeekTeacherExecutionError("HypSeek teacher command timed out") from exc
    except OSError as exc:
        raise HypSeekTeacherExecutionError(
            f"HypSeek teacher command failed to start: {exc}"
        ) from exc
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        detail = stderr or f"exit code {result.returncode}"
        raise HypSeekTeacherExecutionError(f"HypSeek teacher command failed: {detail}")
    return _teacher_score_from_command_output(result.stdout, len(records))


def _teacher_score_from_command_output(output: bytes, record_count: int) -> float:
    try:
        decoded = output.decode("utf-8")
        payload = json.loads(decoded, parse_constant=_reject_json_constant)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise HypSeekTeacherExecutionError(
            "HypSeek teacher command output must be strict JSON"
        ) from exc
    if not isinstance(payload, Mapping):
        raise HypSeekTeacherExecutionError("HypSeek teacher command output must be a JSON object")
    fields = set(payload)
    if fields == {"teacher_score"}:
        return _normalized_teacher_score(payload["teacher_score"], "teacher_score")
    if fields in ({"teacher_distribution"}, {"distribution"}):
        field = next(iter(fields))
        distribution = payload[field]
        if (
            not isinstance(distribution, list)
            or len(distribution) != record_count
            or not distribution
        ):
            raise HypSeekTeacherExecutionError(
                "HypSeek teacher distribution must contain one score per record"
            )
        scores = [
            _normalized_teacher_score(value, f"{field}[{index}]")
            for index, value in enumerate(distribution)
        ]
        return math.fsum(scores) / len(scores)
    raise HypSeekTeacherExecutionError(
        "HypSeek teacher command output must contain exactly teacher_score or teacher_distribution"
    )


def _normalized_teacher_score(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise HypSeekTeacherExecutionError(f"{field} must be numeric")
    score = float(value)
    if not math.isfinite(score):
        raise HypSeekTeacherExecutionError(f"{field} must be finite")
    if score < 0.0 or score > 1.0:
        raise HypSeekTeacherExecutionError(f"{field} must be in [0, 1]")
    return score


def _synthetic_teacher_score(records: list[dict[str, object]]) -> float:
    scores: list[float] = []
    for index, record in enumerate(records):
        outcome = record.get("outcome")
        if isinstance(outcome, str) and outcome.upper() in {
            "PASS",
            "FAIL",
            "AWAITING_EVIDENCE",
            "ERROR",
        }:
            scores.append(1.0 if outcome.upper() == "PASS" else 0.0)
            continue
        passed = record.get("passed")
        if isinstance(passed, bool):
            scores.append(1.0 if passed else 0.0)
            continue
        verdict = record.get("verdict")
        if isinstance(verdict, str) and verdict.upper() in {"PASS", "FAIL"}:
            scores.append(1.0 if verdict.upper() == "PASS" else 0.0)
            continue
        raise ValueError(f"HypSeek synthetic record {index} requires outcome, passed, or verdict")
    return math.fsum(scores) / len(scores)


def _reject_json_constant(value: str) -> Never:
    raise ValueError(f"invalid JSON numeric constant: {value}")


class GeneratorRouterServicer(router_pb2_grpc.GeneratorRouterServiceServicer):
    def __init__(
        self,
        *,
        state_path: str | Path,
        bootstrap: bool,
    ) -> None:
        self.state_path = Path(state_path)
        self.router = TaskAwareRouter(n_generators=len(GENERATOR_NAMES))
        self.kd_layer = CrossParadigmKDLayer(n_generators=len(GENERATOR_NAMES))
        self.proxyless_search_command = os.getenv(
            "TAR_PROXYLESS_SEARCH_COMMAND",
            "",
        ).strip()
        self._state_version = 0
        self._context_state: dict[str, dict[str, dict[str, float]]] = {}
        self._request_context_map: dict[str, str] = {}
        self._request_route_snapshots: dict[str, dict[str, object]] = {}
        self._feedback_ids: set[str] = set()
        self._feedback_payloads: dict[str, str] = {}
        self._feedback_semantic_payloads: dict[str, str] = {}
        self._bootstrap_metadata = {
            "bootstrapped": True,
            "created_at_ns": time.time_ns(),
        }
        self._lock = asyncio.Lock()

        if self.state_path.exists():
            self._load_state()
        elif bootstrap:
            self._persist_state()
        else:
            raise RuntimeError(
                f"Router state is missing at {self.state_path}; "
                "set TAR_BOOTSTRAP=true to initialize it"
            )

    @property
    def state_version(self) -> int:
        return self._state_version

    @property
    def context_state(self) -> dict[str, dict[str, dict[str, float]]]:
        return {
            key: {name: dict(history) for name, history in histories.items()}
            for key, histories in self._context_state.items()
        }

    @property
    def request_context_map(self) -> dict[str, str]:
        return dict(self._request_context_map)

    @property
    def request_route_snapshots(self) -> dict[str, dict[str, object]]:
        return json.loads(json.dumps(self._request_route_snapshots))

    @property
    def feedback_ids(self) -> set[str]:
        return set(self._feedback_ids)

    async def Route(  # noqa: N802
        self,
        request: router_pb2.RouterRequest,
        context: grpc.aio.ServicerContext | None,
    ) -> router_pb2.RouterResponse:
        try:
            async with self._lock:
                prepared = _prepare_router_request(request, self.router)
                previous = self._state_payload()
                state_changed = False
                try:
                    context_history, binding_changed = self._bind_request_context(prepared)
                    state_changed = state_changed or binding_changed
                    request_id = prepared["request_id"]
                    snapshot = self._request_route_snapshots.get(request_id)
                    if snapshot is None:
                        snapshot = _build_route_snapshot(
                            self.router,
                            prepared,
                            context_history,
                        )
                        self._request_route_snapshots[request_id] = snapshot
                        state_changed = True
                    else:
                        state_changed = (
                            _ensure_snapshot_matches_request(snapshot, prepared) or state_changed
                        )

                    if state_changed:
                        self._state_version += 1
                        self._persist_state()
                except BaseException:
                    if state_changed:
                        self._restore_state_payload(previous)
                    raise

                return _route_response(
                    snapshot,
                    warnings=prepared["warnings"],
                    state_version=self._state_version,
                )
        except (TypeError, ValueError) as exc:
            return await _abort_invalid_argument(context, str(exc))

    async def SubmitFeedback(  # noqa: N802
        self,
        request: router_pb2.RouterFeedbackRequest,
        context: grpc.aio.ServicerContext | None,
    ) -> router_pb2.RouterFeedbackResponse:
        try:
            async with self._lock:
                feedback = _validate_feedback_request(request)
                feedback_fingerprint = _feedback_fingerprint(feedback)
                semantic_key = _feedback_semantic_key(feedback)
                semantic_fingerprint = _feedback_content_fingerprint(feedback)
                context_key = self._request_context_map.get(feedback["request_id"])
                if context_key is None:
                    raise ValueError("feedback request_id has no routed context")
                snapshot = self._request_route_snapshots.get(feedback["request_id"])
                if snapshot is None:
                    raise ValueError("feedback request_id has no routed snapshot")
                snapshot_run_id = snapshot["run_id"]
                if snapshot_run_id is not None and feedback["run_id"] != snapshot_run_id:
                    raise ValueError("feedback run_id does not match routed request")
                if feedback["generator_name"] not in snapshot["selected_generators"]:
                    raise ValueError("feedback generator was not selected for this request_id")
                if feedback["feedback_id"] in self._feedback_ids:
                    recorded_fingerprint = self._feedback_payloads.get(feedback["feedback_id"])
                    if recorded_fingerprint != feedback_fingerprint:
                        raise ValueError(
                            "feedback_id was already submitted with a different payload"
                        )
                    return router_pb2.RouterFeedbackResponse(
                        acknowledged=True,
                        duplicate=True,
                        state_version=self._state_version,
                    )
                recorded_semantic_fingerprint = self._feedback_semantic_payloads.get(semantic_key)
                if recorded_semantic_fingerprint is not None:
                    if recorded_semantic_fingerprint != semantic_fingerprint:
                        raise ValueError(
                            "feedback semantic identity was already submitted "
                            "with different content"
                        )
                    return router_pb2.RouterFeedbackResponse(
                        acknowledged=True,
                        duplicate=True,
                        state_version=self._state_version,
                    )
                if snapshot_run_id is None:
                    raise ValueError("feedback request_id must be routed again to bind run_id")
                context_history = self._context_state.get(context_key)
                if context_history is None:
                    raise RuntimeError("Router context state is missing")

                previous = self._state_payload()
                try:
                    _update_history(
                        context_history,
                        feedback["generator_name"],
                        feedback["teacher_score"],
                    )
                    self.router.update_with_feedback(
                        feedback["generator_name"],
                        feedback["teacher_score"],
                    )
                    generator_index = GENERATOR_NAMES.index(feedback["generator_name"])
                    self.kd_layer.update_teacher_scores(
                        feedback["generator_name"],
                        generator_index,
                        [
                            OracleFeedback(
                                oracle_name=feedback["teacher_source"],
                                normalized_score=feedback["teacher_score"],
                            )
                        ],
                    )
                    self._feedback_ids.add(feedback["feedback_id"])
                    self._feedback_payloads[feedback["feedback_id"]] = feedback_fingerprint
                    self._feedback_semantic_payloads[semantic_key] = semantic_fingerprint
                    self._state_version += 1
                    self._persist_state()
                except BaseException:
                    self._restore_state_payload(previous)
                    raise

                return router_pb2.RouterFeedbackResponse(
                    acknowledged=True,
                    duplicate=False,
                    state_version=self._state_version,
                )
        except (TypeError, ValueError) as exc:
            return await _abort_invalid_argument(context, str(exc))

    async def RunProxylessSearch(  # noqa: N802
        self,
        request: router_pb2.RouterProxylessSearchRequest,
        context: grpc.aio.ServicerContext | None,
    ) -> router_pb2.RouterProxylessSearchResponse:
        try:
            payload = _proxyless_search_payload_from_request(request)

            if self.proxyless_search_command:
                return await _abort_failed_precondition(
                    context,
                    "external Proxyless search cannot atomically update Router state",
                )

            async with self._lock:
                previous = self._state_payload()
                try:
                    scheduler = ProxylessSearchScheduler(
                        router=self.router,
                        generator_costs=payload["generator_costs"],
                        cost_weight=payload["cost_weight"],
                        learning_rate=payload["learning_rate"],
                        temperature=payload["temperature"],
                    )
                    result = scheduler.run(payload["reward_batches_by_dataset"])
                    self._state_version += 1
                    result["state_version"] = self._state_version
                    self._persist_state()
                except BaseException:
                    self._restore_state_payload(previous)
                    raise
            return _proxyless_search_response(result)
        except (TypeError, ValueError) as exc:
            return await _abort_invalid_argument(context, str(exc))

    async def GetWeights(  # noqa: N802
        self,
        request: router_pb2.RouterRequest,
        context: grpc.aio.ServicerContext | None,
    ) -> router_pb2.RouterWeightsResponse:
        try:
            async with self._lock:
                prepared = _prepare_router_request(request, self.router)
                previous = self._state_payload()
                state_changed = False
                try:
                    context_history, state_changed = self._bind_request_context(prepared)
                    snapshot = self._request_route_snapshots.get(prepared["request_id"])
                    if snapshot is None:
                        snapshot = _build_route_snapshot(
                            self.router,
                            prepared,
                            context_history,
                        )
                        self._request_route_snapshots[prepared["request_id"]] = snapshot
                        state_changed = True
                    else:
                        state_changed = (
                            _ensure_snapshot_matches_request(snapshot, prepared) or state_changed
                        )
                    weights = snapshot["eligible_weights"]
                    if state_changed:
                        self._state_version += 1
                        self._persist_state()
                except BaseException:
                    if state_changed:
                        self._restore_state_payload(previous)
                    raise
                return router_pb2.RouterWeightsResponse(
                    generator_names=list(weights),
                    weights=list(weights.values()),
                    state_version=self._state_version,
                )
        except (TypeError, ValueError) as exc:
            return await _abort_invalid_argument(context, str(exc))

    def _bind_request_context(
        self,
        prepared: dict[str, object],
    ) -> tuple[dict[str, dict[str, float]], bool]:
        state_changed = False
        context_key = prepared["context_key"]
        request_id = prepared["request_id"]
        existing_context = self._request_context_map.get(request_id)
        if existing_context is not None and existing_context != context_key:
            raise ValueError("request_id is already bound to a different routing context")
        context_history = self._context_state.get(context_key)
        if context_history is None:
            context_history = _empty_history()
            self._context_state[context_key] = context_history
            state_changed = True
        if existing_context is None:
            self._request_context_map[request_id] = context_key
            state_changed = True
        return context_history, state_changed

    def _state_payload(self) -> dict[str, object]:
        return {
            "schema_version": 3,
            "state_version": self._state_version,
            "generator_names": list(GENERATOR_NAMES),
            "bootstrap_metadata": dict(self._bootstrap_metadata),
            "router": {
                "dimensions": {
                    "hciv_dim": self.router.hciv_dim,
                    "task_dim": self.router.task_dim,
                    "hidden_dim": int(self.router.gen_embeddings.shape[1]),
                    "n_generators": self.router.n_generators,
                },
                "tensors": _encode_state_dict(self.router.state_dict()),
                "oracle_history": {
                    name: dict(self.router.oracle_history[name]) for name in GENERATOR_NAMES
                },
            },
            "kd": {
                "dimensions": {
                    "n_generators": self.kd_layer.n_generators,
                    "mode": self.kd_layer.mode,
                },
                "tensors": _encode_state_dict(self.kd_layer.state_dict()),
                "quality_scores": list(self.kd_layer._quality_scores),
                "teacher_embedding_targets": {
                    str(index): _encode_tensor(tensor)
                    for index, tensor in sorted(self.kd_layer._teacher_embedding_targets.items())
                },
            },
            "context_state": self.context_state,
            "request_context_map": self.request_context_map,
            "request_route_snapshots": self.request_route_snapshots,
            "feedback_ids": sorted(self._feedback_ids),
            "feedback_payloads": {
                feedback_id: self._feedback_payloads[feedback_id]
                for feedback_id in sorted(self._feedback_payloads)
            },
            "feedback_semantic_payloads": {
                semantic_key: self._feedback_semantic_payloads[semantic_key]
                for semantic_key in sorted(self._feedback_semantic_payloads)
            },
        }

    def _load_state(self) -> None:
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
            self._restore_state_payload(payload)
        except Exception as exc:
            raise RuntimeError(f"Router state is invalid: {exc}") from exc

    def _restore_state_payload(self, payload: object) -> None:
        clean = _validate_state_payload(
            payload,
            router=self.router,
            kd_layer=self.kd_layer,
        )
        self.router.load_state_dict(clean["router_tensors"], strict=True)
        self.router.oracle_history = clean["oracle_history"]
        self.kd_layer.load_state_dict(clean["kd_tensors"], strict=True)
        self.kd_layer._quality_scores = clean["quality_scores"]
        self.kd_layer._teacher_embedding_targets = clean["teacher_embedding_targets"]
        self._state_version = clean["state_version"]
        self._bootstrap_metadata = clean["bootstrap_metadata"]
        self._context_state = clean["context_state"]
        self._request_context_map = clean["request_context_map"]
        self._request_route_snapshots = clean["request_route_snapshots"]
        self._feedback_ids = clean["feedback_ids"]
        self._feedback_payloads = clean["feedback_payloads"]
        self._feedback_semantic_payloads = clean["feedback_semantic_payloads"]

    def _persist_state(self) -> None:
        payload = json.dumps(
            self._state_payload(),
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        _atomic_write(self.state_path, payload)


async def _abort_invalid_argument(
    context: grpc.aio.ServicerContext | None,
    message: str,
) -> Never:
    if context is not None and hasattr(context, "abort"):
        await context.abort(grpc.StatusCode.INVALID_ARGUMENT, message)
    raise ValueError(message)


async def _abort_failed_precondition(
    context: grpc.aio.ServicerContext | None,
    message: str,
) -> Never:
    if context is not None and hasattr(context, "abort"):
        await context.abort(grpc.StatusCode.FAILED_PRECONDITION, message)
    raise RuntimeError(message)


def _prepare_router_request(
    request: router_pb2.RouterRequest,
    router: TaskAwareRouter,
) -> dict[str, object]:
    warnings = _validate_deprecated_performance(request)
    project_id = str(getattr(request, "project_id", "") or "")
    if not project_id:
        raise ValueError("project_id is required")
    raw_run_id = getattr(request, "run_id", "")
    if raw_run_id is None or raw_run_id == "":
        run_id = None
    elif not isinstance(raw_run_id, str) or raw_run_id != raw_run_id.strip():
        raise ValueError("run_id must be a trimmed string when provided")
    else:
        run_id = raw_run_id
    cig = _cig_from_request(request, project_id)
    profile = _profile_from_request(request)
    hciv = _hciv_from_request(request, router.hciv_dim)
    eligible_names = _eligible_generator_names(request)
    n_select = int(getattr(request, "n_select", 0))
    n_samples = int(getattr(request, "n_samples", 0))
    if n_select <= 0:
        raise ValueError("n_select must be positive")
    if n_select > len(eligible_names):
        raise ValueError("n_select exceeds eligible generators")
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if n_samples < n_select:
        raise ValueError("n_samples must be at least n_select")

    request_id = str(getattr(request, "request_id", "") or "")
    if not request_id:
        raise ValueError("request_id is required")
    context_payload = {
        "project_id": project_id,
        "cig_sha256": hashlib.sha256(cig.SerializeToString(deterministic=True)).hexdigest(),
        "hciv": [float(value) for value in hciv.tolist()],
        "profile": {
            "target_family": profile.target_family,
            "stage": profile.stage,
            "data_richness": profile.data_richness,
            "novelty_demand": profile.novelty_demand,
            "multi_target": profile.multi_target,
            "sa_constraint": profile.sa_constraint,
            "n_samples": profile.n_samples,
            "prior_weights": {
                name: profile.prior_weights[name]
                for name in GENERATOR_NAMES
                if name in profile.prior_weights
            },
        },
        "eligible_generator_names": eligible_names,
        "task_complexity": int(getattr(request, "task_complexity", 0)),
        "n_select": n_select,
    }
    context_json = json.dumps(
        context_payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return {
        "run_id": run_id,
        "request_id": request_id,
        "profile": profile,
        "hciv": hciv,
        "eligible_names": eligible_names,
        "n_select": n_select,
        "n_samples": n_samples,
        "warnings": warnings,
        "context_key": hashlib.sha256(context_json.encode("utf-8")).hexdigest(),
    }


def _cig_from_request(
    request: router_pb2.RouterRequest,
    project_id: str,
) -> cig_pb2.CIG:
    payload = bytes(getattr(request, "cig", b"") or b"")
    if not payload:
        raise ValueError("cig is required")
    message = cig_pb2.CIG()
    try:
        message.ParseFromString(payload)
    except DecodeError as exc:
        raise ValueError("cig must be a serialized CIG") from exc
    if message.project_id != project_id:
        raise ValueError("CIG project_id must match project_id")
    if not message.objectives:
        raise ValueError("CIG objectives must be non-empty")
    return message


def _profile_from_request(request: router_pb2.RouterRequest) -> TaskProfile:
    request_weights = [
        float(value) for value in list(getattr(request, "generator_weights", []) or [])
    ]
    if request_weights and len(request_weights) != len(GENERATOR_NAMES):
        raise ValueError(f"generator_weights must contain {len(GENERATOR_NAMES)} values")
    if any(not math.isfinite(value) for value in request_weights):
        raise ValueError("generator_weights values must be finite")
    if any(value < 0.0 for value in request_weights):
        raise ValueError("generator_weights values must be non-negative")
    prior_weights = {
        name: request_weights[index]
        for index, name in enumerate(GENERATOR_NAMES)
        if request_weights
    }
    data_richness = float(getattr(request, "data_richness", 0.0))
    novelty_demand = float(getattr(request, "novelty_demand", 0.0))
    sa_constraint = float(getattr(request, "sa_constraint", 0.0))
    if not all(math.isfinite(value) for value in (data_richness, novelty_demand, sa_constraint)):
        raise ValueError("task profile numeric values must be finite")
    if data_richness < 0.0:
        raise ValueError("data_richness must be non-negative")
    if novelty_demand < 0.0 or novelty_demand > 1.0:
        raise ValueError("novelty_demand must be in [0, 1]")
    if sa_constraint < 0.0:
        raise ValueError("sa_constraint must be non-negative")
    return TaskProfile(
        target_family=str(getattr(request, "target_family", "")),
        stage=str(getattr(request, "stage", "hit_finding") or "hit_finding"),
        data_richness=data_richness,
        novelty_demand=novelty_demand,
        multi_target=bool(getattr(request, "multi_target", False)),
        sa_constraint=sa_constraint,
        n_samples=int(getattr(request, "n_samples", 0)),
        prior_weights=prior_weights,
    )


def _hciv_from_request(
    request: router_pb2.RouterRequest,
    hciv_dim: int,
) -> torch.Tensor:
    values = [float(item) for item in getattr(request, "hciv", []) or []]
    if any(not math.isfinite(value) for value in values):
        raise ValueError("hciv values must be finite")
    if len(values) != hciv_dim + 1:
        raise ValueError(f"hciv must contain exactly {hciv_dim + 1} values")
    normalized = normalize_lorentz_embedding(
        values,
        expected_dim=hciv_dim + 1,
    )
    if normalized is None:
        raise ValueError("hciv must be a valid Lorentz point")
    return torch.tensor(normalized[1:], dtype=torch.float32)


def _validate_deprecated_performance(
    request: router_pb2.RouterRequest,
) -> list[str]:
    performance = [
        float(value) for value in list(getattr(request, "generator_performance", []) or [])
    ]
    if not performance:
        return []
    if len(performance) != len(GENERATOR_NAMES):
        raise ValueError(f"generator_performance must contain {len(GENERATOR_NAMES)} values")
    if any(not math.isfinite(value) for value in performance):
        raise ValueError("generator_performance values must be finite")
    return ["generator_performance is deprecated and ignored"]


def _eligible_generator_names(request: router_pb2.RouterRequest) -> list[str]:
    requested = [
        str(name) for name in list(getattr(request, "available_generator_names", []) or [])
    ]
    if len(requested) != len(set(requested)):
        raise ValueError("available generator names must be unique")
    unknown = [name for name in requested if name not in GENERATOR_NAMES]
    if unknown:
        raise ValueError(f"unknown available generator: {unknown[0]}")
    available = set(requested or GENERATOR_NAMES)

    complexity = int(getattr(request, "task_complexity", 0))
    complexity_names = {
        router_pb2.TASK_COMPLEXITY_UNSPECIFIED: set(GENERATOR_NAMES),
        router_pb2.TASK_COMPLEXITY_LOW: {"hfm_3d"},
        router_pb2.TASK_COMPLEXITY_MEDIUM: {"hfm_3d", "fragfm"},
        router_pb2.TASK_COMPLEXITY_HIGH: {"mmpt_rag", "fragfm"},
    }
    if complexity not in complexity_names:
        raise ValueError("task_complexity is invalid")
    eligible = [
        name
        for name in GENERATOR_NAMES
        if name in available and name in complexity_names[complexity]
    ]
    if not eligible:
        raise ValueError("no eligible generators")
    return eligible


def _context_weights(
    router: TaskAwareRouter,
    prepared: dict[str, object],
    context_history: dict[str, dict[str, float]],
) -> dict[str, float]:
    raw_weights = router.forward(
        prepared["hciv"],
        prepared["profile"],
        oracle_history=context_history,
    )
    eligible_names = prepared["eligible_names"]
    clean: dict[str, float] = {}
    for name in eligible_names:
        value = float(raw_weights[name])
        if not math.isfinite(value):
            raise ValueError("Router weights must be finite")
        if value < 0.0:
            raise ValueError("Router weights must be non-negative")
        clean[name] = value
    total = sum(clean.values())
    if total == 0.0:
        return {name: 1.0 / len(eligible_names) for name in eligible_names}
    return {name: clean[name] / total for name in eligible_names}


def _top_k_weights(
    weights: dict[str, float],
    n_select: int,
) -> dict[str, float]:
    canonical_index = {name: index for index, name in enumerate(GENERATOR_NAMES)}
    selected_names = sorted(
        weights,
        key=lambda name: (-weights[name], canonical_index[name]),
    )[:n_select]
    return {name: weights[name] for name in selected_names}


def _build_route_snapshot(
    router: TaskAwareRouter,
    prepared: dict[str, object],
    context_history: dict[str, dict[str, float]],
) -> dict[str, object]:
    weights = _context_weights(router, prepared, context_history)
    selected = _top_k_weights(weights, prepared["n_select"])
    selected_weight_sum = sum(selected.values())
    normalized_weights = {name: weight / selected_weight_sum for name, weight in selected.items()}
    allocations = minimum_one_largest_remainder(
        normalized_weights,
        prepared["n_samples"],
    )
    return {
        "run_id": prepared["run_id"],
        "context_key": prepared["context_key"],
        "n_samples": prepared["n_samples"],
        "n_select": prepared["n_select"],
        "eligible_generator_names": list(weights),
        "eligible_weights": dict(weights),
        "selected_generators": list(selected),
        "normalized_weights": normalized_weights,
        "expected_rewards": {name: weights[name] for name in selected},
        "allocations": allocations,
    }


def _ensure_snapshot_matches_request(
    snapshot: dict[str, object],
    prepared: dict[str, object],
) -> bool:
    _validate_route_snapshot(
        snapshot,
        source=f"request_route_snapshots.{prepared['request_id']}",
        expected_context_key=prepared["context_key"],
    )
    if (
        snapshot["n_samples"] != prepared["n_samples"]
        or snapshot["n_select"] != prepared["n_select"]
    ):
        raise ValueError("request_id is already bound to a different routing snapshot")
    if snapshot["run_id"] is None and prepared["run_id"] is not None:
        snapshot["run_id"] = prepared["run_id"]
        run_id_bound = True
    elif prepared["run_id"] is not None and snapshot["run_id"] != prepared["run_id"]:
        raise ValueError("request_id is already bound to a different run_id")
    else:
        run_id_bound = False
    return run_id_bound


def _route_response(
    snapshot: dict[str, object],
    *,
    warnings: list[str],
    state_version: int,
) -> router_pb2.RouterResponse:
    selected = snapshot["selected_generators"]
    normalized_weights = snapshot["normalized_weights"]
    expected_rewards = snapshot["expected_rewards"]
    allocations = snapshot["allocations"]
    return router_pb2.RouterResponse(
        selected_generators=selected,
        selection_weights=[normalized_weights[name] for name in selected],
        strategy="task_aware_router",
        expected_rewards=[expected_rewards[name] for name in selected],
        allocations=[
            router_pb2.GeneratorAllocation(
                generator_name=name,
                n_samples=allocations[name],
                normalized_weight=normalized_weights[name],
                expected_reward=expected_rewards[name],
            )
            for name in selected
        ],
        warnings=warnings,
        state_version=state_version,
    )


def _empty_history() -> dict[str, dict[str, float]]:
    return {name: {"avg_hvi": 0.0, "n_calls": 0.0} for name in GENERATOR_NAMES}


def _update_history(
    history: dict[str, dict[str, float]],
    generator_name: str,
    reward: float,
) -> None:
    record = history[generator_name]
    count = float(record["n_calls"])
    record["avg_hvi"] = float(record["avg_hvi"]) + (reward - float(record["avg_hvi"])) / (
        count + 1.0
    )
    record["n_calls"] = count + 1.0


def _validate_feedback_request(
    request: router_pb2.RouterFeedbackRequest,
) -> dict[str, object]:
    feedback_id = str(getattr(request, "feedback_id", "") or "")
    run_id = str(getattr(request, "run_id", "") or "")
    request_id = str(getattr(request, "request_id", "") or "")
    generator_name = str(getattr(request, "generator_name", "") or "")
    canonical_smiles = str(getattr(request, "canonical_smiles", "") or "")
    teacher_source = str(getattr(request, "teacher_source", "") or "")
    teacher_version = str(getattr(request, "teacher_version", "") or "")
    candidate_ids = [str(value) for value in getattr(request, "candidate_ids", [])]
    evidence_ids = [str(value) for value in getattr(request, "evidence_ids", [])]
    phase = int(getattr(request, "phase", 0))

    if not feedback_id:
        raise ValueError("feedback_id is required")
    if not run_id:
        raise ValueError("run_id is required")
    if not request_id:
        raise ValueError("request_id is required")
    if generator_name not in GENERATOR_NAMES:
        raise ValueError("generator_name is invalid")
    if phase not in {
        router_pb2.ROUTER_FEEDBACK_PHASE_VALIDATION,
        router_pb2.ROUTER_FEEDBACK_PHASE_CRITIC,
    }:
        raise ValueError("feedback phase is invalid")
    if not candidate_ids or any(not value for value in candidate_ids):
        raise ValueError("candidate_ids must be non-empty")
    if not canonical_smiles:
        raise ValueError("canonical_smiles is required")
    if not evidence_ids or any(not value for value in evidence_ids):
        raise ValueError("evidence_ids must be non-empty")
    if not teacher_source:
        raise ValueError("teacher_source is required")
    if not teacher_version:
        raise ValueError("teacher_version is required")
    if not callable(getattr(request, "HasField", None)) or not request.HasField("teacher_score"):
        raise ValueError("teacher_score is required")
    teacher_score = float(request.teacher_score)
    if not math.isfinite(teacher_score):
        raise ValueError("teacher_score must be finite")
    if teacher_score < 0.0 or teacher_score > 1.0:
        raise ValueError("teacher_score must be in [0, 1]")
    return {
        "feedback_id": feedback_id,
        "run_id": run_id,
        "request_id": request_id,
        "iteration": int(request.iteration),
        "phase": phase,
        "generator_name": generator_name,
        "candidate_ids": candidate_ids,
        "canonical_smiles": canonical_smiles,
        "evidence_ids": evidence_ids,
        "teacher_score": teacher_score,
        "teacher_source": teacher_source,
        "teacher_version": teacher_version,
        "synthetic": bool(request.synthetic),
    }


def _feedback_fingerprint(feedback: dict[str, object]) -> str:
    return _canonical_feedback_hash(feedback)


def _feedback_semantic_key(feedback: dict[str, object]) -> str:
    return _canonical_feedback_hash(
        {
            "run_id": feedback["run_id"],
            "request_id": feedback["request_id"],
            "iteration": feedback["iteration"],
            "phase": feedback["phase"],
            "generator_name": feedback["generator_name"],
            "canonical_smiles": feedback["canonical_smiles"],
        }
    )


def _feedback_content_fingerprint(feedback: dict[str, object]) -> str:
    return _canonical_feedback_hash(
        {key: value for key, value in feedback.items() if key != "feedback_id"}
    )


def _canonical_feedback_hash(payload: dict[str, object]) -> str:
    payload = json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _encode_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, dict]:
    return {name: _encode_tensor(tensor) for name, tensor in state_dict.items()}


def _encode_tensor(tensor: torch.Tensor) -> dict[str, object]:
    value = tensor.detach().cpu()
    return {
        "dtype": str(value.dtype).removeprefix("torch."),
        "shape": list(value.shape),
        "values": value.tolist(),
    }


def _decode_state_dict(
    payload: object,
    expected_state: dict[str, torch.Tensor],
    *,
    source: str,
) -> dict[str, torch.Tensor]:
    if not isinstance(payload, dict):
        raise ValueError(f"{source} tensors must be an object")
    if set(payload) != set(expected_state):
        raise ValueError(f"{source} tensor keys do not match")
    return {
        name: _decode_tensor(
            payload[name],
            expected=expected,
            source=f"{source}.{name}",
        )
        for name, expected in expected_state.items()
    }


def _decode_tensor(
    payload: object,
    *,
    expected: torch.Tensor | None = None,
    source: str,
) -> torch.Tensor:
    if not isinstance(payload, dict) or set(payload) != {
        "dtype",
        "shape",
        "values",
    }:
        raise ValueError(f"{source} tensor payload is invalid")
    dtype_name = str(payload["dtype"])
    dtype = getattr(torch, dtype_name, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"{source} tensor dtype is invalid")
    shape = payload["shape"]
    if not isinstance(shape, list) or any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in shape
    ):
        raise ValueError(f"{source} tensor shape is invalid")
    if expected is not None:
        expected_dtype = str(expected.dtype).removeprefix("torch.")
        if dtype_name != expected_dtype:
            raise ValueError(f"{source} tensor dtype does not match")
        if shape != list(expected.shape):
            raise ValueError(f"{source} tensor shape does not match")
    _validate_tensor_values(payload["values"], source=source)
    tensor = torch.tensor(payload["values"], dtype=dtype)
    if list(tensor.shape) != shape:
        raise ValueError(f"{source} tensor values do not match shape")
    if tensor.is_floating_point() and not torch.isfinite(tensor).all():
        raise ValueError(f"{source} tensor values must be finite")
    return tensor


def _validate_state_payload(
    payload: object,
    *,
    router: TaskAwareRouter,
    kd_layer: CrossParadigmKDLayer,
) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise ValueError(
            "Router state lacks the complete replay contract; re-bootstrap or migrate state"
        )
    schema_version = payload.get("schema_version")
    if (
        not isinstance(schema_version, int)
        or isinstance(schema_version, bool)
        or schema_version not in {2, 3}
    ):
        raise ValueError(
            "Router state schema_version is unsupported; re-bootstrap or migrate state"
        )
    expected_keys = {
        "schema_version",
        "state_version",
        "generator_names",
        "bootstrap_metadata",
        "router",
        "kd",
        "context_state",
        "request_context_map",
        "feedback_ids",
    }
    expected_keys.update({"request_route_snapshots", "feedback_payloads"})
    if schema_version >= 3:
        expected_keys.add("feedback_semantic_payloads")
    if set(payload) != expected_keys:
        raise ValueError(
            "Router state lacks the complete replay contract; re-bootstrap or migrate state"
        )
    if payload["generator_names"] != list(GENERATOR_NAMES):
        raise ValueError("Router state generator order does not match")
    state_version = payload["state_version"]
    if not isinstance(state_version, int) or isinstance(state_version, bool) or state_version < 0:
        raise ValueError("Router state state_version is invalid")

    bootstrap_metadata = payload["bootstrap_metadata"]
    if (
        not isinstance(bootstrap_metadata, dict)
        or set(bootstrap_metadata) != {"bootstrapped", "created_at_ns"}
        or bootstrap_metadata["bootstrapped"] is not True
        or not isinstance(bootstrap_metadata["created_at_ns"], int)
        or isinstance(bootstrap_metadata["created_at_ns"], bool)
        or bootstrap_metadata["created_at_ns"] <= 0
    ):
        raise ValueError("Router state bootstrap metadata is invalid")

    router_payload = payload["router"]
    if not isinstance(router_payload, dict) or set(router_payload) != {
        "dimensions",
        "tensors",
        "oracle_history",
    }:
        raise ValueError("Router state router payload is invalid")
    expected_router_dimensions = {
        "hciv_dim": router.hciv_dim,
        "task_dim": router.task_dim,
        "hidden_dim": int(router.gen_embeddings.shape[1]),
        "n_generators": router.n_generators,
    }
    if not _strict_integer_mapping_matches(
        router_payload["dimensions"],
        expected_router_dimensions,
    ):
        raise ValueError("Router state router dimensions do not match")
    router_tensors = _decode_state_dict(
        router_payload["tensors"],
        router.state_dict(),
        source="router",
    )
    oracle_history = _validate_history(
        router_payload["oracle_history"],
        source="oracle_history",
    )

    kd_payload = payload["kd"]
    if not isinstance(kd_payload, dict) or set(kd_payload) != {
        "dimensions",
        "tensors",
        "quality_scores",
        "teacher_embedding_targets",
    }:
        raise ValueError("Router state KD payload is invalid")
    kd_dimensions = kd_payload["dimensions"]
    if (
        not isinstance(kd_dimensions, dict)
        or set(kd_dimensions) != {"n_generators", "mode"}
        or not isinstance(kd_dimensions["n_generators"], int)
        or isinstance(kd_dimensions["n_generators"], bool)
        or kd_dimensions["n_generators"] != kd_layer.n_generators
        or kd_dimensions["mode"] != kd_layer.mode
    ):
        raise ValueError("Router state KD dimensions do not match")
    kd_tensors = _decode_state_dict(
        kd_payload["tensors"],
        kd_layer.state_dict(),
        source="kd",
    )
    quality_scores = kd_payload["quality_scores"]
    if not isinstance(quality_scores, list) or len(quality_scores) != len(GENERATOR_NAMES):
        raise ValueError("Router state KD quality scores are invalid")
    if any(not _is_number(value) for value in quality_scores):
        raise ValueError("Router state KD quality scores are invalid")
    quality_scores = [float(value) for value in quality_scores]
    if any(not math.isfinite(value) for value in quality_scores):
        raise ValueError("Router state KD quality scores must be finite")

    targets_payload = kd_payload["teacher_embedding_targets"]
    if not isinstance(targets_payload, dict):
        raise ValueError("Router state KD teacher targets are invalid")
    teacher_embedding_targets: dict[int, torch.Tensor] = {}
    for raw_index, tensor_payload in targets_payload.items():
        try:
            index = int(raw_index)
        except (TypeError, ValueError) as exc:
            raise ValueError("Router state KD target index is invalid") from exc
        if str(index) != raw_index or index < 0 or index >= len(GENERATOR_NAMES):
            raise ValueError("Router state KD target index is invalid")
        target = _decode_tensor(
            tensor_payload,
            source=f"kd.teacher_embedding_targets.{index}",
        )
        if not target.is_floating_point() or target.numel() == 0:
            raise ValueError("Router state KD teacher target is invalid")
        teacher_embedding_targets[index] = target

    context_payload = payload["context_state"]
    if not isinstance(context_payload, dict):
        raise ValueError("Router state context_state is invalid")
    context_state: dict[str, dict[str, dict[str, float]]] = {}
    for key, history in context_payload.items():
        if (
            not isinstance(key, str)
            or len(key) != 64
            or any(character not in "0123456789abcdef" for character in key)
        ):
            raise ValueError("Router state context key is invalid")
        context_state[key] = _validate_history(
            history,
            source=f"context_state.{key}",
        )

    request_map_payload = payload["request_context_map"]
    if not isinstance(request_map_payload, dict):
        raise ValueError("Router state request_context_map is invalid")
    request_context_map: dict[str, str] = {}
    for request_id, context_key in request_map_payload.items():
        if (
            not isinstance(request_id, str)
            or not request_id
            or not isinstance(context_key, str)
            or context_key not in context_state
        ):
            raise ValueError("Router state request context entry is invalid")
        request_context_map[request_id] = context_key

    snapshots_payload = payload["request_route_snapshots"]
    if not isinstance(snapshots_payload, dict):
        raise ValueError("Router state request_route_snapshots is invalid")
    if set(snapshots_payload) != set(request_context_map):
        raise ValueError("Router state snapshot bindings are incomplete")
    request_route_snapshots: dict[str, dict[str, object]] = {}
    for request_id, snapshot in snapshots_payload.items():
        request_route_snapshots[request_id] = _validate_route_snapshot(
            snapshot,
            source=f"request_route_snapshots.{request_id}",
            expected_context_key=request_context_map[request_id],
            allow_missing_run_id=True,
        )

    feedback_payload = payload["feedback_ids"]
    if (
        not isinstance(feedback_payload, list)
        or any(not isinstance(value, str) or not value for value in feedback_payload)
        or len(feedback_payload) != len(set(feedback_payload))
        or feedback_payload != sorted(feedback_payload)
    ):
        raise ValueError("Router state feedback_ids are invalid")
    feedback_payloads_payload = payload["feedback_payloads"]
    if (
        not isinstance(feedback_payloads_payload, dict)
        or set(feedback_payloads_payload) != set(feedback_payload)
        or any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in feedback_payloads_payload.values()
        )
    ):
        raise ValueError("Router state feedback_payloads are invalid")
    feedback_payloads = {
        feedback_id: feedback_payloads_payload[feedback_id] for feedback_id in feedback_payload
    }
    if schema_version == 2 and feedback_payload:
        raise ValueError(
            "Router state schema_version 2 with feedback cannot be migrated safely; "
            "re-bootstrap or perform an offline migration with the original feedback payloads"
        )
    feedback_semantic_payloads: dict[str, str] = {}
    if schema_version >= 3:
        semantic_payloads = payload["feedback_semantic_payloads"]
        if (
            not isinstance(semantic_payloads, dict)
            or len(semantic_payloads) > len(feedback_payload)
            or any(
                not isinstance(key, str)
                or len(key) != 64
                or any(character not in "0123456789abcdef" for character in key)
                or not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
                for key, value in semantic_payloads.items()
            )
        ):
            raise ValueError("Router state feedback_semantic_payloads are invalid")
        feedback_semantic_payloads = dict(semantic_payloads)

    return {
        "state_version": state_version,
        "bootstrap_metadata": dict(bootstrap_metadata),
        "router_tensors": router_tensors,
        "oracle_history": oracle_history,
        "kd_tensors": kd_tensors,
        "quality_scores": quality_scores,
        "teacher_embedding_targets": teacher_embedding_targets,
        "context_state": context_state,
        "request_context_map": request_context_map,
        "request_route_snapshots": request_route_snapshots,
        "feedback_ids": set(feedback_payload),
        "feedback_payloads": feedback_payloads,
        "feedback_semantic_payloads": feedback_semantic_payloads,
    }


def _validate_history(
    payload: object,
    *,
    source: str,
) -> dict[str, dict[str, float]]:
    if not isinstance(payload, dict) or set(payload) != set(GENERATOR_NAMES):
        raise ValueError(f"Router state {source} generator order does not match")
    clean: dict[str, dict[str, float]] = {}
    for name in GENERATOR_NAMES:
        record = payload[name]
        if not isinstance(record, dict) or set(record) != {"avg_hvi", "n_calls"}:
            raise ValueError(f"Router state {source}.{name} is invalid")
        if not _is_number(record["avg_hvi"]) or not _is_number(record["n_calls"]):
            raise ValueError(f"Router state {source}.{name} must be finite")
        average = float(record["avg_hvi"])
        count = float(record["n_calls"])
        if not math.isfinite(average) or not math.isfinite(count) or count < 0.0:
            raise ValueError(f"Router state {source}.{name} must be finite")
        clean[name] = {"avg_hvi": average, "n_calls": count}
    return clean


def _validate_tensor_values(payload: object, *, source: str) -> None:
    if isinstance(payload, list):
        for value in payload:
            _validate_tensor_values(value, source=source)
        return
    if not _is_number(payload):
        raise ValueError(f"{source} tensor values must be numeric")
    if isinstance(payload, float) and not math.isfinite(payload):
        raise ValueError(f"{source} tensor values must be finite")


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _strict_integer_mapping_matches(
    payload: object,
    expected: dict[str, int],
) -> bool:
    return (
        isinstance(payload, dict)
        and set(payload) == set(expected)
        and all(
            isinstance(payload[name], int)
            and not isinstance(payload[name], bool)
            and payload[name] == value
            for name, value in expected.items()
        )
    )


def _validate_route_snapshot(
    payload: object,
    *,
    source: str,
    expected_context_key: str,
    allow_missing_run_id: bool = False,
) -> dict[str, object]:
    expected_keys = {
        "run_id",
        "context_key",
        "n_samples",
        "n_select",
        "eligible_generator_names",
        "eligible_weights",
        "selected_generators",
        "normalized_weights",
        "expected_rewards",
        "allocations",
    }
    legacy_keys = expected_keys - {"run_id"}
    if not isinstance(payload, dict) or (
        set(payload) != expected_keys and not (allow_missing_run_id and set(payload) == legacy_keys)
    ):
        raise ValueError(f"Router state {source} is invalid")
    payload = dict(payload)
    if "run_id" not in payload:
        payload["run_id"] = None
    run_id = payload["run_id"]
    if run_id is not None and (
        not isinstance(run_id, str) or not run_id or run_id != run_id.strip()
    ):
        raise ValueError(f"Router state {source} run_id is invalid")
    context_key = payload["context_key"]
    if context_key != expected_context_key:
        raise ValueError(f"Router state {source} context_key is invalid")
    n_samples = payload["n_samples"]
    n_select = payload["n_select"]
    if (
        not isinstance(n_samples, int)
        or isinstance(n_samples, bool)
        or n_samples <= 0
        or not isinstance(n_select, int)
        or isinstance(n_select, bool)
        or n_select <= 0
        or n_select > n_samples
    ):
        raise ValueError(f"Router state {source} request counts are invalid")
    eligible = payload["eligible_generator_names"]
    selected = payload["selected_generators"]
    if (
        not isinstance(eligible, list)
        or not eligible
        or eligible != [name for name in GENERATOR_NAMES if name in set(eligible)]
        or len(eligible) != len(set(eligible))
    ):
        raise ValueError(f"Router state {source} eligible generators are invalid")
    if (
        not isinstance(selected, list)
        or not selected
        or len(selected) != n_select
        or len(selected) != len(set(selected))
        or any(name not in eligible for name in selected)
    ):
        raise ValueError(f"Router state {source} selected generators are invalid")
    eligible_weights = _validate_snapshot_float_mapping(
        payload["eligible_weights"],
        names=eligible,
        source=f"{source}.eligible_weights",
    )
    normalized_weights = _validate_snapshot_float_mapping(
        payload["normalized_weights"],
        names=selected,
        source=f"{source}.normalized_weights",
    )
    expected_rewards = _validate_snapshot_float_mapping(
        payload["expected_rewards"],
        names=selected,
        source=f"{source}.expected_rewards",
    )
    if not math.isclose(sum(eligible_weights.values()), 1.0, abs_tol=1e-6):
        raise ValueError(f"Router state {source} eligible weights are invalid")
    if not math.isclose(sum(normalized_weights.values()), 1.0, abs_tol=1e-6):
        raise ValueError(f"Router state {source} normalized weights are invalid")
    selected_weight_sum = sum(eligible_weights[name] for name in selected)
    if selected_weight_sum <= 0.0 or any(
        not math.isclose(
            normalized_weights[name],
            eligible_weights[name] / selected_weight_sum,
            rel_tol=0.0,
            abs_tol=1e-6,
        )
        or not math.isclose(
            expected_rewards[name],
            eligible_weights[name],
            rel_tol=0.0,
            abs_tol=1e-6,
        )
        for name in selected
    ):
        raise ValueError(f"Router state {source} selected weights are inconsistent")
    allocations_payload = payload["allocations"]
    if (
        not isinstance(allocations_payload, dict)
        or set(allocations_payload) != set(selected)
        or any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in allocations_payload.values()
        )
    ):
        raise ValueError(f"Router state {source} allocations are invalid")
    allocations = {name: allocations_payload[name] for name in selected}
    if sum(allocations.values()) != n_samples or allocations != minimum_one_largest_remainder(
        normalized_weights,
        n_samples,
    ):
        raise ValueError(f"Router state {source} snapshot allocation total is invalid")
    return {
        "run_id": run_id,
        "context_key": context_key,
        "n_samples": n_samples,
        "n_select": n_select,
        "eligible_generator_names": list(eligible),
        "eligible_weights": eligible_weights,
        "selected_generators": list(selected),
        "normalized_weights": normalized_weights,
        "expected_rewards": expected_rewards,
        "allocations": allocations,
    }


def _validate_snapshot_float_mapping(
    payload: object,
    *,
    names: list[str],
    source: str,
) -> dict[str, float]:
    if (
        not isinstance(payload, dict)
        or set(payload) != set(names)
        or any(not _is_number(value) for value in payload.values())
    ):
        raise ValueError(f"Router state {source} is invalid")
    clean = {name: float(payload[name]) for name in names}
    if any(not math.isfinite(value) or value < 0.0 for value in clean.values()):
        raise ValueError(f"Router state {source} is invalid")
    return clean


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        try:
            directory_descriptor = os.open(path.parent, directory_flags)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        except OSError as exc:
            logger.warning(
                "directory fsync failed after atomic replace; "
                "state is logically committed but crash durability is uncertain: %s",
                exc,
            )
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _proxyless_search_payload_from_request(
    request: router_pb2.RouterProxylessSearchRequest,
) -> dict:
    generator_costs = _parse_json_object(
        getattr(request, "generator_costs_json", ""),
        "generator_costs_json",
    )
    clean_costs: dict[str, float] = {}
    for name, raw_cost in generator_costs.items():
        if isinstance(raw_cost, bool):
            raise ValueError("generator_costs_json values must be finite and non-negative")
        cost = float(raw_cost)
        if not math.isfinite(cost) or cost < 0.0:
            raise ValueError("generator_costs_json values must be finite and non-negative")
        clean_costs[str(name)] = cost
    cost_weight = float(getattr(request, "cost_weight", 0.0))
    learning_rate = float(getattr(request, "learning_rate", 0.0))
    temperature = float(getattr(request, "temperature", 0.0))
    if not math.isfinite(cost_weight) or cost_weight < 0.0:
        raise ValueError("cost_weight must be finite and non-negative")
    if not math.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError("learning_rate must be finite and positive")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    return {
        "reward_batches_by_dataset": _parse_json_object(
            getattr(request, "reward_batches_json", ""),
            "reward_batches_json",
        ),
        "generator_costs": clean_costs,
        "cost_weight": cost_weight,
        "learning_rate": learning_rate,
        "temperature": temperature,
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


def _validate_proxyless_search_result(result: dict, source: str) -> None:
    if not isinstance(result.get("rounds"), list):
        raise RuntimeError(f"{source} result must contain rounds list")
    if not isinstance(result.get("architecture_probabilities"), dict):
        raise RuntimeError(f"{source} result must contain architecture_probabilities object")


def _proxyless_search_response(result: dict) -> router_pb2.RouterProxylessSearchResponse:
    _validate_proxyless_search_result(result, "Proxyless search")
    architecture_probabilities = result["architecture_probabilities"]
    generator_names = [name for name in GENERATOR_NAMES if name in architecture_probabilities]
    generator_names.extend(
        str(name) for name in architecture_probabilities if str(name) not in set(generator_names)
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


def create_generator_router_servicer_from_env() -> GeneratorRouterServicer:
    state_path = os.getenv("TAR_STATE_PATH", "").strip()
    if not state_path:
        raise RuntimeError("TAR_STATE_PATH is required")
    bootstrap_value = os.getenv("TAR_BOOTSTRAP", "").strip().lower()
    if bootstrap_value not in {"", "false", "true"}:
        raise RuntimeError("TAR_BOOTSTRAP must be true or false")
    return GeneratorRouterServicer(
        state_path=state_path,
        bootstrap=bootstrap_value == "true",
    )


async def serve() -> None:
    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=10))
    router_pb2_grpc.add_GeneratorRouterServiceServicer_to_server(
        create_generator_router_servicer_from_env(),
        server,
    )
    server.add_insecure_port("[::]:50052")
    await server.start()
    logger.info("Generator Router Service running on :50052")
    await server.wait_for_termination()


if __name__ == "__main__":
    asyncio.run(serve())
