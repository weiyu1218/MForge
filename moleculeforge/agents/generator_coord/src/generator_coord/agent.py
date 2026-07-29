"""Generator Coordinator Agent - Coordinates multiple generators based on routing (Agent-2)."""

import asyncio
import hashlib
import importlib
import inspect
import json
import math
import os
import shlex
import struct
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping
from typing import Any, Never

from mf_agents.base.agent import (
    BaseAgent,
    agent_health_check_timeout_seconds,
    close_owned_channel,
    ensure_default_event_loop,
)
from mf_agents.crg.graph import ChemicalReasoningGraph
from mf_core.artifacts import CommandRequirement, check_command, require_available
from mf_core.db.repositories import build_shared_crg_repository_from_env
from mf_core.geometry import normalize_lorentz_embedding
from mf_core.proto_gen.moleculeforge.v1.core import audit_pb2, cig_pb2, humu_pb2
from mf_core.proto_gen.moleculeforge.v1.generator import (
    generator_pb2,
    generator_pb2_grpc,
    router_pb2,
    router_pb2_grpc,
)
from mf_core.routing.task_router import GENERATOR_NAMES

DEFAULT_GENERATORS = ["hfm_3d", "fragfm"]
if not set(DEFAULT_GENERATORS).issubset(GENERATOR_NAMES):
    raise RuntimeError("Default generators must be present in GENERATOR_NAMES")
REFINEMENT_GENERATORS = ["mmpt_rag", "fragfm"]
if not set(REFINEMENT_GENERATORS).issubset(GENERATOR_NAMES):
    raise RuntimeError("Refinement generators must be present in GENERATOR_NAMES")
_UAS_RUNNER_COMMAND = CommandRequirement("uas_runner_command", "UAS_RUNNER_COMMAND")
_GENERATOR_CONTEXT_SCHEMA = "generator_context.v1"
_MOLECULE_PAYLOAD_SCHEMA = "molecule.v1"
_EMBEDDING_PAYLOAD_SCHEMA = "humu.float32.v1"
_HUMU_COORDINATE_COUNT = 129
_HUMU_EMBEDDING_BYTES = _HUMU_COORDINATE_COUNT * 4
_FEEDBACK_ACTION = "generator_coord/feedback/v1"
_TEACHER_POLICY_FIELDS = {
    "teacher_source",
    "teacher_version",
    "allow_synthetic",
}
_TEACHER_OUTPUT_FIELDS = {
    "teacher_score",
    "teacher_source",
    "teacher_version",
    "synthetic",
}


class GeneratorGrpcClient:
    def __init__(self, target: str, generator_name: str | None = None):
        import grpc

        self.target = target
        self.generator_name = str(generator_name or "")
        ensure_default_event_loop()
        self.channel = grpc.aio.insecure_channel(target)
        self.stub = generator_pb2_grpc.GeneratorServiceStub(self.channel)
        self._closed = False

    async def info(self) -> generator_pb2.GeneratorInfo:
        return await self.stub.Info(
            generator_pb2.GeneratorInfo(),
            timeout=agent_health_check_timeout_seconds(),
        )

    async def generate(
        self,
        request: generator_pb2.GenerateRequest | dict,
    ) -> generator_pb2.GenerateResponse:
        proto_request = (
            request
            if isinstance(request, generator_pb2.GenerateRequest)
            else _generator_proto_request(request)
        )
        timeout = float(proto_request.timeout_seconds) or None
        return await self.stub.Generate(proto_request, timeout=timeout)

    async def health_check(self) -> dict:
        response = await self.info()
        if not isinstance(response, generator_pb2.GeneratorInfo):
            return {
                "healthy": False,
                "reason": "generator Info must return GeneratorInfo",
            }
        expected_name = str(getattr(self, "generator_name", "") or response.name or "")
        if not expected_name:
            return {"healthy": False, "reason": "generator Info name is required"}
        reason = _generator_info_unavailable_reason(response, expected_name)
        if reason:
            return {"healthy": False, "reason": reason}
        return {
            "healthy": True,
            "generator_name": response.name,
            "version": str(response.version or ""),
            "requires_gpu": bool(response.requires_gpu),
        }

    async def close(self) -> None:
        await close_owned_channel(self, self.channel)


class GeneratorRouterGrpcClient:
    def __init__(self, target: str):
        import grpc

        self.target = target
        ensure_default_event_loop()
        self.channel = grpc.aio.insecure_channel(target)
        self.stub = router_pb2_grpc.GeneratorRouterServiceStub(self.channel)
        self._closed = False

    async def route(self, request: router_pb2.RouterRequest) -> router_pb2.RouterResponse:
        return await self.stub.Route(
            request,
            timeout=agent_health_check_timeout_seconds(),
        )

    async def submit_feedback(
        self,
        request: router_pb2.RouterFeedbackRequest,
    ) -> router_pb2.RouterFeedbackResponse:
        return await self.stub.SubmitFeedback(
            request,
            timeout=agent_health_check_timeout_seconds(),
        )

    async def health_check(self) -> dict[str, object]:
        try:
            await asyncio.wait_for(
                self.channel.channel_ready(),
                timeout=agent_health_check_timeout_seconds(),
            )
        except TimeoutError:
            return {"healthy": False, "reason": "generator Router channel is unavailable"}
        return {"healthy": True}

    async def close(self) -> None:
        await close_owned_channel(self, self.channel)


class TeacherAdapter:
    async def health_check(self) -> dict[str, object]:
        try:
            teacher_url = _configured_teacher_url()
            timeout_seconds = _positive_environment_float("HYPSEEK_TEACHER_TIMEOUT_SECONDS")
            health = await asyncio.to_thread(
                _get_teacher_health_json,
                _teacher_health_url(teacher_url),
                timeout_seconds,
            )
            if (
                not isinstance(health, Mapping)
                or health.get("status") != "ok"
                or health.get("service") != "hypseek_teacher"
                or health.get("teacher_source") != "hypseek"
                or not isinstance(health.get("teacher_version"), str)
                or not health["teacher_version"].strip()
            ):
                raise RuntimeError("HypSeek teacher health response is invalid")
        except (RuntimeError, TypeError, ValueError) as exc:
            return {"healthy": False, "reason": str(exc)}
        return {
            "healthy": True,
            "teacher_source": health["teacher_source"],
            "teacher_version": health["teacher_version"],
        }

    async def adapt(self, group: Mapping[str, object]) -> dict[str, object]:
        if "teacher" in group:
            raise ValueError("feedback group must not contain teacher output")
        policy = _validated_teacher_policy(group.get("teacher_policy"))
        if policy["teacher_source"] != "hypseek":
            raise ValueError("default TeacherAdapter requires hypseek teacher source")
        url = _configured_teacher_url()
        timeout_seconds = _positive_environment_float("HYPSEEK_TEACHER_TIMEOUT_SECONDS")
        teacher = _validated_teacher_output(
            await asyncio.to_thread(
                _post_teacher_json,
                url,
                {
                    "records": group["records"],
                    "teacher_policy": policy,
                },
                timeout_seconds,
            )
        )
        _ensure_teacher_matches_policy(teacher, policy)
        return teacher


class UASLocalGeneratorClient:
    def __init__(self, command: str | None = None):
        self.command = (command or os.environ.get("UAS_RUNNER_COMMAND", "")).strip()
        self.timeout_seconds = float(os.environ.get("UAS_RUNNER_TIMEOUT_SECONDS", "30"))

    async def health_check(self) -> dict:
        return {
            "healthy": False,
            "generator_name": "uas",
            "version": "0.1.0",
            "reason": "local UAS compatibility client is never production READY",
        }

    async def info(self) -> generator_pb2.GeneratorInfo:
        health = await self.health_check()
        return generator_pb2.GeneratorInfo(
            name="uas",
            version="0.1.0",
            runtime_status=audit_pb2.GENERATOR_RUNTIME_STATUS_UNAVAILABLE,
            status_message=str(health["reason"]),
        )

    async def generate(self, request: dict) -> dict:
        if not self.command:
            raise RuntimeError("UAS_RUNNER_COMMAND is required")
        from mf_generators.uas.generator import UASGenerator

        batch_size = int(request.get("batch_size") or request.get("n_samples") or 1)
        generator_params = dict(request.get("generator_params", {}) or {})
        seed = request.get("seed", generator_params.get("sampling_seed"))
        seed = int(seed) if seed not in (None, "") else None
        start = time.perf_counter()
        generator = UASGenerator(runner=_UASCommandRunner(self.command, self.timeout_seconds))
        candidates = []
        async for molecule in generator.generate(
            request.get("hciv"),
            request.get("intent_cone"),
            request.get("cig") or request.get("objectives"),
            n_samples=batch_size,
            seed=seed,
        ):
            if hasattr(molecule, "model_dump"):
                candidates.append(molecule.model_dump(mode="json"))
            else:
                candidates.append(molecule)
        return {
            "generator_name": "uas",
            "generation_id": str(request.get("project_id") or request.get("request_id") or ""),
            "candidates": candidates,
            "elapsed_ms": int((time.perf_counter() - start) * 1000),
        }


class _UASCommandRunner:
    def __init__(self, command: str, timeout_seconds: float):
        self.command = command
        self.timeout_seconds = timeout_seconds

    def generate(self, **kwargs) -> list:
        _require_command_available(_UAS_RUNNER_COMMAND, self.command)
        payload = dict(kwargs)
        payload["generator"] = "uas"
        result = subprocess.run(
            shlex.split(self.command),
            input=json.dumps(payload, sort_keys=True).encode("utf-8"),
            capture_output=True,
            timeout=self.timeout_seconds,
            check=False,
        )
        if result.returncode != 0:
            stderr = result.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"UAS runner command failed: {stderr}")
        try:
            response = json.loads(result.stdout.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError("UAS runner command returned invalid JSON") from exc
        if isinstance(response, dict):
            candidates = response.get("candidates")
        else:
            candidates = response
        if not isinstance(candidates, list):
            raise RuntimeError("UAS runner command must return candidates")
        return candidates


def _command_status(requirement: CommandRequirement, command: str):
    env = {**os.environ, requirement.env_var: command}
    return check_command(requirement, env=env)


def _require_command_available(
    requirement: CommandRequirement,
    command: str,
) -> None:
    require_available([_command_status(requirement, command)])


def create_uas_generator_client() -> UASLocalGeneratorClient:
    return UASLocalGeneratorClient()


class GeneratorCoordAgent(BaseAgent):
    def __init__(
        self,
        message_bus=None,
        generator_clients: dict[str, Any] | None = None,
        generator_targets: Mapping[str, str] | None = None,
        router_client: Any = None,
        router_target: str | None = None,
        teacher_adapter: Any = None,
        crg_repository: Any = None,
    ):
        super().__init__("generator_coord", message_bus)
        self._subscription_subjects = [
            "agent.generator_coord.request",
            "orchestrator.generate.request",
        ]
        self.crg = ChemicalReasoningGraph()
        self.generators = list(GENERATOR_NAMES)
        self.generator_clients = _build_generator_clients(generator_targets)
        self.generator_clients.update(generator_clients or {})
        self.router_client = router_client or _build_router_client(router_target)
        self.teacher_adapter = teacher_adapter or TeacherAdapter()
        self._submitted_feedback_payloads: dict[tuple[str, str, int, str, str, str], str] = {}
        self._feedback_lock = asyncio.Lock()
        if crg_repository is None:
            self.crg_repository = build_shared_crg_repository_from_env()
            self._owns_crg_repository = self.crg_repository is not None
        else:
            self.crg_repository = crg_repository
            self._owns_crg_repository = False

    def runtime_targets(self) -> Mapping[str, Any]:
        targets: dict[str, Any] = {
            f"generator.{name}": self.generator_clients.get(name) for name in DEFAULT_GENERATORS
        }
        targets.update(
            {f"generator.{name}": client for name, client in self.generator_clients.items()}
        )
        if self.router_client is not None:
            targets["generator_router"] = self.router_client
        targets["teacher"] = self.teacher_adapter
        if self._owns_crg_repository:
            targets["crg_repository"] = self.crg_repository
        return targets

    async def process(self, data):
        if not isinstance(data, Mapping):
            raise TypeError("generator coordinator payload must be a mapping")
        payload = dict(data)
        if payload.get("action") == _FEEDBACK_ACTION:
            return await self._process_feedback(payload)
        return await self._process_generation(payload)

    async def _process_generation(self, data: dict) -> dict:
        if self.router_client is None:
            raise RuntimeError("GENERATOR_ROUTER_TARGET is required")
        if not self.generator_clients:
            raise RuntimeError("at least one generator client is required")
        project_id = _required_string(data, "project_id")
        _required_string(data, "run_id")
        _required_string(data, "request_id")
        context = _typed_generation_context(data)
        _validate_cig_project(context[0], project_id)
        n_samples = _positive_int(data.get("n_samples"), "n_samples")
        strategy = data.get("generation_strategy", "auto")
        objectives = dict(data.get("objectives") or {})
        crg_context = await self._read_generation_crg_context(data, strategy, objectives)
        route_humu_feedback = _route_humu_feedback_from_crg(crg_context)
        jmcg_feedback = _jmcg_feedback_envelope(data, route_humu_feedback)
        dispatch_data = _with_route_humu_feedback(
            data,
            route_humu_feedback,
            jmcg_feedback,
        )
        infos, unavailable = await self._load_generator_infos()
        available_names = _available_names_for_strategy(
            strategy=str(strategy),
            infos=infos,
        )
        if not available_names:
            details = "; ".join(f"{name}: {reason}" for name, reason in sorted(unavailable.items()))
            raise RuntimeError(f"no READY generator is available: {details}")
        n_select = _route_n_select(data, len(available_names), n_samples)
        route_request = _router_request(
            data,
            context,
            available_names=available_names,
            n_samples=n_samples,
            n_select=n_select,
        )
        route_response = await _invoke_router_route(self.router_client, route_request)
        allocations = _validated_allocations(
            route_response,
            available_names=available_names,
            n_samples=n_samples,
            n_select=n_select,
        )
        dispatch_results, candidates = await self._dispatch_generators(
            allocations,
            infos,
            dispatch_data,
            context,
        )
        selected_generators = [allocation.generator_name for allocation in allocations]
        belief = self.crg.add_belief(
            subject=str(data.get("project_id") or data.get("request_id") or strategy),
            predicate="selected_generators",
            obj=",".join(selected_generators),
            confidence=1.0,
            source_agent=self.name,
        )
        await self._persist_belief(
            belief,
            project_id=str(data.get("project_id") or ""),
            run_id=str(data.get("run_id") or data.get("request_id") or ""),
        )
        return {
            "agent": self.name,
            "status": "dispatched",
            "strategy": strategy,
            "selected_generators": selected_generators,
            "available_generators": available_names,
            "router_state_version": int(route_response.state_version),
            "dispatch_results": dispatch_results,
            "candidates": candidates,
            **({"route_humu_feedback": route_humu_feedback} if route_humu_feedback else {}),
            **({"jmcg_feedback": jmcg_feedback} if jmcg_feedback else {}),
        }

    async def _load_generator_infos(
        self,
    ) -> tuple[dict[str, generator_pb2.GeneratorInfo], dict[str, str]]:
        names = [name for name in GENERATOR_NAMES if name in self.generator_clients]
        results = await asyncio.gather(
            *(_load_generator_info(self.generator_clients[name], name) for name in names),
            return_exceptions=True,
        )
        infos: dict[str, generator_pb2.GeneratorInfo] = {}
        unavailable: dict[str, str] = {}
        for name, result in zip(names, results, strict=True):
            if isinstance(result, BaseException):
                unavailable[name] = str(result)
                continue
            reason = _generator_info_unavailable_reason(result, name)
            if reason:
                unavailable[name] = reason
                continue
            infos[name] = result
        return infos, unavailable

    async def _process_feedback(self, data: dict) -> dict:
        async with self._feedback_lock:
            return await self._process_feedback_locked(data)

    async def _process_feedback_locked(self, data: dict) -> dict:
        if self.router_client is None:
            raise RuntimeError("GENERATOR_ROUTER_TARGET is required")
        run_id = _required_string(data, "run_id")
        _required_string(data, "request_id")
        route_request_id = _required_string(data, "route_request_id")
        iteration = _non_negative_int(data.get("iteration"), "iteration")
        groups = data.get("groups")
        if not isinstance(groups, list) or not groups:
            raise ValueError("feedback groups must be a non-empty list")
        submitted = 0
        duplicates = 0
        for raw_group in groups:
            group = _validated_feedback_group(raw_group)
            key = (
                run_id,
                route_request_id,
                iteration,
                group["generator_name"],
                group["canonical_smiles"],
                group["phase"],
            )
            fingerprint = _feedback_group_fingerprint(group)
            submitted_fingerprint = self._submitted_feedback_payloads.get(key)
            if submitted_fingerprint is not None:
                if submitted_fingerprint != fingerprint:
                    raise ValueError(
                        "feedback identity was already submitted with different content"
                    )
                duplicates += 1
                continue
            adapted = self.teacher_adapter.adapt(group)
            if inspect.isawaitable(adapted):
                adapted = await adapted
            teacher = _validated_teacher_output(adapted)
            _ensure_teacher_matches_policy(teacher, group["teacher_policy"])
            feedback_request = _router_feedback_request(
                run_id=run_id,
                request_id=route_request_id,
                iteration=iteration,
                group=group,
                teacher=teacher,
            )
            response = await _invoke_router_feedback(
                self.router_client,
                feedback_request,
            )
            if not isinstance(response, router_pb2.RouterFeedbackResponse):
                raise TypeError("Router SubmitFeedback must return RouterFeedbackResponse")
            if not response.acknowledged:
                raise RuntimeError("Router did not acknowledge feedback")
            self._submitted_feedback_payloads[key] = fingerprint
            if response.duplicate:
                duplicates += 1
            else:
                submitted += 1
        return {
            "agent": self.name,
            "status": "feedback_submitted",
            "action": _FEEDBACK_ACTION,
            "submitted": submitted,
            "duplicates": duplicates,
        }

    async def _persist_belief(self, belief, project_id: str, run_id: str) -> None:
        if self.crg_repository is None:
            return
        write_belief = getattr(self.crg_repository, "write_workflow_belief", None)
        if not callable(write_belief):
            raise TypeError("crg_repository must expose write_workflow_belief(**kwargs)")
        await write_belief(
            project_id=project_id,
            run_id=run_id or belief.subject,
            belief_id=belief.id,
            subject=belief.subject,
            predicate=belief.predicate,
            object_value=belief.object,
            confidence=belief.confidence,
            source_agent=belief.source_agent,
            timestamp_ns=belief.timestamp_ns,
            evidence_ids=list(belief.evidence_ids),
        )

    async def _read_generation_crg_context(
        self,
        data: dict,
        strategy: str,
        objectives: dict,
    ) -> dict:
        run_id = str(data.get("run_id") or data.get("request_id") or "")
        if (
            strategy != "auto"
            or objectives.get("complexity") is not None
            or not run_id
            or self.crg_repository is None
            or not callable(getattr(self.crg_repository, "get_run_crg", None))
        ):
            return {}
        return await self.read_shared_crg(run_id)

    def _select_generators(
        self,
        strategy: str,
        objectives: dict,
        crg_context: dict | None = None,
    ) -> list:
        if strategy == "auto":
            complexity = objectives.get("complexity", "medium")
            if complexity == "medium" and _crg_requests_refinement(crg_context or {}):
                return list(REFINEMENT_GENERATORS)
            if complexity == "high":
                return list(REFINEMENT_GENERATORS)
            elif complexity == "low":
                return ["hfm_3d"]
            return list(DEFAULT_GENERATORS)
        elif strategy == "all":
            return list(self.generators)
        elif strategy in self.generators:
            return [strategy]
        return list(DEFAULT_GENERATORS)

    async def _dispatch_generators(
        self,
        allocations: list[router_pb2.GeneratorAllocation],
        infos: Mapping[str, generator_pb2.GeneratorInfo],
        data: dict,
        context: tuple[cig_pb2.CIG, humu_pb2.HCIV, humu_pb2.IntentCone],
    ) -> tuple[list[dict], list[dict]]:
        tasks = [
            self._dispatch_allocation(allocation, infos[allocation.generator_name], data, context)
            for allocation in allocations
        ]
        results = await asyncio.gather(*tasks)
        dispatch_results = [result[0] for result in results]
        candidates = [
            candidate
            for _dispatch_result, allocation_candidates in results
            for candidate in allocation_candidates
        ]
        expected = sum(int(allocation.n_samples) for allocation in allocations)
        if len(candidates) != expected:
            raise RuntimeError(
                f"generator candidate count mismatch: expected {expected}, got {len(candidates)}"
            )
        return dispatch_results, candidates

    async def _dispatch_allocation(
        self,
        allocation: router_pb2.GeneratorAllocation,
        info: generator_pb2.GeneratorInfo,
        data: dict,
        context: tuple[cig_pb2.CIG, humu_pb2.HCIV, humu_pb2.IntentCone],
    ) -> tuple[dict, list[dict]]:
        generator_name = allocation.generator_name
        client = self.generator_clients[generator_name]
        remaining = int(allocation.n_samples)
        chunk_index = 0
        candidates: list[dict] = []
        while remaining:
            chunk_size = min(remaining, int(info.max_batch_size))
            chunk_id = _chunk_id(data, generator_name, chunk_index)
            chunk_seed = _chunk_seed(data, generator_name, chunk_index)
            request = _chunk_generate_request(
                data=data,
                generator_name=generator_name,
                context=context,
                chunk_id=chunk_id,
                chunk_seed=chunk_seed,
                chunk_size=chunk_size,
            )
            response = await _invoke_generator_client(client, request)
            candidates.extend(
                _strict_response_candidates(
                    response=response,
                    request=request,
                    info=info,
                    chunk_id=chunk_id,
                    chunk_seed=chunk_seed,
                )
            )
            remaining -= chunk_size
            chunk_index += 1
        if len(candidates) != int(allocation.n_samples):
            raise RuntimeError(
                f"{generator_name} allocation count mismatch: "
                f"expected {allocation.n_samples}, got {len(candidates)}"
            )
        return (
            {
                "generator": generator_name,
                "candidate_count": len(candidates),
                "chunk_count": chunk_index,
                "health_status": "ready",
            },
            candidates,
        )


def _typed_generation_context(
    data: Mapping[str, object],
) -> tuple[cig_pb2.CIG, humu_pb2.HCIV, humu_pb2.IntentCone]:
    if "cig" not in data:
        raise ValueError("cig is required")
    if "hciv" not in data:
        raise ValueError("hciv is required")
    if "intent_cone" not in data:
        raise ValueError("intent_cone is required")
    cig = _cig_from_value(data["cig"])
    hciv = _hciv_from_value(data["hciv"])
    cone = _intent_cone_from_value(data["intent_cone"])
    _validate_cig(cig)
    _validate_hciv(hciv)
    _validate_intent_cone(cone, hciv.curvature)
    return cig, hciv, cone


def _cig_from_value(value: object) -> cig_pb2.CIG:
    if isinstance(value, cig_pb2.CIG):
        result = cig_pb2.CIG()
        result.CopyFrom(value)
        return result
    if not isinstance(value, Mapping):
        raise ValueError("cig must be a CIG message or mapping")
    objectives = []
    for raw_objective in value.get("objectives", []):
        if not isinstance(raw_objective, Mapping):
            raise ValueError("cig objectives must be mappings")
        payload: dict[str, object] = {
            "id": str(raw_objective.get("id") or ""),
            "name": str(raw_objective.get("name") or ""),
            "type": _objective_type(raw_objective.get("type")),
            "target_value": _finite_float(
                raw_objective.get("target_value", 0.0),
                "cig objective target_value",
            ),
            "property": str(raw_objective.get("property") or ""),
            "weight": _finite_float(
                raw_objective.get("weight", 0.0),
                "cig objective weight",
            ),
            "pareto_tier": int(raw_objective.get("pareto_tier", 0)),
        }
        if raw_objective.get("target_min") is not None:
            payload["target_min"] = _finite_float(
                raw_objective["target_min"],
                "cig objective target_min",
            )
        if raw_objective.get("target_max") is not None:
            payload["target_max"] = _finite_float(
                raw_objective["target_max"],
                "cig objective target_max",
            )
        objectives.append(cig_pb2.ObjectiveNode(**payload))
    edges = []
    for raw_edge in value.get("edges", []):
        if not isinstance(raw_edge, Mapping):
            raise ValueError("cig edges must be mappings")
        edges.append(
            cig_pb2.ObjectiveEdge(
                source_id=str(raw_edge.get("source_id") or ""),
                target_id=str(raw_edge.get("target_id") or ""),
                relation=str(raw_edge.get("relation") or ""),
                strength=_finite_float(
                    raw_edge.get("strength", 0.0),
                    "cig edge strength",
                ),
            )
        )
    hyperedges = []
    for raw_edge in value.get("hyperedges", []):
        if not isinstance(raw_edge, Mapping):
            raise ValueError("cig hyperedges must be mappings")
        hyperedges.append(
            cig_pb2.ObjectiveHyperedge(
                source_ids=[str(item) for item in raw_edge.get("source_ids", [])],
                target_ids=[str(item) for item in raw_edge.get("target_ids", [])],
                relation=str(raw_edge.get("relation") or ""),
                strength=_finite_float(
                    raw_edge.get("strength", 0.0),
                    "cig hyperedge strength",
                ),
            )
        )
    constraints = value.get("constraints", {})
    if not isinstance(constraints, Mapping):
        raise ValueError("cig constraints must be a mapping")
    return cig_pb2.CIG(
        project_id=str(value.get("project_id") or ""),
        objectives=objectives,
        edges=edges,
        hyperedges=hyperedges,
        constraints={str(key): str(item) for key, item in constraints.items()},
        created_by=str(value.get("created_by") or ""),
    )


def _objective_type(value: object) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    normalized = str(value or "").strip().upper()
    aliases = {
        "CONTINUOUS_MAXIMIZE": "MAXIMIZE",
        "CONTINUOUS_MINIMIZE": "MINIMIZE",
        "MULTI_CONSTRAINT_SATISFY": "CONSTRAINT",
    }
    normalized = aliases.get(normalized, normalized)
    try:
        return cig_pb2.ObjectiveType.Value(normalized)
    except ValueError as exc:
        raise ValueError(f"unsupported CIG objective type: {value}") from exc


def _hciv_from_value(value: object) -> humu_pb2.HCIV:
    if isinstance(value, humu_pb2.HCIV):
        result = humu_pb2.HCIV()
        result.CopyFrom(value)
        return result
    if not isinstance(value, Mapping):
        raise ValueError("hciv must be an HCIV message or mapping")
    coordinates = value.get("coordinates")
    if not isinstance(coordinates, list | tuple):
        raise ValueError("hciv coordinates must be a sequence")
    payload: dict[str, object] = {
        "coordinates": [
            _finite_float(item, "hciv coordinates must be finite") for item in coordinates
        ],
        "curvature": _finite_float(value.get("curvature"), "hciv curvature"),
        "molecule_smiles": str(value.get("molecule_smiles") or ""),
    }
    if value.get("parent_hciv_id") not in (None, ""):
        payload["parent_hciv_id"] = str(value["parent_hciv_id"])
    return humu_pb2.HCIV(**payload)


def _intent_cone_from_value(value: object) -> humu_pb2.IntentCone:
    if isinstance(value, humu_pb2.IntentCone):
        result = humu_pb2.IntentCone()
        result.CopyFrom(value)
        return result
    if not isinstance(value, Mapping):
        raise ValueError("intent_cone must be an IntentCone message or mapping")
    axis = value.get("axis")
    if not isinstance(axis, list | tuple):
        raise ValueError("intent_cone axis must be a sequence")
    weights = value.get("property_weights", {})
    if not isinstance(weights, Mapping):
        raise ValueError("intent_cone property_weights must be a mapping")
    return humu_pb2.IntentCone(
        axis=[_finite_float(item, "intent_cone axis must be finite") for item in axis],
        half_angle=_finite_float(
            value.get("half_angle"),
            "intent_cone half_angle",
        ),
        curvature=_finite_float(
            value.get("curvature"),
            "intent_cone curvature",
        ),
        property_weights={
            str(key): _finite_float(item, "intent_cone property weights")
            for key, item in weights.items()
        },
    )


def _validate_cig(cig: cig_pb2.CIG) -> None:
    if not cig.project_id:
        raise ValueError("cig project_id is required")
    if not cig.objectives:
        raise ValueError("cig must contain at least one objective")
    node_ids = [objective.id for objective in cig.objectives]
    if any(not node_id for node_id in node_ids):
        raise ValueError("cig objective id is required")
    if len(node_ids) != len(set(node_ids)):
        raise ValueError("cig objective ids must be unique")
    known_ids = set(node_ids)
    known_types = {
        cig_pb2.MAXIMIZE,
        cig_pb2.MINIMIZE,
        cig_pb2.TARGET_RANGE,
        cig_pb2.CONSTRAINT,
    }
    for objective in cig.objectives:
        if objective.type not in known_types:
            raise ValueError(f"cig objective {objective.id} type is invalid")
        numeric_values = [objective.target_value, objective.weight]
        if objective.HasField("target_min"):
            numeric_values.append(objective.target_min)
        if objective.HasField("target_max"):
            numeric_values.append(objective.target_max)
        if not all(math.isfinite(value) for value in numeric_values):
            raise ValueError(f"cig objective {objective.id} values must be finite")
        if objective.type == cig_pb2.TARGET_RANGE and (
            not objective.HasField("target_min")
            or not objective.HasField("target_max")
            or objective.target_min > objective.target_max
        ):
            raise ValueError(f"cig objective {objective.id} target range is invalid")
    for edge in cig.edges:
        if edge.source_id not in known_ids or edge.target_id not in known_ids:
            raise ValueError("cig edge references an unknown objective")
        if not math.isfinite(edge.strength):
            raise ValueError("cig edge strength must be finite")
    for edge in cig.hyperedges:
        if (
            not edge.source_ids
            or not edge.target_ids
            or any(item not in known_ids for item in [*edge.source_ids, *edge.target_ids])
        ):
            raise ValueError("cig hyperedge references an unknown objective")
        if not math.isfinite(edge.strength):
            raise ValueError("cig hyperedge strength must be finite")


def _validate_cig_project(cig: cig_pb2.CIG, project_id: str) -> None:
    if cig.project_id != project_id:
        raise ValueError("cig project_id must match request project_id")


def _validate_hciv(hciv: humu_pb2.HCIV) -> None:
    if len(hciv.coordinates) != _HUMU_COORDINATE_COUNT:
        raise ValueError("hciv coordinates must contain exactly 129 values")
    if not math.isfinite(hciv.curvature) or hciv.curvature <= 0.0:
        raise ValueError("hciv curvature must be finite and positive")
    if not all(math.isfinite(item) for item in hciv.coordinates):
        raise ValueError("hciv coordinates must be finite")
    if (
        normalize_lorentz_embedding(
            list(hciv.coordinates),
            expected_dim=_HUMU_COORDINATE_COUNT,
            curvature=hciv.curvature,
        )
        is None
    ):
        raise ValueError("hciv coordinates are outside the configured Lorentz manifold")


def _validate_intent_cone(
    cone: humu_pb2.IntentCone,
    hciv_curvature: float,
) -> None:
    if len(cone.axis) != _HUMU_COORDINATE_COUNT:
        raise ValueError("intent_cone axis must contain exactly 129 values")
    if not all(math.isfinite(item) for item in cone.axis):
        raise ValueError("intent_cone axis must be finite")
    if not math.isfinite(cone.half_angle) or cone.half_angle <= 0.0 or cone.half_angle > math.pi:
        raise ValueError("intent_cone half_angle must be in (0, pi]")
    if not math.isfinite(cone.curvature) or cone.curvature <= 0.0:
        raise ValueError("intent_cone curvature must be finite and positive")
    if not math.isclose(
        cone.curvature,
        hciv_curvature,
        rel_tol=1e-6,
        abs_tol=1e-8,
    ):
        raise ValueError("intent_cone curvature must match hciv curvature")
    if not all(math.isfinite(item) for item in cone.property_weights.values()):
        raise ValueError("intent_cone property weights must be finite")


def _finite_float(value: object, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _positive_int(value: object, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _non_negative_int(value: object, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _required_string(data: Mapping[str, object], field: str) -> str:
    raw_value = data.get(field)
    if not isinstance(raw_value, str) or not raw_value.strip():
        raise ValueError(f"{field} is required")
    return raw_value.strip()


async def _load_generator_info(
    client: Any,
    generator_name: str,
) -> generator_pb2.GeneratorInfo:
    info = getattr(client, "info", None)
    if not callable(info):
        raise RuntimeError(f"{generator_name} client does not expose Info")
    result = info()
    if inspect.isawaitable(result):
        result = await result
    if not isinstance(result, generator_pb2.GeneratorInfo):
        raise TypeError(f"{generator_name} Info must return GeneratorInfo")
    return result


def _generator_info_unavailable_reason(
    info: generator_pb2.GeneratorInfo,
    expected_name: str,
) -> str:
    if info.name != expected_name:
        return f"Info name mismatch: expected {expected_name}, got {info.name or '<empty>'}"
    if info.runtime_status != audit_pb2.GENERATOR_RUNTIME_STATUS_READY:
        return info.status_message or "Info runtime status is not READY"
    if not info.version:
        return "Info version is required"
    if info.max_batch_size <= 0:
        return "Info max_batch_size must be positive"
    if not info.artifacts:
        return "Info must contain artifact refs"
    if not any(artifact.required for artifact in info.artifacts):
        return "Info must contain at least one required artifact"
    for artifact in info.artifacts:
        if artifact.required and (
            not artifact.name or not artifact.version or not artifact.checksum
        ):
            return "required Info artifacts need name, version and checksum"
    return ""


def _available_names_for_strategy(
    *,
    strategy: str,
    infos: Mapping[str, generator_pb2.GeneratorInfo],
) -> list[str]:
    names = [name for name in GENERATOR_NAMES if name in infos]
    if strategy in {"", "auto", "all"}:
        return names
    if strategy not in GENERATOR_NAMES:
        raise ValueError(f"unknown generation_strategy: {strategy}")
    return [strategy] if strategy in infos else []


def _route_n_select(data: Mapping[str, object], available_count: int, n_samples: int) -> int:
    raw = data.get("n_select")
    if raw is None:
        strategy = str(data.get("generation_strategy") or "auto")
        value = 1 if strategy in GENERATOR_NAMES else min(2, available_count, n_samples)
    else:
        value = _positive_int(raw, "n_select")
    if value > available_count:
        raise ValueError("n_select exceeds READY generators")
    if value > n_samples:
        raise ValueError("n_select exceeds n_samples")
    return value


def _router_request(
    data: Mapping[str, object],
    context: tuple[cig_pb2.CIG, humu_pb2.HCIV, humu_pb2.IntentCone],
    *,
    available_names: list[str],
    n_samples: int,
    n_select: int,
) -> router_pb2.RouterRequest:
    cig, hciv, _cone = context
    profile = data.get("task_profile") or {}
    if not isinstance(profile, Mapping):
        raise ValueError("task_profile must be a mapping")
    return router_pb2.RouterRequest(
        project_id=_required_string(data, "project_id"),
        run_id=_required_string(data, "run_id"),
        request_id=_required_string(data, "request_id"),
        cig=cig.SerializeToString(deterministic=True),
        hciv=[float(item) for item in hciv.coordinates],
        n_select=n_select,
        n_samples=n_samples,
        available_generator_names=available_names,
        task_complexity=_task_complexity(data, profile),
        target_family=str(profile.get("target_family") or ""),
        stage=str(profile.get("stage") or ""),
        data_richness=_optional_finite_float(profile, "data_richness"),
        novelty_demand=_optional_finite_float(profile, "novelty_demand"),
        multi_target=_optional_bool(profile, "multi_target"),
        sa_constraint=_optional_finite_float(profile, "sa_constraint"),
    )


def _optional_finite_float(data: Mapping[str, object], field: str) -> float:
    if field not in data:
        return 0.0
    return _finite_float(data[field], f"task_profile {field}")


def _optional_bool(data: Mapping[str, object], field: str) -> bool:
    if field not in data:
        return False
    value = data[field]
    if not isinstance(value, bool):
        raise ValueError(f"task_profile {field} must be boolean")
    return value


def _task_complexity(
    data: Mapping[str, object],
    profile: Mapping[str, object],
) -> int:
    objectives = data.get("objectives")
    objective_complexity = objectives.get("complexity") if isinstance(objectives, Mapping) else None
    value = str(profile.get("task_complexity") or objective_complexity or "").lower()
    mapping = {
        "": router_pb2.TASK_COMPLEXITY_UNSPECIFIED,
        "low": router_pb2.TASK_COMPLEXITY_LOW,
        "medium": router_pb2.TASK_COMPLEXITY_MEDIUM,
        "high": router_pb2.TASK_COMPLEXITY_HIGH,
    }
    if value not in mapping:
        raise ValueError(f"unsupported task complexity: {value}")
    return mapping[value]


async def _invoke_router_route(
    client: Any,
    request: router_pb2.RouterRequest,
) -> router_pb2.RouterResponse:
    method = getattr(client, "route", None) or getattr(client, "Route", None)
    if not callable(method):
        raise TypeError("Router client must expose route(request)")
    result = method(request)
    if inspect.isawaitable(result):
        result = await result
    if not isinstance(result, router_pb2.RouterResponse):
        raise TypeError("Router Route must return RouterResponse")
    return result


def _validated_allocations(
    response: router_pb2.RouterResponse,
    *,
    available_names: list[str],
    n_samples: int,
    n_select: int,
) -> list[router_pb2.GeneratorAllocation]:
    allocations = list(response.allocations)
    if not allocations:
        raise RuntimeError("Router returned no allocations")
    names = [allocation.generator_name for allocation in allocations]
    if len(names) != len(set(names)):
        raise RuntimeError("Router returned duplicate generator allocations")
    unavailable = [name for name in names if name not in available_names]
    if unavailable:
        raise RuntimeError(f"Router allocation references unavailable generator: {unavailable[0]}")
    if any(allocation.n_samples <= 0 for allocation in allocations):
        raise RuntimeError("Router allocations must be positive")
    if sum(int(allocation.n_samples) for allocation in allocations) != n_samples:
        raise RuntimeError("Router allocation sum does not match n_samples")
    if list(response.selected_generators) != names:
        raise RuntimeError("Router selected_generators do not match allocations")
    selection_weights = list(response.selection_weights)
    if len(selection_weights) != len(names):
        raise RuntimeError("Router selection_weights do not match selected_generators")
    expected_rewards = list(response.expected_rewards)
    if len(expected_rewards) != len(names):
        raise RuntimeError("Router expected_rewards do not match selected_generators")
    numeric_values = [
        *selection_weights,
        *expected_rewards,
        *(allocation.normalized_weight for allocation in allocations),
        *(allocation.expected_reward for allocation in allocations),
    ]
    if any(not math.isfinite(value) for value in numeric_values):
        raise RuntimeError("Router weights and expected rewards must be finite")
    if any(value < 0.0 for value in numeric_values):
        raise RuntimeError("Router weights and expected rewards must be non-negative")
    if not math.isclose(sum(selection_weights), 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise RuntimeError("Router selection_weights must sum to one")
    for allocation, selection_weight, expected_reward in zip(
        allocations,
        selection_weights,
        expected_rewards,
        strict=True,
    ):
        if not math.isclose(
            allocation.normalized_weight,
            selection_weight,
            rel_tol=0.0,
            abs_tol=1e-6,
        ) or not math.isclose(
            allocation.expected_reward,
            expected_reward,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise RuntimeError("Router allocation weights do not match response vectors")
    if len(allocations) != n_select:
        raise RuntimeError("Router allocation count does not match n_select")
    return allocations


def _chunk_id(
    data: Mapping[str, object],
    generator_name: str,
    chunk_index: int,
) -> str:
    request_id = _required_string(data, "request_id")
    return f"{request_id}:{generator_name}:chunk-{chunk_index:04d}"


def _chunk_seed(
    data: Mapping[str, object],
    generator_name: str,
    chunk_index: int,
) -> int:
    params = data.get("generator_params") or {}
    if not isinstance(params, Mapping):
        raise ValueError("generator_params must be a mapping")
    raw_seed = params.get("sampling_seed", data.get("seed", 0))
    try:
        base_seed = int(raw_seed)
    except (TypeError, ValueError) as exc:
        raise ValueError("sampling_seed must be an integer") from exc
    generator_index = GENERATOR_NAMES.index(generator_name)
    return (base_seed + (generator_index + 1) * 1_000_003 + chunk_index * 97_409) % 2_147_483_647


def _chunk_generate_request(
    *,
    data: Mapping[str, object],
    generator_name: str,
    context: tuple[cig_pb2.CIG, humu_pb2.HCIV, humu_pb2.IntentCone],
    chunk_id: str,
    chunk_seed: int,
    chunk_size: int,
) -> generator_pb2.GenerateRequest:
    cig, hciv, cone = context
    objectives = data.get("objectives") or {}
    if not isinstance(objectives, Mapping):
        raise ValueError("objectives must be a mapping")
    raw_params = data.get("generator_params") or {}
    if not isinstance(raw_params, Mapping):
        raise ValueError("generator_params must be a mapping")
    params = {str(key): str(value) for key, value in raw_params.items()}
    params.update(
        {
            "generator": generator_name,
            "chunk_id": chunk_id,
            "chunk_seed": str(chunk_seed),
            "sampling_seed": str(chunk_seed),
            "seed": str(chunk_seed),
        }
    )
    request = generator_pb2.GenerateRequest(
        project_id=_required_string(data, "project_id"),
        request_id=chunk_id,
        batch_size=chunk_size,
        total_molecules=chunk_size,
        intent_cone=cone.SerializeToString(deterministic=True),
        target_properties=[str(key) for key in objectives if str(key) != "complexity"],
        property_targets={
            str(key): float(value)
            for key, value in objectives.items()
            if str(key) != "complexity"
            and isinstance(value, int | float)
            and not isinstance(value, bool)
            and math.isfinite(float(value))
        },
        generator_params=params,
        timeout_seconds=int(data.get("timeout_seconds") or 0),
        context_schema_version=_GENERATOR_CONTEXT_SCHEMA,
    )
    request.cig.CopyFrom(cig)
    request.hciv.CopyFrom(hciv)
    return request


def _strict_response_candidates(
    *,
    response: object,
    request: generator_pb2.GenerateRequest,
    info: generator_pb2.GeneratorInfo,
    chunk_id: str,
    chunk_seed: int,
) -> list[dict]:
    if not isinstance(response, generator_pb2.GenerateResponse):
        raise TypeError("generator must return GenerateResponse")
    if response.request_id != request.request_id:
        raise RuntimeError("generator response request_id mismatch")
    if response.generator_name != info.name:
        raise RuntimeError("generator response generator_name mismatch")
    if response.molecule_payload_schema != _MOLECULE_PAYLOAD_SCHEMA:
        raise RuntimeError("generator response molecule payload schema mismatch")
    if response.embedding_payload_schema != _EMBEDDING_PAYLOAD_SCHEMA:
        raise RuntimeError("generator response embedding payload schema mismatch")
    if _artifact_tuples(response.artifacts) != _artifact_tuples(info.artifacts):
        raise RuntimeError("generator response artifacts do not match Info")
    if len(response.molecules) != request.batch_size:
        raise RuntimeError(
            f"generator response count mismatch: expected {request.batch_size}, "
            f"got {len(response.molecules)}"
        )
    embeddings = _decode_humu_embeddings(
        list(response.humu_embeddings),
        molecule_count=len(response.molecules),
    )
    artifact_refs = [_artifact_dict(artifact) for artifact in response.artifacts]
    candidates = []
    for index, payload in enumerate(response.molecules):
        candidate = _decode_strict_molecule(payload)
        candidate["generator"] = info.name
        candidate["generator_name"] = info.name
        candidate["chunk_id"] = chunk_id
        candidate["chunk_seed"] = chunk_seed
        candidate["artifact_refs"] = [dict(item) for item in artifact_refs]
        if embeddings:
            candidate["humu_embedding"] = embeddings[index]
        candidates.append(candidate)
    return candidates


def _decode_strict_molecule(payload: bytes) -> dict:
    try:
        text = payload.decode("utf-8")
        decoded = json.loads(
            text,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"invalid JSON constant: {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError("generator molecule payload is not strict JSON") from exc
    if not isinstance(decoded, dict):
        raise RuntimeError("generator molecule JSON must be an object")
    smiles = decoded.get("smiles")
    canonical_smiles = decoded.get("canonical_smiles")
    if not isinstance(smiles, str) or not smiles:
        raise RuntimeError("generator molecule JSON requires smiles")
    if not isinstance(canonical_smiles, str) or not canonical_smiles:
        raise RuntimeError("generator molecule JSON requires canonical_smiles")
    _validate_json_finite(decoded)
    return dict(decoded)


def _validate_json_finite(value: object) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise RuntimeError("generator molecule JSON numbers must be finite")
    if isinstance(value, Mapping):
        for item in value.values():
            _validate_json_finite(item)
    elif isinstance(value, list | tuple):
        for item in value:
            _validate_json_finite(item)


def _decode_humu_embeddings(
    payloads: list[bytes],
    *,
    molecule_count: int,
) -> list[list[float]]:
    if not payloads:
        return []
    if len(payloads) != molecule_count:
        raise RuntimeError("generator HuMU embedding count must equal molecule count")
    decoded = []
    for payload in payloads:
        if len(payload) != _HUMU_EMBEDDING_BYTES:
            raise RuntimeError("generator HuMU embedding must contain exactly 516 bytes")
        values = list(struct.unpack("<129f", payload))
        if not all(math.isfinite(value) for value in values):
            raise RuntimeError("generator HuMU embedding values must be finite")
        decoded.append(values)
    return decoded


def _artifact_tuples(artifacts) -> list[tuple[str, str, str, bool]]:
    return [
        (
            str(artifact.name),
            str(artifact.version),
            str(artifact.checksum),
            bool(artifact.required),
        )
        for artifact in artifacts
    ]


def _artifact_dict(artifact: audit_pb2.ArtifactRef) -> dict[str, object]:
    return {
        "name": str(artifact.name),
        "version": str(artifact.version),
        "checksum": str(artifact.checksum),
        "required": bool(artifact.required),
    }


def _validated_feedback_group(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("feedback group must be a mapping")
    phase = value.get("phase")
    if not isinstance(phase, str) or phase not in {"validation", "critic"}:
        raise ValueError("feedback phase must be validation or critic")
    generator_name = value.get("generator_name")
    if not isinstance(generator_name, str) or generator_name not in GENERATOR_NAMES:
        raise ValueError("feedback generator_name is invalid")
    canonical_smiles = value.get("canonical_smiles")
    if (
        not isinstance(canonical_smiles, str)
        or not canonical_smiles
        or canonical_smiles != canonical_smiles.strip()
    ):
        raise ValueError("feedback canonical_smiles must be a non-empty trimmed string")
    candidate_ids = _non_empty_string_list(value.get("candidate_ids"), "candidate_ids")
    evidence_ids = _non_empty_string_list(value.get("evidence_ids"), "evidence_ids")
    records = value.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("feedback records must be a non-empty list")
    if not all(isinstance(record, Mapping) for record in records):
        raise ValueError("feedback records must contain mappings")
    teacher_policy = _validated_teacher_policy(value.get("teacher_policy"))
    if "teacher" in value:
        raise ValueError("feedback group must not contain teacher output")
    result = dict(value)
    result.update(
        {
            "phase": phase,
            "generator_name": generator_name,
            "canonical_smiles": canonical_smiles,
            "candidate_ids": candidate_ids,
            "evidence_ids": evidence_ids,
            "records": [dict(record) for record in records],
            "teacher_policy": teacher_policy,
        }
    )
    return result


def _feedback_group_fingerprint(group: Mapping[str, object]) -> str:
    try:
        payload = json.dumps(
            dict(group),
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("feedback group content must be canonical JSON") from exc
    return hashlib.sha256(payload).hexdigest()


def _non_empty_string_list(value: object, field: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"feedback {field} must be a non-empty list")
    if any(not isinstance(item, str) or not item or item != item.strip() for item in value):
        raise ValueError(f"feedback {field} must contain non-empty trimmed strings")
    return list(value)


def _validated_teacher_output(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("TeacherAdapter output must be a mapping")
    if set(value) != _TEACHER_OUTPUT_FIELDS:
        raise ValueError(
            "TeacherAdapter output must contain exactly teacher_score, "
            "teacher_source, teacher_version, and synthetic"
        )
    if "teacher_score" not in value:
        raise ValueError("TeacherAdapter output requires teacher_score")
    score_value = value["teacher_score"]
    if isinstance(score_value, bool):
        raise ValueError("teacher_score must be numeric")
    score = _finite_float(score_value, "teacher_score")
    if not 0.0 <= score <= 1.0:
        raise ValueError("teacher_score must be in [0, 1]")
    source_value = value.get("teacher_source")
    version_value = value.get("teacher_version")
    if not isinstance(source_value, str) or not source_value.strip():
        raise ValueError("teacher source is required")
    if not isinstance(version_value, str) or not version_value.strip():
        raise ValueError("teacher version is required")
    source = source_value.strip()
    version = version_value.strip()
    synthetic = value.get("synthetic")
    if not isinstance(synthetic, bool):
        raise ValueError("teacher synthetic flag must be boolean")
    return {
        "teacher_score": score,
        "teacher_source": source,
        "teacher_version": version,
        "synthetic": synthetic,
    }


def _validated_teacher_policy(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != _TEACHER_POLICY_FIELDS:
        raise ValueError(
            "feedback teacher_policy must contain exactly teacher_source, "
            "teacher_version, and allow_synthetic"
        )
    source = value["teacher_source"]
    version = value["teacher_version"]
    allow_synthetic = value["allow_synthetic"]
    if not isinstance(source, str) or not source.strip():
        raise ValueError("teacher_policy.teacher_source must be a non-empty string")
    if not isinstance(version, str) or not version.strip():
        raise ValueError("teacher_policy.teacher_version must be a non-empty string")
    if not isinstance(allow_synthetic, bool):
        raise ValueError("teacher_policy.allow_synthetic must be a boolean")
    return {
        "teacher_source": source.strip(),
        "teacher_version": version.strip(),
        "allow_synthetic": allow_synthetic,
    }


def _ensure_teacher_matches_policy(
    teacher: Mapping[str, object],
    policy: Mapping[str, object],
) -> None:
    if teacher["teacher_source"] != policy["teacher_source"]:
        raise ValueError("teacher source does not match teacher_policy")
    if teacher["teacher_version"] != policy["teacher_version"]:
        raise ValueError("teacher version does not match teacher_policy")
    if teacher["synthetic"] and not policy["allow_synthetic"]:
        raise ValueError("synthetic teacher output is disallowed by teacher_policy")


def _positive_environment_float(name: str) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        raise RuntimeError(f"{name} is required")
    try:
        value = float(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a finite positive number") from exc
    if not math.isfinite(value) or value <= 0.0:
        raise RuntimeError(f"{name} must be a finite positive number")
    return value


def _configured_teacher_url() -> str:
    url = os.environ.get("HYPSEEK_TEACHER_URL", "").strip()
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise RuntimeError("HYPSEEK_TEACHER_URL must be an HTTP(S) URL")
    return url


def _teacher_health_url(teacher_url: str) -> str:
    parsed = urllib.parse.urlsplit(teacher_url)
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, "/healthz", "", ""))


def _get_teacher_health_json(
    url: str,
    timeout_seconds: float,
) -> dict[str, object]:
    request = urllib.request.Request(url, method="GET")  # noqa: S310
    try:
        with urllib.request.urlopen(  # noqa: S310
            request,
            timeout=timeout_seconds,
        ) as response:
            status = response.getcode()
            response_body = response.read()
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HypSeek teacher health returned HTTP {exc.code}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"HypSeek teacher health request failed: {exc}") from exc
    if status != 200:
        raise RuntimeError(f"HypSeek teacher health returned HTTP {status}")
    try:
        result = json.loads(
            response_body.decode("utf-8"),
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError("HypSeek teacher health response must be strict JSON") from exc
    if not isinstance(result, dict):
        raise RuntimeError("HypSeek teacher health response must be a JSON object")
    return result


def _post_teacher_json(
    url: str,
    payload: dict[str, object],
    timeout_seconds: float,
) -> dict[str, object]:
    try:
        body = json.dumps(
            payload,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("HypSeek teacher request must be canonical JSON") from exc
    request = urllib.request.Request(  # noqa: S310
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(  # noqa: S310
            request,
            timeout=timeout_seconds,
        ) as response:
            status = response.getcode()
            response_body = response.read()
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HypSeek teacher request returned HTTP {exc.code}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"HypSeek teacher request failed: {exc}") from exc
    if status != 200:
        raise RuntimeError(f"HypSeek teacher request returned HTTP {status}")
    try:
        result = json.loads(
            response_body.decode("utf-8"),
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError("HypSeek teacher response must be strict JSON") from exc
    if not isinstance(result, dict):
        raise RuntimeError("HypSeek teacher response must be a JSON object")
    return result


def _reject_json_constant(value: str) -> Never:
    raise ValueError(f"invalid JSON numeric constant: {value}")


def _router_feedback_request(
    *,
    run_id: str,
    request_id: str,
    iteration: int,
    group: Mapping[str, object],
    teacher: Mapping[str, object],
) -> router_pb2.RouterFeedbackRequest:
    identity = json.dumps(
        {
            "run_id": run_id,
            "request_id": request_id,
            "iteration": iteration,
            "phase": group["phase"],
            "generator_name": group["generator_name"],
            "canonical_smiles": group["canonical_smiles"],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    phase = (
        router_pb2.ROUTER_FEEDBACK_PHASE_VALIDATION
        if group["phase"] == "validation"
        else router_pb2.ROUTER_FEEDBACK_PHASE_CRITIC
    )
    return router_pb2.RouterFeedbackRequest(
        feedback_id=f"generator-coord-{hashlib.sha256(identity).hexdigest()}",
        run_id=run_id,
        request_id=request_id,
        iteration=iteration,
        phase=phase,
        generator_name=str(group["generator_name"]),
        candidate_ids=list(group["candidate_ids"]),
        canonical_smiles=str(group["canonical_smiles"]),
        evidence_ids=list(group["evidence_ids"]),
        teacher_score=float(teacher["teacher_score"]),
        teacher_source=str(teacher["teacher_source"]),
        teacher_version=str(teacher["teacher_version"]),
        synthetic=bool(teacher["synthetic"]),
    )


async def _invoke_router_feedback(
    client: Any,
    request: router_pb2.RouterFeedbackRequest,
) -> router_pb2.RouterFeedbackResponse:
    method = getattr(client, "submit_feedback", None) or getattr(client, "SubmitFeedback", None)
    if not callable(method):
        raise TypeError("Router client must expose submit_feedback(request)")
    result = method(request)
    if inspect.isawaitable(result):
        result = await result
    return result


def _build_router_client(target: str | None) -> GeneratorRouterGrpcClient | None:
    resolved = (
        str(target).strip()
        if target is not None
        else os.environ.get("GENERATOR_ROUTER_TARGET", "").strip()
    )
    if not resolved:
        return None
    return GeneratorRouterGrpcClient(resolved)


def _build_generator_clients(
    generator_targets: Mapping[str, str] | None,
) -> dict[str, GeneratorGrpcClient]:
    targets = dict(generator_targets or _generator_targets_from_env())
    clients = {}
    for generator_name, target in targets.items():
        if generator_name not in GENERATOR_NAMES:
            raise ValueError(f"Unknown generator target: {generator_name}")
        if not target:
            raise ValueError(f"Generator target is empty: {generator_name}")
        clients[generator_name] = _generator_client_from_target(
            str(target),
            generator_name,
        )
    return clients


def _generator_client_from_target(target: str, generator_name: str) -> Any:
    if target.startswith(("python://", "python:")):
        return _python_target(target)
    return GeneratorGrpcClient(target, generator_name)


def _python_target(uri: str) -> Any:
    target = uri.removeprefix("python://").removeprefix("python:")
    if ":" not in target:
        raise ValueError("python generator target must be python://module:function")
    module_name, function_name = target.split(":", 1)
    provider = getattr(importlib.import_module(module_name), function_name, None)
    if not callable(provider):
        raise RuntimeError(f"python generator target is not callable: {uri}")
    client = provider()
    if not (callable(client) or callable(getattr(client, "generate", None))):
        raise TypeError("python generator target must return a callable client")
    return client


def _generator_targets_from_env() -> dict[str, str]:
    targets = {}
    discovery_uri = os.environ.get("GENERATOR_DISCOVERY_URI", "")
    if discovery_uri:
        targets.update(_generator_targets_from_discovery_uri(discovery_uri))
    raw = os.environ.get("GENERATOR_CLIENT_TARGETS", "")
    if raw:
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError("GENERATOR_CLIENT_TARGETS must be a JSON object")
        targets.update({str(key): str(value) for key, value in payload.items()})
    for generator_name in GENERATOR_NAMES:
        env_name = f"{generator_name.upper()}_GENERATOR_TARGET"
        target = os.environ.get(env_name, "")
        if target:
            targets[generator_name] = target
    return targets


def _generator_targets_from_discovery_uri(uri: str) -> dict[str, str]:
    if uri.startswith(("http://", "https://")):
        timeout = float(os.environ.get("GENERATOR_DISCOVERY_TIMEOUT_SECONDS", "30"))
        request = urllib.request.Request(uri, method="GET")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Generator discovery request failed: {exc}") from exc
        return _normalize_discovered_targets(payload)

    target = uri.removeprefix("python://").removeprefix("python:")
    if ":" not in target:
        raise ValueError(
            "GENERATOR_DISCOVERY_URI must be http(s)://... or python://module:function"
        )
    module_name, function_name = target.split(":", 1)
    provider = getattr(importlib.import_module(module_name), function_name, None)
    if not callable(provider):
        raise RuntimeError(f"GENERATOR_DISCOVERY_URI target is not callable: {uri}")
    return _normalize_discovered_targets(provider())


def _normalize_discovered_targets(payload: Any) -> dict[str, str]:
    if isinstance(payload, Mapping) and isinstance(payload.get("targets"), Mapping):
        payload = payload["targets"]
    if not isinstance(payload, Mapping):
        raise ValueError("generator discovery payload must be a target mapping")
    return {str(name): str(target) for name, target in payload.items() if str(target)}


def _crg_requests_refinement(crg_context: dict) -> bool:
    for belief in _current_generation_beliefs(crg_context):
        predicate = str(belief.get("predicate") or "")
        value = str(belief.get("object") or belief.get("object_value") or "").lower()
        if predicate == "validation_status" and value == "failed":
            return True
        if predicate == "critic_verdict" and value == "fail":
            return True
        if predicate == "supply_feasibility" and value == "unavailable":
            return True
    return False


def _selected_generators_from_crg(crg_context: dict) -> list[str]:
    belief = _selected_generators_belief_from_crg(crg_context)
    if belief is None:
        return []
    value = str(belief.get("object") or belief.get("object_value") or "")
    return [
        generator_name.strip()
        for generator_name in value.split(",")
        if generator_name.strip() in GENERATOR_NAMES
    ]


def _selected_generators_belief_from_crg(
    crg_context: Mapping[str, object],
) -> Mapping[str, object] | None:
    selected_beliefs = [
        belief
        for belief in _current_generation_beliefs(crg_context)
        if str(belief.get("predicate") or "") == "selected_generators"
    ]
    if not selected_beliefs:
        return None
    return max(selected_beliefs, key=_crg_belief_order_key)


def _latest_refinement_failure_belief(
    crg_context: Mapping[str, object],
) -> Mapping[str, object] | None:
    failure_values = {
        "validation_status": "failed",
        "critic_verdict": "fail",
        "supply_feasibility": "unavailable",
    }
    failures = [
        belief
        for belief in _current_generation_beliefs(crg_context)
        if str(belief.get("object") or belief.get("object_value") or "").lower()
        == failure_values.get(str(belief.get("predicate") or ""))
    ]
    if not failures:
        return None
    return max(failures, key=_crg_belief_order_key)


def _current_generation_beliefs(
    crg_context: Mapping[str, object],
) -> list[Mapping[str, object]]:
    beliefs = crg_context.get("beliefs")
    current: dict[tuple[str, str], tuple[tuple[int, str], Mapping[str, object]]] = {}
    for belief in beliefs if isinstance(beliefs, list) else []:
        if not isinstance(belief, Mapping):
            continue
        predicate = str(belief.get("predicate") or "")
        if predicate not in {
            "selected_generators",
            "validation_status",
            "critic_verdict",
            "supply_feasibility",
        }:
            continue
        key = (str(belief.get("subject") or ""), predicate)
        order_key = _crg_belief_order_key(belief)
        existing = current.get(key)
        if existing is None or order_key > existing[0]:
            current[key] = (order_key, belief)
    return [current[key][1] for key in sorted(current)]


def _crg_belief_order_key(belief: Mapping[str, object]) -> tuple[int, str]:
    raw_timestamp = belief.get("timestamp_ns")
    try:
        timestamp_ns = 0 if raw_timestamp is None else int(raw_timestamp)
    except (TypeError, ValueError):
        timestamp_ns = 0
    tie_breaker = json.dumps(
        {
            "id": str(belief.get("id") or ""),
            "object_value": str(belief.get("object") or belief.get("object_value") or ""),
            "source_agent": str(belief.get("source_agent") or ""),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return timestamp_ns, tie_breaker


def _route_humu_feedback_from_crg(crg_context: dict) -> list[dict]:
    feedback = []
    for belief in crg_context.get("beliefs", []) or []:
        if not isinstance(belief, dict):
            continue
        if str(belief.get("predicate") or "") != "route_humu_embedding":
            continue
        payload = _json_object_from_belief(belief)
        if payload is None:
            continue
        embedding = payload.get("humu_embedding")
        if not isinstance(embedding, list) or not embedding:
            continue
        record = {
            "route_id": str(payload.get("route_id") or ""),
            "curvature": float(payload.get("curvature", 1.0)),
            "humu_embedding": [float(value) for value in embedding],
        }
        for key in (
            "source",
            "weight",
            "polarity",
            "confidence",
            "evidence_ids",
            "metadata",
        ):
            if key in payload:
                record[key] = payload[key]
        feedback.append(record)
    return feedback


def _with_route_humu_feedback(
    data: dict,
    route_humu_feedback: list[dict],
    jmcg_feedback: dict | None = None,
) -> dict:
    if not route_humu_feedback:
        return data
    next_data = dict(data)
    generator_params = dict(next_data.get("generator_params") or {})
    generator_params["route_humu_feedback"] = json.dumps(
        route_humu_feedback,
        sort_keys=True,
    )
    if jmcg_feedback:
        generator_params["jmcg_feedback"] = json.dumps(
            jmcg_feedback,
            sort_keys=True,
        )
    next_data["generator_params"] = generator_params
    return next_data


def _jmcg_feedback_envelope(data: dict, route_humu_feedback: list[dict]) -> dict | None:
    run_id = str(data.get("run_id") or data.get("request_id") or "")
    project_id = str(data.get("project_id") or "")
    records = _existing_jmcg_feedback_records(data)
    records.extend(_jmcg_route_feedback_record(record, run_id) for record in route_humu_feedback)
    if not records:
        return None
    return {
        "schema": "moleculeforge.jmcg.feedback.v1",
        "run_id": run_id,
        "project_id": project_id,
        "records": records,
    }


def _existing_jmcg_feedback_records(data: dict) -> list[dict]:
    generator_params = dict(data.get("generator_params") or {})
    payload = generator_params.get("jmcg_feedback")
    if payload in (None, "", b""):
        return []
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8")
    if isinstance(payload, str):
        payload = json.loads(payload)
    if not isinstance(payload, Mapping):
        return []
    records = payload.get("records")
    if not isinstance(records, list):
        return []
    return [dict(record) for record in records if isinstance(record, Mapping)]


def _jmcg_route_feedback_record(record: dict, run_id: str) -> dict:
    route_id = str(record.get("route_id") or "")
    return {
        "kind": "route",
        "source": str(record.get("source") or "generator_coord"),
        "run_id": run_id,
        "subject": {"type": "route", "id": route_id},
        "humu_embedding": list(record.get("humu_embedding") or []),
        "curvature": float(record.get("curvature", 1.0)),
        "weight": float(record.get("weight", 1.0)),
        "polarity": str(record.get("polarity") or "attract"),
        "confidence": float(record.get("confidence", 1.0)),
        "evidence_ids": _feedback_evidence_ids(record.get("evidence_ids")),
        "metadata": _feedback_metadata(record.get("metadata")),
    }


def _feedback_evidence_ids(value: object) -> list[str]:
    if value in (None, "", b""):
        return []
    if isinstance(value, bytes):
        return [value.decode("utf-8")]
    if isinstance(value, str):
        return [value]
    if isinstance(value, list | tuple | set):
        return [str(item) for item in value]
    return [str(value)]


def _feedback_metadata(value: object) -> dict:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): item for key, item in value.items()}


def _json_object_from_belief(belief: dict) -> dict | None:
    raw_value = belief.get("object_value", belief.get("object"))
    if isinstance(raw_value, dict):
        return raw_value
    if not isinstance(raw_value, str):
        return None
    try:
        payload = json.loads(raw_value)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _generator_proto_request(request: dict) -> generator_pb2.GenerateRequest:
    project_id = _required_string(request, "project_id")
    request_id = _required_string(request, "request_id")
    cig, hciv, cone = _typed_generation_context(request)
    _validate_cig_project(cig, project_id)
    objectives = request.get("objectives") or {}
    if not isinstance(objectives, Mapping):
        raise ValueError("objectives must be a mapping")
    raw_generator_params = request.get("generator_params") or {}
    if not isinstance(raw_generator_params, Mapping):
        raise ValueError("generator_params must be a mapping")
    generator_params = {str(key): str(value) for key, value in raw_generator_params.items()}
    generator_params.setdefault("generator", str(request.get("generator", "")))
    batch_size_value = request.get("batch_size")
    if batch_size_value is None:
        batch_size_value = request.get("n_samples")
    batch_size = _positive_int(batch_size_value, "batch_size")
    total_molecules = (
        batch_size
        if request.get("n_samples") is None
        else _positive_int(request["n_samples"], "n_samples")
    )
    timeout_seconds = (
        0
        if request.get("timeout_seconds") is None
        else _non_negative_int(request["timeout_seconds"], "timeout_seconds")
    )
    proto_request = generator_pb2.GenerateRequest(
        project_id=project_id,
        batch_size=batch_size,
        total_molecules=total_molecules,
        intent_cone=cone.SerializeToString(deterministic=True),
        target_properties=[str(key) for key in objectives if str(key) != "complexity"],
        property_targets={
            str(key): float(value)
            for key, value in objectives.items()
            if str(key) != "complexity"
            and isinstance(value, int | float)
            and not isinstance(value, bool)
            and math.isfinite(float(value))
        },
        generator_params=generator_params,
        timeout_seconds=timeout_seconds,
        request_id=request_id,
        context_schema_version=_GENERATOR_CONTEXT_SCHEMA,
    )
    proto_request.cig.CopyFrom(cig)
    proto_request.hciv.CopyFrom(hciv)
    return proto_request


async def _invoke_generator_client(
    client: Any,
    request: generator_pb2.GenerateRequest,
) -> Any:
    if hasattr(client, "generate"):
        result = client.generate(request)
    elif callable(client):
        result = client(request)
    else:
        raise TypeError("generator client must expose generate(request) or be callable")
    if inspect.isawaitable(result):
        return await result
    return result
