"""Generator Coordinator Agent - Coordinates multiple generators based on routing (Agent-2)."""

import asyncio
import importlib
import inspect
import json
import os
import shlex
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from typing import Any

from mf_agents.base.agent import (
    BaseAgent,
    agent_health_check_timeout_seconds,
    ensure_default_event_loop,
)
from mf_agents.crg.graph import ChemicalReasoningGraph
from mf_core.artifacts import CommandRequirement, check_command, require_available
from mf_core.db.repositories import build_shared_crg_repository_from_env
from mf_core.proto_gen.moleculeforge.v1.generator import generator_pb2, generator_pb2_grpc
from mf_core.routing.task_router import GENERATOR_NAMES

DEFAULT_GENERATORS = ["hfm_3d", "fragfm"]
if not set(DEFAULT_GENERATORS).issubset(GENERATOR_NAMES):
    raise RuntimeError("Default generators must be present in GENERATOR_NAMES")
REFINEMENT_GENERATORS = ["mmpt_rag", "fragfm"]
if not set(REFINEMENT_GENERATORS).issubset(GENERATOR_NAMES):
    raise RuntimeError("Refinement generators must be present in GENERATOR_NAMES")
_UAS_RUNNER_COMMAND = CommandRequirement("uas_runner_command", "UAS_RUNNER_COMMAND")


class GeneratorGrpcClient:
    def __init__(self, target: str):
        import grpc

        self.target = target
        ensure_default_event_loop()
        self.channel = grpc.aio.insecure_channel(target)
        self.stub = generator_pb2_grpc.GeneratorServiceStub(self.channel)

    async def generate(self, request: dict) -> dict:
        response = await self.stub.Generate(_generator_proto_request(request))
        return {
            "generator_name": response.generator_name,
            "generation_id": response.generation_id,
            "candidates": [_decode_molecule_payload(item) for item in response.molecules],
            "aggregate_stats": {
                str(key): float(value) for key, value in response.aggregate_stats.items()
            },
            "elapsed_ms": int(response.elapsed_ms),
        }

    async def health_check(self) -> dict:
        response = await self.stub.Info(
            generator_pb2.GeneratorInfo(),
            timeout=agent_health_check_timeout_seconds(),
        )
        generator_name = str(response.name or "")
        if not generator_name:
            return {
                "healthy": False,
                "reason": "generator info response missing name",
            }
        return {
            "healthy": True,
            "generator_name": generator_name,
            "version": str(response.version or ""),
            "requires_gpu": bool(response.requires_gpu),
        }


class UASLocalGeneratorClient:
    def __init__(self, command: str | None = None):
        self.command = (command or os.environ.get("UAS_RUNNER_COMMAND", "")).strip()
        self.timeout_seconds = float(os.environ.get("UAS_RUNNER_TIMEOUT_SECONDS", "30"))

    async def health_check(self) -> dict:
        if not self.command:
            return {
                "healthy": False,
                "reason": "UAS_RUNNER_COMMAND is required",
            }
        status = _command_status(_UAS_RUNNER_COMMAND, self.command)
        if not status.available:
            return {
                "healthy": False,
                "reason": status.message,
            }
        await asyncio.to_thread(
            _UASCommandRunner(
                self.command,
                agent_health_check_timeout_seconds(),
            ).generate,
            health_check=True,
            dry_run=True,
            n_samples=0,
        )
        return {"healthy": True, "generator_name": "uas", "version": "0.1.0"}

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
        self.crg_repository = (
            crg_repository if crg_repository is not None else build_shared_crg_repository_from_env()
        )

    def runtime_targets(self) -> Mapping[str, Any]:
        targets = {
            f"generator.{name}": self.generator_clients.get(name) for name in DEFAULT_GENERATORS
        }
        targets.update(
            {f"generator.{name}": client for name, client in self.generator_clients.items()}
        )
        return targets

    async def process(self, data):
        """Route generation request to appropriate generator(s) based on objectives.

        Selects generation strategy based on target properties, complexity,
        and available generators. Dispatches to one or more generator backends.
        """
        strategy = data.get("generation_strategy", "auto")
        objectives = data.get("objectives", {})
        crg_context = await self._read_generation_crg_context(data, strategy, objectives)
        route_humu_feedback = _route_humu_feedback_from_crg(crg_context)
        jmcg_feedback = _jmcg_feedback_envelope(data, route_humu_feedback)
        cached_generators = _selected_generators_from_crg(crg_context)
        if cached_generators:
            selected_generators = cached_generators
            cache_source = "shared_crg"
        else:
            selected_generators = self._select_generators(
                strategy,
                objectives,
                crg_context,
            )
            cache_source = ""
        dispatch_results = []
        candidates = []
        status = "selected"
        if self.generator_clients:
            dispatch_data = _with_route_humu_feedback(
                data,
                route_humu_feedback,
                jmcg_feedback,
            )
            dispatch_results, candidates = await self._dispatch_generators(
                selected_generators,
                dispatch_data,
                objectives,
            )
            status = "dispatched"
        if not cache_source:
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
            "status": status,
            "strategy": strategy,
            "selected_generators": selected_generators,
            "available_generators": self.generators,
            "dispatch_results": dispatch_results,
            "candidates": candidates,
            **({"route_humu_feedback": route_humu_feedback} if route_humu_feedback else {}),
            **({"jmcg_feedback": jmcg_feedback} if jmcg_feedback else {}),
            **({"cache_source": cache_source} if cache_source else {}),
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
        selected_generators: list[str],
        data: dict,
        objectives: dict,
    ) -> tuple[list[dict], list[dict]]:
        dispatch_results = []
        all_candidates = []
        for generator_name in selected_generators:
            client = self.generator_clients.get(generator_name)
            if client is None:
                raise RuntimeError(f"Generator client is not configured: {generator_name}")
            health = await _check_generator_health(client, generator_name)
            request = _generator_request(generator_name, data, objectives)
            result = await _invoke_generator_client(client, request)
            candidates = [
                _candidate_dict(candidate, generator_name)
                for candidate in _result_candidates(result)
            ]
            all_candidates.extend(candidates)
            dispatch_results.append(
                {
                    "generator": generator_name,
                    "candidate_count": len(candidates),
                    "health_status": health["status"],
                }
            )
        return dispatch_results, all_candidates


def _generator_request(generator_name: str, data: dict, objectives: dict) -> dict:
    request = {
        "generator": generator_name,
        "objectives": objectives,
    }
    for key in (
        "project_id",
        "request_id",
        "batch_size",
        "n_samples",
        "hciv",
        "intent_cone",
        "task_profile",
        "generator_params",
    ):
        if key in data:
            request[key] = data[key]
    return request


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
        clients[generator_name] = _generator_client_from_target(str(target))
    return clients


def _generator_client_from_target(target: str) -> Any:
    if target.startswith(("python://", "python:")):
        return _python_target(target)
    return GeneratorGrpcClient(target)


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
    for belief in crg_context.get("beliefs", []) or []:
        if not isinstance(belief, dict):
            continue
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
    for belief in crg_context.get("beliefs", []) or []:
        if not isinstance(belief, dict):
            continue
        if str(belief.get("predicate") or "") != "selected_generators":
            continue
        value = str(belief.get("object") or belief.get("object_value") or "")
        selected = [
            generator_name.strip()
            for generator_name in value.split(",")
            if generator_name.strip() in GENERATOR_NAMES
        ]
        if selected:
            return selected
    return []


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
    objectives = dict(request.get("objectives", {}) or {})
    generator_params = {
        str(key): str(value)
        for key, value in dict(request.get("generator_params", {}) or {}).items()
    }
    generator_params.setdefault("generator", str(request.get("generator", "")))
    batch_size = int(request.get("batch_size") or request.get("n_samples") or 1)
    return generator_pb2.GenerateRequest(
        project_id=str(request.get("project_id") or request.get("request_id") or ""),
        batch_size=batch_size,
        total_molecules=int(request.get("n_samples") or batch_size),
        intent_cone=_intent_cone_bytes(request.get("intent_cone")),
        target_properties=[str(key) for key in objectives.keys()],
        property_targets={
            str(key): float(value)
            for key, value in objectives.items()
            if isinstance(value, int | float)
        },
        generator_params=generator_params,
        timeout_seconds=int(request.get("timeout_seconds") or 0),
    )


def _intent_cone_bytes(value: Any) -> bytes:
    if value in (None, "", b"", {}):
        return b""
    if isinstance(value, bytes):
        return value
    if isinstance(value, str):
        return value.encode("utf-8")
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(value, sort_keys=True).encode("utf-8")


def _decode_molecule_payload(payload: bytes) -> dict:
    text = payload.decode("utf-8")
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError:
        return {"payload": text}
    if isinstance(decoded, dict):
        return decoded
    return {"value": decoded}


async def _invoke_generator_client(client: Any, request: dict) -> Any:
    if hasattr(client, "generate"):
        result = client.generate(request)
    elif callable(client):
        result = client(request)
    else:
        raise TypeError("generator client must expose generate(request) or be callable")
    if inspect.isawaitable(result):
        return await result
    return result


async def _check_generator_health(client: Any, generator_name: str) -> dict[str, str]:
    health_check = getattr(client, "health_check", None)
    if not callable(health_check):
        return {"status": "unchecked"}
    result = health_check()
    if inspect.isawaitable(result):
        result = await result
    if result is None:
        return {"status": "healthy"}
    if not isinstance(result, Mapping):
        raise TypeError("generator health_check() must return a mapping")
    if result.get("healthy") is True:
        return {"status": "healthy"}
    reason = str(result.get("reason") or "health check failed")
    raise RuntimeError(f"Generator client is unhealthy: {generator_name}: {reason}")


def _result_candidates(result: Any) -> list:
    if isinstance(result, dict):
        return list(result.get("candidates", []))
    if isinstance(result, list):
        return result
    molecules = getattr(result, "molecules", None)
    if molecules is not None:
        return list(molecules)
    candidates = getattr(result, "candidates", None)
    if candidates is not None:
        return list(candidates)
    return []


def _candidate_dict(candidate: Any, generator_name: str) -> dict:
    if isinstance(candidate, dict):
        normalized = dict(candidate)
    else:
        normalized = {"value": candidate}
    normalized.setdefault("generator", generator_name)
    return normalized
