"""RetroSyn Agent - 3-layer retrosynthesis planning agent (Agent-3)."""

import asyncio
import inspect
import json
import os
import shlex
import struct
import subprocess
from collections.abc import Mapping
from typing import Any

from google.protobuf.json_format import MessageToDict
from mf_agents.base.agent import (
    BaseAgent,
    agent_health_check_timeout_seconds,
    close_owned_channel,
    ensure_default_event_loop,
    run_health_probe_in_daemon,
)
from mf_agents.crg.graph import ChemicalReasoningGraph
from mf_core.artifacts import CommandRequirement, check_command, require_available
from mf_core.db.repositories import build_shared_crg_repository_from_env
from mf_core.geometry import normalize_lorentz_embedding
from mf_core.proto_gen.moleculeforge.v1.humu import encoder_pb2, encoder_pb2_grpc
from mf_core.proto_gen.moleculeforge.v1.retrosyn import retrosyn_pb2, retrosyn_pb2_grpc
from mf_retrosyn._route_validation import (
    RetrosynRouteTypeError,
    RetrosynRouteValueError,
    partition_retrosyn_results,
)

_PLANNER_COMMAND = CommandRequirement(
    "retrosyn_planner_command",
    "RETROSYN_PLANNER_COMMAND",
    required=False,
)
_NAMED_PLANNER_COMMAND_ENVS = (
    ("rascore", "RASCORE_PLANNER_COMMAND"),
    ("rsgpt", "RSGPT_PLANNER_COMMAND"),
    ("ualign", "UALIGN_PLANNER_COMMAND"),
    ("aizynth", "AIZYNTH_PLANNER_COMMAND"),
)
_NAMED_PLANNER_COMMAND_REQUIREMENTS = {
    engine: CommandRequirement(
        f"retrosyn_{engine}_planner_command",
        env_name,
        required=False,
    )
    for engine, env_name in _NAMED_PLANNER_COMMAND_ENVS
}
_SYNTHETIC_VALIDATION_MARKER = "synthetic_pipeline_validation_only"


class RetrosynGrpcClient:
    def __init__(self, target: str):
        target = str(target).strip()
        if not target:
            raise ValueError("retrosynthesis service target is required")
        import grpc

        self.target = target
        ensure_default_event_loop()
        self.channel = grpc.aio.insecure_channel(target)
        self.stub = retrosyn_pb2_grpc.RetrosynServiceStub(self.channel)
        self.timeout = float(os.getenv("RETROSYN_SERVICE_TIMEOUT_SECONDS", "300"))
        if self.timeout <= 0:
            raise ValueError("RETROSYN_SERVICE_TIMEOUT_SECONDS must be positive")
        self._closed = False

    async def plan_routes(
        self,
        smiles: str,
        *,
        max_routes: int,
        request_context: Mapping[str, Any],
    ) -> dict:
        identity = _required_remote_identity(request_context, smiles)
        engine = _required_identity_text(request_context.get("engine"), "engine")
        request = retrosyn_pb2.RetrosynthesisRequest(
            project_id=identity["project_id"],
            request_id=identity["request_id"],
            run_id=identity["run_id"],
            candidate_id=identity["candidate_id"],
            candidate_index=identity["candidate_index"],
            canonical_smiles=identity["canonical_smiles"],
            molecule_smiles=identity["canonical_smiles"],
            max_routes=max_routes,
            engine=engine,
            include_building_blocks=True,
        )
        response = await self.stub.FindRoutes(request, timeout=self.timeout)
        return _retrosyn_response_payload(response, identity)

    async def health_check(self) -> dict[str, bool]:
        try:
            await asyncio.wait_for(
                self.channel.channel_ready(),
                timeout=agent_health_check_timeout_seconds(),
            )
        except Exception:
            return {"healthy": False}
        return {"healthy": True}

    async def close(self) -> None:
        await close_owned_channel(self, self.channel)


class HUMURouteEncoderGrpcClient:
    def __init__(self, target: str):
        import grpc

        ensure_default_event_loop()
        self.channel = grpc.aio.insecure_channel(target)
        self.stub = encoder_pb2_grpc.HUMUEncoderServiceStub(self.channel)
        self._closed = False

    async def encode_route(self, route: dict) -> dict:
        response = await self.stub.Encode(
            encoder_pb2.EncodeRequest(
                entity_type="route",
                input_data=json.dumps(_route_encoder_payload(route)).encode(),
            )
        )
        return {
            "humu_embedding": _float32_embedding_from_bytes(response.humu_embedding),
            "curvature": float(response.curvature),
        }

    async def health_check(self) -> dict[str, bool]:
        response = await self.stub.Encode(
            encoder_pb2.EncodeRequest(
                entity_type="route",
                input_data=json.dumps(
                    {
                        "target_smiles": "C",
                        "reactions": ["C>>C"],
                    },
                    sort_keys=True,
                ).encode(),
            ),
            timeout=agent_health_check_timeout_seconds(),
        )
        try:
            embedding = _float32_embedding_from_bytes(response.humu_embedding)
            curvature = float(response.curvature)
        except (AttributeError, TypeError, ValueError):
            return {"healthy": False}
        return {
            "healthy": normalize_lorentz_embedding(
                embedding,
                expected_dim=129,
                curvature=curvature,
            )
            is not None
        }

    async def close(self) -> None:
        await close_owned_channel(self, self.channel)


class ExternalCommandRetrosynPlanner:
    def __init__(
        self,
        command: str,
        engine: str = "external_command",
        command_requirement: CommandRequirement = _PLANNER_COMMAND,
    ):
        self.command = command
        self.engine = engine or "external_command"
        self.command_requirement = command_requirement
        self.timeout = float(os.getenv("RETROSYN_PLANNER_COMMAND_TIMEOUT_SECONDS", "300"))

    async def find_routes(self, smiles: str, max_routes: int = 10) -> list[dict]:
        return await self._find_routes(
            smiles,
            max_routes=max_routes,
            timeout=self.timeout,
        )

    async def _find_routes(
        self,
        smiles: str,
        *,
        max_routes: int,
        timeout: float,
    ) -> list[dict]:
        payload = {
            "smiles": smiles,
            "max_routes": max_routes,
            "engine": self.engine,
        }
        result = await asyncio.to_thread(self._run, payload, timeout)
        routes = result.get("routes", result)
        if not isinstance(routes, list):
            raise RuntimeError("RETROSYN_PLANNER_COMMAND must return routes as a list")
        for route in routes:
            if not isinstance(route, dict):
                raise RuntimeError("RETROSYN_PLANNER_COMMAND routes must be JSON objects")
            route.setdefault("source_engine", self.engine)
        return routes[:max_routes]

    def _run(self, payload: dict, timeout: float) -> dict | list:
        _require_command_available(self.command_requirement, self.command)
        completed = subprocess.run(
            shlex.split(self.command),
            input=json.dumps(payload, sort_keys=True),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        if completed.returncode != 0:
            stderr = completed.stderr.strip()
            raise RuntimeError(f"RETROSYN_PLANNER_COMMAND failed: {stderr}")
        try:
            parsed = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError("RETROSYN_PLANNER_COMMAND returned invalid JSON") from exc
        if not isinstance(parsed, dict | list):
            raise RuntimeError("RETROSYN_PLANNER_COMMAND must return a JSON object or list")
        return parsed

    async def health_check(self) -> dict[str, bool]:
        routes = await self._find_routes(
            "C",
            max_routes=1,
            timeout=agent_health_check_timeout_seconds(),
        )
        return {"healthy": isinstance(routes, list)}


class _PlannerHealthTarget:
    def __init__(self, planner: Any) -> None:
        self.planner = planner

    @property
    def _close_target(self) -> Any:
        return self.planner

    async def health_check(self) -> dict[str, bool]:
        routes = await run_health_probe_in_daemon(lambda: _run_planner_health_probe(self.planner))
        return {"healthy": isinstance(routes, list)}

    async def close(self) -> None:
        close = getattr(self.planner, "close", None)
        if not callable(close):
            return
        result = close()
        if inspect.isawaitable(result):
            await result


def _run_planner_health_probe(planner: Any) -> list[dict]:
    return asyncio.run(_find_routes_with_planner(planner, "C", 1))


def _planner_health_target(planner: Any) -> Any:
    if planner is None or callable(getattr(planner, "health_check", None)):
        return planner
    if callable(getattr(planner, "find_routes", None)):
        return _PlannerHealthTarget(planner)
    return planner


class RetroSynAgent(BaseAgent):
    def __init__(
        self,
        message_bus=None,
        planner=None,
        route_planners: dict[str, Any] | None = None,
        retrosyn_client=None,
        retrosyn_target: str | None = None,
        route_encoder_client=None,
        route_encoder_target: str | None = None,
        crg_repository: Any = None,
    ):
        super().__init__("retrosyn_agent", message_bus)
        self._subscription_subjects = ["agent.retrosyn.request", "orchestrator.retrosyn.plan"]
        self.crg = ChemicalReasoningGraph()
        explicit_planner = planner is not None or route_planners is not None
        self.retrosyn_client = retrosyn_client or (
            None if explicit_planner else _build_retrosyn_client(retrosyn_target)
        )
        self.planner = planner
        self.route_planners = (
            dict(route_planners)
            if route_planners is not None
            else ({} if self.retrosyn_client is not None else _route_planners_from_env())
        )
        self.planner_command = os.getenv("RETROSYN_PLANNER_COMMAND", "").strip()
        self.route_encoder_client = route_encoder_client or _build_route_encoder_client(
            route_encoder_target
        )
        if crg_repository is None:
            self.crg_repository = build_shared_crg_repository_from_env()
            self._owns_crg_repository = self.crg_repository is not None
        else:
            self.crg_repository = crg_repository
            self._owns_crg_repository = False

    def runtime_targets(self) -> dict[str, Any]:
        targets: dict[str, Any] = {
            "route_encoder": self.route_encoder_client,
        }
        if self.retrosyn_client is not None:
            targets["retrosyn_service"] = self.retrosyn_client
        elif self.route_planners:
            targets.update(
                {
                    f"planner.{name}": _planner_health_target(planner)
                    for name, planner in self.route_planners.items()
                }
            )
        elif self.planner is not None:
            targets["planner"] = _planner_health_target(self.planner)
        elif self.planner_command:
            targets["planner"] = ExternalCommandRetrosynPlanner(self.planner_command)
        elif os.environ.get("AIZYNTH_CONFIG_PATH", "").strip():
            targets["planner"] = _planner_health_target(self._planner())
        else:
            targets["planner"] = None
        if self._owns_crg_repository:
            targets["crg_repository"] = self.crg_repository
        return targets

    async def process(self, data):
        """Plan 3-layer retrosynthesis: strategy -> pathway -> reaction.

        Layer 1: Strategic disconnections and route planning
        Layer 2: Pathway enumeration and ranking
        Layer 3: Individual reaction feasibility and condition selection
        """
        target_smiles = data.get("smiles", "")
        if (
            not isinstance(target_smiles, str)
            or not target_smiles
            or not target_smiles.strip()
            or target_smiles != target_smiles.strip()
        ):
            raise ValueError("smiles is required for retrosynthesis planning")
        raw_max_routes = data.get("max_routes", 10)
        if (
            isinstance(raw_max_routes, bool)
            or not isinstance(raw_max_routes, int)
            or raw_max_routes <= 0
        ):
            raise ValueError("max_routes must be a positive integer")
        max_routes = raw_max_routes
        run_id = str(data.get("run_id") or data.get("request_id") or "")
        candidate_reference = _candidate_reference(data, target_smiles)
        routes, assessments, strategy = await self._find_routes(
            target_smiles,
            max_routes,
            data,
        )
        routes = await self._attach_route_embeddings(routes)
        belief = self.crg.add_belief(
            subject=target_smiles,
            predicate="retrosyn_routes",
            obj=str(len(routes)),
            confidence=1.0,
            source_agent=self.name,
            evidence_ids=[
                str(route["route_id"])
                for route in routes
                if isinstance(route, dict) and route.get("route_id")
            ],
        )
        beliefs_to_persist = [belief]
        for route in routes:
            if not isinstance(route, dict) or "humu_embedding" not in route:
                continue
            route_id = str(route.get("route_id") or "")
            route_belief = self.crg.add_belief(
                subject=target_smiles,
                predicate="route_humu_embedding",
                obj=json.dumps(
                    {
                        "curvature": float(route.get("humu_curvature", 1.0)),
                        "humu_embedding": route["humu_embedding"],
                        "route_id": route_id,
                    },
                    sort_keys=True,
                ),
                confidence=1.0,
                source_agent=self.name,
                evidence_ids=[route_id] if route_id else [],
            )
            beliefs_to_persist.append(route_belief)
        project_id = candidate_reference["project_id"]
        for belief in beliefs_to_persist:
            await self._persist_belief(
                belief,
                project_id=project_id,
                run_id=run_id,
            )
        return {
            "agent": self.name,
            "status": "planned",
            "target_smiles": target_smiles,
            **candidate_reference,
            "routes": routes,
            "assessments": assessments,
            "layers": {
                "strategy": {
                    **strategy,
                    "route_count": len(routes),
                    "assessment_count": len(assessments),
                },
                "pathways": routes,
                "reactions": _route_reactions(routes),
            },
        }

    async def _find_routes(
        self,
        target_smiles: str,
        max_routes: int,
        request_context: Mapping[str, Any],
    ) -> tuple[list[dict], list[dict], dict]:
        if self.retrosyn_client is not None:
            engine = _required_identity_text(request_context.get("engine"), "engine")
            response = await self.retrosyn_client.plan_routes(
                target_smiles,
                max_routes=max_routes,
                request_context={
                    **request_context,
                    "engine": engine,
                },
            )
            return (
                response["routes"],
                response["assessments"],
                {"engine": "retrosyn_service", "target": self.retrosyn_client.target},
            )
        if self.route_planners:
            requested_engine = _required_identity_text(
                request_context.get("engine"),
                "engine",
            )
            if requested_engine == "ensemble":
                selected_planners = self.route_planners
            else:
                planner = self.route_planners.get(requested_engine)
                if planner is None:
                    configured = ", ".join(sorted(self.route_planners))
                    raise ValueError(
                        f"engine must be ensemble or one of configured planners: {configured}"
                    )
                selected_planners = {requested_engine: planner}
            results: list[dict] = []
            for engine, planner in selected_planners.items():
                engine_routes = await _find_routes_with_planner(
                    planner,
                    target_smiles,
                    max_routes,
                )
                for route in engine_routes:
                    route_with_engine = dict(route)
                    route_with_engine.setdefault("source_engine", engine)
                    results.append(route_with_engine)
            routes, assessments = partition_retrosyn_results(
                results,
                "retrosynthesis planner ensemble",
            )
            ranked = _rank_routes(_dedupe_routes(routes))[:max_routes]
            if requested_engine != "ensemble":
                return (
                    ranked,
                    assessments,
                    {"engine": requested_engine},
                )
            return (
                ranked,
                assessments,
                {
                    "engine": "ensemble",
                    "engines": list(self.route_planners.keys()),
                },
            )
        planner = self._planner()
        results = await _find_routes_with_planner(
            planner,
            target_smiles,
            max_routes,
        )
        routes, assessments = partition_retrosyn_results(
            results,
            planner.__class__.__name__,
        )
        return (
            _rank_routes(_dedupe_routes(routes))[:max_routes],
            assessments,
            {"engine": planner.__class__.__name__},
        )

    def _planner(self):
        if self.planner is None:
            if self.planner_command:
                self.planner = ExternalCommandRetrosynPlanner(self.planner_command)
                return self.planner
            from mf_retrosyn.aizynth.retrosyn import AiZynthRetrosyn

            self.planner = AiZynthRetrosyn.from_env()
        return self.planner

    async def _attach_route_embeddings(self, routes: list[dict]) -> list[dict]:
        if self.route_encoder_client is None:
            return routes
        encoded_routes = []
        for route in routes:
            if not isinstance(route, dict):
                encoded_routes.append(route)
                continue
            embedding_record = await _encode_route(self.route_encoder_client, route)
            route_with_embedding = dict(route)
            route_with_embedding["humu_embedding"] = embedding_record["humu_embedding"]
            route_with_embedding["humu_curvature"] = embedding_record["curvature"]
            encoded_routes.append(route_with_embedding)
        return encoded_routes

    async def _persist_belief(self, belief, project_id: str, run_id: str) -> None:
        if self.crg_repository is None:
            return
        write_belief = getattr(self.crg_repository, "write_workflow_belief", None)
        if not callable(write_belief):
            raise TypeError("crg_repository must expose write_workflow_belief(**kwargs)")
        result = write_belief(
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
        if inspect.isawaitable(result):
            await result


def _build_retrosyn_client(retrosyn_target: str | None):
    target = retrosyn_target or os.environ.get("RETROSYN_SERVICE_TARGET", "")
    return RetrosynGrpcClient(target) if target else None


def _candidate_reference(
    data: Mapping[str, Any],
    target_smiles: str,
) -> dict[str, Any]:
    canonical_smiles = data.get("canonical_smiles", target_smiles)
    canonical_smiles = _required_identity_text(canonical_smiles, "canonical_smiles")
    if canonical_smiles != target_smiles:
        raise ValueError("canonical_smiles must equal smiles")
    reference: dict[str, Any] = {
        "project_id": _optional_identity_text(data.get("project_id"), "project_id"),
        "candidate_id": _optional_identity_text(data.get("candidate_id"), "candidate_id"),
        "canonical_smiles": canonical_smiles,
    }
    if "candidate_index" in data:
        reference["candidate_index"] = _candidate_index(data["candidate_index"])
    return reference


def _required_remote_identity(
    context: Mapping[str, Any],
    smiles: str,
) -> dict[str, Any]:
    identity = {
        "project_id": _required_identity_text(context.get("project_id"), "project_id"),
        "run_id": _required_identity_text(context.get("run_id"), "run_id"),
        "request_id": _required_identity_text(context.get("request_id"), "request_id"),
        "candidate_id": _required_identity_text(
            context.get("candidate_id"),
            "candidate_id",
        ),
        "candidate_index": _candidate_index(context.get("candidate_index")),
        "canonical_smiles": _required_identity_text(
            context.get("canonical_smiles"),
            "canonical_smiles",
        ),
    }
    if identity["canonical_smiles"] != smiles:
        raise ValueError("canonical_smiles must equal smiles")
    return identity


def _required_identity_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value or not value.strip() or value != value.strip():
        raise ValueError(f"{field} must be a non-empty trimmed string")
    return value


def _optional_identity_text(value: Any, field: str) -> str:
    if value in (None, ""):
        return ""
    return _required_identity_text(value, field)


def _candidate_index(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("candidate_index must be a non-negative integer")
    return value


def _retrosyn_response_payload(
    response: Any,
    expected_identity: Mapping[str, Any],
) -> dict[str, Any]:
    for field in (
        "request_id",
        "project_id",
        "run_id",
        "candidate_id",
        "canonical_smiles",
    ):
        actual = str(getattr(response, field, "") or "")
        expected = expected_identity[field]
        if actual != expected:
            raise RuntimeError(
                f"retrosynthesis service response {field} mismatch: "
                f"expected {expected!r}, got {actual!r}"
            )
    if (
        not response.HasField("candidate_index")
        or response.candidate_index != expected_identity["candidate_index"]
    ):
        raise RuntimeError("retrosynthesis service response candidate_index mismatch")
    route_records = [_route_from_proto(route) for route in response.routes]
    routes, route_assessments = partition_retrosyn_results(
        route_records,
        "retrosynthesis service response",
    )
    if route_assessments:
        raise RetrosynRouteValueError(
            "retrosynthesis service returned assessments as executable routes"
        )
    if int(response.total_routes_found) != len(routes):
        raise RuntimeError(
            "retrosynthesis service response total_routes_found does not match routes"
        )
    assessment_records = [_assessment_from_proto(assessment) for assessment in response.assessments]
    assessment_routes, assessments = partition_retrosyn_results(
        assessment_records,
        "retrosynthesis service assessments",
    )
    if assessment_routes:
        raise RetrosynRouteValueError(
            "retrosynthesis service assessment payload contains executable routes"
        )
    return {
        **dict(expected_identity),
        "routes": routes,
        "assessments": assessments,
        "total_routes_found": len(routes),
        "elapsed_ms": int(response.elapsed_ms),
    }


def _route_from_proto(route: Any) -> dict[str, Any]:
    _reject_synthetic_validation_result(
        {"source_engine": str(route.source_engine)},
    )
    steps = [_route_step_from_proto(step) for step in route.steps]
    building_blocks = (
        [_struct_to_dict(block) for block in route.building_block_records]
        if route.building_block_records
        else list(route.building_blocks)
    )
    record: dict[str, Any] = {
        "route_id": str(route.route_id),
        "reaction_smiles": list(route.reaction_smiles),
        "predicted_score": float(route.predicted_score),
        "predicted_yield": float(route.predicted_yield),
        "n_steps": int(route.n_steps),
        "building_blocks": building_blocks,
        "estimated_cost_usd_per_g": float(route.estimated_cost_usd_per_g),
        "all_commercially_available": bool(route.all_commercially_available),
        "steps": steps,
    }
    if route.source_engine:
        record["source_engine"] = str(route.source_engine)
    if route.route_type:
        record["route_type"] = str(route.route_type)
    return record


def _route_step_from_proto(step: Any) -> dict[str, Any]:
    if not step.HasField("yield_fraction"):
        raise RetrosynRouteValueError(
            f"retrosynthesis service response step {step.step_id!r} is missing yield_fraction"
        )
    record: dict[str, Any] = {
        "step_id": str(step.step_id),
        "reaction": str(step.reaction),
        "reaction_type": str(step.reaction_type),
        "reactants": [_struct_to_dict(reactant) for reactant in step.reactants],
        "conditions": _struct_to_dict(step.conditions),
        "yield": float(step.yield_fraction),
        "building_blocks": [
            _struct_to_dict(building_block) for building_block in step.building_blocks
        ],
    }
    if step.reagents:
        record["reagents"] = list(step.reagents)
    if step.purification:
        record["purification"] = str(step.purification)
    if step.operation:
        record["operation"] = str(step.operation)
    return record


def _assessment_from_proto(assessment: Any) -> dict[str, Any]:
    details = _struct_to_dict(assessment.details)
    expected = {
        "route_id": str(assessment.assessment_id),
        "route_type": str(assessment.assessment_type),
        "source_engine": str(assessment.source_engine),
        "score": float(assessment.score),
    }
    for field, value in expected.items():
        if field in details and details[field] != value:
            raise RuntimeError(f"retrosynthesis service assessment {field} conflicts with details")
        details[field] = value
    _reject_synthetic_validation_result(details)
    return details


def _struct_to_dict(value: Any) -> dict[str, Any]:
    return MessageToDict(value, preserving_proto_field_name=True)


def _route_reactions(routes: list[dict]) -> list[str]:
    reactions: list[str] = []
    for route in routes:
        for step in route.get("steps") or []:
            reaction = step.get("reaction") if isinstance(step, dict) else None
            if isinstance(reaction, str) and reaction:
                reactions.append(reaction)
    return reactions


def _route_planners_from_env() -> dict[str, ExternalCommandRetrosynPlanner]:
    raw = os.getenv("RETROSYN_PLANNER_COMMANDS_JSON", "").strip()
    if not raw:
        return _named_route_planners_from_env()
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("RETROSYN_PLANNER_COMMANDS_JSON must be a JSON object")
    planners = {}
    for engine, command in payload.items():
        engine_name = str(engine).strip()
        command_value = str(command).strip()
        if not engine_name:
            raise ValueError("RETROSYN_PLANNER_COMMANDS_JSON engine names must be non-empty")
        if not command_value:
            raise ValueError("RETROSYN_PLANNER_COMMANDS_JSON commands must be non-empty")
        planners[engine_name] = ExternalCommandRetrosynPlanner(
            command_value,
            engine=engine_name,
            command_requirement=CommandRequirement(
                f"retrosyn_{engine_name}_planner_command",
                "RETROSYN_PLANNER_COMMANDS_JSON",
                required=False,
            ),
        )
    return planners


def _named_route_planners_from_env() -> dict[str, ExternalCommandRetrosynPlanner]:
    planners = {}
    for engine_name, env_name in _NAMED_PLANNER_COMMAND_ENVS:
        command_value = os.getenv(env_name, "").strip()
        if not command_value:
            continue
        planners[engine_name] = ExternalCommandRetrosynPlanner(
            command_value,
            engine=engine_name,
            command_requirement=_NAMED_PLANNER_COMMAND_REQUIREMENTS[engine_name],
        )
    return planners


def _require_command_available(
    requirement: CommandRequirement,
    command: str,
) -> None:
    required_requirement = CommandRequirement(
        requirement.name,
        requirement.env_var,
        required=True,
    )
    env = {**os.environ, requirement.env_var: command}
    require_available([check_command(required_requirement, env=env)])


async def _find_routes_with_planner(
    planner: Any,
    target_smiles: str,
    max_routes: int,
) -> list[dict]:
    result = planner.find_routes(target_smiles, max_routes=max_routes)
    if inspect.isawaitable(result):
        result = await result
    if not isinstance(result, list):
        raise RetrosynRouteTypeError("retrosynthesis planner must return a list of route dicts")
    for route in result:
        if not isinstance(route, dict):
            raise RetrosynRouteTypeError("retrosynthesis planner routes must be dictionaries")
        _reject_synthetic_validation_result(route)
    return result


def _reject_synthetic_validation_result(record: Mapping[str, Any]) -> None:
    if any(
        record.get(field) == _SYNTHETIC_VALIDATION_MARKER
        for field in ("source_engine", "validation_marker")
    ):
        raise RetrosynRouteValueError(
            "synthetic validation retrosynthesis result cannot satisfy a business request"
        )


def _dedupe_routes(routes: list[dict]) -> list[dict]:
    seen: set[str] = set()
    unique: list[dict] = []
    for route in routes:
        key = _route_key(route)
        if key in seen:
            continue
        seen.add(key)
        unique.append(route)
    return unique


def _route_key(route: dict) -> str:
    route_id = route.get("route_id")
    if isinstance(route_id, str) and route_id:
        return f"id:{route_id}"
    reactions = _route_reactions([route])
    if reactions:
        return "reactions:" + "|".join(reactions)
    return json.dumps(route, sort_keys=True, default=str)


def _rank_routes(routes: list[dict]) -> list[dict]:
    indexed = list(enumerate(routes))
    indexed.sort(key=lambda item: (_route_priority(item[1]), -_route_score(item[1]), item[0]))
    return [route for _, route in indexed]


def _route_priority(route: dict) -> int:
    if str(route.get("route_type") or "") == "retrosynthetic_accessibility_score":
        return 2
    if _route_has_building_blocks(route):
        return 0
    return 1


def _route_has_building_blocks(route: dict) -> bool:
    blocks = route.get("building_blocks")
    if isinstance(blocks, list) and any(_block_smiles(block) for block in blocks):
        return True
    for step in route.get("steps") or []:
        if not isinstance(step, dict):
            continue
        for key in ("building_blocks", "reactants"):
            values = step.get(key)
            if isinstance(values, list) and any(_block_smiles(value) for value in values):
                return True
    return False


def _block_smiles(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return str(value.get("smiles") or value.get("building_block_smiles") or "")
    return ""


def _route_score(route: dict) -> float:
    for key in (
        "score",
        "predicted_score",
        "route_score",
        "estimated_yield",
        "predicted_yield",
    ):
        value = route.get(key)
        if isinstance(value, int | float):
            return float(value)
    return 0.0


def _build_route_encoder_client(route_encoder_target: str | None):
    target = route_encoder_target or os.environ.get("HUMU_ENCODER_TARGET", "")
    return HUMURouteEncoderGrpcClient(target) if target else None


async def _encode_route(client: Any, route: dict) -> dict:
    if hasattr(client, "encode_route"):
        result = client.encode_route(route)
    elif callable(client):
        result = client(route)
    else:
        raise TypeError("route encoder client must expose encode_route(route) or be callable")
    if inspect.isawaitable(result):
        result = await result
    return _normalise_route_embedding(result)


def _normalise_route_embedding(result: object) -> dict:
    if isinstance(result, dict):
        embedding = result.get("humu_embedding", result.get("embedding"))
        curvature = float(result.get("curvature", 1.0))
    else:
        embedding = result
        curvature = 1.0
    if isinstance(embedding, bytes):
        embedding = _float32_embedding_from_bytes(embedding)
    if not isinstance(embedding, list) or not embedding:
        raise ValueError("route encoder result requires a non-empty humu_embedding")
    return {
        "humu_embedding": [float(value) for value in embedding],
        "curvature": curvature,
    }


def _route_encoder_payload(route: dict) -> dict:
    payload = dict(route)
    reactions = payload.get("reactions")
    if not isinstance(reactions, list) or not reactions:
        payload["reactions"] = _route_reactions([route])
    if not payload["reactions"]:
        raise ValueError("route encoder requires at least one reaction")
    return payload


def _float32_embedding_from_bytes(payload: bytes) -> list[float]:
    if len(payload) % 4 != 0:
        raise ValueError("route HUMU embedding bytes must contain float32 values")
    return [float(item[0]) for item in struct.iter_unpack("<f", payload)]
