"""Retrosynthesis Planning Service - gRPC server for AiZynthFinder + RSGPT scoring."""

import asyncio
import json
import logging
import os
import shlex
import subprocess
import sys
import time
import uuid
from concurrent import futures
from typing import Any

import grpc
from mf_core.artifacts import (
    ArtifactRequirement,
    CommandRequirement,
    RequirementStatus,
    check_artifact,
    check_command,
    require_available,
)
from mf_core.proto_gen.moleculeforge.v1.retrosyn import retrosyn_pb2, retrosyn_pb2_grpc
from mf_retrosyn._route_validation import (
    RetrosynRouteError,
    RetrosynRouteTypeError,
    RetrosynRouteValueError,
    partition_retrosyn_results,
)

_SCORER = ArtifactRequirement(
    "retrosyn_scorer",
    "RETROSYN_SCORER_URI",
    kind="uri",
    required=False,
)
_AIZYNTH_CONFIG = ArtifactRequirement("aizynth_config", "AIZYNTH_CONFIG_PATH", kind="file")
_PLANNER_COMMAND = CommandRequirement(
    "retrosyn_planner_command",
    "RETROSYN_PLANNER_COMMAND",
    required=False,
)
_PLANNER_COMMANDS_JSON_ENV = "RETROSYN_PLANNER_COMMANDS_JSON"
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
_VALIDATION_GATE_ENV = "MF_ALLOW_SYNTHETIC_VALIDATION"
_VALIDATION_MARKER = "synthetic_pipeline_validation_only"
_LOGGER = logging.getLogger(__name__)
_VALIDATION_MAX_ROUTES = 64
_VALIDATION_PRECURSORS = {
    "CCO": ("CO", "C"),
    "CCN": ("CN", "C"),
}


def _status_objects() -> list[RequirementStatus]:
    statuses = [
        check_artifact(_AIZYNTH_CONFIG),
        check_artifact(_SCORER),
        check_command(_PLANNER_COMMAND),
        *_json_planner_command_statuses(),
    ]
    for requirement in _NAMED_PLANNER_COMMAND_REQUIREMENTS.values():
        if os.getenv(requirement.env_var, "").strip():
            statuses.append(check_command(requirement))
    return statuses


def _json_planner_command_statuses() -> list[RequirementStatus]:
    raw = os.getenv(_PLANNER_COMMANDS_JSON_ENV, "").strip()
    if not raw:
        return []
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        return [_invalid_json_planner_command_status(raw, f"invalid JSON: {exc}")]
    if not isinstance(payload, dict):
        return [_invalid_json_planner_command_status(raw, "must be a JSON object")]
    if not payload:
        return [_invalid_json_planner_command_status(raw, "must define at least one planner")]
    statuses: list[RequirementStatus] = []
    for engine, command in payload.items():
        engine_name = str(engine).strip()
        command_value = str(command).strip()
        requirement = CommandRequirement(
            _planner_command_status_name(engine_name),
            _PLANNER_COMMANDS_JSON_ENV,
            required=False,
        )
        if not engine_name:
            statuses.append(
                _invalid_json_planner_command_status(
                    command_value,
                    "engine names must be non-empty",
                )
            )
            continue
        if not command_value:
            statuses.append(
                RequirementStatus(
                    name=requirement.name,
                    configured=True,
                    available=False,
                    required=requirement.required,
                    path=None,
                    source=_PLANNER_COMMANDS_JSON_ENV,
                    message=f"{requirement.name} command is empty",
                )
            )
            continue
        statuses.append(
            check_command(
                requirement,
                env={**os.environ, _PLANNER_COMMANDS_JSON_ENV: command_value},
            )
        )
    return statuses


def _invalid_json_planner_command_status(raw: str, message: str) -> RequirementStatus:
    return RequirementStatus(
        name="retrosyn_external_planner_command",
        configured=True,
        available=False,
        required=False,
        path=raw or None,
        source=_PLANNER_COMMANDS_JSON_ENV,
        message=f"{_PLANNER_COMMANDS_JSON_ENV} {message}",
    )


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


def _require_runtime(*requirements: ArtifactRequirement) -> list[RequirementStatus]:
    statuses = [check_artifact(requirement) for requirement in requirements]
    require_available(statuses)
    return statuses


def _require_planner_runtime() -> list[RequirementStatus]:
    statuses = _status_objects()
    aizynth_status = next(status for status in statuses if status.name == _AIZYNTH_CONFIG.name)
    command_statuses = [status for status in statuses if _is_planner_command_status(status)]
    configured_commands = [status for status in command_statuses if status.configured]
    if configured_commands:
        unavailable_commands = [status for status in configured_commands if not status.available]
        if not unavailable_commands:
            return statuses
        details = "; ".join(f"{status.name}: {status.message}" for status in unavailable_commands)
        raise RuntimeError(f"Required artifacts or tools are unavailable: {details}")
    if aizynth_status.available:
        return statuses
    require_available([check_artifact(_AIZYNTH_CONFIG)])
    return statuses


def _is_planner_command_status(status: RequirementStatus) -> bool:
    return status.name == _PLANNER_COMMAND.name or (
        status.name.startswith("retrosyn_") and status.name.endswith("_planner_command")
    )


def runtime_status() -> list[dict]:
    return [status.to_dict() for status in _status_objects()]


async def _abort_unavailable(context, *requirements: ArtifactRequirement):
    statuses = [check_artifact(requirement) for requirement in requirements]
    try:
        require_available(statuses)
    except RuntimeError as exc:
        message = str(exc)
    else:
        message = "Retrosynthesis backend is not configured"
    if context is not None and hasattr(context, "abort"):
        await context.abort(grpc.StatusCode.FAILED_PRECONDITION, message)
    raise RuntimeError(message)


class UnsupportedRetrosynEngineError(ValueError):
    """Requested planner engine is not configured by this service."""


def _validated_request_fields(request: Any) -> dict[str, Any]:
    smiles = _required_request_text(
        getattr(request, "molecule_smiles", ""),
        "molecule_smiles is required",
    )
    canonical_smiles = getattr(request, "canonical_smiles", "")
    if canonical_smiles:
        canonical_smiles = _required_request_text(
            canonical_smiles,
            "canonical_smiles must be a non-empty trimmed string",
        )
        if canonical_smiles != smiles:
            raise ValueError("canonical_smiles must equal molecule_smiles")
    else:
        canonical_smiles = smiles
    raw_max_routes = getattr(request, "max_routes", 0)
    if isinstance(raw_max_routes, bool) or not isinstance(raw_max_routes, int):
        raise TypeError("max_routes must be an integer")
    if raw_max_routes < 0:
        raise ValueError("max_routes must not be negative")
    candidate_index = _optional_candidate_index(request)
    return {
        "project_id": _optional_request_text(
            getattr(request, "project_id", ""),
            "project_id",
        ),
        "request_id": _optional_request_text(
            getattr(request, "request_id", ""),
            "request_id",
        ),
        "run_id": _optional_request_text(
            getattr(request, "run_id", ""),
            "run_id",
        ),
        "candidate_id": _optional_request_text(
            getattr(request, "candidate_id", ""),
            "candidate_id",
        ),
        "candidate_index": candidate_index,
        "canonical_smiles": canonical_smiles,
        "max_routes": raw_max_routes or 10,
    }


def _optional_candidate_index(request: Any) -> int | None:
    has_field = getattr(request, "HasField", None)
    if callable(has_field):
        try:
            has_candidate_index = has_field("candidate_index")
        except ValueError:
            has_candidate_index = True
        if not has_candidate_index:
            return None
    raw_value = getattr(request, "candidate_index", None)
    if raw_value is None:
        return None
    if isinstance(raw_value, bool) or not isinstance(raw_value, int):
        raise TypeError("candidate_index must be an integer")
    if raw_value < 0:
        raise ValueError("candidate_index must not be negative")
    return raw_value


def _required_request_text(value: Any, message: str) -> str:
    if not isinstance(value, str) or not value or not value.strip() or value != value.strip():
        raise ValueError(message)
    return value


def _optional_request_text(value: Any, field: str) -> str:
    if value in (None, ""):
        return ""
    return _required_request_text(
        value,
        f"{field} must be a non-empty trimmed string",
    )


class RetrosynServicer:
    def __init__(self, planner=None, route_planners: dict[str, Any] | None = None):
        self.planner = planner
        self.route_planners = route_planners or _route_planners_from_env()
        self.planner_command = os.getenv("RETROSYN_PLANNER_COMMAND", "").strip()

    async def FindRoutes(self, request, context):  # noqa: N802
        """Plan retrosynthetic routes for a target molecule."""
        try:
            request_fields = _validated_request_fields(request)
        except (TypeError, ValueError) as exc:
            return await _abort(
                context,
                grpc.StatusCode.INVALID_ARGUMENT,
                str(exc),
                ValueError,
            )
        smiles = request_fields["canonical_smiles"]
        max_routes = request_fields["max_routes"]
        start = time.perf_counter()
        try:
            results = await self._find_routes(
                smiles,
                max_routes=max_routes,
                engine=getattr(request, "engine", "aizynth"),
            )
            routes, assessments = partition_retrosyn_results(
                results,
                "retrosynthesis planner",
            )
            routes = _rank_routes(_dedupe_routes(routes))[:max_routes]
        except (TimeoutError, subprocess.TimeoutExpired) as exc:
            return await _abort(
                context,
                grpc.StatusCode.DEADLINE_EXCEEDED,
                str(exc) or "retrosynthesis planner deadline exceeded",
                TimeoutError,
            )
        except RetrosynRouteError as exc:
            return await _abort(
                context,
                grpc.StatusCode.DATA_LOSS,
                str(exc),
                type(exc),
            )
        except UnsupportedRetrosynEngineError as exc:
            return await _abort(
                context,
                grpc.StatusCode.INVALID_ARGUMENT,
                str(exc),
                ValueError,
            )
        except RuntimeError as exc:
            return await _abort(
                context,
                grpc.StatusCode.FAILED_PRECONDITION,
                str(exc),
                RuntimeError,
            )
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        response_fields = {
            "request_id": request_fields["request_id"] or f"retrosyn-{uuid.uuid4().hex[:12]}",
            "routes": [_synthetic_route(route) for route in routes],
            "total_routes_found": len(routes),
            "elapsed_ms": elapsed_ms,
            "project_id": request_fields["project_id"],
            "candidate_id": request_fields["candidate_id"],
            "canonical_smiles": request_fields["canonical_smiles"],
            "run_id": request_fields["run_id"],
            "assessments": [_retrosynthesis_assessment(assessment) for assessment in assessments],
        }
        if request_fields["candidate_index"] is not None:
            response_fields["candidate_index"] = request_fields["candidate_index"]
        return retrosyn_pb2.RetrosynthesisResponse(**response_fields)

    async def FindRoutesStream(self, request_iterator, context):  # noqa: N802
        async for request in request_iterator:
            yield await self.FindRoutes(request, context)

    async def PlanRoutes(self, request, context):  # noqa: N802
        return await self.FindRoutes(request, context)

    async def ScoreRoute(self, request, context):  # noqa: N802
        """Score a specific synthetic route using RSGPT."""
        try:
            _require_runtime(_SCORER)
        except RuntimeError:
            return await _abort_unavailable(context, _SCORER)
        raise RuntimeError("Retrosynthesis route scorer is not configured")

    def _planner(self, engine: str):
        if self.planner is not None:
            return self.planner
        key = (engine or "aizynth").strip().lower()
        if key not in {"aizynth", "aizynthfinder"}:
            raise UnsupportedRetrosynEngineError(f"Unsupported retrosynthesis engine: {engine}")
        from mf_retrosyn.aizynth.retrosyn import AiZynthRetrosyn

        self.planner = AiZynthRetrosyn.from_env()
        return self.planner

    async def _find_routes(
        self,
        smiles: str,
        *,
        max_routes: int,
        engine: str,
    ) -> list[dict]:
        key = (engine or "aizynth").strip().lower()
        if key == "ensemble":
            if not self.route_planners:
                raise RuntimeError("retrosynthesis route planner ensemble is not configured")
            routes: list[dict] = []
            for planner_name, planner in self.route_planners.items():
                for route in await _find_routes_with_planner(planner, smiles, max_routes):
                    route_with_engine = dict(route)
                    route_with_engine.setdefault("source_engine", planner_name)
                    routes.append(route_with_engine)
            return routes
        if self.route_planners and key in self.route_planners:
            return await _find_routes_with_planner(
                self.route_planners[key],
                smiles,
                max_routes,
            )
        if self.planner_command:
            return await _run_planner_command(
                self.planner_command,
                smiles=smiles,
                max_routes=max_routes,
                engine=key,
            )
        planner = self._planner(engine)
        return await _find_routes_with_planner(planner, smiles, max_routes)


async def _maybe_await(value):
    if asyncio.iscoroutine(value):
        return await value
    return value


async def _find_routes_with_planner(
    planner: Any,
    smiles: str,
    max_routes: int,
) -> list[dict]:
    result = await _maybe_await(planner.find_routes(smiles, max_routes=max_routes))
    if not isinstance(result, list):
        raise RetrosynRouteTypeError("retrosynthesis planner must return a list of route dicts")
    for route in result:
        if not isinstance(route, dict):
            raise RetrosynRouteTypeError("retrosynthesis planner routes must be dictionaries")
    return result


class ExternalCommandRoutePlanner:
    def __init__(
        self,
        command: str,
        engine: str,
        command_requirement: CommandRequirement,
    ):
        self.command = command
        self.engine = engine
        self.command_requirement = command_requirement

    async def find_routes(self, smiles: str, max_routes: int = 10) -> list[dict]:
        return await _run_planner_command(
            self.command,
            smiles=smiles,
            max_routes=max_routes,
            engine=self.engine,
            command_requirement=self.command_requirement,
        )


def _route_planners_from_env() -> dict[str, ExternalCommandRoutePlanner]:
    raw = os.getenv(_PLANNER_COMMANDS_JSON_ENV, "").strip()
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
        planners[engine_name] = ExternalCommandRoutePlanner(
            command_value,
            engine_name,
            CommandRequirement(
                _planner_command_status_name(engine_name),
                _PLANNER_COMMANDS_JSON_ENV,
                required=False,
            ),
        )
    return planners


def _named_route_planners_from_env() -> dict[str, ExternalCommandRoutePlanner]:
    planners = {}
    for engine_name, env_name in _NAMED_PLANNER_COMMAND_ENVS:
        command_value = os.getenv(env_name, "").strip()
        if not command_value:
            continue
        planners[engine_name] = ExternalCommandRoutePlanner(
            command_value,
            engine_name,
            _NAMED_PLANNER_COMMAND_REQUIREMENTS[engine_name],
        )
    return planners


def _planner_command_status_name(engine_name: str) -> str:
    normalized = "".join(
        char if char.isalnum() else "_" for char in engine_name.strip().lower()
    ).strip("_")
    return f"retrosyn_{normalized or 'external'}_planner_command"


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
    reactions = route.get("reaction_smiles")
    if isinstance(reactions, list) and reactions:
        return "reactions:" + "|".join(str(item) for item in reactions)
    step_reactions = [
        str(step["reaction"])
        for step in route.get("steps") or []
        if isinstance(step, dict) and isinstance(step.get("reaction"), str)
    ]
    if step_reactions:
        return "reactions:" + "|".join(step_reactions)
    return json.dumps(route, sort_keys=True, default=str)


def _rank_routes(routes: list[dict]) -> list[dict]:
    indexed = list(enumerate(routes))
    indexed.sort(key=lambda item: (_route_priority(item[1]), -_route_score(item[1]), item[0]))
    return [route for _, route in indexed]


def _route_priority(route: dict) -> int:
    if str(route.get("route_type") or "") == "retrosynthetic_accessibility_score":
        return 1
    return 0


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


async def _run_planner_command(
    command: str,
    *,
    smiles: str,
    max_routes: int,
    engine: str,
    command_requirement: CommandRequirement = _PLANNER_COMMAND,
) -> list[dict]:
    payload = {
        "smiles": smiles,
        "max_routes": max_routes,
        "engine": engine,
    }
    result = await asyncio.to_thread(
        _run_planner_command_sync,
        command,
        payload,
        command_requirement,
    )
    routes = result.get("routes", result)
    if not isinstance(routes, list):
        raise RetrosynRouteTypeError("RETROSYN_PLANNER_COMMAND must return routes as a list")
    for route in routes:
        if not isinstance(route, dict):
            raise RetrosynRouteTypeError("RETROSYN_PLANNER_COMMAND routes must be JSON objects")
        route.setdefault("source_engine", engine)
    return routes[:max_routes]


def _run_planner_command_sync(
    command: str,
    payload: dict,
    command_requirement: CommandRequirement = _PLANNER_COMMAND,
) -> dict | list:
    _require_command_available(command_requirement, command)
    timeout = float(os.getenv("RETROSYN_PLANNER_COMMAND_TIMEOUT_SECONDS", "300"))
    completed = subprocess.run(
        shlex.split(command),
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
        raise RetrosynRouteValueError("RETROSYN_PLANNER_COMMAND returned invalid JSON") from exc
    if not isinstance(parsed, dict | list):
        raise RetrosynRouteTypeError("RETROSYN_PLANNER_COMMAND must return a JSON object or list")
    return parsed


async def _abort(context, code, message: str, error_type):
    if context is not None and hasattr(context, "abort"):
        await context.abort(code, message)
    raise error_type(message)


def _synthetic_route(route: dict):
    steps = route.get("steps") if isinstance(route.get("steps"), list) else []
    reaction_smiles = [str(step["reaction"]) for step in steps]
    building_blocks = _building_blocks(route, steps)
    return retrosyn_pb2.SyntheticRoute(
        route_id=str(route.get("route_id", "")),
        reaction_smiles=reaction_smiles,
        predicted_score=float(route.get("predicted_score", route.get("score", 0.0)) or 0.0),
        predicted_yield=float(route.get("predicted_yield", route.get("yield", 0.0)) or 0.0),
        n_steps=int(route.get("n_steps", len(steps)) or len(steps)),
        building_blocks=building_blocks,
        estimated_cost_usd_per_g=float(route.get("estimated_cost_usd_per_g", 0.0) or 0.0),
        all_commercially_available=bool(route.get("all_commercially_available", False)),
        steps=[_synthetic_route_step(step) for step in steps],
        source_engine=str(route.get("source_engine") or ""),
        route_type=str(route.get("route_type") or ""),
        building_block_records=_building_block_records(route, steps),
    )


def _synthetic_route_step(step: dict):
    fields = {
        "step_id": step["step_id"],
        "reaction": step["reaction"],
        "reaction_type": step["reaction_type"],
        "reactants": [dict(reactant) for reactant in step["reactants"]],
        "conditions": dict(step["conditions"]),
        "reagents": list(step.get("reagents") or []),
        "purification": str(step.get("purification") or ""),
        "operation": str(step.get("operation") or ""),
        "building_blocks": [dict(block) for block in step["building_blocks"]],
        "yield_fraction": float(step["yield"]),
    }
    return retrosyn_pb2.SyntheticRouteStep(**fields)


def _retrosynthesis_assessment(assessment: dict):
    return retrosyn_pb2.RetrosynthesisAssessment(
        assessment_id=str(assessment["route_id"]),
        assessment_type=str(assessment["route_type"]),
        source_engine=str(assessment.get("source_engine") or ""),
        score=_route_score(assessment),
        details=dict(assessment),
    )


def _building_blocks(route: dict, steps: list[dict]) -> list[str]:
    direct = route.get("building_blocks")
    if isinstance(direct, list) and direct:
        return [_building_block_smiles(item) for item in direct]
    blocks: list[str] = []
    for step in steps:
        for block in step.get("building_blocks") or []:
            smiles = _building_block_smiles(block)
            if smiles and smiles not in blocks:
                blocks.append(smiles)
    return blocks


def _building_block_records(route: dict, steps: list[dict]) -> list[dict]:
    direct = route.get("building_blocks")
    if isinstance(direct, list) and direct:
        return [_building_block_record(block) for block in direct]
    records: list[dict] = []
    seen: set[str] = set()
    for step in steps:
        for block in step.get("building_blocks") or []:
            record = _building_block_record(block)
            smiles = str(record["smiles"])
            if smiles not in seen:
                seen.add(smiles)
                records.append(record)
    return records


def _building_block_record(block: Any) -> dict:
    if isinstance(block, dict):
        return dict(block)
    return {"smiles": str(block)}


def _building_block_smiles(block) -> str:
    if isinstance(block, dict):
        return str(block.get("smiles", ""))
    return str(block)


async def serve():
    _require_planner_runtime()
    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=8))
    retrosyn_pb2_grpc.add_RetrosynServiceServicer_to_server(RetrosynServicer(), server)
    server.add_insecure_port("[::]:50057")
    await server.start()
    _LOGGER.info("Retrosynthesis Service running on :50057")
    await server.wait_for_termination()


def _validation_response(payload: object) -> dict:
    _require_synthetic_validation_enabled()
    if not isinstance(payload, dict):
        raise ValueError("retrosynthesis validation request must be a JSON object")
    expected_fields = {"smiles", "max_routes", "engine"}
    unexpected = sorted(set(payload) - expected_fields)
    if unexpected:
        raise ValueError(
            "retrosynthesis validation request has unexpected fields: "
            + ", ".join(unexpected)
        )
    missing = sorted(expected_fields - set(payload))
    if missing:
        raise ValueError(
            "retrosynthesis validation request is missing fields: "
            + ", ".join(missing)
        )
    smiles = _validation_text(payload["smiles"], "smiles")
    engine = _validation_text(payload["engine"], "engine")
    max_routes = payload["max_routes"]
    if (
        isinstance(max_routes, bool)
        or not isinstance(max_routes, int)
        or max_routes <= 0
        or max_routes > _VALIDATION_MAX_ROUTES
    ):
        raise ValueError(
            "retrosynthesis validation max_routes must be a positive integer "
            f"not greater than {_VALIDATION_MAX_ROUTES}"
        )
    precursors = _VALIDATION_PRECURSORS.get(smiles)
    if precursors is None:
        supported = ", ".join(sorted(_VALIDATION_PRECURSORS))
        raise ValueError(
            "retrosynthesis validation target is outside the synthetic validation "
            f"dataset; supported targets: {supported}"
        )
    route = _validation_route(
        target_smiles=smiles,
        precursors=precursors,
        requested_engine=engine,
    )
    return {
        "routes": [route],
        "validation_marker": _VALIDATION_MARKER,
    }


def _validation_route(
    *,
    target_smiles: str,
    precursors: tuple[str, str],
    requested_engine: str,
) -> dict:
    reactants = [
        {
            "smiles": precursor,
            "amount_mmol": 1.0 if index == 0 else 1.2,
        }
        for index, precursor in enumerate(precursors)
    ]
    building_blocks = [
        {
            "smiles": precursor,
            "source": _VALIDATION_MARKER,
            "validation_marker": _VALIDATION_MARKER,
        }
        for precursor in precursors
    ]
    route_id = f"validation-{target_smiles.lower()}-route-1"
    route_yield = 0.75
    return {
        "route_id": route_id,
        "source_engine": _VALIDATION_MARKER,
        "requested_engine": requested_engine,
        "validation_marker": _VALIDATION_MARKER,
        "score": 0.75,
        "predicted_yield": route_yield,
        "n_steps": 1,
        "building_blocks": building_blocks,
        "estimated_cost_usd_per_g": 1.0,
        "all_commercially_available": True,
        "steps": [
            {
                "step_id": f"{route_id}-step-1",
                "reaction": f"{'.'.join(precursors)}>>{target_smiles}",
                "reaction_type": "validation_coupling",
                "reactants": reactants,
                "conditions": {
                    "temperature_C": 25.0,
                    "time_h": 2.0,
                    "validation_marker": _VALIDATION_MARKER,
                },
                "yield": route_yield,
                "building_blocks": building_blocks,
                "validation_marker": _VALIDATION_MARKER,
            }
        ],
    }


def _validation_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(
            f"retrosynthesis validation {field_name} must be a non-empty trimmed string"
        )
    return value


def _require_synthetic_validation_enabled() -> None:
    if os.environ.get(_VALIDATION_GATE_ENV) != "true":
        raise RuntimeError(f"{_VALIDATION_GATE_ENV}=true is required")


def _run_validation_runner() -> None:
    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError as exc:
        raise ValueError("retrosynthesis validation request must be valid JSON") from exc
    json.dump(
        _validation_response(payload),
        sys.stdout,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    sys.stdout.write("\n")


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments:
        asyncio.run(serve())
        return 0
    if arguments != ["--validation-runner"]:
        sys.stderr.write(
            "Retrosynthesis service has unexpected command line arguments\n"
        )
        return 2
    try:
        _run_validation_runner()
    except RuntimeError as exc:
        sys.stderr.write(f"{exc}\n")
        return 1
    except ValueError as exc:
        sys.stderr.write(f"{exc}\n")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
