"""CIG Compiler Service - gRPC server for NL-to-Chemical-Intent-Graph parsing."""
import asyncio
import json
import os
import shlex
import subprocess
from concurrent import futures
from time import perf_counter
from types import SimpleNamespace

import grpc
from mf_core.artifacts import (
    CommandRequirement,
    RequirementStatus,
    check_command,
    require_available,
)
from mf_core.proto_gen.moleculeforge.v1.core import cig_pb2, cig_pb2_grpc, humu_pb2

from cig_compiler_svc.domain.compiler import (
    CIGCompiler,
    semantic_parser_command_status,
)

_CIG_REFINEMENT_COMMAND_ENV = "CIG_REFINEMENT_COMMAND"
_CIG_REFINEMENT_TIMEOUT_ENV = "CIG_REFINEMENT_TIMEOUT_SECONDS"
_CIG_REFINEMENT_COMMAND_REQUIREMENT = CommandRequirement(
    "cig_refinement_command",
    _CIG_REFINEMENT_COMMAND_ENV,
)


def runtime_status() -> list[dict]:
    return [status.to_dict() for status in _runtime_statuses()]


def _runtime_statuses() -> list[RequirementStatus]:
    statuses: list[RequirementStatus] = []
    parser_status = semantic_parser_command_status()
    if parser_status is not None:
        statuses.append(parser_status)
    if os.environ.get(_CIG_REFINEMENT_COMMAND_ENV, "").strip():
        statuses.append(check_command(_CIG_REFINEMENT_COMMAND_REQUIREMENT))
    return statuses


def _require_command_available(
    requirement: CommandRequirement,
    command: str,
) -> None:
    env = {**os.environ, requirement.env_var: command}
    require_available([check_command(requirement, env=env)])


async def _abort_unavailable(context, message: str):
    if context is not None and hasattr(context, "abort"):
        await context.abort(grpc.StatusCode.FAILED_PRECONDITION, message)
    raise RuntimeError(message)


class CIGCompilerServicer:
    def __init__(self, compiler: CIGCompiler | None = None) -> None:
        self.compiler = compiler or CIGCompiler()

    async def Compile(self, request, context):
        """Compile natural language into a Chemical Intent Graph (CIG)."""
        start = perf_counter()
        nl_query = getattr(request, "nl_query", "")
        seed = getattr(request, "seed", None)

        try:
            cig, hciv, cone = await self.compiler.compile(nl_query, seed=seed)
        except RuntimeError as exc:
            return await _abort_unavailable(context, str(exc))

        return type(
            "CIGResponse",
            (),
            {
                "cig": cig.model_dump(mode="json"),
                "hciv": hciv.model_dump(mode="json"),
                "intent_cone": cone.model_dump(mode="json"),
                "parse_confidence": None,
                "ambiguities": [],
                "elapsed_ms": int((perf_counter() - start) * 1000),
            },
        )()

    async def Validate(self, request, context):
        """Validate a CIG for consistency and completeness."""
        cig = getattr(request, "cig", {})
        goals = cig.get("goals", [])

        issues = []
        if not goals:
            issues.append("No goals defined in CIG")
        if not cig.get("edges"):
            issues.append("No edges defined in CIG")
        valid = len(issues) == 0

        return type(
            "ValidationResponse",
            (),
            {
                "valid": valid,
                "issues": issues,
                "warnings": [],
                "suggestions": [],
            },
        )()

    async def Refine(self, request, context):
        """Refine a CIG based on feedback or additional context."""
        start = perf_counter()
        try:
            payload = _run_refinement_command(request)
        except RuntimeError as exc:
            return await _abort_unavailable(context, str(exc))
        return type(
            "CIGResponse",
            (),
            {
                "cig": payload["cig"],
                "hciv": payload["hciv"],
                "intent_cone": payload["intent_cone"],
                "parse_confidence": payload.get("parse_confidence"),
                "ambiguities": [
                    str(item) for item in payload.get("ambiguities", [])
                ],
                "elapsed_ms": int(payload.get("elapsed_ms") or (perf_counter() - start) * 1000),
            },
        )()


class CIGCompilerGrpcServicer(cig_pb2_grpc.CIGCompilerServiceServicer):
    def __init__(self, service: CIGCompilerServicer | None = None) -> None:
        self.service = service or CIGCompilerServicer()

    async def Compile(self, request, context):
        seed = request.seed if request.HasField("seed") else None
        response = await self.service.Compile(
            SimpleNamespace(nl_query=request.nl_query, seed=seed),
            context,
        )
        return cig_pb2.CIGCompileResponse(
            cig=_cig_to_proto(response.cig),
            hciv=_hciv_to_proto(response.hciv),
            intent_cone=_intent_cone_to_proto(response.intent_cone),
            ambiguities=list(response.ambiguities),
            elapsed_ms=int(response.elapsed_ms),
        )

    async def Validate(self, request, context):
        issues = []
        if not request.cig.objectives:
            issues.append("No goals defined in CIG")
        if not request.cig.edges:
            issues.append("No edges defined in CIG")
        return cig_pb2.CIGValidationResponse(
            valid=not issues,
            issues=issues,
            warnings=[],
            suggestions=[],
        )

    async def Refine(self, request, context):
        response = await self.service.Refine(request, context)
        return cig_pb2.CIGCompileResponse(
            cig=_cig_to_proto(response.cig),
            hciv=_hciv_to_proto(response.hciv),
            intent_cone=_intent_cone_to_proto(response.intent_cone),
            parse_confidence=response.parse_confidence
            if response.parse_confidence is not None
            else None,
            ambiguities=list(response.ambiguities),
            elapsed_ms=int(response.elapsed_ms),
        )


def _run_refinement_command(request) -> dict:
    raw_command = os.environ.get(_CIG_REFINEMENT_COMMAND_ENV, "").strip()
    if not raw_command:
        raise RuntimeError("CIG refinement runner is not configured")
    _require_command_available(_CIG_REFINEMENT_COMMAND_REQUIREMENT, raw_command)
    try:
        argv = shlex.split(raw_command)
    except ValueError as exc:
        raise RuntimeError(f"{_CIG_REFINEMENT_COMMAND_ENV} is not a valid shell command") from exc
    if not argv:
        raise RuntimeError(f"{_CIG_REFINEMENT_COMMAND_ENV} is empty")
    payload = {
        "cig": _request_cig_to_dict(getattr(request, "cig", {})),
        "feedback": str(getattr(request, "feedback", "")),
        "context": {
            str(key): str(value)
            for key, value in dict(getattr(request, "context", {}) or {}).items()
        },
    }
    completed = subprocess.run(
        argv,
        input=json.dumps(payload),
        capture_output=True,
        check=False,
        text=True,
        timeout=float(os.environ.get(_CIG_REFINEMENT_TIMEOUT_ENV, "60")),
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        raise RuntimeError(f"{_CIG_REFINEMENT_COMMAND_ENV} failed: {stderr}")
    try:
        response = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{_CIG_REFINEMENT_COMMAND_ENV} returned invalid JSON") from exc
    if not isinstance(response, dict):
        raise RuntimeError(f"{_CIG_REFINEMENT_COMMAND_ENV} must return a JSON object")
    for field in ("cig", "hciv", "intent_cone"):
        if not isinstance(response.get(field), dict):
            raise RuntimeError(f"{_CIG_REFINEMENT_COMMAND_ENV} response requires {field}")
    return response


def _request_cig_to_dict(cig) -> dict:
    if isinstance(cig, dict):
        return cig
    objectives = [
        {
            "id": str(item.id),
            "name": str(item.name),
            "type": _objective_type_name(item.type),
            "target_value": float(item.target_value),
            "target_min": float(item.target_min) if item.HasField("target_min") else None,
            "target_max": float(item.target_max) if item.HasField("target_max") else None,
            "property": str(item.property),
            "weight": float(item.weight),
            "pareto_tier": int(item.pareto_tier),
        }
        for item in getattr(cig, "objectives", [])
    ]
    return {
        "project_id": str(getattr(cig, "project_id", "")),
        "objective_nodes": objectives,
        "edges": [
            {
                "source_id": str(item.source_id),
                "target_id": str(item.target_id),
                "relation": str(item.relation),
                "strength": float(item.strength),
            }
            for item in getattr(cig, "edges", [])
        ],
        "hyperedges": [
            {
                "source_ids": [str(value) for value in item.source_ids],
                "target_ids": [str(value) for value in item.target_ids],
                "relation": str(item.relation),
                "strength": float(item.strength),
            }
            for item in getattr(cig, "hyperedges", [])
        ],
        "generative_priors": {
            str(key): str(value)
            for key, value in dict(getattr(cig, "constraints", {}) or {}).items()
        },
        "created_by": str(getattr(cig, "created_by", "")),
    }


def _objective_type_name(value) -> str:
    if value == cig_pb2.MINIMIZE:
        return "continuous_minimize"
    if value == cig_pb2.TARGET_RANGE:
        return "target_range"
    if value == cig_pb2.CONSTRAINT:
        return "multi_constraint_satisfy"
    return "continuous_maximize"


def _objective_type_to_proto(value) -> int:
    raw_value = getattr(value, "value", str(value))
    if raw_value in {"minimize", "continuous_minimize"}:
        return cig_pb2.MINIMIZE
    if raw_value == "target_range":
        return cig_pb2.TARGET_RANGE
    if raw_value in {"constraint", "multi_constraint_satisfy"}:
        return cig_pb2.CONSTRAINT
    return cig_pb2.MAXIMIZE


def _objective_to_proto(objective) -> cig_pb2.ObjectiveNode:
    payload = {
        "id": str(objective.id),
        "name": str(objective.name),
        "type": _objective_type_to_proto(objective.type),
        "target_value": float(objective.target_value),
        "property": str(objective.property or objective.name),
        "weight": float(objective.weight),
        "pareto_tier": int(objective.pareto_tier),
    }
    if objective.target_min is not None:
        payload["target_min"] = float(objective.target_min)
    if objective.target_max is not None:
        payload["target_max"] = float(objective.target_max)
    return cig_pb2.ObjectiveNode(**payload)


def _edge_to_proto(edge) -> cig_pb2.ObjectiveEdge:
    return cig_pb2.ObjectiveEdge(
        source_id=str(edge.source_id),
        target_id=str(edge.target_id),
        relation=str(edge.relation),
        strength=float(edge.strength),
    )


def _hyperedge_to_proto(edge) -> cig_pb2.ObjectiveHyperedge:
    return cig_pb2.ObjectiveHyperedge(
        source_ids=[str(item) for item in edge.source_ids],
        target_ids=[str(item) for item in edge.target_ids],
        relation=str(edge.relation),
        strength=float(edge.strength),
    )


def _cig_to_proto(cig: dict) -> cig_pb2.CIG:
    objectives = [
        _objective_to_proto(SimpleNamespace(**item))
        for item in cig.get("objective_nodes", [])
    ]
    edges = [
        _edge_to_proto(SimpleNamespace(**item))
        for item in cig.get("edges", [])
    ]
    hyperedges = [
        _hyperedge_to_proto(SimpleNamespace(**item))
        for item in cig.get("hyperedges", [])
    ]
    constraints = {
        str(key): str(value)
        for key, value in cig.get("generative_priors", {}).items()
    }
    return cig_pb2.CIG(
        project_id=str(cig.get("intent_id", "")),
        objectives=objectives,
        edges=edges,
        hyperedges=hyperedges,
        constraints=constraints,
        created_by=str(cig.get("created_by") or ""),
    )


def _hciv_to_proto(hciv: dict) -> humu_pb2.HCIV:
    return humu_pb2.HCIV(
        coordinates=[float(item) for item in hciv.get("coordinates", [])],
        curvature=float(hciv.get("curvature", 1.0)),
        molecule_smiles=str(hciv.get("molecule_smiles", "")),
        parent_hciv_id=str(hciv["parent_hciv_id"])
        if hciv.get("parent_hciv_id")
        else None,
    )


def _intent_cone_to_proto(cone: dict) -> humu_pb2.IntentCone:
    return humu_pb2.IntentCone(
        axis=[float(item) for item in cone.get("axis", [])],
        half_angle=float(cone.get("half_angle", cone.get("angle_radians", 0.5))),
        curvature=float(cone.get("curvature", 1.0)),
        property_weights={
            str(key): float(value)
            for key, value in cone.get("property_weights", {}).items()
        },
    )


def register_grpc_services(server) -> None:
    cig_pb2_grpc.add_CIGCompilerServiceServicer_to_server(CIGCompilerGrpcServicer(), server)


async def serve():
    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=10))
    register_grpc_services(server)
    server.add_insecure_port("[::]:50061")
    await server.start()
    print("CIG Compiler Service running on :50061")
    await server.wait_for_termination()


if __name__ == "__main__":
    asyncio.run(serve())
