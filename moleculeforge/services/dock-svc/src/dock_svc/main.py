"""Docking Service - gRPC server for molecular docking (GNINA + DiffDock-L, L1 Oracle)."""
import asyncio
import json
import os
import shlex
import shutil
import subprocess
from concurrent import futures
from types import SimpleNamespace

import grpc
from mf_core.artifacts import (
    ArtifactRequirement,
    PythonPackageRequirement,
    RequirementStatus,
    ToolRequirement,
    check_artifact,
    check_python_package,
    check_tool,
    require_available,
)
from mf_core.proto_gen.moleculeforge.v1.oracle import oracle_pb2, oracle_pb2_grpc

_GNINA_REQUIREMENT = ToolRequirement("gnina", executable="gnina", env_var="GNINA_BINARY")
_DIFFDOCK_REQUIREMENT = ArtifactRequirement("diffdock_model", "DIFFDOCK_MODEL_PATH", kind="path")
_PACKAGES = (PythonPackageRequirement("rdkit", module="rdkit"),)
_DOCK_COMMAND_ENV = "DOCK_ORACLE_COMMAND"
_DOCK_DEFAULT_RECEPTOR_ENV = "DOCK_ORACLE_RECEPTOR_PDB"


def _status_objects() -> list[RequirementStatus]:
    return [
        check_tool(_GNINA_REQUIREMENT),
        check_artifact(_DIFFDOCK_REQUIREMENT),
        _command_status(),
        *(check_python_package(requirement) for requirement in _PACKAGES),
    ]


def _require_runtime(engine: str | None = None) -> list[RequirementStatus]:
    statuses = _status_objects()
    command_status = statuses[2]
    if command_status.configured and not command_status.available:
        raise RuntimeError(
            "Required artifacts or tools are unavailable: "
            f"{command_status.name}: {command_status.message}"
        )
    if statuses[2].available:
        require_available(statuses[3:])
        return statuses
    if engine == "diffdock":
        require_available([statuses[1]])
    elif engine == "gnina":
        require_available([statuses[0]])
    elif not any(status.available for status in statuses[:3]):
        require_available(statuses)
    require_available(statuses[3:])
    return statuses


def runtime_status() -> list[dict]:
    return [status.to_dict() for status in _status_objects()]


async def _abort_unavailable(context, message: str | None = None):
    if message is None:
        statuses = _status_objects()
        try:
            require_available(statuses)
        except RuntimeError as exc:
            message = str(exc)
        else:
            message = "Docking runner is not configured"
    if context is not None and hasattr(context, "abort"):
        await context.abort(grpc.StatusCode.FAILED_PRECONDITION, message)
    raise RuntimeError(message)


class DockServicer:
    async def Dock(self, request, context):
        """Run molecular docking for a protein-ligand pair."""
        try:
            _require_runtime(getattr(request, "engine", "gnina"))
        except RuntimeError as exc:
            return await _abort_unavailable(context, str(exc))
        docking_engine = getattr(request, "engine", "gnina")
        if os.environ.get(_DOCK_COMMAND_ENV):
            return _run_dock_command(request, docking_engine)
        raise RuntimeError(f"{docking_engine} docking runner is not configured")

    async def BatchDock(self, request, context):
        """Batch docking requests."""
        results = []
        for req in getattr(request, "requests", []):
            results.append(await self.Dock(req, context))
        return type(
            "BatchDockResponse",
            (),
            {"results": results, "total_elapsed_ms": 2000},
        )()


class DockOracleServicer(oracle_pb2_grpc.OracleServiceServicer):
    def __init__(self, service: DockServicer | None = None):
        self.service = service or DockServicer()

    async def Evaluate(self, request, context):
        results = []
        for smiles in getattr(request, "molecule_smiles", []):
            response = await self.service.Dock(
                SimpleNamespace(
                    molecule_smiles=smiles,
                    smiles=smiles,
                    engine=_engine_from_request(request),
                ),
                context,
            )
            results.append(
                oracle_pb2.OracleEvaluation(
                    oracle_name=str(getattr(response, "engine", _engine_from_request(request))),
                    molecule_smiles=str(smiles),
                    level=request.level or oracle_pb2.L2_DOCKING,
                    scores=_dock_scores(response),
                    uncertainties=_dock_uncertainties(response),
                    elapsed_ms=int(getattr(response, "elapsed_ms", 0)),
                    success=True,
                )
            )
        return oracle_pb2.OracleBatchResponse(
            evaluations=results,
            batch_id=str(getattr(request, "project_id", "")),
        )

    async def PredictWithUncertainty(self, request, context):
        return await self.Evaluate(request, context)

    async def StreamEvaluate(self, request_iterator, context):
        async for request in request_iterator:
            yield await self.Evaluate(request, context)


def _engine_from_request(request) -> str:
    properties = [str(item).lower() for item in getattr(request, "requested_properties", [])]
    if "diffdock" in properties or "diffdock_l" in properties:
        return "diffdock"
    return "gnina"


def _command_status() -> RequirementStatus:
    raw_command = os.environ.get(_DOCK_COMMAND_ENV, "")
    if not raw_command:
        return RequirementStatus(
            name="dock_oracle_command",
            configured=False,
            available=False,
            required=False,
            path=None,
            source=_DOCK_COMMAND_ENV,
            message=f"{_DOCK_COMMAND_ENV} is not configured",
        )
    try:
        argv = shlex.split(raw_command)
    except ValueError as exc:
        return RequirementStatus(
            name="dock_oracle_command",
            configured=True,
            available=False,
            required=False,
            path=raw_command,
            source=_DOCK_COMMAND_ENV,
            message=str(exc),
        )
    executable = argv[0] if argv else ""
    resolved = shutil.which(executable) if executable else None
    available = bool(resolved or (executable and os.access(executable, os.X_OK)))
    return RequirementStatus(
        name="dock_oracle_command",
        configured=True,
        available=available,
        required=False,
        path=raw_command,
        source=_DOCK_COMMAND_ENV,
        message=(
            "DOCK_ORACLE_COMMAND is available"
            if available
            else f"{_DOCK_COMMAND_ENV} executable is not available: {executable}"
        ),
    )


def _run_dock_command(request, engine: str):
    status = _command_status()
    if not status.available:
        raise RuntimeError(status.message)
    argv = shlex.split(os.environ[_DOCK_COMMAND_ENV])
    timeout = float(os.environ.get("DOCK_ORACLE_TIMEOUT_SECONDS", "120"))
    payload = {
        "engine": str(engine),
        "smiles": str(
            getattr(request, "smiles", None)
            or getattr(request, "molecule_smiles", "")
        ),
    }
    protein_pdb = getattr(request, "protein_pdb", None) or os.environ.get(
        _DOCK_DEFAULT_RECEPTOR_ENV,
        "",
    )
    if protein_pdb:
        payload["protein_pdb"] = str(protein_pdb)
    result = subprocess.run(
        argv,
        input=json.dumps(payload),
        capture_output=True,
        check=False,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "docking command failed")
    try:
        response = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("docking command returned invalid JSON") from exc
    if not isinstance(response, dict):
        raise RuntimeError("docking command must return a JSON object")
    return SimpleNamespace(
        engine=str(response.get("engine") or engine),
        score=response.get("score", response.get("docking_score")),
        scores=_numeric_mapping(response.get("scores")),
        uncertainties=_numeric_mapping(response.get("uncertainties")),
        elapsed_ms=int(response.get("elapsed_ms", 0) or 0),
    )


def _numeric_mapping(value) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    return {
        str(key): float(item)
        for key, item in value.items()
        if isinstance(item, int | float)
    }


def _dock_scores(response) -> dict[str, float]:
    scores = getattr(response, "scores", None)
    if isinstance(scores, dict) and scores:
        return {
            str(key): float(value)
            for key, value in scores.items()
            if isinstance(value, int | float)
        }
    score = getattr(response, "score", None)
    return {"docking_score": float(score)} if isinstance(score, int | float) else {}


def _dock_uncertainties(response) -> dict[str, float]:
    uncertainties = getattr(response, "uncertainties", None)
    if not isinstance(uncertainties, dict):
        return {}
    return {
        str(key): float(value)
        for key, value in uncertainties.items()
        if isinstance(value, int | float)
    }


def register_grpc_services(server) -> None:
    oracle_pb2_grpc.add_OracleServiceServicer_to_server(DockOracleServicer(), server)


async def serve():
    _require_runtime()
    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=10))
    register_grpc_services(server)
    server.add_insecure_port("[::]:50054")
    await server.start()
    print("Docking Service running on :50054")
    await server.wait_for_termination()


if __name__ == "__main__":
    asyncio.run(serve())
