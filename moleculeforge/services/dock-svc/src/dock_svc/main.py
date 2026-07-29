"""Docking Service - gRPC server for molecular docking (GNINA + DiffDock-L, L2 Oracle)."""

import asyncio
import json
import math
import os
import shlex
import shutil
import subprocess
from concurrent import futures
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import unquote, urlparse

import grpc
from mf_core.artifacts import (
    RequirementStatus,
    require_available,
)
from mf_core.plugins.oracle import (
    OracleDataError,
    OracleRequestError,
    OracleUnavailableError,
    abort_oracle_error,
    build_oracle_error_evaluation,
    build_oracle_evaluation,
    build_oracle_response,
    resolve_oracle_artifact_refs,
    validate_oracle_request,
)
from mf_core.proto_gen.moleculeforge.v1.oracle import oracle_pb2, oracle_pb2_grpc

_DOCK_COMMAND_ENV = "DOCK_ORACLE_COMMAND"
_DOCK_DEFAULT_RECEPTOR_ENV = "DOCK_ORACLE_RECEPTOR_PDB"


def _status_objects() -> list[RequirementStatus]:
    return [_command_status()]


def _require_runtime(engine: str | None = None) -> list[RequirementStatus]:
    del engine
    statuses = _status_objects()
    require_available(statuses)
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
            return await asyncio.to_thread(
                _run_dock_command,
                request,
                docking_engine,
            )
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
        self._local_runtime = service is None
        self.service = service or DockServicer()

    async def Evaluate(self, request, context):
        try:
            request_context = validate_oracle_request(
                request,
                expected_level=oracle_pb2.L2_DOCKING,
                require_receptor_uri=True,
                required_parameters=("engine",),
                allowed_parameters=("engine",),
            )
            if request_context.parameters["engine"] not in {"gnina", "diffdock"}:
                raise OracleRequestError("oracle_parameters[engine] must be gnina or diffdock")
            artifacts = await resolve_oracle_artifact_refs(
                (
                    [*_status_objects(), _receptor_status(request_context.receptor_uri)]
                    if self._local_runtime
                    else []
                )
            )
            evaluations = []
            for index, smiles in enumerate(request_context.molecules):
                engine = request_context.parameters["engine"]
                try:
                    response = await self.service.Dock(
                        SimpleNamespace(
                            molecule_smiles=smiles,
                            smiles=smiles,
                            engine=engine,
                            protein_pdb=request_context.receptor_uri,
                        ),
                        context,
                    )
                except (TimeoutError, subprocess.TimeoutExpired):
                    raise
                except OracleUnavailableError:
                    raise
                except OracleDataError:
                    raise
                except RuntimeError as exc:
                    evaluations.append(
                        build_oracle_error_evaluation(
                            request=request_context,
                            index=index,
                            oracle_name=engine,
                            elapsed_ms=0,
                            artifacts=artifacts,
                            error_code="COMPUTATION_ERROR",
                            error_message=str(exc),
                        )
                    )
                    continue
                response_engine = _logical_engine(str(getattr(response, "engine", engine)))
                if response_engine != engine:
                    raise OracleDataError("docking response engine does not match request engine")
                if getattr(response, "smiles", None) != smiles:
                    raise OracleDataError("docking response smiles does not match request")
                if getattr(response, "receptor_uri", None) != request_context.receptor_uri:
                    raise OracleDataError("docking response receptor does not match request")
                evaluations.append(
                    build_oracle_evaluation(
                        request=request_context,
                        index=index,
                        oracle_name=engine,
                        scores=_dock_scores(response),
                        uncertainties=_dock_uncertainties(response),
                        elapsed_ms=_elapsed_milliseconds(getattr(response, "elapsed_ms", 0)),
                        artifacts=artifacts,
                    )
                )
            return build_oracle_response(
                request=request_context,
                evaluations=evaluations,
                total_elapsed_ms=sum(item.elapsed_ms for item in evaluations),
            )
        except (
            OracleRequestError,
            OracleUnavailableError,
            OracleDataError,
            TimeoutError,
            subprocess.TimeoutExpired,
        ) as exc:
            return await abort_oracle_error(context, exc)

    async def PredictWithUncertainty(self, request, context):
        return await self.Evaluate(request, context)

    async def StreamEvaluate(self, request_iterator, context):
        async for request in request_iterator:
            yield await self.Evaluate(request, context)


def _command_status() -> RequirementStatus:
    raw_command = os.environ.get(_DOCK_COMMAND_ENV, "")
    if not raw_command:
        return RequirementStatus(
            name="dock_oracle_command",
            configured=False,
            available=False,
            required=True,
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
            required=True,
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
        required=True,
        path=raw_command,
        source=_DOCK_COMMAND_ENV,
        message=(
            "DOCK_ORACLE_COMMAND is available"
            if available
            else f"{_DOCK_COMMAND_ENV} executable is not available: {executable}"
        ),
    )


def _logical_engine(value: str) -> str:
    return "diffdock" if value == "diffdock_l" else value


def _run_dock_command(request, engine: str):
    status = _command_status()
    if not status.available:
        raise RuntimeError(status.message)
    argv = shlex.split(os.environ[_DOCK_COMMAND_ENV])
    timeout = _positive_timeout(
        os.environ.get("DOCK_ORACLE_TIMEOUT_SECONDS", "120"),
        "DOCK_ORACLE_TIMEOUT_SECONDS",
    )
    request_smiles = str(
        getattr(request, "smiles", None) or getattr(request, "molecule_smiles", "")
    )
    payload = {
        "engine": str(engine),
        "smiles": request_smiles,
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
        raise OracleDataError("docking command returned invalid JSON") from exc
    if not isinstance(response, dict):
        raise OracleDataError("docking command must return a JSON object")
    required_identity = ("smiles", "receptor_uri", "engine")
    missing_identity = [field for field in required_identity if field not in response]
    if missing_identity:
        raise OracleDataError(
            "docking command response missing identity fields: " + ", ".join(missing_identity)
        )
    if response["smiles"] != request_smiles:
        raise OracleDataError("docking response smiles does not match request")
    if response["receptor_uri"] != str(protein_pdb or ""):
        raise OracleDataError("docking response receptor does not match request")
    if _logical_engine(str(response["engine"])) != _logical_engine(str(engine)):
        raise OracleDataError("docking response engine does not match request")
    scores = _command_scores(response)
    return SimpleNamespace(
        engine=str(response["engine"]),
        smiles=str(response["smiles"]),
        receptor_uri=str(response["receptor_uri"]),
        scores=scores,
        uncertainties=_numeric_mapping(response.get("uncertainties")),
        elapsed_ms=_elapsed_milliseconds(response.get("elapsed_ms", 0)),
    )


def _numeric_mapping(value) -> dict[str, float]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise OracleDataError("docking numeric result must be an object")
    output = {}
    for key, item in value.items():
        if isinstance(item, bool) or not isinstance(item, int | float):
            raise OracleDataError(f"docking metric {key} must be numeric")
        number = float(item)
        if not math.isfinite(number):
            raise OracleDataError(f"docking metric {key} must be finite")
        output[str(key)] = number
    return output


def _dock_scores(response) -> dict[str, float]:
    scores = _numeric_mapping(getattr(response, "scores", None))
    score = getattr(response, "score", None)
    docking_score = getattr(response, "docking_score", None)
    scalar_values = [
        _numeric_mapping({"docking_score": value})["docking_score"]
        for value in (score, docking_score)
        if value is not None
    ]
    if len(set(scalar_values)) > 1:
        raise OracleDataError("docking response contains contradictory scalar scores")
    if scalar_values:
        scalar = scalar_values[0]
        mapped = scores.get("docking_score")
        if mapped is not None and mapped != scalar:
            raise OracleDataError("docking response contains contradictory score and scores")
        scores["docking_score"] = scalar
    return scores


def _dock_uncertainties(response) -> dict[str, float]:
    uncertainties = getattr(response, "uncertainties", None)
    return _numeric_mapping(uncertainties)


def _command_scores(response: dict) -> dict[str, float]:
    scores = _numeric_mapping(response.get("scores"))
    scalar_values = [
        _numeric_mapping({"docking_score": response[field]})["docking_score"]
        for field in ("score", "docking_score")
        if field in response and response[field] is not None
    ]
    if len(set(scalar_values)) > 1:
        raise OracleDataError("docking command returned contradictory scalar scores")
    if scalar_values:
        scalar = scalar_values[0]
        if "docking_score" in scores and scores["docking_score"] != scalar:
            raise OracleDataError("docking command returned contradictory score and scores")
        scores["docking_score"] = scalar
    return scores


def _receptor_status(receptor_uri: str) -> RequirementStatus:
    parsed = urlparse(receptor_uri)
    if parsed.scheme == "file":
        path = Path(unquote(parsed.path)).expanduser()
    elif not parsed.scheme:
        path = Path(receptor_uri).expanduser()
    else:
        path = Path("")
    supported_uri = bool(receptor_uri) and (
        not parsed.scheme or (parsed.scheme == "file" and not parsed.netloc)
    )
    available = supported_uri and path.is_file()
    return RequirementStatus(
        name="dock_receptor",
        configured=bool(receptor_uri),
        available=available,
        required=True,
        path=str(path.resolve()) if available else receptor_uri,
        source="request.receptor_uri",
        message=(
            "docking receptor is available"
            if available
            else "request.receptor_uri must reference an available local receptor file"
        ),
    )


def _positive_timeout(value: object, field_name: str) -> float:
    try:
        timeout = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{field_name} must be a finite positive number") from exc
    if isinstance(value, bool) or not math.isfinite(timeout) or timeout <= 0:
        raise RuntimeError(f"{field_name} must be a finite positive number")
    return timeout


def _elapsed_milliseconds(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise OracleDataError("docking elapsed_ms must be numeric")
    elapsed_ms = float(value)
    if not math.isfinite(elapsed_ms) or elapsed_ms < 0:
        raise OracleDataError("docking elapsed_ms must be finite and non-negative")
    return int(elapsed_ms)


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
