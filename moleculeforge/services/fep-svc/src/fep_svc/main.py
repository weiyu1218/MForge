"""FEP Service.

gRPC server for OpenFE RBFE calculations.
"""
import asyncio
import json
import os
import shlex
import subprocess
import time
import uuid
from concurrent import futures
from pathlib import Path

import grpc
from google.protobuf.json_format import MessageToJson, Parse
from mf_core.artifacts import (
    CommandRequirement,
    RequirementStatus,
    ToolRequirement,
    check_command,
    check_tool,
    require_available,
)
from mf_core.proto_gen.moleculeforge.v1.oracle import (
    fep_pb2,
    fep_pb2_grpc,
    oracle_pb2,
    oracle_pb2_grpc,
)

_TOOLS = (ToolRequirement("openfe_runner", executable="openfe", env_var="OPENFE_RUNNER_PATH"),)
_FEP_COMMAND_ENV = "FEP_ORACLE_COMMAND"
_FEP_COMMAND_REQUIREMENT = CommandRequirement("fep_oracle_command", _FEP_COMMAND_ENV)


def _status_objects() -> list[RequirementStatus]:
    if os.environ.get(_FEP_COMMAND_ENV, "").strip():
        return [check_command(_FEP_COMMAND_REQUIREMENT)]
    return [check_tool(requirement) for requirement in _TOOLS]


def _require_runtime() -> list[RequirementStatus]:
    statuses = _status_objects()
    if os.environ.get(_FEP_COMMAND_ENV):
        _fep_command_argv()
        return statuses
    require_available(statuses)
    return statuses


def runtime_status() -> list[dict]:
    return [status.to_dict() for status in _status_objects()]


async def _abort_unavailable(context):
    statuses = _status_objects()
    try:
        require_available(statuses)
    except RuntimeError as exc:
        message = str(exc)
    else:
        message = "OpenFE runner is not configured"
    if context is not None and hasattr(context, "abort"):
        await context.abort(grpc.StatusCode.FAILED_PRECONDITION, message)
    raise RuntimeError(message)


def _epoch_ms() -> int:
    return int(time.time() * 1000)


def _job_status(
    job_id: str,
    state: str,
    *,
    response=None,
    error: str = "",
    submitted_at_ms: int = 0,
    started_at_ms: int = 0,
    completed_at_ms: int = 0,
) -> fep_pb2.FEPJobStatus:
    status = fep_pb2.FEPJobStatus(
        job_id=job_id,
        state=state,
        error=error,
        submitted_at_ms=submitted_at_ms,
        started_at_ms=started_at_ms,
        completed_at_ms=completed_at_ms,
    )
    if response is not None:
        status.response.CopyFrom(response)
    return status


class FEPServicer:
    def __init__(self, job_dir: str | os.PathLike[str] | None = None):
        self.job_dir = Path(job_dir or os.environ.get("FEP_JOB_DIR", "runs/fep-jobs"))
        self._tasks: dict[str, asyncio.Task] = {}

    async def RunFEP(self, request, context):
        """Run Free Energy Perturbation calculation."""
        try:
            _require_runtime()
        except RuntimeError:
            return await _abort_unavailable(context)
        if os.environ.get(_FEP_COMMAND_ENV):
            return _run_fep_command(request)
        raise RuntimeError("OpenFE runner is not configured")

    async def SubmitFEP(self, request, context):
        """Submit a Free Energy Perturbation calculation as a background job."""
        try:
            _require_runtime()
        except RuntimeError:
            return await _abort_unavailable(context)
        job_id = uuid.uuid4().hex
        status = _job_status(job_id, "queued", submitted_at_ms=_epoch_ms())
        self._write_job_status(status)
        job_request = fep_pb2.FEPBatchRequest()
        job_request.CopyFrom(request)
        self._tasks[job_id] = asyncio.create_task(self._run_job(job_id, job_request))
        return status

    async def GetStatus(self, request, context):
        """Get status of a running FEP job."""
        try:
            _require_runtime()
        except RuntimeError:
            return await _abort_unavailable(context)
        try:
            return self._read_job_status(str(getattr(request, "job_id", "")))
        except RuntimeError as exc:
            if context is not None and hasattr(context, "abort"):
                await context.abort(grpc.StatusCode.NOT_FOUND, str(exc))
            raise

    async def _run_job(self, job_id: str, request) -> None:
        started_at_ms = _epoch_ms()
        self._write_job_status(
            _job_status(
                job_id,
                "running",
                submitted_at_ms=self._read_job_status(job_id).submitted_at_ms,
                started_at_ms=started_at_ms,
            )
        )
        try:
            response = await asyncio.to_thread(_run_fep_command, request)
        except Exception as exc:
            submitted_at_ms = self._read_job_status(job_id).submitted_at_ms
            self._write_job_status(
                _job_status(
                    job_id,
                    "failed",
                    error=str(exc),
                    submitted_at_ms=submitted_at_ms,
                    started_at_ms=started_at_ms,
                    completed_at_ms=_epoch_ms(),
                )
            )
            return
        submitted_at_ms = self._read_job_status(job_id).submitted_at_ms
        self._write_job_status(
            _job_status(
                job_id,
                "completed",
                response=response,
                submitted_at_ms=submitted_at_ms,
                started_at_ms=started_at_ms,
                completed_at_ms=_epoch_ms(),
            )
        )

    def _job_path(self, job_id: str) -> Path:
        if not job_id or not all(ch.isalnum() or ch in {"-", "_"} for ch in job_id):
            raise RuntimeError("FEP job_id is invalid")
        return self.job_dir / f"{job_id}.json"

    def _write_job_status(self, status: fep_pb2.FEPJobStatus) -> None:
        self.job_dir.mkdir(parents=True, exist_ok=True)
        path = self._job_path(status.job_id)
        tmp_path = path.with_suffix(".json.tmp")
        tmp_path.write_text(
            MessageToJson(status, preserving_proto_field_name=True),
            encoding="utf-8",
        )
        tmp_path.replace(path)

    def _read_job_status(self, job_id: str) -> fep_pb2.FEPJobStatus:
        path = self._job_path(job_id)
        if not path.is_file():
            raise RuntimeError(f"FEP job not found: {job_id}")
        return Parse(path.read_text(encoding="utf-8"), fep_pb2.FEPJobStatus())


class FEPOracleServicer(oracle_pb2_grpc.OracleServiceServicer):
    def __init__(self, service: FEPServicer | None = None):
        self.service = service or FEPServicer()

    async def Evaluate(self, request, context):
        return await self._evaluate(request, context)

    async def PredictWithUncertainty(self, request, context):
        return await self._evaluate(request, context)

    async def StreamEvaluate(self, request_iterator, context):
        async for request in request_iterator:
            yield await self.Evaluate(request, context)

    async def _evaluate(self, request, context):
        fep_response = await self.service.RunFEP(
            _oracle_request_to_fep_batch(request),
            context,
        )
        return oracle_pb2.OracleBatchResponse(
            evaluations=[
                _oracle_evaluation_from_fep_result(result)
                for result in fep_response.results
            ],
            batch_id=fep_response.batch_id,
            total_elapsed_ms=fep_response.total_elapsed_ms,
        )


def _fep_command_argv() -> list[str]:
    raw_command = os.environ.get(_FEP_COMMAND_ENV, "")
    _require_command_available(raw_command)
    try:
        argv = shlex.split(raw_command)
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc
    if not argv:
        raise RuntimeError(f"{_FEP_COMMAND_ENV} is empty")
    return argv


def _require_command_available(command: str) -> None:
    env = {**os.environ, _FEP_COMMAND_ENV: command}
    require_available([check_command(_FEP_COMMAND_REQUIREMENT, env=env)])


def _run_fep_command(request) -> fep_pb2.FEPBatchResponse:
    result = subprocess.run(
        _fep_command_argv(),
        input=json.dumps(_fep_request_payload(request)),
        capture_output=True,
        check=False,
        text=True,
        timeout=float(os.environ.get("FEP_ORACLE_TIMEOUT_SECONDS", "120")),
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "FEP command failed")
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("FEP command returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("FEP command must return a JSON object")
    records = payload.get("results")
    if not isinstance(records, list) or not records:
        raise RuntimeError("FEP command response requires non-empty results")
    return fep_pb2.FEPBatchResponse(
        results=[_fep_result_from_record(record) for record in records],
        batch_id=str(payload.get("batch_id") or getattr(request, "project_id", "")),
        total_elapsed_ms=int(payload.get("total_elapsed_ms", 0) or 0),
    )


def _fep_request_payload(request) -> dict:
    return {
        "project_id": str(getattr(request, "project_id", "")),
        "protein_pdb_id": str(getattr(request, "protein_pdb_id", "")),
        "reference_ligand_smiles": str(
            getattr(request, "reference_ligand_smiles", "")
        ),
        "test_ligand_smiles": [
            str(smiles) for smiles in getattr(request, "test_ligand_smiles", [])
        ],
        "method": str(getattr(request, "method", "") or "openfe"),
        "n_repeats": int(getattr(request, "n_repeats", 0) or 0),
    }


def _fep_result_from_record(record) -> fep_pb2.FEPResult:
    if not isinstance(record, dict):
        raise RuntimeError("FEP command result records must be JSON objects")
    return fep_pb2.FEPResult(
        ligand_a_smiles=str(record.get("ligand_a_smiles", "")),
        ligand_b_smiles=str(record.get("ligand_b_smiles", "")),
        ddg_kcal_mol=float(record.get("ddg_kcal_mol", 0.0)),
        ddg_uncertainty=float(record.get("ddg_uncertainty", 0.0)),
        n_repeats=int(record.get("n_repeats", 0) or 0),
        method=str(record.get("method", "openfe")),
        per_repeat_ddg=_numeric_mapping(record.get("per_repeat_ddg")),
        converged=bool(record.get("converged", False)),
    )


def _numeric_mapping(value) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    return {
        str(key): float(item)
        for key, item in value.items()
        if isinstance(item, int | float)
    }


def _oracle_request_to_fep_batch(request) -> fep_pb2.FEPBatchRequest:
    reference = os.environ.get("FEP_REFERENCE_LIGAND_SMILES", "")
    if not reference:
        raise RuntimeError("FEP_REFERENCE_LIGAND_SMILES is required for OracleService FEP")
    molecules = [str(smiles) for smiles in getattr(request, "molecule_smiles", [])]
    if not molecules:
        raise ValueError("OracleService FEP requires molecule_smiles")
    return fep_pb2.FEPBatchRequest(
        project_id=str(getattr(request, "project_id", "")),
        reference_ligand_smiles=reference,
        test_ligand_smiles=molecules,
        method=os.environ.get("FEP_METHOD", "openfe"),
        n_repeats=int(os.environ.get("FEP_N_REPEATS", "1")),
    )


def _oracle_evaluation_from_fep_result(result) -> oracle_pb2.OracleEvaluation:
    return oracle_pb2.OracleEvaluation(
        oracle_name=str(result.method or "openfe"),
        molecule_smiles=str(result.ligand_b_smiles),
        level=oracle_pb2.L3_FEP,
        scores={"rbfe": float(result.ddg_kcal_mol)},
        uncertainties={"rbfe": float(result.ddg_uncertainty)},
        elapsed_ms=0,
        success=bool(result.converged),
    )


def register_grpc_services(server) -> None:
    oracle_pb2_grpc.add_OracleServiceServicer_to_server(FEPOracleServicer(), server)


async def serve():
    _require_runtime()
    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=8))
    fep_pb2_grpc.add_FEPServiceServicer_to_server(FEPServicer(), server)
    register_grpc_services(server)
    server.add_insecure_port("[::]:50055")
    await server.start()
    print("FEP Service running on :50055")
    await server.wait_for_termination()


if __name__ == "__main__":
    asyncio.run(serve())
