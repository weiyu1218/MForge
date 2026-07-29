"""FEP Service.

gRPC server for OpenFE RBFE calculations.
"""

import asyncio
import json
import math
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
    check_command,
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
    parse_positive_parameter,
    resolve_oracle_artifact_refs,
    validate_oracle_request,
)
from mf_core.proto_gen.moleculeforge.v1.oracle import (
    fep_pb2,
    fep_pb2_grpc,
    oracle_pb2,
    oracle_pb2_grpc,
)

_FEP_COMMAND_ENV = "FEP_ORACLE_COMMAND"
_FEP_COMMAND_REQUIREMENT = CommandRequirement("fep_oracle_command", _FEP_COMMAND_ENV)


def _status_objects() -> list[RequirementStatus]:
    return [check_command(_FEP_COMMAND_REQUIREMENT)]


def _require_runtime() -> list[RequirementStatus]:
    statuses = _status_objects()
    require_available(statuses)
    _fep_command_argv()
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
        _validate_fep_batch_request(request)
        try:
            _require_runtime()
        except RuntimeError:
            return await _abort_unavailable(context)
        if os.environ.get(_FEP_COMMAND_ENV):
            return await asyncio.to_thread(_run_fep_command, request)
        raise RuntimeError("OpenFE runner is not configured")

    async def SubmitFEP(self, request, context):
        """Submit a Free Energy Perturbation calculation as a background job."""
        _validate_fep_batch_request(request)
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
        self._local_runtime = service is None
        self.service = service or FEPServicer()

    async def Evaluate(self, request, context):
        return await self._evaluate(request, context)

    async def PredictWithUncertainty(self, request, context):
        return await self._evaluate(request, context)

    async def StreamEvaluate(self, request_iterator, context):
        async for request in request_iterator:
            yield await self.Evaluate(request, context)

    async def _evaluate(self, request, context):
        try:
            request_context = validate_oracle_request(
                request,
                expected_level=oracle_pb2.L3_FEP,
                require_protein_pdb_id=True,
                require_reference_ligand=True,
                required_parameters=("method", "n_repeats"),
                allowed_parameters=("method", "n_repeats"),
            )
            artifacts = await resolve_oracle_artifact_refs(
                _status_objects() if self._local_runtime else []
            )
            fep_request = _oracle_request_to_fep_batch(request_context)
            try:
                fep_response = await self.service.RunFEP(
                    fep_request,
                    context,
                )
            except (TimeoutError, subprocess.TimeoutExpired):
                raise
            except OracleUnavailableError:
                raise
            except OracleDataError:
                raise
            except RuntimeError as exc:
                evaluations = [
                    build_oracle_error_evaluation(
                        request=request_context,
                        index=index,
                        oracle_name=request_context.parameters["method"],
                        elapsed_ms=0,
                        artifacts=artifacts,
                        error_code="COMPUTATION_ERROR",
                        error_message=str(exc),
                    )
                    for index in range(len(request_context.molecules))
                ]
                total_elapsed_ms = 0
            else:
                results = list(fep_response.results)
                _validate_fep_response_identity(fep_response, fep_request)
                actual_order = tuple(str(result.ligand_b_smiles) for result in results)
                if actual_order != request_context.molecules:
                    raise OracleDataError("FEP results do not match request molecule order")
                if any(
                    result.ligand_a_smiles != request_context.reference_ligand_smiles
                    for result in results
                ):
                    raise OracleDataError(
                        "FEP result reference does not match reference_ligand_smiles"
                    )
                if any(result.method != request_context.parameters["method"] for result in results):
                    raise OracleDataError(
                        "FEP result method does not match oracle_parameters[method]"
                    )
                if any(result.n_repeats != fep_request.n_repeats for result in results):
                    raise OracleDataError("FEP result n_repeats does not match request")
                for result in results:
                    _validate_repeat_evidence(result.per_repeat_ddg, result.n_repeats)
                evaluations = [
                    _oracle_evaluation_from_fep_result(
                        request_context=request_context,
                        index=index,
                        result=result,
                        artifacts=artifacts,
                    )
                    for index, result in enumerate(results)
                ]
                total_elapsed_ms = int(fep_response.total_elapsed_ms)
            return build_oracle_response(
                request=request_context,
                evaluations=evaluations,
                total_elapsed_ms=total_elapsed_ms,
            )
        except (
            OracleRequestError,
            OracleUnavailableError,
            OracleDataError,
            TimeoutError,
            subprocess.TimeoutExpired,
        ) as exc:
            return await abort_oracle_error(context, exc)


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
        timeout=_positive_timeout(
            os.environ.get("FEP_ORACLE_TIMEOUT_SECONDS", "120"),
            "FEP_ORACLE_TIMEOUT_SECONDS",
        ),
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "FEP command failed")
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise OracleDataError("FEP command returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise OracleDataError("FEP command must return a JSON object")
    _validate_fep_command_identity(payload, request)
    records = payload.get("results")
    if not isinstance(records, list) or not records:
        raise OracleDataError("FEP command response requires non-empty results")
    total_elapsed_ms = payload.get("total_elapsed_ms", 0)
    if (
        isinstance(total_elapsed_ms, bool)
        or not isinstance(total_elapsed_ms, int | float)
        or not math.isfinite(float(total_elapsed_ms))
        or total_elapsed_ms < 0
    ):
        raise OracleDataError("FEP total_elapsed_ms must be a finite non-negative number")
    results = [_fep_result_from_record(record) for record in records]
    _validate_fep_results(results, request)
    return fep_pb2.FEPBatchResponse(
        results=results,
        batch_id=str(payload["batch_id"]),
        total_elapsed_ms=int(total_elapsed_ms),
        project_id=str(payload["project_id"]),
        protein_pdb_id=str(payload["protein_pdb_id"]),
        reference_ligand_smiles=str(payload["reference_ligand_smiles"]),
        test_ligand_smiles=[str(smiles) for smiles in payload["test_ligand_smiles"]],
        method=str(payload["method"]),
        n_repeats=int(payload["n_repeats"]),
    )


def _fep_request_payload(request) -> dict:
    return {
        "project_id": str(getattr(request, "project_id", "")),
        "protein_pdb_id": str(getattr(request, "protein_pdb_id", "")),
        "reference_ligand_smiles": str(getattr(request, "reference_ligand_smiles", "")),
        "test_ligand_smiles": [
            str(smiles) for smiles in getattr(request, "test_ligand_smiles", [])
        ],
        "method": str(getattr(request, "method", "")),
        "n_repeats": int(getattr(request, "n_repeats", 0) or 0),
    }


def _fep_result_from_record(record) -> fep_pb2.FEPResult:
    if not isinstance(record, dict):
        raise OracleDataError("FEP command result records must be JSON objects")
    required = (
        "ligand_a_smiles",
        "ligand_b_smiles",
        "ddg_kcal_mol",
        "ddg_uncertainty",
        "n_repeats",
        "method",
        "converged",
    )
    missing = [name for name in required if name not in record]
    if missing:
        raise OracleDataError(f"FEP command result missing fields: {', '.join(missing)}")
    if not str(record["ligand_a_smiles"]).strip():
        raise OracleDataError("FEP ligand_a_smiles must be non-empty")
    if not str(record["ligand_b_smiles"]).strip():
        raise OracleDataError("FEP ligand_b_smiles must be non-empty")
    if not str(record["method"]).strip():
        raise OracleDataError("FEP method must be non-empty")
    ddg = _finite_output(record["ddg_kcal_mol"], "ddg_kcal_mol")
    uncertainty = _finite_output(record["ddg_uncertainty"], "ddg_uncertainty")
    n_repeats = record["n_repeats"]
    if isinstance(n_repeats, bool) or not isinstance(n_repeats, int) or n_repeats <= 0:
        raise OracleDataError("FEP n_repeats must be a positive integer")
    if not isinstance(record["converged"], bool):
        raise OracleDataError("FEP converged must be a boolean")
    per_repeat_ddg = _numeric_mapping(record.get("per_repeat_ddg"))
    _validate_repeat_evidence(per_repeat_ddg, n_repeats)
    return fep_pb2.FEPResult(
        ligand_a_smiles=str(record["ligand_a_smiles"]),
        ligand_b_smiles=str(record["ligand_b_smiles"]),
        ddg_kcal_mol=ddg,
        ddg_uncertainty=uncertainty,
        n_repeats=n_repeats,
        method=str(record["method"]),
        per_repeat_ddg=per_repeat_ddg,
        converged=record["converged"],
    )


def _numeric_mapping(value) -> dict[str, float]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise OracleDataError("FEP per_repeat_ddg must be an object")
    return {str(key): _finite_output(item, f"per_repeat_ddg[{key}]") for key, item in value.items()}


def _finite_output(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise OracleDataError(f"FEP {field_name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise OracleDataError(f"FEP {field_name} must be finite")
    return number


def _validate_fep_command_identity(payload: dict, request) -> None:
    expected = {
        "batch_id": str(getattr(request, "project_id", "")),
        "project_id": str(getattr(request, "project_id", "")),
        "protein_pdb_id": str(getattr(request, "protein_pdb_id", "")),
        "reference_ligand_smiles": str(getattr(request, "reference_ligand_smiles", "")),
        "test_ligand_smiles": [
            str(smiles) for smiles in getattr(request, "test_ligand_smiles", [])
        ],
        "method": str(getattr(request, "method", "")),
        "n_repeats": int(getattr(request, "n_repeats", 0)),
    }
    missing = [field for field in expected if field not in payload]
    if missing:
        raise OracleDataError("FEP command response missing identity fields: " + ", ".join(missing))
    response_repeats = payload["n_repeats"]
    if (
        isinstance(response_repeats, bool)
        or not isinstance(response_repeats, int)
        or response_repeats <= 0
    ):
        raise OracleDataError("FEP command response n_repeats must be a positive integer")
    mismatched = [field for field, value in expected.items() if payload[field] != value]
    if mismatched:
        raise OracleDataError(
            "FEP command response identity does not match request: " + ", ".join(mismatched)
        )


def _validate_fep_batch_request(request) -> None:
    required = {
        "project_id": str(getattr(request, "project_id", "")).strip(),
        "protein_pdb_id": str(getattr(request, "protein_pdb_id", "")).strip(),
        "reference_ligand_smiles": str(getattr(request, "reference_ligand_smiles", "")).strip(),
        "method": str(getattr(request, "method", "")).strip(),
    }
    missing = [field for field, value in required.items() if not value]
    test_ligands = [str(smiles).strip() for smiles in getattr(request, "test_ligand_smiles", [])]
    if not test_ligands or any(not smiles for smiles in test_ligands):
        missing.append("test_ligand_smiles")
    n_repeats = getattr(request, "n_repeats", 0)
    if isinstance(n_repeats, bool) or not isinstance(n_repeats, int) or n_repeats <= 0:
        missing.append("n_repeats")
    if missing:
        raise OracleRequestError("FEP request requires valid: " + ", ".join(missing))


def _validate_fep_results(results: list[fep_pb2.FEPResult], request) -> None:
    requested_ligands = tuple(str(smiles) for smiles in getattr(request, "test_ligand_smiles", []))
    actual_ligands = tuple(result.ligand_b_smiles for result in results)
    if actual_ligands != requested_ligands:
        raise OracleDataError("FEP command results do not match requested test ligands")
    reference = str(getattr(request, "reference_ligand_smiles", ""))
    method = str(getattr(request, "method", ""))
    repeats = int(getattr(request, "n_repeats", 0))
    if any(result.ligand_a_smiles != reference for result in results):
        raise OracleDataError("FEP command result reference ligand does not match request")
    if any(result.method != method for result in results):
        raise OracleDataError("FEP command result method does not match request")
    if any(result.n_repeats != repeats for result in results):
        raise OracleDataError("FEP command result n_repeats does not match request")


def _validate_fep_response_identity(response, request) -> None:
    expected = {
        "batch_id": request.project_id,
        "project_id": request.project_id,
        "protein_pdb_id": request.protein_pdb_id,
        "reference_ligand_smiles": request.reference_ligand_smiles,
        "test_ligand_smiles": tuple(request.test_ligand_smiles),
        "method": request.method,
        "n_repeats": request.n_repeats,
    }
    actual = {
        "batch_id": str(response.batch_id),
        "project_id": str(response.project_id),
        "protein_pdb_id": str(response.protein_pdb_id),
        "reference_ligand_smiles": str(response.reference_ligand_smiles),
        "test_ligand_smiles": tuple(str(item) for item in response.test_ligand_smiles),
        "method": str(response.method),
        "n_repeats": int(response.n_repeats),
    }
    mismatched = [field for field, value in expected.items() if actual[field] != value]
    if mismatched:
        raise OracleDataError(
            "FEP response identity does not match request: " + ", ".join(mismatched)
        )


def _validate_repeat_evidence(per_repeat_ddg, n_repeats: int) -> None:
    actual_keys = set(per_repeat_ddg)
    expected_keys = {f"repeat_{index}" for index in range(1, n_repeats + 1)}
    if actual_keys != expected_keys:
        raise OracleDataError("FEP per_repeat_ddg must contain exactly one value for each repeat")


def _positive_timeout(value: object, field_name: str) -> float:
    try:
        timeout = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{field_name} must be a finite positive number") from exc
    if isinstance(value, bool) or not math.isfinite(timeout) or timeout <= 0:
        raise RuntimeError(f"{field_name} must be a finite positive number")
    return timeout


def _oracle_request_to_fep_batch(request) -> fep_pb2.FEPBatchRequest:
    return fep_pb2.FEPBatchRequest(
        project_id=request.project_id,
        protein_pdb_id=request.protein_pdb_id,
        reference_ligand_smiles=request.reference_ligand_smiles,
        test_ligand_smiles=request.molecules,
        method=request.parameters["method"],
        n_repeats=parse_positive_parameter(request.parameters, "n_repeats"),
    )


def _oracle_evaluation_from_fep_result(
    *,
    request_context,
    index: int,
    result,
    artifacts,
) -> oracle_pb2.OracleEvaluation:
    oracle_name = str(result.method)
    if not result.converged:
        return build_oracle_error_evaluation(
            request=request_context,
            index=index,
            oracle_name=oracle_name,
            elapsed_ms=0,
            artifacts=artifacts,
            error_code="NOT_CONVERGED",
            error_message="FEP calculation did not converge",
        )
    return build_oracle_evaluation(
        request=request_context,
        index=index,
        oracle_name=oracle_name,
        scores={"rbfe": result.ddg_kcal_mol},
        uncertainties={"rbfe": result.ddg_uncertainty},
        elapsed_ms=0,
        artifacts=artifacts,
        units={"rbfe": "kcal/mol"},
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
