"""FEP Service.

gRPC server for OpenFE RBFE calculations.
"""

import asyncio
import json
import logging
import math
import os
import shlex
import shutil
import signal
import statistics
import subprocess
import sys
import time
import uuid
from concurrent import futures
from contextlib import suppress
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
_OPENFE_RUNNER_ENV = "OPENFE_RUNNER_PATH"
_OPENFE_RUNNER_REQUIREMENT = CommandRequirement(
    "openfe_runner_command",
    _OPENFE_RUNNER_ENV,
)
_OPENFE_CLI_ENV = "OPENFE_CLI_PATH"
_OPENFE_CLI_REQUIREMENT = CommandRequirement("openfe_cli_command", _OPENFE_CLI_ENV)
_OPENFE_INPUT_ENVS = (
    ("openfe_result_replay", "OPENFE_RESULT_REPLAY_PATH"),
    ("openfe_result_registry", "OPENFE_RESULT_REGISTRY"),
    ("openfe_transformation_registry", "OPENFE_TRANSFORMATION_REGISTRY"),
)
_FEP_DATA_ERROR_EXIT_CODE = 2
_FEP_TIMEOUT_EXIT_CODE = 124
_UPSTREAM_PROCESS_GROUP_ENV = "_MFORGE_FEP_UPSTREAM_PROCESS_GROUP"
_OPENFE_REQUEST_WORK_DIR_ENV = "_MFORGE_OPENFE_REQUEST_WORK_DIR"
_LOGGER = logging.getLogger(__name__)
_VALIDATION_GATE_ENV = "MF_ALLOW_SYNTHETIC_VALIDATION"
_VALIDATION_MARKER = "synthetic_pipeline_validation_only"
_VALIDATION_MAX_BATCH_SIZE = 256
_VALIDATION_MAX_REPEATS = 64


def _status_objects() -> list[RequirementStatus]:
    command_status = _check_command_with_script(_FEP_COMMAND_REQUIREMENT)
    statuses = [command_status, _fep_concurrency_status()]
    if not _uses_builtin_fep_wrapper():
        return statuses
    statuses.append(_check_command_with_script(_OPENFE_RUNNER_REQUIREMENT))
    input_statuses = _openfe_input_statuses()
    statuses.extend(input_statuses)
    statuses.append(_openfe_timeout_status())
    transformation_configured = bool(
        os.environ.get("OPENFE_TRANSFORMATION_REGISTRY", "").strip()
    )
    if transformation_configured:
        statuses.append(_check_command_with_script(_OPENFE_CLI_REQUIREMENT))
    return statuses


def _require_runtime() -> list[RequirementStatus]:
    statuses = _status_objects()
    require_available(statuses)
    _fep_command_argv()
    return statuses


def _check_command_with_script(
    requirement: CommandRequirement,
) -> RequirementStatus:
    status = check_command(requirement)
    if not status.available:
        return status
    try:
        argv = shlex.split(os.environ.get(requirement.env_var, ""))
    except ValueError:
        return status
    for argument in argv[1:]:
        if not argument.endswith(".py"):
            continue
        script_path = Path(argument).expanduser()
        if script_path.is_file():
            continue
        return RequirementStatus(
            name=status.name,
            configured=True,
            available=False,
            required=status.required,
            path=status.path,
            source=status.source,
            message=f"{status.name} script is not available: {script_path}",
        )
    return status


def _uses_builtin_fep_wrapper() -> bool:
    try:
        argv = shlex.split(os.environ.get(_FEP_COMMAND_ENV, ""))
    except ValueError:
        return False
    return any(Path(argument).name == "fep_oracle_wrapper.py" for argument in argv)


def _openfe_input_statuses() -> list[RequirementStatus]:
    configured = [
        (name, env_name, os.environ.get(env_name, "").strip())
        for name, env_name in _OPENFE_INPUT_ENVS
        if os.environ.get(env_name, "").strip()
    ]
    source_statuses = [
        _openfe_input_file_status(name, env_name, raw_path)
        for name, env_name, raw_path in configured
    ]
    available = bool(source_statuses) and all(
        status.available for status in source_statuses
    )
    aggregate = RequirementStatus(
        name="openfe_input_source",
        configured=bool(configured),
        available=available,
        required=True,
        path=",".join(raw_path for _, _, raw_path in configured) or None,
        source=",".join(env_name for _, env_name, _ in configured)
        or "OPENFE_RESULT_REPLAY_PATH|OPENFE_RESULT_REGISTRY|OPENFE_TRANSFORMATION_REGISTRY",
        message=(
            "OpenFE input source is available"
            if available
            else "at least one valid OpenFE replay, result registry, or "
            "transformation registry is required"
        ),
    )
    return [*source_statuses, aggregate]


def _openfe_input_file_status(
    name: str,
    env_name: str,
    raw_path: str,
) -> RequirementStatus:
    path = Path(raw_path).expanduser()
    message = f"{name} is available"
    available = path.is_file()
    if available:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or not payload:
                raise ValueError("JSON root must be a non-empty object")
            if env_name == "OPENFE_TRANSFORMATION_REGISTRY":
                _require_registry_transformation_files(payload, path.parent)
            elif env_name == "OPENFE_RESULT_REGISTRY":
                _require_result_registry_payload(payload)
            elif env_name == "OPENFE_RESULT_REPLAY_PATH":
                _require_replay_payload(payload)
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            available = False
            message = f"{name} is invalid: {exc}"
    else:
        message = f"{name} is not available: {path}"
    return RequirementStatus(
        name=name,
        configured=True,
        available=available,
        required=True,
        path=str(path),
        source=env_name,
        message=message,
    )


def _require_registry_transformation_files(
    registry: dict,
    registry_directory: Path,
) -> None:
    found = False
    for protein_id, entries in registry.items():
        if (
            not isinstance(protein_id, str)
            or not protein_id
            or protein_id != protein_id.strip()
        ):
            raise ValueError("transformation registry protein identity is invalid")
        if not isinstance(entries, dict) or not entries:
            raise ValueError("transformation registry protein entry must be non-empty")
        for pair_key, value in entries.items():
            if not isinstance(pair_key, str) or pair_key.count(">>") != 1:
                raise ValueError("transformation registry pair identity is invalid")
            ligand_a, ligand_b = pair_key.split(">>")
            if (
                not ligand_a
                or ligand_a != ligand_a.strip()
                or not ligand_b
                or ligand_b != ligand_b.strip()
            ):
                raise ValueError("transformation registry pair identity is invalid")
            paths = _transformation_paths_from_registry_value(value)
            if not paths:
                raise ValueError("transformation registry pair has no transformation path")
            for raw_path in paths:
                path = Path(str(raw_path)).expanduser()
                if not path.is_absolute():
                    path = registry_directory / path
                if not path.resolve().is_file():
                    raise ValueError(f"transformation file is not available: {path}")
                _require_single_protocol_repeat(path.resolve())
                found = True
    if not found:
        raise ValueError("transformation registry contains no transformation files")


def _require_result_registry_payload(registry: dict) -> None:
    for protein_id, entries in registry.items():
        if not isinstance(protein_id, str) or not protein_id.strip():
            raise ValueError("result registry protein identity is invalid")
        if not isinstance(entries, dict) or not entries:
            raise ValueError("result registry protein entry must be non-empty")
        for pair_key, record in entries.items():
            if not isinstance(pair_key, str) or ">>" not in pair_key:
                raise ValueError("result registry pair identity is invalid")
            _require_runtime_result_record(record, "result registry")


def _require_replay_payload(payload: dict) -> None:
    required_identity = (
        "project_id",
        "protein_pdb_id",
        "reference_ligand_smiles",
        "test_ligand_smiles",
        "method",
        "n_repeats",
    )
    missing = [field for field in required_identity if field not in payload]
    if missing:
        raise ValueError("replay missing fields: " + ", ".join(missing))
    for field in required_identity[:3] + ("method",):
        value = payload[field]
        if not isinstance(value, str) or not value or value != value.strip():
            raise ValueError(f"replay {field} is invalid")
    ligands = payload["test_ligand_smiles"]
    if not isinstance(ligands, list) or not ligands or any(
        not isinstance(value, str) or not value or value != value.strip()
        for value in ligands
    ):
        raise ValueError("replay test_ligand_smiles is invalid")
    repeats = payload["n_repeats"]
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats <= 0:
        raise ValueError("replay n_repeats is invalid")
    records = payload.get("results", payload.get("rows"))
    if not isinstance(records, list) or len(records) != len(ligands):
        raise ValueError("replay results must match test_ligand_smiles")
    for index, record in enumerate(records):
        _require_runtime_result_record(record, "replay")
        if (
            record["ligand_a_smiles"] != payload["reference_ligand_smiles"]
            or record["ligand_b_smiles"] != ligands[index]
            or record["method"] != payload["method"]
            or record["n_repeats"] != repeats
        ):
            raise ValueError("replay result identity does not match replay metadata")


def _require_runtime_result_record(record: object, label: str) -> None:
    if not isinstance(record, dict):
        raise ValueError(f"{label} result must be an object")
    required = (
        "ligand_a_smiles",
        "ligand_b_smiles",
        "ddg_kcal_mol",
        "ddg_uncertainty",
        "n_repeats",
        "method",
        "per_repeat_ddg",
        "converged",
    )
    missing = [field for field in required if field not in record]
    if missing:
        raise ValueError(f"{label} result missing fields: " + ", ".join(missing))
    for field in ("ligand_a_smiles", "ligand_b_smiles", "method"):
        value = record[field]
        if not isinstance(value, str) or not value or value != value.strip():
            raise ValueError(f"{label} result {field} is invalid")
    ddg = _runtime_finite_number(record["ddg_kcal_mol"], f"{label} ddg")
    uncertainty = _runtime_finite_number(
        record["ddg_uncertainty"],
        f"{label} uncertainty",
    )
    if uncertainty < 0:
        raise ValueError(f"{label} uncertainty must be non-negative")
    repeats = record["n_repeats"]
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats <= 0:
        raise ValueError(f"{label} n_repeats is invalid")
    if not isinstance(record["converged"], bool):
        raise ValueError(f"{label} converged must be a boolean")
    per_repeat = record["per_repeat_ddg"]
    expected_keys = {f"repeat_{index}" for index in range(1, repeats + 1)}
    if not isinstance(per_repeat, dict) or set(per_repeat) != expected_keys:
        raise ValueError(f"{label} per_repeat_ddg does not match n_repeats")
    values = [
        _runtime_finite_number(per_repeat[key], f"{label} {key}")
        for key in sorted(per_repeat)
    ]
    if not math.isclose(
        sum(values) / repeats,
        ddg,
        rel_tol=1e-6,
        abs_tol=1e-6,
    ):
        raise ValueError(f"{label} per_repeat_ddg mean does not match ddg")


def _runtime_finite_number(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field_name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field_name} must be finite")
    return number


def _transformation_paths_from_registry_value(value: object) -> list[object]:
    if isinstance(value, str):
        raise ValueError(
            "RBFE transformation registry requires explicit complex and solvent paths"
        )
    if isinstance(value, list):
        raise ValueError(
            "RBFE transformation registry requires explicit complex and solvent paths"
        )
    if not isinstance(value, dict):
        return []
    complex_path = value.get("complex")
    solvent_path = value.get("solvent")
    if complex_path or solvent_path:
        if not complex_path or not solvent_path:
            raise ValueError(
                "RBFE transformation registry requires both complex and solvent paths"
            )
        return [complex_path, solvent_path]
    raise ValueError(
        "RBFE transformation registry requires explicit complex and solvent paths"
    )


def _require_single_protocol_repeat(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    protocol = payload.get("protocol") if isinstance(payload, dict) else None
    settings = protocol.get("settings") if isinstance(protocol, dict) else None
    if not isinstance(settings, dict):
        raise ValueError(
            "transformation requires protocol settings with protocol_repeats=1"
        )
    repeats = settings.get("protocol_repeats", settings.get("n_repeats"))
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats != 1:
        raise ValueError("transformation must set protocol_repeats=1")


def _openfe_timeout_status() -> RequirementStatus:
    try:
        quickrun_timeout = _positive_timeout(
            os.environ.get("OPENFE_QUICKRUN_TIMEOUT_SECONDS", "3600"),
            "OPENFE_QUICKRUN_TIMEOUT_SECONDS",
        )
        gather_timeout = _positive_timeout(
            os.environ.get("OPENFE_GATHER_TIMEOUT_SECONDS", "600"),
            "OPENFE_GATHER_TIMEOUT_SECONDS",
        )
        maximum_transformations = _positive_integer(
            os.environ.get("OPENFE_MAX_TRANSFORMATIONS_PER_PAIR", "2"),
            "OPENFE_MAX_TRANSFORMATIONS_PER_PAIR",
        )
        minimum_runner = (
            maximum_transformations * quickrun_timeout + gather_timeout + 60.0
        )
        raw_runner = os.environ.get("OPENFE_RUNNER_TIMEOUT_SECONDS", "").strip()
        runner_timeout = (
            _positive_timeout(raw_runner, "OPENFE_RUNNER_TIMEOUT_SECONDS")
            if raw_runner
            else minimum_runner
        )
        if runner_timeout < minimum_runner:
            raise RuntimeError(
                "OPENFE_RUNNER_TIMEOUT_SECONDS is shorter than the minimum "
                f"OpenFE execution budget ({minimum_runner:g}s)"
            )
        minimum_fep = runner_timeout + 30.0
        raw_fep = os.environ.get("FEP_ORACLE_TIMEOUT_SECONDS", "").strip()
        if raw_fep:
            fep_timeout = _positive_timeout(raw_fep, "FEP_ORACLE_TIMEOUT_SECONDS")
            if fep_timeout < minimum_fep:
                raise RuntimeError(
                    "FEP_ORACLE_TIMEOUT_SECONDS is shorter than the minimum "
                    f"FEP execution budget ({minimum_fep:g}s)"
                )
    except RuntimeError as exc:
        return RequirementStatus(
            name="openfe_timeout_configuration",
            configured=True,
            available=False,
            required=True,
            path=None,
            source=(
                "FEP_ORACLE_TIMEOUT_SECONDS|OPENFE_RUNNER_TIMEOUT_SECONDS|"
                "OPENFE_QUICKRUN_TIMEOUT_SECONDS|OPENFE_GATHER_TIMEOUT_SECONDS|"
                "OPENFE_MAX_TRANSFORMATIONS_PER_PAIR"
            ),
            message=str(exc),
        )
    return RequirementStatus(
        name="openfe_timeout_configuration",
        configured=True,
        available=True,
        required=True,
        path=None,
        source=(
            "FEP_ORACLE_TIMEOUT_SECONDS|OPENFE_RUNNER_TIMEOUT_SECONDS|"
            "OPENFE_QUICKRUN_TIMEOUT_SECONDS|OPENFE_GATHER_TIMEOUT_SECONDS|"
            "OPENFE_MAX_TRANSFORMATIONS_PER_PAIR"
        ),
        message="OpenFE timeout hierarchy is valid",
    )


def _fep_concurrency_status() -> RequirementStatus:
    try:
        maximum = _positive_integer(
            os.environ.get("FEP_MAX_CONCURRENT_JOBS", "1"),
            "FEP_MAX_CONCURRENT_JOBS",
        )
    except RuntimeError as exc:
        return RequirementStatus(
            name="fep_concurrency_configuration",
            configured=True,
            available=False,
            required=True,
            path=None,
            source="FEP_MAX_CONCURRENT_JOBS",
            message=str(exc),
        )
    return RequirementStatus(
        name="fep_concurrency_configuration",
        configured=True,
        available=True,
        required=True,
        path=None,
        source="FEP_MAX_CONCURRENT_JOBS",
        message=f"FEP execution concurrency is limited to {maximum}",
    )


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


async def _abort_internal(context, error: RuntimeError):
    if context is not None and hasattr(context, "abort"):
        await context.abort(grpc.StatusCode.INTERNAL, str(error) or type(error).__name__)
    raise error


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
    request_id: str = "",
    batch_id: str = "",
) -> fep_pb2.FEPJobStatus:
    status = fep_pb2.FEPJobStatus(
        job_id=job_id,
        state=state,
        error=error,
        submitted_at_ms=submitted_at_ms,
        started_at_ms=started_at_ms,
        completed_at_ms=completed_at_ms,
        request_id=request_id,
        batch_id=batch_id,
    )
    if response is not None:
        status.response.CopyFrom(response)
    return status


class FEPServicer:
    def __init__(
        self,
        job_dir: str | os.PathLike[str] | None = None,
        *,
        max_concurrent_jobs: int | None = None,
    ):
        self.job_dir = Path(job_dir or os.environ.get("FEP_JOB_DIR", "runs/fep-jobs"))
        maximum = _positive_integer(
            (
                os.environ.get("FEP_MAX_CONCURRENT_JOBS", "1")
                if max_concurrent_jobs is None
                else max_concurrent_jobs
            ),
            "FEP_MAX_CONCURRENT_JOBS",
        )
        self._execution_semaphore = asyncio.Semaphore(maximum)
        self._tasks: dict[str, asyncio.Task] = {}

    async def RunFEP(self, request, context):
        """Run Free Energy Perturbation calculation."""
        try:
            _validate_fep_batch_request(request)
        except OracleRequestError as exc:
            return await abort_oracle_error(context, exc)
        try:
            _require_runtime()
        except RuntimeError:
            return await _abort_unavailable(context)
        if os.environ.get(_FEP_COMMAND_ENV):
            try:
                async with self._execution_semaphore:
                    return await _run_fep_command_async(request)
            except (
                OracleDataError,
                TimeoutError,
                subprocess.TimeoutExpired,
            ) as exc:
                return await abort_oracle_error(context, exc)
            except RuntimeError as exc:
                return await _abort_internal(context, exc)
        raise RuntimeError("OpenFE runner is not configured")

    async def SubmitFEP(self, request, context):
        """Submit a Free Energy Perturbation calculation as a background job."""
        try:
            _validate_fep_batch_request(request)
        except OracleRequestError as exc:
            return await abort_oracle_error(context, exc)
        try:
            _require_runtime()
        except RuntimeError:
            return await _abort_unavailable(context)
        job_id = uuid.uuid4().hex
        status = _job_status(
            job_id,
            "queued",
            submitted_at_ms=_epoch_ms(),
            request_id=request.request_id,
            batch_id=request.batch_id,
        )
        self._write_job_status(status)
        job_request = fep_pb2.FEPBatchRequest()
        job_request.CopyFrom(request)
        task = asyncio.create_task(self._run_job(job_id, job_request))
        self._tasks[job_id] = task
        task.add_done_callback(lambda _task, job_key=job_id: self._tasks.pop(job_key, None))
        return status

    async def GetStatus(self, request, context):
        """Get status of a running FEP job."""
        try:
            return self._read_job_status(str(getattr(request, "job_id", "")))
        except (OracleRequestError, OracleDataError) as exc:
            return await abort_oracle_error(context, exc)
        except RuntimeError as exc:
            if context is not None and hasattr(context, "abort"):
                await context.abort(grpc.StatusCode.NOT_FOUND, str(exc))
            raise

    async def _run_job(self, job_id: str, request) -> None:
        async with self._execution_semaphore:
            started_at_ms = _epoch_ms()
            queued = self._read_job_status(job_id)
            self._write_job_status(
                _job_status(
                    job_id,
                    "running",
                    submitted_at_ms=queued.submitted_at_ms,
                    started_at_ms=started_at_ms,
                    request_id=queued.request_id,
                    batch_id=queued.batch_id,
                )
            )
            try:
                response = await _run_fep_command_async(request)
            except Exception as exc:
                submitted_at_ms = self._read_job_status(job_id).submitted_at_ms
                self._write_job_status(
                    _job_status(
                        job_id,
                        "failed",
                        error=str(exc) or type(exc).__name__,
                        submitted_at_ms=submitted_at_ms,
                        started_at_ms=started_at_ms,
                        completed_at_ms=_epoch_ms(),
                        request_id=request.request_id,
                        batch_id=request.batch_id,
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
                    request_id=request.request_id,
                    batch_id=request.batch_id,
                )
            )

    def _job_path(self, job_id: str) -> Path:
        if not job_id or not all(ch.isalnum() or ch in {"-", "_"} for ch in job_id):
            raise OracleRequestError("FEP job_id is invalid")
        return self.job_dir / f"{job_id}.json"

    def _write_job_status(self, status: fep_pb2.FEPJobStatus) -> None:
        _validate_fep_job_status(status, expected_job_id=status.job_id)
        self.job_dir.mkdir(parents=True, exist_ok=True)
        path = self._job_path(status.job_id)
        tmp_path = path.with_suffix(".json.tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            handle.write(MessageToJson(status, preserving_proto_field_name=True))
            handle.flush()
            os.fsync(handle.fileno())
        tmp_path.replace(path)
        directory_fd = os.open(self.job_dir, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)

    def _read_job_status(self, job_id: str) -> fep_pb2.FEPJobStatus:
        path = self._job_path(job_id)
        if not path.is_file():
            raise RuntimeError(f"FEP job not found: {job_id}")
        try:
            status = Parse(path.read_text(encoding="utf-8"), fep_pb2.FEPJobStatus())
        except Exception as exc:
            raise OracleDataError(f"FEP job status is invalid: {job_id}") from exc
        _validate_fep_job_status(status, expected_job_id=job_id)
        return status

    def recover_interrupted_jobs(self) -> int:
        if not self.job_dir.is_dir():
            return 0
        recovered = 0
        for path in sorted(self.job_dir.glob("*.json")):
            try:
                status = self._read_job_status(path.stem)
            except (OracleRequestError, OracleDataError) as exc:
                status = self._migrate_recoverable_job_status(path)
                if status is None:
                    quarantined_path = self._quarantine_job_status(path)
                    _LOGGER.error(
                        "Quarantined invalid FEP job status %s as %s: %s",
                        path,
                        quarantined_path,
                        exc,
                    )
                    continue
            if status.state not in {"queued", "running"}:
                continue
            now = _epoch_ms()
            started_at_ms = status.started_at_ms or max(status.submitted_at_ms, now)
            self._write_job_status(
                _job_status(
                    status.job_id,
                    "failed",
                    error="FEP service restarted before job completion",
                    submitted_at_ms=status.submitted_at_ms,
                    started_at_ms=started_at_ms,
                    completed_at_ms=max(started_at_ms, now),
                    request_id=status.request_id,
                    batch_id=status.batch_id,
                )
            )
            recovered += 1
        return recovered

    def _migrate_recoverable_job_status(
        self,
        path: Path,
    ) -> fep_pb2.FEPJobStatus | None:
        try:
            status = Parse(
                path.read_text(encoding="utf-8"),
                fep_pb2.FEPJobStatus(),
            )
        except Exception:
            return None
        if status.job_id != path.stem or not status.HasField("response"):
            return None
        request_id = str(status.response.request_id)
        batch_id = str(status.response.batch_id)
        if (
            not request_id
            or request_id != request_id.strip()
            or not batch_id
            or batch_id != batch_id.strip()
            or status.request_id not in {"", request_id}
            or status.batch_id not in {"", batch_id}
        ):
            return None
        status.request_id = request_id
        status.batch_id = batch_id
        try:
            _validate_fep_job_status(status, expected_job_id=path.stem)
        except OracleDataError:
            return None
        self._write_job_status(status)
        return status

    def _quarantine_job_status(self, path: Path) -> Path:
        destination = path.with_name(f"{path.name}.invalid")
        if destination.exists():
            destination = path.with_name(f"{path.name}.invalid-{uuid.uuid4().hex}")
        path.replace(destination)
        directory_fd = os.open(self.job_dir, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return destination


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
                    None,
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
                    _validate_repeat_evidence(
                        result.per_repeat_ddg,
                        result.n_repeats,
                        result.ddg_kcal_mol,
                    )
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


async def _run_fep_command_async(request) -> fep_pb2.FEPBatchResponse:
    command = _fep_command_argv()
    timeout = _fep_timeout_seconds(request)
    child_env = os.environ.copy()
    child_env[_UPSTREAM_PROCESS_GROUP_ENV] = "1"
    request_work_path = _openfe_request_work_path(child_env)
    process = None
    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            env=child_env,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        stdout, stderr = await asyncio.wait_for(
            process.communicate(
                json.dumps(_fep_request_payload(request)).encode("utf-8")
            ),
            timeout=timeout,
        )
    except TimeoutError as exc:
        raise subprocess.TimeoutExpired(command, timeout) from exc
    except asyncio.CancelledError:
        raise
    finally:
        if process is not None:
            await asyncio.shield(_terminate_async_process_group(process))
        if request_work_path is not None:
            await asyncio.shield(
                asyncio.to_thread(
                    _remove_openfe_request_work_directory,
                    request_work_path,
                )
            )
    result = subprocess.CompletedProcess(
        command,
        process.returncode,
        stdout.decode("utf-8", errors="replace"),
        stderr.decode("utf-8", errors="replace"),
    )
    return _fep_response_from_completed_process(result, request)


def _openfe_request_work_path(child_env: dict[str, str]) -> Path | None:
    if not _uses_builtin_fep_wrapper():
        return None
    raw_base = os.environ.get("OPENFE_WORK_DIR", "").strip()
    if not raw_base:
        return None
    base_path = Path(raw_base).expanduser()
    base_path.mkdir(parents=True, exist_ok=True)
    request_path = base_path / f"mforge-openfe-{uuid.uuid4().hex}"
    child_env[_OPENFE_REQUEST_WORK_DIR_ENV] = str(request_path)
    return request_path


def _remove_openfe_request_work_directory(path: Path) -> bool:
    if not path.exists() and not path.is_symlink():
        return False
    try:
        if path.is_symlink():
            path.unlink()
        else:
            shutil.rmtree(path)
    except OSError as exc:
        _LOGGER.error("Failed to remove OpenFE request work directory %s: %s", path, exc)
        return False
    return True


def _cleanup_stale_openfe_work_directories() -> int:
    raw_base = os.environ.get("OPENFE_WORK_DIR", "").strip()
    if not raw_base:
        return 0
    base_path = Path(raw_base).expanduser()
    if not base_path.is_dir():
        return 0
    removed = 0
    for path in base_path.glob("mforge-openfe-*"):
        removed += int(_remove_openfe_request_work_directory(path))
    return removed


async def _terminate_async_process_group(
    process: asyncio.subprocess.Process,
) -> None:
    with suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGTERM)
    deadline = time.monotonic() + 5.0
    while process.returncode is None and time.monotonic() < deadline:
        await asyncio.sleep(0.05)
    with suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGKILL)
    await process.communicate()


def _fep_response_from_completed_process(
    result: subprocess.CompletedProcess[str],
    request,
) -> fep_pb2.FEPBatchResponse:
    if result.returncode == _FEP_DATA_ERROR_EXIT_CODE:
        raise OracleDataError(result.stderr.strip() or "FEP command returned invalid data")
    if result.returncode == _FEP_TIMEOUT_EXIT_CODE:
        raise TimeoutError(result.stderr.strip() or "FEP command timed out")
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
        request_id=str(payload["request_id"]),
    )


def _fep_timeout_seconds(request) -> float:
    if _uses_builtin_fep_wrapper():
        quickrun_timeout = _positive_timeout(
            os.environ.get("OPENFE_QUICKRUN_TIMEOUT_SECONDS", "3600"),
            "OPENFE_QUICKRUN_TIMEOUT_SECONDS",
        )
        gather_timeout = _positive_timeout(
            os.environ.get("OPENFE_GATHER_TIMEOUT_SECONDS", "600"),
            "OPENFE_GATHER_TIMEOUT_SECONDS",
        )
        maximum_transformations = _positive_integer(
            os.environ.get("OPENFE_MAX_TRANSFORMATIONS_PER_PAIR", "2"),
            "OPENFE_MAX_TRANSFORMATIONS_PER_PAIR",
        )
        runner_minimum = (
            int(getattr(request, "n_repeats", 0))
            * len(getattr(request, "test_ligand_smiles", []))
            * (maximum_transformations * quickrun_timeout + gather_timeout)
            + 60.0
        )
        configured_runner = os.environ.get("OPENFE_RUNNER_TIMEOUT_SECONDS", "").strip()
        runner_timeout = (
            _positive_timeout(
                configured_runner,
                "OPENFE_RUNNER_TIMEOUT_SECONDS",
            )
            if configured_runner
            else runner_minimum
        )
        if runner_timeout < runner_minimum:
            raise RuntimeError(
                "OPENFE_RUNNER_TIMEOUT_SECONDS is shorter than the configured "
                f"OpenFE execution budget ({runner_minimum:g}s)"
            )
        minimum = runner_timeout + 30.0
    configured = os.environ.get("FEP_ORACLE_TIMEOUT_SECONDS", "").strip()
    if not _uses_builtin_fep_wrapper():
        return (
            _positive_timeout(configured, "FEP_ORACLE_TIMEOUT_SECONDS")
            if configured
            else 120.0
        )
    if not configured:
        return minimum
    timeout = _positive_timeout(configured, "FEP_ORACLE_TIMEOUT_SECONDS")
    if timeout < minimum:
        raise RuntimeError(
            "FEP_ORACLE_TIMEOUT_SECONDS is shorter than the configured "
            f"FEP execution budget ({minimum:g}s)"
        )
    return timeout


def _positive_integer(value: object, field_name: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{field_name} must be a positive integer") from exc
    if isinstance(value, bool) or str(number) != str(value).strip() or number <= 0:
        raise RuntimeError(f"{field_name} must be a positive integer")
    return number


def _fep_request_payload(request) -> dict:
    return {
        "project_id": str(getattr(request, "project_id", "")),
        "request_id": str(getattr(request, "request_id", "")),
        "batch_id": str(getattr(request, "batch_id", "")),
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
    if uncertainty < 0:
        raise OracleDataError("FEP ddg_uncertainty must be non-negative")
    n_repeats = record["n_repeats"]
    if isinstance(n_repeats, bool) or not isinstance(n_repeats, int) or n_repeats <= 0:
        raise OracleDataError("FEP n_repeats must be a positive integer")
    if not isinstance(record["converged"], bool):
        raise OracleDataError("FEP converged must be a boolean")
    per_repeat_ddg = _numeric_mapping(record.get("per_repeat_ddg"))
    _validate_repeat_evidence(per_repeat_ddg, n_repeats, ddg)
    model_version = (
        _VALIDATION_MARKER
        if record.get("validation_marker") == _VALIDATION_MARKER
        else str(record.get("model_version") or record.get("validation_marker") or "")
    )
    return fep_pb2.FEPResult(
        ligand_a_smiles=str(record["ligand_a_smiles"]),
        ligand_b_smiles=str(record["ligand_b_smiles"]),
        ddg_kcal_mol=ddg,
        ddg_uncertainty=uncertainty,
        n_repeats=n_repeats,
        method=str(record["method"]),
        per_repeat_ddg=per_repeat_ddg,
        converged=record["converged"],
        model_version=model_version,
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
        "request_id": str(getattr(request, "request_id", "")),
        "batch_id": str(getattr(request, "batch_id", "")),
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
        "project_id": str(getattr(request, "project_id", "")),
        "request_id": str(getattr(request, "request_id", "")),
        "batch_id": str(getattr(request, "batch_id", "")),
        "protein_pdb_id": str(getattr(request, "protein_pdb_id", "")),
        "reference_ligand_smiles": str(getattr(request, "reference_ligand_smiles", "")),
        "method": str(getattr(request, "method", "")),
    }
    invalid = [
        field for field, value in required.items() if not value or value != value.strip()
    ]
    test_ligands = [str(smiles) for smiles in getattr(request, "test_ligand_smiles", [])]
    if not test_ligands or any(
        not smiles or smiles != smiles.strip() for smiles in test_ligands
    ):
        invalid.append("test_ligand_smiles")
    n_repeats = getattr(request, "n_repeats", 0)
    if isinstance(n_repeats, bool) or not isinstance(n_repeats, int) or n_repeats <= 0:
        invalid.append("n_repeats")
    if invalid:
        raise OracleRequestError("FEP request requires valid: " + ", ".join(invalid))


def _validate_fep_job_status(status, *, expected_job_id: str) -> None:
    identity = {
        "job_id": str(getattr(status, "job_id", "")),
        "request_id": str(getattr(status, "request_id", "")),
        "batch_id": str(getattr(status, "batch_id", "")),
    }
    if any(not value or value != value.strip() for value in identity.values()):
        raise OracleDataError("FEP job status identity is invalid")
    if identity["job_id"] != expected_job_id:
        raise OracleDataError("FEP job status job_id does not match the requested job")
    state = str(getattr(status, "state", ""))
    if state not in {"queued", "running", "completed", "failed"}:
        raise OracleDataError("FEP job status state is invalid")
    submitted_at_ms = int(getattr(status, "submitted_at_ms", 0))
    started_at_ms = int(getattr(status, "started_at_ms", 0))
    completed_at_ms = int(getattr(status, "completed_at_ms", 0))
    if submitted_at_ms <= 0:
        raise OracleDataError("FEP job status submitted_at_ms is invalid")
    if state == "queued":
        timestamps_valid = started_at_ms == 0 and completed_at_ms == 0
    elif state == "running":
        timestamps_valid = started_at_ms >= submitted_at_ms and completed_at_ms == 0
    else:
        timestamps_valid = (
            started_at_ms >= submitted_at_ms and completed_at_ms >= started_at_ms
        )
    if not timestamps_valid:
        raise OracleDataError("FEP job status timestamps are invalid")
    has_response = status.HasField("response")
    error = str(getattr(status, "error", ""))
    if state == "completed":
        if not has_response or error:
            raise OracleDataError("completed FEP job status requires only a response")
        if (
            status.response.request_id != identity["request_id"]
            or status.response.batch_id != identity["batch_id"]
        ):
            raise OracleDataError("FEP job response identity does not match the job status")
    elif state == "failed":
        if has_response or not error:
            raise OracleDataError("failed FEP job status requires only an error")
    elif has_response or error:
        raise OracleDataError("pending FEP job status cannot contain a response or error")


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
        "request_id": request.request_id,
        "batch_id": request.batch_id,
        "project_id": request.project_id,
        "protein_pdb_id": request.protein_pdb_id,
        "reference_ligand_smiles": request.reference_ligand_smiles,
        "test_ligand_smiles": tuple(request.test_ligand_smiles),
        "method": request.method,
        "n_repeats": request.n_repeats,
    }
    actual = {
        "request_id": str(response.request_id),
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


def _validate_repeat_evidence(
    per_repeat_ddg,
    n_repeats: int,
    ddg_kcal_mol: float,
) -> None:
    actual_keys = set(per_repeat_ddg)
    expected_keys = {f"repeat_{index}" for index in range(1, n_repeats + 1)}
    if actual_keys != expected_keys:
        raise OracleDataError("FEP per_repeat_ddg must contain exactly one value for each repeat")
    repeat_mean = sum(float(value) for value in per_repeat_ddg.values()) / n_repeats
    if not math.isclose(repeat_mean, float(ddg_kcal_mol), rel_tol=1e-6, abs_tol=1e-6):
        raise OracleDataError("FEP per_repeat_ddg mean must match ddg_kcal_mol")


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
        request_id=request.request_id,
        batch_id=request.request_id,
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
            model_version=str(result.model_version),
        )
    return build_oracle_evaluation(
        request=request_context,
        index=index,
        oracle_name=oracle_name,
        scores={"rbfe": result.ddg_kcal_mol},
        uncertainties={"rbfe": result.ddg_uncertainty},
        elapsed_ms=0,
        artifacts=artifacts,
        model_version=str(result.model_version),
        units={"rbfe": "kcal/mol"},
    )


def register_grpc_services(server, service: FEPServicer | None = None) -> None:
    oracle_pb2_grpc.add_OracleServiceServicer_to_server(
        FEPOracleServicer(service=service),
        server,
    )


async def serve():
    _require_runtime()
    _cleanup_stale_openfe_work_directories()
    service = FEPServicer()
    service.recover_interrupted_jobs()
    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=8))
    fep_pb2_grpc.add_FEPServiceServicer_to_server(service, server)
    register_grpc_services(server, service)
    server.add_insecure_port("[::]:50055")
    await server.start()
    _LOGGER.info("FEP Service running on :50055")
    await server.wait_for_termination()


def _validation_response(payload: object) -> dict:
    _require_synthetic_validation_enabled()
    if not isinstance(payload, dict):
        raise ValueError("FEP validation request must be a JSON object")
    expected_fields = {
        "project_id",
        "request_id",
        "batch_id",
        "protein_pdb_id",
        "reference_ligand_smiles",
        "test_ligand_smiles",
        "method",
        "n_repeats",
    }
    unexpected = sorted(set(payload) - expected_fields)
    if unexpected:
        raise ValueError(
            "FEP validation request has unexpected fields: " + ", ".join(unexpected)
        )
    missing = sorted(expected_fields - set(payload))
    if missing:
        raise ValueError(
            "FEP validation request is missing fields: " + ", ".join(missing)
        )
    identity = {
        field_name: _validation_text(payload[field_name], field_name)
        for field_name in (
            "project_id",
            "request_id",
            "batch_id",
            "protein_pdb_id",
            "reference_ligand_smiles",
            "method",
        )
    }
    test_ligands = _validation_text_list(
        payload["test_ligand_smiles"],
        "test_ligand_smiles",
        maximum=_VALIDATION_MAX_BATCH_SIZE,
    )
    n_repeats = payload["n_repeats"]
    if (
        isinstance(n_repeats, bool)
        or not isinstance(n_repeats, int)
        or n_repeats <= 0
        or n_repeats > _VALIDATION_MAX_REPEATS
    ):
        raise ValueError(
            "FEP validation n_repeats must be a positive integer "
            f"not greater than {_VALIDATION_MAX_REPEATS}"
        )
    results = [
        _validation_fep_result(
            reference_ligand_smiles=identity["reference_ligand_smiles"],
            test_ligand_smiles=test_ligand_smiles,
            method=identity["method"],
            n_repeats=n_repeats,
        )
        for test_ligand_smiles in test_ligands
    ]
    return {
        **identity,
        "test_ligand_smiles": test_ligands,
        "n_repeats": n_repeats,
        "total_elapsed_ms": 0,
        "results": results,
        "validation_marker": _VALIDATION_MARKER,
    }


def _validation_fep_result(
    *,
    reference_ligand_smiles: str,
    test_ligand_smiles: str,
    method: str,
    n_repeats: int,
) -> dict:
    fingerprint = _validation_fingerprint(
        f"{reference_ligand_smiles}|{test_ligand_smiles}|{method}"
    )
    center_value = ((fingerprint % 401) - 200) / 100.0
    repeat_center = (n_repeats - 1) / 2.0
    repeat_values = [
        round(center_value + 0.1 * (repeat_index - repeat_center), 6)
        for repeat_index in range(n_repeats)
    ]
    ddg = sum(repeat_values) / n_repeats
    return {
        "ligand_a_smiles": reference_ligand_smiles,
        "ligand_b_smiles": test_ligand_smiles,
        "ddg_kcal_mol": ddg,
        "ddg_uncertainty": statistics.pstdev(repeat_values),
        "n_repeats": n_repeats,
        "method": method,
        "per_repeat_ddg": {
            f"repeat_{repeat_index}": repeat_value
            for repeat_index, repeat_value in enumerate(repeat_values, start=1)
        },
        "converged": True,
        "validation_marker": _VALIDATION_MARKER,
    }


def _validation_fingerprint(value: str) -> int:
    return sum(index * ord(character) for index, character in enumerate(value, start=1))


def _validation_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(
            f"FEP validation {field_name} must be a non-empty trimmed string"
        )
    return value


def _validation_text_list(value: object, field_name: str, *, maximum: int) -> list[str]:
    if not isinstance(value, list) or not value or len(value) > maximum:
        raise ValueError(
            f"FEP validation {field_name} must be a non-empty list "
            f"with at most {maximum} items"
        )
    if any(
        not isinstance(item, str)
        or not item
        or item != item.strip()
        for item in value
    ):
        raise ValueError(
            f"FEP validation {field_name} must contain non-empty trimmed strings"
        )
    return list(value)


def _require_synthetic_validation_enabled() -> None:
    if os.environ.get(_VALIDATION_GATE_ENV) != "true":
        raise RuntimeError(f"{_VALIDATION_GATE_ENV}=true is required")


def _run_validation_runner() -> None:
    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError as exc:
        raise ValueError("FEP validation request must be valid JSON") from exc
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
        sys.stderr.write("FEP service has unexpected command line arguments\n")
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
