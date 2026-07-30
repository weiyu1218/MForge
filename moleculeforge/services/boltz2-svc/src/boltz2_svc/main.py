"""Boltz-2 Binding Affinity Service.

gRPC server for protein-ligand affinity prediction.
"""

import asyncio
import inspect
import json
import logging
import math
import os
import re
import shlex
import signal
import statistics
import subprocess
import sys
import time
from concurrent import futures
from contextlib import suppress
from pathlib import Path
from typing import Any

import grpc
import yaml
from mf_core.artifacts import (
    ArtifactRequirement,
    CommandRequirement,
    PythonPackageRequirement,
    RequirementStatus,
    ToolRequirement,
    check_artifact,
    check_command,
    check_python_package,
    check_tool,
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
    boltz2_pb2,
    boltz2_pb2_grpc,
    oracle_pb2,
    oracle_pb2_grpc,
)

_ARTIFACTS = (
    ArtifactRequirement("boltz_model", "BOLTZ_MODEL_PATH", kind="path"),
    ArtifactRequirement(
        "boltz_input_templates",
        "BOLTZ_INPUT_TEMPLATE_DIR",
        kind="directory",
    ),
)
_TOOLS = (ToolRequirement("boltz", executable="boltz", env_var="BOLTZ_BINARY"),)
_PACKAGES = (
    PythonPackageRequirement("rdkit", module="rdkit"),
    PythonPackageRequirement("pyyaml", module="yaml"),
)
_COMMAND_ENV = "BOLTZ2_ORACLE_COMMAND"
_COMMAND_TIMEOUT_ENV = "BOLTZ2_ORACLE_TIMEOUT_SECONDS"
_COMMAND_REQUIREMENT = CommandRequirement("boltz2_oracle_command", _COMMAND_ENV)
_VALIDATION_GATE_ENV = "MF_ALLOW_SYNTHETIC_VALIDATION"
_VALIDATION_MARKER = "synthetic_pipeline_validation_only"
_LOGGER = logging.getLogger(__name__)
_VALIDATION_MAX_BATCH_SIZE = 256
_VALIDATION_MAX_ENSEMBLE_SIZE = 64


def _status_objects() -> list[RequirementStatus]:
    command_status = _command_status()
    if command_status.configured:
        return [command_status, _timeout_status()]
    return [
        *(check_artifact(requirement) for requirement in _ARTIFACTS),
        *(check_tool(requirement) for requirement in _TOOLS),
        *(check_python_package(requirement) for requirement in _PACKAGES),
        _timeout_status(),
    ]


def _artifact_status_objects() -> list[RequirementStatus]:
    command_status = _command_status()
    if command_status.configured:
        return [command_status]
    return [
        *(check_artifact(requirement) for requirement in _ARTIFACTS),
        *(check_tool(requirement) for requirement in _TOOLS),
    ]


def _require_runtime() -> list[RequirementStatus]:
    statuses = _status_objects()
    require_available(statuses)
    return statuses


def runtime_status() -> list[dict]:
    return [status.to_dict() for status in _status_objects()]


def _command_status() -> RequirementStatus:
    command = os.environ.get(_COMMAND_ENV, "").strip()
    if command:
        return check_command(_COMMAND_REQUIREMENT)
    return RequirementStatus(
        name=_COMMAND_REQUIREMENT.name,
        configured=False,
        available=False,
        required=False,
        path=None,
        source=_COMMAND_ENV,
        message=f"{_COMMAND_ENV} is not configured",
    )


def _timeout_status() -> RequirementStatus:
    raw_value = os.environ.get(_COMMAND_TIMEOUT_ENV, "300")
    try:
        value = float(raw_value)
    except ValueError:
        value = 0.0
    available = math.isfinite(value) and value > 0
    return RequirementStatus(
        name="boltz2_oracle_timeout",
        configured=True,
        available=available,
        required=True,
        path=None,
        source=_COMMAND_TIMEOUT_ENV,
        message=(
            f"{_COMMAND_TIMEOUT_ENV} is configured"
            if available
            else f"{_COMMAND_TIMEOUT_ENV} must be a finite positive number"
        ),
    )


def _require_command_available(command: str) -> None:
    env = {**os.environ, _COMMAND_ENV: command}
    require_available([check_command(_COMMAND_REQUIREMENT, env=env)])


async def _abort_unavailable(context):
    statuses = _status_objects()
    try:
        require_available(statuses)
    except RuntimeError as exc:
        message = str(exc)
    else:
        message = "Boltz-2 runner is not configured"
    if context is not None and hasattr(context, "abort"):
        await context.abort(grpc.StatusCode.FAILED_PRECONDITION, message)
    raise RuntimeError(message)


class Boltz2Servicer:
    def __init__(self, runner=None):
        self.runner = runner

    async def PredictAffinity(self, request, context):
        """Run Boltz-2 binding affinity prediction for a protein-ligand complex."""
        protein_pdb_id = str(getattr(request, "protein_pdb_id", "") or "")
        ligand_smiles = [str(item) for item in getattr(request, "ligand_smiles", []) if item]
        if not protein_pdb_id:
            return await _abort_invalid_argument(context, "protein_pdb_id is required")
        if not ligand_smiles:
            return await _abort_invalid_argument(context, "ligand_smiles is required")
        ensemble_size = int(getattr(request, "ensemble_size", 0) or 5)
        start = time.perf_counter()
        try:
            runner = self._runner()
        except RuntimeError as exc:
            return await _abort_message(context, str(exc))
        predict_affinity = runner.predict_affinity
        if inspect.iscoroutinefunction(predict_affinity):
            rows = await predict_affinity(
                protein_pdb_id,
                ligand_smiles,
                ensemble_size,
            )
        else:
            rows = await asyncio.to_thread(
                predict_affinity,
                protein_pdb_id,
                ligand_smiles,
                ensemble_size,
            )
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        return boltz2_pb2.Boltz2BatchResponse(
            protein_pdb_id=protein_pdb_id,
            affinities=[_binding_affinity(row) for row in rows],
            elapsed_ms=elapsed_ms,
        )

    async def BatchPredict(self, request, context):
        """Batch affinity prediction."""
        results = []
        for req in getattr(request, "requests", []):
            results.append(await self.PredictAffinity(req, context))
        return type(
            "BatchAffinityResponse",
            (),
            {"results": results, "total_elapsed_ms": 500},
        )()

    def _runner(self):
        if self.runner is not None:
            return self.runner
        command = os.environ.get(_COMMAND_ENV, "").strip()
        if command:
            _require_runtime()
            self.runner = BoltzCommandRunner(command)
            return self.runner
        try:
            _require_runtime()
        except RuntimeError:
            raise
        self.runner = BoltzCliRunner.from_env()
        return self.runner


class Boltz2OracleServicer(oracle_pb2_grpc.OracleServiceServicer):
    def __init__(self, service: Boltz2Servicer | None = None):
        self._local_runtime = service is None
        self.service = service or Boltz2Servicer()

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
                expected_level=oracle_pb2.L1_ML_SURROGATE,
                require_protein_pdb_id=True,
                allowed_parameters=("ensemble_size",),
            )
            artifacts = await resolve_oracle_artifact_refs(
                _artifact_status_objects() if self._local_runtime else []
            )
            boltz_request = _oracle_request_to_boltz_batch(request_context)
            total_elapsed_ms = 0
            try:
                boltz_response = await self.service.PredictAffinity(
                    boltz_request,
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
                        oracle_name="boltz2",
                        elapsed_ms=0,
                        artifacts=artifacts,
                        error_code="COMPUTATION_ERROR",
                        error_message=str(exc),
                    )
                    for index in range(len(request_context.molecules))
                ]
            else:
                affinities = list(boltz_response.affinities)
                if boltz_response.protein_pdb_id != request_context.protein_pdb_id:
                    raise OracleDataError(
                        "Boltz-2 response protein does not match request protein_pdb_id"
                    )
                actual_order = tuple(str(affinity.ligand_smiles) for affinity in affinities)
                if actual_order != request_context.molecules:
                    raise OracleDataError("Boltz-2 affinities do not match request molecule order")
                if any(
                    affinity.protein_pdb_id != request_context.protein_pdb_id
                    for affinity in affinities
                ):
                    raise OracleDataError(
                        "Boltz-2 affinity protein does not match request protein_pdb_id"
                    )
                if any(
                    affinity.ensemble_size != boltz_request.ensemble_size for affinity in affinities
                ):
                    raise OracleDataError("Boltz-2 affinity ensemble_size does not match request")
                if any(
                    len(affinity.per_member_dg) != affinity.ensemble_size for affinity in affinities
                ):
                    raise OracleDataError(
                        "Boltz-2 affinity per_member_dg count does not match ensemble_size"
                    )
                evaluations = [
                    _oracle_evaluation_from_affinity(
                        request_context=request_context,
                        index=index,
                        affinity=affinity,
                        elapsed_ms=boltz_response.elapsed_ms,
                        artifacts=artifacts,
                    )
                    for index, affinity in enumerate(affinities)
                ]
                total_elapsed_ms = int(boltz_response.elapsed_ms)
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


class BoltzCommandRunner:
    def __init__(self, command: str) -> None:
        self.command = command
        self.timeout = _positive_timeout(
            os.environ.get(_COMMAND_TIMEOUT_ENV, "300"),
            _COMMAND_TIMEOUT_ENV,
        )

    def predict_affinity(
        self,
        protein_pdb_id: str,
        ligand_smiles: list[str],
        ensemble_size: int = 5,
    ) -> list[dict]:
        _require_command_available(self.command)
        payload = {
            "protein_pdb_id": protein_pdb_id,
            "ligand_smiles": list(ligand_smiles),
            "ensemble_size": int(ensemble_size),
        }
        completed = _run_process_group(
            shlex.split(self.command),
            input_text=json.dumps(payload, sort_keys=True),
            timeout=self.timeout,
        )
        if completed.returncode != 0:
            stderr = completed.stderr.strip()
            raise RuntimeError(f"{_COMMAND_ENV} failed: {stderr}")
        try:
            response = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise OracleDataError(f"{_COMMAND_ENV} returned invalid JSON") from exc
        rows = _affinity_rows_from_command_response(response)
        for row in rows:
            _require_affinity_row(row)
        return rows


def _affinity_rows_from_command_response(response: object) -> list[dict]:
    if isinstance(response, list):
        rows = response
    elif isinstance(response, dict):
        rows = response.get("affinities", response.get("results", response.get("rows")))
    else:
        raise OracleDataError(f"{_COMMAND_ENV} must return a JSON object or list")
    if not isinstance(rows, list):
        raise OracleDataError(f"{_COMMAND_ENV} response requires affinities")
    if not all(isinstance(row, dict) for row in rows):
        raise OracleDataError(f"{_COMMAND_ENV} affinities must be JSON objects")
    return rows


def _require_affinity_row(row: dict) -> None:
    required = (
        "protein_pdb_id",
        "ligand_smiles",
        "delta_g_kcal_mol",
        "uncertainty",
        "ki_nm",
        "ensemble_size",
    )
    missing = [field for field in required if field not in row or row[field] in ("", None)]
    if missing:
        raise OracleDataError(f"{_COMMAND_ENV} affinity row missing fields: {', '.join(missing)}")
    _finite_output(row["delta_g_kcal_mol"], "delta_g_kcal_mol")
    _finite_output(row["uncertainty"], "uncertainty")
    _finite_output(row["ki_nm"], "ki_nm")
    ensemble_size = row["ensemble_size"]
    if isinstance(ensemble_size, bool) or not isinstance(ensemble_size, int) or ensemble_size <= 0:
        raise OracleDataError(f"{_COMMAND_ENV} ensemble_size must be a positive integer")
    per_member = row.get("per_member_dg", [])
    if not isinstance(per_member, list):
        raise OracleDataError(f"{_COMMAND_ENV} per_member_dg must be a list")
    for value in per_member:
        _finite_output(value, "per_member_dg")
    if len(per_member) != ensemble_size:
        raise OracleDataError(f"{_COMMAND_ENV} per_member_dg count must match ensemble_size")


class BoltzCliRunner:
    def __init__(
        self,
        model_path: str | Path,
        template_dir: str | Path,
        work_dir: str | Path,
        boltz_executable: str = "boltz",
        run_command=None,
        timeout: float | None = None,
    ):
        self.model_path = Path(model_path).expanduser()
        self.template_dir = Path(template_dir).expanduser()
        self.work_dir = Path(work_dir).expanduser()
        self.boltz_executable = boltz_executable
        self.run_command = run_command
        self.timeout = _positive_timeout(
            timeout if timeout is not None else os.environ.get(_COMMAND_TIMEOUT_ENV, "300"),
            _COMMAND_TIMEOUT_ENV,
        )

    @classmethod
    def from_env(cls) -> "BoltzCliRunner":
        return cls(
            model_path=os.environ["BOLTZ_MODEL_PATH"],
            template_dir=os.environ["BOLTZ_INPUT_TEMPLATE_DIR"],
            work_dir=os.environ.get("BOLTZ_WORK_DIR", "runs/boltz2"),
            boltz_executable=os.environ.get("BOLTZ_BINARY", "boltz"),
        )

    def predict_affinity(
        self,
        protein_pdb_id: str,
        ligand_smiles: list[str],
        ensemble_size: int = 5,
    ) -> list[dict]:
        if not ligand_smiles:
            raise ValueError("ligand_smiles is required")
        results: list[dict] = []
        output_dir = self.work_dir / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        for index, smiles in enumerate(ligand_smiles):
            input_path = self._write_input(protein_pdb_id, smiles, index)
            command = self._command(input_path, output_dir, ensemble_size)
            if self.run_command is None:
                completed = _run_process_group(
                    command,
                    timeout=self.timeout,
                    env=os.environ.copy(),
                )
            else:
                completed = self.run_command(
                    command,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    env=os.environ.copy(),
                )
            if getattr(completed, "returncode", 0) != 0:
                stderr = getattr(completed, "stderr", "")
                raise RuntimeError(f"Boltz prediction failed for {smiles}: {stderr}")
            results.append(
                _read_affinity_result(
                    _affinity_output_path(output_dir, input_path.stem),
                    protein_pdb_id,
                    smiles,
                    ensemble_size,
                )
            )
        return results

    def _write_input(self, protein_pdb_id: str, ligand_smiles: str, index: int) -> Path:
        template_path = self._template_path(protein_pdb_id)
        with template_path.open(encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
        payload = _replace_ligand_smiles(payload, ligand_smiles)
        input_dir = self.work_dir / "inputs"
        input_dir.mkdir(parents=True, exist_ok=True)
        input_path = input_dir / f"{_safe_name(protein_pdb_id)}_{index}.yaml"
        with input_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(payload, handle, sort_keys=False)
        return input_path

    def _template_path(self, protein_pdb_id: str) -> Path:
        for suffix in (".yaml", ".yml"):
            path = self.template_dir / f"{protein_pdb_id}{suffix}"
            if path.is_file():
                return path
        raise RuntimeError(
            f"Boltz input template not found for {protein_pdb_id} in {self.template_dir}"
        )

    def _command(self, input_path: Path, output_dir: Path, ensemble_size: int) -> list[str]:
        command = [
            self.boltz_executable,
            "predict",
            str(input_path),
            "--out_dir",
            str(output_dir),
            "--cache",
            str(self._cache_dir()),
            "--accelerator",
            os.environ.get("BOLTZ_ACCELERATOR", "gpu"),
            "--devices",
            os.environ.get("BOLTZ_DEVICES", "1"),
            "--model",
            "boltz2",
            "--recycling_steps",
            os.environ.get("BOLTZ_RECYCLING_STEPS", "3"),
            "--sampling_steps",
            os.environ.get("BOLTZ_SAMPLING_STEPS", "200"),
            "--sampling_steps_affinity",
            os.environ.get("BOLTZ_SAMPLING_STEPS_AFFINITY", "200"),
            "--diffusion_samples_affinity",
            str(max(int(ensemble_size), 1)),
            "--num_workers",
            os.environ.get("BOLTZ_NUM_WORKERS", "0"),
            "--override",
        ]
        command.extend(self._checkpoint_args())
        if os.environ.get("BOLTZ_USE_MSA_SERVER") == "1":
            command.append("--use_msa_server")
        if os.environ.get("BOLTZ_USE_POTENTIALS") == "1":
            command.append("--use_potentials")
        if os.environ.get("BOLTZ_NO_KERNELS") == "1":
            command.append("--no_kernels")
        return command

    def _cache_dir(self) -> Path:
        if os.environ.get("BOLTZ_CACHE"):
            return Path(os.environ["BOLTZ_CACHE"]).expanduser()
        if self.model_path.is_dir():
            return self.model_path
        return self.model_path.parent

    def _checkpoint_args(self) -> list[str]:
        if self.model_path.is_dir():
            structure_checkpoint = self.model_path / "boltz2_conf.ckpt"
            affinity_checkpoint = self.model_path / "boltz2_aff.ckpt"
            if not structure_checkpoint.is_file():
                raise RuntimeError(f"Boltz structure checkpoint not found: {structure_checkpoint}")
            if not affinity_checkpoint.is_file():
                raise RuntimeError(f"Boltz affinity checkpoint not found: {affinity_checkpoint}")
            return [
                "--checkpoint",
                str(structure_checkpoint),
                "--affinity_checkpoint",
                str(affinity_checkpoint),
            ]
        return [
            "--checkpoint",
            str(self.model_path),
            "--affinity_checkpoint",
            str(self.model_path),
        ]


async def _abort_invalid_argument(context, message: str):
    if context is not None and hasattr(context, "abort"):
        await context.abort(grpc.StatusCode.INVALID_ARGUMENT, message)
    raise ValueError(message)


async def _abort_message(context, message: str):
    if context is not None and hasattr(context, "abort"):
        await context.abort(grpc.StatusCode.FAILED_PRECONDITION, message)
    raise RuntimeError(message)


def _binding_affinity(row: dict) -> boltz2_pb2.Boltz2BindingAffinity:
    _require_affinity_row(row)
    model_version = (
        _VALIDATION_MARKER
        if row.get("validation_marker") == _VALIDATION_MARKER
        else str(row.get("model_version") or row.get("validation_marker") or "")
    )
    return boltz2_pb2.Boltz2BindingAffinity(
        protein_pdb_id=str(row["protein_pdb_id"]),
        ligand_smiles=str(row["ligand_smiles"]),
        delta_g_kcal_mol=float(row["delta_g_kcal_mol"]),
        uncertainty=float(row["uncertainty"]),
        ki_nm=float(row["ki_nm"]),
        ensemble_size=int(row["ensemble_size"]),
        per_member_dg=[float(value) for value in row.get("per_member_dg", [])],
        model_version=model_version,
    )


def _finite_output(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise OracleDataError(f"{_COMMAND_ENV} {field_name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise OracleDataError(f"{_COMMAND_ENV} {field_name} must be finite")
    return number


def _positive_timeout(value: object, field_name: str) -> float:
    try:
        timeout = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{field_name} must be a finite positive number") from exc
    if isinstance(value, bool) or not math.isfinite(timeout) or timeout <= 0:
        raise RuntimeError(f"{field_name} must be a finite positive number")
    return timeout


def _run_process_group(
    command: list[str],
    *,
    timeout: float,
    input_text: str | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess:
    parsed_timeout = _positive_timeout(timeout, _COMMAND_TIMEOUT_ENV)
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE if input_text is not None else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(input=input_text, timeout=parsed_timeout)
    except subprocess.TimeoutExpired:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        stdout, stderr = process.communicate()
        raise subprocess.TimeoutExpired(
            command,
            parsed_timeout,
            output=stdout,
            stderr=stderr,
        ) from None
    return subprocess.CompletedProcess(
        command,
        int(process.returncode),
        stdout,
        stderr,
    )


def _read_affinity_result(
    path: Path,
    protein_pdb_id: str,
    ligand_smiles: str,
    ensemble_size: int,
) -> dict:
    if not path.is_file():
        raise RuntimeError(f"Boltz affinity output not found: {path}")
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if "affinity_pred_value" not in data:
        raise RuntimeError(f"Boltz affinity output missing affinity_pred_value: {path}")
    prediction_value = float(data["affinity_pred_value"])
    member_values = [
        float(value)
        for key, value in sorted(data.items())
        if re.fullmatch(r"affinity_pred_value\d+", key)
    ]
    if not member_values:
        member_values = [prediction_value]
    per_member_dg = [_boltz_affinity_to_delta_g(value) for value in member_values]
    return {
        "protein_pdb_id": protein_pdb_id,
        "ligand_smiles": ligand_smiles,
        "delta_g_kcal_mol": _boltz_affinity_to_delta_g(prediction_value),
        "uncertainty": statistics.pstdev(per_member_dg) if len(per_member_dg) > 1 else 0.0,
        "ki_nm": 1000.0 * math.pow(10.0, prediction_value),
        "ensemble_size": ensemble_size,
        "per_member_dg": per_member_dg,
    }


def _affinity_output_path(output_dir: Path, input_stem: str) -> Path:
    candidates = (
        output_dir / "predictions" / input_stem / f"affinity_{input_stem}.json",
        output_dir
        / f"boltz_results_{input_stem}"
        / "predictions"
        / input_stem
        / f"affinity_{input_stem}.json",
    )
    for path in candidates:
        if path.is_file():
            return path
    return candidates[-1]


def _boltz_affinity_to_delta_g(value: float) -> float:
    return -(6.0 - float(value)) * 1.364


def _replace_ligand_smiles(value: Any, ligand_smiles: str) -> Any:
    if isinstance(value, dict):
        return {key: _replace_ligand_smiles(item, ligand_smiles) for key, item in value.items()}
    if isinstance(value, list):
        return [_replace_ligand_smiles(item, ligand_smiles) for item in value]
    if isinstance(value, str) and value in {
        "__LIGAND_SMILES__",
        "{{LIGAND_SMILES}}",
        "${LIGAND_SMILES}",
    }:
        return ligand_smiles
    return value


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "boltz_input"


def _oracle_request_to_boltz_batch(request) -> boltz2_pb2.Boltz2BatchRequest:
    return boltz2_pb2.Boltz2BatchRequest(
        project_id=request.project_id,
        protein_pdb_id=request.protein_pdb_id,
        ligand_smiles=request.molecules,
        ensemble_size=parse_positive_parameter(
            request.parameters,
            "ensemble_size",
            default=5,
        ),
    )


def _oracle_evaluation_from_affinity(
    *,
    request_context,
    index: int,
    affinity,
    elapsed_ms: int,
    artifacts,
) -> oracle_pb2.OracleEvaluation:
    return build_oracle_evaluation(
        request=request_context,
        index=index,
        oracle_name="boltz2",
        scores={
            "affinity": float(affinity.delta_g_kcal_mol),
            "ki_nm": float(affinity.ki_nm),
        },
        uncertainties={"affinity": float(affinity.uncertainty)},
        elapsed_ms=int(elapsed_ms),
        artifacts=artifacts,
        model_version=str(affinity.model_version),
        units={"affinity": "kcal/mol", "ki_nm": "nM"},
    )


def register_grpc_services(server) -> None:
    oracle_pb2_grpc.add_OracleServiceServicer_to_server(Boltz2OracleServicer(), server)


async def serve():
    _require_runtime()
    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=10))
    boltz2_pb2_grpc.add_Boltz2ServiceServicer_to_server(Boltz2Servicer(), server)
    register_grpc_services(server)
    server.add_insecure_port("[::]:50053")
    await server.start()
    _LOGGER.info("Boltz-2 Binding Affinity Service running on :50053")
    await server.wait_for_termination()


def _validation_response(payload: object) -> dict:
    _require_synthetic_validation_enabled()
    if not isinstance(payload, dict):
        raise ValueError("Boltz-2 validation request must be a JSON object")
    expected_fields = {"protein_pdb_id", "ligand_smiles", "ensemble_size"}
    unexpected = sorted(set(payload) - expected_fields)
    if unexpected:
        raise ValueError(
            "Boltz-2 validation request has unexpected fields: " + ", ".join(unexpected)
        )
    missing = sorted(expected_fields - set(payload))
    if missing:
        raise ValueError(
            "Boltz-2 validation request is missing fields: " + ", ".join(missing)
        )
    protein_pdb_id = _validation_text(payload["protein_pdb_id"], "protein_pdb_id")
    ligand_smiles = _validation_text_list(
        payload["ligand_smiles"],
        "ligand_smiles",
        maximum=_VALIDATION_MAX_BATCH_SIZE,
    )
    ensemble_size = payload["ensemble_size"]
    if (
        isinstance(ensemble_size, bool)
        or not isinstance(ensemble_size, int)
        or ensemble_size <= 0
        or ensemble_size > _VALIDATION_MAX_ENSEMBLE_SIZE
    ):
        raise ValueError(
            "Boltz-2 validation ensemble_size must be a positive integer "
            f"not greater than {_VALIDATION_MAX_ENSEMBLE_SIZE}"
        )
    affinities = [
        _validation_affinity_row(
            protein_pdb_id,
            molecule_smiles,
            ensemble_size,
        )
        for molecule_smiles in ligand_smiles
    ]
    return {
        "protein_pdb_id": protein_pdb_id,
        "affinities": affinities,
        "elapsed_ms": 0,
        "validation_marker": _VALIDATION_MARKER,
    }


def _validation_affinity_row(
    protein_pdb_id: str,
    ligand_smiles: str,
    ensemble_size: int,
) -> dict:
    fingerprint = _validation_fingerprint(f"{protein_pdb_id}|{ligand_smiles}")
    affinity_value = ((fingerprint % 201) - 100) / 1000.0
    center = (ensemble_size - 1) / 2.0
    member_affinities = [
        affinity_value + 0.02 * (member_index - center)
        for member_index in range(ensemble_size)
    ]
    per_member_dg = [
        round(_boltz_affinity_to_delta_g(value), 12)
        for value in member_affinities
    ]
    delta_g = sum(per_member_dg) / ensemble_size
    return {
        "protein_pdb_id": protein_pdb_id,
        "ligand_smiles": ligand_smiles,
        "delta_g_kcal_mol": delta_g,
        "uncertainty": statistics.pstdev(per_member_dg),
        "ki_nm": round(1000.0 * math.pow(10.0, affinity_value), 12),
        "ensemble_size": ensemble_size,
        "per_member_dg": per_member_dg,
        "validation_marker": _VALIDATION_MARKER,
    }


def _validation_fingerprint(value: str) -> int:
    return sum(index * ord(character) for index, character in enumerate(value, start=1))


def _validation_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(
            f"Boltz-2 validation {field_name} must be a non-empty trimmed string"
        )
    return value


def _validation_text_list(value: object, field_name: str, *, maximum: int) -> list[str]:
    if not isinstance(value, list) or not value or len(value) > maximum:
        raise ValueError(
            f"Boltz-2 validation {field_name} must be a non-empty list "
            f"with at most {maximum} items"
        )
    if any(
        not isinstance(item, str)
        or not item
        or item != item.strip()
        for item in value
    ):
        raise ValueError(
            f"Boltz-2 validation {field_name} must contain non-empty trimmed strings"
        )
    return list(value)


def _require_synthetic_validation_enabled() -> None:
    if os.environ.get(_VALIDATION_GATE_ENV) != "true":
        raise RuntimeError(f"{_VALIDATION_GATE_ENV}=true is required")


def _run_validation_runner() -> None:
    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError as exc:
        raise ValueError("Boltz-2 validation request must be valid JSON") from exc
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
        sys.stderr.write("Boltz-2 service has unexpected command line arguments\n")
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
