"""ADMET Prediction Service.

gRPC server for ADMET property prediction and L0 drug-likeness filtering.
"""
import asyncio
import json
import os
import shlex
import shutil
import subprocess
import time
from concurrent import futures
from types import SimpleNamespace

import grpc
from mf_core.artifacts import (
    ArtifactRequirement,
    PythonPackageRequirement,
    RequirementStatus,
    check_artifact,
    check_python_package,
    require_available,
)
from mf_core.proto_gen.moleculeforge.v1.oracle import oracle_pb2, oracle_pb2_grpc

_REQUIREMENTS = (ArtifactRequirement("admet_model", "ADMET_MODEL_PATH", kind="directory"),)
_PACKAGES = (PythonPackageRequirement("rdkit", module="rdkit"),)
_COMMAND_ENV = "ADMET_ORACLE_COMMAND"
_COMMAND_TIMEOUT_ENV = "ADMET_ORACLE_TIMEOUT_SECONDS"


def _require_runtime() -> list[RequirementStatus]:
    statuses = _status_objects()
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
        message = "ADMET model runner is not configured"
    if context is not None and hasattr(context, "abort"):
        await context.abort(grpc.StatusCode.FAILED_PRECONDITION, message)
    raise RuntimeError(message)


def _status_objects() -> list[RequirementStatus]:
    command_status = _command_status()
    if command_status.configured:
        return [command_status]
    return [
        *(check_artifact(requirement) for requirement in _REQUIREMENTS),
        *(check_python_package(requirement) for requirement in _PACKAGES),
        command_status,
    ]


class ADMETServicer:
    def __init__(self, runner=None):
        self.runner = runner

    def _runner(self):
        if self.runner is not None:
            return self.runner
        command = os.environ.get(_COMMAND_ENV, "").strip()
        if command:
            self.runner = ADMETCommandRunner(command)
            return self.runner
        from mf_oracles.admet_ai.oracle import ADMETHTTPRunner

        self.runner = ADMETHTTPRunner.from_env()
        return self.runner

    def _uses_command_runner(self) -> bool:
        return isinstance(self.runner, ADMETCommandRunner) or (
            self.runner is None and bool(os.environ.get(_COMMAND_ENV, "").strip())
        )

    async def Predict(self, request, context):
        """Predict ADMET properties for a molecule."""
        smiles = _request_smiles(request)
        properties = _request_properties(request)
        start = time.perf_counter()
        if self._uses_command_runner():
            rows = await _maybe_await(
                self._runner().predict([smiles], properties, return_uncertainty=False)
            )
            predictions = _admet_predictions_for_smiles(rows, smiles)
            return SimpleNamespace(
                smiles=smiles,
                predictions=predictions,
                properties=predictions,
                elapsed_ms=_elapsed_ms_from_rows(rows, smiles, start),
            )
        try:
            _require_runtime()
        except RuntimeError:
            return await _abort_unavailable(context)
        from mf_oracles.admet_ai.oracle import ADMETAIOracle

        result = await ADMETAIOracle(runner=self._runner()).evaluate([smiles], properties)
        predictions = result[smiles]
        return SimpleNamespace(
            smiles=smiles,
            predictions=predictions,
            properties=predictions,
            elapsed_ms=int((time.perf_counter() - start) * 1000),
        )

    async def BatchPredict(self, request, context):
        """Batch ADMET prediction."""
        results = []
        for req in getattr(request, "requests", []):
            results.append(await self.Predict(req, context))
        return type(
            "BatchADMETResponse",
            (),
            {"results": results, "total_elapsed_ms": 200},
        )()

    async def Screen(self, request, context):
        """Quick L0 screen: returns pass/fail with filter reasons."""
        smiles = _request_smiles(request)
        properties = _request_properties(request)
        if self._uses_command_runner():
            rows = await _maybe_await(
                self._runner().predict([smiles], properties, return_uncertainty=True)
            )
            uncertainties = _admet_uncertainties_for_smiles(rows, smiles)
            return SimpleNamespace(
                smiles=smiles,
                result={
                    smiles: {
                        key: {
                            "value": value,
                            "uncertainty": uncertainties.get(key, 0.0),
                        }
                        for key, value in _admet_predictions_for_smiles(rows, smiles).items()
                    }
                },
            )
        try:
            _require_runtime()
        except RuntimeError:
            return await _abort_unavailable(context)
        from mf_oracles.admet_ai.oracle import ADMETAIOracle

        result = await ADMETAIOracle(runner=self._runner()).predict_with_uncertainty(
            [smiles],
            properties,
        )
        return SimpleNamespace(smiles=smiles, result=result)


class ADMETOracleServicer(oracle_pb2_grpc.OracleServiceServicer):
    def __init__(self, service: ADMETServicer | None = None):
        self.service = service or ADMETServicer()

    async def Evaluate(self, request, context):
        start = time.perf_counter()
        evaluations = []
        properties = list(getattr(request, "requested_properties", []))
        for smiles in getattr(request, "molecule_smiles", []):
            response = await self.service.Predict(
                SimpleNamespace(smiles=smiles, properties=properties),
                context,
            )
            evaluations.append(
                oracle_pb2.OracleEvaluation(
                    oracle_name="admet_ai",
                    molecule_smiles=str(smiles),
                    level=request.level or oracle_pb2.L1_ML_SURROGATE,
                    scores=_numeric_map(response.predictions),
                    elapsed_ms=int(response.elapsed_ms),
                    success=True,
                )
            )
        return oracle_pb2.OracleBatchResponse(
            evaluations=evaluations,
            batch_id=str(getattr(request, "project_id", "")),
            total_elapsed_ms=int((time.perf_counter() - start) * 1000),
        )

    async def PredictWithUncertainty(self, request, context):
        start = time.perf_counter()
        evaluations = []
        properties = list(getattr(request, "requested_properties", []))
        for smiles in getattr(request, "molecule_smiles", []):
            response = await self.service.Screen(
                SimpleNamespace(smiles=smiles, properties=properties),
                context,
            )
            scores, uncertainties = _split_uncertainty_result(response.result, str(smiles))
            evaluations.append(
                oracle_pb2.OracleEvaluation(
                    oracle_name="admet_ai",
                    molecule_smiles=str(smiles),
                    level=request.level or oracle_pb2.L1_ML_SURROGATE,
                    scores=scores,
                    uncertainties=uncertainties,
                    success=True,
                )
            )
        return oracle_pb2.OracleBatchResponse(
            evaluations=evaluations,
            batch_id=str(getattr(request, "project_id", "")),
            total_elapsed_ms=int((time.perf_counter() - start) * 1000),
        )

    async def StreamEvaluate(self, request_iterator, context):
        async for request in request_iterator:
            yield await self.Evaluate(request, context)


class ADMETCommandRunner:
    def __init__(self, command: str) -> None:
        self.command = command
        self.timeout = float(os.environ.get(_COMMAND_TIMEOUT_ENV, "120"))

    def predict(
        self,
        smiles: list[str],
        properties: list[str],
        return_uncertainty: bool,
    ) -> list[dict]:
        payload = {
            "smiles": list(smiles),
            "properties": list(properties),
            "return_uncertainty": bool(return_uncertainty),
        }
        completed = subprocess.run(
            _command_argv(self.command),
            input=json.dumps(payload, sort_keys=True),
            capture_output=True,
            check=False,
            text=True,
            timeout=self.timeout,
        )
        if completed.returncode != 0:
            stderr = completed.stderr.strip()
            raise RuntimeError(f"{_COMMAND_ENV} failed: {stderr}")
        try:
            response = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"{_COMMAND_ENV} returned invalid JSON") from exc
        return _admet_rows_from_command_response(response)


def _command_status() -> RequirementStatus:
    raw_command = os.environ.get(_COMMAND_ENV, "").strip()
    if not raw_command:
        return RequirementStatus(
            name="admet_oracle_command",
            configured=False,
            available=False,
            required=False,
            path=None,
            source=_COMMAND_ENV,
            message=f"{_COMMAND_ENV} is not configured",
        )
    try:
        argv = shlex.split(raw_command)
    except ValueError as exc:
        return RequirementStatus(
            name="admet_oracle_command",
            configured=True,
            available=False,
            required=True,
            path=raw_command,
            source=_COMMAND_ENV,
            message=str(exc),
        )
    executable = argv[0] if argv else ""
    available = bool(shutil.which(executable) or (executable and os.access(executable, os.X_OK)))
    return RequirementStatus(
        name="admet_oracle_command",
        configured=True,
        available=available,
        required=True,
        path=raw_command,
        source=_COMMAND_ENV,
        message=(
            "ADMET_ORACLE_COMMAND is available"
            if available
            else f"{_COMMAND_ENV} executable is not available: {executable}"
        ),
    )


def _command_argv(raw_command: str | None = None) -> list[str]:
    raw_command = (raw_command or os.environ.get(_COMMAND_ENV, "")).strip()
    try:
        argv = shlex.split(raw_command)
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc
    if not argv:
        raise RuntimeError(f"{_COMMAND_ENV} is empty")
    executable = argv[0]
    if not shutil.which(executable) and not os.access(executable, os.X_OK):
        raise RuntimeError(f"{_COMMAND_ENV} executable is not available: {executable}")
    return argv


def _admet_rows_from_command_response(response: object) -> list[dict]:
    if isinstance(response, list):
        rows = response
    elif isinstance(response, dict):
        rows = response.get("results", response.get("rows"))
        if rows is None and "predictions" in response:
            rows = [response]
    else:
        raise RuntimeError(f"{_COMMAND_ENV} must return a JSON object or list")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError(f"{_COMMAND_ENV} response requires non-empty results")
    if not all(isinstance(row, dict) for row in rows):
        raise RuntimeError(f"{_COMMAND_ENV} result rows must be JSON objects")
    return rows


def _admet_predictions_for_smiles(rows: list[dict], smiles: str) -> dict[str, float]:
    row = _admet_row_for_smiles(rows, smiles)
    predictions = _numeric_map(row.get("predictions", row.get("scores", {})))
    if not predictions:
        raise RuntimeError(f"{_COMMAND_ENV} result requires numeric predictions")
    return predictions


def _admet_uncertainties_for_smiles(rows: list[dict], smiles: str) -> dict[str, float]:
    row = _admet_row_for_smiles(rows, smiles)
    return _numeric_map(row.get("uncertainties"))


def _admet_row_for_smiles(rows: list[dict], smiles: str) -> dict:
    for row in rows:
        row_smiles = str(row.get("smiles") or row.get("molecule_smiles") or "")
        if row_smiles == smiles:
            return row
    raise RuntimeError(f"{_COMMAND_ENV} response missing result for {smiles}")


def _elapsed_ms_from_rows(rows: list[dict], smiles: str, start: float) -> int:
    row = _admet_row_for_smiles(rows, smiles)
    elapsed_ms = row.get("elapsed_ms")
    if isinstance(elapsed_ms, int | float):
        return int(elapsed_ms)
    return int((time.perf_counter() - start) * 1000)


async def _maybe_await(value):
    if hasattr(value, "__await__"):
        return await value
    return value


def _numeric_map(values) -> dict[str, float]:
    if not isinstance(values, dict):
        return {}
    output = {}
    for key, value in values.items():
        if isinstance(value, int | float):
            output[str(key)] = float(value)
    return output


def _split_uncertainty_result(
    result: dict,
    smiles: str,
) -> tuple[dict[str, float], dict[str, float]]:
    row = result.get(smiles, result) if isinstance(result, dict) else {}
    scores = {}
    uncertainties = {}
    if not isinstance(row, dict):
        return scores, uncertainties
    for key, value in row.items():
        if isinstance(value, dict):
            if isinstance(value.get("value"), int | float):
                scores[str(key)] = float(value["value"])
            if isinstance(value.get("uncertainty"), int | float):
                uncertainties[str(key)] = float(value["uncertainty"])
        elif isinstance(value, int | float):
            scores[str(key)] = float(value)
    return scores, uncertainties


def _request_smiles(request) -> str:
    smiles = (
        getattr(request, "smiles", None)
        or getattr(request, "molecule_smiles", None)
        or getattr(request, "canonical_smiles", None)
    )
    if not smiles:
        raise ValueError("request.smiles is required")
    return str(smiles)


def _request_properties(request) -> list[str]:
    properties = (
        getattr(request, "properties", None)
        or getattr(request, "requested_properties", None)
        or []
    )
    if isinstance(properties, str):
        properties = [properties]
    properties = [str(item) for item in properties if str(item)]
    if properties:
        return properties
    import os

    return [item.strip() for item in os.environ.get("ADMET_TARGETS", "").split(",") if item.strip()]


def register_grpc_services(server) -> None:
    oracle_pb2_grpc.add_OracleServiceServicer_to_server(ADMETOracleServicer(), server)


async def serve():
    _require_runtime()
    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=16))
    register_grpc_services(server)
    server.add_insecure_port("[::]:50056")
    await server.start()
    print("ADMET Prediction Service running on :50056")
    await server.wait_for_termination()


if __name__ == "__main__":
    asyncio.run(serve())
