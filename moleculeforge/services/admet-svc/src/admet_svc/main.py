"""ADMET Prediction Service.

gRPC server for L1 ADMET property prediction.
"""

import asyncio
import json
import logging
import math
import os
import shlex
import shutil
import subprocess
import sys
import time
from concurrent import futures
from types import SimpleNamespace
from urllib.parse import urlparse

import grpc
from mf_core.artifacts import (
    PythonPackageRequirement,
    RequirementStatus,
    check_python_package,
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
from mf_core.proto_gen.moleculeforge.v1.core import audit_pb2
from mf_core.proto_gen.moleculeforge.v1.oracle import oracle_pb2, oracle_pb2_grpc

_PACKAGES = (
    PythonPackageRequirement("rdkit", module="rdkit"),
    PythonPackageRequirement("httpx", module="httpx"),
)
_COMMAND_ENV = "ADMET_ORACLE_COMMAND"
_COMMAND_TIMEOUT_ENV = "ADMET_ORACLE_TIMEOUT_SECONDS"
_VALIDATION_GATE_ENV = "MF_ALLOW_SYNTHETIC_VALIDATION"
_VALIDATION_MARKER = "synthetic_pipeline_validation_only"
_LOGGER = logging.getLogger(__name__)
_VALIDATION_MAX_BATCH_SIZE = 256
_VALIDATION_MAX_PROPERTIES = 64


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
        _http_url_status(
            name="admet_service_url",
            env_var="ADMET_SERVICE_URL",
        ),
        _configuration_status(
            name="admet_targets",
            env_var="ADMET_TARGETS",
            configured=bool(_configured_targets()),
        ),
        _positive_integer_status(
            name="admet_batch_size",
            env_var="ADMET_BATCH_SIZE",
            default=64,
        ),
        _positive_number_status(
            name="admet_timeout",
            env_var=_COMMAND_TIMEOUT_ENV,
            default=120.0,
        ),
        *(check_python_package(requirement) for requirement in _PACKAGES),
    ]


def _artifact_status_objects() -> list[RequirementStatus]:
    command_status = _command_status()
    return [command_status] if command_status.configured else []


def _configuration_status(
    *,
    name: str,
    env_var: str,
    configured: bool,
) -> RequirementStatus:
    return RequirementStatus(
        name=name,
        configured=configured,
        available=configured,
        required=True,
        path=None,
        source=env_var,
        message=(f"{env_var} is configured" if configured else f"{env_var} is required for {name}"),
    )


def _http_url_status(
    *,
    name: str,
    env_var: str,
) -> RequirementStatus:
    value = os.environ.get(env_var, "").strip()
    parsed = urlparse(value)
    available = bool(value and parsed.scheme in {"http", "https"} and parsed.netloc)
    return RequirementStatus(
        name=name,
        configured=bool(value),
        available=available,
        required=True,
        path=None,
        source=env_var,
        message=(
            f"{env_var} is configured"
            if available
            else f"{env_var} must be an absolute http(s) URL"
        ),
    )


def _configured_targets() -> list[str]:
    return [item.strip() for item in os.environ.get("ADMET_TARGETS", "").split(",") if item.strip()]


def _positive_integer_status(
    *,
    name: str,
    env_var: str,
    default: int,
) -> RequirementStatus:
    raw_value = os.environ.get(env_var, str(default))
    try:
        value = int(raw_value)
    except ValueError:
        value = 0
    available = value > 0
    return RequirementStatus(
        name=name,
        configured=True,
        available=available,
        required=True,
        path=None,
        source=env_var,
        message=(
            f"{env_var} is configured" if available else f"{env_var} must be a positive integer"
        ),
    )


def _positive_number_status(
    *,
    name: str,
    env_var: str,
    default: float,
) -> RequirementStatus:
    raw_value = os.environ.get(env_var, str(default))
    try:
        value = float(raw_value)
    except ValueError:
        value = 0.0
    available = math.isfinite(value) and value > 0
    return RequirementStatus(
        name=name,
        configured=True,
        available=available,
        required=True,
        path=None,
        source=env_var,
        message=(
            f"{env_var} is configured"
            if available
            else f"{env_var} must be a finite positive number"
        ),
    )


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
            rows = await asyncio.to_thread(
                self._runner().predict,
                [smiles],
                properties,
                False,
            )
            predictions = _admet_predictions_for_smiles(rows, smiles)
            return SimpleNamespace(
                smiles=smiles,
                predictions=predictions,
                properties=predictions,
                elapsed_ms=_elapsed_ms_from_rows(rows, smiles, start),
                model_version=_admet_model_version_for_smiles(rows, smiles),
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
            model_version=str(getattr(result, "model_version", "")),
            model_artifact=_remote_model_artifact(result),
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
        """Return ADMET predictions with available uncertainty estimates."""
        smiles = _request_smiles(request)
        properties = _request_properties(request)
        if self._uses_command_runner():
            rows = await asyncio.to_thread(
                self._runner().predict,
                [smiles],
                properties,
                True,
            )
            uncertainties = _admet_uncertainties_for_smiles(rows, smiles)
            return SimpleNamespace(
                smiles=smiles,
                result={
                    smiles: {
                        key: {
                            "value": value,
                            **({"uncertainty": uncertainties[key]} if key in uncertainties else {}),
                        }
                        for key, value in _admet_predictions_for_smiles(rows, smiles).items()
                    }
                },
                model_version=_admet_model_version_for_smiles(rows, smiles),
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
        return SimpleNamespace(
            smiles=smiles,
            result=result,
            model_version=str(getattr(result, "model_version", "")),
            model_artifact=_remote_model_artifact(result),
        )


class ADMETOracleServicer(oracle_pb2_grpc.OracleServiceServicer):
    def __init__(self, service: ADMETServicer | None = None):
        self._local_runtime = service is None
        self.service = service or ADMETServicer()

    async def Evaluate(self, request, context):
        return await self._evaluate(request, context, return_uncertainty=False)

    async def PredictWithUncertainty(self, request, context):
        return await self._evaluate(request, context, return_uncertainty=True)

    async def _evaluate(self, request, context, *, return_uncertainty: bool):
        start = time.perf_counter()
        try:
            request_context = validate_oracle_request(
                request,
                expected_level=oracle_pb2.L1_ML_SURROGATE,
                allowed_parameters=(),
            )
            artifacts = await resolve_oracle_artifact_refs(
                _artifact_status_objects() if self._local_runtime else []
            )
            evaluations = []
            for index, smiles in enumerate(request_context.molecules):
                item_start = time.perf_counter()
                try:
                    if return_uncertainty:
                        response = await self.service.Screen(
                            SimpleNamespace(
                                smiles=smiles,
                                properties=list(request_context.properties),
                            ),
                            context,
                        )
                        scores, uncertainties = _split_uncertainty_result(
                            response.result,
                            smiles,
                        )
                        elapsed_ms = int((time.perf_counter() - item_start) * 1000)
                    else:
                        response = await self.service.Predict(
                            SimpleNamespace(
                                smiles=smiles,
                                properties=list(request_context.properties),
                            ),
                            context,
                        )
                        scores = response.predictions
                        uncertainties = {}
                        elapsed_ms = int(response.elapsed_ms)
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
                            oracle_name="admet_ai",
                            elapsed_ms=int((time.perf_counter() - item_start) * 1000),
                            artifacts=artifacts,
                            error_code="COMPUTATION_ERROR",
                            error_message=str(exc),
                        )
                    )
                    continue
                evaluation_artifacts = list(artifacts)
                model_artifact = getattr(response, "model_artifact", None)
                if model_artifact is not None:
                    if not isinstance(model_artifact, audit_pb2.ArtifactRef):
                        raise OracleDataError("ADMET model_artifact must be an ArtifactRef")
                    evaluation_artifacts.append(model_artifact)
                evaluations.append(
                    build_oracle_evaluation(
                        request=request_context,
                        index=index,
                        oracle_name="admet_ai",
                        scores=scores,
                        uncertainties=uncertainties,
                        elapsed_ms=elapsed_ms,
                        artifacts=evaluation_artifacts,
                        model_version=str(getattr(response, "model_version", "")),
                    )
                )
            return build_oracle_response(
                request=request_context,
                evaluations=evaluations,
                total_elapsed_ms=int((time.perf_counter() - start) * 1000),
            )
        except (
            OracleRequestError,
            OracleUnavailableError,
            OracleDataError,
            TimeoutError,
            subprocess.TimeoutExpired,
        ) as exc:
            return await abort_oracle_error(context, exc)

    async def StreamEvaluate(self, request_iterator, context):
        async for request in request_iterator:
            yield await self.Evaluate(request, context)


class ADMETCommandRunner:
    def __init__(self, command: str) -> None:
        self.command = command
        self.timeout = _positive_timeout(
            os.environ.get(_COMMAND_TIMEOUT_ENV, "120"),
            _COMMAND_TIMEOUT_ENV,
        )

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
            raise OracleDataError(f"{_COMMAND_ENV} returned invalid JSON") from exc
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
        raise OracleDataError(f"{_COMMAND_ENV} must return a JSON object or list")
    if not isinstance(rows, list) or not rows:
        raise OracleDataError(f"{_COMMAND_ENV} response requires non-empty results")
    if not all(isinstance(row, dict) for row in rows):
        raise OracleDataError(f"{_COMMAND_ENV} result rows must be JSON objects")
    return rows


def _admet_predictions_for_smiles(rows: list[dict], smiles: str) -> dict[str, float]:
    row = _admet_row_for_smiles(rows, smiles)
    predictions = _numeric_map(row.get("predictions", row.get("scores", {})))
    return predictions


def _admet_uncertainties_for_smiles(rows: list[dict], smiles: str) -> dict[str, float]:
    row = _admet_row_for_smiles(rows, smiles)
    return _numeric_map(row.get("uncertainties"))


def _admet_row_for_smiles(rows: list[dict], smiles: str) -> dict:
    for row in rows:
        row_smiles = str(row.get("smiles") or row.get("molecule_smiles") or "")
        if row_smiles == smiles:
            return row
    raise OracleDataError(f"{_COMMAND_ENV} response missing result for {smiles}")


def _admet_model_version_for_smiles(rows: list[dict], smiles: str) -> str:
    row = _admet_row_for_smiles(rows, smiles)
    if row.get("validation_marker") == _VALIDATION_MARKER:
        return _VALIDATION_MARKER
    return str(row.get("model_version") or row.get("validation_marker") or "")


def _elapsed_ms_from_rows(rows: list[dict], smiles: str, start: float) -> int:
    row = _admet_row_for_smiles(rows, smiles)
    elapsed_ms = row.get("elapsed_ms")
    if isinstance(elapsed_ms, int | float):
        return int(elapsed_ms)
    return int((time.perf_counter() - start) * 1000)


def _numeric_map(values) -> dict[str, float]:
    if not isinstance(values, dict):
        raise OracleDataError("oracle numeric result must be an object")
    output = {}
    for key, value in values.items():
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise OracleDataError(f"oracle metric {key} must be numeric")
        number = float(value)
        if not math.isfinite(number):
            raise OracleDataError(f"oracle metric {key} must be finite")
        output[str(key)] = number
    return output


def _remote_model_artifact(result) -> audit_pb2.ArtifactRef | None:
    model_version = getattr(result, "model_version", "")
    artifact_name = getattr(result, "artifact_name", "")
    artifact_checksum = getattr(result, "artifact_checksum", "")
    if not model_version and not artifact_name and not artifact_checksum:
        return None
    if not all(
        isinstance(value, str) and value.strip()
        for value in (model_version, artifact_name, artifact_checksum)
    ):
        raise OracleDataError("ADMET remote model metadata is incomplete")
    return audit_pb2.ArtifactRef(
        name=artifact_name.strip(),
        version=model_version.strip(),
        checksum=artifact_checksum.strip(),
        required=True,
    )


def _positive_timeout(value: object, field_name: str) -> float:
    try:
        timeout = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{field_name} must be a finite positive number") from exc
    if isinstance(value, bool) or not math.isfinite(timeout) or timeout <= 0:
        raise RuntimeError(f"{field_name} must be a finite positive number")
    return timeout


def _split_uncertainty_result(
    result: object,
    smiles: str,
) -> tuple[dict[str, float], dict[str, float]]:
    row = result.get(smiles, result) if isinstance(result, dict) else {}
    if isinstance(row, tuple | list):
        if len(row) != 2:
            raise OracleDataError("ADMET uncertainty result must contain scores and uncertainties")
        return _numeric_map(row[0]), _numeric_map(row[1])
    scores = {}
    uncertainties = {}
    if not isinstance(row, dict):
        return scores, uncertainties
    for key, value in row.items():
        if isinstance(value, dict):
            if "value" in value:
                scores.update(_numeric_map({str(key): value["value"]}))
            if "uncertainty" in value:
                uncertainties.update(_numeric_map({str(key): value["uncertainty"]}))
        else:
            scores.update(_numeric_map({str(key): value}))
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
        getattr(request, "properties", None) or getattr(request, "requested_properties", None) or []
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
    _LOGGER.info("ADMET Prediction Service running on :50056")
    await server.wait_for_termination()


def _validation_response(payload: object) -> dict:
    _require_synthetic_validation_enabled()
    if not isinstance(payload, dict):
        raise ValueError("ADMET validation request must be a JSON object")
    expected_fields = {"smiles", "properties", "return_uncertainty"}
    unexpected = sorted(set(payload) - expected_fields)
    if unexpected:
        raise ValueError(
            "ADMET validation request has unexpected fields: " + ", ".join(unexpected)
        )
    missing = sorted(expected_fields - set(payload))
    if missing:
        raise ValueError(
            "ADMET validation request is missing fields: " + ", ".join(missing)
        )
    smiles = _validation_text_list(
        payload["smiles"],
        "smiles",
        maximum=_VALIDATION_MAX_BATCH_SIZE,
    )
    properties = _validation_text_list(
        payload["properties"],
        "properties",
        maximum=_VALIDATION_MAX_PROPERTIES,
    )
    if len(set(properties)) != len(properties):
        raise ValueError("ADMET validation properties must be unique")
    return_uncertainty = payload["return_uncertainty"]
    if not isinstance(return_uncertainty, bool):
        raise ValueError("ADMET validation return_uncertainty must be a boolean")
    rows = []
    for molecule_smiles in smiles:
        predictions = {
            property_name: _validation_metric(molecule_smiles, property_name)
            for property_name in properties
        }
        uncertainties = (
            {
                property_name: _validation_uncertainty(molecule_smiles, property_name)
                for property_name in properties
            }
            if return_uncertainty
            else {}
        )
        rows.append(
            {
                "smiles": molecule_smiles,
                "predictions": predictions,
                "uncertainties": uncertainties,
                "elapsed_ms": 0,
                "model_version": _VALIDATION_MARKER,
                "validation_marker": _VALIDATION_MARKER,
            }
        )
    return {
        "validation_marker": _VALIDATION_MARKER,
        "results": rows,
    }


def _validation_metric(smiles: str, property_name: str) -> float:
    fingerprint = _validation_fingerprint(f"{smiles}|{property_name}")
    return round((fingerprint % 1000) / 1000.0, 6)


def _validation_uncertainty(smiles: str, property_name: str) -> float:
    fingerprint = _validation_fingerprint(f"{property_name}|{smiles}")
    return round(0.05 + (fingerprint % 100) / 1000.0, 6)


def _validation_fingerprint(value: str) -> int:
    return sum(index * ord(character) for index, character in enumerate(value, start=1))


def _validation_text_list(value: object, field_name: str, *, maximum: int) -> list[str]:
    if not isinstance(value, list) or not value or len(value) > maximum:
        raise ValueError(
            f"ADMET validation {field_name} must be a non-empty list "
            f"with at most {maximum} items"
        )
    if any(
        not isinstance(item, str)
        or not item
        or item != item.strip()
        for item in value
    ):
        raise ValueError(
            f"ADMET validation {field_name} must contain non-empty trimmed strings"
        )
    return list(value)


def _require_synthetic_validation_enabled() -> None:
    if os.environ.get(_VALIDATION_GATE_ENV) != "true":
        raise RuntimeError(f"{_VALIDATION_GATE_ENV}=true is required")


def _run_validation_runner() -> None:
    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError as exc:
        raise ValueError("ADMET validation request must be valid JSON") from exc
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
        sys.stderr.write("ADMET service has unexpected command line arguments\n")
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
