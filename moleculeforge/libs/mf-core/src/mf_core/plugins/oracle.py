from __future__ import annotations

import asyncio
import hashlib
import math
import shlex
import shutil
import struct
import subprocess
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse

import grpc

from mf_core.artifacts import RequirementStatus
from mf_core.proto_gen.moleculeforge.v1.core import audit_pb2
from mf_core.proto_gen.moleculeforge.v1.oracle import oracle_pb2


class OracleRequestError(ValueError):
    """The public oracle request violates the wire contract."""


class OracleUnavailableError(RuntimeError):
    """The configured oracle runtime is unavailable."""


class OracleDataError(RuntimeError):
    """The oracle runtime returned corrupt or contradictory data."""


SYNTHETIC_VALIDATION_MARKER = "synthetic_pipeline_validation_only"


@dataclass(frozen=True)
class OracleRequestContext:
    project_id: str
    request_id: str
    molecules: tuple[str, ...]
    properties: tuple[str, ...]
    level: int
    receptor_uri: str
    protein_pdb_id: str
    reference_ligand_smiles: str
    parameters: Mapping[str, str]


class OraclePlugin(ABC):
    @abstractmethod
    async def evaluate(
        self, molecules: list[str], properties: list[str]
    ) -> dict[str, dict[str, float]]: ...

    @abstractmethod
    async def predict_with_uncertainty(
        self, molecules: list[str], properties: list[str]
    ) -> dict[str, tuple[dict, dict]]: ...

    @abstractmethod
    def oracle_level(self) -> int: ...


def validate_oracle_request(
    request: oracle_pb2.OracleBatchRequest,
    *,
    expected_level: int,
    require_receptor_uri: bool = False,
    require_protein_pdb_id: bool = False,
    require_reference_ligand: bool = False,
    required_parameters: Sequence[str] = (),
    allowed_parameters: Sequence[str] = (),
) -> OracleRequestContext:
    project_id = str(getattr(request, "project_id", "")).strip()
    request_id = str(getattr(request, "request_id", "")).strip()
    molecules = tuple(str(value).strip() for value in request.molecule_smiles)
    properties = tuple(str(value).strip() for value in request.requested_properties)
    level = int(getattr(request, "level", 0))
    receptor_uri = str(getattr(request, "receptor_uri", "")).strip()
    protein_pdb_id = str(getattr(request, "protein_pdb_id", "")).strip()
    reference_ligand = str(getattr(request, "reference_ligand_smiles", "")).strip()
    raw_parameters = [
        (str(key).strip(), str(value).strip())
        for key, value in getattr(request, "oracle_parameters", {}).items()
    ]
    parameters = dict(raw_parameters)

    if not project_id:
        raise OracleRequestError("project_id is required")
    if not request_id:
        raise OracleRequestError("request_id is required")
    if not molecules or any(not value for value in molecules):
        raise OracleRequestError("molecule_smiles must contain non-empty values")
    if not properties or any(not value for value in properties):
        raise OracleRequestError("requested_properties must contain non-empty values")
    if len(set(properties)) != len(properties):
        raise OracleRequestError("requested_properties must not contain duplicates")
    if any(not key for key, _ in raw_parameters):
        raise OracleRequestError("oracle_parameters keys must be non-empty")
    if len(parameters) != len(raw_parameters):
        raise OracleRequestError("oracle_parameters keys must be unique")
    unknown_parameters = sorted(set(parameters) - set(allowed_parameters))
    if unknown_parameters:
        raise OracleRequestError(f"unknown oracle_parameters: {', '.join(unknown_parameters)}")
    if level != expected_level:
        level_name = oracle_pb2.OracleLevel.Name(expected_level)
        raise OracleRequestError(f"level must be {level_name}")
    if require_receptor_uri and not receptor_uri:
        raise OracleRequestError("receptor_uri is required")
    if require_protein_pdb_id and not protein_pdb_id:
        raise OracleRequestError("protein_pdb_id is required")
    if require_reference_ligand and not reference_ligand:
        raise OracleRequestError("reference_ligand_smiles is required")
    missing_parameters = [name for name in required_parameters if not parameters.get(name, "")]
    if missing_parameters:
        raise OracleRequestError(f"oracle_parameters requires: {', '.join(missing_parameters)}")
    return OracleRequestContext(
        project_id=project_id,
        request_id=request_id,
        molecules=molecules,
        properties=properties,
        level=level,
        receptor_uri=receptor_uri,
        protein_pdb_id=protein_pdb_id,
        reference_ligand_smiles=reference_ligand,
        parameters=parameters,
    )


def build_oracle_evaluation(
    *,
    request: OracleRequestContext,
    index: int,
    oracle_name: str,
    scores: Mapping[str, object],
    uncertainties: Mapping[str, object] | None,
    elapsed_ms: int,
    artifacts: Sequence[audit_pb2.ArtifactRef],
    oracle_version: str = "",
    model_version: str = "",
    units: Mapping[str, str] | None = None,
) -> oracle_pb2.OracleEvaluation:
    molecule = _molecule_at(request, index)
    resolved_model_version = str(model_version or "")
    if resolved_model_version == SYNTHETIC_VALIDATION_MARKER:
        return build_oracle_error_evaluation(
            request=request,
            index=index,
            oracle_name=oracle_name,
            elapsed_ms=elapsed_ms,
            artifacts=artifacts,
            error_code="SYNTHETIC_VALIDATION_ONLY",
            error_message="synthetic validation output cannot satisfy an oracle evaluation",
            oracle_version=oracle_version,
            model_version=resolved_model_version,
        )
    numeric_scores = _numeric_mapping(scores, "scores")
    numeric_uncertainties = _numeric_mapping(
        uncertainties or {},
        "uncertainties",
    )
    if any(value < 0 for value in numeric_uncertainties.values()):
        raise OracleDataError("uncertainties must be non-negative")
    missing = [name for name in request.properties if name not in numeric_scores]
    if missing:
        return build_oracle_error_evaluation(
            request=request,
            index=index,
            oracle_name=oracle_name,
            elapsed_ms=elapsed_ms,
            artifacts=artifacts,
            error_code="MISSING_METRIC",
            error_message=f"missing requested metrics: {', '.join(missing)}",
            oracle_version=oracle_version,
            model_version=model_version,
        )

    requested_scores = {name: numeric_scores[name] for name in request.properties}
    requested_uncertainties = {
        name: numeric_uncertainties[name]
        for name in request.properties
        if name in numeric_uncertainties
    }
    metric_units = units or {}
    metrics = []
    for property_name in request.properties:
        kwargs = {
            "property": property_name,
            "value": requested_scores[property_name],
            "unit": str(metric_units.get(property_name, "")),
        }
        if property_name in requested_uncertainties:
            kwargs["uncertainty"] = requested_uncertainties[property_name]
        metrics.append(oracle_pb2.OracleMetric(**kwargs))
    return oracle_pb2.OracleEvaluation(
        oracle_name=str(oracle_name),
        molecule_smiles=molecule,
        level=request.level,
        scores=requested_scores,
        uncertainties=requested_uncertainties,
        elapsed_ms=_elapsed_milliseconds(elapsed_ms, "elapsed_ms"),
        success=True,
        outcome=oracle_pb2.ORACLE_OUTCOME_PASS,
        oracle_version=str(oracle_version or ""),
        model_version=resolved_model_version,
        artifact_refs=artifacts,
        evidence_id=f"{request.request_id}:{oracle_name}:{index}",
        metrics=metrics,
    )


def build_oracle_error_evaluation(
    *,
    request: OracleRequestContext,
    index: int,
    oracle_name: str,
    elapsed_ms: int,
    artifacts: Sequence[audit_pb2.ArtifactRef],
    error_code: str,
    error_message: str,
    oracle_version: str = "",
    model_version: str = "",
) -> oracle_pb2.OracleEvaluation:
    return oracle_pb2.OracleEvaluation(
        oracle_name=str(oracle_name),
        molecule_smiles=_molecule_at(request, index),
        level=request.level,
        elapsed_ms=_elapsed_milliseconds(elapsed_ms, "elapsed_ms"),
        success=False,
        error_message=str(error_message),
        outcome=oracle_pb2.ORACLE_OUTCOME_ERROR,
        oracle_version=str(oracle_version or ""),
        model_version=str(model_version or ""),
        artifact_refs=artifacts,
        evidence_id=f"{request.request_id}:{oracle_name}:{index}",
        error_code=str(error_code),
    )


def build_oracle_response(
    *,
    request: OracleRequestContext,
    evaluations: Sequence[oracle_pb2.OracleEvaluation],
    total_elapsed_ms: int,
) -> oracle_pb2.OracleBatchResponse:
    if len(evaluations) != len(request.molecules):
        raise OracleDataError(
            f"oracle returned {len(evaluations)} evaluations, expected {len(request.molecules)}"
        )
    actual_order = tuple(evaluation.molecule_smiles for evaluation in evaluations)
    if actual_order != request.molecules:
        raise OracleDataError("oracle evaluations do not match request molecule order")
    return oracle_pb2.OracleBatchResponse(
        evaluations=evaluations,
        batch_id=request.request_id,
        total_elapsed_ms=_elapsed_milliseconds(
            total_elapsed_ms,
            "total_elapsed_ms",
        ),
    )


async def abort_oracle_error(context, error: BaseException):
    if isinstance(error, OracleRequestError):
        code = grpc.StatusCode.INVALID_ARGUMENT
    elif isinstance(error, OracleUnavailableError):
        code = grpc.StatusCode.FAILED_PRECONDITION
    elif isinstance(error, TimeoutError | subprocess.TimeoutExpired):
        code = grpc.StatusCode.DEADLINE_EXCEEDED
    elif isinstance(error, OracleDataError):
        code = grpc.StatusCode.DATA_LOSS
    else:
        raise error
    if context is not None and hasattr(context, "abort"):
        await context.abort(code, str(error) or type(error).__name__)
    raise error


def oracle_artifact_refs(
    statuses: Sequence[RequirementStatus],
) -> list[audit_pb2.ArtifactRef]:
    return [
        audit_pb2.ArtifactRef(
            name=status.name,
            version="",
            checksum=_artifact_checksum(status),
            required=status.required,
        )
        for status in statuses
    ]


async def resolve_oracle_artifact_refs(
    statuses: Sequence[RequirementStatus],
) -> list[audit_pb2.ArtifactRef]:
    refs = await asyncio.to_thread(oracle_artifact_refs, statuses)
    for status, ref in zip(statuses, refs, strict=True):
        if status.required and not status.available:
            raise OracleUnavailableError(f"required oracle artifact is unavailable: {status.name}")
        if status.required and not ref.checksum:
            raise OracleUnavailableError(f"required oracle artifact has no checksum: {status.name}")
    return refs


def parse_positive_parameter(
    parameters: Mapping[str, str],
    name: str,
    *,
    default: int | None = None,
) -> int:
    raw = parameters.get(name)
    if raw in (None, ""):
        if default is None:
            raise OracleRequestError(f"oracle_parameters[{name}] is required")
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise OracleRequestError(f"oracle_parameters[{name}] must be a positive integer") from exc
    if value <= 0:
        raise OracleRequestError(f"oracle_parameters[{name}] must be a positive integer")
    return value


def _molecule_at(request: OracleRequestContext, index: int) -> str:
    if isinstance(index, bool) or index < 0 or index >= len(request.molecules):
        raise OracleDataError("oracle evaluation index is out of range")
    return request.molecules[index]


def _numeric_mapping(
    values: Mapping[str, object],
    field_name: str,
) -> dict[str, float]:
    if not isinstance(values, Mapping):
        raise OracleDataError(f"{field_name} must be an object")
    output: dict[str, float] = {}
    for key, value in values.items():
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise OracleDataError(f"{field_name}[{key}] must be numeric")
        number = float(value)
        if not math.isfinite(number):
            raise OracleDataError(f"{field_name}[{key}] must be finite")
        output[str(key)] = number
    return output


def _elapsed_milliseconds(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise OracleDataError(f"{field_name} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise OracleDataError(f"{field_name} must be finite and non-negative")
    return int(number)


def _artifact_checksum(status: RequirementStatus) -> str:
    if not status.available or not status.path:
        return ""
    normalized_arguments, paths = _artifact_material(status.path)
    if not normalized_arguments or not paths:
        return ""
    digest = hashlib.sha256()
    digest.update(b"moleculeforge.oracle-artifact.v1")
    for argument in normalized_arguments:
        encoded = argument.encode("utf-8")
        digest.update(struct.pack("<Q", len(encoded)))
        digest.update(encoded)
    for index, path in enumerate(paths):
        digest.update(struct.pack("<Q", index))
        encoded_path = str(path).encode("utf-8")
        digest.update(struct.pack("<Q", len(encoded_path)))
        digest.update(encoded_path)
        if path.is_file():
            _update_checksum(digest, path)
            continue
        for child in sorted(item for item in path.rglob("*") if item.is_file()):
            relative = child.relative_to(path).as_posix().encode("utf-8")
            digest.update(struct.pack("<Q", len(relative)))
            digest.update(relative)
            _update_checksum(digest, child)
    return f"sha256:{digest.hexdigest()}"


def _artifact_material(raw: str) -> tuple[tuple[str, ...], tuple[Path, ...]]:
    parsed = urlparse(raw)
    if parsed.scheme == "file":
        path = Path(unquote(parsed.path)).expanduser()
        if not path.exists():
            return (), ()
        resolved = path.resolve()
        return (str(resolved),), (resolved,)
    candidate = Path(raw).expanduser()
    if candidate.exists():
        resolved = candidate.resolve()
        return (str(resolved),), (resolved,)
    try:
        command = shlex.split(raw)
    except ValueError:
        return (), ()
    if not command:
        return (), ()
    normalized = []
    paths: list[Path] = []
    executable = Path(command[0]).expanduser()
    if executable.is_file():
        resolved_executable = executable.resolve()
    else:
        resolved = shutil.which(command[0])
        if not resolved:
            return (), ()
        resolved_executable = Path(resolved).resolve()
    normalized.append(str(resolved_executable))
    paths.append(resolved_executable)
    for argument in command[1:]:
        normalized_argument, argument_paths = _normalize_artifact_argument(argument)
        normalized.append(normalized_argument)
        for path in argument_paths:
            if path not in paths:
                paths.append(path)
    return tuple(normalized), tuple(paths)


def _normalize_artifact_argument(argument: str) -> tuple[str, tuple[Path, ...]]:
    prefix = ""
    value = argument
    if "=" in argument:
        option, possible_path = argument.split("=", 1)
        if option.startswith("-"):
            prefix = f"{option}="
            value = possible_path
    parsed = urlparse(value)
    if parsed.scheme == "file":
        candidate = Path(unquote(parsed.path)).expanduser()
    else:
        candidate = Path(value).expanduser()
    if not candidate.exists():
        return argument, ()
    resolved = candidate.resolve()
    return f"{prefix}{resolved}", (resolved,)


def _update_checksum(digest, path: Path) -> None:
    digest.update(struct.pack("<Q", path.stat().st_size))
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
