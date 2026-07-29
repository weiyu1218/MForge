from __future__ import annotations

import base64
import hashlib
import inspect
import json
import math
import shlex
import shutil
import struct
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Protocol
from urllib.parse import unquote, urlparse

from google.protobuf.message import DecodeError

from mf_core.artifacts import RequirementStatus
from mf_core.geometry.lorentz import normalize_lorentz_embedding
from mf_core.proto_gen.moleculeforge.v1.core import audit_pb2, cig_pb2, humu_pb2
from mf_core.proto_gen.moleculeforge.v1.generator import generator_pb2
from mf_core.types.cig import (
    CIG,
    ObjectiveEdge,
    ObjectiveHyperedge,
    ObjectiveNode,
    ObjectiveType,
)
from mf_core.types.humu import HCIV, IntentCone
from mf_core.types.molecule import Molecule

GENERATOR_CONTEXT_SCHEMA = "generator_context.v1"
MOLECULE_PAYLOAD_SCHEMA = "molecule.v1"
HUMU_EMBEDDING_SCHEMA = "humu.float32.v1"
HUMU_COORDINATE_COUNT = 129
HUMU_FLOAT32_BYTES = HUMU_COORDINATE_COUNT * 4

_OBJECTIVE_TYPES = {
    cig_pb2.MAXIMIZE: ObjectiveType.MAXIMIZE,
    cig_pb2.MINIMIZE: ObjectiveType.MINIMIZE,
    cig_pb2.TARGET_RANGE: ObjectiveType.TARGET_RANGE,
    cig_pb2.CONSTRAINT: ObjectiveType.CONSTRAINT,
}


class GeneratorRequestError(ValueError):
    """The public generator request violates the wire contract."""


class GeneratorResultError(RuntimeError):
    """A generator returned an invalid or incomplete result."""


class _Digest(Protocol):
    def update(self, data: bytes) -> None: ...


@dataclass(frozen=True)
class GeneratorRequestContext:
    project_id: str
    request_id: str
    batch_size: int
    cig: CIG
    hciv: HCIV
    intent_cone: IntentCone


def validate_generate_request(
    request: generator_pb2.GenerateRequest,
    *,
    max_batch_size: int,
) -> GeneratorRequestContext:
    if isinstance(max_batch_size, bool) or max_batch_size <= 0:
        raise ValueError("max_batch_size must be a positive integer")
    project_id = str(getattr(request, "project_id", "")).strip()
    request_id = str(getattr(request, "request_id", "")).strip()
    schema_version = str(getattr(request, "context_schema_version", "")).strip()
    try:
        batch_size = int(getattr(request, "batch_size", 0))
    except (TypeError, ValueError) as exc:
        raise GeneratorRequestError("batch_size must be an integer") from exc

    if not project_id:
        raise GeneratorRequestError("project_id is required")
    if not request_id:
        raise GeneratorRequestError("request_id is required")
    if batch_size <= 0:
        raise GeneratorRequestError("batch_size must be positive")
    if batch_size > max_batch_size:
        raise GeneratorRequestError(f"batch_size must not exceed generator limit {max_batch_size}")
    if schema_version != GENERATOR_CONTEXT_SCHEMA:
        raise GeneratorRequestError(f"context_schema_version must be {GENERATOR_CONTEXT_SCHEMA}")

    has_field = getattr(request, "HasField", None)
    if not callable(has_field) or not has_field("cig"):
        raise GeneratorRequestError("cig is required")
    if not has_field("hciv"):
        raise GeneratorRequestError("hciv is required")

    cig_message = request.cig
    cig = _validate_cig(cig_message)
    if cig.project_id != project_id:
        raise GeneratorRequestError("cig.project_id must match project_id")

    hciv_message = request.hciv
    curvature = _finite_positive(hciv_message.curvature, "hciv.curvature")
    coordinates = normalize_lorentz_embedding(
        list(hciv_message.coordinates),
        expected_dim=HUMU_COORDINATE_COUNT,
        curvature=curvature,
    )
    if coordinates is None:
        raise GeneratorRequestError(
            "hciv.coordinates must be a finite 129-coordinate Lorentz point"
        )
    hciv = HCIV(
        coordinates=coordinates,
        dim=HUMU_COORDINATE_COUNT - 1,
        curvature=curvature,
        molecule_smiles=str(hciv_message.molecule_smiles),
        parent_hciv_id=(
            str(hciv_message.parent_hciv_id) if hciv_message.HasField("parent_hciv_id") else None
        ),
    )

    intent_cone = _validate_intent_cone(
        getattr(request, "intent_cone", b""),
        curvature=curvature,
    )
    return GeneratorRequestContext(
        project_id=project_id,
        request_id=request_id,
        batch_size=batch_size,
        cig=cig,
        hciv=hciv,
        intent_cone=intent_cone,
    )


def serialize_generator_results(
    molecules: Iterable[object],
    *,
    expected_count: int,
    canonicalize_smiles: Callable[[str], str],
) -> tuple[list[bytes], list[bytes]]:
    try:
        values = list(molecules)
    except Exception as exc:
        raise GeneratorResultError("generator result must be iterable") from exc
    if len(values) != expected_count:
        raise GeneratorResultError(
            f"generator returned {len(values)} molecules, expected {expected_count}"
        )

    payloads: list[bytes] = []
    embeddings: list[bytes | None] = []
    for molecule in values:
        payload, embedding = _normalize_molecule(
            molecule,
            canonicalize_smiles=canonicalize_smiles,
        )
        payloads.append(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        )
        embeddings.append(embedding)

    present_embeddings = [value for value in embeddings if value is not None]
    if present_embeddings and len(present_embeddings) != len(embeddings):
        raise GeneratorResultError(
            "HuMU embeddings must be absent for all molecules or present for every molecule"
        )
    return payloads, present_embeddings


def build_generate_response(
    *,
    generator_name: str,
    request: generator_pb2.GenerateRequest,
    molecules: Iterable[object],
    statuses: Sequence[RequirementStatus],
    elapsed_ms: int,
    canonicalize_smiles: Callable[[str], str],
) -> generator_pb2.GenerateResponse:
    batch_size = int(getattr(request, "batch_size", 0))
    molecule_payloads, embeddings = serialize_generator_results(
        molecules,
        expected_count=batch_size,
        canonicalize_smiles=canonicalize_smiles,
    )
    request_id = str(getattr(request, "request_id", ""))
    return generator_pb2.GenerateResponse(
        generator_name=generator_name,
        generation_id=str(getattr(request, "project_id", "")),
        molecules=molecule_payloads,
        humu_embeddings=embeddings,
        elapsed_ms=max(0, int(elapsed_ms)),
        request_id=request_id,
        artifacts=artifact_refs(statuses),
        molecule_payload_schema=MOLECULE_PAYLOAD_SCHEMA,
        embedding_payload_schema=HUMU_EMBEDDING_SCHEMA,
    )


async def build_generator_info(
    *,
    generator_name: str,
    generator: object,
    statuses: Sequence[RequirementStatus],
    fallback: Mapping[str, object],
) -> generator_pb2.GeneratorInfo:
    details = dict(fallback)
    info_method = getattr(generator, "info", None)
    if callable(info_method):
        result = info_method()
        if inspect.isawaitable(result):
            result = await result
        if not isinstance(result, Mapping):
            raise GeneratorResultError("generator info must be an object")
        details.update(result)

    ready, status_message = runtime_readiness(statuses)
    max_batch_size = int(details.get("max_batch_size", 0))
    if max_batch_size <= 0:
        ready = False
        status_message = "generator max_batch_size must be positive"
    return generator_pb2.GeneratorInfo(
        name=generator_name,
        version=str(details.get("version", "")),
        description=str(details.get("description", "")),
        supported_properties=[str(value) for value in details.get("supported_properties", [])],
        max_batch_size=max(0, max_batch_size),
        supports_streaming=bool(details.get("supports_streaming", False)),
        requires_gpu=bool(details.get("requires_gpu", False)),
        runtime_status=(
            audit_pb2.GENERATOR_RUNTIME_STATUS_READY
            if ready
            else audit_pb2.GENERATOR_RUNTIME_STATUS_UNAVAILABLE
        ),
        status_message=status_message,
        artifacts=artifact_refs(statuses),
    )


def runtime_readiness(
    statuses: Sequence[RequirementStatus],
) -> tuple[bool, str]:
    unavailable = [
        status
        for status in statuses
        if (status.required or status.configured) and not status.available
    ]
    if unavailable:
        return False, "; ".join(status.message for status in unavailable)
    if not statuses:
        return False, "generator runtime requirements are not configured"
    return True, "ready"


def artifact_refs(
    statuses: Sequence[RequirementStatus],
) -> list[audit_pb2.ArtifactRef]:
    return [
        audit_pb2.ArtifactRef(
            name=status.name,
            version="runtime.v1",
            checksum=_status_checksum(status) if status.available else "",
            required=status.required,
        )
        for status in statuses
    ]


def _validate_cig(message: cig_pb2.CIG) -> CIG:
    objectives: list[ObjectiveNode] = []
    for objective in message.objectives:
        objective_type = _OBJECTIVE_TYPES.get(objective.type)
        if objective_type is None:
            raise GeneratorRequestError(
                f"cig objective {objective.id or '<unknown>'} has an invalid type"
            )
        target_value = _finite(objective.target_value, "cig objective target_value")
        weight = _finite(objective.weight, "cig objective weight")
        target_min = (
            _finite(objective.target_min, "cig objective target_min")
            if objective.HasField("target_min")
            else None
        )
        target_max = (
            _finite(objective.target_max, "cig objective target_max")
            if objective.HasField("target_max")
            else None
        )
        objectives.append(
            ObjectiveNode(
                id=str(objective.id),
                name=str(objective.name),
                type=objective_type,
                target_value=target_value,
                target_min=target_min,
                target_max=target_max,
                property=str(objective.property),
                weight=weight,
                pareto_tier=int(objective.pareto_tier),
            )
        )
    edges = [
        ObjectiveEdge(
            source_id=str(edge.source_id),
            target_id=str(edge.target_id),
            relation=str(edge.relation),
            strength=_finite(edge.strength, "cig edge strength"),
        )
        for edge in message.edges
    ]
    hyperedges = [
        ObjectiveHyperedge(
            source_ids=[str(source_id) for source_id in edge.source_ids],
            target_ids=[str(target_id) for target_id in edge.target_ids],
            relation=str(edge.relation),
            strength=_finite(edge.strength, "cig hyperedge strength"),
        )
        for edge in message.hyperedges
    ]
    cig = CIG(
        project_id=str(message.project_id),
        objectives=objectives,
        edges=edges,
        hyperedges=hyperedges,
        constraints={str(key): str(value) for key, value in message.constraints.items()},
        created_by=str(message.created_by),
    )
    issues = cig.validate_consistency()
    if issues:
        raise GeneratorRequestError("; ".join(issues))
    return cig


def _validate_intent_cone(raw: object, *, curvature: float) -> IntentCone:
    if not isinstance(raw, bytes | bytearray) or not raw:
        raise GeneratorRequestError("intent_cone is required")
    try:
        message = humu_pb2.IntentCone.FromString(bytes(raw))
    except DecodeError as exc:
        raise GeneratorRequestError("intent_cone must be a serialized IntentCone") from exc

    raw_axis = [_finite(value, "intent_cone.axis") for value in message.axis]
    half_angle = _finite_positive(message.half_angle, "intent_cone.half_angle")
    if half_angle > math.pi:
        raise GeneratorRequestError("intent_cone.half_angle must not exceed pi")
    cone_curvature = _finite_positive(message.curvature, "intent_cone.curvature")
    if not math.isclose(cone_curvature, curvature, rel_tol=1e-6, abs_tol=1e-8):
        raise GeneratorRequestError("intent_cone.curvature must match hciv.curvature")
    axis = normalize_lorentz_embedding(
        raw_axis,
        expected_dim=HUMU_COORDINATE_COUNT,
        curvature=cone_curvature,
    )
    if axis is None:
        raise GeneratorRequestError(
            "intent_cone.axis must be a finite 129-coordinate Lorentz point"
        )
    property_weights = {
        str(key): _finite(value, f"intent_cone.property_weights[{key}]")
        for key, value in message.property_weights.items()
    }
    return IntentCone(
        axis=axis,
        half_angle=half_angle,
        angle_radians=half_angle,
        curvature=cone_curvature,
        property_weights=property_weights,
    )


def _normalize_molecule(
    molecule: object,
    *,
    canonicalize_smiles: Callable[[str], str],
) -> tuple[dict[str, object], bytes | None]:
    if hasattr(molecule, "model_dump"):
        raw = molecule.model_dump(mode="python")
    elif isinstance(molecule, Mapping):
        raw = dict(molecule)
    else:
        raise GeneratorResultError(f"unsupported molecule payload: {type(molecule)!r}")
    if not isinstance(raw, dict):
        raise GeneratorResultError("molecule payload must be an object")
    smiles = raw.get("canonical_smiles") or raw.get("smiles")
    if not isinstance(smiles, str) or not smiles.strip():
        raise GeneratorResultError("molecule payload requires a nonempty smiles")
    try:
        canonical_smiles = canonicalize_smiles(smiles.strip())
    except Exception as exc:
        raise GeneratorResultError("molecule payload contains invalid SMILES") from exc
    if not isinstance(canonical_smiles, str) or not canonical_smiles.strip():
        raise GeneratorResultError("SMILES canonicalizer returned an empty value")
    raw["smiles"] = canonical_smiles.strip()
    raw["canonical_smiles"] = canonical_smiles.strip()

    embedding_value = raw.pop("humu_embedding", None)
    if embedding_value is None:
        embedding_value = raw.pop("embedding", None)
    else:
        raw.pop("embedding", None)
    embedding = _normalize_embedding(embedding_value)
    return _json_safe(raw), embedding


def _normalize_embedding(value: object) -> bytes | None:
    if value is None:
        return None
    if isinstance(value, bytes | bytearray):
        payload = bytes(value)
        if len(payload) != HUMU_FLOAT32_BYTES:
            raise GeneratorResultError(f"HuMU embedding must be {HUMU_FLOAT32_BYTES} bytes")
        coordinates = struct.unpack("<129f", payload)
    elif isinstance(value, Sequence) and not isinstance(value, str):
        if len(value) != HUMU_COORDINATE_COUNT:
            raise GeneratorResultError("HuMU embedding must contain 129 coordinates")
        coordinates = tuple(_finite_result(item, "HuMU embedding") for item in value)
        try:
            payload = struct.pack("<129f", *coordinates)
        except (OverflowError, struct.error) as exc:
            raise GeneratorResultError("HuMU embedding is not float32 encodable") from exc
    else:
        raise GeneratorResultError("HuMU embedding has an unsupported representation")
    if not all(math.isfinite(value) for value in coordinates):
        raise GeneratorResultError("HuMU embedding must contain only finite values")
    return payload


def _json_safe(value: object) -> object:
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise GeneratorResultError("molecule payload contains a non-finite value")
        return value
    if isinstance(value, bytes | bytearray):
        return base64.b64encode(bytes(value)).decode("ascii")
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence):
        return [_json_safe(item) for item in value]
    raise GeneratorResultError(f"molecule payload contains an unsupported value: {type(value)!r}")


def _status_checksum(status: RequirementStatus) -> str:
    if not status.path:
        return ""
    path = _path_from_status(status.path)
    if path is None:
        return ""
    resolved = path.resolve()
    return _checksum_path(resolved, _checksum_signature(resolved))


@lru_cache(maxsize=128)
def _checksum_path(
    path: Path,
    signature: tuple[tuple[str, int, int], ...],
) -> str:
    del signature
    digest = hashlib.sha256()
    if path.is_file():
        _update_digest_from_file(digest, path)
        return f"sha256:{digest.hexdigest()}"
    if path.is_dir():
        files = sorted(item for item in path.rglob("*") if item.is_file())
        for item in files:
            relative = item.relative_to(path).as_posix().encode("utf-8")
            digest.update(struct.pack("<Q", len(relative)))
            digest.update(relative)
            _update_digest_from_file(digest, item)
        return f"sha256:{digest.hexdigest()}"
    return ""


def _checksum_signature(path: Path) -> tuple[tuple[str, int, int], ...]:
    if path.is_file():
        status = path.stat()
        return (("", status.st_mtime_ns, status.st_size),)
    if path.is_dir():
        signature = []
        for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
            status = item.stat()
            signature.append(
                (item.relative_to(path).as_posix(), status.st_mtime_ns, status.st_size)
            )
        return tuple(signature)
    return ()


def _path_from_status(raw: str) -> Path | None:
    parsed = urlparse(raw)
    if parsed.scheme == "file":
        return Path(unquote(parsed.path))
    candidate = Path(raw).expanduser()
    if candidate.exists():
        return candidate
    try:
        command = shlex.split(raw)
    except ValueError:
        return None
    if not command:
        return None
    executable = Path(command[0]).expanduser()
    if executable.is_file():
        return executable
    resolved = shutil.which(command[0])
    return Path(resolved) if resolved else None


def _update_digest_from_file(digest: _Digest, path: Path) -> None:
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)


def _finite(value: object, field: str) -> float:
    if isinstance(value, bool):
        raise GeneratorRequestError(f"{field} must be numeric")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise GeneratorRequestError(f"{field} must be numeric") from exc
    if not math.isfinite(parsed):
        raise GeneratorRequestError(f"{field} must be finite")
    return parsed


def _finite_positive(value: object, field: str) -> float:
    parsed = _finite(value, field)
    if parsed <= 0.0:
        raise GeneratorRequestError(f"{field} must be positive")
    return parsed


def _finite_result(value: object, field: str) -> float:
    if isinstance(value, bool):
        raise GeneratorResultError(f"{field} must be numeric")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise GeneratorResultError(f"{field} must be numeric") from exc
    if not math.isfinite(parsed):
        raise GeneratorResultError(f"{field} must be finite")
    return parsed


class GeneratorPlugin(ABC):
    @abstractmethod
    async def generate(
        self,
        batch_size: int,
        intent_cone: IntentCone | None = None,
        **kwargs: object,
    ) -> list[Molecule]: ...

    @abstractmethod
    async def info(self) -> dict[str, object]: ...

    async def generate_stream(
        self, batch_size: int, total: int, **kwargs: object
    ) -> AsyncIterator[list[Molecule]]:
        for _ in range(0, total, batch_size):
            yield await self.generate(batch_size, **kwargs)
