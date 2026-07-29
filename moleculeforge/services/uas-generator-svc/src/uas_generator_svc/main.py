"""Production gRPC service for unfamiliarity-aware sampling."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import signal
import time
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Never

import grpc
import torch
from mf_chem.molecule.parsing import canonicalize
from mf_core.artifacts import RequirementStatus
from mf_core.plugins.generator import (
    GENERATOR_CONTEXT_SCHEMA,
    GeneratorRequestError,
    GeneratorResultError,
    build_generate_response,
    build_generator_info,
    validate_generate_request,
)
from mf_core.proto_gen.moleculeforge.v1.core import cig_pb2
from mf_core.proto_gen.moleculeforge.v1.generator import generator_pb2, generator_pb2_grpc
from mf_generators.uas.autoencoder.molecule_ae import MoleculeAutoencoder
from mf_generators.uas.generator import UASGenerator

_COMMAND_SCHEMA = "uas.command.v1"
_MANIFEST_SCHEMA = "uas_training.v1"
_HUMU_DIM = 129
_MAX_SAMPLES = 256
_CHECKSUM_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")
_DECODER_CANDIDATE_FIELDS = frozenset({"id", "smiles", "canonical_smiles", "properties"})
_REQUIRED_ENV_VARS = (
    "UAS_AUTOENCODER_PATH",
    "UAS_ARTIFACT_MANIFEST_PATH",
    "UAS_CANDIDATE_SOURCE_COMMAND",
    "UAS_DECODER_COMMAND",
)


@dataclass(frozen=True)
class UASRuntime:
    checkpoint_path: Path
    manifest_path: Path
    checksum: str
    dim: int
    latent_dim: int
    model: MoleculeAutoencoder
    candidate_command: tuple[str, ...]
    decoder_command: tuple[str, ...]
    command_timeout_seconds: float
    validated: bool = False


@dataclass(frozen=True)
class _ValidatedRequest:
    project_id: str
    request_id: str
    n_samples: int
    seed: int | None
    context: dict[str, Any]


class _RequestCommandAdapter:
    def __init__(self, runtime: UASRuntime, request: _ValidatedRequest) -> None:
        self.runtime = runtime
        self.request = request
        self.attempt = 0

    async def candidates(self, n_samples: int) -> torch.Tensor:
        self.attempt += 1
        return await _candidate_embeddings(
            self.runtime,
            operation="sample",
            n_samples=n_samples,
            seed=self.request.seed,
            attempt=self.attempt,
            context=self.request.context,
        )

    async def decode(self, embeddings: torch.Tensor) -> list[dict[str, Any]]:
        return await _decode_embeddings(
            self.runtime,
            operation="decode",
            embeddings=embeddings,
            context=self.request.context,
        )


class UASGeneratorServicer(generator_pb2_grpc.GeneratorServiceServicer):
    def __init__(self, runtime: UASRuntime) -> None:
        if not runtime.validated:
            raise RuntimeError("UAS runtime must complete validation before serving")
        self.runtime = runtime

    async def Generate(  # noqa: N802
        self,
        request: generator_pb2.GenerateRequest,
        context: grpc.aio.ServicerContext,
    ) -> generator_pb2.GenerateResponse:
        try:
            validated = _validate_generate_request(request)
        except (GeneratorRequestError, TypeError, ValueError) as exc:
            return await _abort(context, grpc.StatusCode.INVALID_ARGUMENT, str(exc), exc)
        adapter = _RequestCommandAdapter(self.runtime, validated)
        generator = UASGenerator(
            dim=self.runtime.dim,
            autoencoder=self.runtime.model,
            candidate_source=adapter.candidates,
            decoder=adapter.decode,
        )
        started = time.perf_counter()
        try:
            molecules = [
                molecule
                async for molecule in generator.generate(
                    validated.context["hciv"],
                    validated.context["intent_cone"],
                    validated.context["cig"],
                    n_samples=validated.n_samples,
                    seed=validated.seed,
                )
            ]
            if len(molecules) != validated.n_samples:
                raise RuntimeError("UAS generation did not return the requested molecule count")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            return await _abort(
                context,
                grpc.StatusCode.FAILED_PRECONDITION,
                str(exc),
                exc,
            )
        try:
            return build_generate_response(
                generator_name="uas",
                request=request,
                molecules=molecules,
                statuses=_runtime_statuses(self.runtime),
                elapsed_ms=int((time.perf_counter() - started) * 1000),
                canonicalize_smiles=canonicalize,
            )
        except GeneratorResultError as exc:
            return await _abort(context, grpc.StatusCode.INTERNAL, str(exc), exc)

    async def GenerateStream(  # noqa: N802
        self,
        request_iterator: AsyncIterator[generator_pb2.GenerateRequest],
        context: grpc.aio.ServicerContext,
    ) -> AsyncIterator[generator_pb2.GenerateResponse]:
        async for request in request_iterator:
            yield await self.Generate(request, context)

    async def BatchGenerate(  # noqa: N802
        self,
        request_iterator: AsyncIterator[generator_pb2.GenerateRequest],
        context: grpc.aio.ServicerContext,
    ) -> AsyncIterator[generator_pb2.GenerateResponse]:
        async for request in request_iterator:
            yield await self.Generate(request, context)

    async def Info(  # noqa: N802
        self,
        request: generator_pb2.GeneratorInfo,
        context: grpc.aio.ServicerContext,
    ) -> generator_pb2.GeneratorInfo:
        del request, context
        return await build_generator_info(
            generator_name="uas",
            generator=self.runtime,
            statuses=_runtime_statuses(self.runtime),
            fallback={
                "version": "0.1.0",
                "description": "Unfamiliarity-aware sampling generator",
                "supported_properties": [],
                "max_batch_size": _MAX_SAMPLES,
                "supports_streaming": True,
                "requires_gpu": False,
            },
        )


async def load_runtime(
    env: Mapping[str, str] | None = None,
    *,
    command_timeout_seconds: float = 30.0,
) -> UASRuntime:
    configured = os.environ if env is None else env
    values = {name: str(configured.get(name, "")).strip() for name in _REQUIRED_ENV_VARS}
    missing = [name for name, value in values.items() if not value]
    if missing:
        raise RuntimeError(f"Missing required UAS environment variables: {', '.join(missing)}")
    if not math.isfinite(command_timeout_seconds) or command_timeout_seconds <= 0:
        raise ValueError("command_timeout_seconds must be positive")

    checkpoint_path = _required_file(values["UAS_AUTOENCODER_PATH"], "UAS_AUTOENCODER_PATH")
    manifest_path = _required_file(
        values["UAS_ARTIFACT_MANIFEST_PATH"],
        "UAS_ARTIFACT_MANIFEST_PATH",
    )
    candidate_command = _command_argv(
        values["UAS_CANDIDATE_SOURCE_COMMAND"],
        "UAS_CANDIDATE_SOURCE_COMMAND",
    )
    decoder_command = _command_argv(
        values["UAS_DECODER_COMMAND"],
        "UAS_DECODER_COMMAND",
    )
    manifest = _load_manifest(manifest_path)
    dim = _positive_int(manifest.get("dim"), "manifest dim")
    latent_dim = _positive_int(manifest.get("latent_dim"), "manifest latent_dim")
    if dim != _HUMU_DIM:
        raise RuntimeError(f"UAS manifest dim must be {_HUMU_DIM}")
    manifest_checkpoint = _manifest_checkpoint_path(manifest_path, manifest)
    if manifest_checkpoint != checkpoint_path:
        raise RuntimeError(
            "UAS_AUTOENCODER_PATH must match the checkpoint referenced by the manifest"
        )
    expected_checksum = manifest.get("autoencoder_sha256")
    if not isinstance(expected_checksum, str) or not _CHECKSUM_PATTERN.fullmatch(expected_checksum):
        raise RuntimeError("UAS manifest autoencoder_sha256 must be sha256:<64 lowercase hex>")
    actual_checksum = _sha256(checkpoint_path)
    if actual_checksum != expected_checksum:
        raise RuntimeError("UAS autoencoder checksum does not match the manifest")
    model = _load_autoencoder(checkpoint_path, dim=dim, latent_dim=latent_dim)
    runtime = UASRuntime(
        checkpoint_path=checkpoint_path,
        manifest_path=manifest_path,
        checksum=actual_checksum,
        dim=dim,
        latent_dim=latent_dim,
        model=model,
        candidate_command=candidate_command,
        decoder_command=decoder_command,
        command_timeout_seconds=float(command_timeout_seconds),
    )
    await _probe_runtime(runtime)
    return replace(runtime, validated=True)


def _validate_generate_request(request: object) -> _ValidatedRequest:
    request_context = validate_generate_request(
        request,
        max_batch_size=_MAX_SAMPLES,
    )
    cig_payload = _cig_payload(request.cig)
    _validate_hyperedges(cig_payload)
    generator_params = {str(key): str(value) for key, value in request.generator_params.items()}
    seed = _request_seed(generator_params)
    return _ValidatedRequest(
        project_id=request_context.project_id,
        request_id=request_context.request_id,
        n_samples=request_context.batch_size,
        seed=seed,
        context={
            "project_id": request_context.project_id,
            "request_id": request_context.request_id,
            "context_schema_version": GENERATOR_CONTEXT_SCHEMA,
            "cig": cig_payload,
            "hciv": request_context.hciv.model_dump(mode="json"),
            "intent_cone": request_context.intent_cone.model_dump(mode="json"),
            "generator_params": generator_params,
        },
    )


def _cig_payload(cig: cig_pb2.CIG) -> dict[str, Any]:
    return {
        "project_id": str(cig.project_id),
        "objectives": [
            {
                "id": str(item.id),
                "name": str(item.name),
                "type": cig_pb2.ObjectiveType.Name(item.type).lower(),
                "target_value": float(item.target_value),
                "target_min": (float(item.target_min) if item.HasField("target_min") else None),
                "target_max": (float(item.target_max) if item.HasField("target_max") else None),
                "property": str(item.property),
                "weight": float(item.weight),
                "pareto_tier": int(item.pareto_tier),
            }
            for item in cig.objectives
        ],
        "edges": [
            {
                "source_id": str(item.source_id),
                "target_id": str(item.target_id),
                "relation": str(item.relation),
                "strength": float(item.strength),
            }
            for item in cig.edges
        ],
        "hyperedges": [
            {
                "source_ids": [str(value) for value in item.source_ids],
                "target_ids": [str(value) for value in item.target_ids],
                "relation": str(item.relation),
                "strength": float(item.strength),
            }
            for item in cig.hyperedges
        ],
        "constraints": {str(key): str(value) for key, value in cig.constraints.items()},
        "created_by": str(cig.created_by),
    }


def _validate_hyperedges(cig: Mapping[str, Any]) -> None:
    node_ids = {objective["id"] for objective in cig["objectives"]}
    for hyperedge in cig["hyperedges"]:
        if not hyperedge["source_ids"] or not hyperedge["target_ids"]:
            raise ValueError("cig hyperedges require source_ids and target_ids")
        unknown = (set(hyperedge["source_ids"]) | set(hyperedge["target_ids"])) - node_ids
        if unknown:
            raise ValueError("cig hyperedge references an unknown objective")
        if not math.isfinite(hyperedge["strength"]):
            raise ValueError("cig hyperedge strength must be finite")


def _request_seed(generator_params: Mapping[str, str]) -> int | None:
    raw_seed = generator_params.get("sampling_seed", generator_params.get("seed", ""))
    if not raw_seed:
        return None
    try:
        return int(raw_seed)
    except ValueError as exc:
        raise ValueError("generator sampling seed must be an integer") from exc


async def _abort(
    context: grpc.aio.ServicerContext | None,
    code: grpc.StatusCode,
    message: str,
    cause: Exception,
) -> Never:
    if context is not None and hasattr(context, "abort"):
        await context.abort(code, message)
    raise cause


def _runtime_statuses(runtime: UASRuntime) -> list[RequirementStatus]:
    return [
        RequirementStatus(
            name="uas_autoencoder",
            configured=True,
            available=runtime.validated,
            required=True,
            path=str(runtime.checkpoint_path),
            source="UAS_AUTOENCODER_PATH",
            message="available" if runtime.validated else "runtime validation is incomplete",
        )
    ]


def _required_file(value: str, env_var: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_file():
        raise RuntimeError(f"{env_var} must reference an existing file")
    return path.resolve()


def _command_argv(value: str, env_var: str) -> tuple[str, ...]:
    try:
        argv = tuple(shlex.split(value))
    except ValueError as exc:
        raise RuntimeError(f"{env_var} is not a valid command") from exc
    if not argv:
        raise RuntimeError(f"{env_var} must not be empty")
    if shutil.which(argv[0]) is None:
        raise RuntimeError(f"{env_var} executable not found: {argv[0]}")
    return argv


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("UAS artifact manifest must be valid UTF-8 JSON") from exc
    if not isinstance(manifest, dict):
        raise RuntimeError("UAS artifact manifest must be a JSON object")
    if manifest.get("schema_version") != _MANIFEST_SCHEMA:
        raise RuntimeError(f"UAS artifact manifest schema must be {_MANIFEST_SCHEMA}")
    return manifest


def _manifest_checkpoint_path(manifest_path: Path, manifest: Mapping[str, Any]) -> Path:
    value = manifest.get("autoencoder_path")
    if not isinstance(value, str) or not value:
        raise RuntimeError("UAS manifest autoencoder_path is required")
    relative_path = Path(value)
    if relative_path.is_absolute() or relative_path.name != value:
        raise RuntimeError("UAS manifest autoencoder_path must be a relative basename")
    checkpoint_path = manifest_path.parent / relative_path
    if not checkpoint_path.is_file():
        raise RuntimeError("UAS manifest checkpoint does not exist")
    return checkpoint_path.resolve()


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RuntimeError(f"{name} must be a positive integer")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as artifact_file:
        for chunk in iter(lambda: artifact_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _load_autoencoder(
    path: Path,
    *,
    dim: int,
    latent_dim: int,
) -> MoleculeAutoencoder:
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise RuntimeError("UAS autoencoder checkpoint could not be loaded") from exc
    if not isinstance(state, Mapping) or not state:
        raise RuntimeError("UAS autoencoder checkpoint must contain a state dictionary")
    for name, value in state.items():
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            raise RuntimeError("UAS autoencoder state dictionary must contain tensors")
        if (value.is_floating_point() or value.is_complex()) and not bool(
            torch.isfinite(value).all()
        ):
            raise RuntimeError(f"UAS autoencoder tensor is not finite: {name}")
    model = MoleculeAutoencoder(input_dim=dim, latent_dim=latent_dim)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            "UAS autoencoder state dictionary does not match manifest dimensions"
        ) from exc
    model.eval()
    return model


async def _probe_runtime(runtime: UASRuntime) -> None:
    context = _probe_context()
    embeddings = await _candidate_embeddings(
        runtime,
        operation="probe",
        n_samples=1,
        seed=0,
        attempt=1,
        context=context,
    )
    with torch.no_grad():
        reconstruction, _ = runtime.model(embeddings)
    if reconstruction.shape != embeddings.shape or not bool(torch.isfinite(reconstruction).all()):
        raise RuntimeError("UAS autoencoder probe produced an invalid reconstruction")
    candidates = await _decode_embeddings(
        runtime,
        operation="probe",
        embeddings=embeddings,
        context=context,
    )
    if len(candidates) != 1:
        raise RuntimeError("UAS decoder probe must return exactly one candidate")


def _probe_context() -> dict[str, Any]:
    coordinates = [1.0, *([0.0] * 128)]
    return {
        "project_id": "uas-startup-probe",
        "request_id": "uas-startup-probe",
        "context_schema_version": GENERATOR_CONTEXT_SCHEMA,
        "cig": {
            "project_id": "uas-startup-probe",
            "objectives": [
                {
                    "id": "probe-objective",
                    "name": "probe",
                    "type": "maximize",
                    "target_value": 1.0,
                    "target_min": None,
                    "target_max": None,
                    "property": "qed",
                    "weight": 1.0,
                    "pareto_tier": 1,
                }
            ],
            "edges": [],
            "hyperedges": [],
            "constraints": {},
            "created_by": "uas-generator-svc",
        },
        "hciv": {
            "coordinates": coordinates,
            "dim": 128,
            "curvature": 1.0,
            "manifold_type": "lorentz",
            "molecule_smiles": "",
            "parent_hciv_id": None,
        },
        "intent_cone": {
            "axis": coordinates,
            "half_angle": 0.5,
            "angle_radians": 0.5,
            "curvature": 1.0,
            "property_weights": {},
            "apex": None,
            "axis_direction": None,
            "length": 1.0,
        },
        "generator_params": {},
    }


async def _candidate_embeddings(
    runtime: UASRuntime,
    *,
    operation: str,
    n_samples: int,
    seed: int | None,
    attempt: int,
    context: Mapping[str, Any],
) -> torch.Tensor:
    response = await _run_json_command(
        runtime.candidate_command,
        {
            "schema_version": _COMMAND_SCHEMA,
            "operation": operation,
            "project_id": context["project_id"],
            "request_id": context["request_id"],
            "n_samples": n_samples,
            "embedding_dim": runtime.dim,
            "seed": seed,
            "attempt": attempt,
            "hciv": context["hciv"],
            "intent_cone": context["intent_cone"],
            "cig": context["cig"],
            "generator_params": context["generator_params"],
        },
        timeout_seconds=runtime.command_timeout_seconds,
        name="UAS_CANDIDATE_SOURCE_COMMAND",
    )
    if set(response) != {"schema_version", "embeddings"}:
        raise RuntimeError("UAS candidate response has invalid fields")
    if response.get("schema_version") != _COMMAND_SCHEMA:
        raise RuntimeError(f"UAS candidate response schema must be {_COMMAND_SCHEMA}")
    rows = response.get("embeddings")
    if not isinstance(rows, list) or len(rows) != n_samples:
        raise RuntimeError("UAS candidate response must contain the requested embedding count")
    normalized = [_finite_float_row(row, runtime.dim, "UAS candidate embedding") for row in rows]
    embeddings = torch.tensor(normalized, dtype=torch.float32)
    if not bool(torch.isfinite(embeddings).all()):
        raise RuntimeError("UAS candidate embeddings must contain finite float32 values")
    return embeddings


async def _decode_embeddings(
    runtime: UASRuntime,
    *,
    operation: str,
    embeddings: torch.Tensor,
    context: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows = embeddings.detach().cpu().tolist()
    response = await _run_json_command(
        runtime.decoder_command,
        {
            "schema_version": _COMMAND_SCHEMA,
            "operation": operation,
            "project_id": context["project_id"],
            "request_id": context["request_id"],
            "embeddings": rows,
            "context": {
                "context_schema_version": context["context_schema_version"],
                "hciv": context["hciv"],
                "intent_cone": context["intent_cone"],
                "cig": context["cig"],
                "generator_params": context["generator_params"],
            },
        },
        timeout_seconds=runtime.command_timeout_seconds,
        name="UAS_DECODER_COMMAND",
    )
    if set(response) != {"schema_version", "candidates"}:
        raise RuntimeError("UAS decoder response has invalid fields")
    if response.get("schema_version") != _COMMAND_SCHEMA:
        raise RuntimeError(f"UAS decoder response schema must be {_COMMAND_SCHEMA}")
    candidates = response.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != len(rows):
        raise RuntimeError("UAS decoder response count must match the embedding count")
    normalized = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise RuntimeError("UAS decoder candidates must be JSON objects")
        if set(candidate) - _DECODER_CANDIDATE_FIELDS:
            raise RuntimeError("UAS decoder candidate has invalid fields")
        if "id" in candidate and not isinstance(candidate["id"], str):
            raise RuntimeError("UAS decoder candidate id must be a string")
        if "canonical_smiles" in candidate and not isinstance(
            candidate["canonical_smiles"],
            str,
        ):
            raise RuntimeError("UAS decoder candidate canonical_smiles must be a string")
        if "properties" in candidate and not isinstance(candidate["properties"], dict):
            raise RuntimeError("UAS decoder candidate properties must be a JSON object")
        smiles = candidate.get("smiles")
        if not isinstance(smiles, str) or not smiles.strip():
            raise RuntimeError("UAS decoder candidate requires a non-empty smiles field")
        normalized_smiles = smiles.strip()
        try:
            canonical_smiles = canonicalize(normalized_smiles)
        except ValueError as exc:
            raise RuntimeError(f"UAS decoder produced invalid SMILES: {smiles}") from exc
        normalized_candidate = dict(candidate)
        normalized_candidate["smiles"] = normalized_smiles
        normalized_candidate["canonical_smiles"] = canonical_smiles
        normalized.append(normalized_candidate)
    return normalized


def _finite_float_row(value: object, dim: int, name: str) -> list[float]:
    if not isinstance(value, list) or len(value) != dim:
        raise RuntimeError(f"{name} must contain exactly {dim} values")
    normalized = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int | float):
            raise RuntimeError(f"{name} values must be numeric")
        try:
            number = float(item)
        except OverflowError as exc:
            raise RuntimeError(f"{name} values must be finite") from exc
        if not math.isfinite(number):
            raise RuntimeError(f"{name} values must be finite")
        normalized.append(number)
    return normalized


async def _run_json_command(
    argv: Sequence[str],
    payload: Mapping[str, Any],
    *,
    timeout_seconds: float,
    name: str,
) -> dict[str, Any]:
    try:
        request_bytes = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{name} request is not valid JSON") from exc
    process = await asyncio.create_subprocess_exec(
        *argv,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    try:
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(request_bytes),
                timeout=timeout_seconds,
            )
        except TimeoutError as exc:
            raise RuntimeError(f"{name} timed out") from exc
        if process.returncode != 0:
            message = stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"{name} failed with exit code {process.returncode}: {message}")
        try:
            response = json.loads(
                stdout.decode("utf-8"),
                parse_constant=_reject_json_constant,
            )
        except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
            raise RuntimeError(f"{name} returned invalid JSON") from exc
        if not isinstance(response, dict):
            raise RuntimeError(f"{name} must return a JSON object")
        _validate_finite_json_numbers(response, name)
        return response
    finally:
        await _stop_process_uninterruptibly(process)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant: {value}")


def _validate_finite_json_numbers(value: object, name: str) -> None:
    pending = [value]
    while pending:
        current = pending.pop()
        if isinstance(current, float) and not math.isfinite(current):
            raise RuntimeError(f"{name} returned a non-finite JSON number")
        if isinstance(current, dict):
            pending.extend(current.values())
        elif isinstance(current, list):
            pending.extend(current)


async def _stop_process(process: asyncio.subprocess.Process) -> None:
    _signal_process_group(process.pid, signal.SIGTERM)
    try:
        if process.returncode is None:
            await asyncio.wait_for(process.wait(), timeout=1.0)
    except TimeoutError:
        pass
    _signal_process_group(process.pid, signal.SIGKILL)
    if process.returncode is None:
        await process.wait()


async def _stop_process_uninterruptibly(
    process: asyncio.subprocess.Process,
) -> None:
    cleanup_task = asyncio.create_task(_stop_process(process))
    cancellation: asyncio.CancelledError | None = None
    while not cleanup_task.done():
        try:
            await asyncio.shield(cleanup_task)
        except asyncio.CancelledError as exc:
            cancellation = exc
    await cleanup_task
    if cancellation is not None:
        raise cancellation


def _signal_process_group(process_group_id: int, signal_number: int) -> None:
    try:
        os.killpg(process_group_id, signal_number)
    except (PermissionError, ProcessLookupError):
        pass


async def start_server(
    runtime: UASRuntime,
    address: str,
) -> tuple[grpc.aio.Server, int]:
    server = grpc.aio.server()
    generator_pb2_grpc.add_GeneratorServiceServicer_to_server(
        UASGeneratorServicer(runtime),
        server,
    )
    port = server.add_insecure_port(address)
    if port == 0:
        raise RuntimeError(f"UAS generator service could not bind {address}")
    await server.start()
    return server, port


async def serve() -> None:
    runtime = await load_runtime()
    server, _ = await start_server(runtime, "[::]:50068")
    await server.wait_for_termination()


def main() -> None:
    asyncio.run(serve())


if __name__ == "__main__":
    main()
