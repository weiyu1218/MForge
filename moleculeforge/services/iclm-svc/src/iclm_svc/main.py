"""ICLM Service - gRPC server for Incremental Causal Language Model with EWC."""

import asyncio
import base64
import binascii
import hashlib
import hmac
import inspect
import json
import logging
import math
import os
import shlex
import signal
import struct
import sys
import tempfile
import time
from concurrent import futures
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

import grpc
from google.protobuf.message import DecodeError
from mf_chem.molecule.parsing import canonicalize
from mf_core.artifacts import (
    ArtifactRequirement,
    CommandRequirement,
    RequirementStatus,
    check_artifact,
    check_command,
    require_available,
)
from mf_core.plugins.generator import (
    GeneratorRequestError,
    GeneratorResultError,
    artifact_refs,
    build_generate_response,
    build_generator_info,
    validate_generate_request,
)
from mf_core.proto_gen.moleculeforge.v1.core import audit_pb2
from mf_core.proto_gen.moleculeforge.v1.generator import generator_pb2, generator_pb2_grpc

_REQUIREMENTS = (ArtifactRequirement("iclm_model", "ICLM_MODEL_PATH", kind="path"),)
_GENERATOR_NAME = "iclm"
_MAX_BATCH_SIZE = 64
_UPDATE_COMMAND_ENV = "ICLM_UPDATE_COMMAND"
_UPDATE_TIMEOUT_ENV = "ICLM_UPDATE_TIMEOUT_SECONDS"
_CHECKPOINT_DIRECTORY_ENV = "ICLM_CHECKPOINT_DIRECTORY"
_BUILTIN_UPDATE_MODULE = "mf_generators.incremental_clm.hf_runner"
_BUILTIN_UPDATE_LABEL = "built-in ICLM update runner"
_CONTINUAL_STATE_FILE = "moleculeforge_continual_state.pt"
_EWC_REPLAY_FILE = "moleculeforge_ewc_replay.json"
_VALIDATION_MODEL_METADATA_FILE = "moleculeforge_validation_model.json"
_VALIDATION_MODEL_SCHEMA = "iclm-validation-model.v1"
_ALLOW_VALIDATION_MODEL_ENV = "ICLM_ALLOW_VALIDATION_MODEL"
_STATE_PATH_ENV = "ICLM_STATE_PATH"
_INTERNAL_SERVICE_TOKEN_ENV = "INTERNAL_SERVICE_TOKEN"
_SERVICE_TOKEN_METADATA_KEY = "x-moleculeforge-service-token"
_UPDATE_COMMAND_REQUIREMENT = CommandRequirement(
    "iclm_update_command",
    _UPDATE_COMMAND_ENV,
)
_TRAINING_BATCH_SCHEMA = "training-batch.v1"
_STATE_SCHEMA = "iclm-state.v3"
_LEGACY_STATE_SCHEMA = "iclm-state.v2"
_UPDATE_FINGERPRINT_SCHEMA = "iclm-update+base.v1"
_LEGACY_UPDATE_FINGERPRINT_SCHEMA = "iclm-update.v1"
_LOGGER = logging.getLogger(__name__)


class ModelUpdateRequestError(ValueError):
    pass


class UpdateRunnerUnavailable(RuntimeError):
    pass


@dataclass
class _UpdateRecord:
    fingerprint: str
    fingerprint_schema: str
    state: str
    base_checkpoint_path: str
    base_version: str
    base_checkpoint_checksum: str
    response: bytes | None = None


def _require_runtime(
    *,
    include_update_command: bool = False,
    model_path: str | None = None,
) -> list[RequirementStatus]:
    statuses = _runtime_statuses(
        include_update_command=include_update_command,
        model_path=model_path,
    )
    require_available(statuses)
    return statuses


def runtime_status() -> list[dict]:
    return [status.to_dict() for status in _runtime_statuses(include_update_command=True)]


def _runtime_statuses(
    *,
    include_update_command: bool,
    model_path: str | None = None,
) -> list[RequirementStatus]:
    requirements = (
        (
            ArtifactRequirement(
                "iclm_model",
                "ICLM_MODEL_PATH",
                kind="path",
                path=model_path,
            ),
        )
        if model_path
        else _REQUIREMENTS
    )
    statuses = [check_artifact(requirement) for requirement in requirements]
    resolved_model_path = model_path or os.environ.get("ICLM_MODEL_PATH", "").strip()
    validation_model_status = _validation_model_opt_in_status(resolved_model_path)
    if validation_model_status is not None:
        statuses.append(validation_model_status)
    if include_update_command:
        if os.environ.get(_UPDATE_COMMAND_ENV, "").strip():
            statuses.append(check_command(_UPDATE_COMMAND_REQUIREMENT))
        else:
            statuses.append(_ewc_baseline_status(resolved_model_path))
    return statuses


def _ewc_baseline_status(model_path: str) -> RequirementStatus:
    if not model_path:
        return RequirementStatus(
            name="iclm_ewc_baseline",
            configured=False,
            available=False,
            required=True,
            path=None,
            source="ICLM_MODEL_PATH",
            message="ICLM_MODEL_PATH is required for iclm_ewc_baseline",
        )
    checkpoint_path = Path(model_path).expanduser()
    continual_state_path = checkpoint_path / _CONTINUAL_STATE_FILE
    if continual_state_path.is_file():
        try:
            from mf_generators.incremental_clm.hf_runner import (
                validate_continual_checkpoint,
            )

            validate_continual_checkpoint(checkpoint_path)
        except Exception as exc:
            return RequirementStatus(
                name="iclm_ewc_baseline",
                configured=True,
                available=False,
                required=True,
                path=str(continual_state_path),
                source="manifest",
                message=str(exc),
            )
        return RequirementStatus(
            name="iclm_ewc_baseline",
            configured=True,
            available=True,
            required=True,
            path=str(continual_state_path),
            source="manifest",
            message="ICLM continual-learning state is available",
        )
    replay_path = checkpoint_path / _EWC_REPLAY_FILE
    replay_status = check_artifact(
        ArtifactRequirement(
            "iclm_ewc_baseline",
            "ICLM_MODEL_PATH",
            kind="file",
            path=str(replay_path),
        )
    )
    if not replay_status.available:
        return replay_status
    try:
        from mf_generators.incremental_clm.hf_runner import validate_ewc_replay

        validate_ewc_replay(replay_path)
    except Exception as exc:
        return RequirementStatus(
            name="iclm_ewc_baseline",
            configured=True,
            available=False,
            required=True,
            path=str(replay_path),
            source="manifest",
            message=str(exc),
        )
    return replay_status


def _validation_model_opt_in_status(
    model_path: str,
) -> RequirementStatus | None:
    if not model_path:
        return None
    metadata_path = Path(model_path).expanduser() / _VALIDATION_MODEL_METADATA_FILE
    if not metadata_path.exists():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        message = f"ICLM validation model metadata is invalid: {exc}"
        valid_metadata = False
    else:
        valid_metadata = (
            isinstance(metadata, dict)
            and metadata.get("schema_version") == _VALIDATION_MODEL_SCHEMA
            and metadata.get("purpose") == "synthetic_pipeline_validation_only"
            and metadata.get("seed") == 7
        )
        message = (
            "ICLM validation model metadata is valid"
            if valid_metadata
            else "ICLM validation model metadata is invalid"
        )
    opted_in = os.environ.get(_ALLOW_VALIDATION_MODEL_ENV, "").strip() == "true"
    available = valid_metadata and opted_in
    if valid_metadata and not opted_in:
        message = f"{_ALLOW_VALIDATION_MODEL_ENV}=true is required"
    return RequirementStatus(
        name="iclm_validation_model_opt_in",
        configured=opted_in,
        available=available,
        required=True,
        path=str(metadata_path),
        source=_ALLOW_VALIDATION_MODEL_ENV,
        message=message,
    )


async def _abort_unavailable(
    context,
    message: str | None = None,
    *,
    include_update_command: bool = False,
):
    statuses = _runtime_statuses(include_update_command=include_update_command)
    if message is None:
        try:
            require_available(statuses)
        except RuntimeError as exc:
            message = str(exc)
        else:
            message = "ICLM runner is not configured"
    if context is not None and hasattr(context, "abort"):
        await context.abort(grpc.StatusCode.FAILED_PRECONDITION, message)
    raise RuntimeError(message)


async def _abort_invalid_argument(context, message: str):
    if context is not None and hasattr(context, "abort"):
        await context.abort(grpc.StatusCode.INVALID_ARGUMENT, message)
    raise ValueError(message)


async def _abort_internal(context, message: str):
    if context is not None and hasattr(context, "abort"):
        await context.abort(grpc.StatusCode.INTERNAL, message)
    raise RuntimeError(message)


def _build_generator(model_path: str | None = None):
    from mf_generators.incremental_clm.generator import IncrementalCLMGenerator
    from mf_generators.incremental_clm.hf_runner import HuggingFaceCausalLMRunner

    model_path = str(
        Path(model_path or os.environ["ICLM_MODEL_PATH"]).expanduser().resolve()
    )
    device = os.environ.get("ICLM_DEVICE", "cpu")
    runner = HuggingFaceCausalLMRunner(model_path=model_path, device=device)
    return IncrementalCLMGenerator(
        checkpoint_path=model_path,
        device=device,
        runner=runner,
    )


def _record_update_failure(
    record: _UpdateRecord,
) -> None:
    record.state = "retryable"
    record.response = None


class ICLMServicer(
    generator_pb2_grpc.GeneratorServiceServicer,
    generator_pb2_grpc.IncrementalGeneratorServiceServicer,
):
    def __init__(self, generator=None, generator_factory=None, state_path=None):
        self.generator = generator if generator is not None else _build_generator()
        self._generator_factory = generator_factory or _build_generator
        configured_checkpoint_path = str(
            getattr(self.generator, "checkpoint_path", "")
            or os.environ.get("ICLM_MODEL_PATH", "")
        ).strip()
        self._active_checkpoint_path = (
            str(Path(configured_checkpoint_path).expanduser().resolve())
            if configured_checkpoint_path
            else ""
        )
        self._active_version = str(getattr(self.generator, "version", "")).strip() or "0.1.0"
        self._activation_lock = asyncio.Lock()
        self._initialization_lock = asyncio.Lock()
        self._update_lock = asyncio.Lock()
        self._update_records: dict[str, _UpdateRecord] = {}
        configured_state_path = (
            str(state_path).strip()
            if state_path is not None
            else os.environ.get(_STATE_PATH_ENV, "").strip()
        )
        self._state_path = Path(configured_state_path) if configured_state_path else None
        self._active_checkpoint_checksum: str | None = None
        self._recovery_checkpoint_path: str | None = None
        self._recovery_checkpoint_checksum: str | None = None
        self._bootstrap_state_pending = (
            self._state_path is not None and not self._state_path.exists()
        )
        self._initialized = True
        if self._state_path is not None and self._state_path.exists():
            (
                self._active_checkpoint_path,
                self._active_version,
                self._active_checkpoint_checksum,
                self._update_records,
            ) = _load_service_state(self._state_path)
            self._recovery_checkpoint_path = self._active_checkpoint_path
            self._recovery_checkpoint_checksum = self._active_checkpoint_checksum
            self._initialized = False

    async def initialize(self) -> None:
        async with self._initialization_lock:
            if self._initialized and not self._bootstrap_state_pending:
                return
            checkpoint_path = self._recovery_checkpoint_path
            if checkpoint_path is None:
                checkpoint_path = self._active_checkpoint_path
                if not checkpoint_path:
                    raise RuntimeError("ICLM active checkpoint_path is required")
                _require_runtime(model_path=checkpoint_path)
                await _eager_validate_generator(self.generator)
                checkpoint_checksum = await asyncio.to_thread(
                    _checkpoint_artifact_sha256,
                    checkpoint_path,
                )
                self._persist_state(
                    checkpoint_path=checkpoint_path,
                    checkpoint_checksum=checkpoint_checksum,
                    active_version=self._active_version,
                    records=self._update_records,
                )
                self._active_checkpoint_checksum = checkpoint_checksum
                self._bootstrap_state_pending = False
                self._initialized = True
                return
            expected_checksum = self._recovery_checkpoint_checksum
            if expected_checksum is None:
                raise RuntimeError("ICLM recovery checkpoint checksum is required")
            actual_checksum = await asyncio.to_thread(
                _checkpoint_artifact_sha256,
                checkpoint_path,
            )
            if not hmac.compare_digest(actual_checksum, expected_checksum):
                raise RuntimeError(
                    "ICLM active checkpoint checksum does not match persisted state"
                )
            _require_runtime(model_path=checkpoint_path)
            recovered_generator = await self._construct_generator(
                checkpoint_path,
                eager_validate=True,
            )
            async with self._activation_lock:
                self.generator = recovered_generator
                self._recovery_checkpoint_path = None
                self._recovery_checkpoint_checksum = None
                self._bootstrap_state_pending = False
                self._initialized = True

    async def Generate(self, request, context):
        """Generate molecular structures via incremental causal LM."""
        await self.initialize()
        generator, checkpoint_path, _ = await self._active_snapshot()
        try:
            statuses = _require_runtime(model_path=checkpoint_path)
        except RuntimeError as exc:
            return await _abort_unavailable(context, str(exc))
        if generator is None:
            return await _abort_unavailable(context)
        try:
            request_context = validate_generate_request(
                request,
                max_batch_size=_MAX_BATCH_SIZE,
            )
        except GeneratorRequestError as exc:
            return await _abort_invalid_argument(context, str(exc))
        params = dict(getattr(request, "generator_params", {}) or {})
        start = time.perf_counter()
        try:
            molecules = await generator.generate(
                batch_size=request_context.batch_size,
                intent_cone=request_context.intent_cone,
                **params,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            return await _abort_internal(context, str(exc))
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        try:
            return build_generate_response(
                generator_name=_GENERATOR_NAME,
                request=request,
                molecules=molecules,
                statuses=statuses,
                elapsed_ms=elapsed_ms,
                canonicalize_smiles=canonicalize,
            )
        except GeneratorResultError as exc:
            return await _abort_internal(context, str(exc))

    async def GenerateStream(self, request_iterator, context):
        async for request in request_iterator:
            yield await self.Generate(request, context)

    async def BatchGenerate(self, request_iterator, context):
        async for request in request_iterator:
            yield await self.Generate(request, context)

    async def Info(self, request, context):
        await self.initialize()
        generator, checkpoint_path, active_version = await self._active_snapshot()
        include_update_runtime = bool(
            os.environ.get(_UPDATE_COMMAND_ENV, "").strip()
        ) or _is_builtin_huggingface_runner(getattr(generator, "runner", None))
        response = await build_generator_info(
            generator_name=_GENERATOR_NAME,
            generator=generator,
            statuses=_runtime_statuses(
                include_update_command=include_update_runtime,
                model_path=checkpoint_path,
            ),
            fallback={
                "version": "0.1.0",
                "description": "Incremental causal language model generator",
                "supported_properties": ["qed", "sa_score", "mw", "logp"],
                "max_batch_size": _MAX_BATCH_SIZE,
                "supports_streaming": False,
                "requires_gpu": True,
            },
        )
        response.version = active_version
        student_embedding_dimension = await _student_embedding_dimension(generator)
        if student_embedding_dimension is not None:
            response.default_params["student_embedding_dim"] = str(
                student_embedding_dimension
            )
        return response

    async def UpdateModel(self, request, context):
        """Update ICLM via Elastic Weight Consolidation (EWC) with new data."""
        await _authenticate_model_update(context)
        await self.initialize()
        try:
            update_payload = _validate_model_update_request(request)
        except ModelUpdateRequestError as exc:
            return await _abort_invalid_argument(context, str(exc))
        async with self._update_lock:
            (
                active_generator,
                active_checkpoint_path,
                active_version,
            ) = await self._active_snapshot()
            record = self._update_records.get(request.request_id)
            if request.dim > 0 and (
                record is None or record.state != "succeeded"
            ):
                student_embedding_dimension = await _student_embedding_dimension(
                    active_generator
                )
                if (
                    student_embedding_dimension is not None
                    and request.dim != student_embedding_dimension
                ):
                    return await _abort_invalid_argument(
                        context,
                        f"teacher embedding dimension {request.dim} does not match "
                        "student embedding dimension "
                        f"{student_embedding_dimension}",
                    )
            actual_active_checksum = await asyncio.to_thread(
                _checkpoint_artifact_sha256,
                active_checkpoint_path,
            )
            if self._active_checkpoint_checksum is not None and not hmac.compare_digest(
                actual_active_checksum,
                self._active_checkpoint_checksum,
            ):
                return await _abort_unavailable(
                    context,
                    "ICLM active checkpoint checksum no longer matches persisted state",
                )
            retrying_bound_request = record is not None
            if record is not None:
                fingerprint_payload = update_payload
                if record.fingerprint_schema == _LEGACY_UPDATE_FINGERPRINT_SCHEMA:
                    try:
                        fingerprint_payload = _legacy_model_update_payload(request)
                    except ModelUpdateRequestError as exc:
                        return await _abort_invalid_argument(context, str(exc))
                request_fingerprint = _model_update_fingerprint(
                    fingerprint_payload,
                    fingerprint_schema=record.fingerprint_schema,
                    base_checkpoint_path=record.base_checkpoint_path,
                    base_version=record.base_version,
                    base_checkpoint_checksum=record.base_checkpoint_checksum,
                )
                if record.fingerprint != request_fingerprint:
                    return await _abort_invalid_argument(
                        context,
                        "request_id is already bound to a different model update request",
                    )
                if record.state == "succeeded" and record.response is not None:
                    return generator_pb2.ModelUpdateResponse.FromString(record.response)
            else:
                retryable_request_ids = [
                    request_id
                    for request_id, update_record in self._update_records.items()
                    if update_record.state == "retryable"
                ]
                if retryable_request_ids:
                    return await _abort_unavailable(
                        context,
                        "retryable update must complete before a new ICLM update",
                    )
                base_checkpoint_path = str(Path(active_checkpoint_path).resolve())
                base_checkpoint_checksum = actual_active_checksum
                request_fingerprint = _model_update_fingerprint(
                    update_payload,
                    fingerprint_schema=_UPDATE_FINGERPRINT_SCHEMA,
                    base_checkpoint_path=base_checkpoint_path,
                    base_version=active_version,
                    base_checkpoint_checksum=base_checkpoint_checksum,
                )
                record = _UpdateRecord(
                    fingerprint=request_fingerprint,
                    fingerprint_schema=_UPDATE_FINGERPRINT_SCHEMA,
                    state="in_progress",
                    base_checkpoint_path=base_checkpoint_path,
                    base_version=active_version,
                    base_checkpoint_checksum=base_checkpoint_checksum,
                )
                self._update_records[request.request_id] = record
            if record.state != "succeeded":
                if (
                    Path(active_checkpoint_path).resolve()
                    != Path(record.base_checkpoint_path).resolve()
                    or active_version != record.base_version
                ):
                    return await _abort_unavailable(
                        context,
                        "ICLM update base checkpoint is no longer active",
                    )
                if retrying_bound_request and not hmac.compare_digest(
                    actual_active_checksum,
                    record.base_checkpoint_checksum,
                ):
                    return await _abort_unavailable(
                        context,
                        "ICLM update base checkpoint is no longer active",
                    )
            record.state = "in_progress"
            record.response = None
            record.state = "retryable"
            try:
                self._persist_state(
                    checkpoint_path=active_checkpoint_path,
                    active_version=active_version,
                    records=self._update_records,
                )
            except Exception as exc:
                return await _abort_internal(context, str(exc))
            if not any(
                _sample_learning_strength(sample) > 0.0
                for sample in update_payload["samples"]
            ):
                response = _skipped_model_update_response(
                    request,
                    update_payload,
                    checkpoint_path=active_checkpoint_path,
                    active_version=active_version,
                )
                successful_record = _UpdateRecord(
                    fingerprint=request_fingerprint,
                    fingerprint_schema=record.fingerprint_schema,
                    state="succeeded",
                    base_checkpoint_path=record.base_checkpoint_path,
                    base_version=record.base_version,
                    base_checkpoint_checksum=record.base_checkpoint_checksum,
                    response=response.SerializeToString(deterministic=True),
                )
                try:
                    async with self._activation_lock:
                        persisted_records = {
                            **self._update_records,
                            request.request_id: successful_record,
                        }
                        self._persist_state(
                            checkpoint_path=active_checkpoint_path,
                            active_version=active_version,
                            records=persisted_records,
                        )
                        self._update_records[request.request_id] = successful_record
                except Exception as exc:
                    return await _abort_internal(context, str(exc))
                return response
            record.state = "in_progress"
            try:
                if os.environ.get(_UPDATE_COMMAND_ENV, "").strip():
                    update_generator = active_generator
                    include_update_runtime = True
                else:
                    update_generator = await self._construct_generator(active_checkpoint_path)
                    include_update_runtime = _is_builtin_huggingface_runner(
                        getattr(update_generator, "runner", None)
                    )
                _require_runtime(
                    include_update_command=include_update_runtime,
                    model_path=active_checkpoint_path,
                )
            except RuntimeError as exc:
                try:
                    self._persist_update_failure(
                        record,
                        checkpoint_path=active_checkpoint_path,
                        active_version=active_version,
                    )
                except Exception as state_exc:
                    return await _abort_internal(context, str(state_exc))
                return await _abort_unavailable(context, str(exc))
            try:
                result = await _run_update(update_payload, update_generator)
                checkpoint_path = _new_checkpoint_path(
                    result,
                    active_checkpoint_path=active_checkpoint_path,
                )
                response = _update_model_response(request, update_payload, result)
                activated_generator = await self._construct_generator(
                    checkpoint_path,
                    eager_validate=True,
                )
                checkpoint_checksum = (
                    await asyncio.to_thread(
                        _checkpoint_artifact_sha256,
                        checkpoint_path,
                    )
                    if self._state_path is not None
                    else None
                )
                serialized_response = response.SerializeToString(deterministic=True)
                successful_record = _UpdateRecord(
                    fingerprint=request_fingerprint,
                    fingerprint_schema=record.fingerprint_schema,
                    state="succeeded",
                    base_checkpoint_path=record.base_checkpoint_path,
                    base_version=record.base_version,
                    base_checkpoint_checksum=record.base_checkpoint_checksum,
                    response=serialized_response,
                )
                async with self._activation_lock:
                    persisted_records = {
                        **self._update_records,
                        request.request_id: successful_record,
                    }
                    self._persist_state(
                        checkpoint_path=checkpoint_path,
                        checkpoint_checksum=checkpoint_checksum,
                        active_version=response.active_version,
                        records=persisted_records,
                    )
                    self.generator = activated_generator
                    self._active_checkpoint_path = checkpoint_path
                    self._active_checkpoint_checksum = checkpoint_checksum
                    self._active_version = response.active_version
                    self._update_records[request.request_id] = successful_record
                return response
            except asyncio.CancelledError:
                try:
                    self._persist_update_failure(
                        record,
                        checkpoint_path=active_checkpoint_path,
                        active_version=active_version,
                    )
                except Exception:
                    _LOGGER.exception(
                        "ICLM retryable update state could not be persisted during cancellation"
                    )
                raise
            except UpdateRunnerUnavailable as exc:
                try:
                    self._persist_update_failure(
                        record,
                        checkpoint_path=active_checkpoint_path,
                        active_version=active_version,
                    )
                except Exception as state_exc:
                    return await _abort_internal(context, str(state_exc))
                return await _abort_unavailable(context, str(exc))
            except Exception as exc:
                try:
                    self._persist_update_failure(
                        record,
                        checkpoint_path=active_checkpoint_path,
                        active_version=active_version,
                    )
                except Exception as state_exc:
                    return await _abort_internal(context, str(state_exc))
                return await _abort_internal(context, str(exc))

    def _persist_update_failure(
        self,
        record: _UpdateRecord,
        *,
        checkpoint_path: str,
        active_version: str,
    ) -> None:
        _record_update_failure(record)
        self._persist_state(
            checkpoint_path=checkpoint_path,
            active_version=active_version,
            records=self._update_records,
        )

    def _persist_state(
        self,
        *,
        checkpoint_path: str,
        checkpoint_checksum: str | None = None,
        active_version: str,
        records: dict[str, _UpdateRecord],
    ) -> None:
        if self._state_path is None:
            return
        if checkpoint_checksum is None:
            if Path(checkpoint_path).resolve() != Path(
                self._active_checkpoint_path
            ).resolve():
                raise RuntimeError(
                    "ICLM state checkpoint checksum is required for a new checkpoint"
                )
            checkpoint_checksum = self._active_checkpoint_checksum
        if checkpoint_checksum is None:
            raise RuntimeError("ICLM active checkpoint checksum is required")
        _write_service_state(
            self._state_path,
            checkpoint_path=checkpoint_path,
            checkpoint_checksum=checkpoint_checksum,
            active_version=active_version,
            records=records,
        )

    async def _active_snapshot(self) -> tuple[object, str, str]:
        async with self._activation_lock:
            return (
                self.generator,
                self._active_checkpoint_path,
                self._active_version,
            )

    async def _construct_generator(
        self,
        checkpoint_path: str,
        *,
        eager_validate: bool = False,
    ):
        if not checkpoint_path:
            raise RuntimeError("ICLM active checkpoint_path is required")
        generator = await _call_maybe_async(
            self._generator_factory,
            checkpoint_path,
        )
        if generator is None:
            raise RuntimeError("ICLM generator factory returned no runner")
        runner_checkpoint = str(getattr(generator, "checkpoint_path", "")).strip()
        if not runner_checkpoint:
            raise RuntimeError("ICLM generator runner requires checkpoint_path")
        if Path(runner_checkpoint).resolve() != Path(checkpoint_path).resolve():
            raise RuntimeError(
                "ICLM generator runner checkpoint_path does not match activated checkpoint"
            )
        if eager_validate:
            await _eager_validate_generator(generator)
        return generator


def _load_service_state(
    state_path: Path,
) -> tuple[str, str, str, dict[str, _UpdateRecord]]:
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"ICLM state cannot be loaded from {state_path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") not in {
        _STATE_SCHEMA,
        _LEGACY_STATE_SCHEMA,
    }:
        raise RuntimeError(
            f"ICLM state schema_version must be {_STATE_SCHEMA} or {_LEGACY_STATE_SCHEMA}"
        )
    state_schema = payload["schema_version"]
    checkpoint_path = str(
        Path(
            _required_state_string(
                payload.get("active_checkpoint_path"),
                "active_checkpoint_path",
            )
        ).expanduser().resolve()
    )
    active_version = _required_state_string(
        payload.get("active_version"),
        "active_version",
    )
    active_checkpoint_checksum = _required_checkpoint_checksum(
        payload.get("active_checkpoint_checksum"),
        "active_checkpoint_checksum",
    )
    raw_records = payload.get("successful_updates")
    if not isinstance(raw_records, dict):
        raise RuntimeError("ICLM state successful_updates must be a JSON object")
    records: dict[str, _UpdateRecord] = {}
    for request_id, raw_record in raw_records.items():
        if not isinstance(request_id, str) or not request_id.strip():
            raise RuntimeError("ICLM state request_id must be a non-empty string")
        if not isinstance(raw_record, dict):
            raise RuntimeError("ICLM state update record must be a JSON object")
        fingerprint = _required_state_string(
            raw_record.get("fingerprint"),
            "fingerprint",
        )
        if len(fingerprint) != 64 or any(
            character not in "0123456789abcdef" for character in fingerprint
        ):
            raise RuntimeError("ICLM state fingerprint must be a sha256 hex digest")
        encoded_response = _required_state_string(
            raw_record.get("response"),
            "response",
        )
        try:
            response_bytes = base64.b64decode(encoded_response, validate=True)
            response = generator_pb2.ModelUpdateResponse.FromString(response_bytes)
        except (binascii.Error, DecodeError, ValueError) as exc:
            raise RuntimeError("ICLM state response is invalid") from exc
        if not response.acknowledged:
            raise RuntimeError("ICLM state response must acknowledge a successful update")
        base_checkpoint_path, base_version, base_checkpoint_checksum = (
            _loaded_update_base_identity(
                raw_record,
                state_schema=state_schema,
                active_checkpoint_path=checkpoint_path,
                active_version=active_version,
                active_checkpoint_checksum=active_checkpoint_checksum,
            )
        )
        records[request_id] = _UpdateRecord(
            fingerprint=fingerprint,
            fingerprint_schema=(
                _LEGACY_UPDATE_FINGERPRINT_SCHEMA
                if state_schema == _LEGACY_STATE_SCHEMA
                else _required_update_fingerprint_schema(
                    raw_record,
                    allow_legacy=True,
                )
            ),
            state="succeeded",
            base_checkpoint_path=base_checkpoint_path,
            base_version=base_version,
            base_checkpoint_checksum=base_checkpoint_checksum,
            response=response_bytes,
        )
    raw_retryable_records = payload.get("retryable_updates", {})
    if not isinstance(raw_retryable_records, dict):
        raise RuntimeError("ICLM state retryable_updates must be a JSON object")
    for request_id, raw_record in raw_retryable_records.items():
        if not isinstance(request_id, str) or not request_id.strip():
            raise RuntimeError("ICLM state request_id must be a non-empty string")
        if request_id in records:
            raise RuntimeError("ICLM state request_id must be unique")
        if not isinstance(raw_record, dict):
            raise RuntimeError("ICLM state update record must be a JSON object")
        if state_schema == _LEGACY_STATE_SCHEMA:
            raise RuntimeError(
                "ICLM legacy retryable update has no base checkpoint identity"
            )
        fingerprint = _required_state_string(
            raw_record.get("fingerprint"),
            "fingerprint",
        )
        if len(fingerprint) != 64 or any(
            character not in "0123456789abcdef" for character in fingerprint
        ):
            raise RuntimeError("ICLM state fingerprint must be a sha256 hex digest")
        base_checkpoint_path, base_version, base_checkpoint_checksum = (
            _loaded_update_base_identity(
                raw_record,
                state_schema=state_schema,
                active_checkpoint_path=checkpoint_path,
                active_version=active_version,
                active_checkpoint_checksum=active_checkpoint_checksum,
            )
        )
        records[request_id] = _UpdateRecord(
            fingerprint=fingerprint,
            fingerprint_schema=_required_update_fingerprint_schema(raw_record),
            state="retryable",
            base_checkpoint_path=base_checkpoint_path,
            base_version=base_version,
            base_checkpoint_checksum=base_checkpoint_checksum,
        )
    if sum(record.state == "retryable" for record in records.values()) > 1:
        raise RuntimeError("ICLM state must contain at most one retryable update")
    return checkpoint_path, active_version, active_checkpoint_checksum, records


def _loaded_update_base_identity(
    raw_record: dict[str, object],
    *,
    state_schema: str,
    active_checkpoint_path: str,
    active_version: str,
    active_checkpoint_checksum: str,
) -> tuple[str, str, str]:
    if state_schema == _LEGACY_STATE_SCHEMA:
        return (
            str(Path(active_checkpoint_path).resolve()),
            active_version,
            active_checkpoint_checksum,
        )
    return (
        str(
            Path(
                _required_state_string(
                    raw_record.get("base_checkpoint_path"),
                    "base_checkpoint_path",
                )
            ).resolve()
        ),
        _required_state_string(raw_record.get("base_version"), "base_version"),
        _required_checkpoint_checksum(
            raw_record.get("base_checkpoint_checksum"),
            "base_checkpoint_checksum",
        ),
    )


def _required_update_fingerprint_schema(
    raw_record: dict[str, object],
    *,
    allow_legacy: bool = False,
) -> str:
    schema = _required_state_string(
        raw_record.get("fingerprint_schema"),
        "fingerprint_schema",
    )
    allowed_schemas = {_UPDATE_FINGERPRINT_SCHEMA}
    if allow_legacy:
        allowed_schemas.add(_LEGACY_UPDATE_FINGERPRINT_SCHEMA)
    if schema not in allowed_schemas:
        expected = " or ".join(sorted(allowed_schemas))
        raise RuntimeError(
            f"ICLM state fingerprint_schema must be {expected}"
        )
    return schema


def _write_service_state(
    state_path: Path,
    *,
    checkpoint_path: str,
    checkpoint_checksum: str,
    active_version: str,
    records: dict[str, _UpdateRecord],
) -> None:
    active_checkpoint_checksum = _required_checkpoint_checksum(
        checkpoint_checksum,
        "active_checkpoint_checksum",
    )
    successful_updates = {}
    retryable_updates = {}
    for request_id, record in sorted(records.items()):
        allowed_fingerprint_schemas = {_UPDATE_FINGERPRINT_SCHEMA}
        if record.state == "succeeded":
            allowed_fingerprint_schemas.add(_LEGACY_UPDATE_FINGERPRINT_SCHEMA)
        if record.fingerprint_schema not in allowed_fingerprint_schemas:
            raise RuntimeError("ICLM update record fingerprint_schema is invalid")
        base_identity = {
            "fingerprint_schema": record.fingerprint_schema,
            "base_checkpoint_path": str(Path(record.base_checkpoint_path).resolve()),
            "base_version": _required_state_string(
                record.base_version,
                "base_version",
            ),
            "base_checkpoint_checksum": _required_checkpoint_checksum(
                record.base_checkpoint_checksum,
                "base_checkpoint_checksum",
            ),
        }
        if record.state == "retryable":
            if record.response is not None:
                raise RuntimeError("retryable ICLM update record cannot contain a response")
            retryable_updates[request_id] = {
                "fingerprint": record.fingerprint,
                **base_identity,
            }
            continue
        if record.state != "succeeded":
            continue
        if record.response is None:
            raise RuntimeError("successful ICLM update record requires a response")
        successful_updates[request_id] = {
            "fingerprint": record.fingerprint,
            **base_identity,
            "response": base64.b64encode(record.response).decode("ascii"),
        }
    payload = {
        "schema_version": _STATE_SCHEMA,
        "active_checkpoint_path": str(Path(checkpoint_path).expanduser().resolve()),
        "active_checkpoint_checksum": active_checkpoint_checksum,
        "active_version": active_version,
        "retryable_updates": retryable_updates,
        "successful_updates": successful_updates,
    }
    serialized = (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    )
    state_path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_path = tempfile.mkstemp(
        prefix=f".{state_path.name}.",
        suffix=".tmp",
        dir=state_path.parent,
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, state_path)
        directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        directory_descriptor = os.open(state_path.parent, directory_flags)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        Path(temporary_path).unlink(missing_ok=True)


def _required_state_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"ICLM state {field} must be a non-empty string")
    return value.strip()


def _required_checkpoint_checksum(value: object, field: str) -> str:
    checksum = _required_state_string(value, field)
    prefix = "sha256:"
    digest = checksum.removeprefix(prefix)
    if (
        not checksum.startswith(prefix)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise RuntimeError(f"ICLM state {field} must be a sha256 checksum")
    return checksum


def _checkpoint_artifact_sha256(checkpoint_path: str) -> str:
    path = Path(checkpoint_path).expanduser()
    digest = hashlib.sha256()
    try:
        if path.is_file():
            _update_checkpoint_digest(digest, path)
        elif path.is_dir():
            for child in sorted(item for item in path.rglob("*") if item.is_file()):
                relative_path = child.relative_to(path).as_posix().encode("utf-8")
                digest.update(struct.pack("<Q", len(relative_path)))
                digest.update(relative_path)
                _update_checkpoint_digest(digest, child)
        else:
            raise RuntimeError(
                f"ICLM active checkpoint cannot be checksummed: {checkpoint_path}"
            )
    except OSError as exc:
        raise RuntimeError(
            f"ICLM active checkpoint cannot be checksummed: {checkpoint_path}: {exc}"
        ) from exc
    return f"sha256:{digest.hexdigest()}"


def _update_checkpoint_digest(digest, path: Path) -> None:
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)


def _required_state_path() -> str:
    state_path = os.environ.get(_STATE_PATH_ENV, "").strip()
    if not state_path:
        raise RuntimeError(f"{_STATE_PATH_ENV} is required")
    return state_path


def _required_internal_service_token() -> str:
    service_token = os.environ.get(_INTERNAL_SERVICE_TOKEN_ENV, "").strip()
    if not service_token:
        raise RuntimeError(f"{_INTERNAL_SERVICE_TOKEN_ENV} is required")
    return service_token


async def _authenticate_model_update(context) -> None:
    if context is None:
        return
    expected_token = _required_internal_service_token()
    metadata: dict[str, str] = {}
    for item in context.invocation_metadata():
        if isinstance(item, tuple):
            key, value = item
        else:
            key, value = item.key, item.value
        metadata[str(key).lower()] = str(value)
    supplied_token = metadata.get(_SERVICE_TOKEN_METADATA_KEY, "")
    if supplied_token and hmac.compare_digest(
        supplied_token.encode("utf-8"),
        expected_token.encode("utf-8"),
    ):
        return
    if hasattr(context, "abort"):
        await context.abort(
            grpc.StatusCode.UNAUTHENTICATED,
            "Invalid internal service token",
        )
    raise PermissionError("Invalid internal service token")


def _update_model_response(
    request: generator_pb2.ModelUpdateRequest,
    update_payload: dict[str, object],
    payload: dict[str, object],
) -> generator_pb2.ModelUpdateResponse:
    checkpoint_path = str(payload.get("checkpoint_path", "")).strip()
    if not checkpoint_path:
        raise RuntimeError("ICLM update response requires checkpoint_path")
    checkpoint_status = check_artifact(
        ArtifactRequirement(
            "iclm_checkpoint",
            "ICLM_MODEL_PATH",
            kind="path",
            path=checkpoint_path,
        )
    )
    require_available([checkpoint_status])
    updated_samples = _strict_positive_int(
        payload.get("updated_samples"),
        "updated_samples",
    )
    if updated_samples != request.rows:
        raise RuntimeError(
            f"ICLM update processed {updated_samples} samples, expected {request.rows}"
        )
    checkpoint_ref = artifact_refs([checkpoint_status])[0]
    checkpoint_ref.version = request.target_checkpoint_version
    teacher_ref = audit_pb2.ArtifactRef(
        name=request.teacher_source,
        version=request.teacher_version,
        checksum=_teacher_supervision_checksum(request, update_payload),
        required=True,
    )
    return generator_pb2.ModelUpdateResponse(
        acknowledged=True,
        active_version=request.target_checkpoint_version,
        artifacts=[checkpoint_ref, teacher_ref],
        updated_samples=updated_samples,
        status=generator_pb2.MODEL_UPDATE_STATUS_APPLIED,
    )


def _skipped_model_update_response(
    request: generator_pb2.ModelUpdateRequest,
    update_payload: dict[str, object],
    *,
    checkpoint_path: str,
    active_version: str,
) -> generator_pb2.ModelUpdateResponse:
    checkpoint_status = check_artifact(
        ArtifactRequirement(
            "iclm_checkpoint",
            "ICLM_MODEL_PATH",
            kind="path",
            path=checkpoint_path,
        )
    )
    require_available([checkpoint_status])
    checkpoint_ref = artifact_refs([checkpoint_status])[0]
    checkpoint_ref.version = active_version
    teacher_ref = audit_pb2.ArtifactRef(
        name=request.teacher_source,
        version=request.teacher_version,
        checksum=_teacher_supervision_checksum(request, update_payload),
        required=True,
    )
    return generator_pb2.ModelUpdateResponse(
        acknowledged=True,
        active_version=active_version,
        artifacts=[checkpoint_ref, teacher_ref],
        updated_samples=0,
        status=generator_pb2.MODEL_UPDATE_STATUS_SKIPPED,
    )


def _teacher_supervision_checksum(
    request: generator_pb2.ModelUpdateRequest,
    update_payload: dict[str, object],
) -> str:
    samples = update_payload.get("samples")
    if not isinstance(samples, list) or not samples:
        raise RuntimeError("ICLM teacher supervision requires training samples")
    canonical_samples: list[dict[str, object]] = []
    for sample in samples:
        if not isinstance(sample, dict):
            raise RuntimeError("ICLM teacher supervision samples must be objects")
        canonical_sample = {
            "candidate_id": sample.get("candidate_id"),
            "smiles": sample.get("smiles"),
            "reward": sample.get("reward"),
        }
        if "outcome" in sample:
            canonical_sample["outcome"] = sample["outcome"]
        canonical_samples.append(canonical_sample)
    envelope = {
        "schema_version": "iclm-teacher-supervision.v1",
        "teacher_source": request.teacher_source,
        "teacher_version": request.teacher_version,
        "teacher_weight": update_payload.get("teacher_weight"),
        "samples": canonical_samples,
        "teacher_embeddings_sha256": (
            f"sha256:{hashlib.sha256(request.teacher_embeddings).hexdigest()}"
        ),
    }
    serialized = json.dumps(
        envelope,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(serialized).hexdigest()}"


def _new_checkpoint_path(
    payload: dict[str, object],
    *,
    active_checkpoint_path: str,
) -> str:
    checkpoint_path = str(payload.get("checkpoint_path", "")).strip()
    if not checkpoint_path:
        raise RuntimeError("ICLM update must return a new checkpoint_path")
    resolved_checkpoint = Path(checkpoint_path).resolve()
    if resolved_checkpoint == Path(active_checkpoint_path).resolve():
        raise RuntimeError(
            "ICLM update must return a new checkpoint path distinct from the active checkpoint"
        )
    checkpoint_directory = _required_checkpoint_directory()
    if (
        resolved_checkpoint == checkpoint_directory
        or not resolved_checkpoint.is_relative_to(checkpoint_directory)
    ):
        raise RuntimeError(
            f"ICLM update checkpoint must be within {_CHECKPOINT_DIRECTORY_ENV}"
        )
    return str(resolved_checkpoint)


def _required_checkpoint_directory() -> Path:
    checkpoint_directory_value = os.environ.get(_CHECKPOINT_DIRECTORY_ENV, "").strip()
    if not checkpoint_directory_value:
        raise RuntimeError(f"{_CHECKPOINT_DIRECTORY_ENV} is required")
    return Path(checkpoint_directory_value).resolve()


async def _run_update(payload: dict[str, object], generator: object) -> dict[str, object]:
    _required_checkpoint_directory()
    if os.environ.get(_UPDATE_COMMAND_ENV, "").strip():
        model_path = str(
            getattr(generator, "checkpoint_path", "") or os.environ.get("ICLM_MODEL_PATH", "")
        )
        return await _run_update_command(payload, model_path)
    runner = getattr(generator, "runner", None)
    if _is_builtin_huggingface_runner(runner):
        model_path = str(
            getattr(generator, "checkpoint_path", "") or os.environ.get("ICLM_MODEL_PATH", "")
        )
        return await _run_builtin_huggingface_update(payload, model_path)
    return await _run_online_learner_update(payload, generator)


def _is_builtin_huggingface_runner(runner: object) -> bool:
    from mf_generators.incremental_clm.hf_runner import HuggingFaceCausalLMRunner

    return isinstance(runner, HuggingFaceCausalLMRunner)


def _builtin_hf_update_argv() -> tuple[str, ...]:
    return (sys.executable, "-m", _BUILTIN_UPDATE_MODULE)


async def _run_builtin_huggingface_update(
    payload: dict[str, object],
    model_path: str,
) -> dict[str, object]:
    return await _run_json_update_process(
        payload,
        model_path=model_path,
        argv=_builtin_hf_update_argv(),
        command_label=_BUILTIN_UPDATE_LABEL,
    )


async def _run_online_learner_update(
    payload: dict[str, object],
    generator: object,
) -> dict[str, object]:
    from mf_generators.incremental_clm.learning.online_learner import OnlineLearner

    learner = getattr(generator, "online_learner", None)
    if learner is None or not hasattr(learner, "update"):
        raise UpdateRunnerUnavailable("ICLM update runner is not configured")
    if isinstance(learner, OnlineLearner):
        raise UpdateRunnerUnavailable(
            "legacy OnlineLearner is not a production update runner; "
            "use HuggingFaceCausalLMRunner"
        )
    result = await _call_maybe_async(learner.update, payload)
    if not isinstance(result, dict):
        raise RuntimeError("online learner update must return a new checkpoint result object")
    response = dict(result)
    if not response.get("checkpoint_path"):
        raise RuntimeError("online learner update must return a new checkpoint_path")
    return response


async def _run_update_command(
    payload: dict[str, object],
    model_path: str | None = None,
) -> dict[str, object]:
    raw_command = os.environ.get(_UPDATE_COMMAND_ENV, "").strip()
    if not raw_command:
        raise UpdateRunnerUnavailable("ICLM update runner is not configured")
    _require_command_available(_UPDATE_COMMAND_REQUIREMENT, raw_command)
    try:
        argv = shlex.split(raw_command)
    except ValueError as exc:
        raise RuntimeError(f"{_UPDATE_COMMAND_ENV} is not a valid shell command") from exc
    if not argv:
        raise RuntimeError(f"{_UPDATE_COMMAND_ENV} is empty")
    return await _run_json_update_process(
        payload,
        model_path=model_path or os.environ["ICLM_MODEL_PATH"],
        argv=tuple(argv),
        command_label=_UPDATE_COMMAND_ENV,
    )


async def _run_json_update_process(
    payload: dict[str, object],
    *,
    model_path: str,
    argv: tuple[str, ...],
    command_label: str,
) -> dict[str, object]:
    command_payload = {
        **payload,
        "model_path": model_path,
        "device": os.environ.get("ICLM_DEVICE", "cpu"),
    }
    timeout = _finite_positive_float(
        os.environ.get(_UPDATE_TIMEOUT_ENV, "300"),
        _UPDATE_TIMEOUT_ENV,
    )
    process = None
    try:
        process = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        stdout, stderr = await asyncio.wait_for(
            process.communicate(
                json.dumps(
                    command_payload,
                    sort_keys=True,
                    allow_nan=False,
                ).encode("utf-8")
            ),
            timeout=timeout,
        )
    except TimeoutError as exc:
        raise RuntimeError(
            f"{command_label} execution failed: timed out after {timeout} seconds"
        ) from exc
    except asyncio.CancelledError:
        raise
    except OSError as exc:
        raise RuntimeError(f"{command_label} execution failed: {exc}") from exc
    finally:
        if process is not None:
            await _cleanup_process_group(process)
    if process.returncode != 0:
        error_message = stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"{command_label} failed: {error_message}")
    try:
        response = json.loads(stdout.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{command_label} returned invalid JSON") from exc
    if not isinstance(response, dict):
        raise RuntimeError(f"{command_label} must return a JSON object")
    if not response.get("checkpoint_path"):
        raise RuntimeError(f"{command_label} response requires checkpoint_path")
    return response


async def _cleanup_process_group(process: asyncio.subprocess.Process) -> None:
    cleanup = asyncio.create_task(_terminate_process_group(process))
    while not cleanup.done():
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError:
            continue
    cleanup.result()


async def _terminate_process_group(process: asyncio.subprocess.Process) -> None:
    with suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGTERM)
    if process.returncode is None:
        with suppress(TimeoutError):
            await asyncio.wait_for(process.wait(), timeout=0.2)
    with suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGKILL)
    if process.returncode is None:
        await process.wait()


async def _call_maybe_async(callable_object, *args):
    if inspect.iscoroutinefunction(callable_object):
        return await callable_object(*args)
    worker = asyncio.create_task(asyncio.to_thread(callable_object, *args))
    result = await _wait_for_sync_worker(worker)
    if inspect.isawaitable(result):
        return await result
    return result


async def _wait_for_sync_worker(worker: asyncio.Task):
    cancelled = False
    while True:
        try:
            result = await asyncio.shield(worker)
            break
        except asyncio.CancelledError:
            cancelled = True
            continue
        except Exception:
            if cancelled:
                raise asyncio.CancelledError from None
            raise
    if cancelled:
        raise asyncio.CancelledError
    return result


async def _eager_validate_generator(generator: object) -> None:
    validation_hook = getattr(generator, "validate_checkpoint", None)
    if not callable(validation_hook):
        runner = getattr(generator, "runner", None)
        for hook_name in ("validate_checkpoint", "validate", "load", "_load"):
            candidate = getattr(runner, hook_name, None)
            if callable(candidate):
                validation_hook = candidate
                break
    if not callable(validation_hook):
        raise RuntimeError("ICLM activated runner requires an eager checkpoint validation hook")
    result = await _call_maybe_async(validation_hook)
    if result is False:
        raise RuntimeError("ICLM activated runner rejected the checkpoint")


async def _student_embedding_dimension(generator: object) -> int | None:
    dimension_hook = getattr(generator, "embedding_dimension", None)
    if not callable(dimension_hook):
        runner = getattr(generator, "runner", None)
        dimension_hook = getattr(runner, "embedding_dimension", None)
    if not callable(dimension_hook):
        return None
    dimension = await _call_maybe_async(dimension_hook)
    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
        raise RuntimeError("ICLM student embedding dimension must be a positive integer")
    return dimension


def _model_update_fingerprint(
    payload: dict[str, object],
    *,
    fingerprint_schema: str,
    base_checkpoint_path: str,
    base_version: str,
    base_checkpoint_checksum: str,
) -> str:
    if fingerprint_schema == _LEGACY_UPDATE_FINGERPRINT_SCHEMA:
        fingerprint_payload: object = payload
    elif fingerprint_schema == _UPDATE_FINGERPRINT_SCHEMA:
        fingerprint_payload = {
            "base_checkpoint_path": str(Path(base_checkpoint_path).resolve()),
            "base_version": base_version,
            "base_checkpoint_checksum": base_checkpoint_checksum,
            "update": payload,
        }
    else:
        raise RuntimeError("ICLM update fingerprint schema is unsupported")
    serialized = json.dumps(
        fingerprint_payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _legacy_model_update_payload(
    request: generator_pb2.ModelUpdateRequest,
) -> dict[str, object]:
    run_id = _required_string(request.run_id, "run_id")
    request_id = _required_string(request.request_id, "request_id")
    teacher_source = _required_string(request.teacher_source, "teacher_source")
    teacher_version = _required_string(request.teacher_version, "teacher_version")
    target_version = _required_string(
        request.target_checkpoint_version,
        "target_checkpoint_version",
    )
    rows = _strict_positive_int(request.rows, "rows")
    dim = _strict_positive_int(request.dim, "dim")
    teacher_bytes = bytes(request.teacher_embeddings)
    if len(teacher_bytes) != rows * dim * 4:
        raise ModelUpdateRequestError(
            "teacher_embeddings byte length must equal rows * dim * 4"
        )
    try:
        flat_embeddings = struct.unpack(f"<{rows * dim}f", teacher_bytes)
    except struct.error as exc:
        raise ModelUpdateRequestError(
            "teacher_embeddings must contain float32 values"
        ) from exc
    if not all(math.isfinite(value) for value in flat_embeddings):
        raise ModelUpdateRequestError("teacher_embeddings must contain finite values")
    embeddings = [
        list(flat_embeddings[row_index * dim : (row_index + 1) * dim])
        for row_index in range(rows)
    ]
    try:
        training_batch = json.loads(str(request.training_batch_json))
    except json.JSONDecodeError as exc:
        raise ModelUpdateRequestError("training_batch_json must be valid JSON") from exc
    if not isinstance(training_batch, dict):
        raise ModelUpdateRequestError("training_batch_json must contain a JSON object")
    if training_batch.get("schema_version") != _TRAINING_BATCH_SCHEMA:
        raise ModelUpdateRequestError(
            f"training_batch_json schema_version must be {_TRAINING_BATCH_SCHEMA}"
        )
    samples = training_batch.get("samples")
    if not isinstance(samples, list) or len(samples) != rows:
        raise ModelUpdateRequestError(
            "training_batch_json samples length must match rows"
        )
    for sample in samples:
        if not isinstance(sample, dict):
            raise ModelUpdateRequestError(
                "training batch samples must be JSON objects"
            )
        if not isinstance(sample.get("smiles"), str) or not sample["smiles"].strip():
            raise ModelUpdateRequestError("training batch samples require smiles")
    kd_weight = _finite_positive_float(training_batch.get("kd_weight"), "kd_weight")
    return {
        **training_batch,
        "kd_weight": kd_weight,
        "run_id": run_id,
        "request_id": request_id,
        "kd_teacher_embeddings": embeddings,
        "teacher_source": teacher_source,
        "teacher_version": teacher_version,
        "target_checkpoint_version": target_version,
    }


def _validate_model_update_request(
    request: generator_pb2.ModelUpdateRequest,
) -> dict[str, object]:
    run_id = _required_string(request.run_id, "run_id")
    request_id = _required_string(request.request_id, "request_id")
    teacher_source = _required_string(request.teacher_source, "teacher_source")
    teacher_version = _required_string(request.teacher_version, "teacher_version")
    target_version = _required_string(
        request.target_checkpoint_version,
        "target_checkpoint_version",
    )
    target_path = Path(target_version)
    if (
        target_version in {".", ".."}
        or target_path.is_absolute()
        or "/" in target_version
        or "\\" in target_version
    ):
        raise ModelUpdateRequestError(
            "target_checkpoint_version must be a file-safe name"
    )
    rows = _strict_positive_int(request.rows, "rows")
    dim = _strict_non_negative_int(request.dim, "dim")
    expected_bytes = rows * dim * 4
    teacher_bytes = bytes(request.teacher_embeddings)
    if len(teacher_bytes) != expected_bytes:
        raise ModelUpdateRequestError("teacher_embeddings byte length must equal rows * dim * 4")
    try:
        flat_embeddings = struct.unpack(f"<{rows * dim}f", teacher_bytes)
    except struct.error as exc:
        raise ModelUpdateRequestError("teacher_embeddings must contain float32 values") from exc
    if not all(math.isfinite(value) for value in flat_embeddings):
        raise ModelUpdateRequestError("teacher_embeddings must contain finite values")
    embeddings = [
        list(flat_embeddings[row_index * dim : (row_index + 1) * dim]) for row_index in range(rows)
    ]

    raw_batch = str(request.training_batch_json)
    try:
        training_batch = json.loads(raw_batch)
    except json.JSONDecodeError as exc:
        raise ModelUpdateRequestError("training_batch_json must be valid JSON") from exc
    if not isinstance(training_batch, dict):
        raise ModelUpdateRequestError("training_batch_json must contain a JSON object")
    if training_batch.get("schema_version") != _TRAINING_BATCH_SCHEMA:
        raise ModelUpdateRequestError(
            f"training_batch_json schema_version must be {_TRAINING_BATCH_SCHEMA}"
        )
    samples = training_batch.get("samples")
    if not isinstance(samples, list) or len(samples) != rows:
        raise ModelUpdateRequestError("training_batch_json samples length must match rows")
    validated_samples: list[dict[str, object]] = []
    for sample_index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise ModelUpdateRequestError("training batch samples must be JSON objects")
        if not isinstance(sample.get("smiles"), str) or not sample["smiles"].strip():
            raise ModelUpdateRequestError("training batch samples require smiles")
        try:
            canonical_smiles = canonicalize(sample["smiles"].strip())
        except (ImportError, ValueError) as exc:
            raise ModelUpdateRequestError(
                "training batch sample smiles must be a valid molecule"
            ) from exc
        candidate_id = sample.get("candidate_id")
        if candidate_id is None:
            candidate_id = f"{request_id}:{sample_index + 1}"
        elif (
            not isinstance(candidate_id, str)
            or not candidate_id
            or candidate_id != candidate_id.strip()
        ):
            raise ModelUpdateRequestError(
                "training batch samples require a non-empty candidate_id"
            )
        reward_value = sample.get("reward", 1.0)
        if isinstance(reward_value, bool) or not isinstance(reward_value, int | float):
            raise ModelUpdateRequestError("training batch sample reward must be numeric")
        reward = float(reward_value)
        if not math.isfinite(reward) or not 0.0 <= reward <= 1.0:
            raise ModelUpdateRequestError(
                "training batch sample reward must be finite and in [0, 1]"
            )
        outcome = sample.get("outcome")
        if outcome is not None and outcome not in {"PASS", "FAIL"}:
            raise ModelUpdateRequestError(
                "training batch sample outcome must be PASS or FAIL"
            )
        validated_samples.append(
            {
                **sample,
                "smiles": canonical_smiles,
                "candidate_id": candidate_id,
                "reward": reward,
            }
        )
    training_batch["samples"] = validated_samples
    raw_teacher_weight = training_batch.get(
        "teacher_weight",
        training_batch.get("kd_weight"),
    )
    teacher_weight = _finite_unit_interval_float(
        raw_teacher_weight,
        "teacher_weight",
    )
    if teacher_weight > 0.0 and dim == 0:
        raise ModelUpdateRequestError(
            "positive teacher_weight requires teacher_embeddings"
        )
    training_batch.pop("kd_weight", None)
    return {
        **training_batch,
        "teacher_weight": teacher_weight,
        "run_id": run_id,
        "request_id": request_id,
        "teacher_embeddings": embeddings,
        "teacher_source": teacher_source,
        "teacher_version": teacher_version,
        "target_checkpoint_version": target_version,
    }


def _sample_learning_strength(sample: object) -> float:
    if not isinstance(sample, dict):
        raise RuntimeError("validated ICLM training sample must be an object")
    reward = float(sample["reward"])
    return 1.0 - reward if sample.get("outcome") == "FAIL" else reward


def _required_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ModelUpdateRequestError(
            f"{field} must be a non-empty trimmed string"
        )
    return value


def _strict_positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ModelUpdateRequestError(f"{field} must be an integer")
    if value <= 0:
        raise ModelUpdateRequestError(f"{field} must be positive")
    return value


def _strict_non_negative_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ModelUpdateRequestError(f"{field} must be an integer")
    if value < 0:
        raise ModelUpdateRequestError(f"{field} must be non-negative")
    return value


def _finite_positive_float(value: object, field: str) -> float:
    if isinstance(value, bool):
        raise ModelUpdateRequestError(f"{field} must be numeric")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ModelUpdateRequestError(f"{field} must be numeric") from exc
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise ModelUpdateRequestError(f"{field} must be finite and positive")
    return parsed


def _finite_unit_interval_float(value: object, field: str) -> float:
    if isinstance(value, bool):
        raise ModelUpdateRequestError(f"{field} must be numeric")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ModelUpdateRequestError(f"{field} must be numeric") from exc
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise ModelUpdateRequestError(f"{field} must be finite and in [0, 1]")
    return parsed


def _require_command_available(
    requirement: CommandRequirement,
    command: str,
) -> None:
    required_requirement = CommandRequirement(
        requirement.name,
        requirement.env_var,
        required=True,
    )
    env = {**os.environ, requirement.env_var: command}
    require_available([check_command(required_requirement, env=env)])


async def serve():
    _required_internal_service_token()
    state_path = _required_state_path()
    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=8))
    servicer = ICLMServicer(state_path=state_path)
    await servicer.initialize()
    _, checkpoint_path, _ = await servicer._active_snapshot()
    _require_runtime(include_update_command=True, model_path=checkpoint_path)
    generator_pb2_grpc.add_GeneratorServiceServicer_to_server(servicer, server)
    generator_pb2_grpc.add_IncrementalGeneratorServiceServicer_to_server(
        servicer,
        server,
    )
    server.add_insecure_port("[::]:50067")
    await server.start()
    print("ICLM Service running on :50067")
    await server.wait_for_termination()


if __name__ == "__main__":
    asyncio.run(serve())
