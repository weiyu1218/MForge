"""ICLM Service - gRPC server for Incremental Causal Language Model with EWC."""

import asyncio
import hashlib
import inspect
import json
import math
import os
import shlex
import signal
import struct
import time
from concurrent import futures
from dataclasses import dataclass
from pathlib import Path

import grpc
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
_UPDATE_COMMAND_REQUIREMENT = CommandRequirement(
    "iclm_update_command",
    _UPDATE_COMMAND_ENV,
)
_TRAINING_BATCH_SCHEMA = "training-batch.v1"


class ModelUpdateRequestError(ValueError):
    pass


class UpdateRunnerUnavailable(RuntimeError):
    pass


@dataclass
class _UpdateRecord:
    fingerprint: str
    state: str
    response: bytes | None = None
    failure_code: grpc.StatusCode | None = None
    failure_message: str = ""


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
    if include_update_command and os.environ.get(_UPDATE_COMMAND_ENV, "").strip():
        statuses.append(check_command(_UPDATE_COMMAND_REQUIREMENT))
    return statuses


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

    model_path = model_path or os.environ["ICLM_MODEL_PATH"]
    device = os.environ.get("ICLM_DEVICE", "cpu")
    return IncrementalCLMGenerator(
        checkpoint_path=model_path,
        device=device,
        runner=HuggingFaceCausalLMRunner(model_path=model_path, device=device),
    )


def _record_update_failure(
    record: _UpdateRecord,
    code: grpc.StatusCode,
    message: str,
) -> None:
    record.state = "failed"
    record.response = None
    record.failure_code = code
    record.failure_message = message or "ICLM model update failed"


async def _replay_update_failure(context, record: _UpdateRecord):
    code = record.failure_code or grpc.StatusCode.INTERNAL
    message = record.failure_message or "ICLM model update failed"
    if context is not None and hasattr(context, "abort"):
        await context.abort(code, message)
    raise RuntimeError(message)


class ICLMServicer(
    generator_pb2_grpc.GeneratorServiceServicer,
    generator_pb2_grpc.IncrementalGeneratorServiceServicer,
):
    def __init__(self, generator=None, generator_factory=None):
        self.generator = generator if generator is not None else _build_generator()
        self._generator_factory = generator_factory or _build_generator
        self._active_checkpoint_path = str(
            getattr(self.generator, "checkpoint_path", "") or os.environ.get("ICLM_MODEL_PATH", "")
        )
        self._active_version = str(getattr(self.generator, "version", "")).strip() or "0.1.0"
        self._activation_lock = asyncio.Lock()
        self._update_lock = asyncio.Lock()
        self._update_records: dict[str, _UpdateRecord] = {}

    async def Generate(self, request, context):
        """Generate molecular structures via incremental causal LM."""
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
        generator, checkpoint_path, active_version = await self._active_snapshot()
        response = await build_generator_info(
            generator_name=_GENERATOR_NAME,
            generator=generator,
            statuses=_runtime_statuses(
                include_update_command=False,
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
        return response

    async def UpdateModel(self, request, context):
        """Update ICLM via Elastic Weight Consolidation (EWC) with new data."""
        try:
            update_payload = _validate_model_update_request(request)
        except ModelUpdateRequestError as exc:
            return await _abort_invalid_argument(context, str(exc))
        request_fingerprint = _model_update_fingerprint(update_payload)
        async with self._update_lock:
            record = self._update_records.get(request.request_id)
            if record is not None:
                if record.fingerprint != request_fingerprint:
                    return await _abort_invalid_argument(
                        context,
                        "request_id is already bound to a different model update request",
                    )
                if record.state == "succeeded" and record.response is not None:
                    return generator_pb2.ModelUpdateResponse.FromString(record.response)
                if record.state == "failed":
                    return await _replay_update_failure(context, record)
            else:
                record = _UpdateRecord(
                    fingerprint=request_fingerprint,
                    state="in_progress",
                )
                self._update_records[request.request_id] = record
            record.state = "in_progress"
            record.response = None
            record.failure_code = None
            record.failure_message = ""
            active_generator, active_checkpoint_path, _ = await self._active_snapshot()
            try:
                _require_runtime(
                    include_update_command=True,
                    model_path=active_checkpoint_path,
                )
            except RuntimeError as exc:
                _record_update_failure(
                    record,
                    grpc.StatusCode.FAILED_PRECONDITION,
                    str(exc),
                )
                return await _abort_unavailable(context, str(exc))
            try:
                if os.environ.get(_UPDATE_COMMAND_ENV, "").strip():
                    update_generator = active_generator
                else:
                    update_generator = await self._construct_generator(active_checkpoint_path)
                result = await _run_update(update_payload, update_generator)
                checkpoint_path = _new_checkpoint_path(
                    result,
                    active_checkpoint_path=active_checkpoint_path,
                )
                response = _update_model_response(request, result)
                activated_generator = await self._construct_generator(
                    checkpoint_path,
                    eager_validate=True,
                )
                serialized_response = response.SerializeToString(deterministic=True)
                async with self._activation_lock:
                    self.generator = activated_generator
                    self._active_checkpoint_path = checkpoint_path
                    self._active_version = response.active_version
                    record.state = "succeeded"
                    record.response = serialized_response
                return response
            except asyncio.CancelledError:
                _record_update_failure(
                    record,
                    grpc.StatusCode.CANCELLED,
                    "ICLM model update was cancelled",
                )
                raise
            except UpdateRunnerUnavailable as exc:
                _record_update_failure(
                    record,
                    grpc.StatusCode.FAILED_PRECONDITION,
                    str(exc),
                )
                return await _abort_unavailable(context, str(exc))
            except Exception as exc:
                _record_update_failure(
                    record,
                    grpc.StatusCode.INTERNAL,
                    str(exc),
                )
                return await _abort_internal(context, str(exc))

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


def _update_model_response(
    request: generator_pb2.ModelUpdateRequest,
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
        checksum=f"sha256:{hashlib.sha256(request.teacher_embeddings).hexdigest()}",
        required=True,
    )
    return generator_pb2.ModelUpdateResponse(
        acknowledged=True,
        active_version=request.target_checkpoint_version,
        artifacts=[checkpoint_ref, teacher_ref],
        updated_samples=updated_samples,
    )


def _new_checkpoint_path(
    payload: dict[str, object],
    *,
    active_checkpoint_path: str,
) -> str:
    checkpoint_path = str(payload.get("checkpoint_path", "")).strip()
    if not checkpoint_path:
        raise RuntimeError("ICLM update must return a new checkpoint_path")
    if Path(checkpoint_path).resolve() == Path(active_checkpoint_path).resolve():
        raise RuntimeError(
            "ICLM update must return a new checkpoint path distinct from the active checkpoint"
        )
    return checkpoint_path


async def _run_update(payload: dict[str, object], generator: object) -> dict[str, object]:
    if os.environ.get(_UPDATE_COMMAND_ENV, "").strip():
        model_path = str(
            getattr(generator, "checkpoint_path", "") or os.environ.get("ICLM_MODEL_PATH", "")
        )
        return await _run_update_command(payload, model_path)
    return await _run_online_learner_update(payload, generator)


async def _run_online_learner_update(
    payload: dict[str, object],
    generator: object,
) -> dict[str, object]:
    learner = getattr(generator, "online_learner", None)
    if learner is None or not hasattr(learner, "update"):
        raise UpdateRunnerUnavailable("ICLM update runner is not configured")
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
    command_payload = {
        **payload,
        "model_path": model_path or os.environ["ICLM_MODEL_PATH"],
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
            f"{_UPDATE_COMMAND_ENV} execution failed: timed out after {timeout} seconds"
        ) from exc
    except asyncio.CancelledError:
        raise
    except OSError as exc:
        raise RuntimeError(f"{_UPDATE_COMMAND_ENV} execution failed: {exc}") from exc
    finally:
        if process is not None:
            await _cleanup_process_group(process)
    if process.returncode != 0:
        error_message = stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"{_UPDATE_COMMAND_ENV} failed: {error_message}")
    try:
        response = json.loads(stdout.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{_UPDATE_COMMAND_ENV} returned invalid JSON") from exc
    if not isinstance(response, dict):
        raise RuntimeError(f"{_UPDATE_COMMAND_ENV} must return a JSON object")
    if not response.get("checkpoint_path"):
        raise RuntimeError(f"{_UPDATE_COMMAND_ENV} response requires checkpoint_path")
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
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    if process.returncode is None:
        try:
            await asyncio.wait_for(process.wait(), timeout=0.2)
        except TimeoutError:
            pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
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


def _model_update_fingerprint(payload: dict[str, object]) -> str:
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


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
    rows = _strict_positive_int(request.rows, "rows")
    dim = _strict_positive_int(request.dim, "dim")
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
    for sample in samples:
        if not isinstance(sample, dict):
            raise ModelUpdateRequestError("training batch samples must be JSON objects")
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


def _required_string(value: object, field: str) -> str:
    parsed = str(value).strip()
    if not parsed:
        raise ModelUpdateRequestError(f"{field} is required")
    return parsed


def _strict_positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ModelUpdateRequestError(f"{field} must be an integer")
    if value <= 0:
        raise ModelUpdateRequestError(f"{field} must be positive")
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
    _require_runtime()
    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=8))
    servicer = ICLMServicer()
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
