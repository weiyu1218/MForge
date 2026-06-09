"""ICLM Service - gRPC server for Incremental Causal Language Model with EWC."""
import asyncio
import inspect
import json
import os
import shlex
import subprocess
import time
from concurrent import futures

import grpc
from mf_core.artifacts import (
    ArtifactRequirement,
    CommandRequirement,
    RequirementStatus,
    check_artifact,
    check_command,
    require_available,
)
from mf_core.proto_gen.moleculeforge.v1.generator import generator_pb2, generator_pb2_grpc
from mf_core.types.humu import IntentCone

_REQUIREMENTS = (ArtifactRequirement("iclm_model", "ICLM_MODEL_PATH", kind="path"),)
_GENERATOR_NAME = "iclm"
_UPDATE_COMMAND_ENV = "ICLM_UPDATE_COMMAND"
_UPDATE_TIMEOUT_ENV = "ICLM_UPDATE_TIMEOUT_SECONDS"
_UPDATE_COMMAND_REQUIREMENT = CommandRequirement(
    "iclm_update_command",
    _UPDATE_COMMAND_ENV,
)


def _require_runtime(*, include_update_command: bool = False) -> list[RequirementStatus]:
    statuses = _runtime_statuses(include_update_command=include_update_command)
    require_available(statuses)
    return statuses


def runtime_status() -> list[dict]:
    return [status.to_dict() for status in _runtime_statuses(include_update_command=True)]


def _runtime_statuses(*, include_update_command: bool) -> list[RequirementStatus]:
    statuses = [check_artifact(requirement) for requirement in _REQUIREMENTS]
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


def _batch_size(request) -> int:
    value = int(getattr(request, "batch_size", 0))
    if value <= 0:
        raise ValueError("batch_size must be positive")
    return value


def _serialize_molecule(molecule) -> bytes:
    if hasattr(molecule, "model_dump_json"):
        return molecule.model_dump_json().encode("utf-8")
    if isinstance(molecule, dict):
        return json.dumps(molecule, sort_keys=True).encode("utf-8")
    raise TypeError(f"Unsupported molecule payload: {type(molecule)!r}")


async def _abort_invalid_argument(context, message: str):
    if context is not None and hasattr(context, "abort"):
        await context.abort(grpc.StatusCode.INVALID_ARGUMENT, message)
    raise ValueError(message)


def _intent_cone_from_request(request) -> IntentCone | None:
    raw = getattr(request, "intent_cone", None)
    if raw in (None, "", b"", {}):
        return None
    if isinstance(raw, IntentCone):
        return raw
    if isinstance(raw, bytes):
        raw = json.loads(raw.decode("utf-8"))
    elif isinstance(raw, str):
        raw = json.loads(raw)
    elif hasattr(raw, "model_dump"):
        raw = raw.model_dump(mode="json")
    if isinstance(raw, dict):
        return IntentCone.model_validate(raw)
    raise TypeError(f"Unsupported intent_cone payload: {type(raw)!r}")


def _build_generator():
    from mf_generators.incremental_clm.generator import IncrementalCLMGenerator
    from mf_generators.incremental_clm.hf_runner import HuggingFaceCausalLMRunner

    model_path = os.environ["ICLM_MODEL_PATH"]
    device = os.environ.get("ICLM_DEVICE", "cpu")
    return IncrementalCLMGenerator(
        checkpoint_path=model_path,
        device=device,
        runner=HuggingFaceCausalLMRunner(model_path=model_path, device=device),
    )


class ICLMServicer:
    def __init__(self, generator=None):
        self.generator = generator if generator is not None else _build_generator()

    async def Generate(self, request, context):
        """Generate molecular structures via incremental causal LM."""
        try:
            _require_runtime()
        except RuntimeError:
            return await _abort_unavailable(context)
        if self.generator is None:
            return await _abort_unavailable(context)
        try:
            batch_size = _batch_size(request)
        except ValueError as exc:
            return await _abort_invalid_argument(context, str(exc))
        params = dict(getattr(request, "generator_params", {}) or {})
        start = time.perf_counter()
        molecules = await self.generator.generate(
            batch_size=batch_size,
            intent_cone=_intent_cone_from_request(request),
            **params,
        )
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        return type(
            "GenerateResponse",
            (),
            {
                "generator_name": _GENERATOR_NAME,
                "generation_id": getattr(request, "project_id", ""),
                "molecules": [_serialize_molecule(mol) for mol in molecules],
                "humu_embeddings": [],
                "aggregate_stats": {},
                "elapsed_ms": elapsed_ms,
            },
        )()

    async def GenerateStream(self, request_iterator, context):
        async for request in request_iterator:
            yield await self.Generate(request, context)

    async def BatchGenerate(self, request_iterator, context):
        async for request in request_iterator:
            yield await self.Generate(request, context)

    async def Info(self, request, context):
        return generator_pb2.GeneratorInfo(
            name=_GENERATOR_NAME,
            version="0.1.0",
            description="Incremental causal language model generator",
            supported_properties=["qed", "sa_score", "mw", "logp"],
            max_batch_size=256,
            supports_streaming=True,
            requires_gpu=False,
        )

    async def UpdateModel(self, request, context):
        """Update ICLM via Elastic Weight Consolidation (EWC) with new data."""
        try:
            _require_runtime(include_update_command=True)
        except RuntimeError:
            return await _abort_unavailable(context, include_update_command=True)
        try:
            payload = await _run_update(request, self.generator)
        except RuntimeError as exc:
            return await _abort_unavailable(context, str(exc))
        return _update_model_response(payload)


def _update_model_response(payload: dict):
    return type(
        "UpdateModelResponse",
        (),
        {
            "checkpoint_path": str(payload["checkpoint_path"]),
            "updated_samples": int(payload.get("updated_samples", 0)),
            "ewc_loss": float(payload.get("ewc_loss", 0.0)),
            "kd_loss": float(payload.get("kd_loss", 0.0)),
            "metadata": {
                str(key): str(value)
                for key, value in dict(payload.get("metadata", {}) or {}).items()
            },
        },
    )()


async def _run_update(request, generator) -> dict:
    if os.environ.get(_UPDATE_COMMAND_ENV, "").strip():
        return _run_update_command(request)
    return await _run_online_learner_update(request, generator)


async def _run_online_learner_update(request, generator) -> dict:
    learner = getattr(generator, "online_learner", None)
    if learner is None or not hasattr(learner, "update"):
        raise RuntimeError("ICLM update runner is not configured")
    payload = _request_to_json_mapping(request)
    result = learner.update(payload)
    if inspect.isawaitable(result):
        result = await result
    if isinstance(result, dict):
        response = dict(result)
    else:
        samples = payload.get("training_samples", [])
        task_loss = getattr(learner, "last_task_loss", result)
        kd_loss = getattr(learner, "last_kd_loss", 0.0)
        response = {
            "checkpoint_path": str(
                getattr(generator, "checkpoint_path", "")
                or os.environ.get("ICLM_MODEL_PATH", "")
            ),
            "updated_samples": len(samples) if isinstance(samples, list) else 0,
            "ewc_loss": float(task_loss),
            "kd_loss": float(kd_loss),
            "metadata": {"mode": "online_learner"},
        }
    if not response.get("checkpoint_path"):
        raise RuntimeError("online learner update response requires checkpoint_path")
    return response


def _run_update_command(request) -> dict:
    raw_command = os.environ.get(_UPDATE_COMMAND_ENV, "").strip()
    if not raw_command:
        raise RuntimeError("ICLM update runner is not configured")
    _require_command_available(_UPDATE_COMMAND_REQUIREMENT, raw_command)
    try:
        argv = shlex.split(raw_command)
    except ValueError as exc:
        raise RuntimeError(f"{_UPDATE_COMMAND_ENV} is not a valid shell command") from exc
    if not argv:
        raise RuntimeError(f"{_UPDATE_COMMAND_ENV} is empty")
    payload = _request_to_json_mapping(request)
    payload["model_path"] = os.environ["ICLM_MODEL_PATH"]
    payload["device"] = os.environ.get("ICLM_DEVICE", "cpu")
    completed = subprocess.run(
        argv,
        input=json.dumps(payload),
        capture_output=True,
        check=False,
        text=True,
        timeout=float(os.environ.get(_UPDATE_TIMEOUT_ENV, "300")),
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        raise RuntimeError(f"{_UPDATE_COMMAND_ENV} failed: {stderr}")
    try:
        response = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{_UPDATE_COMMAND_ENV} returned invalid JSON") from exc
    if not isinstance(response, dict):
        raise RuntimeError(f"{_UPDATE_COMMAND_ENV} must return a JSON object")
    if not response.get("checkpoint_path"):
        raise RuntimeError(f"{_UPDATE_COMMAND_ENV} response requires checkpoint_path")
    return response


def _request_to_json_mapping(request) -> dict:
    if isinstance(request, dict):
        return _json_safe_dict(request)
    if hasattr(request, "model_dump"):
        return _json_safe_dict(request.model_dump(mode="json"))
    return _json_safe_dict(vars(request))


def _json_safe_dict(payload: dict) -> dict:
    return {str(key): _json_safe_value(value) for key, value in payload.items()}


def _json_safe_value(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, dict):
        return _json_safe_dict(value)
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(item) for item in value]
    return value


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
    generator_pb2_grpc.add_GeneratorServiceServicer_to_server(ICLMServicer(), server)
    server.add_insecure_port("[::]:50067")
    await server.start()
    print("ICLM Service running on :50067")
    await server.wait_for_termination()


if __name__ == "__main__":
    asyncio.run(serve())
