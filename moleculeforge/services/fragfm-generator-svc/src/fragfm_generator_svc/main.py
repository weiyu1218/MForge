"""FragFM Generator Service - gRPC server for fragment-based molecule generation."""
import asyncio
import json
import os
import time
from concurrent import futures

import grpc
from mf_core.artifacts import (
    ArtifactRequirement,
    RequirementStatus,
    check_artifact,
    require_available,
)
from mf_core.proto_gen.moleculeforge.v1.generator import generator_pb2, generator_pb2_grpc
from mf_core.types.humu import IntentCone

_REQUIREMENTS = (
    ArtifactRequirement("fragfm_vocabulary", "FRAGFM_VOCAB_PATH"),
    ArtifactRequirement(
        "fragfm_checkpoint",
        "FRAGFM_CHECKPOINT_PATH",
        required=False,
    ),
    ArtifactRequirement(
        "fragfm_rate_matrix",
        "FRAGFM_RATE_MATRIX_PATH",
        required=False,
    ),
)
_GENERATOR_NAME = "fragfm"


def _require_runtime() -> list[RequirementStatus]:
    statuses = [check_artifact(requirement) for requirement in _REQUIREMENTS]
    require_available(statuses)
    _require_configured_artifacts_available(statuses)
    return statuses


def runtime_status() -> list[dict]:
    return [check_artifact(requirement).to_dict() for requirement in _REQUIREMENTS]


def _require_configured_artifacts_available(statuses: list[RequirementStatus]) -> None:
    missing = [
        status
        for status in statuses
        if status.configured and not status.available
    ]
    if missing:
        details = "; ".join(f"{status.name}: {status.message}" for status in missing)
        raise RuntimeError(f"Configured FragFM artifacts are unavailable: {details}")


async def _abort_unavailable(context):
    statuses = [check_artifact(requirement) for requirement in _REQUIREMENTS]
    try:
        require_available(statuses)
        _require_configured_artifacts_available(statuses)
    except RuntimeError as exc:
        message = str(exc)
    else:
        message = "FragFM generator runner is not configured"
    if context is not None and hasattr(context, "abort"):
        await context.abort(grpc.StatusCode.FAILED_PRECONDITION, message)
    raise RuntimeError(message)


async def _abort_invalid_argument(context, message: str):
    if context is not None and hasattr(context, "abort"):
        await context.abort(grpc.StatusCode.INVALID_ARGUMENT, message)
    raise ValueError(message)


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


class SharedHUMULatentSampler:
    def __init__(self, curvature: float = 1.0):
        from mf_humu.manifold.lorentz import LorentzManifold

        self.curvature = curvature
        self.manifold = LorentzManifold(curvature=curvature)

    def sample(self, *, batch_size: int, intent_cone: IntentCone | None):
        if intent_cone is None:
            return None
        from mf_humu.operations.intent_cone import sample_within_cone

        latents = sample_within_cone(
            intent_cone,
            n_samples=batch_size,
            manifold=self.manifold,
        )
        return latents.detach().cpu().tolist()


def _build_generator():
    from mf_generators.fragfm.generator import FragFMGenerator

    return FragFMGenerator(
        checkpoint_path=os.environ.get("FRAGFM_CHECKPOINT_PATH", ""),
        rate_matrix_path=os.environ.get("FRAGFM_RATE_MATRIX_PATH", ""),
        vocab_path=os.environ["FRAGFM_VOCAB_PATH"],
        humu_latent_sampler=SharedHUMULatentSampler(
            curvature=float(os.environ.get("FRAGFM_HUMU_CURVATURE", "1.0"))
        ),
    )


class FragFMGeneratorServicer:
    def __init__(self, generator=None):
        self.generator = generator if generator is not None else _build_generator()

    async def Generate(self, request, context):  # noqa: N802
        """Generate molecules via FragFM fragment assembly."""
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
        try:
            intent_cone = _intent_cone_from_request(request)
        except (TypeError, ValueError) as exc:
            return await _abort_invalid_argument(context, f"intent_cone is invalid: {exc}")
        start = time.perf_counter()
        molecules = await self.generator.generate(
            batch_size=batch_size,
            intent_cone=intent_cone,
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

    async def GenerateStream(self, request_iterator, context):  # noqa: N802
        async for request in request_iterator:
            yield await self.Generate(request, context)

    async def BatchGenerate(self, request_iterator, context):  # noqa: N802
        async for request in request_iterator:
            yield await self.Generate(request, context)

    async def Info(self, request, context):  # noqa: N802
        return generator_pb2.GeneratorInfo(
            name=_GENERATOR_NAME,
            version="0.1.0",
            description="FragFM fragment assembly generator",
            supported_properties=["qed", "sa_score", "mw", "logp"],
            max_batch_size=512,
            supports_streaming=True,
            requires_gpu=False,
        )


async def serve():
    _require_runtime()
    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=8))
    generator_pb2_grpc.add_GeneratorServiceServicer_to_server(FragFMGeneratorServicer(), server)
    server.add_insecure_port("[::]:50065")
    await server.start()
    print("FragFM Generator Service running on :50065")
    await server.wait_for_termination()


if __name__ == "__main__":
    asyncio.run(serve())
