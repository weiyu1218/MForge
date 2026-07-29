"""FragFM Generator Service - gRPC server for fragment-based molecule generation."""

import asyncio
import os
import time
from concurrent import futures

import grpc
from mf_chem.molecule.parsing import canonicalize
from mf_core.artifacts import (
    ArtifactRequirement,
    RequirementStatus,
    check_artifact,
    require_available,
)
from mf_core.plugins.generator import (
    GeneratorRequestError,
    GeneratorResultError,
    build_generate_response,
    build_generator_info,
    validate_generate_request,
)
from mf_core.proto_gen.moleculeforge.v1.generator import generator_pb2_grpc
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
_MAX_BATCH_SIZE = 512


def _require_runtime() -> list[RequirementStatus]:
    statuses = _runtime_statuses()
    require_available(statuses)
    _require_configured_artifacts_available(statuses)
    return statuses


def runtime_status() -> list[dict]:
    return [status.to_dict() for status in _runtime_statuses()]


def _runtime_statuses() -> list[RequirementStatus]:
    return [check_artifact(requirement) for requirement in _REQUIREMENTS]


def _require_configured_artifacts_available(statuses: list[RequirementStatus]) -> None:
    missing = [status for status in statuses if status.configured and not status.available]
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


async def _abort_internal(context, message: str):
    if context is not None and hasattr(context, "abort"):
        await context.abort(grpc.StatusCode.INTERNAL, message)
    raise RuntimeError(message)


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
            statuses = _require_runtime()
        except RuntimeError:
            return await _abort_unavailable(context)
        if self.generator is None:
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
            molecules = await self.generator.generate(
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

    async def GenerateStream(self, request_iterator, context):  # noqa: N802
        async for request in request_iterator:
            yield await self.Generate(request, context)

    async def BatchGenerate(self, request_iterator, context):  # noqa: N802
        async for request in request_iterator:
            yield await self.Generate(request, context)

    async def Info(self, request, context):  # noqa: N802
        return await build_generator_info(
            generator_name=_GENERATOR_NAME,
            generator=self.generator,
            statuses=_runtime_statuses(),
            fallback={
                "version": "0.1.0",
                "description": "FragFM fragment assembly generator",
                "supported_properties": ["qed", "sa_score", "mw", "logp"],
                "max_batch_size": _MAX_BATCH_SIZE,
                "supports_streaming": True,
                "requires_gpu": False,
            },
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
