"""HFM Molecule Generator Service - gRPC server for Hyperbolic Flow Matching generation."""

import asyncio
import os
import time
from concurrent import futures

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
    build_generate_response,
    build_generator_info,
    validate_generate_request,
)
from mf_core.proto_gen.moleculeforge.v1.generator import generator_pb2_grpc

_CHECKPOINT_REQUIREMENT = ArtifactRequirement("hfm_checkpoint", "HFM_CHECKPOINT_PATH")
_DECODER_REQUIREMENT = ArtifactRequirement("hfm_decoder", "HFM_DECODER_PATH")
_MOLECULAR_DECODER_COMMAND = CommandRequirement(
    "hfm_molecular_decoder_command",
    "HFM_MOLECULAR_DECODER_COMMAND",
)
_GENERATOR_NAME = "hfm_3d"
_MAX_BATCH_SIZE = 1024


def _require_runtime() -> list[RequirementStatus]:
    statuses = _runtime_statuses()
    require_available(statuses)
    return statuses


def runtime_status() -> list[dict]:
    return [status.to_dict() for status in _runtime_statuses()]


def _runtime_statuses() -> list[RequirementStatus]:
    statuses = [check_artifact(_CHECKPOINT_REQUIREMENT)]
    decoder_command = os.environ.get("HFM_MOLECULAR_DECODER_COMMAND", "").strip()
    if decoder_command:
        statuses.append(check_command(_MOLECULAR_DECODER_COMMAND))
    else:
        statuses.append(check_artifact(_DECODER_REQUIREMENT))
    return statuses


async def _abort_unavailable(context):
    statuses = _runtime_statuses()
    try:
        require_available(statuses)
    except RuntimeError as exc:
        message = str(exc)
    else:
        message = "HFM generator runner is not configured"
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


def _build_generator():
    from mf_generators.hfm_3d.generator import HFM3DGenerator

    return HFM3DGenerator(
        checkpoint_path=os.environ["HFM_CHECKPOINT_PATH"],
        decoder_path=os.environ.get("HFM_DECODER_PATH", ""),
    )


class HFMGeneratorServicer:
    def __init__(self, generator=None):
        self.generator = generator if generator is not None else _build_generator()

    async def Generate(self, request, context):
        """Generate molecules via Hyperbolic Flow Matching in Lorentz manifold."""
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

    async def GenerateStream(self, request_iterator, context):
        async for request in request_iterator:
            yield await self.Generate(request, context)

    async def BatchGenerate(self, request_iterator, context):
        async for request in request_iterator:
            yield await self.Generate(request, context)

    async def Info(self, request, context):
        return await build_generator_info(
            generator_name=_GENERATOR_NAME,
            generator=self.generator,
            statuses=_runtime_statuses(),
            fallback={
                "version": "0.1.0",
                "description": "HFM-3D Lorentz flow generator",
                "supported_properties": ["qed", "sa_score", "mw", "logp"],
                "max_batch_size": _MAX_BATCH_SIZE,
                "supports_streaming": True,
                "requires_gpu": True,
            },
        )


async def serve():
    _require_runtime()
    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=6))
    generator_pb2_grpc.add_GeneratorServiceServicer_to_server(HFMGeneratorServicer(), server)
    server.add_insecure_port("[::]:50066")
    await server.start()
    print("HFM Generator Service running on :50066")
    await server.wait_for_termination()


if __name__ == "__main__":
    asyncio.run(serve())
