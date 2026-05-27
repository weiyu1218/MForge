"""HFM Molecule Generator Service - gRPC server for Hyperbolic Flow Matching generation."""
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
from mf_core.proto_gen.moleculeforge.v1.generator import generator_pb2_grpc

_REQUIREMENTS = (
    ArtifactRequirement("hfm_checkpoint", "HFM_CHECKPOINT_PATH"),
    ArtifactRequirement("hfm_decoder", "HFM_DECODER_PATH"),
)
_GENERATOR_NAME = "hfm_3d"


def _require_runtime() -> list[RequirementStatus]:
    statuses = [check_artifact(requirement) for requirement in _REQUIREMENTS]
    require_available(statuses)
    return statuses


def runtime_status() -> list[dict]:
    return [check_artifact(requirement).to_dict() for requirement in _REQUIREMENTS]


async def _abort_unavailable(context):
    statuses = [check_artifact(requirement) for requirement in _REQUIREMENTS]
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


def _build_generator():
    from mf_generators.hfm_3d.generator import HFM3DGenerator

    return HFM3DGenerator(
        checkpoint_path=os.environ["HFM_CHECKPOINT_PATH"],
        decoder_path=os.environ["HFM_DECODER_PATH"],
    )


class HFMGeneratorServicer:
    def __init__(self, generator=None):
        self.generator = generator if generator is not None else _build_generator()

    async def Generate(self, request, context):
        """Generate molecules via Hyperbolic Flow Matching in Lorentz manifold."""
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
            intent_cone=None,
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
        return request


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
