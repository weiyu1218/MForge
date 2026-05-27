"""LaMGen-3D Generator Service - gRPC server for multi-target molecule generation."""
import asyncio
from concurrent import futures

import grpc
from mf_core.artifacts import (
    ArtifactRequirement,
    RequirementStatus,
    check_artifact,
    require_available,
)
from mf_core.proto_gen.moleculeforge.v1.generator import generator_pb2_grpc

_REQUIREMENTS = (ArtifactRequirement("lamgen_model", "LAMGEN_MODEL_PATH", kind="path"),)


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
        message = "LaMGen generator runner is not configured"
    if context is not None and hasattr(context, "abort"):
        await context.abort(grpc.StatusCode.FAILED_PRECONDITION, message)
    raise RuntimeError(message)


class LaMGenGeneratorServicer:
    async def Generate(self, request, context):
        """Generate multi-target molecules via LaMGen-3D."""
        try:
            _require_runtime()
        except RuntimeError:
            return await _abort_unavailable(context)
        raise RuntimeError("LaMGen generator runner is not configured")

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
    generator_pb2_grpc.add_GeneratorServiceServicer_to_server(LaMGenGeneratorServicer(), server)
    server.add_insecure_port("[::]:50068")
    await server.start()
    print("LaMGen Generator Service running on :50068")
    await server.wait_for_termination()


if __name__ == "__main__":
    asyncio.run(serve())
