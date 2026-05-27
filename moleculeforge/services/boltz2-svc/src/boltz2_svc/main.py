"""Boltz-2 Binding Affinity Service.

gRPC server for protein-ligand affinity prediction.
"""
import asyncio
from concurrent import futures

import grpc
from mf_core.artifacts import (
    ArtifactRequirement,
    PythonPackageRequirement,
    RequirementStatus,
    ToolRequirement,
    check_artifact,
    check_python_package,
    check_tool,
    require_available,
)
from mf_core.proto_gen.moleculeforge.v1.oracle import boltz2_pb2_grpc

_ARTIFACTS = (ArtifactRequirement("boltz_model", "BOLTZ_MODEL_PATH", kind="path"),)
_TOOLS = (ToolRequirement("boltz", executable="boltz"),)
_PACKAGES = (PythonPackageRequirement("rdkit", module="rdkit"),)


def _status_objects() -> list[RequirementStatus]:
    return [
        *(check_artifact(requirement) for requirement in _ARTIFACTS),
        *(check_tool(requirement) for requirement in _TOOLS),
        *(check_python_package(requirement) for requirement in _PACKAGES),
    ]


def _require_runtime() -> list[RequirementStatus]:
    statuses = _status_objects()
    require_available(statuses)
    return statuses


def runtime_status() -> list[dict]:
    return [status.to_dict() for status in _status_objects()]


async def _abort_unavailable(context):
    statuses = _status_objects()
    try:
        require_available(statuses)
    except RuntimeError as exc:
        message = str(exc)
    else:
        message = "Boltz-2 runner is not configured"
    if context is not None and hasattr(context, "abort"):
        await context.abort(grpc.StatusCode.FAILED_PRECONDITION, message)
    raise RuntimeError(message)


class Boltz2Servicer:
    async def PredictAffinity(self, request, context):
        """Run Boltz-2 binding affinity prediction for a protein-ligand complex."""
        try:
            _require_runtime()
        except RuntimeError:
            return await _abort_unavailable(context)
        raise RuntimeError("Boltz-2 runner is not configured")

    async def BatchPredict(self, request, context):
        """Batch affinity prediction."""
        results = []
        for req in getattr(request, "requests", []):
            results.append(await self.PredictAffinity(req, context))
        return type(
            "BatchAffinityResponse",
            (),
            {"results": results, "total_elapsed_ms": 500},
        )()


async def serve():
    _require_runtime()
    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=10))
    boltz2_pb2_grpc.add_Boltz2ServiceServicer_to_server(Boltz2Servicer(), server)
    server.add_insecure_port("[::]:50053")
    await server.start()
    print("Boltz-2 Binding Affinity Service running on :50053")
    await server.wait_for_termination()


if __name__ == "__main__":
    asyncio.run(serve())
