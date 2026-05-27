"""FEP Service.

gRPC server for OpenFE RBFE calculations.
"""
import asyncio
from concurrent import futures

import grpc
from mf_core.artifacts import (
    PythonPackageRequirement,
    RequirementStatus,
    ToolRequirement,
    check_python_package,
    check_tool,
    require_available,
)
from mf_core.proto_gen.moleculeforge.v1.oracle import fep_pb2_grpc

_TOOLS = (ToolRequirement("openfe_runner", executable="openfe", env_var="OPENFE_RUNNER_PATH"),)
_PACKAGES = (PythonPackageRequirement("openfe", module="openfe"),)


def _status_objects() -> list[RequirementStatus]:
    return [
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
        message = "OpenFE runner is not configured"
    if context is not None and hasattr(context, "abort"):
        await context.abort(grpc.StatusCode.FAILED_PRECONDITION, message)
    raise RuntimeError(message)


class FEPServicer:
    async def RunFEP(self, request, context):
        """Run Free Energy Perturbation calculation."""
        try:
            _require_runtime()
        except RuntimeError:
            return await _abort_unavailable(context)
        raise RuntimeError("OpenFE runner is not configured")

    async def GetStatus(self, request, context):
        """Get status of a running FEP job."""
        try:
            _require_runtime()
        except RuntimeError:
            return await _abort_unavailable(context)
        raise RuntimeError("OpenFE job status backend is not configured")


async def serve():
    _require_runtime()
    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=8))
    fep_pb2_grpc.add_FEPServiceServicer_to_server(FEPServicer(), server)
    server.add_insecure_port("[::]:50055")
    await server.start()
    print("FEP Service running on :50055")
    await server.wait_for_termination()


if __name__ == "__main__":
    asyncio.run(serve())
