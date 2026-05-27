"""EvoMol-RL Pareto Optimizer Service - gRPC server for multi-objective molecular optimization."""
import asyncio
from concurrent import futures

import grpc
from mf_core.artifacts import (
    ArtifactRequirement,
    RequirementStatus,
    check_artifact,
    require_available,
)

_REQUIREMENTS = (ArtifactRequirement("evomol_runner", "EVOMOL_RUNNER_URI", kind="uri"),)


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
        message = "EvoMol-RL runner is not configured"
    if context is not None and hasattr(context, "abort"):
        await context.abort(grpc.StatusCode.FAILED_PRECONDITION, message)
    raise RuntimeError(message)


class EvoMolRLServicer:
    async def Optimize(self, request, context):
        """Run multi-objective optimization via EvoMol-RL."""
        try:
            _require_runtime()
        except RuntimeError:
            return await _abort_unavailable(context)
        raise RuntimeError("EvoMol-RL optimizer runner is not configured")

    async def GetStatus(self, request, context):
        """Get optimization run status."""
        try:
            _require_runtime()
        except RuntimeError:
            return await _abort_unavailable(context)
        raise RuntimeError("EvoMol-RL status backend is not configured")


async def serve():
    _require_runtime()
    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=8))
    server.add_insecure_port("[::]:50064")
    await server.start()
    print("EvoMol-RL Optimizer Service running on :50064")
    await server.wait_for_termination()


if __name__ == "__main__":
    asyncio.run(serve())
