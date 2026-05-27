"""Docking Service - gRPC server for molecular docking (GNINA + DiffDock-L, L1 Oracle)."""
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

_GNINA_REQUIREMENT = ToolRequirement("gnina", executable="gnina", env_var="GNINA_BINARY")
_DIFFDOCK_REQUIREMENT = ArtifactRequirement("diffdock_model", "DIFFDOCK_MODEL_PATH", kind="path")
_PACKAGES = (PythonPackageRequirement("rdkit", module="rdkit"),)


def _status_objects() -> list[RequirementStatus]:
    return [
        check_tool(_GNINA_REQUIREMENT),
        check_artifact(_DIFFDOCK_REQUIREMENT),
        *(check_python_package(requirement) for requirement in _PACKAGES),
    ]


def _require_runtime(engine: str | None = None) -> list[RequirementStatus]:
    statuses = _status_objects()
    if engine == "diffdock":
        require_available([statuses[1]])
    elif engine == "gnina":
        require_available([statuses[0]])
    elif not any(status.available for status in statuses[:2]):
        require_available(statuses)
    require_available(statuses[2:])
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
        message = "Docking runner is not configured"
    if context is not None and hasattr(context, "abort"):
        await context.abort(grpc.StatusCode.FAILED_PRECONDITION, message)
    raise RuntimeError(message)


class DockServicer:
    async def Dock(self, request, context):
        """Run molecular docking for a protein-ligand pair."""
        try:
            _require_runtime(getattr(request, "engine", "gnina"))
        except RuntimeError:
            return await _abort_unavailable(context)
        docking_engine = getattr(request, "engine", "gnina")
        raise RuntimeError(f"{docking_engine} docking runner is not configured")

    async def BatchDock(self, request, context):
        """Batch docking requests."""
        results = []
        for req in getattr(request, "requests", []):
            results.append(await self.Dock(req, context))
        return type(
            "BatchDockResponse",
            (),
            {"results": results, "total_elapsed_ms": 2000},
        )()


async def serve():
    _require_runtime()
    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=10))
    server.add_insecure_port("[::]:50054")
    await server.start()
    print("Docking Service running on :50054")
    await server.wait_for_termination()


if __name__ == "__main__":
    asyncio.run(serve())
