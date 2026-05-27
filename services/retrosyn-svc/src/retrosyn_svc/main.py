"""Retrosynthesis Planning Service - gRPC server for AiZynthFinder + RSGPT scoring."""
import asyncio
import time
import uuid
from concurrent import futures

import grpc
from mf_core.artifacts import (
    ArtifactRequirement,
    RequirementStatus,
    check_artifact,
    require_available,
)
from mf_core.proto_gen.moleculeforge.v1.retrosyn import retrosyn_pb2, retrosyn_pb2_grpc

_RUNNER = ArtifactRequirement("retrosyn_runner", "RETROSYN_RUNNER_URI", kind="uri")
_SCORER = ArtifactRequirement("retrosyn_scorer", "RETROSYN_SCORER_URI", kind="uri")
_AIZYNTH_CONFIG = ArtifactRequirement("aizynth_config", "AIZYNTH_CONFIG_PATH", kind="file")


def _status_objects() -> list[RequirementStatus]:
    return [check_artifact(_RUNNER), check_artifact(_SCORER), check_artifact(_AIZYNTH_CONFIG)]


def _require_runtime(*requirements: ArtifactRequirement) -> list[RequirementStatus]:
    statuses = [check_artifact(requirement) for requirement in requirements]
    require_available(statuses)
    return statuses


def runtime_status() -> list[dict]:
    return [status.to_dict() for status in _status_objects()]


async def _abort_unavailable(context, *requirements: ArtifactRequirement):
    statuses = [check_artifact(requirement) for requirement in requirements]
    try:
        require_available(statuses)
    except RuntimeError as exc:
        message = str(exc)
    else:
        message = "Retrosynthesis backend is not configured"
    if context is not None and hasattr(context, "abort"):
        await context.abort(grpc.StatusCode.FAILED_PRECONDITION, message)
    raise RuntimeError(message)


class RetrosynServicer:
    def __init__(self, planner=None):
        self.planner = planner

    async def FindRoutes(self, request, context):  # noqa: N802
        """Plan retrosynthetic routes for a target molecule."""
        smiles = getattr(request, "molecule_smiles", "")
        if not isinstance(smiles, str) or not smiles:
            message = "molecule_smiles is required"
            if context is not None and hasattr(context, "abort"):
                await context.abort(grpc.StatusCode.INVALID_ARGUMENT, message)
            raise ValueError(message)
        max_routes = int(getattr(request, "max_routes", 0) or 10)
        start = time.perf_counter()
        try:
            planner = self._planner(getattr(request, "engine", "aizynth"))
            routes = await _maybe_await(planner.find_routes(smiles, max_routes=max_routes))
        except RuntimeError as exc:
            return await _abort_message(context, str(exc))
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        return retrosyn_pb2.RetrosynthesisResponse(
            request_id=f"retrosyn-{uuid.uuid4().hex[:12]}",
            routes=[_synthetic_route(route) for route in routes],
            total_routes_found=len(routes),
            elapsed_ms=elapsed_ms,
        )

    async def FindRoutesStream(self, request_iterator, context):  # noqa: N802
        async for request in request_iterator:
            yield await self.FindRoutes(request, context)

    async def PlanRoutes(self, request, context):  # noqa: N802
        return await self.FindRoutes(request, context)

    async def ScoreRoute(self, request, context):  # noqa: N802
        """Score a specific synthetic route using RSGPT."""
        try:
            _require_runtime(_SCORER)
        except RuntimeError:
            return await _abort_unavailable(context, _SCORER)
        raise RuntimeError("Retrosynthesis route scorer is not configured")

    def _planner(self, engine: str):
        if self.planner is not None:
            return self.planner
        key = (engine or "aizynth").strip().lower()
        if key not in {"aizynth", "aizynthfinder"}:
            raise RuntimeError(f"Unsupported retrosynthesis engine: {engine}")
        from mf_retrosyn.aizynth.retrosyn import AiZynthRetrosyn

        self.planner = AiZynthRetrosyn.from_env()
        return self.planner


async def _maybe_await(value):
    if asyncio.iscoroutine(value):
        return await value
    return value


async def _abort_message(context, message: str):
    if context is not None and hasattr(context, "abort"):
        await context.abort(grpc.StatusCode.FAILED_PRECONDITION, message)
    raise RuntimeError(message)


def _synthetic_route(route: dict):
    steps = route.get("steps") if isinstance(route.get("steps"), list) else []
    reaction_smiles = route.get("reaction_smiles")
    if not isinstance(reaction_smiles, list):
        reaction_smiles = [
            str(step["reaction"])
            for step in steps
            if isinstance(step, dict) and isinstance(step.get("reaction"), str)
        ]
    building_blocks = _building_blocks(route, steps)
    return retrosyn_pb2.SyntheticRoute(
        route_id=str(route.get("route_id", "")),
        reaction_smiles=reaction_smiles,
        predicted_score=float(route.get("predicted_score", route.get("score", 0.0)) or 0.0),
        predicted_yield=float(route.get("predicted_yield", route.get("yield", 0.0)) or 0.0),
        n_steps=int(route.get("n_steps", len(steps)) or len(steps)),
        building_blocks=building_blocks,
        estimated_cost_usd_per_g=float(route.get("estimated_cost_usd_per_g", 0.0) or 0.0),
        all_commercially_available=bool(
            route.get("all_commercially_available", bool(building_blocks))
        ),
    )


def _building_blocks(route: dict, steps: list[dict]) -> list[str]:
    direct = route.get("building_blocks")
    if isinstance(direct, list) and direct:
        return [_building_block_smiles(item) for item in direct]
    blocks: list[str] = []
    for step in steps:
        for block in step.get("building_blocks") or []:
            smiles = _building_block_smiles(block)
            if smiles and smiles not in blocks:
                blocks.append(smiles)
    return blocks


def _building_block_smiles(block) -> str:
    if isinstance(block, dict):
        return str(block.get("smiles", ""))
    return str(block)


async def serve():
    _require_runtime(_AIZYNTH_CONFIG)
    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=8))
    retrosyn_pb2_grpc.add_RetrosynServiceServicer_to_server(RetrosynServicer(), server)
    server.add_insecure_port("[::]:50057")
    await server.start()
    print("Retrosynthesis Service running on :50057")
    await server.wait_for_termination()


if __name__ == "__main__":
    asyncio.run(serve())
