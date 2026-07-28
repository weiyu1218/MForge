"""SSE streaming endpoints backed by persisted Orchestrator events."""
from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from api_gateway.routers.design import orchestrator_event_stream, orchestrator_get

router = APIRouter()


@router.get("/{job_id}")
async def stream_job(job_id: str) -> StreamingResponse:
    snapshot, _ = await orchestrator_get(f"/v1/orchestrator/runs/{job_id}")
    return StreamingResponse(
        orchestrator_event_stream(job_id, initial_snapshot=snapshot),
        media_type="text/event-stream",
    )


@router.get("/{job_id}/thinking")
async def stream_thinking(job_id: str) -> StreamingResponse:
    return await stream_job(job_id)
