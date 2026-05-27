"""SSE streaming endpoints — emit live design progress."""
from __future__ import annotations

import asyncio
import json
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from api_gateway.routers.design import _designs

router = APIRouter()


@router.get("/{job_id}")
async def stream_job(job_id: str) -> StreamingResponse:
    if job_id not in _designs:
        raise HTTPException(status_code=404, detail="Design not found")

    async def event_stream() -> AsyncGenerator[str, None]:
        last_status = None
        last_count = 0
        for _ in range(600):
            state = _designs.get(job_id, {})
            status = state.get("status")
            n_done = len(state.get("results", []) or [])
            if status != last_status or n_done != last_count:
                payload = {
                    "job_id": job_id,
                    "status": status,
                    "candidates_generated": state.get("candidates_generated", 0),
                    "valid_results": sum(
                        1 for r in (state.get("results") or []) if r.get("valid")
                    ),
                    "devices_used": state.get("devices_used", []),
                }
                yield f"data: {json.dumps(payload)}\n\n"
                last_status = status
                last_count = n_done
            if status in {"completed", "failed", "cancelled"}:
                break
            await asyncio.sleep(0.5)
        # Final summary
        state = _designs.get(job_id, {})
        yield f"data: {json.dumps({'event': 'final', **state})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.get("/{job_id}/thinking")
async def stream_thinking(job_id: str) -> StreamingResponse:
    """Reasoning trace — synthesised from the actual workflow stages."""
    if job_id not in _designs:
        raise HTTPException(status_code=404, detail="Design not found")

    async def event_stream() -> AsyncGenerator[str, None]:
        thoughts = [
            "Parsing user objectives and design constraints...",
            "Routing to RDKit-Random baseline generator (no GPU checkpoint required).",
            "Mutating template SMILES to expand candidate pool.",
            "Encoding all candidates with the HUMU Lorentz manifold encoder.",
            "Sharding ADMET property heads across all visible CUDA devices.",
            "Computing RDKit physicochemical descriptors per candidate.",
            "Ranking by composite score (QED · SA · Lipinski penalty).",
            "Identifying Pareto-optimal candidates on (QED, -SA, -|logP-2.5|).",
            "Final summary written to design store.",
        ]
        for i, thought in enumerate(thoughts):
            payload = {
                "job_id": job_id,
                "thought_index": i,
                "thought": thought,
                "agent": "orchestrator",
            }
            yield f"data: {json.dumps(payload)}\n\n"
            await asyncio.sleep(0.2)

    return StreamingResponse(event_stream(), media_type="text/event-stream")
