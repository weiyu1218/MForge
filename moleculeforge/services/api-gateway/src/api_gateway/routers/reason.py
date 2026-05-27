"""Reasoning workbench endpoints — power the natural-language frontend.

Routes:
  POST /v1/reason/runs           submit a NL design intent → run_id
  GET  /v1/reason/runs           list recent runs (DB)
  GET  /v1/reason/runs/{id}      full run snapshot (steps + results)
  GET  /v1/reason/runs/{id}/stream   SSE stream of reasoning steps
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncGenerator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from orchestrator.pipeline import get_pipeline
from mf_core.db import store

router = APIRouter()


class RunRequest(BaseModel):
    intent: str = Field(..., min_length=4, description="Natural-language design intent.")
    project_id: str | None = Field(default=None)


@router.post("/runs")
async def submit_run(req: RunRequest) -> dict[str, Any]:
    pl = get_pipeline()
    run_id = pl.submit(intent=req.intent, project_id=req.project_id)
    return {"run_id": run_id, "status": "queued"}


@router.get("/runs")
async def list_runs(limit: int = 30) -> dict[str, Any]:
    runs = store.list_runs(limit=limit)
    return {"runs": runs, "n": len(runs)}


@router.get("/runs/{run_id}")
async def get_run(run_id: str) -> dict[str, Any]:
    pl = get_pipeline()
    snap = pl.get(run_id)
    if not snap:
        raise HTTPException(status_code=404, detail="run not found")
    # Always pull persisted steps + results so the response is stable.
    db_run = store.get_run(run_id) or {}
    payload = {
        "run_id": snap["run_id"],
        "project_id": snap.get("project_id"),
        "intent": snap.get("intent"),
        "status": snap.get("status"),
        "created_at": snap.get("created_at"),
        "finished_at": snap.get("finished_at") or db_run.get("finished_at"),
        "objectives": snap.get("objectives"),
        "summary": db_run.get("summary"),
        "devices_used": db_run.get("devices_used") or [],
        "n_candidates": db_run.get("n_candidates", 0),
        "n_novel": db_run.get("n_novel", 0),
        "n_known": db_run.get("n_known", 0),
        "steps": store.get_reasoning_steps(run_id),
        "results": store.get_run_results(run_id),
    }
    return payload


@router.get("/runs/{run_id}/stream")
async def stream_run(run_id: str) -> StreamingResponse:
    pl = get_pipeline()
    snap = pl.get(run_id)
    if not snap:
        raise HTTPException(status_code=404, detail="run not found")

    async def event_stream() -> AsyncGenerator[str, None]:
        q = await pl.subscribe(run_id)
        # Push everything we already have
        if snap.get("status") in ("completed", "failed"):
            yield f"data: {json.dumps({'type': 'done', 'run_id': run_id})}\n\n"
            return
        try:
            while True:
                event = await asyncio.wait_for(q.get(), timeout=120.0)
                yield f"data: {json.dumps(event)}\n\n"
                if event.get("type") == "done":
                    break
        except asyncio.TimeoutError:
            yield f"data: {json.dumps({'type': 'timeout'})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.get("/known")
async def known_catalog(query: str = "", limit: int = 50) -> dict[str, Any]:
    """Look up the seed known-molecule catalog (debug/help endpoint)."""
    with store.cursor() as cur:
        if query:
            rows = cur.execute(
                "SELECT inchi_key, canonical_smiles, name, drugbank_id, indications, target "
                "FROM known_molecules WHERE name LIKE ? OR canonical_smiles LIKE ? LIMIT ?",
                (f"%{query}%", f"%{query}%", limit),
            ).fetchall()
        else:
            rows = cur.execute(
                "SELECT inchi_key, canonical_smiles, name, drugbank_id, indications, target "
                "FROM known_molecules LIMIT ?", (limit,),
            ).fetchall()
    return {"items": [dict(r) for r in rows], "n": len(rows)}
