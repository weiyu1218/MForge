"""Reasoning workbench compatibility routes backed by Orchestrator APIs."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse
from mf_core.db import store
from pydantic import BaseModel, Field

from api_gateway.routers.design import (
    orchestrator_event_stream,
    orchestrator_get,
    orchestrator_post,
)

router = APIRouter()


class RunRequest(BaseModel):
    intent: str = Field(..., min_length=4, description="Natural-language design intent.")
    workflow_scope: str = Field(..., min_length=1)
    validation_passed: bool
    max_refinements: int = Field(..., ge=0)
    project_id: str | None = None


@router.post("/runs")
async def submit_run(req: RunRequest) -> JSONResponse:
    payload = {
        "intent": req.intent,
        "workflow_scope": req.workflow_scope,
        "validation_passed": req.validation_passed,
        "max_refinements": req.max_refinements,
        "project_id": req.project_id,
    }
    response, status_code = await orchestrator_post(
        "/v1/orchestrator/design",
        payload,
    )
    return JSONResponse(content=response, status_code=status_code)


@router.get("/runs")
async def list_runs(
    page_size: int = 30,
    page_token: str | None = None,
) -> JSONResponse:
    params: dict[str, object] = {"page_size": page_size}
    if page_token is not None:
        params["page_token"] = page_token
    response, status_code = await orchestrator_get(
        "/v1/orchestrator/runs",
        params=params,
    )
    return JSONResponse(content=response, status_code=status_code)


@router.get("/runs/{run_id}")
async def get_run(run_id: str) -> JSONResponse:
    response, status_code = await orchestrator_get(f"/v1/orchestrator/runs/{run_id}")
    return JSONResponse(content=response, status_code=status_code)


@router.get("/runs/{run_id}/stream")
async def stream_run(run_id: str) -> StreamingResponse:
    snapshot, _ = await orchestrator_get(f"/v1/orchestrator/runs/{run_id}")
    return StreamingResponse(
        orchestrator_event_stream(run_id, initial_snapshot=snapshot),
        media_type="text/event-stream",
    )


@router.get("/known")
async def known_catalog(query: str = "", limit: int = 50) -> dict[str, Any]:
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
                "FROM known_molecules LIMIT ?",
                (limit,),
            ).fetchall()
    return {"items": [dict(row) for row in rows], "n": len(rows)}
