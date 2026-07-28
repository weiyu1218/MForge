"""Molecular design endpoints backed by the Orchestrator run store."""
from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncGenerator
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

router = APIRouter()


def _orchestrator_base_url() -> str:
    return os.environ.get(
        "ORCHESTRATOR_SVC_URL",
        "http://orchestrator-svc:8011",
    ).rstrip("/")


async def orchestrator_get(
    path: str,
    *,
    params: dict[str, object] | None = None,
) -> tuple[dict[str, Any], int]:
    url = f"{_orchestrator_base_url()}{path}"
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url, params=params)
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"orchestrator service unavailable: {exc}",
        ) from exc
    return _decode_upstream(response)


async def orchestrator_post(
    path: str,
    payload: dict[str, Any],
) -> tuple[dict[str, Any], int]:
    url = f"{_orchestrator_base_url()}{path}"
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(url, json=payload)
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"orchestrator service unavailable: {exc}",
        ) from exc
    return _decode_upstream(response)


def _decode_upstream(response: httpx.Response) -> tuple[dict[str, Any], int]:
    try:
        payload = response.json()
    except ValueError as exc:
        raise HTTPException(
            status_code=502,
            detail="orchestrator service returned non-JSON response",
        ) from exc
    if not isinstance(payload, dict):
        raise HTTPException(
            status_code=502,
            detail="orchestrator service returned invalid response",
        )
    if response.status_code >= 400:
        raise HTTPException(
            status_code=response.status_code,
            detail=payload.get("detail", payload),
        )
    return payload, response.status_code


async def orchestrator_event_stream(
    run_id: str,
    *,
    initial_snapshot: dict[str, Any],
) -> AsyncGenerator[str, None]:
    after_step = -1
    snapshot = initial_snapshot
    while True:
        event_page, _ = await orchestrator_get(
            f"/v1/orchestrator/runs/{run_id}/events",
            params={"after_step": after_step},
        )
        for event in event_page.get("events", []):
            after_step = int(event["step_index"])
            yield f"data: {json.dumps(event)}\n\n"
        if snapshot.get("status") in {
            "completed",
            "rejected",
            "failed",
            "interrupted",
        }:
            done = {
                "type": "done",
                "run_id": run_id,
                "status": snapshot["status"],
            }
            if snapshot["status"] in {"failed", "interrupted"}:
                done["error_type"] = snapshot.get("error_type")
                done["error_message"] = snapshot.get("error_message")
            yield f"data: {json.dumps(done)}\n\n"
            return
        await asyncio.sleep(0.5)
        snapshot, _ = await orchestrator_get(
            f"/v1/orchestrator/runs/{run_id}"
        )


@router.post("/")
async def create_design(request: dict[str, Any]) -> JSONResponse:
    payload, status_code = await orchestrator_post(
        "/v1/orchestrator/design",
        request,
    )
    return JSONResponse(content=payload, status_code=status_code)


@router.get("/{design_id}")
async def get_design(design_id: str) -> JSONResponse:
    payload, status_code = await orchestrator_get(
        f"/v1/orchestrator/runs/{design_id}"
    )
    return JSONResponse(content=payload, status_code=status_code)


@router.get("/{design_id}/status")
async def get_design_status(design_id: str) -> JSONResponse:
    payload, status_code = await orchestrator_get(
        f"/v1/orchestrator/runs/{design_id}"
    )
    return JSONResponse(content=payload, status_code=status_code)


@router.get("/{design_id}/results")
async def get_design_results(design_id: str) -> JSONResponse:
    payload, status_code = await orchestrator_get(
        f"/v1/orchestrator/runs/{design_id}"
    )
    return JSONResponse(content=payload, status_code=status_code)


@router.post("/{design_id}/cancel")
async def cancel_design(design_id: str) -> None:
    raise HTTPException(
        status_code=405,
        detail=f"run cancellation is not supported: {design_id}",
    )
