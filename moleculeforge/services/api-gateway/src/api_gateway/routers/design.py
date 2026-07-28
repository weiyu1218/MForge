"""Molecular design endpoints backed by the Orchestrator run store."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncGenerator
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError

router = APIRouter()

_POLICY_FIELDS = frozenset({"workflow_scope", "validation_passed", "max_refinements"})
_INTERNAL_LEGACY_DESIGN_REQUEST = "_mforge_internal_legacy_design_request"


class DesignRequest(BaseModel):
    project_id: str = ""
    objectives: list[str] = Field(default_factory=lambda: ["qed", "sa_score", "logp"])
    constraints: dict[str, Any] = Field(default_factory=dict)
    n_samples: int = Field(default=64, ge=1, le=2048)
    seed_smiles: list[str] = Field(min_length=1)
    seed: int | None = None


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


async def orchestrator_delete(path: str) -> tuple[dict[str, Any], int]:
    url = f"{_orchestrator_base_url()}{path}"
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.delete(url)
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


def _canonical_design_request(request: dict[str, Any]) -> dict[str, Any]:
    if _POLICY_FIELDS & request.keys():
        canonical_request = dict(request)
        canonical_request.pop(_INTERNAL_LEGACY_DESIGN_REQUEST, None)
        return canonical_request
    try:
        legacy_request = DesignRequest.model_validate(request).model_dump()
    except ValidationError as exc:
        errors = [
            {
                **error,
                "loc": ("body", *error["loc"]),
            }
            for error in exc.errors(include_url=False)
        ]
        raise RequestValidationError(errors, body=request) from exc
    canonical_request = dict(legacy_request)
    if canonical_request["project_id"] == "":
        canonical_request.pop("project_id")
    intent_inputs = {
        "constraints": legacy_request["constraints"],
        "objectives": legacy_request["objectives"],
    }
    return {
        **canonical_request,
        "intent": (
            "Legacy molecular design: "
            f"{json.dumps(intent_inputs, sort_keys=True, separators=(',', ':'))}"
        ),
        "workflow_scope": "engineering",
        "validation_passed": True,
        "max_refinements": 0,
        _INTERNAL_LEGACY_DESIGN_REQUEST: True,
    }


def _snapshot_state(snapshot: dict[str, Any]) -> dict[str, Any]:
    state = snapshot.get("state")
    return state if isinstance(state, dict) else {}


def _snapshot_results(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    state = _snapshot_state(snapshot)
    results = state.get("results")
    if isinstance(results, list):
        return [row for row in results if isinstance(row, dict)]
    validation = state.get("validation")
    if isinstance(validation, dict):
        results = validation.get("results")
        if isinstance(results, list):
            return [row for row in results if isinstance(row, dict)]
    return []


def _legacy_status_snapshot(
    design_id: str,
    snapshot: dict[str, Any],
) -> dict[str, Any]:
    state = _snapshot_state(snapshot)
    candidates = state.get("candidates")
    if isinstance(candidates, list):
        candidates_generated = len(candidates)
    else:
        persisted_count = snapshot.get("n_candidates", 0)
        candidates_generated = (
            persisted_count
            if isinstance(persisted_count, int) and not isinstance(persisted_count, bool)
            else 0
        )
    results = _snapshot_results(snapshot)
    status = snapshot.get("status")
    if status == "queued":
        progress = 5.0
    elif status == "running":
        progress = 60.0 if candidates_generated else 20.0
    elif status == "completed":
        progress = 100.0
    elif status == "failed":
        progress = 0.0
    else:
        progress = 5.0
    devices_used = snapshot.get("devices_used", state.get("devices_used", []))
    if not isinstance(devices_used, list):
        devices_used = []
    return {
        "design_id": design_id,
        "status": status,
        "progress_pct": progress,
        "current_stage": status,
        "candidates_generated": candidates_generated,
        "valid_results": sum(1 for result in results if result.get("valid")),
        "devices_used": devices_used,
    }


def _legacy_results_snapshot(
    design_id: str,
    snapshot: dict[str, Any],
) -> dict[str, Any]:
    state = _snapshot_state(snapshot)
    request = state.get("request")
    objectives = state.get("objectives")
    if not isinstance(objectives, list) and isinstance(request, dict):
        objectives = request.get("objectives")
    if not isinstance(objectives, list):
        objectives = []
    status = snapshot.get("status")
    if status != "completed":
        return {
            "design_id": design_id,
            "status": status,
            "results": [],
        }
    results = _snapshot_results(snapshot)
    devices_used = snapshot.get("devices_used", state.get("devices_used", []))
    if not isinstance(devices_used, list):
        devices_used = []
    return {
        "design_id": design_id,
        "status": status,
        "results": results,
        "n_results": len(results),
        "objectives": objectives,
        "devices_used": devices_used,
    }


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
        snapshot, _ = await orchestrator_get(f"/v1/orchestrator/runs/{run_id}")


@router.post("/")
async def create_design(request: dict[str, Any]) -> JSONResponse:
    payload, status_code = await orchestrator_post(
        "/v1/orchestrator/design",
        _canonical_design_request(request),
    )
    return JSONResponse(content=payload, status_code=status_code)


@router.get("/{design_id}")
async def get_design(design_id: str) -> JSONResponse:
    payload, status_code = await orchestrator_get(f"/v1/orchestrator/runs/{design_id}")
    return JSONResponse(content=payload, status_code=status_code)


@router.get("/{design_id}/status")
async def get_design_status(design_id: str) -> JSONResponse:
    payload, status_code = await orchestrator_get(f"/v1/orchestrator/runs/{design_id}")
    return JSONResponse(
        content=_legacy_status_snapshot(design_id, payload),
        status_code=status_code,
    )


@router.get("/{design_id}/results")
async def get_design_results(design_id: str) -> JSONResponse:
    payload, status_code = await orchestrator_get(f"/v1/orchestrator/runs/{design_id}")
    return JSONResponse(
        content=_legacy_results_snapshot(design_id, payload),
        status_code=status_code,
    )


@router.post("/{design_id}/cancel")
async def cancel_design(design_id: str) -> JSONResponse:
    payload, status_code = await orchestrator_post(
        f"/v1/orchestrator/runs/{design_id}/cancel",
        {},
    )
    return JSONResponse(content=payload, status_code=status_code)
