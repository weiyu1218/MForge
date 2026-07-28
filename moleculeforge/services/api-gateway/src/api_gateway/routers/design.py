"""Molecular design endpoints backed by the Orchestrator run store."""

from __future__ import annotations

import asyncio
import json
import os
import re
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
_LEGACY_INTENT_PREFIX = "Legacy molecular design: "
_LEGACY_DESIGN_ID_PATTERN = re.compile(r"design-[0-9a-f]{10}")
_LEGACY_REQUEST_FIELDS = frozenset(
    {"objectives", "constraints", "n_samples", "seed_smiles", "seed"}
)


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
        "intent": _LEGACY_INTENT_PREFIX
        + json.dumps(intent_inputs, sort_keys=True, separators=(",", ":")),
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


def _is_legacy_design_snapshot(
    design_id: str,
    snapshot: dict[str, Any],
) -> bool:
    if (
        _LEGACY_DESIGN_ID_PATTERN.fullmatch(design_id) is None
        or snapshot.get("run_id") != design_id
    ):
        return False
    request = _snapshot_state(snapshot).get("request")
    if not isinstance(request, dict):
        return False
    intent = request.get("intent")
    return (
        isinstance(intent, str)
        and intent.startswith(_LEGACY_INTENT_PREFIX)
        and _LEGACY_REQUEST_FIELDS <= request.keys()
        and request.get("workflow_scope") == "engineering"
        and request.get("validation_passed") is True
        and request.get("max_refinements") == 0
        and "run_id" not in request
    )


def _legacy_design_status(status: object) -> object:
    return "cancelled" if status == "interrupted" else status


def _legacy_design_snapshot(
    design_id: str,
    snapshot: dict[str, Any],
) -> dict[str, Any]:
    state = _snapshot_state(snapshot)
    request = state["request"]
    result = {
        "design_id": design_id,
        "project_id": request.get("project_id") or snapshot.get("project_id") or "",
        "objectives": request["objectives"],
        "constraints": request["constraints"],
        "n_samples": request["n_samples"],
        "seed_smiles": request["seed_smiles"],
        "seed": request["seed"],
        "status": _legacy_design_status(snapshot.get("status")),
        "created_at": snapshot["created_at"],
    }
    candidates = state.get("candidates")
    if isinstance(candidates, list):
        result["candidates_generated"] = len(candidates)
    elif isinstance(snapshot.get("n_candidates"), int) and snapshot["n_candidates"] > 0:
        result["candidates_generated"] = snapshot["n_candidates"]
    validation = state.get("validation")
    has_results = "results" in state or (isinstance(validation, dict) and "results" in validation)
    if has_results:
        result["results"] = _snapshot_results(snapshot)
    devices_used = snapshot.get("devices_used", state.get("devices_used"))
    if isinstance(devices_used, list) and devices_used:
        result["devices_used"] = devices_used
    started_at = state.get("started_at")
    if isinstance(started_at, str):
        result["started_at"] = started_at
    finished_at = snapshot.get("finished_at") or state.get("finished_at")
    if isinstance(finished_at, str):
        result["finished_at"] = finished_at
    error_message = snapshot.get("error_message")
    error_type = snapshot.get("error_type")
    if isinstance(error_message, str):
        error = (
            f"{error_type}: {error_message}"
            if isinstance(error_type, str) and error_type
            else error_message
        )
    else:
        error = state.get("error")
    if result["status"] == "failed" and isinstance(error, str):
        result["error"] = error
    return result


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
    status = _legacy_design_status(snapshot.get("status"))
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
    status = _legacy_design_status(snapshot.get("status"))
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
    legacy_request = not bool(_POLICY_FIELDS & request.keys())
    payload, status_code = await orchestrator_post(
        "/v1/orchestrator/design",
        _canonical_design_request(request),
    )
    if legacy_request:
        design_id = str(payload.get("design_id") or payload["run_id"])
        return JSONResponse(
            content=_legacy_design_snapshot(design_id, payload),
            status_code=200,
        )
    return JSONResponse(content=payload, status_code=status_code)


@router.get("/{design_id}")
async def get_design(design_id: str) -> JSONResponse:
    payload, status_code = await orchestrator_get(f"/v1/orchestrator/runs/{design_id}")
    if _is_legacy_design_snapshot(design_id, payload):
        payload = _legacy_design_snapshot(design_id, payload)
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
    if _LEGACY_DESIGN_ID_PATTERN.fullmatch(design_id) is not None:
        snapshot, status_code = await orchestrator_get(f"/v1/orchestrator/runs/{design_id}")
        if _is_legacy_design_snapshot(design_id, snapshot):
            if snapshot.get("status") not in {"queued", "running"}:
                return JSONResponse(
                    content={
                        "design_id": design_id,
                        "status": _legacy_design_status(snapshot.get("status")),
                    },
                    status_code=status_code,
                )
            try:
                payload, status_code = await orchestrator_post(
                    f"/v1/orchestrator/runs/{design_id}/cancel",
                    {},
                )
            except HTTPException as exc:
                if exc.status_code != 409:
                    raise
                current, _ = await orchestrator_get(f"/v1/orchestrator/runs/{design_id}")
                if not _is_legacy_design_snapshot(design_id, current) or current.get("status") in {
                    "queued",
                    "running",
                }:
                    raise
                payload = current
                status_code = 200
            return JSONResponse(
                content={
                    "design_id": design_id,
                    "status": _legacy_design_status(payload.get("status")),
                },
                status_code=status_code,
            )
    payload, status_code = await orchestrator_post(
        f"/v1/orchestrator/runs/{design_id}/cancel",
        {},
    )
    return JSONResponse(content=payload, status_code=status_code)
