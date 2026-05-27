"""Molecular design lifecycle endpoints.

Real wiring:
- POST /v1/design/  -> spawn a design job that scores caller-provided seed
  molecules through the shared ``MolPredictEngine``.
- GET  /v1/design/{id}                -> job snapshot
- GET  /v1/design/{id}/results        -> Pareto-style ranked candidates
- POST /v1/design/{id}/cancel         -> mark a job as cancelled
"""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException
from mf_chem.predict import get_default_engine
from pydantic import BaseModel, Field

router = APIRouter()

_designs: dict[str, dict] = {}


class DesignRequest(BaseModel):
    project_id: str = ""
    objectives: list[str] = Field(default_factory=lambda: ["qed", "sa_score", "logp"])
    constraints: dict[str, Any] = Field(default_factory=dict)
    n_samples: int = Field(default=64, ge=1, le=2048)
    seed_smiles: list[str] = Field(min_length=1)
    seed: int | None = None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _generate_seed_pool(seed: int | None, count: int, custom: list[str]) -> list[str]:
    """Select deterministic seed molecules for local API scoring."""
    base = list(custom)
    if not base:
        raise ValueError("seed_smiles must not be empty")
    offset = abs(int(seed or 0)) % len(base)
    return [base[(i + offset) % len(base)] for i in range(count)]


def _rank_pareto(scored: list[dict]) -> list[dict]:
    """Sort by composite score; flag the non-dominated front on (qed, -sa, -logp)."""
    valid = [s for s in scored if s.get("valid")]
    if not valid:
        return []
    objectives = [
        (s["qed"] or 0.0, -(s["sa_score"] or 10.0), -abs((s["logp"] or 5.0) - 2.5))
        for s in valid
    ]
    fronts = [True] * len(valid)
    for i, oi in enumerate(objectives):
        for j, oj in enumerate(objectives):
            if i == j:
                continue
            if all(oj[k] >= oi[k] for k in range(len(oi))) and any(
                oj[k] > oi[k] for k in range(len(oi))
            ):
                fronts[i] = False
                break
    valid.sort(key=lambda s: -(s.get("composite_score") or 0.0))
    out = []
    for rank, s in enumerate(valid, start=1):
        idx = scored.index(s)
        s_out = dict(s)
        s_out["rank"] = rank
        s_out["pareto_optimal"] = bool(fronts[scored.index(s)] if scored[idx] is s else False)
        out.append(s_out)
    return out


async def _run_design(design_id: str) -> None:
    state = _designs.get(design_id)
    if state is None:
        return
    state["status"] = "running"
    state["started_at"] = _now()
    try:
        seed_pool = _generate_seed_pool(
            state.get("seed"), state["n_samples"], state.get("seed_smiles"),
        )
        state["candidates_generated"] = len(seed_pool)
        engine = get_default_engine()
        # `predict_batch` is sync; offload to a thread so we don't block the loop.
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None, engine.predict_batch, seed_pool,
        )
        scored = [r.to_dict() for r in results]
        state["results"] = _rank_pareto(scored)
        state["devices_used"] = engine.devices
        state["status"] = "completed"
        state["finished_at"] = _now()
    except Exception as e:  # noqa: BLE001
        state["status"] = "failed"
        state["error"] = f"{type(e).__name__}: {e}"


@router.post("/")
async def create_design(request: DesignRequest, background: BackgroundTasks) -> dict[str, Any]:
    design_id = f"design-{uuid.uuid4().hex[:10]}"
    _designs[design_id] = {
        "design_id": design_id,
        "project_id": request.project_id,
        "objectives": request.objectives,
        "constraints": request.constraints,
        "n_samples": request.n_samples,
        "seed_smiles": request.seed_smiles,
        "seed": request.seed,
        "status": "queued",
        "created_at": _now(),
    }
    background.add_task(_run_design, design_id)
    return _designs[design_id]


@router.get("/{design_id}")
async def get_design(design_id: str) -> dict[str, Any]:
    state = _designs.get(design_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Design not found")
    return state


@router.get("/{design_id}/status")
async def get_design_status(design_id: str) -> dict[str, Any]:
    state = _designs.get(design_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Design not found")
    candidates = state.get("candidates_generated", 0)
    results = state.get("results", []) or []
    n_valid = len([r for r in results if r.get("valid")])
    if state["status"] == "completed":
        progress = 100.0
    elif state["status"] == "running":
        progress = 60.0 if candidates else 20.0
    elif state["status"] == "failed":
        progress = 0.0
    else:
        progress = 5.0
    return {
        "design_id": design_id,
        "status": state["status"],
        "progress_pct": progress,
        "current_stage": state["status"],
        "candidates_generated": candidates,
        "valid_results": n_valid,
        "devices_used": state.get("devices_used", []),
    }


@router.get("/{design_id}/results")
async def get_design_results(design_id: str) -> dict[str, Any]:
    state = _designs.get(design_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Design not found")
    if state["status"] != "completed":
        return {"design_id": design_id, "status": state["status"], "results": []}
    return {
        "design_id": design_id,
        "status": state["status"],
        "results": state.get("results", []),
        "n_results": len(state.get("results", [])),
        "objectives": state.get("objectives", []),
        "devices_used": state.get("devices_used", []),
    }


@router.post("/{design_id}/cancel")
async def cancel_design(design_id: str) -> dict[str, Any]:
    state = _designs.get(design_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Design not found")
    if state["status"] in {"queued", "running"}:
        state["status"] = "cancelled"
        state["finished_at"] = _now()
    return {"design_id": design_id, "status": state["status"]}
