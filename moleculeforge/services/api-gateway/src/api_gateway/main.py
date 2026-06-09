"""MoleculeForge API Gateway - unified REST entry point.

Real end-to-end molecular property prediction + design submission + streaming
updates. All RDKit descriptors are computed from the actual molecule; the
HUMU hyperbolic embedding + ADMET heads run on every visible GPU via the
shared ``MolPredictEngine`` singleton.
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from api_gateway.auth.oidc import OIDCAuth  # noqa: F401 (kept for downstream wiring)
from api_gateway.routers import (
    design,
    molecules,
    pareto,
    projects,
    reason,
    routes,
    stream,
)

from mf_chem.predict import MolPredictEngine, get_default_engine
from mf_core.db.store import init_db


@asynccontextmanager
async def _lifespan(app: FastAPI):
    # Initialise SQLite (creates schema + seeds known-molecule catalog).
    init_db()
    # Eagerly boot the predictor so every worker hits a warm engine on 1st request.
    engine = get_default_engine()
    app.state.engine = engine
    yield


app = FastAPI(
    title="MoleculeForge API Gateway",
    version="0.1.0",
    description="Molecular Inverse Design Platform API",
    lifespan=_lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(projects.router, prefix="/v1/projects", tags=["projects"])
app.include_router(design.router, prefix="/v1/design", tags=["design"])
app.include_router(molecules.router, prefix="/v1/molecules", tags=["molecules"])
app.include_router(pareto.router, prefix="/v1/pareto", tags=["pareto"])
app.include_router(routes.router, prefix="/v1/routes", tags=["routes"])
app.include_router(stream.router, prefix="/v1/stream", tags=["stream"])
app.include_router(reason.router, prefix="/v1/reason", tags=["reason"])


def _orchestrator_base_url() -> str:
    return os.environ.get("ORCHESTRATOR_SVC_URL", "http://orchestrator-svc:8011").rstrip("/")


def _upstream_json(response: httpx.Response) -> dict[str, Any]:
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
    return payload


@app.post("/v1/orchestrator/design", tags=["orchestrator"])
async def orchestrator_design(payload: dict[str, Any]) -> dict[str, Any]:
    """Proxy the full design workflow request to orchestrator-svc."""
    url = f"{_orchestrator_base_url()}/v1/orchestrator/design"
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(url, json=payload)
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"orchestrator service unavailable: {exc}",
        ) from exc
    upstream_payload = _upstream_json(response)
    if response.status_code >= 400:
        raise HTTPException(status_code=response.status_code, detail=upstream_payload)
    return upstream_payload


@app.get("/v1/orchestrator/{design_id}", tags=["orchestrator"])
async def orchestrator_status(design_id: str) -> dict[str, Any]:
    """Proxy design workflow status lookup to orchestrator-svc."""
    url = f"{_orchestrator_base_url()}/v1/orchestrator/{design_id}"
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url)
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"orchestrator service unavailable: {exc}",
        ) from exc
    upstream_payload = _upstream_json(response)
    if response.status_code >= 400:
        raise HTTPException(status_code=response.status_code, detail=upstream_payload)
    return upstream_payload


class PredictRequest(BaseModel):
    smiles: str | list[str] = Field(
        ..., description="Single SMILES or list of SMILES to predict",
    )


@app.post("/v1/predict", tags=["predict"])
async def predict(request: PredictRequest) -> dict[str, Any]:
    """Single-call prediction endpoint (convenience wrapper)."""
    engine: MolPredictEngine = app.state.engine
    if isinstance(request.smiles, str):
        result = engine.predict_one(request.smiles)
        payload = result.to_dict()
        if not result.valid:
            raise HTTPException(status_code=400, detail=payload)
        return {"result": payload, "devices_used": engine.devices}
    results = engine.predict_batch(request.smiles)
    return {
        "results": [r.to_dict() for r in results],
        "n_total": len(results),
        "n_valid": sum(1 for r in results if r.valid),
        "devices_used": engine.devices,
    }


@app.get("/health")
async def health() -> dict[str, Any]:
    try:
        import torch
        gpu = {
            "available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }
    except Exception:
        gpu = {"available": False, "device_count": 0}
    engine = getattr(app.state, "engine", None)
    return {
        "status": "healthy",
        "gpu": gpu,
        "devices": engine.devices if engine else [],
    }


# ---------------------------------------------------------------------------
# Static frontend (mounted last so API routes shadow it).
# ---------------------------------------------------------------------------
_UI_DIR = os.environ.get("MF_UI_DIR", "/workspace/MForge/moleculeforge/ui/public")
if os.path.isdir(_UI_DIR):
    app.mount("/", StaticFiles(directory=_UI_DIR, html=True), name="ui")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
