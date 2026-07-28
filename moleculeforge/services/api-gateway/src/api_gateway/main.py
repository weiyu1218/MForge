"""MoleculeForge API Gateway - unified REST entry point.

Real end-to-end molecular property prediction + design submission + streaming
updates. All RDKit descriptors are computed from the actual molecule; the
HUMU hyperbolic embedding + ADMET heads run on every visible GPU via the
shared ``MolPredictEngine`` singleton.
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from mf_chem.predict import MolPredictEngine, get_default_engine
from mf_core.db.store import init_db
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


@app.post("/v1/orchestrator/design", tags=["orchestrator"])
async def orchestrator_design(payload: dict[str, Any]) -> JSONResponse:
    """Proxy the full design workflow request to orchestrator-svc."""
    upstream_payload, status_code = await design.orchestrator_post(
        "/v1/orchestrator/design",
        payload,
    )
    return JSONResponse(content=upstream_payload, status_code=status_code)


@app.get("/v1/orchestrator/{design_id}", tags=["orchestrator"])
async def orchestrator_status(design_id: str) -> JSONResponse:
    """Proxy design workflow status lookup to orchestrator-svc."""
    upstream_payload, status_code = await design.orchestrator_get(
        f"/v1/orchestrator/runs/{design_id}"
    )
    return JSONResponse(content=upstream_payload, status_code=status_code)


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
_REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
_UI_DIR = Path(
    os.environ.get("MF_UI_DIR", str(_REPOSITORY_ROOT / "ui" / "public"))
)
if _UI_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(_UI_DIR), html=True), name="ui")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
