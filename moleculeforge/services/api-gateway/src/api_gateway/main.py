"""MoleculeForge API Gateway - unified REST entry point.

Molecular descriptor calculation, design submission, and streaming updates.
The local ``MolPredictEngine`` exposes only properties computed from the
actual molecule; learned-model fields remain unavailable until a real model
service is explicitly connected.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from mf_chem.predict import MolPredictEngine, get_default_engine
from mf_core.db.store import init_db
from pydantic import BaseModel, Field

from api_gateway.auth.oidc import OIDCAuth
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
_OIDC_AUTH = OIDCAuth.from_environment()
app.state.oidc_auth = _OIDC_AUTH


async def _require_authenticated_user(request: Request) -> dict[str, Any]:
    authenticator = getattr(request.app.state, "oidc_auth", _OIDC_AUTH)
    user = await authenticator.authenticate(request)
    if user.get("anonymous"):
        raise HTTPException(status_code=401, detail="Authentication required")
    return user


AuthenticatedUser = Annotated[dict[str, Any], Depends(_require_authenticated_user)]


async def _optional_authenticated_principal(request: Request) -> str | None:
    authenticator = getattr(request.app.state, "oidc_auth", _OIDC_AUTH)
    user = await authenticator.authenticate(request)
    if user.get("anonymous"):
        return None
    principal = user.get("sub")
    if not isinstance(principal, str) or not principal.strip():
        raise HTTPException(status_code=401, detail="Authentication required")
    return principal.strip()


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
async def orchestrator_design(payload: dict[str, Any], request: Request) -> JSONResponse:
    """Proxy the full design workflow request to orchestrator-svc."""
    public_payload = dict(payload)
    public_payload.pop(design._INTERNAL_LEGACY_DESIGN_REQUEST, None)
    principal_id = None
    if public_payload.get("workflow_scope") == "full":
        authenticated_user = await _require_authenticated_user(request)
        principal_id = str(authenticated_user["sub"])
    upstream_payload, status_code = await design.orchestrator_post(
        "/v1/orchestrator/design",
        public_payload,
        principal_id=principal_id,
    )
    return JSONResponse(content=upstream_payload, status_code=status_code)


@app.get("/v1/orchestrator/runs", tags=["orchestrator"])
async def orchestrator_runs(
    authenticated_user: AuthenticatedUser,
    page_size: int = 50,
    page_token: str | None = None,
) -> JSONResponse:
    params: dict[str, object] = {"page_size": page_size}
    if page_token is not None:
        params["page_token"] = page_token
    upstream_payload, status_code = await design.orchestrator_get(
        "/v1/orchestrator/runs",
        params=params,
        principal_id=str(authenticated_user["sub"]),
    )
    return JSONResponse(content=upstream_payload, status_code=status_code)


@app.get("/v1/orchestrator/runs/{run_id}", tags=["orchestrator"])
async def orchestrator_run(
    run_id: str,
    authenticated_user: AuthenticatedUser,
) -> JSONResponse:
    upstream_payload, status_code = await design.orchestrator_get(
        f"/v1/orchestrator/runs/{run_id}",
        principal_id=str(authenticated_user["sub"]),
    )
    return JSONResponse(content=upstream_payload, status_code=status_code)


@app.get("/v1/orchestrator/runs/{run_id}/events", tags=["orchestrator"])
async def orchestrator_run_events(
    run_id: str,
    authenticated_user: AuthenticatedUser,
    after_step: int = -1,
) -> JSONResponse:
    upstream_payload, status_code = await design.orchestrator_get(
        f"/v1/orchestrator/runs/{run_id}/events",
        params={"after_step": after_step},
        principal_id=str(authenticated_user["sub"]),
    )
    return JSONResponse(content=upstream_payload, status_code=status_code)


async def _orchestrator_run_action(
    run_id: str,
    action: str,
    principal_id: str,
    payload: dict[str, Any] | None = None,
) -> JSONResponse:
    upstream_payload, status_code = await design.orchestrator_post(
        f"/v1/orchestrator/runs/{run_id}/{action}",
        payload or {},
        principal_id=principal_id,
    )
    return JSONResponse(content=upstream_payload, status_code=status_code)


@app.post("/v1/orchestrator/runs/{run_id}/pause", tags=["orchestrator"])
async def orchestrator_pause_run(
    run_id: str,
    authenticated_user: AuthenticatedUser,
) -> JSONResponse:
    return await _orchestrator_run_action(
        run_id,
        "pause",
        str(authenticated_user["sub"]),
    )


@app.post("/v1/orchestrator/runs/{run_id}/resume", tags=["orchestrator"])
async def orchestrator_resume_run(
    run_id: str,
    authenticated_user: AuthenticatedUser,
) -> JSONResponse:
    return await _orchestrator_run_action(
        run_id,
        "resume",
        str(authenticated_user["sub"]),
    )


@app.post("/v1/orchestrator/runs/{run_id}/evidence/resume", tags=["orchestrator"])
async def orchestrator_resume_run_evidence(
    run_id: str,
    payload: dict[str, Any],
    authenticated_user: AuthenticatedUser,
) -> JSONResponse:
    return await _orchestrator_run_action(
        run_id,
        "evidence/resume",
        str(authenticated_user["sub"]),
        payload,
    )


@app.post("/v1/orchestrator/runs/{run_id}/cancel", tags=["orchestrator"])
async def orchestrator_cancel_run(
    run_id: str,
    authenticated_user: AuthenticatedUser,
) -> JSONResponse:
    return await _orchestrator_run_action(
        run_id,
        "cancel",
        str(authenticated_user["sub"]),
    )


@app.get("/v1/orchestrator/{design_id}", tags=["orchestrator"])
async def orchestrator_status(
    design_id: str,
    request: Request,
) -> JSONResponse:
    """Proxy design workflow status lookup to orchestrator-svc."""
    principal_id = await _optional_authenticated_principal(request)
    upstream_payload, status_code = await design.orchestrator_get(
        f"/v1/orchestrator/runs/{design_id}",
        principal_id=principal_id,
    )
    return JSONResponse(content=upstream_payload, status_code=status_code)


@app.post("/v1/orchestrator/{design_id}/evidence/resume", tags=["orchestrator"])
async def orchestrator_resume_evidence(
    design_id: str,
    payload: dict[str, Any],
    authenticated_user: dict[str, Any] = Depends(_require_authenticated_user),
) -> JSONResponse:
    """Resume a persisted full workflow with external validation evidence."""
    upstream_payload, status_code = await design.orchestrator_post(
        f"/v1/orchestrator/runs/{design_id}/evidence/resume",
        payload,
        principal_id=str(authenticated_user["sub"]),
    )
    return JSONResponse(content=upstream_payload, status_code=status_code)


class PredictRequest(BaseModel):
    smiles: str | list[str] = Field(
        ...,
        description="Single SMILES or list of SMILES to predict",
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
_UI_DIR = Path(os.environ.get("MF_UI_DIR", str(_REPOSITORY_ROOT / "ui" / "public")))
if _UI_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(_UI_DIR), html=True), name="ui")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
