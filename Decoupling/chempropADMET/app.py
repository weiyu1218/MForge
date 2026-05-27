"""Chemprop ADMET inference microservice.

Start:  uvicorn app:app --host 0.0.0.0 --port 8901
Test:   curl -X POST http://localhost:8901/predict \
          -H 'Content-Type: application/json' \
          -d '{"smiles": ["CCO", "c1ccccc1", "CC(=O)O"]}'
"""

from __future__ import annotations

import logging
import time

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import config
from model_manager import ModelManager
from models import HealthResponse, PredictRequest, PredictResponse, MoleculeResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# App lifecycle
# ------------------------------------------------------------------

app = FastAPI(
    title="Chemprop ADMET Service",
    description="Stateless ADMET prediction microservice powered by chemprop MPNN",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

manager = ModelManager(device=config.DEVICE)


@app.on_event("startup")
async def startup() -> None:
    """Register all configured endpoints (lazy-loaded on first request)."""
    manager.register_many(config.ADMET_ENDPOINTS)
    logger.info(
        "Service ready — %d endpoints registered on %s",
        len(manager.available_endpoints),
        manager.device,
    )


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        available_endpoints=manager.available_endpoints,
        device=str(manager.device),
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    """Batch ADMET prediction.

    Accepts a list of SMILES and returns per-molecule ADMET property dicts.
    Single-molecule requests are supported but **batching is strongly
    recommended** — MPNN forward passes have high fixed overhead.
    """
    # Resolve endpoints
    available = set(manager.available_endpoints)
    if req.endpoints is not None:
        unknown = set(req.endpoints) - available
        if unknown:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown endpoints: {unknown}. Available: {available}",
            )
        endpoints = req.endpoints
    else:
        endpoints = list(available)

    if not endpoints:
        raise HTTPException(status_code=400, detail="No endpoints available")

    t0 = time.perf_counter()
    try:
        raw = manager.predict_batch(
            smiles=req.smiles,
            endpoints=endpoints,
            batch_size=min(req.batch_size, config.MAX_BATCH_SIZE),
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(exc))

    elapsed = time.perf_counter() - t0
    logger.info(
        "Predicted %d molecules × %d endpoints in %.2fs",
        len(req.smiles),
        len(endpoints),
        elapsed,
    )

    results = [
        MoleculeResult(smiles=s, predictions=p) for s, p in zip(req.smiles, raw)
    ]
    return PredictResponse(
        results=results,
        n_molecules=len(req.smiles),
        endpoints_used=endpoints,
    )


# ------------------------------------------------------------------
# Entrypoint
# ------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=config.HOST, port=config.PORT)
