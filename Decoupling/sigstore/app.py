"""Sigstore signing microservice.

Start:
    uvicorn app:app --host 0.0.0.0 --port 8902

Endpoints:
    GET  /health          — service status, OIDC token presence
    POST /sign/file       — sign a file on disk
    POST /sign/bytes      — sign raw bytes (base64)
    POST /sign/json       — sign a JSON object (canonical encoding)
    POST /verify/file     — verify a file against a bundle + identity policy
    POST /verify/bytes    — verify raw bytes against a bundle + identity policy
"""

from __future__ import annotations

import base64
import logging
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import config
from sigstore_manager import SigstoreConfig, SigstoreManager
from models import (
    HealthResponse,
    SignBytesRequest,
    SignFileRequest,
    SignJsonRequest,
    SignResponse,
    VerifyBytesRequest,
    VerifyFileRequest,
    VerifyResponse,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Sigstore Signing Service",
    description="Stateless artifact signing & verification via Sigstore (Fulcio + Rekor)",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Manager (singleton)
# ---------------------------------------------------------------------------

_manager: SigstoreManager | None = None


def get_manager() -> SigstoreManager:
    global _manager
    if _manager is None:
        cfg = SigstoreConfig(
            env=config.SIGSTORE_ENV,
            oidc_strategy=config.OIDC_STRATEGY,
            bundle_dir=config.BUNDLE_DIR,
        )
        _manager = SigstoreManager(config=cfg)
    return _manager


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        sigstore_env=config.SIGSTORE_ENV,
        oidc_strategy=config.OIDC_STRATEGY,
        oidc_token_present=bool(os.environ.get("SIGSTORE_ID_TOKEN")),
    )


@app.post("/sign/file", response_model=SignResponse)
async def sign_file(req: SignFileRequest):
    mgr = get_manager()
    try:
        result = mgr.sign_file(req.file_path, req.bundle_path)
        return SignResponse(
            success=True,
            artifact_path=result.artifact_path,
            bundle_path=result.bundle_path,
            digest_hex=result.digest_hex,
            rekor_log_index=result.rekor_log_index,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.exception("sign/file failed")
        return SignResponse(
            success=False,
            artifact_path=req.file_path,
            bundle_path="",
            digest_hex="",
            error=str(exc),
        )


@app.post("/sign/bytes", response_model=SignResponse)
async def sign_bytes(req: SignBytesRequest):
    mgr = get_manager()
    try:
        data = base64.b64decode(req.data_base64)
        result = mgr.sign_bytes(data, req.bundle_path)
        return SignResponse(
            success=True,
            artifact_path=result.artifact_path,
            bundle_path=result.bundle_path,
            digest_hex=result.digest_hex,
            rekor_log_index=result.rekor_log_index,
        )
    except Exception as exc:
        logger.exception("sign/bytes failed")
        return SignResponse(
            success=False,
            artifact_path="<bytes>",
            bundle_path=req.bundle_path,
            digest_hex="",
            error=str(exc),
        )


@app.post("/sign/json", response_model=SignResponse)
async def sign_json(req: SignJsonRequest):
    mgr = get_manager()
    try:
        result = mgr.sign_json(req.data, req.bundle_path)
        return SignResponse(
            success=True,
            artifact_path=result.artifact_path,
            bundle_path=result.bundle_path,
            digest_hex=result.digest_hex,
            rekor_log_index=result.rekor_log_index,
        )
    except Exception as exc:
        logger.exception("sign/json failed")
        return SignResponse(
            success=False,
            artifact_path="<json>",
            bundle_path=req.bundle_path,
            digest_hex="",
            error=str(exc),
        )


@app.post("/verify/file", response_model=VerifyResponse)
async def verify_file(req: VerifyFileRequest):
    mgr = get_manager()
    try:
        result = mgr.verify_file(
            req.file_path, req.bundle_path, req.identity, req.issuer,
        )
        return VerifyResponse(
            valid=result.valid,
            artifact_path=result.artifact_path,
            bundle_path=result.bundle_path,
            identity=result.identity,
            issuer=result.issuer,
            rekor_log_index=result.rekor_log_index,
            error=result.error,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.post("/verify/bytes", response_model=VerifyResponse)
async def verify_bytes(req: VerifyBytesRequest):
    mgr = get_manager()
    try:
        data = base64.b64decode(req.data_base64)
        result = mgr.verify_bytes(data, req.bundle_path, req.identity, req.issuer)
        return VerifyResponse(
            valid=result.valid,
            artifact_path=result.artifact_path,
            bundle_path=result.bundle_path,
            identity=result.identity,
            issuer=result.issuer,
            rekor_log_index=result.rekor_log_index,
            error=result.error,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.HOST, port=config.PORT)
