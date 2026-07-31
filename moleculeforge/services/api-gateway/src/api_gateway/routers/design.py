"""HTTP client helpers for the Orchestrator service."""

from __future__ import annotations

import os
from typing import Any

import httpx
from fastapi import HTTPException

_SERVICE_TOKEN_HEADER = "X-MoleculeForge-Service-Token"
_PRINCIPAL_HEADER = "X-MoleculeForge-Principal"


def _orchestrator_base_url() -> str:
    return os.environ.get(
        "ORCHESTRATOR_SVC_URL",
        "http://orchestrator-svc:8011",
    ).rstrip("/")


async def orchestrator_get(
    path: str,
    *,
    params: dict[str, object] | None = None,
    principal_id: str | None = None,
) -> tuple[dict[str, Any], int]:
    url = f"{_orchestrator_base_url()}{path}"
    headers = _orchestrator_headers(principal_id)
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            request_kwargs: dict[str, object] = {"params": params}
            if headers:
                request_kwargs["headers"] = headers
            response = await client.get(url, **request_kwargs)
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"orchestrator service unavailable: {exc}",
        ) from exc
    return _decode_upstream(response)


async def orchestrator_post(
    path: str,
    payload: dict[str, Any],
    *,
    principal_id: str | None = None,
) -> tuple[dict[str, Any], int]:
    url = f"{_orchestrator_base_url()}{path}"
    headers = _orchestrator_headers(principal_id)
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            request_kwargs: dict[str, object] = {"json": payload}
            if headers:
                request_kwargs["headers"] = headers
            response = await client.post(url, **request_kwargs)
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"orchestrator service unavailable: {exc}",
        ) from exc
    return _decode_upstream(response)


async def orchestrator_delete(
    path: str,
    *,
    principal_id: str | None = None,
) -> tuple[dict[str, Any], int]:
    url = f"{_orchestrator_base_url()}{path}"
    headers = _orchestrator_headers(principal_id)
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            if headers:
                response = await client.delete(url, headers=headers)
            else:
                response = await client.delete(url)
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"orchestrator service unavailable: {exc}",
        ) from exc
    return _decode_upstream(response)


def _orchestrator_headers(principal_id: str | None) -> dict[str, str]:
    headers: dict[str, str] = {}
    service_token = os.environ.get("INTERNAL_SERVICE_TOKEN", "").strip()
    if service_token:
        headers[_SERVICE_TOKEN_HEADER] = service_token
    if principal_id is not None and service_token:
        normalized_principal = principal_id.strip()
        if not normalized_principal:
            raise HTTPException(status_code=401, detail="Authenticated principal is required")
        headers[_PRINCIPAL_HEADER] = normalized_principal
    return headers


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
