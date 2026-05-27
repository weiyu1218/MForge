"""JSON HTTP helpers for grounding tools."""
from __future__ import annotations

import asyncio
import json
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

USER_AGENT = "moleculeforge-cig-grounding/0.1"


async def get_json(
    url: str,
    params: dict[str, Any] | None = None,
    timeout: float = 20.0,
) -> tuple[dict[str, Any], str | None]:
    return await asyncio.to_thread(_request_json, "GET", url, params, None, timeout)


async def post_json(
    url: str,
    payload: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
    timeout: float = 20.0,
) -> tuple[dict[str, Any], str | None]:
    return await asyncio.to_thread(_request_json, "POST", url, params, payload, timeout)


def _request_json(
    method: str,
    url: str,
    params: dict[str, Any] | None,
    payload: dict[str, Any] | None,
    timeout: float,
) -> tuple[dict[str, Any], str | None]:
    if params:
        separator = "&" if "?" in url else "?"
        url = f"{url}{separator}{urlencode(params, doseq=True)}"

    data = None
    headers = {
        "Accept": "application/json",
        "User-Agent": USER_AGENT,
    }
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = Request(url, data=data, headers=headers, method=method)
    try:
        with urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            timestamp = response.headers.get("Last-Modified") or response.headers.get("Date")
    except Exception as exc:
        raise RuntimeError(f"grounding HTTP request failed for {url}: {exc}") from exc

    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"grounding HTTP response was not JSON for {url}") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError(f"grounding HTTP response was not a JSON object for {url}")
    return parsed, timestamp
