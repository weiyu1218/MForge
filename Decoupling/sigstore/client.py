"""Lightweight HTTP client for the Sigstore signing service.

Usage:
    from client import SigstoreClient

    client = SigstoreClient("http://localhost:8902")
    result = client.sign_file("/data/dataset.csv")
    ok = client.verify_file(
        "/data/dataset.csv",
        result["bundle_path"],
        identity="https://github.com/org/repo/...",
        issuer="https://token.actions.githubusercontent.com",
    )
"""

from __future__ import annotations

import base64
import json
import os
from typing import Any

import httpx

BASE_URL = os.environ.get("SIGSTORE_SERVICE_URL", "http://localhost:8902")
TIMEOUT = 60.0


class SigstoreClient:
    """Typed client for the Sigstore signing microservice."""

    def __init__(self, base_url: str | None = None, token: str | None = None):
        self.base_url = (base_url or BASE_URL).rstrip("/")
        headers: dict[str, str] = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self._http = httpx.Client(
            base_url=self.base_url,
            headers=headers,
            timeout=TIMEOUT,
        )

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def health(self) -> dict[str, Any]:
        resp = self._http.get("/health")
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # Sign
    # ------------------------------------------------------------------

    def sign_file(self, file_path: str, bundle_path: str | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {"file_path": file_path}
        if bundle_path:
            payload["bundle_path"] = bundle_path
        resp = self._http.post("/sign/file", json=payload)
        resp.raise_for_status()
        return resp.json()

    def sign_bytes(self, data: bytes, bundle_path: str) -> dict[str, Any]:
        resp = self._http.post(
            "/sign/bytes",
            json={
                "data_base64": base64.b64encode(data).decode(),
                "bundle_path": bundle_path,
            },
        )
        resp.raise_for_status()
        return resp.json()

    def sign_json(self, obj: Any, bundle_path: str) -> dict[str, Any]:
        resp = self._http.post(
            "/sign/json",
            json={"data": obj, "bundle_path": bundle_path},
        )
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # Verify
    # ------------------------------------------------------------------

    def verify_file(
        self,
        file_path: str,
        bundle_path: str,
        identity: str,
        issuer: str,
    ) -> dict[str, Any]:
        resp = self._http.post(
            "/verify/file",
            json={
                "file_path": file_path,
                "bundle_path": bundle_path,
                "identity": identity,
                "issuer": issuer,
            },
        )
        resp.raise_for_status()
        return resp.json()

    def verify_bytes(
        self,
        data: bytes,
        bundle_path: str,
        identity: str,
        issuer: str,
    ) -> dict[str, Any]:
        resp = self._http.post(
            "/verify/bytes",
            json={
                "data_base64": base64.b64encode(data).decode(),
                "bundle_path": bundle_path,
                "identity": identity,
                "issuer": issuer,
            },
        )
        resp.raise_for_status()
        return resp.json()


# ---------------------------------------------------------------------------
# CLI quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    url = sys.argv[1] if len(sys.argv) > 1 else BASE_URL
    c = SigstoreClient(url)

    print("Health check:")
    try:
        print(json.dumps(c.health(), indent=2))
    except httpx.ConnectError:
        print("ERROR: Cannot connect. Is the service running?")
        print(f"  Start:  python app.py")
        sys.exit(1)
