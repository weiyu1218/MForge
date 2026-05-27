"""Lightweight client for the ADMET microservice.

Can be used standalone or wrapped as a LangChain @tool.

Usage:
    from client import admet_predict
    results = admet_predict(["CCO", "c1ccccc1"])
"""

from __future__ import annotations

import os
from typing import Any

import httpx

# ---------------------------------------------------------------------------
# Configurable base URL — override via env var ADMET_SERVICE_URL
# ---------------------------------------------------------------------------
BASE_URL = os.environ.get("ADMET_SERVICE_URL", "http://localhost:8901")
TIMEOUT = 120.0  # seconds — large batches can be slow on CPU


def admet_predict(
    smiles: list[str],
    endpoints: list[str] | None = None,
    batch_size: int = 64,
    base_url: str | None = None,
) -> list[dict[str, Any]]:
    """Call the ADMET service and return per-molecule prediction dicts.

    Parameters
    ----------
    smiles : list[str]
        SMILES strings to predict.
    endpoints : list[str] | None
        Specific ADMET endpoints (None = all available).
    batch_size : int
        Inference batch size passed to the service.
    base_url : str | None
        Override the default service URL.

    Returns
    -------
    list[dict]
        One dict per molecule: {"smiles": ..., "predictions": {...}}
    """
    url = (base_url or BASE_URL).rstrip("/")
    payload: dict[str, Any] = {"smiles": smiles, "batch_size": batch_size}
    if endpoints is not None:
        payload["endpoints"] = endpoints

    resp = httpx.post(f"{url}/predict", json=payload, timeout=TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    return data["results"]


def admet_health(base_url: str | None = None) -> dict[str, Any]:
    """Check service health."""
    url = (base_url or BASE_URL).rstrip("/")
    resp = httpx.get(f"{url}/health", timeout=10)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# LangChain @tool wrapper (optional — uncomment if using LangChain)
# ---------------------------------------------------------------------------
#
# from langchain_core.tools import tool
#
# @tool
# def admet_tool(smiles_csv: str) -> str:
#     """Predict ADMET properties for molecules.
#
#     Args:
#         smiles_csv: Comma-separated SMILES strings, e.g. 'CCO,c1ccccc1'
#
#     Returns:
#         JSON string of per-molecule ADMET predictions.
#     """
#     import json
#     smiles = [s.strip() for s in smiles_csv.split(",") if s.strip()]
#     results = admet_predict(smiles)
#     return json.dumps(results, indent=2)


# ---------------------------------------------------------------------------
# CLI quick-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json
    import sys

    test_smiles = sys.argv[1:] if len(sys.argv) > 1 else ["CCO", "c1ccccc1", "CC(=O)O"]
    print(f"Querying {BASE_URL}/predict with {len(test_smiles)} SMILES …")
    try:
        results = admet_predict(test_smiles)
        print(json.dumps(results, indent=2))
    except httpx.ConnectError:
        print("ERROR: Cannot connect. Is the service running?")
        print("  Start:  python app.py")
    except httpx.HTTPStatusError as exc:
        print(f"HTTP {exc.response.status_code}: {exc.response.text}")
