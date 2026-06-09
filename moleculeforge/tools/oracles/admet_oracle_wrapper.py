#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


def main() -> int:
    try:
        request = _read_request()
        response = _run(request)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    json.dump(response, sys.stdout, sort_keys=True)
    sys.stdout.write("\n")
    return 0


def _read_request() -> dict:
    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError as exc:
        raise RuntimeError("admet wrapper requires JSON stdin") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("admet wrapper request must be a JSON object")
    return payload


def _run(request: dict) -> dict:
    smiles = _smiles_list(request)
    properties = _properties(request)
    service_url = _service_url()
    payload = {
        "smiles": smiles,
        "endpoints": properties,
        "batch_size": int(os.environ.get("ADMET_BATCH_SIZE", "64")),
    }
    if bool(request.get("return_uncertainty", False)):
        payload["return_uncertainty"] = True
    start = time.perf_counter()
    response = _post_json(f"{service_url}/predict", payload)
    elapsed_ms = int((time.perf_counter() - start) * 1000)
    return {
        "results": _result_rows(response, properties),
        "total_elapsed_ms": elapsed_ms,
    }


def _smiles_list(request: dict) -> list[str]:
    raw = request.get("smiles")
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list) or not raw:
        raise RuntimeError("admet wrapper requires non-empty smiles")
    smiles = [str(item) for item in raw if str(item)]
    if len(smiles) != len(raw):
        raise RuntimeError("admet wrapper smiles items must be non-empty strings")
    return smiles


def _properties(request: dict) -> list[str]:
    raw = request.get("properties") or []
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        raise RuntimeError("admet wrapper properties must be a list of strings")
    properties = [str(item) for item in raw if str(item)]
    if properties:
        return properties
    properties = [
        item.strip()
        for item in os.environ.get("ADMET_TARGETS", "").split(",")
        if item.strip()
    ]
    if not properties:
        raise RuntimeError("admet wrapper requires properties or ADMET_TARGETS")
    return properties


def _service_url() -> str:
    service_url = os.environ.get("ADMET_SERVICE_URL", "").strip().rstrip("/")
    if not service_url:
        raise RuntimeError("ADMET_SERVICE_URL is required")
    parsed = urlsplit(service_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise RuntimeError("ADMET_SERVICE_URL must be an http or https URL")
    return service_url


def _post_json(url: str, payload: dict[str, object]) -> dict[str, Any]:
    body = json.dumps(payload, sort_keys=True).encode("utf-8")
    request = Request(  # noqa: S310
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    timeout = float(os.environ.get("ADMET_ORACLE_TIMEOUT_SECONDS", "120"))
    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310
            response_body = response.read()
    except HTTPError as exc:
        body_text = exc.read().decode("utf-8", "replace")
        raise RuntimeError(f"ADMET service HTTP {exc.code}: {body_text}") from exc
    except URLError as exc:
        raise RuntimeError(f"ADMET service request failed: {exc.reason}") from exc
    try:
        decoded = json.loads(response_body.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError("ADMET service returned invalid JSON") from exc
    if not isinstance(decoded, dict):
        raise RuntimeError("ADMET service response must be a JSON object")
    return decoded


def _result_rows(response: dict[str, Any], properties: list[str]) -> list[dict]:
    rows = response.get("results", response.get("rows"))
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("ADMET service response requires non-empty results")
    output = []
    for row in rows:
        if not isinstance(row, dict):
            raise RuntimeError("ADMET service result rows must be JSON objects")
        smiles = str(row.get("smiles") or row.get("molecule_smiles") or "")
        predictions = _numeric_map(row.get("predictions", row.get("scores")))
        missing = [name for name in properties if name not in predictions]
        if not smiles or missing:
            raise RuntimeError("ADMET service result missing smiles or predictions")
        output_row = {
            "smiles": smiles,
            "predictions": predictions,
        }
        uncertainties = _numeric_map(row.get("uncertainties"))
        if uncertainties:
            output_row["uncertainties"] = uncertainties
        if isinstance(row.get("elapsed_ms"), int | float):
            output_row["elapsed_ms"] = int(row["elapsed_ms"])
        output.append(output_row)
    return output


def _numeric_map(values: object) -> dict[str, float]:
    if not isinstance(values, dict):
        return {}
    output = {}
    for key, value in values.items():
        if isinstance(value, int | float):
            output[str(key)] = float(value)
    return output


if __name__ == "__main__":
    raise SystemExit(main())
