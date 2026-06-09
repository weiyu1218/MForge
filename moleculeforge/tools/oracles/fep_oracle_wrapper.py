#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
import time


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
        raise RuntimeError("fep wrapper requires JSON stdin") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("fep wrapper request must be a JSON object")
    return payload


def _run(request: dict) -> dict:
    _require_request(request)
    start = time.perf_counter()
    completed = subprocess.run(  # noqa: S603
        _runner_argv(),
        input=json.dumps(request, sort_keys=True),
        capture_output=True,
        check=False,
        text=True,
        timeout=float(os.environ.get("FEP_ORACLE_TIMEOUT_SECONDS", "120")),
    )
    elapsed_ms = int((time.perf_counter() - start) * 1000)
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        raise RuntimeError(stderr or "OpenFE runner command failed")
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("OpenFE runner returned invalid JSON") from exc
    rows, total_elapsed_ms = _rows_from_response(payload, elapsed_ms)
    return {
        "batch_id": str(payload.get("batch_id") or request.get("project_id") or ""),
        "results": rows,
        "total_elapsed_ms": total_elapsed_ms,
    }


def _require_request(request: dict) -> None:
    required = (
        "project_id",
        "protein_pdb_id",
        "reference_ligand_smiles",
        "test_ligand_smiles",
        "method",
        "n_repeats",
    )
    missing = [key for key in required if key not in request or request[key] in ("", None)]
    if missing:
        raise RuntimeError("fep wrapper request missing fields: " + ", ".join(missing))
    if not isinstance(request["test_ligand_smiles"], list) or not request["test_ligand_smiles"]:
        raise RuntimeError("fep wrapper requires non-empty test_ligand_smiles list")


def _runner_argv() -> list[str]:
    raw = os.environ.get("OPENFE_RUNNER_PATH", "").strip() or "openfe"
    argv = shlex.split(raw)
    if not argv:
        raise RuntimeError("OPENFE_RUNNER_PATH is empty")
    executable = argv[0]
    if not shutil.which(executable) and not os.access(executable, os.X_OK):
        raise RuntimeError(f"OpenFE runner executable is not available: {executable}")
    return argv


def _rows_from_response(payload: object, elapsed_ms: int) -> tuple[list[dict], int]:
    if not isinstance(payload, dict):
        raise RuntimeError("OpenFE runner response must be a JSON object")
    rows = payload.get("results", payload.get("rows"))
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("OpenFE runner response requires non-empty results")
    normalized = []
    for row in rows:
        if not isinstance(row, dict):
            raise RuntimeError("OpenFE runner result rows must be JSON objects")
        normalized.append(_normalized_row(row))
    raw_elapsed = payload.get("total_elapsed_ms")
    total_elapsed_ms = int(raw_elapsed) if isinstance(raw_elapsed, int | float) else elapsed_ms
    return normalized, total_elapsed_ms


def _normalized_row(row: dict) -> dict:
    required = (
        "ligand_a_smiles",
        "ligand_b_smiles",
        "ddg_kcal_mol",
        "ddg_uncertainty",
        "n_repeats",
        "method",
        "converged",
    )
    missing = [key for key in required if key not in row or row[key] in ("", None)]
    if missing:
        raise RuntimeError("OpenFE runner result missing fields: " + ", ".join(missing))
    return {
        "ligand_a_smiles": str(row["ligand_a_smiles"]),
        "ligand_b_smiles": str(row["ligand_b_smiles"]),
        "ddg_kcal_mol": float(row["ddg_kcal_mol"]),
        "ddg_uncertainty": float(row["ddg_uncertainty"]),
        "n_repeats": int(row["n_repeats"]),
        "method": str(row["method"]),
        "per_repeat_ddg": _numeric_map(row.get("per_repeat_ddg")),
        "converged": bool(row["converged"]),
    }


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
