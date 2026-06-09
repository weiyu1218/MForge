#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


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
        raise RuntimeError("OpenADMET JSON runner requires JSON stdin") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("OpenADMET JSON runner request must be a JSON object")
    return payload


def _run(request: dict) -> dict:
    smiles = _smiles_list(request)
    properties = _properties(request)
    model_dirs = _model_dirs(properties)
    prediction_columns = _column_mapping("OPENADMET_PROPERTY_COLUMNS")
    uncertainty_columns = _column_mapping("OPENADMET_UNCERTAINTY_COLUMNS")
    start = time.perf_counter()
    rows = [{"smiles": item, "predictions": {}} for item in smiles]
    include_uncertainty = bool(request.get("return_uncertainty", False))
    if include_uncertainty:
        for row in rows:
            row["uncertainties"] = {}
    with tempfile.TemporaryDirectory(prefix="mforge-openadmet-") as work_dir:
        work_path = Path(work_dir)
        input_path = work_path / "input.csv"
        _write_input_csv(input_path, smiles)
        for prop in properties:
            output_path = work_path / f"{_safe_name(prop)}-predictions.csv"
            _run_openadmet(input_path, model_dirs[prop], output_path)
            prediction_rows = _read_prediction_csv(output_path)
            _merge_property(
                rows,
                smiles,
                prop,
                prediction_rows,
                prediction_columns,
                uncertainty_columns,
                include_uncertainty,
            )
    elapsed_ms = int((time.perf_counter() - start) * 1000)
    return {"results": rows, "total_elapsed_ms": elapsed_ms}


def _smiles_list(request: dict) -> list[str]:
    raw = request.get("smiles")
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list) or not raw:
        raise RuntimeError("OpenADMET JSON runner requires non-empty smiles")
    smiles = [str(item) for item in raw if isinstance(item, str) and item]
    if len(smiles) != len(raw):
        raise RuntimeError("OpenADMET JSON runner smiles items must be non-empty strings")
    return smiles


def _properties(request: dict) -> list[str]:
    raw = request.get("properties") or []
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        raise RuntimeError("OpenADMET JSON runner properties must be a list of strings")
    properties = [str(item) for item in raw if isinstance(item, str) and item]
    if properties:
        return properties
    properties = [
        item.strip()
        for item in os.environ.get("ADMET_TARGETS", "").split(",")
        if item.strip()
    ]
    if not properties:
        raise RuntimeError("OpenADMET JSON runner requires properties or ADMET_TARGETS")
    return properties


def _model_dirs(properties: list[str]) -> dict[str, Path]:
    mapped = _path_mapping("OPENADMET_MODEL_DIRS")
    global_model_dir = os.environ.get("OPENADMET_MODEL_DIR", "").strip()
    output: dict[str, Path] = {}
    for prop in properties:
        raw = mapped.get(prop) or global_model_dir
        if not raw:
            raise RuntimeError(
                "OpenADMET JSON runner requires OPENADMET_MODEL_DIR or "
                f"OPENADMET_MODEL_DIRS entry for {prop}"
            )
        path = Path(raw).expanduser()
        if not path.is_dir():
            raise RuntimeError(f"OpenADMET model directory not found for {prop}: {path}")
        output[prop] = path
    return output


def _path_mapping(env_name: str) -> dict[str, str]:
    output: dict[str, str] = {}
    raw = os.environ.get(env_name, "").strip()
    if not raw:
        return output
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise RuntimeError(f"{env_name} entries must use property=path")
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or not value:
            raise RuntimeError(f"{env_name} entries must use non-empty property=path")
        output[key] = value
    return output


def _column_mapping(env_name: str) -> dict[str, str]:
    output: dict[str, str] = {}
    raw = os.environ.get(env_name, "").strip()
    if not raw:
        return output
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise RuntimeError(f"{env_name} entries must use property=column")
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or not value:
            raise RuntimeError(f"{env_name} entries must use non-empty property=column")
        output[key] = value
    return output


def _write_input_csv(path: Path, smiles: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["smiles"])
        writer.writeheader()
        for item in smiles:
            writer.writerow({"smiles": item})


def _run_openadmet(input_path: Path, model_dir: Path, output_path: Path) -> None:
    command = [
        *_openadmet_argv(),
        "predict",
        "--input-path",
        str(input_path),
        "--input-col",
        "smiles",
        "--model-dir",
        str(model_dir),
        "--output-csv",
        str(output_path),
        "--accelerator",
        os.environ.get("OPENADMET_ACCELERATOR", "gpu"),
    ]
    completed = subprocess.run(  # noqa: S603
        command,
        capture_output=True,
        check=False,
        text=True,
        timeout=float(os.environ.get("OPENADMET_TIMEOUT_SECONDS", "3600")),
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        raise RuntimeError(stderr or "OpenADMET predict failed")
    if not output_path.is_file():
        raise RuntimeError("OpenADMET predict did not write output CSV")


def _openadmet_argv() -> list[str]:
    raw = os.environ.get("OPENADMET_BINARY", "").strip() or "openadmet"
    argv = shlex.split(raw)
    if not argv:
        raise RuntimeError("OPENADMET_BINARY is empty")
    executable = argv[0]
    if not shutil.which(executable) and not os.access(executable, os.X_OK):
        raise RuntimeError(f"OpenADMET executable is not available: {executable}")
    return argv


def _read_prediction_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError("OpenADMET output CSV contains no prediction rows")
    return rows


def _merge_property(
    output_rows: list[dict[str, Any]],
    input_smiles: list[str],
    prop: str,
    prediction_rows: list[dict[str, str]],
    prediction_columns: dict[str, str],
    uncertainty_columns: dict[str, str],
    include_uncertainty: bool,
) -> None:
    by_smiles = _rows_by_smiles(prediction_rows, input_smiles)
    for output_row in output_rows:
        smiles = str(output_row["smiles"])
        source_row = by_smiles[smiles]
        pred_col = _select_column(source_row, prop, prediction_columns, "OADMET_PRED")
        prediction = _float_value(source_row.get(pred_col), pred_col)
        output_row["predictions"][prop] = prediction
        if include_uncertainty:
            uncertainty_col = _select_optional_column(
                source_row,
                prop,
                uncertainty_columns,
                "OADMET_STD",
            )
            if uncertainty_col is not None:
                output_row["uncertainties"][prop] = _float_value(
                    source_row.get(uncertainty_col),
                    uncertainty_col,
                )


def _rows_by_smiles(
    prediction_rows: list[dict[str, str]],
    input_smiles: list[str],
) -> dict[str, dict[str, str]]:
    if len(prediction_rows) != len(input_smiles):
        raise RuntimeError("OpenADMET output row count does not match input smiles count")
    output: dict[str, dict[str, str]] = {}
    for index, row in enumerate(prediction_rows):
        row_smiles = row.get("smiles") or row.get("OPENADMET_SMILES") or input_smiles[index]
        output[str(row_smiles)] = row
    missing = [smiles for smiles in input_smiles if smiles not in output]
    if missing:
        raise RuntimeError("OpenADMET output missing smiles: " + ", ".join(missing))
    return output


def _select_column(
    row: dict[str, str],
    prop: str,
    mapping: dict[str, str],
    prefix: str,
) -> str:
    column = _select_optional_column(row, prop, mapping, prefix)
    if column is None:
        raise RuntimeError(f"OpenADMET output missing prediction column for {prop}")
    return column


def _select_optional_column(
    row: dict[str, str],
    prop: str,
    mapping: dict[str, str],
    prefix: str,
) -> str | None:
    if prop in mapping:
        column = mapping[prop]
        if column not in row:
            raise RuntimeError(f"OpenADMET output missing configured column: {column}")
        return column
    for candidate in (prop, f"{prefix}_{prop}"):
        if candidate in row and _is_float(row[candidate]):
            return candidate
    candidates = [
        key
        for key, value in row.items()
        if key.startswith(prefix) and _is_float(value)
    ]
    if len(candidates) == 1:
        return candidates[0]
    return None


def _float_value(raw: object, column: str) -> float:
    try:
        return float(str(raw))
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"OpenADMET output column is not numeric: {column}") from exc


def _is_float(raw: object) -> bool:
    try:
        float(str(raw))
    except (TypeError, ValueError):
        return False
    return True


def _safe_name(value: str) -> str:
    output = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)
    return output.strip("_") or "property"


if __name__ == "__main__":
    raise SystemExit(main())
