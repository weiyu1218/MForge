#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import csv
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path


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
        raise RuntimeError("OpenFE JSON runner requires JSON stdin") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("OpenFE JSON runner request must be a JSON object")
    return payload


def _run(request: dict) -> dict:
    _require_request(request)
    replay = _replay_path(request)
    if replay is not None:
        return _response_from_payload(_read_json_file(replay), request, 0)
    registry_rows = _result_registry_rows(request, required=False)
    if registry_rows:
        return {
            "batch_id": str(request.get("project_id") or ""),
            "results": registry_rows,
            "total_elapsed_ms": 0,
        }
    transformations = _transformation_paths(request)
    from_registry = False
    if not transformations:
        transformations = _registry_transformation_paths(request)
        from_registry = bool(transformations)
    if not transformations:
        raise RuntimeError(
            "OpenFE JSON runner requires OPENFE_RESULT_REPLAY_PATH or "
            "openfe_transformation_json_paths or OPENFE_TRANSFORMATION_REGISTRY "
            "or OPENFE_RESULT_REGISTRY"
        )
    start = time.perf_counter()
    with _openfe_work_path() as work_path:
        result_paths = []
        for index, transformation_path in enumerate(transformations):
            run_path = work_path / f"run-{index}"
            run_path.mkdir(parents=True, exist_ok=True)
            result_path = run_path / "openfe-result.json"
            _run_quickrun(transformation_path, result_path, run_path)
            result_paths.append(result_path)
        rows = _rows_from_gather(result_paths, request, work_path, required=from_registry)
        if rows is None:
            rows = [
                _row_from_openfe_result(_read_json_file(result_path), request, index)
                for index, result_path in enumerate(result_paths)
            ]
    elapsed_ms = int((time.perf_counter() - start) * 1000)
    return {
        "batch_id": str(request.get("project_id") or ""),
        "results": rows,
        "total_elapsed_ms": elapsed_ms,
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
        raise RuntimeError(
            "OpenFE JSON runner request missing fields: " + ", ".join(missing)
        )
    if not isinstance(request["test_ligand_smiles"], list) or not request["test_ligand_smiles"]:
        raise RuntimeError("OpenFE JSON runner requires non-empty test_ligand_smiles list")


def _replay_path(request: dict) -> Path | None:
    raw = request.get("openfe_result_path") or os.environ.get("OPENFE_RESULT_REPLAY_PATH")
    if not raw:
        return None
    path = Path(str(raw)).expanduser()
    if not path.is_file():
        raise RuntimeError(f"OpenFE replay result not found: {path}")
    return path


def _result_registry_rows(request: dict, *, required: bool = True) -> list[dict] | None:
    raw = os.environ.get("OPENFE_RESULT_REGISTRY", "").strip()
    if not raw:
        return None
    registry_path = Path(raw).expanduser()
    if not registry_path.is_file():
        raise RuntimeError(f"OpenFE result registry not found: {registry_path}")
    registry = _read_json_file(registry_path)
    if not isinstance(registry, dict):
        raise RuntimeError("OpenFE result registry must be a JSON object")
    protein_id = str(request["protein_pdb_id"])
    entries = registry.get(protein_id)
    if entries is None:
        entries = registry
    if not isinstance(entries, dict):
        raise RuntimeError(f"OpenFE result registry entry is invalid: {protein_id}")
    rows = []
    for index, ligand_b in enumerate(request["test_ligand_smiles"]):
        pair_key = _pair_key(request["reference_ligand_smiles"], ligand_b)
        value = entries.get(pair_key)
        if value is None:
            if not required:
                return None
            raise RuntimeError(
                f"OpenFE result registry missing pair {protein_id} {pair_key}"
            )
        rows.append(_result_registry_value_to_row(value, request, index))
    return rows


def _result_registry_value_to_row(value: object, request: dict, index: int) -> dict:
    if not isinstance(value, dict):
        raise RuntimeError("OpenFE result registry value must be a JSON object")
    return {
        "ligand_a_smiles": str(
            value.get("ligand_a_smiles") or request["reference_ligand_smiles"]
        ),
        "ligand_b_smiles": str(
            value.get("ligand_b_smiles") or request["test_ligand_smiles"][index]
        ),
        "ddg_kcal_mol": float(value["ddg_kcal_mol"]),
        "ddg_uncertainty": float(value.get("ddg_uncertainty", 0.0)),
        "n_repeats": int(value.get("n_repeats", request.get("n_repeats", 1)) or 1),
        "method": str(value.get("method", request.get("method", "openfe"))),
        "per_repeat_ddg": _numeric_map(value.get("per_repeat_ddg")),
        "converged": bool(value.get("converged", True)),
    }


def _transformation_paths(request: dict) -> list[Path]:
    raw = request.get("openfe_transformation_json_paths")
    if raw is None:
        raw = request.get("openfe_transformation_json_path")
    if raw is None:
        return []
    if isinstance(raw, str):
        values = [raw]
    elif isinstance(raw, list):
        values = raw
    elif isinstance(raw, dict):
        ligand_smiles = [str(item) for item in request["test_ligand_smiles"]]
        values = [raw.get(smiles) for smiles in ligand_smiles]
    else:
        raise RuntimeError("openfe_transformation_json_paths must be a path, list, or map")
    paths = []
    for value in values:
        if not value:
            raise RuntimeError("openfe_transformation_json_paths contains an empty path")
        path = Path(str(value)).expanduser()
        if not path.is_file():
            raise RuntimeError(f"OpenFE transformation JSON not found: {path}")
        paths.append(path)
    return paths


def _registry_transformation_paths(request: dict) -> list[Path]:
    raw = os.environ.get("OPENFE_TRANSFORMATION_REGISTRY", "").strip()
    if not raw:
        return []
    registry_path = Path(raw).expanduser()
    if not registry_path.is_file():
        raise RuntimeError(f"OpenFE transformation registry not found: {registry_path}")
    registry = _read_json_file(registry_path)
    if not isinstance(registry, dict):
        raise RuntimeError("OpenFE transformation registry must be a JSON object")
    protein_id = str(request["protein_pdb_id"])
    entries = registry.get(protein_id)
    if entries is None:
        entries = registry
    if not isinstance(entries, dict):
        raise RuntimeError(f"OpenFE transformation registry entry is invalid: {protein_id}")
    paths = []
    for ligand_b in request["test_ligand_smiles"]:
        pair_key = _pair_key(request["reference_ligand_smiles"], ligand_b)
        value = entries.get(pair_key)
        if value is None:
            raise RuntimeError(
                f"OpenFE transformation registry missing pair {protein_id} {pair_key}"
            )
        paths.extend(_registry_paths_from_value(value))
    return paths


def _registry_paths_from_value(value: object) -> list[Path]:
    if isinstance(value, str):
        return [_validated_transformation_path(value)]
    if isinstance(value, list):
        return [_validated_transformation_path(item) for item in value]
    if not isinstance(value, dict):
        raise RuntimeError("OpenFE transformation registry value must be a path")
    leg_values = [value.get(key) for key in ("complex", "solvent") if value.get(key)]
    if leg_values:
        return [_validated_transformation_path(item) for item in leg_values]
    for key in (
        "transformation",
        "path",
        "openfe_transformation_json_path",
    ):
        if value.get(key):
            return [_validated_transformation_path(value[key])]
    raise RuntimeError("OpenFE transformation registry value must be a path")


def _validated_transformation_path(raw_path: object) -> Path:
    path = Path(str(raw_path)).expanduser()
    if not path.is_file():
        raise RuntimeError(f"OpenFE transformation JSON not found: {path}")
    return path


def _pair_key(ligand_a_smiles: object, ligand_b_smiles: object) -> str:
    return f"{_canonical_smiles(ligand_a_smiles)}>>{_canonical_smiles(ligand_b_smiles)}"


def _canonical_smiles(raw_smiles: object) -> str:
    smiles = str(raw_smiles)
    try:
        from rdkit import Chem
    except Exception:
        return smiles
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    return Chem.MolToSmiles(mol, canonical=True)


@contextmanager
def _openfe_work_path():
    raw = os.environ.get("OPENFE_WORK_DIR", "").strip()
    if raw:
        path = Path(raw).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        yield path
        return
    with tempfile.TemporaryDirectory(prefix="mforge-openfe-") as work_dir:
        yield Path(work_dir)


def _run_quickrun(transformation_path: Path, result_path: Path, work_path: Path) -> None:
    openfe_argv = _openfe_cli_argv()
    command = [
        *openfe_argv,
        "quickrun",
        str(transformation_path),
        "-d",
        str(work_path),
        "-o",
        str(result_path),
    ]
    completed = subprocess.run(  # noqa: S603
        command,
        capture_output=True,
        check=False,
        env=_openfe_subprocess_env(openfe_argv),
        text=True,
        timeout=float(os.environ.get("OPENFE_QUICKRUN_TIMEOUT_SECONDS", "3600")),
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        raise RuntimeError(stderr or "OpenFE quickrun failed")
    if not result_path.is_file():
        raise RuntimeError("OpenFE quickrun did not write a result JSON")


def _rows_from_gather(
    result_paths: list[Path],
    request: dict,
    work_path: Path,
    *,
    required: bool,
) -> list[dict] | None:
    report_path = work_path / "openfe-ddg.tsv"
    openfe_argv = _openfe_cli_argv()
    command = [
        *openfe_argv,
        "gather",
        "--report",
        "ddg",
        "--tsv",
        "-o",
        str(report_path),
        *[str(path) for path in result_paths],
    ]
    completed = subprocess.run(  # noqa: S603
        command,
        capture_output=True,
        check=False,
        env=_openfe_subprocess_env(openfe_argv),
        text=True,
        timeout=float(os.environ.get("OPENFE_GATHER_TIMEOUT_SECONDS", "600")),
    )
    if completed.returncode != 0:
        if required:
            stderr = completed.stderr.strip()
            raise RuntimeError(stderr or "OpenFE gather failed")
        return None
    if not report_path.is_file():
        if required:
            raise RuntimeError("OpenFE gather did not write a TSV report")
        return None
    rows = _parse_gather_tsv(report_path, request)
    if not rows and required:
        raise RuntimeError("OpenFE gather did not return any ddG rows")
    return rows or None


def _parse_gather_tsv(report_path: Path, request: dict) -> list[dict]:
    with report_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fieldnames = reader.fieldnames or []
        if len(fieldnames) < 3:
            return []
        ligand_a_key = _gather_column(fieldnames, ("ligand_i", "ligand_a", "ligand 1"), 0)
        ligand_b_key = _gather_column(fieldnames, ("ligand_j", "ligand_b", "ligand 2"), 1)
        ddg_key = _gather_column(fieldnames, ("ddg", "delta delta", "delta-delta"), None)
        uncertainty_key = _gather_column(
            fieldnames,
            ("uncert", "standard error", "stderr", "error"),
            None,
        )
        if ddg_key is None:
            return []
        rows = []
        for index, item in enumerate(reader):
            ddg = _number_from_text(item.get(ddg_key))
            if ddg is None:
                continue
            uncertainty = (
                _number_from_text(item.get(uncertainty_key))
                if uncertainty_key is not None
                else None
            )
            rows.append(
                {
                    "ligand_a_smiles": str(
                        item.get(ligand_a_key) or request["reference_ligand_smiles"]
                    ),
                    "ligand_b_smiles": str(
                        item.get(ligand_b_key) or request["test_ligand_smiles"][index]
                    ),
                    "ddg_kcal_mol": float(ddg),
                    "ddg_uncertainty": float(uncertainty) if uncertainty is not None else 0.0,
                    "n_repeats": int(request["n_repeats"]),
                    "method": str(request["method"]),
                    "per_repeat_ddg": {},
                    "converged": True,
                }
            )
    return rows


def _gather_column(
    fieldnames: list[str],
    candidates: tuple[str, ...],
    fallback_index: int | None,
) -> str | None:
    normalized_candidates = tuple(_normalize_header(candidate) for candidate in candidates)
    for fieldname in fieldnames:
        normalized = _normalize_header(fieldname)
        if any(candidate in normalized for candidate in normalized_candidates):
            return fieldname
    if fallback_index is not None and fallback_index < len(fieldnames):
        return fieldnames[fallback_index]
    return None


def _normalize_header(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _number_from_text(value: object) -> float | None:
    if isinstance(value, int | float):
        return float(value)
    if not isinstance(value, str):
        return None
    match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", value)
    if not match:
        return None
    return float(match.group(0))


def _openfe_cli_argv() -> list[str]:
    raw = os.environ.get("OPENFE_CLI_PATH", "").strip()
    if not raw:
        candidate = Path(sys.executable).with_name("openfe")
        raw = str(candidate) if candidate.exists() else "openfe"
    argv = shlex.split(raw)
    if not argv:
        raise RuntimeError("OPENFE_CLI_PATH is empty")
    executable = argv[0]
    if not shutil.which(executable) and not os.access(executable, os.X_OK):
        raise RuntimeError(f"OpenFE CLI executable is not available: {executable}")
    return argv


def _openfe_subprocess_env(openfe_argv: list[str]) -> dict[str, str]:
    env = os.environ.copy()
    executable = openfe_argv[0]
    if not os.path.isabs(executable):
        return env
    bin_dir = str(Path(executable).resolve().parent)
    current_path = env.get("PATH", "")
    path_parts = [part for part in current_path.split(os.pathsep) if part]
    if bin_dir not in path_parts:
        env["PATH"] = bin_dir + (os.pathsep + current_path if current_path else "")
    env.setdefault("CONDA_PREFIX", str(Path(bin_dir).parent))
    return env


def _read_json_file(path: Path) -> object:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _response_from_payload(payload: object, request: dict, elapsed_ms: int) -> dict:
    if not isinstance(payload, dict):
        raise RuntimeError("OpenFE replay result must be a JSON object")
    rows = payload.get("results", payload.get("rows"))
    if not isinstance(rows, list) or not rows:
        rows = [_row_from_openfe_result(payload, request, 0)]
    normalized = [_normalize_row(row, request, index) for index, row in enumerate(rows)]
    raw_elapsed = payload.get("total_elapsed_ms")
    total_elapsed_ms = int(raw_elapsed) if isinstance(raw_elapsed, int | float) else elapsed_ms
    return {
        "batch_id": str(payload.get("batch_id") or request.get("project_id") or ""),
        "results": normalized,
        "total_elapsed_ms": total_elapsed_ms,
    }


def _row_from_openfe_result(payload: object, request: dict, index: int) -> dict:
    if not isinstance(payload, dict):
        raise RuntimeError("OpenFE quickrun result must be a JSON object")
    ddg = _first_number(
        payload,
        (
            "ddg_kcal_mol",
            "estimate",
            "estimate_kcal_mol",
            "magnitude",
            "value",
        ),
    )
    uncertainty = _first_number(
        payload,
        (
            "ddg_uncertainty",
            "uncertainty",
            "uncertainty_kcal_mol",
            "standard_error",
        ),
    )
    if ddg is None:
        nested = payload.get("unit_result") or payload.get("protocol_result")
        if isinstance(nested, dict):
            ddg = _first_number(nested, ("estimate", "magnitude", "value"))
            uncertainty = uncertainty if uncertainty is not None else _first_number(
                nested,
                ("uncertainty", "standard_error"),
            )
    if ddg is None:
        raise RuntimeError("OpenFE quickrun result does not contain a ddG estimate")
    return {
        "ligand_a_smiles": str(request["reference_ligand_smiles"]),
        "ligand_b_smiles": str(request["test_ligand_smiles"][index]),
        "ddg_kcal_mol": float(ddg),
        "ddg_uncertainty": float(uncertainty) if uncertainty is not None else 0.0,
        "n_repeats": int(request["n_repeats"]),
        "method": str(request["method"]),
        "per_repeat_ddg": _numeric_map(payload.get("per_repeat_ddg")),
        "converged": bool(payload.get("converged", True)),
    }


def _normalize_row(row: object, request: dict, index: int) -> dict:
    if not isinstance(row, dict):
        raise RuntimeError("OpenFE result rows must be JSON objects")
    if "ddg_kcal_mol" not in row:
        return _row_from_openfe_result(row, request, index)
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
        raise RuntimeError("OpenFE result row missing fields: " + ", ".join(missing))
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


def _first_number(payload: dict, keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, int | float):
            return float(value)
        if isinstance(value, dict):
            magnitude = value.get("magnitude", value.get("value"))
            if isinstance(magnitude, int | float):
                return float(magnitude)
    return None


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
