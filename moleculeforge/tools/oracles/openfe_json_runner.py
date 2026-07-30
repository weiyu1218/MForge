#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

_DATA_ERROR_EXIT_CODE = 2
_TIMEOUT_EXIT_CODE = 124
_UPSTREAM_PROCESS_GROUP_ENV = "_MFORGE_FEP_UPSTREAM_PROCESS_GROUP"
_OPENFE_REQUEST_WORK_DIR_ENV = "_MFORGE_OPENFE_REQUEST_WORK_DIR"


class RunnerDataError(RuntimeError):
    pass


def main() -> int:
    try:
        request = _read_request()
        response = _run(request)
    except RunnerDataError as exc:
        print(str(exc), file=sys.stderr)
        return _DATA_ERROR_EXIT_CODE
    except (TimeoutError, subprocess.TimeoutExpired) as exc:
        print(str(exc), file=sys.stderr)
        return _TIMEOUT_EXIT_CODE
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
        raise RunnerDataError("OpenFE JSON runner requires JSON stdin") from exc
    if not isinstance(payload, dict):
        raise RunnerDataError("OpenFE JSON runner request must be a JSON object")
    return payload


def _run(request: dict) -> dict:
    _require_request(request)
    replay = _replay_path(request)
    if replay is not None:
        return _response_from_payload(_read_json_file(replay), request, 0)
    registry_rows = _result_registry_rows(request, required=False)
    if registry_rows:
        return _response(request, registry_rows, 0)
    if request["method"] != "openfe":
        raise RunnerDataError(
            "OpenFE transformations require method=openfe or an exact result registry entry"
        )
    transformations = _transformation_paths(request)
    _require_transformation_limit(transformations)
    transformation_groups = [transformations] if transformations else []
    from_registry = False
    gather_identity_aliases = [{}] if transformations else []
    if not transformation_groups:
        transformation_groups, gather_identity_aliases = (
            _registry_transformation_paths(request)
        )
        from_registry = bool(transformation_groups)
    if not transformation_groups:
        raise RuntimeError(
            "OpenFE JSON runner requires OPENFE_RESULT_REPLAY_PATH or "
            "openfe_transformation_json_paths or OPENFE_TRANSFORMATION_REGISTRY "
            "or OPENFE_RESULT_REGISTRY"
        )
    start = time.perf_counter()
    with _openfe_work_path() as work_path:
        repeat_rows = []
        for repeat_index in range(int(request["n_repeats"])):
            repeat_path = work_path / f"repeat-{repeat_index + 1}"
            repeat_path.mkdir(parents=True, exist_ok=False)
            rows = []
            for group_index, group in enumerate(transformation_groups):
                group_path = repeat_path / f"group-{group_index}"
                group_path.mkdir(parents=True, exist_ok=False)
                result_paths = []
                for transformation_index, transformation_path in enumerate(group):
                    run_path = group_path / f"run-{transformation_index}"
                    run_path.mkdir(parents=True, exist_ok=False)
                    result_path = run_path / "openfe-result.json"
                    _run_quickrun(transformation_path, result_path, run_path)
                    result_paths.append(result_path)
                group_request = request
                if from_registry:
                    group_request = {
                        **request,
                        "test_ligand_smiles": [
                            request["test_ligand_smiles"][group_index]
                        ],
                    }
                group_rows = _rows_from_gather(
                    result_paths,
                    group_request,
                    group_path,
                    required=from_registry,
                    identity_aliases=gather_identity_aliases[group_index],
                    bind_single_pair=from_registry,
                )
                if from_registry and group_rows is not None:
                    group_rows = [
                        _row_from_rbfe_result_pair(
                            [_read_json_file(path) for path in result_paths],
                            group_request,
                            group_rows[0],
                        )
                    ]
                if group_rows is None:
                    if len(result_paths) != len(group_request["test_ligand_smiles"]):
                        raise RunnerDataError(
                            "OpenFE quickrun results cannot be mapped to requested test ligands"
                        )
                    group_rows = [
                        _row_from_openfe_result(
                            _read_json_file(result_path),
                            group_request,
                            index,
                        )
                        for index, result_path in enumerate(result_paths)
                    ]
                rows.extend(group_rows)
            repeat_rows.append(rows)
        rows = _aggregate_repeat_rows(repeat_rows, request)
    elapsed_ms = int((time.perf_counter() - start) * 1000)
    return _response(request, rows, elapsed_ms)


def _require_request(request: dict) -> None:
    required = (
        "project_id",
        "request_id",
        "batch_id",
        "protein_pdb_id",
        "reference_ligand_smiles",
        "test_ligand_smiles",
        "method",
        "n_repeats",
    )
    missing = [key for key in required if key not in request or request[key] in ("", None)]
    if missing:
        raise RunnerDataError(
            "OpenFE JSON runner request missing fields: " + ", ".join(missing)
        )
    string_fields = (
        "project_id",
        "request_id",
        "batch_id",
        "protein_pdb_id",
        "reference_ligand_smiles",
        "method",
    )
    invalid_strings = [
        field
        for field in string_fields
        if not isinstance(request[field], str)
        or not request[field]
        or request[field] != request[field].strip()
    ]
    if invalid_strings:
        raise RunnerDataError(
            "OpenFE JSON runner request requires normalized strings: "
            + ", ".join(invalid_strings)
        )
    if not isinstance(request["test_ligand_smiles"], list) or not request["test_ligand_smiles"]:
        raise RunnerDataError("OpenFE JSON runner requires non-empty test_ligand_smiles list")
    if any(
        not isinstance(smiles, str) or not smiles or smiles != smiles.strip()
        for smiles in request["test_ligand_smiles"]
    ):
        raise RunnerDataError(
            "OpenFE JSON runner test_ligand_smiles must contain normalized strings"
        )
    n_repeats = request["n_repeats"]
    if isinstance(n_repeats, bool) or not isinstance(n_repeats, int) or n_repeats <= 0:
        raise RunnerDataError("OpenFE JSON runner n_repeats must be a positive integer")


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
        if not required:
            return None
        raise RunnerDataError(f"OpenFE result registry missing protein: {protein_id}")
    if not isinstance(entries, dict):
        raise RunnerDataError(f"OpenFE result registry entry is invalid: {protein_id}")
    rows = []
    for index, ligand_b in enumerate(request["test_ligand_smiles"]):
        pair_key = _pair_key(request["reference_ligand_smiles"], ligand_b)
        value = entries.get(pair_key)
        if value is None:
            if not required:
                return None
            raise RunnerDataError(
                f"OpenFE result registry missing pair {protein_id} {pair_key}"
            )
        row = _result_registry_value_to_row(
            value,
            request,
            index,
            required=required,
        )
        if row is None:
            return None
        rows.append(row)
    return rows


def _result_registry_value_to_row(
    value: object,
    request: dict,
    index: int,
    *,
    required: bool,
) -> dict | None:
    if not isinstance(value, dict):
        raise RunnerDataError("OpenFE result registry value must be a JSON object")
    required_fields = (
        "ligand_a_smiles",
        "ligand_b_smiles",
        "ddg_kcal_mol",
        "ddg_uncertainty",
        "n_repeats",
        "method",
        "per_repeat_ddg",
        "converged",
    )
    missing = [field for field in required_fields if field not in value]
    if missing:
        raise RunnerDataError(
            "OpenFE result registry value missing fields: " + ", ".join(missing)
        )
    stored_ligand_a = value["ligand_a_smiles"]
    stored_ligand_b = value["ligand_b_smiles"]
    method = value["method"]
    n_repeats = value["n_repeats"]
    if not isinstance(stored_ligand_a, str) or not isinstance(stored_ligand_b, str):
        raise RunnerDataError("OpenFE result registry ligand identities must be strings")
    if not isinstance(method, str) or not method:
        raise RunnerDataError("OpenFE result registry method must be a non-empty string")
    if isinstance(n_repeats, bool) or not isinstance(n_repeats, int) or n_repeats <= 0:
        raise RunnerDataError("OpenFE result registry n_repeats must be a positive integer")
    expected_ligand_a = request["reference_ligand_smiles"]
    expected_ligand_b = request["test_ligand_smiles"][index]
    applicable = (
        _canonical_smiles(stored_ligand_a) == _canonical_smiles(expected_ligand_a)
        and _canonical_smiles(stored_ligand_b) == _canonical_smiles(expected_ligand_b)
        and method == request["method"]
        and n_repeats == request["n_repeats"]
    )
    if not applicable:
        if not required:
            return None
        raise RunnerDataError("OpenFE result registry value does not match request")
    ddg = _finite_number(value["ddg_kcal_mol"], "result registry ddg_kcal_mol")
    uncertainty = _finite_number(
        value["ddg_uncertainty"],
        "result registry ddg_uncertainty",
    )
    if uncertainty < 0:
        raise RunnerDataError(
            "OpenFE result registry ddg_uncertainty must be non-negative"
        )
    if not isinstance(value["converged"], bool):
        raise RunnerDataError("OpenFE result registry converged must be a boolean")
    per_repeat = _required_repeat_map(value["per_repeat_ddg"], n_repeats, ddg)
    return {
        "ligand_a_smiles": expected_ligand_a,
        "ligand_b_smiles": expected_ligand_b,
        "ddg_kcal_mol": ddg,
        "ddg_uncertainty": uncertainty,
        "n_repeats": n_repeats,
        "method": method,
        "per_repeat_ddg": per_repeat,
        "converged": value["converged"],
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
        _require_single_protocol_repeat(path)
        paths.append(path)
    return paths


def _registry_transformation_paths(
    request: dict,
) -> tuple[
    list[list[Path]],
    list[dict[tuple[str, str], tuple[str, str]]],
]:
    raw = os.environ.get("OPENFE_TRANSFORMATION_REGISTRY", "").strip()
    if not raw:
        return [], []
    registry_path = Path(raw).expanduser()
    if not registry_path.is_file():
        raise RuntimeError(f"OpenFE transformation registry not found: {registry_path}")
    registry = _read_json_file(registry_path)
    if not isinstance(registry, dict):
        raise RuntimeError("OpenFE transformation registry must be a JSON object")
    protein_id = str(request["protein_pdb_id"])
    entries = registry.get(protein_id)
    if entries is None:
        raise RuntimeError(f"OpenFE transformation registry missing protein: {protein_id}")
    if not isinstance(entries, dict):
        raise RuntimeError(f"OpenFE transformation registry entry is invalid: {protein_id}")
    groups = []
    identity_aliases = []
    for ligand_b in request["test_ligand_smiles"]:
        pair_key = _pair_key(request["reference_ligand_smiles"], ligand_b)
        value = entries.get(pair_key)
        if value is None:
            raise RuntimeError(
                f"OpenFE transformation registry missing pair {protein_id} {pair_key}"
            )
        group = _registry_paths_from_value(value, registry_path.parent)
        _require_transformation_limit(group)
        groups.append(group)
        identity_aliases.append(
            _registry_identity_aliases(
                value,
                request["reference_ligand_smiles"],
                ligand_b,
            )
        )
    return groups, identity_aliases


def _registry_identity_aliases(
    value: object,
    expected_ligand_a: str,
    expected_ligand_b: str,
) -> dict[tuple[str, str], tuple[str, str]]:
    if not isinstance(value, dict):
        return {}
    expected = (expected_ligand_a, expected_ligand_b)
    aliases = {}
    for left_field, right_field in (
        ("ligand_a_name", "ligand_b_name"),
        ("ligand_a_smiles", "ligand_b_smiles"),
    ):
        left = value.get(left_field)
        right = value.get(right_field)
        if left is None and right is None:
            continue
        if (
            not isinstance(left, str)
            or not left
            or not isinstance(right, str)
            or not right
        ):
            raise RunnerDataError(
                "OpenFE transformation registry identity aliases are invalid"
            )
        if left_field.endswith("smiles") and (
            _canonical_smiles(left) != _canonical_smiles(expected_ligand_a)
            or _canonical_smiles(right) != _canonical_smiles(expected_ligand_b)
        ):
            raise RunnerDataError(
                "OpenFE transformation registry ligand identity does not match request"
            )
        alias = (left, right)
        existing = aliases.get(alias)
        if existing is not None and existing != expected:
            raise RunnerDataError(
                "OpenFE transformation registry identity alias is ambiguous"
            )
        aliases[alias] = expected
    return aliases


def _registry_paths_from_value(value: object, registry_directory: Path) -> list[Path]:
    if isinstance(value, str):
        raise RunnerDataError(
            "OpenFE RBFE transformation registry requires explicit complex and "
            "solvent paths"
        )
    if isinstance(value, list):
        raise RunnerDataError(
            "OpenFE RBFE transformation registry requires explicit complex and "
            "solvent paths"
        )
    if not isinstance(value, dict):
        raise RuntimeError("OpenFE transformation registry value must be a path")
    complex_path = value.get("complex")
    solvent_path = value.get("solvent")
    if complex_path or solvent_path:
        if not complex_path or not solvent_path:
            raise RunnerDataError(
                "OpenFE RBFE transformation registry requires both complex and solvent paths"
            )
        return [
            _validated_transformation_path(item, registry_directory)
            for item in (complex_path, solvent_path)
        ]
    raise RunnerDataError(
        "OpenFE RBFE transformation registry requires explicit complex and solvent paths"
    )


def _validated_transformation_path(
    raw_path: object,
    registry_directory: Path,
) -> Path:
    path = Path(str(raw_path)).expanduser()
    if not path.is_absolute():
        path = registry_directory / path
    path = path.resolve()
    if not path.is_file():
        raise RuntimeError(f"OpenFE transformation JSON not found: {path}")
    _require_single_protocol_repeat(path)
    return path


def _require_single_protocol_repeat(path: Path) -> None:
    payload = _read_json_file(path)
    if not isinstance(payload, dict):
        raise RunnerDataError("OpenFE transformation must be a JSON object")
    protocol = payload.get("protocol")
    settings = protocol.get("settings") if isinstance(protocol, dict) else None
    if not isinstance(settings, dict):
        raise RunnerDataError(
            "OpenFE transformation requires protocol settings with protocol_repeats=1"
        )
    repeats = settings.get("protocol_repeats", settings.get("n_repeats"))
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats != 1:
        raise RunnerDataError(
            "OpenFE transformation must set protocol_repeats=1 so request "
            "n_repeats remains the independent repeat count"
        )


def _require_transformation_limit(paths: list[Path]) -> None:
    maximum = _positive_integer(
        os.environ.get("OPENFE_MAX_TRANSFORMATIONS_PER_PAIR", "2"),
        "OPENFE_MAX_TRANSFORMATIONS_PER_PAIR",
    )
    if len(paths) > maximum:
        raise RunnerDataError(
            "OpenFE transformation count exceeds "
            f"OPENFE_MAX_TRANSFORMATIONS_PER_PAIR={maximum}"
        )


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
    request_path_raw = os.environ.get(_OPENFE_REQUEST_WORK_DIR_ENV, "").strip()
    if request_path_raw:
        request_path = Path(request_path_raw).expanduser()
        request_path.mkdir(parents=True, exist_ok=False)
        try:
            yield request_path
        finally:
            shutil.rmtree(request_path, ignore_errors=True)
        return
    raw = os.environ.get("OPENFE_WORK_DIR", "").strip()
    if raw:
        base_path = Path(raw).expanduser()
        base_path.mkdir(parents=True, exist_ok=True)
        temporary_parent = str(base_path)
    else:
        temporary_parent = None
    with tempfile.TemporaryDirectory(
        prefix="mforge-openfe-",
        dir=temporary_parent,
    ) as work_dir:
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
    completed = _run_subprocess(
        command,
        env=_openfe_subprocess_env(openfe_argv),
        timeout=_positive_number(
            os.environ.get("OPENFE_QUICKRUN_TIMEOUT_SECONDS", "3600"),
            "OPENFE_QUICKRUN_TIMEOUT_SECONDS",
        ),
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
    identity_aliases: dict[tuple[str, str], tuple[str, str]],
    bind_single_pair: bool,
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
    completed = _run_subprocess(
        command,
        env=_openfe_subprocess_env(openfe_argv),
        timeout=_positive_number(
            os.environ.get("OPENFE_GATHER_TIMEOUT_SECONDS", "600"),
            "OPENFE_GATHER_TIMEOUT_SECONDS",
        ),
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
    rows = _parse_gather_tsv(
        report_path,
        request,
        identity_aliases=identity_aliases,
        bind_single_pair=bind_single_pair,
    )
    if not rows and required:
        raise RuntimeError("OpenFE gather did not return any ddG rows")
    if rows and len(rows) != len(request["test_ligand_smiles"]):
        if required:
            raise RunnerDataError(
                "OpenFE gather result count does not match requested test ligands"
            )
        return None
    return rows or None


def _parse_gather_tsv(
    report_path: Path,
    request: dict,
    *,
    identity_aliases: dict[tuple[str, str], tuple[str, str]],
    bind_single_pair: bool,
) -> list[dict]:
    with report_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fieldnames = reader.fieldnames or []
        if len(fieldnames) < 3:
            return []
        ligand_a_key = _gather_column(
            fieldnames,
            ("ligand_i", "ligand_a", "ligand 1"),
            0,
        )
        ligand_b_key = _gather_column(
            fieldnames,
            ("ligand_j", "ligand_b", "ligand 2"),
            1,
        )
        ddg_key = _gather_column(fieldnames, ("ddg", "delta delta", "delta-delta"), None)
        uncertainty_key = _gather_column(
            fieldnames,
            ("uncert", "standard error", "stderr", "error"),
            None,
        )
        if ddg_key is None:
            return []
        if uncertainty_key is None:
            raise RunnerDataError(
                "OpenFE gather report is missing an uncertainty column"
            )
        expected_pairs = {
            (
                _canonical_smiles(request["reference_ligand_smiles"]),
                _canonical_smiles(ligand_b),
            ): (request["reference_ligand_smiles"], ligand_b)
            for ligand_b in request["test_ligand_smiles"]
        }
        if len(expected_pairs) != len(request["test_ligand_smiles"]):
            raise RunnerDataError(
                "OpenFE request contains duplicate canonical test ligands"
            )
        rows_by_pair = {}
        for item in reader:
            raw_ligand_a = item.get(ligand_a_key)
            raw_ligand_b = item.get(ligand_b_key)
            if not raw_ligand_a or not raw_ligand_b:
                raise RunnerDataError("OpenFE gather row is missing ligand identity")
            alias_pair = identity_aliases.get((raw_ligand_a, raw_ligand_b))
            if alias_pair is None:
                canonical_pair = (
                    _canonical_smiles(raw_ligand_a),
                    _canonical_smiles(raw_ligand_b),
                )
            else:
                canonical_pair = (
                    _canonical_smiles(alias_pair[0]),
                    _canonical_smiles(alias_pair[1]),
                )
            if bind_single_pair and canonical_pair != next(iter(expected_pairs)):
                raise RunnerDataError(
                    "OpenFE gather row ligand identity contradicts transformation registry"
                )
            expected_identity = expected_pairs.get(canonical_pair)
            if expected_identity is None:
                raise RunnerDataError(
                    "OpenFE gather row ligand identity does not match request"
                )
            if canonical_pair in rows_by_pair:
                raise RunnerDataError(
                    "OpenFE gather returned duplicate rows for a ligand pair"
                )
            ddg = _number_from_text(item.get(ddg_key))
            if ddg is None:
                raise RunnerDataError("OpenFE gather row ddG is invalid")
            uncertainty = (
                _number_from_text(item.get(uncertainty_key))
                if uncertainty_key is not None
                else None
            )
            if uncertainty is None or uncertainty < 0:
                raise RunnerDataError(
                    "OpenFE gather row uncertainty is invalid"
                )
            rows_by_pair[canonical_pair] = {
                "ligand_a_smiles": expected_identity[0],
                "ligand_b_smiles": expected_identity[1],
                "ddg_kcal_mol": float(ddg),
                "ddg_uncertainty": float(uncertainty),
                "n_repeats": int(request["n_repeats"]),
                "method": str(request["method"]),
                "per_repeat_ddg": {},
                "converged": False,
            }
    if set(rows_by_pair) != set(expected_pairs):
        raise RunnerDataError(
            "OpenFE gather results do not contain every requested ligand pair"
        )
    return [
        rows_by_pair[
            (
                _canonical_smiles(request["reference_ligand_smiles"]),
                _canonical_smiles(ligand_b),
            )
        ]
        for ligand_b in request["test_ligand_smiles"]
    ]


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
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        number = float(value)
        return number if math.isfinite(number) else None
    if not isinstance(value, str):
        return None
    match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", value)
    if not match:
        return None
    number = float(match.group(0))
    return number if math.isfinite(number) else None


def _openfe_cli_argv() -> list[str]:
    raw = os.environ.get("OPENFE_CLI_PATH", "").strip()
    if not raw:
        raise RuntimeError("OPENFE_CLI_PATH is required")
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
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as exc:
        raise RunnerDataError(f"OpenFE JSON data is invalid: {path}") from exc


def _response_from_payload(payload: object, request: dict, elapsed_ms: int) -> dict:
    if not isinstance(payload, dict):
        raise RunnerDataError("OpenFE replay result must be a JSON object")
    scientific_identity = {
        "project_id": request["project_id"],
        "protein_pdb_id": request["protein_pdb_id"],
        "reference_ligand_smiles": request["reference_ligand_smiles"],
        "test_ligand_smiles": request["test_ligand_smiles"],
        "method": request["method"],
        "n_repeats": request["n_repeats"],
    }
    missing = [field for field in scientific_identity if field not in payload]
    if missing:
        raise RunnerDataError(
            "OpenFE replay missing scientific identity fields: " + ", ".join(missing)
        )
    replay_string_fields = (
        "project_id",
        "protein_pdb_id",
        "reference_ligand_smiles",
        "method",
    )
    if any(
        not isinstance(payload[field], str)
        or not payload[field]
        or payload[field] != payload[field].strip()
        for field in replay_string_fields
    ):
        raise RunnerDataError(
            "OpenFE replay scientific identity fields must be normalized strings"
        )
    if not isinstance(payload["test_ligand_smiles"], list) or any(
        not isinstance(smiles, str) or not smiles or smiles != smiles.strip()
        for smiles in payload["test_ligand_smiles"]
    ):
        raise RunnerDataError(
            "OpenFE replay test_ligand_smiles must contain normalized strings"
        )
    replay_repeats = payload["n_repeats"]
    if (
        isinstance(replay_repeats, bool)
        or not isinstance(replay_repeats, int)
        or replay_repeats <= 0
    ):
        raise RunnerDataError("OpenFE replay n_repeats must be a positive integer")
    for field in ("request_id", "batch_id"):
        if field in payload and (
            not isinstance(payload[field], str)
            or not payload[field]
            or payload[field] != payload[field].strip()
        ):
            raise RunnerDataError(
                f"OpenFE replay {field} must be a normalized string when present"
            )
    mismatched = [
        field for field, value in scientific_identity.items() if payload[field] != value
    ]
    for field in ("request_id", "batch_id"):
        if field in payload and payload[field] != request[field]:
            mismatched.append(field)
    if mismatched:
        raise RunnerDataError(
            "OpenFE replay identity does not match request: " + ", ".join(mismatched)
        )
    rows = payload.get("results", payload.get("rows"))
    if not isinstance(rows, list) or not rows:
        raise RunnerDataError("OpenFE replay requires non-empty results")
    if len(rows) != len(request["test_ligand_smiles"]):
        raise RunnerDataError(
            "OpenFE replay result count does not match requested test ligands"
        )
    normalized = [_normalize_row(row, request, index) for index, row in enumerate(rows)]
    raw_elapsed = payload.get("total_elapsed_ms")
    if raw_elapsed is None:
        total_elapsed_ms = elapsed_ms
    else:
        elapsed = _finite_number(raw_elapsed, "replay total_elapsed_ms")
        if elapsed < 0:
            raise RunnerDataError(
                "OpenFE replay total_elapsed_ms must be non-negative"
            )
        total_elapsed_ms = int(elapsed)
    return _response(request, normalized, total_elapsed_ms)


def _response(request: dict, results: list[dict], total_elapsed_ms: int) -> dict:
    return {
        **_response_identity(request),
        "results": results,
        "total_elapsed_ms": total_elapsed_ms,
    }


def _response_identity(request: dict) -> dict:
    return {
        "request_id": str(request["request_id"]),
        "batch_id": str(request["batch_id"]),
        "project_id": str(request["project_id"]),
        "protein_pdb_id": str(request["protein_pdb_id"]),
        "reference_ligand_smiles": str(request["reference_ligand_smiles"]),
        "test_ligand_smiles": [str(item) for item in request["test_ligand_smiles"]],
        "method": str(request["method"]),
        "n_repeats": int(request["n_repeats"]),
    }


def _row_from_rbfe_result_pair(
    payloads: list[object],
    request: dict,
    identity_row: dict,
) -> dict:
    if len(payloads) != 2 or len(request["test_ligand_smiles"]) != 1:
        raise RunnerDataError(
            "OpenFE RBFE execution requires one complex and one solvent result "
            "per ligand pair"
        )
    complex_estimate, complex_uncertainty, complex_complete = (
        _openfe_leg_result(payloads[0])
    )
    solvent_estimate, solvent_uncertainty, solvent_complete = (
        _openfe_leg_result(payloads[1])
    )
    return {
        **identity_row,
        "ddg_kcal_mol": complex_estimate - solvent_estimate,
        "ddg_uncertainty": math.sqrt(
            complex_uncertainty**2 + solvent_uncertainty**2
        ),
        "converged": complex_complete and solvent_complete,
    }


def _openfe_leg_result(payload: object) -> tuple[float, float, bool]:
    if not isinstance(payload, dict):
        raise RunnerDataError("OpenFE quickrun result must be a JSON object")
    reported_estimate = _openfe_kcal_quantity(payload.get("estimate"), "estimate")
    protocol_result = payload.get("protocol_result")
    data = (
        protocol_result.get("data")
        if isinstance(protocol_result, dict)
        else None
    )
    if not isinstance(data, dict) or len(data) != 1:
        raise RunnerDataError(
            "OpenFE quickrun result must contain exactly one protocol repeat"
        )
    result_chain = next(iter(data.values()))
    if not isinstance(result_chain, list) or not result_chain:
        raise RunnerDataError("OpenFE quickrun protocol repeat is incomplete")
    final_result = result_chain[-1]
    outputs = (
        final_result.get("outputs")
        if isinstance(final_result, dict) and "exception" not in final_result
        else None
    )
    if not isinstance(outputs, dict):
        raise RunnerDataError("OpenFE quickrun protocol repeat did not complete")
    estimate = _openfe_kcal_quantity(
        outputs.get("unit_estimate"),
        "unit_estimate",
    )
    uncertainty = _openfe_kcal_quantity(
        outputs.get("unit_estimate_error"),
        "unit_estimate_error",
    )
    if not math.isclose(
        estimate,
        reported_estimate,
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        raise RunnerDataError(
            "OpenFE quickrun protocol estimate does not match the reported estimate"
        )
    if uncertainty < 0:
        raise RunnerDataError("OpenFE quickrun uncertainty must be non-negative")
    unit_results = payload.get("unit_results")
    if not isinstance(unit_results, dict) or not unit_results:
        raise RunnerDataError(
            "OpenFE quickrun result does not contain protocol unit results"
        )
    unit_successes: dict[str, bool] = {}
    for result_key, value in unit_results.items():
        if not isinstance(value, dict):
            raise RunnerDataError("OpenFE protocol unit results must be JSON objects")
        source_key = value.get("source_key", result_key)
        if not isinstance(source_key, str) or not source_key:
            raise RunnerDataError(
                "OpenFE protocol unit result source_key must be a non-empty string"
            )
        unit_successes[source_key] = (
            unit_successes.get(source_key, False) or "exception" not in value
        )
    converged = all(unit_successes.values()) and all(
        isinstance(value, dict)
        for value in result_chain
    )
    return estimate, uncertainty, converged


def _openfe_kcal_quantity(value: object, field_name: str) -> float:
    if not isinstance(value, dict):
        raise RunnerDataError(
            f"OpenFE quickrun {field_name} must be a unit-bearing quantity"
        )
    unit_name = value.get("unit")
    normalized_unit = (
        re.sub(r"[^a-z]+", "", unit_name.lower())
        if isinstance(unit_name, str)
        else ""
    )
    if normalized_unit not in {
        "kilocaloriemole",
        "kilocaloriespermole",
        "kilocaloriepermole",
        "kcalmole",
        "kcalpermole",
    }:
        raise RunnerDataError(
            f"OpenFE quickrun {field_name} must use kilocalories per mole"
        )
    return _finite_number(
        value.get("magnitude"),
        f"quickrun {field_name} magnitude",
    )


def _row_from_openfe_result(payload: object, request: dict, index: int) -> dict:
    if not isinstance(payload, dict):
        raise RunnerDataError("OpenFE quickrun result must be a JSON object")
    if index >= len(request["test_ligand_smiles"]):
        raise RunnerDataError(
            "OpenFE quickrun returned more results than requested test ligands"
        )
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
        raise RunnerDataError("OpenFE quickrun result does not contain a ddG estimate")
    if uncertainty is None:
        raise RunnerDataError(
            "OpenFE quickrun result does not contain an uncertainty estimate"
        )
    uncertainty_value = uncertainty
    if uncertainty_value < 0:
        raise RunnerDataError("OpenFE quickrun uncertainty must be non-negative")
    if "converged" not in payload:
        raise RunnerDataError(
            "OpenFE quickrun result does not contain convergence evidence"
        )
    converged = payload["converged"]
    if not isinstance(converged, bool):
        raise RunnerDataError("OpenFE quickrun converged must be a boolean")
    return {
        "ligand_a_smiles": str(request["reference_ligand_smiles"]),
        "ligand_b_smiles": str(request["test_ligand_smiles"][index]),
        "ddg_kcal_mol": float(ddg),
        "ddg_uncertainty": uncertainty_value,
        "n_repeats": int(request["n_repeats"]),
        "method": str(request["method"]),
        "per_repeat_ddg": {},
        "converged": converged,
    }


def _normalize_row(row: object, request: dict, index: int) -> dict:
    if not isinstance(row, dict):
        raise RunnerDataError("OpenFE result rows must be JSON objects")
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
        raise RunnerDataError("OpenFE result row missing fields: " + ", ".join(missing))
    if index >= len(request["test_ligand_smiles"]):
        raise RunnerDataError(
            "OpenFE result contains more rows than requested test ligands"
        )
    expected_ligand_a = request["reference_ligand_smiles"]
    expected_ligand_b = request["test_ligand_smiles"][index]
    if row["ligand_a_smiles"] != expected_ligand_a or row["ligand_b_smiles"] != expected_ligand_b:
        raise RunnerDataError("OpenFE result row ligand identity does not match request")
    if row["method"] != request["method"]:
        raise RunnerDataError("OpenFE result row method does not match request")
    n_repeats = row["n_repeats"]
    if isinstance(n_repeats, bool) or not isinstance(n_repeats, int) or n_repeats <= 0:
        raise RunnerDataError("OpenFE result row n_repeats must be a positive integer")
    if n_repeats != request["n_repeats"]:
        raise RunnerDataError("OpenFE result row n_repeats does not match request")
    ddg = _finite_number(row["ddg_kcal_mol"], "result row ddg_kcal_mol")
    uncertainty = _finite_number(
        row["ddg_uncertainty"],
        "result row ddg_uncertainty",
    )
    if uncertainty < 0:
        raise RunnerDataError("OpenFE result row ddg_uncertainty must be non-negative")
    if not isinstance(row["converged"], bool):
        raise RunnerDataError("OpenFE result row converged must be a boolean")
    per_repeat = _required_repeat_map(row.get("per_repeat_ddg"), n_repeats, ddg)
    return {
        "ligand_a_smiles": expected_ligand_a,
        "ligand_b_smiles": expected_ligand_b,
        "ddg_kcal_mol": ddg,
        "ddg_uncertainty": uncertainty,
        "n_repeats": n_repeats,
        "method": request["method"],
        "per_repeat_ddg": per_repeat,
        "converged": row["converged"],
    }


def _first_number(payload: dict, keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, int | float) and not isinstance(value, bool):
            number = float(value)
            return number if math.isfinite(number) else None
        if isinstance(value, dict):
            magnitude = value.get("magnitude", value.get("value"))
            if isinstance(magnitude, int | float) and not isinstance(magnitude, bool):
                number = float(magnitude)
                return number if math.isfinite(number) else None
    return None


def _aggregate_repeat_rows(repeat_rows: list[list[dict]], request: dict) -> list[dict]:
    n_repeats = request["n_repeats"]
    if len(repeat_rows) != n_repeats:
        raise RunnerDataError("OpenFE repeat result count does not match n_repeats")
    expected_count = len(request["test_ligand_smiles"])
    if any(len(rows) != expected_count for rows in repeat_rows):
        raise RunnerDataError(
            "OpenFE repeat results do not match requested test ligands"
        )
    aggregated = []
    for index, ligand_b in enumerate(request["test_ligand_smiles"]):
        values = []
        uncertainties = []
        converged = True
        for rows in repeat_rows:
            row = rows[index]
            if not isinstance(row, dict):
                raise RunnerDataError("OpenFE repeat rows must be JSON objects")
            if (
                row.get("ligand_a_smiles") != request["reference_ligand_smiles"]
                or row.get("ligand_b_smiles") != ligand_b
                or row.get("method") != request["method"]
            ):
                raise RunnerDataError("OpenFE repeat row identity does not match request")
            ddg = _finite_number(row.get("ddg_kcal_mol"), "repeat ddg_kcal_mol")
            uncertainty = _finite_number(
                row.get("ddg_uncertainty"),
                "repeat ddg_uncertainty",
            )
            if uncertainty < 0:
                raise RunnerDataError(
                    "OpenFE repeat ddg_uncertainty must be non-negative"
                )
            if not isinstance(row.get("converged"), bool):
                raise RunnerDataError("OpenFE repeat converged must be a boolean")
            values.append(ddg)
            uncertainties.append(uncertainty)
            converged = converged and row["converged"]
        mean_ddg = sum(values) / n_repeats
        within_variance = sum(value * value for value in uncertainties) / (
            n_repeats * n_repeats
        )
        between_variance = 0.0
        if n_repeats > 1:
            between_variance = sum(
                (value - mean_ddg) ** 2 for value in values
            ) / (n_repeats * (n_repeats - 1))
        aggregated.append(
            {
                "ligand_a_smiles": request["reference_ligand_smiles"],
                "ligand_b_smiles": ligand_b,
                "ddg_kcal_mol": mean_ddg,
                "ddg_uncertainty": math.sqrt(within_variance + between_variance),
                "n_repeats": n_repeats,
                "method": request["method"],
                "per_repeat_ddg": {
                    f"repeat_{repeat_index + 1}": value
                    for repeat_index, value in enumerate(values)
                },
                "converged": converged,
            }
        )
    return aggregated


def _finite_number(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise RunnerDataError(f"OpenFE {field_name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise RunnerDataError(f"OpenFE {field_name} must be finite")
    return number


def _positive_number(value: object, field_name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{field_name} must be a finite positive number") from exc
    if isinstance(value, bool) or not math.isfinite(number) or number <= 0:
        raise RuntimeError(f"{field_name} must be a finite positive number")
    return number


def _positive_integer(value: object, field_name: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{field_name} must be a positive integer") from exc
    if isinstance(value, bool) or str(number) != str(value).strip() or number <= 0:
        raise RuntimeError(f"{field_name} must be a positive integer")
    return number


def _run_subprocess(
    command: list[str],
    *,
    env: dict[str, str],
    timeout: float,
) -> subprocess.CompletedProcess[str]:
    owns_process_group = os.environ.get(_UPSTREAM_PROCESS_GROUP_ENV) != "1"
    child_env = dict(env)
    if owns_process_group:
        child_env[_UPSTREAM_PROCESS_GROUP_ENV] = "1"
    process = subprocess.Popen(  # noqa: S603
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=child_env,
        start_new_session=owns_process_group,
        text=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        terminate = (
            _terminate_process_group if owns_process_group else _terminate_process
        )
        stdout, stderr = terminate(process)
        raise subprocess.TimeoutExpired(
            command,
            timeout,
            output=stdout,
            stderr=stderr,
        ) from exc
    except BaseException:
        terminate = (
            _terminate_process_group if owns_process_group else _terminate_process
        )
        terminate(process)
        raise
    if owns_process_group:
        _terminate_process_group(process)
    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


def _terminate_process(process: subprocess.Popen[str]) -> tuple[str, str]:
    if process.poll() is None:
        process.terminate()
    deadline = time.monotonic() + 5.0
    while process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.05)
    if process.poll() is None:
        process.kill()
    return process.communicate()


def _terminate_process_group(process: subprocess.Popen[str]) -> tuple[str, str]:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    deadline = time.monotonic() + 5.0
    while process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.05)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    return process.communicate()


def _required_repeat_map(values: object, n_repeats: int, ddg: float) -> dict[str, float]:
    if not isinstance(values, dict):
        raise RunnerDataError("OpenFE per_repeat_ddg must be a JSON object")
    expected_keys = {f"repeat_{index + 1}" for index in range(n_repeats)}
    actual_keys = {str(key) for key in values}
    if actual_keys != expected_keys:
        raise RunnerDataError("OpenFE per_repeat_ddg keys do not match n_repeats")
    output = {
        str(key): _finite_number(value, f"per_repeat_ddg[{key}]")
        for key, value in values.items()
    }
    repeat_mean = sum(output.values()) / n_repeats
    if not math.isclose(repeat_mean, ddg, rel_tol=1e-6, abs_tol=1e-6):
        raise RunnerDataError("OpenFE per_repeat_ddg mean does not match ddg_kcal_mol")
    return output


if __name__ == "__main__":
    raise SystemExit(main())
