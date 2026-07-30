#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time

_DATA_ERROR_EXIT_CODE = 2
_TIMEOUT_EXIT_CODE = 124
_UPSTREAM_PROCESS_GROUP_ENV = "_MFORGE_FEP_UPSTREAM_PROCESS_GROUP"


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
        raise RunnerDataError("fep wrapper requires JSON stdin") from exc
    if not isinstance(payload, dict):
        raise RunnerDataError("fep wrapper request must be a JSON object")
    return payload


def _run(request: dict) -> dict:
    _require_request(request)
    start = time.perf_counter()
    completed = _run_subprocess(
        _runner_argv(),
        input=json.dumps(request, sort_keys=True),
        timeout=_runner_timeout_seconds(request),
    )
    elapsed_ms = int((time.perf_counter() - start) * 1000)
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        if completed.returncode == _DATA_ERROR_EXIT_CODE:
            raise RunnerDataError(stderr or "OpenFE runner returned invalid data")
        if completed.returncode == _TIMEOUT_EXIT_CODE:
            raise TimeoutError(stderr or "OpenFE runner timed out")
        raise RuntimeError(stderr or "OpenFE runner command failed")
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RunnerDataError("OpenFE runner returned invalid JSON") from exc
    _validate_response_identity(payload, request)
    rows, total_elapsed_ms = _rows_from_response(payload, elapsed_ms)
    return {
        "request_id": str(payload["request_id"]),
        "batch_id": str(payload["batch_id"]),
        "project_id": str(payload["project_id"]),
        "protein_pdb_id": str(payload["protein_pdb_id"]),
        "reference_ligand_smiles": str(payload["reference_ligand_smiles"]),
        "test_ligand_smiles": [str(item) for item in payload["test_ligand_smiles"]],
        "method": str(payload["method"]),
        "n_repeats": int(payload["n_repeats"]),
        "results": rows,
        "total_elapsed_ms": total_elapsed_ms,
    }


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
        raise RunnerDataError("fep wrapper request missing fields: " + ", ".join(missing))
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
            "fep wrapper request requires normalized strings: "
            + ", ".join(invalid_strings)
        )
    if not isinstance(request["test_ligand_smiles"], list) or not request["test_ligand_smiles"]:
        raise RunnerDataError("fep wrapper requires non-empty test_ligand_smiles list")
    if any(
        not isinstance(smiles, str) or not smiles or smiles != smiles.strip()
        for smiles in request["test_ligand_smiles"]
    ):
        raise RunnerDataError(
            "fep wrapper test_ligand_smiles must contain normalized strings"
        )
    n_repeats = request["n_repeats"]
    if isinstance(n_repeats, bool) or not isinstance(n_repeats, int) or n_repeats <= 0:
        raise RunnerDataError("fep wrapper n_repeats must be a positive integer")


def _validate_response_identity(payload: object, request: dict) -> None:
    if not isinstance(payload, dict):
        raise RunnerDataError("OpenFE runner response must be a JSON object")
    expected = {
        "request_id": str(request["request_id"]),
        "batch_id": str(request["batch_id"]),
        "project_id": str(request["project_id"]),
        "protein_pdb_id": str(request["protein_pdb_id"]),
        "reference_ligand_smiles": str(request["reference_ligand_smiles"]),
        "test_ligand_smiles": [str(item) for item in request["test_ligand_smiles"]],
        "method": str(request["method"]),
        "n_repeats": int(request["n_repeats"]),
    }
    missing = [field for field in expected if field not in payload]
    if missing:
        raise RunnerDataError(
            "OpenFE runner response missing identity fields: " + ", ".join(missing)
        )
    string_fields = (
        "request_id",
        "batch_id",
        "project_id",
        "protein_pdb_id",
        "reference_ligand_smiles",
        "method",
    )
    if any(
        not isinstance(payload[field], str)
        or not payload[field]
        or payload[field] != payload[field].strip()
        for field in string_fields
    ):
        raise RunnerDataError(
            "OpenFE runner response identity fields must be normalized strings"
        )
    if not isinstance(payload["test_ligand_smiles"], list) or any(
        not isinstance(smiles, str) or not smiles or smiles != smiles.strip()
        for smiles in payload["test_ligand_smiles"]
    ):
        raise RunnerDataError(
            "OpenFE runner response test_ligand_smiles must be normalized strings"
        )
    response_repeats = payload["n_repeats"]
    if (
        isinstance(response_repeats, bool)
        or not isinstance(response_repeats, int)
        or response_repeats <= 0
    ):
        raise RunnerDataError(
            "OpenFE runner response n_repeats must be a positive integer"
        )
    mismatched = [field for field, value in expected.items() if payload[field] != value]
    if mismatched:
        raise RunnerDataError(
            "OpenFE runner response identity does not match request: "
            + ", ".join(mismatched)
        )


def _runner_argv() -> list[str]:
    raw = os.environ.get("OPENFE_RUNNER_PATH", "").strip()
    if not raw:
        raise RuntimeError("OPENFE_RUNNER_PATH is required")
    argv = shlex.split(raw)
    if not argv:
        raise RuntimeError("OPENFE_RUNNER_PATH is empty")
    executable = argv[0]
    if not shutil.which(executable) and not os.access(executable, os.X_OK):
        raise RuntimeError(f"OpenFE runner executable is not available: {executable}")
    return argv


def _runner_timeout_seconds(request: dict) -> float:
    quickrun_timeout = _positive_number(
        os.environ.get("OPENFE_QUICKRUN_TIMEOUT_SECONDS", "3600"),
        "OPENFE_QUICKRUN_TIMEOUT_SECONDS",
    )
    gather_timeout = _positive_number(
        os.environ.get("OPENFE_GATHER_TIMEOUT_SECONDS", "600"),
        "OPENFE_GATHER_TIMEOUT_SECONDS",
    )
    maximum_transformations = _positive_integer(
        os.environ.get("OPENFE_MAX_TRANSFORMATIONS_PER_PAIR", "2"),
        "OPENFE_MAX_TRANSFORMATIONS_PER_PAIR",
    )
    minimum = (
        int(request["n_repeats"])
        * len(request["test_ligand_smiles"])
        * (maximum_transformations * quickrun_timeout + gather_timeout)
        + 60.0
    )
    configured = os.environ.get("OPENFE_RUNNER_TIMEOUT_SECONDS")
    if configured is None or not configured.strip():
        return minimum
    timeout = _positive_number(configured, "OPENFE_RUNNER_TIMEOUT_SECONDS")
    if timeout < minimum:
        raise RuntimeError(
            "OPENFE_RUNNER_TIMEOUT_SECONDS is shorter than the configured "
            f"OpenFE execution budget ({minimum:g}s)"
        )
    return timeout


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
    input: str,
    timeout: float,
) -> subprocess.CompletedProcess[str]:
    owns_process_group = os.environ.get(_UPSTREAM_PROCESS_GROUP_ENV) != "1"
    child_env = os.environ.copy()
    if owns_process_group:
        child_env[_UPSTREAM_PROCESS_GROUP_ENV] = "1"
    process = subprocess.Popen(  # noqa: S603
        command,
        env=child_env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=owns_process_group,
        text=True,
    )
    try:
        stdout, stderr = process.communicate(input=input, timeout=timeout)
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


def _rows_from_response(payload: object, elapsed_ms: int) -> tuple[list[dict], int]:
    if not isinstance(payload, dict):
        raise RunnerDataError("OpenFE runner response must be a JSON object")
    rows = payload.get("results", payload.get("rows"))
    if not isinstance(rows, list) or not rows:
        raise RunnerDataError("OpenFE runner response requires non-empty results")
    expected_ligands = payload["test_ligand_smiles"]
    if len(rows) != len(expected_ligands):
        raise RunnerDataError(
            "OpenFE runner result count does not match test_ligand_smiles"
        )
    normalized = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise RunnerDataError("OpenFE runner result rows must be JSON objects")
        normalized.append(
            _normalized_row(
                row,
                reference_ligand=payload["reference_ligand_smiles"],
                test_ligand=expected_ligands[index],
                method=payload["method"],
                n_repeats=payload["n_repeats"],
            )
        )
    raw_elapsed = payload.get("total_elapsed_ms")
    if raw_elapsed is None:
        total_elapsed_ms = elapsed_ms
    else:
        elapsed = _finite_number(raw_elapsed, "total_elapsed_ms")
        if elapsed < 0:
            raise RunnerDataError(
                "OpenFE runner total_elapsed_ms must be non-negative"
            )
        total_elapsed_ms = int(elapsed)
    return normalized, total_elapsed_ms


def _normalized_row(
    row: dict,
    *,
    reference_ligand: str,
    test_ligand: str,
    method: str,
    n_repeats: int,
) -> dict:
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
        raise RunnerDataError("OpenFE runner result missing fields: " + ", ".join(missing))
    if row["ligand_a_smiles"] != reference_ligand or row["ligand_b_smiles"] != test_ligand:
        raise RunnerDataError("OpenFE runner result ligand identity does not match request")
    if row["method"] != method:
        raise RunnerDataError("OpenFE runner result method does not match request")
    row_repeats = row["n_repeats"]
    if isinstance(row_repeats, bool) or not isinstance(row_repeats, int) or row_repeats <= 0:
        raise RunnerDataError("OpenFE runner result n_repeats must be a positive integer")
    if row_repeats != n_repeats:
        raise RunnerDataError("OpenFE runner result n_repeats does not match request")
    ddg = _finite_number(row["ddg_kcal_mol"], "ddg_kcal_mol")
    uncertainty = _finite_number(row["ddg_uncertainty"], "ddg_uncertainty")
    if uncertainty < 0:
        raise RunnerDataError(
            "OpenFE runner result ddg_uncertainty must be non-negative"
        )
    if not isinstance(row["converged"], bool):
        raise RunnerDataError("OpenFE runner result converged must be a boolean")
    per_repeat = _required_repeat_map(row.get("per_repeat_ddg"), n_repeats, ddg)
    return {
        "ligand_a_smiles": reference_ligand,
        "ligand_b_smiles": test_ligand,
        "ddg_kcal_mol": ddg,
        "ddg_uncertainty": uncertainty,
        "n_repeats": n_repeats,
        "method": method,
        "per_repeat_ddg": per_repeat,
        "converged": row["converged"],
    }


def _finite_number(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise RunnerDataError(f"OpenFE runner {field_name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise RunnerDataError(f"OpenFE runner {field_name} must be finite")
    return number


def _required_repeat_map(
    values: object,
    n_repeats: int,
    ddg: float,
) -> dict[str, float]:
    if not isinstance(values, dict):
        raise RunnerDataError("OpenFE runner per_repeat_ddg must be a JSON object")
    expected_keys = {f"repeat_{index + 1}" for index in range(n_repeats)}
    actual_keys = {str(key) for key in values}
    if actual_keys != expected_keys:
        raise RunnerDataError(
            "OpenFE runner per_repeat_ddg keys do not match n_repeats"
        )
    output = {
        str(key): _finite_number(value, f"per_repeat_ddg[{key}]")
        for key, value in values.items()
    }
    repeat_mean = sum(output.values()) / n_repeats
    if not math.isclose(repeat_mean, ddg, rel_tol=1e-6, abs_tol=1e-6):
        raise RunnerDataError(
            "OpenFE runner per_repeat_ddg mean does not match ddg_kcal_mol"
        )
    return output


if __name__ == "__main__":
    raise SystemExit(main())
