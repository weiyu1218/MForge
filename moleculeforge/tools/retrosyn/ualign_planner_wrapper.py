#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_UALIGN_SOURCE_DIR = ROOT / "models" / "artifacts" / "ualign_source" / "UAlign"
DEFAULT_UALIGN_MODEL_ARCH_PATH = DEFAULT_UALIGN_SOURCE_DIR / "model_arch" / "uspto_50k.json"
DEFAULT_UALIGN_CHECKPOINT_PATH = (
    ROOT / "models" / "artifacts" / "ualign" / "checkpoints" / "USPTO-50K" / "class_unknown.pth"
)
DEFAULT_UALIGN_TOKEN_CKPT_PATH = (
    ROOT / "models" / "artifacts" / "ualign" / "checkpoints" / "USPTO-50K" / "class_unknown.pkl"
)


def main() -> int:
    try:
        response = _run(_read_request())
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    json.dump(response, sys.stdout, sort_keys=True)
    sys.stdout.write("\n")
    return 0


def _read_request() -> dict[str, object]:
    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError as exc:
        raise RuntimeError("UAlign planner wrapper requires JSON stdin") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("UAlign planner wrapper request must be a JSON object")
    return payload


def _run(payload: dict[str, object]) -> dict[str, object]:
    smiles = str(payload.get("smiles") or "")
    if not smiles:
        raise RuntimeError("UAlign planner wrapper requires smiles")
    max_routes = int(payload.get("max_routes") or 10)
    if max_routes <= 0:
        raise RuntimeError("UAlign planner wrapper requires max_routes > 0")
    engine = str(payload.get("engine") or "ualign").strip().lower()
    if engine != "ualign":
        raise RuntimeError(f"Unsupported UAlign planner engine: {engine}")

    start = time.perf_counter()
    result = _run_ualign_inference(smiles, max_routes=max_routes)
    routes = _routes_from_result(result, smiles=smiles, max_routes=max_routes)
    elapsed_ms = int((time.perf_counter() - start) * 1000)
    normalized = _validated_routes(routes)
    return {
        "routes": normalized,
        "total_routes_found": len(normalized),
        "elapsed_ms": elapsed_ms,
    }


def _run_ualign_inference(smiles: str, max_routes: int) -> dict[str, object]:
    paths = _ualign_paths_from_env()
    beams = _positive_int(
        os.environ.get("UALIGN_BEAMS", "").strip() or str(max_routes),
        "UALIGN_BEAMS",
    )
    max_len = _positive_int(
        os.environ.get("UALIGN_MAX_LEN", "").strip() or "300",
        "UALIGN_MAX_LEN",
    )
    command = [
        sys.executable,
        str(paths.source_dir / "inference_one.py"),
        "--model_arch_path",
        str(paths.model_arch_path),
        "--checkpoint",
        str(paths.checkpoint_path),
        "--token_ckpt",
        str(paths.token_ckpt_path),
        "--device",
        os.environ.get("UALIGN_DEVICE", "0").strip() or "0",
        "--beams",
        str(beams),
        "--max_len",
        str(max_len),
        "--product_smiles",
        smiles,
    ]
    if _bool_env("UALIGN_USE_CLASS", default=False):
        command.append("--use_class")
        command.extend(
            [
                "--input_class",
                os.environ.get("UALIGN_INPUT_CLASS", "-1").strip() or "-1",
            ]
        )
    if _bool_env("UALIGN_ORG_OUTPUT", default=False):
        command.append("--org_output")
    if _bool_env("UALIGN_DISABLE_KV_CACHE", default=False):
        command.append("--disable_kv_cache")
    aug_time = os.environ.get("UALIGN_AUG_TIME", "").strip()
    if aug_time:
        command.extend(["--aug_time", str(_positive_int(aug_time, "UALIGN_AUG_TIME"))])
    rank_compute = os.environ.get("UALIGN_RANK_COMPUTE", "").strip()
    if rank_compute:
        command.extend(["--rank_compute", rank_compute])
    score_alpha = os.environ.get("UALIGN_SCORE_ALPHA", "").strip()
    if score_alpha:
        command.extend(["--score_alpha", score_alpha])

    completed = subprocess.run(  # noqa: S603
        command,
        capture_output=True,
        check=False,
        cwd=ROOT,
        text=True,
        timeout=float(os.environ.get("UALIGN_COMMAND_TIMEOUT_SECONDS", "300")),
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        stdout = completed.stdout.strip()
        message = stderr or stdout or f"exit code {completed.returncode}"
        raise RuntimeError(f"UAlign inference failed: {message}")
    if completed.stderr.strip():
        print(completed.stderr.strip(), file=sys.stderr)
    return _parse_ualign_result(completed.stdout)


class _UAlignPaths:
    def __init__(
        self,
        *,
        source_dir: Path,
        model_arch_path: Path,
        checkpoint_path: Path,
        token_ckpt_path: Path,
    ) -> None:
        self.source_dir = source_dir
        self.model_arch_path = model_arch_path
        self.checkpoint_path = checkpoint_path
        self.token_ckpt_path = token_ckpt_path


def _ualign_paths_from_env() -> _UAlignPaths:
    return _UAlignPaths(
        source_dir=_required_dir("UALIGN_SOURCE_DIR", DEFAULT_UALIGN_SOURCE_DIR),
        model_arch_path=_required_file("UALIGN_MODEL_ARCH_PATH", DEFAULT_UALIGN_MODEL_ARCH_PATH),
        checkpoint_path=_required_file("UALIGN_CHECKPOINT_PATH", DEFAULT_UALIGN_CHECKPOINT_PATH),
        token_ckpt_path=_required_file("UALIGN_TOKEN_CKPT_PATH", DEFAULT_UALIGN_TOKEN_CKPT_PATH),
    )


def _required_dir(env_name: str, default: Path) -> Path:
    path = Path(os.environ.get(env_name, "").strip() or default)
    if not path.is_dir():
        raise RuntimeError(f"{env_name} directory not found: {path}")
    return path


def _required_file(env_name: str, default: Path) -> Path:
    path = Path(os.environ.get(env_name, "").strip() or default)
    if not path.is_file():
        raise RuntimeError(f"{env_name} file not found: {path}")
    return path


def _parse_ualign_result(stdout: str) -> dict[str, object]:
    marker = "[RESULT]"
    marker_index = stdout.rfind(marker)
    if marker_index < 0:
        raise RuntimeError("UAlign inference stdout did not contain [RESULT]")
    raw_json = stdout[marker_index + len(marker) :].strip()
    try:
        parsed = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise RuntimeError("UAlign inference returned invalid result JSON") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError("UAlign inference result must be a JSON object")
    return parsed


def _routes_from_result(
    result: dict[str, object],
    *,
    smiles: str,
    max_routes: int,
) -> list[dict[str, object]]:
    answers = result.get("answers")
    probs = result.get("probs")
    if not isinstance(answers, list):
        raise RuntimeError("UAlign inference result requires answers list")
    prob_values = probs if isinstance(probs, list) else []
    routes = []
    seen: set[str] = set()
    for index, answer in enumerate(answers):
        if not isinstance(answer, str):
            continue
        reactants = answer.strip()
        if not reactants:
            continue
        reaction = f"{reactants}>>{smiles}"
        if reaction in seen:
            continue
        seen.add(reaction)
        blocks = _building_blocks_from_answer(reactants)
        route_index = len(routes) + 1
        route: dict[str, object] = {
            "route_id": f"ualign-{route_index}",
            "smiles": smiles,
            "source_engine": "ualign",
            "reaction_smiles": [reaction],
            "steps": [
                {
                    "step_id": "ualign-1",
                    "reaction": reaction,
                    "reactants": [{"smiles": item} for item in blocks],
                    "building_blocks": [{"smiles": item} for item in blocks],
                    "conditions": {"source": "ualign"},
                }
            ],
            "building_blocks": [{"smiles": item} for item in blocks],
            "n_steps": 1,
        }
        score = _score_at(prob_values, index)
        if score is not None:
            route["score"] = score
        routes.append(route)
        if len(routes) >= max_routes:
            break
    return routes


def _building_blocks_from_answer(answer: str) -> list[str]:
    blocks = [item.strip() for item in answer.split(".") if item.strip()]
    if not blocks:
        raise RuntimeError(f"UAlign answer must contain reactants: {answer}")
    return blocks


def _score_at(values: list[object], index: int) -> float | None:
    if index >= len(values):
        return None
    value = values[index]
    if isinstance(value, int | float):
        return float(value)
    return None


def _validated_routes(routes: object) -> list[dict[str, object]]:
    if not isinstance(routes, list):
        raise RuntimeError("UAlign planner must return a list of route dictionaries")
    if not routes:
        raise RuntimeError("UAlign planner returned no routes")
    normalized = []
    for index, route in enumerate(routes):
        if not isinstance(route, dict):
            raise RuntimeError("UAlign planner routes must be JSON objects")
        route_id = str(route.get("route_id") or f"ualign-{index + 1}")
        steps = route.get("steps")
        reaction_smiles = route.get("reaction_smiles")
        if not isinstance(steps, list) and not isinstance(reaction_smiles, list):
            raise RuntimeError("UAlign planner route requires steps or reaction_smiles")
        normalized_route = dict(route)
        normalized_route["route_id"] = route_id
        normalized.append(normalized_route)
    return normalized


def _positive_int(value: str, name: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise RuntimeError(f"{name} must be positive")
    return parsed


def _bool_env(name: str, *, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(f"{name} must be a boolean")


if __name__ == "__main__":
    raise SystemExit(main())
