#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RASCORE_MODEL_PATH = (
    ROOT / "models" / "artifacts" / "rascore" / "XGB_chembl_ecfp_counts" / "model.json"
)
RASCORE_ROUTE_TYPE = "retrosynthetic_accessibility_score"


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
        raise RuntimeError("RAscore planner wrapper requires JSON stdin") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("RAscore planner wrapper request must be a JSON object")
    return payload


def _run(payload: dict[str, object]) -> dict[str, object]:
    smiles = str(payload.get("smiles") or "")
    if not smiles:
        raise RuntimeError("RAscore planner wrapper requires smiles")
    max_routes = int(payload.get("max_routes") or 10)
    if max_routes <= 0:
        raise RuntimeError("RAscore planner wrapper requires max_routes > 0")
    engine = str(payload.get("engine") or "rascore").strip().lower()
    if engine != "rascore":
        raise RuntimeError(f"Unsupported RAscore planner engine: {engine}")

    start = time.perf_counter()
    score = _score_smiles(smiles, _rascore_model_path_from_env())
    elapsed_ms = int((time.perf_counter() - start) * 1000)
    routes = [_accessibility_route(smiles, score)]
    return {
        "routes": routes[:max_routes],
        "total_routes_found": min(len(routes), max_routes),
        "elapsed_ms": elapsed_ms,
    }


def _score_smiles(smiles: str, model_path: Path) -> float:
    import xgboost as xgb
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise RuntimeError(f"RAscore planner received invalid SMILES: {smiles}")
    booster = xgb.Booster()
    booster.load_model(str(model_path))
    fingerprint = AllChem.GetMorganFingerprint(
        mol,
        3,
        useCounts=True,
        useFeatures=False,
    )
    values = np.zeros((1, 2048), dtype=np.float32)
    for index, count in fingerprint.GetNonzeroElements().items():
        values[0, index % 2048] += float(count)
    score = float(booster.inplace_predict(values)[0])
    if not 0.0 <= score <= 1.0:
        raise RuntimeError(f"RAscore model returned out-of-range score: {score}")
    return score


def _rascore_model_path_from_env() -> Path:
    path = Path(os.environ.get("RASCORE_MODEL_PATH", "").strip() or DEFAULT_RASCORE_MODEL_PATH)
    if not path.is_file():
        raise RuntimeError(f"RASCORE_MODEL_PATH file not found: {path}")
    return path


def _accessibility_route(smiles: str, score: float) -> dict[str, object]:
    return {
        "route_id": "rascore-1",
        "smiles": smiles,
        "source_engine": "rascore",
        "route_type": RASCORE_ROUTE_TYPE,
        "model": "RAscore XGB ChEMBL ECFP counts",
        "score": score,
        "predicted_score": score,
        "accessibility_score": score,
        "reaction_smiles": [],
        "steps": [],
        "building_blocks": [],
        "n_steps": 0,
    }


if __name__ == "__main__":
    raise SystemExit(main())
