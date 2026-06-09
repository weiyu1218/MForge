#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BOLTZ2_SERVICE_SRC = ROOT / "services" / "boltz2-svc" / "src"
if str(BOLTZ2_SERVICE_SRC) not in sys.path:
    sys.path.insert(0, str(BOLTZ2_SERVICE_SRC))


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
        raise RuntimeError("boltz2 wrapper requires JSON stdin") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("boltz2 wrapper request must be a JSON object")
    return payload


def _run(request: dict) -> dict:
    protein_pdb_id = str(request.get("protein_pdb_id") or "")
    if not protein_pdb_id:
        raise RuntimeError("boltz2 wrapper requires protein_pdb_id")
    ligand_smiles = request.get("ligand_smiles")
    if not isinstance(ligand_smiles, list) or not all(
        isinstance(item, str) for item in ligand_smiles
    ):
        raise RuntimeError("boltz2 wrapper requires ligand_smiles as a list of strings")
    if not ligand_smiles:
        raise RuntimeError("boltz2 wrapper requires at least one ligand_smiles item")
    ensemble_size = int(request.get("ensemble_size") or 5)
    from boltz2_svc.main import BoltzCliRunner

    start = time.perf_counter()
    rows = BoltzCliRunner.from_env().predict_affinity(
        protein_pdb_id,
        ligand_smiles,
        ensemble_size,
    )
    elapsed_ms = int((time.perf_counter() - start) * 1000)
    return {
        "affinities": rows,
        "total_elapsed_ms": elapsed_ms,
    }


if __name__ == "__main__":
    raise SystemExit(main())
