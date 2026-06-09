#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def score_pharmacophore_records(
    smiles_list: list[str],
    *,
    reference_sdf: str,
    random_seed: int = 61453,
) -> dict[str, dict[str, object]]:
    if not smiles_list:
        raise ValueError("CReM pharmacophore scorer requires at least one SMILES")
    reference = _load_reference(reference_sdf)
    records: dict[str, dict[str, object]] = {}
    for smiles in smiles_list:
        probe = _mol_from_smiles(smiles, random_seed=random_seed)
        shape_score, color_score = _align_to_reference(reference, probe)
        records[str(smiles)] = {
            "pharmacophore_score": float(shape_score + color_score),
            "shape_score": float(shape_score),
            "pharmacophore_color_score": float(color_score),
            "pharmacophore_reference": str(Path(reference_sdf)),
        }
    return records


def _load_reference(reference_sdf: str):
    from rdkit import Chem
    from rdkit.Chem import AllChem

    path = Path(reference_sdf)
    if not path.is_file():
        raise FileNotFoundError(f"pharmacophore reference SDF not found: {path}")
    supplier = Chem.SDMolSupplier(str(path), removeHs=False)
    mol = next((item for item in supplier if item is not None), None)
    if mol is None:
        raise RuntimeError(f"pharmacophore reference SDF contains no molecule: {path}")
    if mol.GetNumConformers() == 0:
        mol = Chem.AddHs(mol)
        status = AllChem.EmbedMolecule(mol, randomSeed=61453)
        if status != 0:
            raise RuntimeError(f"failed to embed pharmacophore reference: {path}")
    return mol


def _mol_from_smiles(smiles: str, *, random_seed: int):
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise RuntimeError(f"invalid SMILES for pharmacophore scoring: {smiles}")
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = int(random_seed)
    if AllChem.EmbedMolecule(mol, params) != 0:
        raise RuntimeError(f"failed to embed molecule for pharmacophore scoring: {smiles}")
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
    except Exception:
        AllChem.UFFOptimizeMolecule(mol, maxIters=200)
    return mol


def _align_to_reference(reference, probe) -> tuple[float, float]:
    from rdkit.Chem import rdShapeAlign

    shape_score, color_score = rdShapeAlign.AlignMol(
        reference,
        probe,
        useColors=True,
        opt_param=0.5,
    )
    return float(shape_score), float(color_score)


def _read_request() -> dict[str, Any]:
    payload = json.load(sys.stdin)
    if not isinstance(payload, dict):
        raise RuntimeError("CReM pharmacophore scorer request must be a JSON object")
    return payload


def _smiles_from_request(payload: Mapping[str, object]) -> list[str]:
    raw = payload.get("smiles")
    if not isinstance(raw, list) or not raw:
        raise RuntimeError("CReM pharmacophore scorer request requires non-empty smiles list")
    smiles = [str(item) for item in raw]
    if not all(item for item in smiles):
        raise RuntimeError("CReM pharmacophore scorer request contains empty SMILES")
    return smiles


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CReM pharmacophore JSON scorer")
    parser.add_argument(
        "--reference-sdf",
        default=os.environ.get("CREM_PHARMACOPHORE_REFERENCE_SDF", ""),
    )
    args = parser.parse_args(argv)
    try:
        if not args.reference_sdf:
            raise RuntimeError(
                "pharmacophore reference is required via --reference-sdf or "
                "CREM_PHARMACOPHORE_REFERENCE_SDF"
            )
        records = score_pharmacophore_records(
            _smiles_from_request(_read_request()),
            reference_sdf=args.reference_sdf,
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    json.dump({"records": records}, sys.stdout, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
