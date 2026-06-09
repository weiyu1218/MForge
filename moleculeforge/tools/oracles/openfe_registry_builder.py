#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def main() -> int:
    try:
        args = _parse_args()
        ligand_smiles = _load_ligand_smiles(args.ligands_sdf)
        if args.transformation_registry_output:
            transformation_registry = _build_transformation_registry(
                args.protein_id,
                ligand_smiles,
                args.transformations_dir,
            )
            _write_json(args.transformation_registry_output, transformation_registry)
        if args.result_registry_output:
            if args.ddg_tsv is None:
                raise RuntimeError("--ddg-tsv is required with --result-registry-output")
            result_registry = _build_result_registry(
                args.protein_id,
                ligand_smiles,
                args.ddg_tsv,
            )
            _write_json(args.result_registry_output, result_registry)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protein-id", required=True)
    parser.add_argument("--ligands-sdf", type=Path, required=True)
    parser.add_argument("--transformations-dir", type=Path)
    parser.add_argument("--transformation-registry-output", type=Path)
    parser.add_argument("--ddg-tsv", type=Path)
    parser.add_argument("--result-registry-output", type=Path)
    args = parser.parse_args()
    if args.transformation_registry_output and args.transformations_dir is None:
        raise RuntimeError(
            "--transformations-dir is required with --transformation-registry-output"
        )
    if not args.transformation_registry_output and not args.result_registry_output:
        raise RuntimeError("at least one registry output path is required")
    return args


def _load_ligand_smiles(sdf_path: Path) -> dict[str, str]:
    if not sdf_path.is_file():
        raise RuntimeError(f"ligands SDF not found: {sdf_path}")
    try:
        from rdkit import Chem
    except Exception as exc:
        raise RuntimeError("openfe registry builder requires rdkit") from exc

    ligands = {}
    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    for index, mol in enumerate(supplier):
        if mol is None:
            raise RuntimeError(f"failed to parse ligand at SDF index {index}")
        name = mol.GetProp("_Name").strip() if mol.HasProp("_Name") else ""
        if not name:
            raise RuntimeError(f"ligand at SDF index {index} is missing _Name")
        smiles = Chem.MolToSmiles(Chem.RemoveHs(mol), canonical=True)
        if not smiles:
            raise RuntimeError(f"ligand {name} has empty canonical SMILES")
        ligands[name] = smiles
    if not ligands:
        raise RuntimeError(f"ligands SDF is empty: {sdf_path}")
    return ligands


def _build_transformation_registry(
    protein_id: str,
    ligand_smiles: dict[str, str],
    transformations_dir: Path,
) -> dict[str, dict[str, dict[str, str]]]:
    if not transformations_dir.is_dir():
        raise RuntimeError(f"transformations directory not found: {transformations_dir}")
    grouped: dict[tuple[str, str], dict[str, str]] = {}
    for path in sorted(transformations_dir.glob("*.json")):
        parsed = _parse_transformation_filename(path.name)
        if parsed is None:
            continue
        ligand_a, ligand_b, phase = parsed
        _require_ligand(ligand_smiles, ligand_a)
        _require_ligand(ligand_smiles, ligand_b)
        grouped.setdefault((ligand_a, ligand_b), {})[phase] = str(path.resolve())

    entries = {}
    for (ligand_a, ligand_b), phases in grouped.items():
        if "complex" not in phases or "solvent" not in phases:
            continue
        key = f"{ligand_smiles[ligand_a]}>>{ligand_smiles[ligand_b]}"
        entries[key] = {
            "complex": phases["complex"],
            "solvent": phases["solvent"],
        }
    if not entries:
        raise RuntimeError(f"no complete complex/solvent transformations found: {transformations_dir}")
    return {protein_id: entries}


def _parse_transformation_filename(filename: str) -> tuple[str, str, str] | None:
    if not filename.startswith("rbfe_") or not filename.endswith(".json"):
        return None
    stem = filename[len("rbfe_") : -len(".json")]
    for phase in ("complex", "solvent"):
        suffix = f"_{phase}"
        marker = f"_{phase}_"
        if stem.endswith(suffix) and marker in stem:
            body = stem[: -len(suffix)]
            ligand_a, ligand_b = body.split(marker, 1)
            return ligand_a, ligand_b, phase
    return None


def _build_result_registry(
    protein_id: str,
    ligand_smiles: dict[str, str],
    ddg_tsv: Path,
) -> dict[str, dict[str, dict[str, object]]]:
    if not ddg_tsv.is_file():
        raise RuntimeError(f"ddG TSV not found: {ddg_tsv}")
    entries = {}
    with ddg_tsv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            ligand_a = _required_row_value(row, "ligand_i")
            ligand_b = _required_row_value(row, "ligand_j")
            _require_ligand(ligand_smiles, ligand_a)
            _require_ligand(ligand_smiles, ligand_b)
            ddg = float(_required_row_value(row, "DDG(i->j) (kcal/mol)"))
            uncertainty = float(_required_row_value(row, "uncertainty (kcal/mol)"))
            key = f"{ligand_smiles[ligand_a]}>>{ligand_smiles[ligand_b]}"
            entries[key] = _result_entry(
                ligand_smiles[ligand_a],
                ligand_smiles[ligand_b],
                ddg,
                uncertainty,
            )
            reverse_key = f"{ligand_smiles[ligand_b]}>>{ligand_smiles[ligand_a]}"
            entries[reverse_key] = _result_entry(
                ligand_smiles[ligand_b],
                ligand_smiles[ligand_a],
                -ddg,
                uncertainty,
            )
    if not entries:
        raise RuntimeError(f"ddG TSV contains no result rows: {ddg_tsv}")
    return {protein_id: entries}


def _result_entry(
    ligand_a_smiles: str,
    ligand_b_smiles: str,
    ddg: float,
    uncertainty: float,
) -> dict[str, object]:
    return {
        "ligand_a_smiles": ligand_a_smiles,
        "ligand_b_smiles": ligand_b_smiles,
        "ddg_kcal_mol": ddg,
        "ddg_uncertainty": uncertainty,
        "n_repeats": 3,
        "method": "openfe",
        "converged": True,
    }


def _require_ligand(ligand_smiles: dict[str, str], ligand_name: str) -> None:
    if ligand_name not in ligand_smiles:
        raise RuntimeError(f"ligand missing from SDF: {ligand_name}")


def _required_row_value(row: dict[str, str | None], key: str) -> str:
    value = row.get(key)
    if value in (None, ""):
        raise RuntimeError(f"ddG TSV row missing {key}")
    return value


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
