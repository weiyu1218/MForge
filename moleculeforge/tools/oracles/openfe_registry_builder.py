#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
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
                args.transformation_registry_output,
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
        if args.experimental_registry_output:
            if args.experimental_binding_json is None:
                raise RuntimeError(
                    "--experimental-binding-json is required with "
                    "--experimental-registry-output"
                )
            experimental_registry = _build_experimental_registry(
                args.protein_id,
                ligand_smiles,
                args.experimental_binding_json,
            )
            _write_json(args.experimental_registry_output, experimental_registry)
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
    parser.add_argument("--experimental-binding-json", type=Path)
    parser.add_argument("--experimental-registry-output", type=Path)
    args = parser.parse_args()
    if args.transformation_registry_output and args.transformations_dir is None:
        raise RuntimeError(
            "--transformations-dir is required with --transformation-registry-output"
        )
    if not (
        args.transformation_registry_output
        or args.result_registry_output
        or args.experimental_registry_output
    ):
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
    registry_output: Path,
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
        _require_single_protocol_repeat(path)
        grouped.setdefault((ligand_a, ligand_b), {})[phase] = _portable_path(
            path,
            registry_output.parent,
        )

    entries = {}
    for (ligand_a, ligand_b), phases in grouped.items():
        if "complex" not in phases or "solvent" not in phases:
            raise RuntimeError(
                "incomplete complex/solvent transformations for "
                f"{ligand_a}>>{ligand_b}"
            )
        key = f"{ligand_smiles[ligand_a]}>>{ligand_smiles[ligand_b]}"
        entries[key] = {
            "complex": phases["complex"],
            "solvent": phases["solvent"],
            "ligand_a_name": ligand_a,
            "ligand_b_name": ligand_b,
            "ligand_a_smiles": ligand_smiles[ligand_a],
            "ligand_b_smiles": ligand_smiles[ligand_b],
        }
    if not entries:
        raise RuntimeError(
            "no complete complex/solvent transformations found: "
            f"{transformations_dir}"
        )
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
            ddg = _finite_number(
                _required_row_value(row, "DDG(i->j) (kcal/mol)"),
                "DDG(i->j) (kcal/mol)",
            )
            uncertainty = _finite_number(
                _required_row_value(row, "uncertainty (kcal/mol)"),
                "uncertainty (kcal/mol)",
            )
            if uncertainty < 0:
                raise RuntimeError("ddG TSV uncertainty must be non-negative")
            converged = _boolean_value(
                _required_row_value(row, "converged"),
                "converged",
            )
            n_repeats, per_repeat_ddg = _repeat_evidence(row, ddg)
            key = f"{ligand_smiles[ligand_a]}>>{ligand_smiles[ligand_b]}"
            entries[key] = _result_entry(
                ligand_smiles[ligand_a],
                ligand_smiles[ligand_b],
                ddg,
                uncertainty,
                converged,
                n_repeats,
                per_repeat_ddg,
            )
            reverse_key = f"{ligand_smiles[ligand_b]}>>{ligand_smiles[ligand_a]}"
            entries[reverse_key] = _result_entry(
                ligand_smiles[ligand_b],
                ligand_smiles[ligand_a],
                -ddg,
                uncertainty,
                converged,
                n_repeats,
                {key: -value for key, value in per_repeat_ddg.items()},
            )
    if not entries:
        raise RuntimeError(f"ddG TSV contains no result rows: {ddg_tsv}")
    return {protein_id: entries}


def _build_experimental_registry(
    protein_id: str,
    ligand_smiles: dict[str, str],
    experimental_binding_json: Path,
) -> dict[str, dict[str, dict[str, object]]]:
    if not experimental_binding_json.is_file():
        raise RuntimeError(f"experimental binding JSON not found: {experimental_binding_json}")
    payload = json.loads(experimental_binding_json.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("experimental binding JSON must be an object")

    absolute_entries = {}
    for ligand_name, record in payload.items():
        _require_ligand(ligand_smiles, ligand_name)
        if not isinstance(record, dict) or not isinstance(record.get("dg"), dict):
            raise RuntimeError(f"experimental binding record missing dg: {ligand_name}")
        unit = str(_required_row_value(record["dg"], "unit"))
        if unit != "kilocalories_per_mole":
            raise RuntimeError(f"unsupported experimental binding unit for {ligand_name}: {unit}")
        uncertainty = _finite_number(
            _required_row_value(record["dg"], "uncertainty"),
            f"experimental uncertainty for {ligand_name}",
        )
        if uncertainty < 0:
            raise RuntimeError(
                f"experimental uncertainty must be non-negative: {ligand_name}"
            )
        converged = record.get("converged")
        if not isinstance(converged, bool):
            raise RuntimeError(
                f"experimental binding record requires boolean converged: {ligand_name}"
            )
        absolute_entries[ligand_name] = {
            "ligand_smiles": ligand_smiles[ligand_name],
            "dg_kcal_mol": _finite_number(
                _required_row_value(record["dg"], "magnitude"),
                f"experimental magnitude for {ligand_name}",
            ),
            "uncertainty": uncertainty,
            "converged": converged,
            "reference": str(record.get("reference", "")),
        }

    entries = {}
    ligand_names = sorted(absolute_entries)
    for ligand_a in ligand_names:
        for ligand_b in ligand_names:
            if ligand_a == ligand_b:
                continue
            left = absolute_entries[ligand_a]
            right = absolute_entries[ligand_b]
            key = f"{left['ligand_smiles']}>>{right['ligand_smiles']}"
            entries[key] = {
                "ligand_a_smiles": left["ligand_smiles"],
                "ligand_b_smiles": right["ligand_smiles"],
                "ddg_kcal_mol": right["dg_kcal_mol"] - left["dg_kcal_mol"],
                "ddg_uncertainty": math.sqrt(
                    left["uncertainty"] ** 2 + right["uncertainty"] ** 2
                ),
                "n_repeats": 1,
                "method": "experimental_binding_free_energy",
                "per_repeat_ddg": {
                    "repeat_1": right["dg_kcal_mol"] - left["dg_kcal_mol"]
                },
                "converged": left["converged"] and right["converged"],
                "source_ligand_a": ligand_a,
                "source_ligand_b": ligand_b,
                "source_reference": right["reference"] or left["reference"],
            }
    if not entries:
        raise RuntimeError(
            "experimental binding JSON contains fewer than two ligands: "
            f"{experimental_binding_json}"
        )
    return {protein_id: entries}


def _result_entry(
    ligand_a_smiles: str,
    ligand_b_smiles: str,
    ddg: float,
    uncertainty: float,
    converged: bool,
    n_repeats: int,
    per_repeat_ddg: dict[str, float],
) -> dict[str, object]:
    return {
        "ligand_a_smiles": ligand_a_smiles,
        "ligand_b_smiles": ligand_b_smiles,
        "ddg_kcal_mol": ddg,
        "ddg_uncertainty": uncertainty,
        "n_repeats": n_repeats,
        "method": "openfe",
        "per_repeat_ddg": per_repeat_ddg,
        "converged": converged,
    }


def _repeat_evidence(
    row: dict[str, str | None],
    ddg: float,
) -> tuple[int, dict[str, float]]:
    raw_repeats = _required_row_value(row, "n_repeats")
    try:
        n_repeats = int(raw_repeats)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("ddG TSV n_repeats must be a positive integer") from exc
    if str(n_repeats) != str(raw_repeats).strip() or n_repeats <= 0:
        raise RuntimeError("ddG TSV n_repeats must be a positive integer")
    raw_evidence = _required_row_value(row, "per_repeat_ddg")
    try:
        evidence = json.loads(raw_evidence)
    except json.JSONDecodeError as exc:
        raise RuntimeError("ddG TSV per_repeat_ddg must be a JSON object") from exc
    expected_keys = {f"repeat_{index}" for index in range(1, n_repeats + 1)}
    if not isinstance(evidence, dict) or set(evidence) != expected_keys:
        raise RuntimeError("ddG TSV per_repeat_ddg does not match n_repeats")
    normalized = {
        key: _finite_number(value, f"per_repeat_ddg[{key}]")
        for key, value in evidence.items()
    }
    if not math.isclose(
        sum(normalized.values()) / n_repeats,
        ddg,
        rel_tol=1e-6,
        abs_tol=1e-6,
    ):
        raise RuntimeError("ddG TSV per_repeat_ddg mean does not match DDG")
    return n_repeats, normalized


def _require_single_protocol_repeat(path: Path) -> None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"transformation JSON is invalid: {path}") from exc
    protocol = payload.get("protocol") if isinstance(payload, dict) else None
    settings = protocol.get("settings") if isinstance(protocol, dict) else None
    if not isinstance(settings, dict):
        raise RuntimeError(
            f"transformation must define protocol settings with protocol_repeats=1: {path}"
        )
    repeats = settings.get("protocol_repeats", settings.get("n_repeats"))
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats != 1:
        raise RuntimeError(f"transformation must set protocol_repeats=1: {path}")


def _require_ligand(ligand_smiles: dict[str, str], ligand_name: str) -> None:
    if ligand_name not in ligand_smiles:
        raise RuntimeError(f"ligand missing from SDF: {ligand_name}")


def _required_row_value(row: dict[str, str | None], key: str) -> str:
    value = row.get(key)
    if value in (None, ""):
        raise RuntimeError(f"ddG TSV row missing {key}")
    return value


def _portable_path(path: Path, registry_directory: Path) -> str:
    return Path(
        os.path.relpath(path.resolve(), registry_directory.resolve())
    ).as_posix()


def _finite_number(value: object, field_name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{field_name} must be a finite number") from exc
    if isinstance(value, bool) or not math.isfinite(number):
        raise RuntimeError(f"{field_name} must be a finite number")
    return number


def _boolean_value(value: object, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise RuntimeError(f"{field_name} must be true or false")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
