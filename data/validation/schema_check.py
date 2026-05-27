"""Schema validation for molecular data files.

Validates CSV/SDF/JSON files against expected schemas before ingestion.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def validate_molecule_csv(path: str | Path) -> dict[str, Any]:
    """Validate a molecular CSV file.

    Expected columns: smiles (required), name/id/source (optional).

    Returns dict with keys: valid, n_rows, n_valid_smiles, n_invalid_smiles,
    missing_required_columns, issues.
    """
    issues: list[str] = []
    path = Path(path)

    if not path.exists():
        return {"valid": False, "issues": [f"File not found: {path}"]}

    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames or []

        if "smiles" not in columns:
            issues.append("Missing required column: 'smiles'")

        n_rows = 0
        n_valid = 0
        n_invalid = 0

        from rdkit import Chem

        for row in reader:
            n_rows += 1
            smiles = row.get("smiles", "").strip()
            if smiles and Chem.MolFromSmiles(smiles) is not None:
                n_valid += 1
            else:
                n_invalid += 1
                if n_invalid <= 5:
                    issues.append(f"Row {n_rows}: invalid SMILES '{smiles[:50]}'")

    return {
        "valid": len(issues) == 0 or all("Missing" not in i for i in issues),
        "n_rows": n_rows,
        "n_valid_smiles": n_valid,
        "n_invalid_smiles": n_invalid,
        "columns": columns,
        "issues": issues,
    }


def validate_reaction_csv(path: str | Path) -> dict[str, Any]:
    """Validate a reaction CSV file.

    Expected columns: reactants, products (required); catalyst/solvent/yield (optional).
    """
    issues: list[str] = []
    path = Path(path)

    if not path.exists():
        return {"valid": False, "issues": [f"File not found: {path}"]}

    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames or []

        missing = []
        for col in ["reactants", "products"]:
            if col not in columns:
                missing.append(col)
        if missing:
            issues.append(f"Missing required columns: {missing}")

        n_rows = 0
        n_valid = 0
        from rdkit import Chem

        for row in reader:
            n_rows += 1
            r = row.get("reactants", "")
            p = row.get("products", "")
            if r and p and Chem.MolFromSmiles(r) and Chem.MolFromSmiles(p):
                n_valid += 1

    return {
        "valid": len(issues) == 0,
        "n_rows": n_rows,
        "n_valid_reactions": n_valid,
        "columns": columns,
        "issues": issues,
    }


def validate_json_schema(path: str | Path, schema: dict) -> dict[str, Any]:
    """Validate a JSON file against a JSON Schema (basic checks)."""
    import jsonschema

    path = Path(path)
    if not path.exists():
        return {"valid": False, "issues": [f"File not found: {path}"]}

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    try:
        jsonschema.validate(data, schema)
        return {"valid": True, "issues": []}
    except jsonschema.ValidationError as e:
        return {"valid": False, "issues": [str(e)]}


def validate_conformer_count(
    smiles: str, *, expected_min: int = 10
) -> dict[str, Any]:
    """Check that a molecule can generate a minimum number of conformers."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"valid": False, "issues": ["Invalid SMILES"]}

    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    n_conf = AllChem.EmbedMultipleConfs(mol, numConfs=expected_min * 2, params=params)

    valid = len(n_conf) >= expected_min
    return {
        "valid": valid,
        "n_conformers": len(n_conf),
        "issues": [] if valid else [f"Only {len(n_conf)} conformers (min={expected_min})"],
    }
