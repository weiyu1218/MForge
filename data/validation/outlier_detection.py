"""Outlier detection for molecular properties and 3D conformers.

Detects molecules with anomalous properties (LogP, MW, TPSA) and
conformational outliers (unusual bond lengths, angles).
"""

from __future__ import annotations

from typing import Any

import numpy as np


def detect_property_outliers(
    properties: list[dict[str, float]],
    *,
    sigma: float = 3.0,
) -> dict[str, Any]:
    """Detect outlier molecules based on molecular properties (> sigma std from mean).

    Args:
        properties: list of dicts with keys like 'mw', 'logp', 'tpsa', 'qed'
        sigma: number of standard deviations for outlier threshold

    Returns dict with: n_total, n_outliers, outlier_indices, per_property_stats.
    """
    if not properties:
        return {"n_total": 0, "n_outliers": 0, "outlier_indices": [], "per_property_stats": {}}

    keys = list(properties[0].keys())
    n = len(properties)

    # Collect per-property arrays
    props: dict[str, np.ndarray] = {}
    for key in keys:
        values = np.array([p.get(key, np.nan) for p in properties], dtype=np.float64)
        props[key] = values

    # Detect outliers
    outlier_scores = np.zeros(n)
    per_prop_stats: dict[str, dict] = {}

    for key, values in props.items():
        valid = ~np.isnan(values)
        if not valid.any():
            continue
        mu = np.mean(values[valid])
        std = np.std(values[valid])
        per_prop_stats[key] = {"mean": float(mu), "std": float(std)}
        if std < 1e-9:
            continue
        z_scores = np.abs((values - mu) / std)
        outlier_scores += np.where(valid & (z_scores > sigma), 1.0, 0.0)

    outlier_indices = [int(i) for i in np.where(outlier_scores > 0)[0]]

    return {
        "n_total": n,
        "n_outliers": len(outlier_indices),
        "outlier_indices": outlier_indices,
        "per_property_stats": per_prop_stats,
    }


def detect_conformer_outliers(
    conformer_positions: list[np.ndarray],
    *,
    max_bond_length: float = 2.5,
    min_bond_length: float = 0.8,
) -> dict[str, Any]:
    """Detect conformers with unreasonable bond lengths.

    Args:
        conformer_positions: list of (N, 3) position arrays
        max_bond_length: Ångström, bonds longer than this are suspicious
        min_bond_length: Ångström, bonds shorter than this are suspicious
    """
    issues: list[dict] = []
    for idx, pos in enumerate(conformer_positions):
        n_atoms = pos.shape[0]
        for i in range(n_atoms):
            for j in range(i + 1, min(i + 5, n_atoms)):
                dist = float(np.linalg.norm(pos[i] - pos[j]))
                if dist > max_bond_length:
                    issues.append({
                        "conformer_idx": idx,
                        "atom_pair": (i, j),
                        "distance": dist,
                        "issue": "too_long",
                    })
                elif dist < min_bond_length:
                    issues.append({
                        "conformer_idx": idx,
                        "atom_pair": (i, j),
                        "distance": dist,
                        "issue": "too_short",
                    })

    return {
        "n_conformers": len(conformer_positions),
        "n_issues": len(issues),
        "issues": issues[:20],  # cap for large datasets
    }


def detect_charge_outliers(
    smiles_list: list[str],
    *,
    max_absolute_charge: int = 3,
) -> dict[str, Any]:
    """Detect molecules with unusually high formal charges."""
    from rdkit import Chem

    issues: list[dict] = []
    for idx, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        for atom in mol.GetAtoms():
            charge = atom.GetFormalCharge()
            if abs(charge) > max_absolute_charge:
                issues.append({
                    "molecule_idx": idx,
                    "smiles": smi,
                    "atom_idx": atom.GetIdx(),
                    "charge": charge,
                    "element": atom.GetSymbol(),
                })

    return {
        "n_molecules": len(smiles_list),
        "n_charge_outliers": len(issues),
        "issues": issues[:20],
    }
