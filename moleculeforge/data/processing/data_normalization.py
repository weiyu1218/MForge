"""Data normalization and preprocessing utilities.

Standardizes molecular property ranges, handles missing values,
and prepares data for model training.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def normalize_logp(logp: float) -> float:
    """Normalize LogP to [0, 1] range (typical drug range: -5 to 10)."""
    lo, hi = -5.0, 10.0
    return (np.clip(logp, lo, hi) - lo) / (hi - lo)


def normalize_molecular_weight(mw: float) -> float:
    """Normalize molecular weight to [0, 1] (typical range: 100-900 Da)."""
    lo, hi = 100.0, 900.0
    return (np.clip(mw, lo, hi) - lo) / (hi - lo)


def normalize_tpsa(tpsa: float) -> float:
    """Normalize TPSA to [0, 1] (typical range: 0-200 Å²)."""
    return np.clip(tpsa, 0.0, 200.0) / 200.0


def normalize_qed(qed: float) -> float:
    """QED is already [0, 1]; just clamp."""
    return float(np.clip(qed, 0.0, 1.0))


def normalize_sa_score(sa: float) -> float:
    """Normalize SA score to [0, 1] (1=easy, 10=hard)."""
    return 1.0 - (np.clip(sa, 1.0, 10.0) - 1.0) / 9.0


def normalize_binding_affinity(pkd: float) -> float:
    """Normalize pKd/pKi/pIC50 to [0, 1] (typical: 4-12)."""
    lo, hi = 4.0, 12.0
    return (np.clip(pkd, lo, hi) - lo) / (hi - lo)


def impute_missing_property(
    values: np.ndarray, *, strategy: str = "median"
) -> np.ndarray:
    """Impute missing (NaN) values in a property array.

    Args:
        values: 1D array potentially containing NaN
        strategy: 'median', 'mean', or 'zero'
    """
    result = values.copy()
    nan_mask = np.isnan(result)

    if not nan_mask.any():
        return result

    if strategy == "median":
        fill = np.nanmedian(result)
    elif strategy == "mean":
        fill = np.nanmean(result)
    elif strategy == "zero":
        fill = 0.0
    else:
        raise ValueError(f"Unknown imputation strategy: {strategy}")

    result[nan_mask] = fill
    return result


def standard_scale(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Z-score normalization: (x - mu) / sigma.

    Returns (normalized, mu, sigma).
    """
    mu = float(np.mean(values))
    sigma = float(np.std(values))
    if sigma < 1e-9:
        return np.zeros_like(values), mu, sigma
    return (values - mu) / sigma, mu, sigma


def compute_molecular_properties_vector(
    smiles: str,
) -> dict[str, Optional[float]]:
    """Compute a standard set of molecular properties from SMILES.

    Returns dict with keys: mw, logp, tpsa, hbd, hba, rotatable_bonds,
    aromatic_rings, qed, fraction_csp3.
    """
    from rdkit import Chem
    from rdkit.Chem import Descriptors, QED, Lipinski

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}

    return {
        "mw": Descriptors.MolWt(mol),
        "logp": Descriptors.MolLogP(mol),
        "tpsa": Descriptors.TPSA(mol),
        "hbd": Lipinski.NumHDonors(mol),
        "hba": Lipinski.NumHAcceptors(mol),
        "rotatable_bonds": Descriptors.NumRotatableBonds(mol),
        "aromatic_rings": Descriptors.NumAromaticRings(mol),
        "qed": QED.default(mol),
        "fraction_csp3": Descriptors.FractionCSP3(mol),
    }
