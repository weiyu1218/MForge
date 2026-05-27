"""Duplicate detection using canonical SMILES and Tanimoto similarity.

Identifies exact duplicates and near-duplicates (Tanimoto >= threshold)
in molecular datasets before ingestion.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


def find_exact_duplicates(smiles_list: list[str]) -> dict[str, Any]:
    """Find exact duplicate SMILES (after canonicalization).

    Returns dict with: n_total, n_unique, n_duplicates, duplicate_groups.
    """
    from rdkit import Chem

    canonical: dict[str, list[int]] = defaultdict(list)

    for idx, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            can = Chem.MolToSmiles(mol, canonical=True)
            canonical[can].append(idx)

    duplicates = {k: v for k, v in canonical.items() if len(v) > 1}

    return {
        "n_total": len(smiles_list),
        "n_unique": len(canonical),
        "n_duplicate_groups": len(duplicates),
        "n_duplicate_molecules": sum(len(v) - 1 for v in duplicates.values()),
        "duplicate_groups": duplicates,
    }


def find_near_duplicates(
    smiles_list: list[str],
    *,
    threshold: float = 0.95,
    batch_size: int = 1000,
) -> dict[str, Any]:
    """Find near-duplicate pairs using ECFP4 Tanimoto similarity.

    Molecules with Tanimoto similarity >= threshold are flagged as
    near-duplicates (potential registration errors or congeneric series).

    Returns dict with: n_total, n_pairs, pairs (list of (idx_a, idx_b, similarity)).
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    # Compute valid fingerprints
    valid_indices: list[int] = []
    fingerprints: list = []
    for idx, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
            fingerprints.append(fp)
            valid_indices.append(idx)

    # Bulk Tanimoto calculation (within batches)
    pairs: list[tuple[int, int, float]] = []
    n = len(fingerprints)

    for i in range(0, n, batch_size):
        batch_end = min(i + batch_size, n)
        sims = AllChem.GetBulkTanimotoSimilarity(
            fingerprints[i], fingerprints[i + 1 : batch_end]
        )
        for j_offset, sim in enumerate(sims):
            if sim >= threshold:
                j = i + 1 + j_offset
                pairs.append((valid_indices[i], valid_indices[j], float(sim)))

    return {
        "n_total": len(smiles_list),
        "n_valid": n,
        "n_near_duplicate_pairs": len(pairs),
        "threshold": threshold,
        "pairs": pairs,
    }


def compute_dataset_diversity(
    smiles_list: list[str], *, n_sample: int = 1000
) -> dict[str, Any]:
    """Estimate molecular diversity of a dataset.

    Randomly samples n_sample molecules and computes pairwise
    Tanimoto distance statistics.
    """
    rng = np.random.default_rng(42)
    n = len(smiles_list)
    indices = rng.choice(n, size=min(n_sample, n), replace=False)

    from rdkit import Chem
    from rdkit.Chem import AllChem

    fps = []
    for idx in indices:
        mol = Chem.MolFromSmiles(smiles_list[idx])
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
            fps.append(fp)

    if len(fps) < 2:
        return {"n_sample": len(fps), "mean_similarity": 0.0, "std_similarity": 0.0}

    # Sample pairs
    n_pairs = min(5000, len(fps) * (len(fps) - 1) // 2)
    pair_indices = rng.choice(len(fps), size=(n_pairs, 2))
    sims = []
    for a, b in pair_indices:
        if a != b:
            sim = AllChem.DataStructs.TanimotoSimilarity(fps[a], fps[b])
            sims.append(sim)

    sims_arr = np.array(sims)
    return {
        "n_sample": len(fps),
        "n_pairs": len(sims),
        "mean_similarity": float(sims_arr.mean()),
        "std_similarity": float(sims_arr.std()),
        "median_similarity": float(np.median(sims_arr)),
        "min_similarity": float(sims_arr.min()),
        "max_similarity": float(sims_arr.max()),
    }
