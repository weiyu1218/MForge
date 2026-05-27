"""Molecular fingerprints for similarity search and model input.

Supported types:
- ECFP4 (Morgan, radius=2, 1024-bit) — primary
- MACCS (166-bit structural keys) — substructure screening
- RDKit topological (2048-bit) — fallback descriptor
"""

from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, RDKFingerprint


def compute_ecfp4(
    smiles: str, *, n_bits: int = 1024, radius: int = 2
) -> np.ndarray | None:
    """Morgan fingerprint (ECFP4 equivalent)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    arr = np.zeros(n_bits, dtype=np.int8)
    Chem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def compute_ecfp4_batch(
    smiles_list: list[str], *, n_bits: int = 1024, radius: int = 2
) -> list[np.ndarray | None]:
    """Batch ECFP4 computation."""
    return [compute_ecfp4(s, n_bits=n_bits, radius=radius) for s in smiles_list]


def compute_maccs(smiles: str) -> np.ndarray | None:
    """MACCS 166-bit structural key."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = MACCSkeys.GenMACCSKeys(mol)
    arr = np.zeros(167, dtype=np.int8)  # MACCS is 167 bits (1-indexed)
    Chem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr[1:]  # drop bit 0 (always 0)


def compute_rdkit_topo(smiles: str, *, n_bits: int = 2048) -> np.ndarray | None:
    """RDKit topological (path-based) fingerprint."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = RDKFingerprint(mol, fpSize=n_bits)
    arr = np.zeros(n_bits, dtype=np.int8)
    Chem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def tanimoto_similarity(fp_a: np.ndarray, fp_b: np.ndarray) -> float:
    """Tanimoto (Jaccard) similarity between two binary fingerprint arrays."""
    intersection = np.logical_and(fp_a, fp_b).sum()
    union = np.logical_or(fp_a, fp_b).sum()
    return float(intersection / union) if union > 0 else 0.0


def tanimoto_distance_matrix(fps: list[np.ndarray]) -> np.ndarray:
    """Compute pairwise Tanimoto distance matrix."""
    n = len(fps)
    dist = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            d = 1.0 - tanimoto_similarity(fps[i], fps[j])
            dist[i, j] = dist[j, i] = d
    return dist
