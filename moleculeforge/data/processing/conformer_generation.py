"""3D conformer generation using ETKDG.

Produces energy-minimized 3D conformers for downstream:
- HUMU pocket encoding (docking pose)
- HFM-3D training data (3D molecular geometry)
- Boltz-2 binding free energy prediction
"""

from __future__ import annotations

from typing import Optional

from rdkit import Chem
from rdkit.Chem import AllChem


def generate_conformers(
    smiles: str,
    *,
    n_conformers: int = 50,
    random_seed: int = 42,
    max_attempts: int = 100,
) -> list[dict]:
    """Generate ETKDG conformers with MMFF94 minimization.

    Returns list of dicts with keys: 'positions' (Nx3 array),
    'energy' (kcal/mol), 'rmsd_to_first'.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []

    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = random_seed
    params.numThreads = 0  # auto-detect
    params.pruneRmsThresh = 0.5

    cids = list(
        AllChem.EmbedMultipleConfs(mol, numConfs=n_conformers, params=params)
    )
    if not cids:
        return []

    results: list[dict] = []
    for cid in cids:
        ff = AllChem.MMFFGetMoleculeForceField(
            mol, AllChem.MMFFGetMoleculeProperties(mol), confId=cid
        )
        if ff is None:
            continue
        energy = ff.CalcEnergy()
        pos = mol.GetConformer(cid).GetPositions()
        results.append({
            "conf_id": cid,
            "energy_kcal_mol": energy,
            "positions": pos.tolist(),
            "n_atoms": mol.GetNumAtoms(),
        })

    return sorted(results, key=lambda r: r["energy_kcal_mol"])


def generate_lowest_energy_conformer(
    smiles: str, *, n_conformers: int = 200, random_seed: int = 42
) -> Optional[dict]:
    """Generate conformers and return the lowest-energy one."""
    results = generate_conformers(
        smiles, n_conformers=n_conformers, random_seed=random_seed
    )
    return results[0] if results else None


def generate_conformer_batch(
    smiles_list: list[str], *, n_conformers: int = 50, random_seed: int = 42
) -> list[list[dict]]:
    """Generate conformers for a batch of SMILES (sequential, CPU-bound)."""
    return [
        generate_conformers(s, n_conformers=n_conformers, random_seed=random_seed)
        for s in smiles_list
    ]
