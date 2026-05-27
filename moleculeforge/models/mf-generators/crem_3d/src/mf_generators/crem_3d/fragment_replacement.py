"""CReM (Chemically Reasonable Mutations) — fragment replacement core."""
from __future__ import annotations

try:
    from rdkit import Chem
    from rdkit.Chem import RWMol
    _RDKIT = True
except ImportError:
    _RDKIT = False


def get_attachment_points(mol) -> list[int]:
    if not _RDKIT:
        raise ImportError("RDKit required: conda install -c conda-forge rdkit")
    if mol is None:
        return []

    points = []
    for atom in mol.GetAtoms():
        if atom.GetDegree() == 1 and atom.GetAtomicNum() != 1:
            points.append(atom.GetIdx())
            continue
        if atom.GetDegree() == 2:
            non_h_neighbors = sum(
                1 for n in atom.GetNeighbors() if n.GetAtomicNum() != 1
            )
            if non_h_neighbors >= 2 and atom.GetSymbol() in ("C", "N", "O"):
                points.append(atom.GetIdx())

    return points if points else [0]


def replace_fragment(mol, idx: int, frag_smi: str):
    if not _RDKIT:
        raise ImportError("RDKit required")
    if mol is None:
        return None
    if idx < 0 or idx >= mol.GetNumAtoms():
        raise ValueError("fragment replacement attachment index is out of range")
    if not isinstance(frag_smi, str) or not frag_smi:
        raise ValueError("fragment_smiles is required")

    rwmol = RWMol(mol)
    frag = Chem.MolFromSmiles(frag_smi.replace("*", "[H]"))
    if frag is None:
        raise ValueError(f"fragment_smiles is invalid: {frag_smi}")

    combined = Chem.CombineMols(rwmol, frag)
    editable = RWMol(combined)
    n_orig = mol.GetNumAtoms()
    editable.AddBond(idx, n_orig, Chem.BondType.SINGLE)

    new_mol = editable.GetMol()
    try:
        Chem.SanitizeMol(new_mol)
    except Exception:
        return None
    return new_mol
