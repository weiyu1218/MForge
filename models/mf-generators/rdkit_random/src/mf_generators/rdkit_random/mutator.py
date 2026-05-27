"""Molecular mutation operations (RDKit-based)."""
from __future__ import annotations
import random

try:
    from rdkit import Chem
    from rdkit.Chem import RWMol
    _RDKIT = True
except ImportError:
    _RDKIT = False


ATOM_VOCAB = ["C", "N", "O", "F", "S", "Cl"]


def mutate_atom_type(mol):
    if not _RDKIT or mol is None:
        return None

    try:
        rwmol = RWMol(mol)
        atoms = list(rwmol.GetAtoms())
        if not atoms:
            return None

        candidates = [a for a in atoms if a.GetAtomicNum() != 1]
        if not candidates:
            return None
        atom = random.choice(candidates)

        current = atom.GetSymbol()
        new_choices = [s for s in ATOM_VOCAB if s != current]
        new_symbol = random.choice(new_choices)

        new_atomic_num = Chem.GetPeriodicTable().GetAtomicNumber(new_symbol)
        rwmol.GetAtomWithIdx(atom.GetIdx()).SetAtomicNum(new_atomic_num)

        new_mol = rwmol.GetMol()
        try:
            Chem.SanitizeMol(new_mol)
            return new_mol
        except Exception:
            return None
    except Exception:
        return None


def random_mutate(smiles: str, n_mutations: int = 1) -> str | None:
    if not _RDKIT:
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    for _ in range(n_mutations):
        new_mol = mutate_atom_type(mol)
        if new_mol is None:
            continue
        mol = new_mol

    try:
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None
