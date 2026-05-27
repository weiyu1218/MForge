"""Fragment decomposition using BRICS and RECAP rules.

Produces fragment vocabularies for FragFM and CReM-3D generators.
"""

from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import BRICS, Recap


def brics_decompose(smiles: str) -> list[str]:
    """Decompose molecule into BRICS fragments.

    BRICS (Breaking of Retrosynthetically Interesting Chemical Substructures)
    cuts at 16 bond types corresponding to common synthetic reactions.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    fragments = BRICS.BRICSDecompose(mol, keepNonLeafNodes=True)
    return list(fragments)


def recap_decompose(smiles: str) -> list[str]:
    """Decompose molecule using RECAP rules.

    RECAP (Retrosynthetic Combinatorial Analysis Procedure) cleaves
    at 11 bond types: amide, ester, amine, urea, ether, olefin,
    quaternary N, aromatic N-aliphatic C, lactam N-aliphatic C,
    aromatic C-aromatic C, and sulfonamide.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    tree = Recap.RecapDecompose(mol)
    fragments: list[str] = []

    def _collect(node):
        if node.mol is not None:
            fragments.append(Chem.MolToSmiles(node.mol, canonical=True))
        for child in node.children:
            _collect(child)

    _collect(tree)
    return fragments


def extract_scaffold(smiles: str) -> str | None:
    """Extract Bemis-Murcko scaffold from a molecule."""
    from rdkit.Chem.Scaffolds import MurckoScaffold

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    if scaffold is None:
        return None
    return Chem.MolToSmiles(scaffold, canonical=True)


def extract_r_groups(smiles: str) -> list[str]:
    """Extract R-group substituents by removing the scaffold."""
    scaffold_smi = extract_scaffold(smiles)
    if scaffold_smi is None:
        return []

    mol = Chem.MolFromSmiles(smiles)
    scaffold_mol = Chem.MolFromSmiles(scaffold_smi)
    if mol is None or scaffold_mol is None:
        return []

    # Find scaffold substructure and remove it
    match = mol.GetSubstructMatch(scaffold_mol)
    if not match:
        return []

    rw_mol = Chem.RWMol(mol)
    atoms_to_remove = set(match)
    # Keep attachment points
    for idx in sorted(atoms_to_remove, reverse=True):
        rw_mol.RemoveAtom(idx)

    try:
        frag_smi = Chem.MolToSmiles(rw_mol.GetMol(), canonical=True)
        return [frag_smi] if frag_smi else []
    except Exception:
        return []
