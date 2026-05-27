"""Canonical SMILES generation and standardization.

Produces a single canonical representation for each molecule,
essential for deduplication and consistent database lookups.
"""

from __future__ import annotations

from typing import Optional

from rdkit import Chem
from rdkit.Chem import MolStandardize


def canonicalize_smiles(smiles: str, /, *, remove_stereo: bool = False) -> Optional[str]:
    """Return canonical SMILES or None for unparseable input.

    Steps:
    1. Parse with RDKit (fail-soft)
    2. Remove stereochemistry if requested
    3. Normalize: charge, tautomer, fragment selection
    4. Generate canonical SMILES (isomeric unless remove_stereo)
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    if remove_stereo:
        Chem.RemoveStereochemistry(mol)

    # RDKit MolStandardize pipeline
    normalizer = MolStandardize.normalize.Normalizer()
    mol = normalizer.normalize(mol)

    tautomer = MolStandardize.tautomer.TautomerCanonicalizer()
    mol = tautomer.canonicalize(mol)

    fragment = MolStandardize.fragment.LargestFragmentChooser()
    mol = fragment.choose(mol)

    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=not remove_stereo)


def canonicalize_batch(
    smiles_list: list[str], *, remove_stereo: bool = False
) -> list[Optional[str]]:
    """Canonicalize a batch of SMILES strings."""
    return [canonicalize_smiles(s, remove_stereo=remove_stereo) for s in smiles_list]


def smiles_to_inchi(smiles: str) -> Optional[str]:
    """Convert SMILES to InChI (for cross-database linking)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToInchi(mol)


def smiles_to_inchikey(smiles: str) -> Optional[str]:
    """Convert SMILES to InChI Key (14-char hash)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.InchiToInchiKey(Chem.MolToInchi(mol))
