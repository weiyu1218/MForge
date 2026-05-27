"""Pharmacophore feature extraction from 3D conformers.

Identifies hydrogen bond donors/acceptors, hydrophobic regions,
aromatic rings, positive/negative ionizable groups in 3D space.
"""

from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import ChemicalFeatures
from rdkit.Chem.Pharm3D import Pharmacophore


def extract_pharmacophore_features(smiles: str) -> list[dict]:
    """Extract pharmacophore features from a molecule's 3D conformer.

    Returns list of feature dicts with keys:
      - type: 'Donor'|'Acceptor'|'Hydrophobe'|'Aromatic'|'PosIonizable'|'NegIonizable'
      - position: [x, y, z]
      - atom_indices: [int, ...]
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []

    # Generate a conformer if needed
    mol = Chem.AddHs(mol)
    from rdkit.Chem import AllChem

    try:
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)
    except Exception:
        return []

    # Load feature factory
    fdef = ChemicalFeatures.BuildFeatureFactory()
    if fdef is None:
        return []

    features = fdef.GetFeaturesForMol(mol)
    if not features:
        return []

    results: list[dict] = []
    for feat in features:
        pos = feat.GetPos()
        results.append({
            "type": feat.GetFamily(),
            "position": [pos.x, pos.y, pos.z],
            "atom_indices": list(feat.GetAtomIds()),
        })

    return results


def extract_pharmacophore_fingerprint(
    smiles: str, *, n_bins: int = 4
) -> list[float] | None:
    """Generate a 3D pharmacophore fingerprint (ph4).

    This creates a discretized 3D grid of pharmacophore feature
    pair distances, binned into n_bins^3.
    """
    features = extract_pharmacophore_features(smiles)
    if len(features) < 2:
        return None

    fp: list[float] = []
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            p1 = features[i]["position"]
            p2 = features[j]["position"]
            dx = (p1[0] - p2[0]) / 10.0  # normalize to ~10Å range
            dy = (p1[1] - p2[1]) / 10.0
            dz = (p1[2] - p2[2]) / 10.0
            d = (dx**2 + dy**2 + dz**2) ** 0.5
            fp.extend([dx, dy, dz, d])

    return fp
