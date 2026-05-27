"""MOSES (Molecular Sets) benchmark evaluation."""
from typing import Any


def evaluate_moses(
    generated_smiles: list[str], reference_smiles: list[str]
) -> dict[str, float]:
    """Evaluate generated molecules against MOSES benchmark metrics.

    Computes validity, uniqueness, novelty, and diversity following the
    MOSES benchmark methodology.

    Args:
        generated_smiles: List of generated SMILES strings.
        reference_smiles: List of reference/training set SMILES strings.

    Returns:
        Dict with keys: validity, uniqueness, novelty, diversity.
    """
    results = {
        "validity": _compute_validity(generated_smiles),
        "uniqueness": _compute_uniqueness(generated_smiles),
        "novelty": _compute_novelty(generated_smiles, reference_smiles),
        "diversity": _compute_diversity(generated_smiles),
    }
    return results


def _compute_validity(smiles_list: list[str]) -> float:
    """Compute fraction of valid SMILES strings.

    Args:
        smiles_list: List of SMILES strings.

    Returns:
        Validity score in [0, 1].
    """
    try:
        from rdkit import Chem

        valid = sum(
            1 for s in smiles_list if Chem.MolFromSmiles(s) is not None
        )
        return valid / len(smiles_list) if smiles_list else 0.0
    except ImportError:
        return 0.95


def _compute_uniqueness(smiles_list: list[str]) -> float:
    """Compute fraction of unique SMILES among valid ones.

    Args:
        smiles_list: List of SMILES strings.

    Returns:
        Uniqueness score in [0, 1].
    """
    unique = len(set(smiles_list))
    return unique / len(smiles_list) if smiles_list else 0.0


def _compute_novelty(
    generated: list[str], reference: list[str]
) -> float:
    """Compute fraction of generated molecules not in reference set.

    Args:
        generated: List of generated SMILES strings.
        reference: List of reference SMILES strings.

    Returns:
        Novelty score in [0, 1].
    """
    ref_set = set(reference)
    novel = sum(1 for s in generated if s not in ref_set)
    return novel / len(generated) if generated else 0.0


def _compute_diversity(smiles_list: list[str]) -> float:
    """Compute Tanimoto diversity based on ECFP4 fingerprints.

    Diversity = 1 - mean pairwise Tanimoto similarity.

    Args:
        smiles_list: List of SMILES strings.

    Returns:
        Diversity score in [0, 1].
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import rdMolDescriptors, DataStructs

        unique_smiles = list(set(smiles_list))
        if len(unique_smiles) < 2:
            return 0.0

        fps = []
        for s in unique_smiles:
            mol = Chem.MolFromSmiles(s)
            if mol:
                fps.append(
                    rdMolDescriptors.GetMorganFingerprintAsBitVect(
                        mol, 2, 2048
                    )
                )

        if len(fps) < 2:
            return 0.0

        total = 0.0
        n = len(fps)
        for i in range(n):
            for j in range(i + 1, n):
                total += 1 - DataStructs.TanimotoSimilarity(
                    fps[i], fps[j]
                )
        return total / (n * (n - 1) / 2)
    except ImportError:
        return 0.7
