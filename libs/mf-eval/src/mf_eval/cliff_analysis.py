"""Activity cliff analysis utilities."""
from __future__ import annotations

import torch


def find_activity_cliffs(
    smiles: list[str],
    activities: list[float],
    similarity_threshold: float,
    activity_delta_threshold: float,
) -> list[dict]:
    if len(smiles) != len(activities):
        raise ValueError("smiles and activities must have the same length")
    try:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import rdMolDescriptors
    except ImportError as exc:
        raise RuntimeError("RDKit is required for activity cliff analysis") from exc

    molecules = [Chem.MolFromSmiles(item) for item in smiles]
    if any(mol is None for mol in molecules):
        raise ValueError("activity cliff analysis requires valid SMILES")
    fingerprints = [
        rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, 2048)
        for mol in molecules
    ]
    cliffs = []
    for i in range(len(fingerprints)):
        for j in range(i + 1, len(fingerprints)):
            similarity = DataStructs.TanimotoSimilarity(fingerprints[i], fingerprints[j])
            activity_delta = abs(float(activities[i]) - float(activities[j]))
            if (
                similarity >= similarity_threshold
                and activity_delta >= activity_delta_threshold
            ):
                cliffs.append(
                    {
                        "i": i,
                        "j": j,
                        "similarity": float(similarity),
                        "activity_delta": activity_delta,
                    }
                )
    return cliffs


def cliff_separation_auroc(embeddings, cliff_labels: list[bool]) -> float | None:
    embedding_tensor = (
        embeddings.float()
        if isinstance(embeddings, torch.Tensor)
        else torch.tensor(embeddings, dtype=torch.float32)
    )
    labels = torch.tensor(cliff_labels, dtype=torch.bool, device=embedding_tensor.device)
    if embedding_tensor.ndim != 2 or embedding_tensor.shape[0] != labels.numel():
        raise ValueError("embeddings rows must match cliff_labels")
    if labels.sum() == 0 or (~labels).sum() == 0:
        return None
    positive_centroid = embedding_tensor[labels].mean(dim=0, keepdim=True)
    scores = -torch.linalg.norm(embedding_tensor - positive_centroid, dim=1)
    return _binary_auroc(scores, labels)


def _binary_auroc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    positive = scores[labels]
    negative = scores[~labels]
    comparisons = positive[:, None] - negative[None, :]
    wins = (comparisons > 0).float().sum()
    ties = (comparisons == 0).float().sum() * 0.5
    return float(((wins + ties) / comparisons.numel()).cpu().item())
