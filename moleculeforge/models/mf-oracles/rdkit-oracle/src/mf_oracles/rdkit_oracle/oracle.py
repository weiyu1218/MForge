"""RDKitOracle — L0 oracle using RDKit descriptors and scores."""
from __future__ import annotations

from enum import Enum


class OracleLevel(Enum):
    L0 = "L0"
    L1 = "L1"
    L2 = "L2"


class RDKitOracle:
    """L0 oracle computing fast RDKit-based scores."""

    def __init__(self):
        self.name = "rdkit_oracle_l0"
        self.oracle_level = OracleLevel.L0

    async def predict(self, mol, mode: str = "any") -> float:
        from mf_oracles.rdkit_oracle.scorer import compute_composite_score
        return compute_composite_score(mol.smiles)

    async def predict_with_uncertainty(self, mol, mode: str = "any") -> tuple[float, float]:
        score = await self.predict(mol, mode)
        return score, 0.0

    async def evaluate(self, molecules: list[str], properties: list[str]) -> dict:
        from mf_oracles.rdkit_oracle.scorer import (
            compute_composite_score,
            has_pains_alert,
            pains_alerts,
        )

        result = {}
        for smiles in molecules:
            score = compute_composite_score(smiles)
            values = {prop: score for prop in properties}
            values["pains_alert"] = has_pains_alert(smiles)
            values["pains_alerts"] = pains_alerts(smiles)
            result[smiles] = values
        return result
