"""RDKitOracle — L0 oracle using RDKit descriptors and scores."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mf_core.types.molecule import MoleculeModel


class OracleLevel(Enum):
    L0 = "L0"
    L1 = "L1"
    L2 = "L2"


class RDKitOracle:
    """L0 oracle computing fast RDKit-based scores."""

    def __init__(self) -> None:
        self.name = "rdkit_oracle_l0"
        self.oracle_level = OracleLevel.L0

    async def predict(self, mol: MoleculeModel, mode: str = "any") -> float:
        from mf_oracles.rdkit_oracle.scorer import compute_composite_score

        return compute_composite_score(mol.smiles)

    async def predict_with_uncertainty(
        self,
        mol: MoleculeModel,
        mode: str = "any",
    ) -> tuple[float, float]:
        score = await self.predict(mol, mode)
        return score, 0.0

    async def evaluate(self, molecules: list[str], properties: list[str]) -> dict:
        from mf_oracles.rdkit_oracle.scorer import (
            compute_composite_score,
            compute_qed,
            compute_sa_score,
            count_lipinski_violations,
            has_pains_alert,
            pains_alerts,
        )
        from rdkit import Chem
        from rdkit.Chem import Crippen

        result = {}
        for smiles in molecules:
            molecule = Chem.MolFromSmiles(smiles)
            if molecule is None:
                raise ValueError(f"invalid SMILES: {smiles}")
            qed = compute_qed(smiles)
            sa_score = compute_sa_score(smiles)
            if qed is None or sa_score is None:
                raise RuntimeError(f"RDKit failed to compute L0 metrics for {smiles}")
            values = {
                "qed": float(qed),
                "sa_score": float(sa_score),
                "logp": float(Crippen.MolLogP(molecule)),
                "lipinski_violations": count_lipinski_violations(smiles),
                "admet_score": compute_composite_score(smiles),
            }
            values["pains_alert"] = has_pains_alert(smiles)
            values["pains_alerts"] = pains_alerts(smiles)
            result[smiles] = values
        return result
