"""CReM-3D: Chemically Reasonable Mutations in 3D."""
from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from mf_core.plugins.generator import GeneratorPlugin
from mf_core.types.humu import IntentCone
from mf_core.types.molecule import Molecule
from mf_generators.crem_3d.fragment_replacement import get_attachment_points, replace_fragment

try:
    from rdkit import Chem
except ImportError:  # pragma: no cover
    Chem = None


class CReM3DGenerator(GeneratorPlugin):
    name = "crem_3d"

    def __init__(
        self,
        mmp_db_path: str = "",
        mode: str = "production_real",
    ):
        if mode not in {"production_real", "local_demo"}:
            raise ValueError(f"Unknown CReM3DGenerator mode: {mode}")
        self.mode = mode
        self.mmp_db_path = mmp_db_path
        if mmp_db_path:
            self.mutations = self._load_mmp_database(mmp_db_path)
        elif mode == "local_demo":
            self.mutations = [
                {
                    "id": "local_demo_fluoro",
                    "seed_smiles": "c1ccccc1",
                    "product": "Fc1ccccc1",
                }
            ]
        else:
            raise RuntimeError(
                "CReM-3D production generation requires an MMP database artifact"
            )

    def _load_mmp_database(self, mmp_db_path: str) -> list[dict[str, object]]:
        path = Path(mmp_db_path)
        if not path.exists():
            raise FileNotFoundError(f"CReM MMP database artifact not found: {mmp_db_path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("CReM MMP database artifact must be a JSON object")
        mutations = payload.get("mutations")
        if not isinstance(mutations, list) or not mutations:
            raise ValueError("CReM MMP database artifact requires mutations")
        return [
            self._normalize_mutation(idx, mutation)
            for idx, mutation in enumerate(mutations)
        ]

    def _normalize_mutation(
        self,
        idx: int,
        mutation: object,
    ) -> dict[str, object]:
        if not isinstance(mutation, Mapping):
            raise ValueError("CReM mutation record must be a JSON object")
        product = mutation.get("product")
        fragment_smiles = mutation.get("fragment_smiles")
        if (not isinstance(product, str) or not product) and (
            not isinstance(fragment_smiles, str) or not fragment_smiles
        ):
            raise ValueError("CReM mutation record requires product or fragment_smiles")
        seed_smiles = mutation.get("seed_smiles", "")
        if seed_smiles and not isinstance(seed_smiles, str):
            raise ValueError("CReM mutation seed_smiles must be a string")
        attachment_index = mutation.get("attachment_index")
        if attachment_index is not None and not isinstance(attachment_index, int):
            raise ValueError("CReM mutation attachment_index must be an integer")
        return {
            "id": str(mutation.get("id", idx)),
            "seed_smiles": self._canonical_smiles(seed_smiles) if seed_smiles else "",
            "fragment_smiles": fragment_smiles or "",
            "attachment_index": attachment_index,
            "product": self._canonical_smiles(product) if product else "",
        }

    def _canonical_smiles(self, smiles: str) -> str:
        if Chem is None:
            raise ImportError("RDKit is required for CReM validity checks")
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"CReM mutation produced invalid SMILES: {smiles}")
        return Chem.MolToSmiles(mol)

    async def generate(
        self,
        batch_size: int,
        intent_cone: IntentCone | None = None,
        **kwargs,
    ) -> list[Molecule]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        seed_smiles = kwargs.get("seed_smiles", "")
        if seed_smiles and not isinstance(seed_smiles, str):
            raise ValueError("seed_smiles must be a string")
        seed_key = self._canonical_smiles(seed_smiles) if seed_smiles else ""
        mutations = [
            mutation for mutation in self.mutations
            if not seed_key or mutation["seed_smiles"] in {"", seed_key}
        ]
        if not mutations:
            raise RuntimeError("CReM MMP database contains no mutation for seed_smiles")
        results = []
        for i in range(batch_size):
            mutation = mutations[i % len(mutations)]
            smiles = self._product_for_mutation(mutation, seed_key)
            fragment_replacement = bool(mutation.get("fragment_smiles"))
            results.append(
                Molecule(
                    smiles=smiles,
                    metadata={
                        "generator_name": self.name,
                        "mmp_database": self.mmp_db_path,
                        "mutation_id": mutation["id"],
                        "fragment_replacement": str(fragment_replacement).lower(),
                    },
                )
            )
        return results

    def _product_for_mutation(self, mutation: dict[str, object], seed_key: str) -> str:
        fragment_smiles = mutation.get("fragment_smiles")
        if not fragment_smiles:
            product = mutation.get("product")
            if not isinstance(product, str) or not product:
                raise RuntimeError("CReM mutation product is required")
            return product
        seed_smiles = seed_key or mutation.get("seed_smiles")
        if not isinstance(seed_smiles, str) or not seed_smiles:
            raise RuntimeError("CReM fragment replacement requires seed_smiles")
        seed_mol = Chem.MolFromSmiles(seed_smiles)
        if seed_mol is None:
            raise ValueError(f"CReM seed_smiles is invalid: {seed_smiles}")
        attachment_points = get_attachment_points(seed_mol)
        attachment_index = mutation.get("attachment_index")
        if attachment_index is None:
            attachment_index = attachment_points[0]
        if attachment_index not in attachment_points:
            raise RuntimeError("CReM attachment_index is not valid for seed_smiles")
        product_mol = replace_fragment(seed_mol, int(attachment_index), str(fragment_smiles))
        if product_mol is None:
            raise RuntimeError("CReM fragment replacement failed RDKit sanitization")
        return Chem.MolToSmiles(product_mol)

    async def info(self) -> dict:
        return {
            "name": "crem_3d",
            "version": "0.1.0",
            "supports_streaming": False,
            "requires_gpu": False,
        }
