"""MMPT-RAG (Matched Molecular Pair Transformer + Retrieval-Augmented Generation)."""
from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from mf_core.types.molecule import MoleculeModel


class MMPTRAGGenerator:
    name = "mmpt_rag"
    version = "0.1.0"
    supported_modes = ["lead_opt", "scaffold_hop"]

    def __init__(self, mmp_database: list[dict] | None = None, index_path: str = ""):
        self.index_path = index_path
        loaded = _load_index(index_path) if index_path else None
        self.mmp_database = mmp_database or loaded or [
            {"pattern": "F", "replacement": "Cl"},
            {"pattern": "Cl", "replacement": "Br"},
            {"pattern": "OC", "replacement": "OCC"},
        ]

    async def generate(
        self,
        hciv: Any,
        cone: Any,
        cig: Any,
        n_samples: int = 10,
        seed: int | None = None,
    ) -> AsyncIterator[MoleculeModel]:
        seeds = ["c1ccccc1F", "CC(=O)Oc1ccccc1F", "Fc1ccc(N)cc1"]

        count = 0
        for seed_smi in seeds:
            for mmp in self.mmp_database:
                if count >= n_samples:
                    return
                new_smi = self._simple_replace(seed_smi, mmp["pattern"], mmp["replacement"])
                if new_smi is None:
                    continue
                yield MoleculeModel(
                    smiles=new_smi,
                    canonical_smiles=new_smi,
                    generator_name=self.name,
                    humu_embedding=None,
                )
                count += 1

    def _simple_replace(
        self,
        mol_smi: str,
        pattern_smi: str,
        replacement_smi: str,
    ) -> str | None:
        if pattern_smi not in mol_smi:
            return None
        return mol_smi.replace(pattern_smi, replacement_smi, 1)


def _load_index(index_path: str) -> list[dict]:
    path = Path(index_path)
    if not path.exists():
        raise FileNotFoundError(f"MMPT index artifact not found: {index_path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("MMPT index artifact must be a JSON object")
    transforms = payload.get("transforms")
    if not isinstance(transforms, list) or not transforms:
        raise ValueError("MMPT index artifact requires transforms")
    normalized = []
    for index, transform in enumerate(transforms):
        if not isinstance(transform, dict):
            raise ValueError("MMPT transform must be a JSON object")
        pattern = transform.get("pattern")
        replacement = transform.get("replacement")
        if not isinstance(pattern, str) or not pattern:
            raise ValueError("MMPT transform requires pattern")
        if not isinstance(replacement, str) or not replacement:
            raise ValueError("MMPT transform requires replacement")
        normalized.append(
            {
                "id": str(transform.get("id", index)),
                "pattern": pattern,
                "replacement": replacement,
            }
        )
    return normalized
