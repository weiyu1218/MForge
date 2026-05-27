"""RDKit-Random generator (lightweight baseline for testing and fallback)."""
from __future__ import annotations
import random
from typing import Any, AsyncIterator
from mf_core.types.molecule import MoleculeModel
from mf_generators.rdkit_random.mutator import random_mutate

try:
    from rdkit import Chem
    _RDKIT = True
except ImportError:
    _RDKIT = False


TEMPLATE_SMILES = [
    "CCO",
    "c1ccccc1",
    "CC(=O)Oc1ccccc1C(=O)O",
    "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
    "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
    "CCN(CC)CC",
    "OC(=O)c1ccccc1O",
    "Nc1ccc(S(N)(=O)=O)cc1",
]


class RDKitRandomGenerator:
    name = "rdkit_random"
    version = "0.1.0"
    supported_modes = ["hit_finding", "lead_opt", "scaffold_hop", "baseline"]

    def __init__(self, seed: int | None = None):
        self.seed = seed
        if seed is not None:
            random.seed(seed)

    async def generate(
        self,
        hciv: Any,
        cone: Any,
        cig: Any,
        n_samples: int = 10,
        seed: int | None = None,
    ) -> AsyncIterator[MoleculeModel]:
        if seed is not None:
            random.seed(seed)
        elif self.seed is not None:
            random.seed(self.seed)

        for i in range(n_samples):
            template = random.choice(TEMPLATE_SMILES)
            mutated = random_mutate(template, n_mutations=random.randint(1, 3))

            if mutated is None:
                mutated = template

            if _RDKIT:
                mol = Chem.MolFromSmiles(mutated)
                canonical = Chem.MolToSmiles(mol, canonical=True) if mol else mutated
            else:
                canonical = mutated

            yield MoleculeModel(
                smiles=mutated,
                canonical_smiles=canonical,
                generator_name=self.name,
                humu_embedding=None,
            )
