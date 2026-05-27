"""EvoMol-RL: RL-genetic algorithm hybrid with Pareto hypervolume reward."""
from __future__ import annotations

import inspect

from mf_core.plugins.generator import GeneratorPlugin
from mf_core.types.humu import IntentCone
from mf_core.types.molecule import Molecule


class EvoMolRLGenerator(GeneratorPlugin):
    def __init__(self, checkpoint_path: str = "", device: str = "cpu", runner=None):
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.runner = runner

    async def generate(
        self,
        batch_size: int,
        intent_cone: IntentCone | None = None,
        **kwargs,
    ) -> list[Molecule]:
        """Generate molecules via RL-guided evolutionary search with Pareto optimization."""
        if self.runner is None:
            raise RuntimeError("EVOMOL_RUNNER is required")
        result = self.runner.generate(
            batch_size=batch_size,
            intent_cone=intent_cone,
            checkpoint_path=self.checkpoint_path,
            device=self.device,
            **kwargs,
        )
        if inspect.isawaitable(result):
            result = await result
        return [_to_molecule(item) for item in result]

    async def info(self) -> dict:
        return {
            "name": "evomol_rl",
            "version": "0.1.0",
            "description": "RL-genetic algorithm hybrid with Pareto hypervolume reward",
            "supported_properties": ["qed", "logp", "sa_score", "mw"],
            "max_batch_size": 128,
            "supports_streaming": True,
            "requires_gpu": False,
        }


def _to_molecule(item) -> Molecule:
    if isinstance(item, Molecule):
        return item
    if not isinstance(item, dict) or not item.get("smiles"):
        raise ValueError("EvoMol-RL runner output must contain a smiles field")
    return Molecule(
        smiles=str(item["smiles"]),
        sdf_bytes=item.get("sdf_bytes"),
        metadata={
            key: str(value)
            for key, value in item.get("metadata", {}).items()
            if value is not None
        },
    )
