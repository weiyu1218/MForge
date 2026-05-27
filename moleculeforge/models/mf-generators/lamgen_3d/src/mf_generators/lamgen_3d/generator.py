"""LaMGen-3D: Multi-target attention-based 3D molecular generation."""
import inspect

from mf_core.plugins.generator import GeneratorPlugin
from mf_core.types.humu import IntentCone
from mf_core.types.molecule import Molecule


class LaMGen3DGenerator(GeneratorPlugin):
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
        """Generate molecules via multi-target attention on Lorentz manifold."""
        if self.runner is None:
            raise RuntimeError("LAMGEN_RUNNER is required")
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
            "name": "lamgen_3d",
            "version": "0.1.0",
            "description": "Multi-target LLM-based 3D molecular generation",
            "supported_properties": ["qed", "logp", "sa_score", "mw"],
            "max_batch_size": 256,
            "supports_streaming": True,
            "requires_gpu": True,
        }


def _to_molecule(item) -> Molecule:
    if isinstance(item, Molecule):
        return item
    if not isinstance(item, dict) or not item.get("smiles"):
        raise ValueError("LaMGen runner output must contain a smiles field")
    metadata = {
        key: str(value)
        for key, value in item.get("metadata", {}).items()
        if value is not None
    }
    return Molecule(
        smiles=str(item["smiles"]),
        sdf_bytes=item.get("sdf_bytes"),
        metadata=metadata,
    )
