"""Incremental CLM: Online continual learning for SAR series refinement."""
from __future__ import annotations

import inspect

from mf_core.plugins.generator import GeneratorPlugin
from mf_core.types.molecule import Molecule
from mf_core.types.humu import IntentCone


class IncrementalCLMGenerator(GeneratorPlugin):
    def __init__(
        self,
        checkpoint_path: str = "",
        device: str = "cpu",
        runner=None,
        model=None,
        tokenizer=None,
        decoder=None,
        online_learner=None,
        ewc_regularizer=None,
        packnet=None,
    ):
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.runner = runner
        self._model = model
        self.tokenizer = tokenizer
        self.decoder = decoder
        self.online_learner = online_learner
        self.ewc_regularizer = ewc_regularizer
        self.packnet = packnet
        self._task_count = 0

    async def generate(self, batch_size: int, intent_cone: IntentCone | None = None, **kwargs) -> list[Molecule]:
        """Generate molecules via continual learning with EWC/PackNet regularization."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.packnet is not None:
            self.packnet.apply_mask()
        if self.runner is not None:
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
        if self._model is None or self.decoder is None:
            raise RuntimeError("IncrementalCLM model or runner is required")

        online_batch = kwargs.pop("online_batch", None)
        if online_batch is not None and self.online_learner is not None:
            update_result = self.online_learner.update(online_batch)
            if inspect.isawaitable(update_result):
                await update_result

        model_output = self._model(
            batch_size=batch_size,
            intent_cone=intent_cone,
            tokenizer=self.tokenizer,
            checkpoint_path=self.checkpoint_path,
            device=self.device,
            **kwargs,
        )
        if inspect.isawaitable(model_output):
            model_output = await model_output
        decoded = self.decoder(model_output, batch_size=batch_size)
        if inspect.isawaitable(decoded):
            decoded = await decoded

        metadata = {}
        if self.ewc_regularizer is not None:
            ewc_loss = self.ewc_regularizer.ewc_loss()
            if hasattr(ewc_loss, "detach"):
                ewc_loss = ewc_loss.detach().cpu().item()
            metadata["ewc_loss"] = str(ewc_loss)
        results = [_to_molecule(item, metadata=metadata) for item in decoded[:batch_size]]
        if len(results) < batch_size:
            raise RuntimeError("IncrementalCLM decoder returned fewer molecules than requested")
        self._task_count += 1
        return results

    async def info(self) -> dict:
        return {
            "name": "incremental_clm",
            "version": "0.1.0",
            "description": "Online continual learning for SAR series refinement",
            "max_batch_size": 64,
            "supports_streaming": False,
            "requires_gpu": True,
        }


def _to_molecule(item, metadata: dict[str, str] | None = None) -> Molecule:
    base_metadata = metadata or {}
    if isinstance(item, Molecule):
        merged = {**item.metadata, **base_metadata}
        return item.model_copy(update={"metadata": merged})
    if isinstance(item, str):
        return Molecule(smiles=item, metadata=base_metadata)
    if not isinstance(item, dict) or not item.get("smiles"):
        raise ValueError("IncrementalCLM output must contain a smiles field")
    return Molecule(
        smiles=str(item["smiles"]),
        sdf_bytes=item.get("sdf_bytes"),
        metadata={
            **{
                key: str(value)
                for key, value in item.get("metadata", {}).items()
                if value is not None
            },
            **base_metadata,
        },
    )
