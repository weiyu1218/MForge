"""Model loading and batch inference engine.

Handles lazy-loading chemprop MPNN checkpoints and running
batched predictions with automatic device selection.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)


def _resolve_device(preference: str = "auto") -> torch.device:
    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(preference)


class ModelManager:
    """Lazy-loading registry of chemprop MPNN models."""

    def __init__(self, device: str = "auto"):
        self.device = _resolve_device(device)
        self._models: dict[str, Any] = {}  # endpoint_name -> loaded model
        self._paths: dict[str, Path] = {}
        logger.info("ModelManager device=%s", self.device)

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, name: str, path: Path) -> None:
        """Register a model checkpoint path (does NOT load yet)."""
        self._paths[name] = path

    def register_many(self, mapping: dict[str, Path]) -> None:
        for name, path in mapping.items():
            self.register(name, path)

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------

    def _load(self, name: str) -> Any:
        if name in self._models:
            return self._models[name]

        path = self._paths.get(name)
        if path is None:
            raise KeyError(f"Unknown endpoint: {name}")
        if not path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found for '{name}': {path}\n"
                "Place your trained chemprop checkpoint in this directory."
            )

        from chemprop.models import MPNN  # lazy import — slow

        logger.info("Loading model '%s' from %s …", name, path)
        model = MPNN.load_from_checkpoint(str(path / "model.ckpt"), map_location=self.device)
        model.to(self.device)
        model.eval()
        self._models[name] = model
        return model

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @property
    def available_endpoints(self) -> list[str]:
        return list(self._paths.keys())

    def predict_batch(
        self,
        smiles: list[str],
        endpoints: list[str],
        batch_size: int = 64,
    ) -> list[dict[str, float | None]]:
        """Run batched prediction across requested endpoints.

        Returns one dict per SMILES: {endpoint_name: prediction_value | None}.
        Failed molecules (invalid SMILES, model errors) get None.
        """
        from chemprop.data import MoleculeDatapoint, MoleculeDataset, collate_batch

        # Build dataset once
        datapoints = [MoleculeDatapoint.from_smi(s) for s in smiles]
        dataset = MoleculeDataset(datapoints)

        results: list[dict[str, float | None]] = [{} for _ in smiles]

        for ep_name in endpoints:
            model = self._load(ep_name)
            preds = self._run_model(model, dataset, batch_size)
            for i, val in enumerate(preds):
                results[i][ep_name] = val

        return results

    def _run_model(self, model: Any, dataset, batch_size: int) -> list[float | None]:
        """Run a single model on a full dataset with batching."""
        from chemprop.data import MoleculeDataset, collate_batch

        all_preds: list[float | None] = []
        n = len(dataset)

        with torch.no_grad():
            for start in range(0, n, batch_size):
                batch_slice = dataset[start : start + batch_size]
                batch = collate_batch(batch_slice)
                # Move to device
                bmg = batch.batch_graph()
                V = bmg.V.to(self.device)
                E = bmg.E.to(self.device)
                bmg_on_device = bmg.__class__(V, E, bmg.batch)

                try:
                    out = model(bmg_on_device, batch.features_batch if hasattr(batch, 'features_batch') else None)
                    preds = out.cpu().numpy().flatten().tolist()
                    # Convert NaN → None
                    preds = [float(p) if np.isfinite(p) else None for p in preds]
                except Exception:
                    logger.exception("Batch inference failed for a slice")
                    preds = [None] * len(batch_slice)

                all_preds.extend(preds)

        return all_preds
