"""UAS (Unfamiliarity-Aware Sampling) — OOD detection via reconstruction error."""
from __future__ import annotations

import inspect
from collections.abc import AsyncIterator
from typing import Any

import torch
from mf_core.types.molecule import MoleculeModel
from mf_generators.uas.autoencoder.molecule_ae import MoleculeAutoencoder
from mf_generators.uas.sampler.ood_aware_sampling import _safety_probability
from mf_generators.uas.unfamiliarity_estimator import compute_unfamiliarity

try:
    from rdkit import Chem
except ImportError:  # pragma: no cover
    Chem = None


_Autoencoder = MoleculeAutoencoder


class UASGenerator:
    name = "uas"
    version = "0.1.0"
    supported_modes = ["hit_finding", "lead_opt"]

    def __init__(
        self,
        dim: int = 128,
        runner=None,
        candidate_source=None,
        reference_embeddings=None,
        decoder=None,
        unfamiliarity_threshold: float = 0.5,
        autoencoder=None,
    ):
        self.dim = dim
        self.ae = (
            autoencoder
            if autoencoder is not None
            else MoleculeAutoencoder(input_dim=dim, latent_dim=max(1, dim // 2))
        )
        self._uses_trained_autoencoder = autoencoder is not None
        self.runner = runner
        self.candidate_source = candidate_source
        self.reference_embeddings = reference_embeddings
        self.decoder = decoder
        self.unfamiliarity_threshold = unfamiliarity_threshold

    async def generate(
        self,
        hciv: Any,
        cone: Any,
        cig: Any,
        n_samples: int = 10,
        seed: int | None = None,
    ) -> AsyncIterator[MoleculeModel]:
        if self.runner is not None:
            result = self.runner.generate(
                hciv=hciv,
                cone=cone,
                cig=cig,
                n_samples=n_samples,
                seed=seed,
                dim=self.dim,
            )
            if inspect.isawaitable(result):
                result = await result
            if hasattr(result, "__aiter__"):
                async for item in result:
                    yield _to_molecule_model(item)
                return
            for item in result:
                yield _to_molecule_model(item)
            return

        if (
            self.candidate_source is None
            or self.decoder is None
            or (not self._uses_trained_autoencoder and self.reference_embeddings is None)
        ):
            raise RuntimeError(
                "UAS_RUNNER is required or UAS candidate_source, "
                "reference_embeddings, and decoder are required"
            )

        reference = (
            _as_tensor(self.reference_embeddings)
            if self.reference_embeddings is not None
            else torch.empty((0, self.dim), dtype=torch.float32)
        )

        def estimator(candidates: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                return compute_unfamiliarity(
                    candidates,
                    reference,
                    model=self.ae if self._uses_trained_autoencoder else None,
                )

        accepted, safety_probabilities = await _sample_candidates(
            self.candidate_source,
            estimator,
            n_samples,
            rejection_threshold=self.unfamiliarity_threshold,
        )
        decoded = self.decoder(accepted)
        if inspect.isawaitable(decoded):
            decoded = await decoded
        for index, item in enumerate(decoded[:n_samples]):
            molecule = _to_molecule_model(item)
            molecule.humu_embedding = [
                float(value) for value in accepted[index].detach().cpu().tolist()
            ]
            if index < len(safety_probabilities):
                molecule.properties["uas_safety_probability"] = float(
                    safety_probabilities[index]
                )
            _validate_smiles(molecule.smiles)
            yield molecule


async def _sample_candidates(
    candidate_source,
    estimator,
    n_samples: int,
    *,
    rejection_threshold: float,
    max_attempts: int = 10,
) -> tuple[torch.Tensor, list[float]]:
    accepted = []
    accepted_probabilities = []
    attempts = 0
    while len(accepted) < n_samples and attempts < max_attempts:
        candidates = candidate_source(n_samples)
        if inspect.isawaitable(candidates):
            candidates = await candidates
        candidates = _as_tensor(candidates)
        unfamiliarity = estimator(candidates)
        safety_probability = _safety_probability(
            unfamiliarity,
            rejection_threshold,
            1.0,
        )
        mask = safety_probability > 0.5
        accepted.extend(candidates[mask])
        accepted_probabilities.extend(safety_probability[mask])
        attempts += 1
    if len(accepted) < n_samples:
        raise RuntimeError("UAS candidate source did not produce enough in-domain samples")
    return (
        torch.stack(accepted[:n_samples]),
        [
            float(value.detach().cpu().item())
            for value in accepted_probabilities[:n_samples]
        ],
    )


def _to_molecule_model(item) -> MoleculeModel:
    if isinstance(item, MoleculeModel):
        return item
    if isinstance(item, str):
        return MoleculeModel(smiles=item, canonical_smiles=item, generator_name="uas")
    if not isinstance(item, dict) or not item.get("smiles"):
        raise ValueError("UAS runner output must contain a smiles field")
    return MoleculeModel(
        id=str(item.get("id", "")),
        smiles=str(item["smiles"]),
        canonical_smiles=str(item.get("canonical_smiles", item["smiles"])),
        generator_name="uas",
        humu_embedding=item.get("humu_embedding"),
        properties=item.get("properties", {}),
        embedding=item.get("embedding"),
    )


def _as_tensor(value) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.float()
    return torch.tensor(value, dtype=torch.float32)


def _validate_smiles(smiles: str) -> None:
    if Chem is None:
        return
    if Chem.MolFromSmiles(smiles) is None:
        raise ValueError(f"UAS decoder produced invalid SMILES: {smiles}")
