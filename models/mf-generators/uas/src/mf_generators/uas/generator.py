"""UAS (Unfamiliarity-Aware Sampling) — OOD detection via reconstruction error."""
from __future__ import annotations

import inspect
from typing import Any, AsyncIterator

import torch
from mf_core.types.molecule import MoleculeModel
from mf_generators.uas.autoencoder.molecule_ae import MoleculeAutoencoder
from mf_generators.uas.sampler.ood_aware_sampling import OODAwareSampler
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
    ):
        self.dim = dim
        self.ae = MoleculeAutoencoder(input_dim=dim, latent_dim=max(1, dim // 2))
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
            or self.reference_embeddings is None
            or self.decoder is None
        ):
            raise RuntimeError(
                "UAS_RUNNER is required or UAS candidate_source, "
                "reference_embeddings, and decoder are required"
            )

        reference = _as_tensor(self.reference_embeddings)

        def estimator(candidates: torch.Tensor) -> torch.Tensor:
            return compute_unfamiliarity(candidates, reference)

        sampler = OODAwareSampler(
            estimator,
            rejection_threshold=self.unfamiliarity_threshold,
            candidate_source=self.candidate_source,
        )
        accepted = sampler.sample(n_samples)
        _ = self.ae.reconstruction_loss(accepted)
        decoded = self.decoder(accepted)
        if inspect.isawaitable(decoded):
            decoded = await decoded
        for item in decoded[:n_samples]:
            molecule = _to_molecule_model(item)
            _validate_smiles(molecule.smiles)
            yield molecule


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
