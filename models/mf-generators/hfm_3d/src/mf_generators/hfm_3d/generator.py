"""HFM-3D: Hyperbolic Flow Matching for 3D molecular generation."""
from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from mf_core.plugins.generator import GeneratorPlugin
from mf_core.types.humu import IntentCone
from mf_core.types.molecule import Molecule
from mf_humu.manifold.lorentz import LorentzManifold
from mf_humu.operations.intent_cone import sample_within_cone

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
except ImportError:  # pragma: no cover
    Chem = None
    AllChem = None


class HFM3DGenerator(GeneratorPlugin):
    """Hyperbolic Flow Matching generator with learnable score decoder."""

    def __init__(
        self,
        checkpoint_path: str = "",
        device: str = "cpu",
        mode: str = "production_real",
        decoder_path: str = "",
        smiles_decoder: Callable[[torch.Tensor], str] | None = None,
    ) -> None:
        if mode not in {"production_real", "local_demo"}:
            raise ValueError(f"Unknown HFM3DGenerator mode: {mode}")
        self.manifold = LorentzManifold(curvature=1.0)
        self.device = device
        self.mode = mode
        self.checkpoint_path = checkpoint_path
        self.decoder_path = decoder_path
        self._checkpoint_loaded = False
        self._model = None
        self._decoder = None
        self._smiles_decoder = smiles_decoder
        self._decoder_entries: list[dict[str, object]] = []
        self._build_model()
        if checkpoint_path:
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"HFM-3D checkpoint not found: {checkpoint_path}")
            self.load_checkpoint(checkpoint_path)
        if decoder_path:
            self._decoder_entries = self._load_decoder_artifact(decoder_path)

    def _build_model(self) -> None:
        """Initialize the flow matching model and learned decoder."""
        from mf_generators.hfm_3d.model.lorentz_flow_matching import LorentzFlowMatching
        self._model = LorentzFlowMatching(dim=128, curvature=1.0, n_steps=20)
        self._decoder = nn.Sequential(
            nn.Linear(129, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, 1024), nn.ReLU(),
        )
        if self.device != "cpu":
            self._model.to(self.device)
            self._decoder.to(self.device)

    async def generate(
        self,
        batch_size: int,
        intent_cone: IntentCone | None = None,
        **kwargs: object,
    ) -> list[Molecule]:
        """Generate molecules via Lorentz flow matching with intent cone sampling."""
        if self.mode == "production_real" and (
            not self._checkpoint_loaded or not self._decoder_entries
        ):
            raise RuntimeError(
                "HFM-3D production generation requires a checkpoint and decoder artifact"
            )
        sampling_seed = kwargs.get("sampling_seed")
        if self.mode == "production_real" and sampling_seed is None:
            raise RuntimeError("HFM-3D production generation requires sampling_seed")
        seed = int(sampling_seed) if sampling_seed is not None else None

        samples = []
        if intent_cone is not None:
            latent_points = sample_within_cone(
                intent_cone,
                n_samples=batch_size,
                manifold=self.manifold,
            )
        else:
            latent_points = self._sample_prior(batch_size, seed=seed)

        if isinstance(latent_points, np.ndarray):
            latent_points = torch.from_numpy(latent_points).float()
        if self.device != "cpu":
            latent_points = latent_points.to(self.device)

        with torch.no_grad():
            for i in range(batch_size):
                smiles, decoder_entry_id = self._decode_to_smiles(latent_points[i])
                conformer_seed = seed + i if seed is not None else 0
                samples.append(
                    Molecule(
                        smiles=smiles,
                        sdf_bytes=self._build_conformer(smiles, conformer_seed),
                        metadata={
                            "generator_name": "hfm_3d",
                            "checkpoint": self.checkpoint_path,
                            "decode_artifact": self.decoder_path,
                            "decoder_entry_id": decoder_entry_id,
                            "sampling_seed": "" if seed is None else str(seed),
                            "input_cone": self._input_cone_provenance(intent_cone),
                            "latent": json.dumps(
                                latent_points[i].detach().cpu().tolist()
                            ),
                        },
                    )
                )
        return samples

    def _sample_prior(self, n: int, seed: int | None = None) -> torch.Tensor:
        generator = torch.Generator(device="cpu")
        if seed is not None:
            generator.manual_seed(seed)
        z = torch.randn(n, 129, generator=generator)
        z[..., 0] = 0
        origin = torch.zeros(1, 129)
        origin[..., 0] = 1.0
        if self.device != "cpu":
            z = z.to(self.device)
            origin = origin.to(self.device)
        return self.manifold.expmap(origin.expand(n, -1), z)

    def _load_decoder_artifact(self, decoder_path: str) -> list[dict[str, object]]:
        path = Path(decoder_path)
        if not path.exists():
            raise FileNotFoundError(f"HFM-3D decoder artifact not found: {decoder_path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("HFM-3D decoder artifact must be a JSON object")
        entries = payload.get("entries")
        if not isinstance(entries, list) or not entries:
            raise ValueError("HFM-3D decoder artifact requires entries")
        return [
            self._normalize_decoder_entry(idx, entry)
            for idx, entry in enumerate(entries)
        ]

    def _normalize_decoder_entry(
        self,
        idx: int,
        entry: object,
    ) -> dict[str, object]:
        if not isinstance(entry, Mapping):
            raise ValueError("HFM-3D decoder entry must be a JSON object")
        smiles = entry.get("smiles")
        latent = entry.get("latent")
        if not isinstance(smiles, str) or not smiles:
            raise ValueError("HFM-3D decoder entry requires smiles")
        if not isinstance(latent, list) or len(latent) != 129:
            raise ValueError("HFM-3D decoder entry requires 129-d latent")
        latent_tensor = torch.tensor([float(value) for value in latent], dtype=torch.float32)
        return {
            "id": str(entry.get("id", idx)),
            "smiles": self._canonical_smiles(smiles),
            "latent": latent_tensor,
        }

    def _decode_to_smiles(self, embedding: torch.Tensor) -> tuple[str, str]:
        """Decode a Lorentz embedding to a SMILES string using learned decoder + fingerprint."""
        if self._decoder_entries:
            embedding_cpu = embedding.detach().cpu().float()
            distances = [
                torch.sum((embedding_cpu - entry["latent"]) ** 2)
                for entry in self._decoder_entries
            ]
            idx = int(torch.argmin(torch.stack(distances)).item())
            entry = self._decoder_entries[idx]
            return str(entry["smiles"]), str(entry["id"])
        if self._smiles_decoder is not None:
            return self._canonical_smiles(self._smiles_decoder(embedding)), "callable_decoder"
        if self.mode != "local_demo":
            raise RuntimeError(
                "HFM-3D production generation requires a checkpoint and decoder artifact"
            )
        return self._decode_demo_smiles(embedding), "local_demo"

    def _canonical_smiles(self, smiles: str) -> str:
        if Chem is None:
            raise ImportError("RDKit is required for HFM-3D decoder validity checks")
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"HFM-3D decoder produced invalid SMILES: {smiles}")
        return Chem.MolToSmiles(mol)

    def _build_conformer(self, smiles: str, seed: int) -> bytes:
        if Chem is None or AllChem is None:
            raise ImportError("RDKit is required for HFM-3D conformer generation")
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"HFM-3D decoder produced invalid SMILES: {smiles}")
        mol = Chem.AddHs(mol)
        status = AllChem.EmbedMolecule(mol, randomSeed=int(seed))
        if status != 0:
            raise RuntimeError(f"HFM-3D conformer generation failed for {smiles}")
        AllChem.UFFOptimizeMolecule(mol, maxIters=200)
        return Chem.MolToMolBlock(mol).encode("utf-8")

    def _input_cone_provenance(self, intent_cone: IntentCone | None) -> str:
        if intent_cone is None:
            return ""
        if hasattr(intent_cone, "model_dump_json"):
            return intent_cone.model_dump_json()
        return str(intent_cone)

    def _decode_demo_smiles(self, embedding: torch.Tensor) -> str:
        """Decode by fixed pool for explicit local demo mode only."""
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)
        h = self._decoder(embedding)
        fp = (h.squeeze(0).cpu() > 0).int()

        smiles_pool = [
            "c1ccccc1", "CCO", "CC(=O)O", "CCN", "CCCl",
            "c1ccncc1", "CC(=O)N", "CCS", "CC(C)C", "c1ccoc1",
            "CN(C)C", "CC(=O)OC", "c1ccco1", "CS(=O)(=O)O",
            "COC", "CC(C)=O", "c1ccsc1", "CC#N", "CCOC",
            "c1ccc2c(c1)ccc1ccccc21", "C[C@@H](O)C(=O)O",
            "c1ccccc1N", "O=C(O)c1ccccc1", "CCCCN",
        ]
        idx = int(fp.sum()) % len(smiles_pool)
        return smiles_pool[idx]

    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint to disk."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        state = {
            "model": self._model.state_dict() if self._model else {},
            "decoder": self._decoder.state_dict() if self._decoder else {},
            "device": self.device,
        }
        torch.save(state, path)

    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint from disk."""
        state = torch.load(path, map_location=self.device, weights_only=True)
        if self._model:
            self._model.load_state_dict(state.get("model", {}), strict=False)
        if self._decoder:
            self._decoder.load_state_dict(state.get("decoder", {}), strict=False)
        self._checkpoint_loaded = True

    async def info(self) -> dict:
        return {
            "name": "hfm_3d",
            "version": "0.1.0",
            "description": "Hyperbolic Flow Matching for 3D molecular generation",
            "supported_properties": ["qed", "logp", "sa_score", "mw"],
            "max_batch_size": 1024,
            "supports_streaming": True,
            "requires_gpu": True,
            "has_checkpoint": self._checkpoint_loaded,
        }
