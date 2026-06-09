"""HUMU pocket encoder backed by atom coordinates and residue features."""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from mf_humu.encoders.lorentz_attention import LorentzAttention
from mf_humu.manifold.learnable_lorentz import LearnableLorentzManifold
from mf_humu.manifold.lorentz import LorentzManifold

_POCKET_FEATURE_DIM = 12
_HYDROPHOBIC = {"ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP", "PRO"}
_POSITIVE = {"LYS", "ARG", "HIS"}
_NEGATIVE = {"ASP", "GLU"}


class HUMUPocketEncoder(nn.Module):
    """Encode protein binding pocket point clouds into Lorentz embeddings."""

    def __init__(
        self,
        dim: int = 128,
        curvature: float = 1.0,
        learnable_curvature: bool = False,
        use_esm2: bool = False,
        esm2_checkpoint: str | None = None,
        esm2_layer: int = 33,
        esm2_dim: int = 1280,
        esm2_batch_tokens: int = 8192,
        esm2_max_sequence_length: int | None = None,
    ):
        super().__init__()
        manifold_cls = LearnableLorentzManifold if learnable_curvature else LorentzManifold
        self.manifold = manifold_cls(curvature=curvature)
        self.dim = dim
        self.use_esm2 = use_esm2
        self.esm2_checkpoint = esm2_checkpoint
        self.esm2_layer = esm2_layer
        self.esm2_dim = esm2_dim
        self.esm2_batch_tokens = esm2_batch_tokens
        self.esm2_max_sequence_length = esm2_max_sequence_length
        self._point_projection = nn.Linear(_POCKET_FEATURE_DIM, dim + 1)
        self._esm2_projection = nn.Linear(esm2_dim, dim + 1) if use_esm2 else None
        self._attention = LorentzAttention(
            dim=dim,
            heads=8,
            curvature=curvature,
            learnable_curvature=learnable_curvature,
        )
        self._esm2_model = None
        self._esm2_batch_converter = None
        self._esm2_sequence_cache: dict[str, torch.Tensor] = {}

    def forward(self, pocket_data: dict | list[dict]) -> torch.Tensor:
        if isinstance(pocket_data, list):
            return self.encode_batch(pocket_data)
        return self.encode(pocket_data)

    def encode(self, pocket_data: dict) -> torch.Tensor:
        esm2_embedding = self._esm2_embedding(pocket_data) if self.use_esm2 else None
        return self._encode_with_esm2_embedding(pocket_data, esm2_embedding)

    def _encode_with_esm2_embedding(
        self,
        pocket_data: dict,
        esm2_embedding: torch.Tensor | None,
    ) -> torch.Tensor:
        coords, elements, residues = self._validate_pocket(pocket_data)
        coords = coords.to(self._param_device())
        features = self._point_features(coords, elements, residues)
        x = self._point_projection(features)
        if esm2_embedding is not None:
            x = x + self._esm2_projection(esm2_embedding).expand_as(x)
        x = x.unsqueeze(0)
        x = self.manifold._project(x)
        x = self._attention(x)
        embedding = x.mean(dim=1)
        return self.manifold._project(embedding)

    def encode_batch(self, pocket_data_list: list[dict]) -> torch.Tensor:
        if not pocket_data_list:
            raise ValueError("pocket encoder requires at least one pocket record")
        if not self.use_esm2:
            return torch.cat([self.encode(pocket) for pocket in pocket_data_list], dim=0)
        esm2_embeddings = self._esm2_embeddings(pocket_data_list)
        return torch.cat(
            [
                self._encode_with_esm2_embedding(pocket, esm2_embeddings[index])
                for index, pocket in enumerate(pocket_data_list)
            ],
            dim=0,
        )

    def _param_device(self) -> torch.device:
        return self._point_projection.weight.device

    def _validate_pocket(self, pocket_data: dict) -> tuple[torch.Tensor, list[str], list[str]]:
        if "coords" not in pocket_data:
            raise ValueError("pocket encoder requires coords for every pocket atom")
        coords = pocket_data["coords"]
        if not isinstance(coords, torch.Tensor):
            coords = torch.tensor(coords, dtype=torch.float32)
        coords = coords.float()
        if coords.ndim != 2 or coords.shape[1] != 3 or coords.shape[0] == 0:
            raise ValueError("pocket encoder requires coords with shape (n_atoms, 3)")
        if not torch.isfinite(coords).all():
            raise ValueError("pocket encoder requires finite coords")

        elements = pocket_data.get("elements")
        residues = pocket_data.get("residue_types")
        if not isinstance(elements, list) or len(elements) != coords.shape[0]:
            raise ValueError("pocket encoder requires one element per coordinate")
        if not isinstance(residues, list) or len(residues) != coords.shape[0]:
            raise ValueError("pocket encoder requires one residue type per coordinate")
        return coords, [str(e).upper() for e in elements], [str(r).upper() for r in residues]

    def _esm2_embedding(self, pocket_data: dict) -> torch.Tensor:
        if "esm2_embedding" in pocket_data:
            return self._validate_esm2_embedding(pocket_data["esm2_embedding"])

        sequence = pocket_data.get("protein_sequence") or pocket_data.get("sequence")
        if not isinstance(sequence, str) or not sequence:
            raise ValueError("ESM-2 input requires protein_sequence, sequence, or esm2_embedding")
        self._validate_esm2_sequence_length(sequence)
        cached = self._esm2_sequence_cache.get(sequence)
        if cached is None:
            cached = self._compute_esm2_embedding(sequence).detach().cpu()
            self._esm2_sequence_cache[sequence] = cached
        return cached.to(self._param_device())

    def _esm2_embeddings(self, pocket_data_list: list[dict]) -> torch.Tensor:
        outputs: list[torch.Tensor | None] = [None] * len(pocket_data_list)
        missing_sequences: list[str] = []
        missing_positions_by_sequence: dict[str, list[int]] = {}
        for index, pocket_data in enumerate(pocket_data_list):
            if "esm2_embedding" in pocket_data:
                outputs[index] = self._validate_esm2_embedding(pocket_data["esm2_embedding"])
                continue

            sequence = pocket_data.get("protein_sequence") or pocket_data.get("sequence")
            if not isinstance(sequence, str) or not sequence:
                raise ValueError(
                    "ESM-2 input requires protein_sequence, sequence, or esm2_embedding"
                )
            self._validate_esm2_sequence_length(sequence)
            cached = self._esm2_sequence_cache.get(sequence)
            if cached is not None:
                outputs[index] = cached.to(self._param_device())
                continue
            if sequence not in missing_positions_by_sequence:
                missing_sequences.append(sequence)
                missing_positions_by_sequence[sequence] = []
            missing_positions_by_sequence[sequence].append(index)

        if missing_sequences:
            computed = self._compute_esm2_batch_embeddings(missing_sequences)
            for sequence, embedding in zip(missing_sequences, computed, strict=True):
                cached = embedding.detach().cpu()
                self._esm2_sequence_cache[sequence] = cached
                for index in missing_positions_by_sequence[sequence]:
                    outputs[index] = cached.to(self._param_device())

        return torch.stack([embedding for embedding in outputs if embedding is not None])

    def _validate_esm2_sequence_length(self, sequence: str) -> None:
        max_length = self.esm2_max_sequence_length
        if max_length is None or int(max_length) <= 0:
            return
        if len(sequence) > int(max_length):
            raise ValueError(
                "ESM-2 sequence length "
                f"{len(sequence)} exceeds configured maximum {int(max_length)}"
            )

    def _validate_esm2_embedding(self, embedding) -> torch.Tensor:
        if not isinstance(embedding, torch.Tensor):
            embedding = torch.tensor(embedding, dtype=torch.float32)
        embedding = embedding.float().to(self._param_device()).reshape(-1)
        if embedding.numel() != self.esm2_dim:
            raise ValueError(
                f"ESM-2 embedding must have {self.esm2_dim} values, "
                f"got {embedding.numel()}"
            )
        if not torch.isfinite(embedding).all():
            raise ValueError("ESM-2 embedding must contain finite values")
        return embedding

    def _compute_esm2_embedding(self, sequence: str) -> torch.Tensor:
        return self._compute_esm2_batch_embeddings([sequence])[0]

    def _compute_esm2_batch_embeddings(self, sequences: list[str]) -> torch.Tensor:
        model, batch_converter = self._load_esm2()
        outputs: list[torch.Tensor | None] = [None] * len(sequences)
        for batch_indices in self._esm2_sequence_batches(sequences):
            raw_batch = [(f"pocket_{index}", sequences[index]) for index in batch_indices]
            _, _, tokens = batch_converter(raw_batch)
            tokens = tokens.to(self._param_device())
            with torch.no_grad():
                result = model(tokens, repr_layers=[self.esm2_layer], return_contacts=False)
            representations = result["representations"][self.esm2_layer]
            for row, sequence_index in enumerate(batch_indices):
                sequence = sequences[sequence_index]
                pooled = representations[row, 1 : len(sequence) + 1].mean(dim=0)
                pooled = pooled.float().reshape(-1)
                if pooled.numel() != self.esm2_dim:
                    raise ValueError(
                        f"ESM-2 model returned {pooled.numel()} values, "
                        f"expected {self.esm2_dim}"
                    )
                outputs[sequence_index] = pooled.to(self._param_device())
        return torch.stack([embedding for embedding in outputs if embedding is not None])

    def _esm2_sequence_batches(self, sequences: list[str]) -> list[list[int]]:
        sorted_indices = sorted(range(len(sequences)), key=lambda index: len(sequences[index]))
        batches: list[list[int]] = []
        current: list[int] = []
        current_max_len = 0
        max_tokens = max(1, int(self.esm2_batch_tokens))
        for index in sorted_indices:
            seq_len = len(sequences[index]) + 2
            candidate_max_len = max(current_max_len, seq_len)
            candidate_tokens = candidate_max_len * (len(current) + 1)
            if current and candidate_tokens > max_tokens:
                batches.append(current)
                current = []
                current_max_len = 0
            current.append(index)
            current_max_len = max(current_max_len, seq_len)
        if current:
            batches.append(current)
        return batches

    def _load_esm2(self):
        if self._esm2_model is not None and self._esm2_batch_converter is not None:
            return self._esm2_model, self._esm2_batch_converter
        if not self.esm2_checkpoint:
            raise RuntimeError(
                "esm2_checkpoint is required when ESM-2 sequence encoding is enabled"
            )
        checkpoint = Path(self.esm2_checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"ESM-2 checkpoint not found: {checkpoint}")
        try:
            from esm import pretrained
        except ImportError as exc:
            raise RuntimeError("esm package is required for ESM-2 sequence encoding") from exc

        regression_path = Path(str(checkpoint.with_suffix("")) + "-contact-regression.pt")
        if checkpoint.stem.startswith("esm2") and not regression_path.is_file():
            model, alphabet = self._load_esm2_without_contact_regression(
                checkpoint,
                pretrained,
            )
        else:
            model, alphabet = pretrained.load_model_and_alphabet(str(checkpoint))
        model.eval()
        model.to(self._param_device())
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        self._esm2_model = model
        self._esm2_batch_converter = alphabet.get_batch_converter()
        return self._esm2_model, self._esm2_batch_converter

    def _load_esm2_without_contact_regression(self, checkpoint: Path, pretrained):
        model_data = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
        if "cfg" in model_data and "model" in model_data:
            return pretrained.load_model_and_alphabet_core(
                checkpoint.stem,
                model_data,
                regression_data=None,
            )
        if "model_state_dict" not in model_data:
            raise RuntimeError(f"Unsupported ESM-2 checkpoint format: {checkpoint}")

        import esm
        from esm.model.esm2 import ESM2

        alphabet = esm.data.Alphabet.from_architecture("ESM-1b")
        model = ESM2(
            num_layers=int(model_data["num_layers"]),
            embed_dim=int(model_data["embed_dim"]),
            attention_heads=int(model_data["attention_heads"]),
            alphabet=alphabet,
            token_dropout=True,
        )
        missing, unexpected = model.load_state_dict(
            model_data["model_state_dict"],
            strict=False,
        )
        allowed_missing = {
            "contact_head.regression.weight",
            "contact_head.regression.bias",
        }
        unexpected = set(unexpected)
        missing = set(missing) - allowed_missing
        if missing or unexpected:
            raise RuntimeError(
                "Unsupported ESM-2 checkpoint state dict: "
                f"missing={sorted(missing)} unexpected={sorted(unexpected)}"
            )
        return model, alphabet

    def _point_features(
        self,
        coords: torch.Tensor,
        elements: list[str],
        residues: list[str],
    ) -> torch.Tensor:
        centered = coords - coords.mean(dim=0, keepdim=True)
        pairwise = torch.cdist(centered, centered)
        non_self = ~torch.eye(coords.shape[0], dtype=torch.bool, device=coords.device)
        masked = pairwise.masked_fill(~non_self, 0.0)
        neighbor_count = max(int(coords.shape[0]) - 1, 1)
        radial_distance = torch.linalg.vector_norm(centered, dim=1)
        mean_neighbor_distance = masked.sum(dim=-1) / float(neighbor_count)
        max_neighbor_distance = pairwise.max(dim=-1).values
        min_neighbor_distance = pairwise.masked_fill(
            ~non_self,
            float("inf"),
        ).min(dim=-1).values
        if coords.shape[0] == 1:
            min_neighbor_distance = torch.zeros_like(min_neighbor_distance)
        distance_scale = 20.0

        rows = []
        for idx, element in enumerate(elements):
            residue = residues[idx]
            rows.append([
                radial_distance[idx] / distance_scale,
                mean_neighbor_distance[idx] / distance_scale,
                max_neighbor_distance[idx] / distance_scale,
                min_neighbor_distance[idx] / distance_scale,
                float(element == "C"),
                float(element == "N"),
                float(element == "O"),
                float(element == "S"),
                float(element in {"F", "CL", "BR", "I"}),
                float(residue in _HYDROPHOBIC),
                float(residue in _POSITIVE),
                float(residue in _NEGATIVE),
            ])
        return torch.tensor(rows, dtype=torch.float32, device=self._param_device())
