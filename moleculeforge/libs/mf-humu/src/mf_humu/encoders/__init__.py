"""Shared HUMU encoder runtime contracts."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

import torch
import torch.nn as nn

from mf_humu.manifold.lorentz import LorentzManifold

HUMU_CHECKPOINT_SCHEMA = "humu-checkpoint.v1"
HUMU_VALIDATION_ARTIFACT_SCHEMA = "humu-validation-artifact.v1"
HUMU_VALIDATION_ARTIFACT_PURPOSE = "synthetic_pipeline_validation_only"
HUMU_VALIDATION_ARTIFACT_SEED = 7


class HUMUEncoderWrapper(nn.Module):
    """Apply the trainable shared projection used by HUMU pretraining."""

    def __init__(
        self,
        inner: nn.Module,
        dim: int,
        device: torch.device | str,
        curvature: float = 1.0,
        model_config: Mapping[str, Any] | None = None,
    ):
        super().__init__()
        self.inner = inner
        self.proj = nn.Linear(dim + 1, dim + 1)
        self._manifold = LorentzManifold(curvature=curvature)
        self.humu_model_config = deepcopy(dict(model_config)) if model_config is not None else None
        self.to(device)

    @property
    def manifold(self) -> LorentzManifold:
        return self._manifold

    def forward(self, data):
        if isinstance(data, list):
            embedding = self.inner.encode_batch(data)
        else:
            embedding = self.inner.encode(data)
        device = self.proj.weight.device
        if embedding.device != device:
            embedding = embedding.to(device)
        return self._manifold._project(self.proj(embedding))

    def encode_batch(self, data: list) -> torch.Tensor:
        return self.forward(data)

    def encode(self, data) -> torch.Tensor:
        return self.forward(data)


def build_humu_model_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize training configuration into the serving architecture contract."""
    dim = int(config.get("embed_dim", 129)) - 1
    if dim <= 0:
        raise ValueError("embed_dim must be greater than one")
    curvature = float(config.get("curvature", 1.0))
    encoder_config = config.get("encoders") or {}
    if not isinstance(encoder_config, Mapping):
        raise ValueError("encoders must be a mapping")
    mol = encoder_config.get("mol") or {}
    pocket = encoder_config.get("pocket") or {}
    route = encoder_config.get("route") or {}
    for name, value in (("mol", mol), ("pocket", pocket), ("route", route)):
        if not isinstance(value, Mapping):
            raise ValueError(f"encoders.{name} must be a mapping")

    return validate_humu_model_config(
        {
            "embedding_dim": dim,
            "curvature": curvature,
            "learnable_curvature": bool(config.get("learnable_curvature", False)),
            "encoders": {
                "mol": {
                    "hidden_dim": int(mol.get("hidden_dim") or dim),
                    "n_layers": int(mol.get("n_layers", 2)),
                    "n_heads": int(mol.get("n_heads", 8)),
                    "dropout": float(mol.get("dropout", 0.0)),
                    "use_3d_geometry": bool(mol.get("use_3d_geometry", True)),
                },
                "pocket": {
                    "hidden_dim": int(pocket.get("hidden_dim") or dim),
                    "n_layers": int(pocket.get("n_layers", 1)),
                    "n_heads": int(pocket.get("n_heads", 8)),
                    "dropout": float(pocket.get("dropout", 0.0)),
                    "radius_angstrom": float(pocket.get("radius_angstrom", 20.0)),
                    "max_neighbors": (
                        int(pocket["max_neighbors"])
                        if pocket.get("max_neighbors") is not None
                        else None
                    ),
                    "use_3d_geometry": bool(pocket.get("use_3d_geometry", True)),
                    "use_esm2": bool(pocket.get("use_esm2", False)),
                    "esm2_checkpoint": pocket.get("esm2_checkpoint"),
                    "esm2_layer": int(pocket.get("esm2_layer", 33)),
                    "esm2_dim": int(pocket.get("esm2_dim", 1280)),
                    "esm2_batch_tokens": int(pocket.get("esm2_batch_tokens", 8192)),
                    "esm2_max_sequence_length": (
                        int(pocket["esm2_max_sequence_length"])
                        if pocket.get("esm2_max_sequence_length") is not None
                        else None
                    ),
                    "esm2_required_sources": sorted(
                        {
                            str(source).strip().lower()
                            for source in (pocket.get("esm2_required_sources") or [])
                            if str(source).strip()
                        }
                    ),
                },
                "route": {
                    "hidden_dim": int(route.get("hidden_dim") or dim),
                    "n_layers": int(route.get("n_layers", 2)),
                    "n_heads": int(route.get("n_heads", 8)),
                    "dropout": float(route.get("dropout", 0.0)),
                    "use_tree_pooling": bool(route.get("use_tree_pooling", True)),
                },
            },
        }
    )


def validate_humu_model_config(
    model_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a versioned checkpoint architecture without implicit defaults."""
    if not isinstance(model_config, Mapping):
        raise ValueError("HUMU model_config must be a mapping")
    required_top_level = {
        "embedding_dim",
        "curvature",
        "learnable_curvature",
        "encoders",
    }
    missing_top_level = required_top_level - set(model_config)
    if missing_top_level:
        raise ValueError("HUMU model_config is missing: " + ", ".join(sorted(missing_top_level)))
    encoders = model_config["encoders"]
    if not isinstance(encoders, Mapping):
        raise ValueError("HUMU model_config.encoders must be a mapping")

    required_encoder_fields = {
        "mol": {
            "hidden_dim",
            "n_layers",
            "n_heads",
            "dropout",
            "use_3d_geometry",
        },
        "pocket": {
            "hidden_dim",
            "n_layers",
            "n_heads",
            "dropout",
            "radius_angstrom",
            "max_neighbors",
            "use_3d_geometry",
            "use_esm2",
            "esm2_checkpoint",
            "esm2_layer",
            "esm2_dim",
            "esm2_batch_tokens",
            "esm2_max_sequence_length",
            "esm2_required_sources",
        },
        "route": {
            "hidden_dim",
            "n_layers",
            "n_heads",
            "dropout",
            "use_tree_pooling",
        },
    }
    for name, required_fields in required_encoder_fields.items():
        value = encoders.get(name)
        if not isinstance(value, Mapping):
            raise ValueError(f"HUMU model_config.encoders.{name} must be a mapping")
        missing_fields = required_fields - set(value)
        if missing_fields:
            raise ValueError(
                f"HUMU model_config.encoders.{name} is missing: "
                + ", ".join(sorted(missing_fields))
            )

    dim = int(model_config["embedding_dim"])
    if dim <= 0:
        raise ValueError("HUMU model_config.embedding_dim must be positive")
    curvature = float(model_config["curvature"])
    if curvature != 1.0:
        raise ValueError("HUMU model_config.curvature must be 1.0")

    mol = encoders["mol"]
    pocket = encoders["pocket"]
    route = encoders["route"]
    esm2_checkpoint = pocket["esm2_checkpoint"]
    if esm2_checkpoint is not None and not isinstance(esm2_checkpoint, str):
        raise ValueError("HUMU model_config pocket esm2_checkpoint must be a string or null")
    esm2_required_sources = pocket["esm2_required_sources"]
    if not isinstance(esm2_required_sources, list | tuple | set):
        raise ValueError("HUMU model_config pocket esm2_required_sources must be a sequence")

    return {
        "embedding_dim": dim,
        "curvature": curvature,
        "learnable_curvature": bool(model_config["learnable_curvature"]),
        "encoders": {
            "mol": {
                "hidden_dim": int(mol["hidden_dim"]),
                "n_layers": int(mol["n_layers"]),
                "n_heads": int(mol["n_heads"]),
                "dropout": float(mol["dropout"]),
                "use_3d_geometry": bool(mol["use_3d_geometry"]),
            },
            "pocket": {
                "hidden_dim": int(pocket["hidden_dim"]),
                "n_layers": int(pocket["n_layers"]),
                "n_heads": int(pocket["n_heads"]),
                "dropout": float(pocket["dropout"]),
                "radius_angstrom": float(pocket["radius_angstrom"]),
                "max_neighbors": (
                    int(pocket["max_neighbors"]) if pocket["max_neighbors"] is not None else None
                ),
                "use_3d_geometry": bool(pocket["use_3d_geometry"]),
                "use_esm2": bool(pocket["use_esm2"]),
                "esm2_checkpoint": esm2_checkpoint,
                "esm2_layer": int(pocket["esm2_layer"]),
                "esm2_dim": int(pocket["esm2_dim"]),
                "esm2_batch_tokens": int(pocket["esm2_batch_tokens"]),
                "esm2_max_sequence_length": (
                    int(pocket["esm2_max_sequence_length"])
                    if pocket["esm2_max_sequence_length"] is not None
                    else None
                ),
                "esm2_required_sources": sorted(
                    {
                        str(source).strip().lower()
                        for source in esm2_required_sources
                        if str(source).strip()
                    }
                ),
            },
            "route": {
                "hidden_dim": int(route["hidden_dim"]),
                "n_layers": int(route["n_layers"]),
                "n_heads": int(route["n_heads"]),
                "dropout": float(route["dropout"]),
                "use_tree_pooling": bool(route["use_tree_pooling"]),
            },
        },
    }
