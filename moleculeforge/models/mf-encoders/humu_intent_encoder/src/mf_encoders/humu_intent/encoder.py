"""HUMU intent encoder backed by CIG objective features."""
from __future__ import annotations

import torch
import torch.nn as nn
from mf_core.types.humu import IntentCone
from mf_humu.manifold.learnable_lorentz import LearnableLorentzManifold
from mf_humu.manifold.lorentz import LorentzManifold

_INTENT_PROPERTIES = [
    "binding_affinity",
    "selectivity",
    "solubility",
    "permeability",
    "clearance",
    "toxicity",
    "hERG",
    "logp",
    "mw",
    "sa_score",
]
_INTENT_FEATURE_DIM = len(_INTENT_PROPERTIES) * 3 + 4


class HUMUIntentEncoder(nn.Module):
    """Encode CIG objective graphs into Lorentz manifold embeddings."""

    def __init__(
        self,
        dim: int = 128,
        curvature: float = 1.0,
        learnable_curvature: bool = False,
    ):
        super().__init__()
        manifold_cls = LearnableLorentzManifold if learnable_curvature else LorentzManifold
        self.manifold = manifold_cls(curvature=curvature)
        self.dim = dim
        self._projection = nn.Sequential(
            nn.Linear(_INTENT_FEATURE_DIM, dim),
            nn.ReLU(),
            nn.Linear(dim, dim + 1),
        )

    def forward(self, intent_spec: dict | object) -> torch.Tensor:
        if isinstance(intent_spec, list):
            return self.encode_batch(intent_spec)
        return self.encode(intent_spec)

    def encode(self, intent_spec: dict | object) -> torch.Tensor:
        """Encode a CIG or explicit intent dict without hash or random fallback."""
        features, weights = self._intent_features(intent_spec)
        x = self._projection(features.to(self._param_device())).unsqueeze(0)
        x = self.manifold._project(x)
        if weights:
            tangent = torch.zeros_like(x)
            total_weight = sum(abs(value) for value in weights.values())
            if total_weight > 0:
                for prop, value in weights.items():
                    if prop in _INTENT_PROPERTIES:
                        component_idx = _INTENT_PROPERTIES.index(prop) + 1
                        tangent[0, component_idx] = float(value) / total_weight
                x = self.manifold.expmap(x, tangent)
        return self.manifold._project(x)

    def encode_batch(self, intent_specs: list[dict | object]) -> torch.Tensor:
        if not intent_specs:
            raise ValueError("intent encoder requires at least one intent record")
        return torch.cat([self.encode(intent_spec) for intent_spec in intent_specs], dim=0)

    def encode_to_cone(self, intent_spec: dict) -> IntentCone:
        """Encode a design intent as an IntentCone for guided generation.

        Args:
            intent_spec: Design intent specification dictionary.

        Returns:
            IntentCone with axis on the Lorentz manifold.
        """
        embedding = self.encode(intent_spec)
        axis = embedding.squeeze(0).tolist()
        _, weights = self._intent_features(intent_spec)
        half_angle = 0.1 + 0.4 * (1.0 / max(1, len(weights)))
        return IntentCone(
            axis=axis,
            half_angle=half_angle,
            curvature=float(self.manifold.k.detach().cpu().item())
            if hasattr(self.manifold.k, "detach")
            else self.manifold.k,
            property_weights=weights,
        )

    def _param_device(self) -> torch.device:
        return self._projection[0].weight.device

    def _intent_features(self, intent_spec: dict | object) -> tuple[torch.Tensor, dict[str, float]]:
        targets, weights, constraints, objective_count, edge_count = self._normalize_intent(
            intent_spec
        )
        if not targets and not weights and objective_count == 0:
            raise ValueError("intent encoder requires CIG objectives or explicit targets")

        values = []
        for prop in _INTENT_PROPERTIES:
            values.append(float(targets.get(prop, 0.0)))
            values.append(float(weights.get(prop, 0.0)))
            constraint = constraints.get(prop)
            if isinstance(constraint, (list, tuple)) and len(constraint) == 2:
                values.append(float(constraint[1]) - float(constraint[0]))
            else:
                values.append(0.0)
        values.extend([
            objective_count / 64.0,
            edge_count / 128.0,
            float(bool(constraints)),
            float(bool(weights)),
        ])
        return torch.tensor(values, dtype=torch.float32), weights

    def _normalize_intent(
        self,
        intent_spec: dict | object,
    ) -> tuple[dict[str, float], dict[str, float], dict, float, float]:
        if isinstance(intent_spec, dict):
            targets = {
                self._canonical_property(key): float(value)
                for key, value in intent_spec.get("targets", {}).items()
            }
            weights = {
                self._canonical_property(key): float(value)
                for key, value in intent_spec.get("weights", {}).items()
            }
            constraints = {
                self._canonical_property(key): value
                for key, value in intent_spec.get("constraints", {}).items()
            }
            return targets, weights, constraints, float(len(targets)), 0.0

        objective_nodes = getattr(intent_spec, "objective_nodes", [])
        targets: dict[str, float] = {}
        weights: dict[str, float] = {}
        constraints: dict = {}
        for node in objective_nodes:
            prop = self._canonical_property(
                getattr(node, "property", "") or getattr(node, "name", "")
            )
            targets[prop] = float(getattr(node, "target_value", 0.0))
            weights[prop] = float(getattr(node, "weight", 1.0))
            target_min = getattr(node, "target_min", None)
            target_max = getattr(node, "target_max", None)
            if target_min is not None and target_max is not None:
                constraints[prop] = [float(target_min), float(target_max)]
        edge_count = float(len(getattr(intent_spec, "edges", [])))
        return targets, weights, constraints, float(len(objective_nodes)), edge_count

    def _canonical_property(self, prop: str) -> str:
        normalized = prop.strip()
        aliases = {
            "affinity": "binding_affinity",
            "delta_g": "binding_affinity",
            "herg": "hERG",
            "molecular_weight": "mw",
            "synthetic_accessibility": "sa_score",
        }
        return aliases.get(normalized, normalized)
