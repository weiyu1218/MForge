"""HUMU route encoder backed by reaction graph features."""
from __future__ import annotations

import torch
import torch.nn as nn
from mf_humu.manifold.learnable_lorentz import LearnableLorentzManifold
from mf_humu.manifold.lorentz import LorentzManifold

_ROUTE_FEATURE_DIM = 18


class HUMURouteEncoder(nn.Module):
    """Encode synthetic route trees into Lorentz manifold embeddings."""

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
        self._route_projection = nn.Sequential(
            nn.Linear(_ROUTE_FEATURE_DIM, dim),
            nn.ReLU(),
            nn.Linear(dim, dim + 1),
        )

    def forward(self, route_data: dict | list[dict]) -> torch.Tensor:
        if isinstance(route_data, list):
            return self.encode_batch(route_data)
        return self.encode(route_data)

    def encode(self, route_data: dict) -> torch.Tensor:
        reactions = self._validate_reactions(route_data)
        features = self._route_features(route_data, reactions).to(self._param_device())
        x = self._route_projection(features).unsqueeze(0)
        return self.manifold._project(x)

    def encode_batch(self, route_data_list: list[dict]) -> torch.Tensor:
        if not route_data_list:
            raise ValueError("route encoder requires at least one route record")
        return torch.cat([self.encode(route_data) for route_data in route_data_list], dim=0)

    def _param_device(self) -> torch.device:
        return self._route_projection[0].weight.device

    def _validate_reactions(self, route_data: dict) -> list[str]:
        reactions = route_data.get("reactions")
        if not isinstance(reactions, list) or not reactions:
            raise ValueError("route encoder requires reactions from a reaction graph")
        clean = [str(reaction).strip() for reaction in reactions if str(reaction).strip()]
        if not clean or any(">>" not in reaction for reaction in clean):
            raise ValueError("route encoder requires reactions in reactants>>products form")
        return clean

    def _route_features(self, route_data: dict, reactions: list[str]) -> torch.Tensor:
        reactant_count = 0
        product_count = 0
        mapped_atoms = 0
        ring_tokens = 0
        hetero_tokens = 0
        carbon_tokens = 0
        halogen_tokens = 0
        charge_tokens = 0
        branch_tokens = 0
        max_reaction_len = 0

        for reaction in reactions:
            lhs, rhs = reaction.split(">>", 1)
            reactant_count += len([part for part in lhs.split(".") if part])
            product_count += len([part for part in rhs.split(".") if part])
            mapped_atoms += reaction.count(":")
            ring_tokens += sum(ch.isdigit() for ch in reaction)
            hetero_tokens += sum(ch in "NOSP" for ch in reaction)
            carbon_tokens += reaction.count("C") + reaction.count("c")
            halogen_tokens += reaction.count("Cl") + reaction.count("Br") + reaction.count("F")
            charge_tokens += reaction.count("+") + reaction.count("-")
            branch_tokens += reaction.count("(") + reaction.count(")")
            max_reaction_len = max(max_reaction_len, len(reaction))

        steps_value = route_data.get("steps", route_data.get("n_steps", len(reactions)))
        steps = float(steps_value if isinstance(steps_value, int | float) else len(reactions))
        score = float(route_data.get("score", 0.0))
        intermediates = route_data.get("intermediates", [])
        if not isinstance(intermediates, list):
            intermediates = []

        denom = max(float(sum(len(r) for r in reactions)), 1.0)
        values = [
            len(reactions) / 32.0,
            steps / 32.0,
            reactant_count / 128.0,
            product_count / 128.0,
            len(intermediates) / 128.0,
            max_reaction_len / 512.0,
            mapped_atoms / 512.0,
            ring_tokens / denom,
            hetero_tokens / denom,
            carbon_tokens / denom,
            halogen_tokens / denom,
            charge_tokens / denom,
            branch_tokens / denom,
            max(0.0, min(score, 1.0)),
            float(route_data.get("route_found", True)),
            float(any("@" in reaction for reaction in reactions)),
            float(any("=" in reaction for reaction in reactions)),
            float(any("#" in reaction for reaction in reactions)),
        ]
        return torch.tensor(values, dtype=torch.float32)
