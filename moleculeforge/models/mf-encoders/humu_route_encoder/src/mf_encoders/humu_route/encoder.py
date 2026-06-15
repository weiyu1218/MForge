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
        hidden_dim: int | None = None,
        n_layers: int = 2,
        n_heads: int = 8,
        dropout: float = 0.0,
        use_tree_pooling: bool = True,
    ):
        super().__init__()
        hidden_dim = int(hidden_dim or dim)
        n_layers = max(1, int(n_layers))
        n_heads = max(1, int(n_heads))
        if hidden_dim % n_heads != 0:
            raise ValueError("route encoder hidden_dim must be divisible by n_heads")
        manifold_cls = LearnableLorentzManifold if learnable_curvature else LorentzManifold
        self.manifold = manifold_cls(curvature=curvature)
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dropout_p = float(dropout)
        self.use_tree_pooling = bool(use_tree_pooling)
        self._feature_projection = nn.Linear(_ROUTE_FEATURE_DIM, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=max(hidden_dim * 2, hidden_dim),
            dropout=self.dropout_p,
            batch_first=True,
        )
        self._route_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
        )
        self._output_projection = nn.Linear(hidden_dim, dim + 1)

    def forward(self, route_data: dict | list[dict]) -> torch.Tensor:
        if isinstance(route_data, list):
            return self.encode_batch(route_data)
        return self.encode(route_data)

    def encode(self, route_data: dict) -> torch.Tensor:
        reactions = self._validate_reactions(route_data)
        features = self._route_features(route_data, reactions).to(self._param_device())
        hidden = torch.relu(self._feature_projection(features)).unsqueeze(0).unsqueeze(0)
        hidden = self._route_transformer(hidden)
        if self.use_tree_pooling:
            pooled = hidden[:, 0, :]
        else:
            pooled = hidden.mean(dim=1)
        x = self._output_projection(pooled)
        return self.manifold._project(x)

    def encode_batch(self, route_data_list: list[dict]) -> torch.Tensor:
        if not route_data_list:
            raise ValueError("route encoder requires at least one route record")
        feature_rows = []
        for route_data in route_data_list:
            reactions = self._validate_reactions(route_data)
            feature_rows.append(self._route_features(route_data, reactions))
        features = torch.stack(feature_rows, dim=0).to(self._param_device())
        hidden = torch.relu(self._feature_projection(features)).unsqueeze(1)
        hidden = self._route_transformer(hidden)
        if self.use_tree_pooling:
            pooled = hidden[:, 0, :]
        else:
            pooled = hidden.mean(dim=1)
        x = self._output_projection(pooled)
        return self.manifold._project(x)

    def _param_device(self) -> torch.device:
        return self._feature_projection.weight.device

    def _validate_reactions(self, route_data: dict) -> list[str]:
        reactions = route_data.get("reactions")
        if reactions is None:
            steps = route_data.get("steps")
            if isinstance(steps, list):
                reactions = [
                    step.get("reaction")
                    for step in steps
                    if isinstance(step, dict) and step.get("reaction")
                ]
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
        tree_stats = self._route_tree_stats(route_data, reactions)
        if isinstance(steps_value, list):
            steps = float(tree_stats["step_count"])
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
            (len(intermediates) + tree_stats["leaf_count"]) / 128.0,
            max_reaction_len / 512.0,
            mapped_atoms / 512.0,
            ring_tokens / denom,
            hetero_tokens / denom,
            carbon_tokens / denom,
            halogen_tokens / denom,
            charge_tokens / denom,
            (branch_tokens + tree_stats["branching_edges"]) / denom,
            max(0.0, min(score, 1.0)),
            tree_stats["max_depth"] / 32.0,
            float(any("@" in reaction for reaction in reactions)),
            float(any("=" in reaction for reaction in reactions)),
            float(any("#" in reaction for reaction in reactions)),
        ]
        return torch.tensor(values, dtype=torch.float32)

    def _route_tree_stats(self, route_data: dict, reactions: list[str]) -> dict[str, float]:
        steps = route_data.get("steps")
        if not isinstance(steps, list) or not steps:
            step_count = float(route_data.get("n_steps", len(reactions)) or len(reactions))
            return {
                "step_count": step_count,
                "branching_edges": 0.0,
                "leaf_count": max(step_count, 1.0),
                "max_depth": max(step_count, 1.0),
            }

        children_by_step: dict[str, set[str]] = {}
        parent_by_step: dict[str, str] = {}
        step_ids: list[str] = []
        for index, step in enumerate(steps):
            if not isinstance(step, dict):
                continue
            step_id = str(step.get("step_id") or step.get("id") or index)
            step_ids.append(step_id)
            children_by_step.setdefault(step_id, set())
            parent = step.get("parent_step_id", step.get("parent_id"))
            if parent is not None and str(parent):
                parent_id = str(parent)
                parent_by_step[step_id] = parent_id
                children_by_step.setdefault(parent_id, set()).add(step_id)
            children = step.get("children", step.get("child_step_ids", []))
            if isinstance(children, list):
                for child in children:
                    child_id = str(child)
                    if not child_id:
                        continue
                    parent_by_step[child_id] = step_id
                    children_by_step.setdefault(step_id, set()).add(child_id)
                    children_by_step.setdefault(child_id, set())

        if not step_ids:
            step_ids = [str(index) for index, _reaction in enumerate(reactions)]
            for step_id in step_ids:
                children_by_step.setdefault(step_id, set())

        def depth(step_id: str, seen: set[str] | None = None) -> int:
            seen = set() if seen is None else set(seen)
            if step_id in seen:
                return 1
            seen.add(step_id)
            parent_id = parent_by_step.get(step_id)
            if parent_id is None:
                return 1
            return 1 + depth(parent_id, seen)

        leaf_count = sum(
            1
            for step_id in step_ids
            if not children_by_step.get(step_id)
        )
        return {
            "step_count": float(len(step_ids)),
            "branching_edges": float(
                sum(max(len(children) - 1, 0) for children in children_by_step.values())
            ),
            "leaf_count": float(max(leaf_count, 1)),
            "max_depth": float(max(depth(step_id) for step_id in step_ids)),
        }
