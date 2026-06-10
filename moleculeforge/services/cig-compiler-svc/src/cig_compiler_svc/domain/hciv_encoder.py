"""HCIV encoder (canonical, hash, and learned)."""
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from mf_core.types.cig import ChemicalIntentGraph
from mf_core.types.humu import HCIV, IntentCone


def cig_to_features(cig: ChemicalIntentGraph, feature_dim: int = 64) -> torch.Tensor:
    features = torch.zeros(feature_dim)
    obj_ids = [o.id for o in cig.objective_nodes]
    edges = getattr(cig, "edges", [])
    hyperedges = getattr(cig, "hyperedges", [])
    if any("admet" in oid for oid in obj_ids):
        features[29] = 1.0
    if any("affinity" in oid for oid in obj_ids):
        features[30] = 1.0
    features[0] = float(len(cig.objective_nodes))
    if edges:
        features[31] = float(len(edges))
        features[32] = float(sum(edge.strength for edge in edges) / len(edges))
        features[33] = float(sum(1 for edge in edges if edge.relation == "trade_off"))
    if hyperedges:
        features[34] = float(len(hyperedges))
        features[35] = float(sum(edge.strength for edge in hyperedges) / len(hyperedges))
        features[36] = float(
            sum(len(edge.source_ids) + len(edge.target_ids) for edge in hyperedges)
            / len(hyperedges)
        )
    return features


class HCIVEncoder(nn.Module):
    def __init__(self, dim: int = 32, curvature: float = 1.0, hidden_dim: int = 64):
        super().__init__()
        self.dim = dim
        self.curvature = curvature
        self.hidden_dim = hidden_dim
        self.node_encoder = nn.Linear(64, hidden_dim)
        self.edge_encoder = nn.Linear(64, hidden_dim)
        self.hyperedge_encoder = nn.Linear(64, hidden_dim)
        self.graph_projection = nn.Sequential(
            nn.Linear(hidden_dim + 64, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 64),
        )
        self.fc = nn.Linear(64, dim * 2)

    def encode(self, cig: ChemicalIntentGraph) -> tuple[HCIV, IntentCone]:
        coords = self.forward_coordinates(cig)
        hciv = HCIV(
            coordinates=coords.detach().tolist(),
            dim=self.dim,
            curvature=self.curvature,
        )
        half_angle = 0.5
        cone = IntentCone(
            apex=hciv,
            axis_direction=hciv,
            axis=list(hciv.coordinates),
            half_angle=half_angle,
            angle_radians=half_angle,
            length=1.0,
            curvature=self.curvature,
        )
        return hciv, cone

    def forward_coordinates(self, cig: ChemicalIntentGraph) -> torch.Tensor:
        device = next(self.parameters()).device
        features = cig_to_features(cig).to(device)
        graph_embedding = self._encode_directed_hypergraph(cig, features)
        raw = self.fc(graph_embedding)
        spatial = raw[: self.dim] * 0.3
        time = torch.sqrt(1.0 + (spatial**2).sum())
        return torch.cat([time.unsqueeze(0), spatial], dim=0)

    def _encode_directed_hypergraph(
        self,
        cig: ChemicalIntentGraph,
        global_features: torch.Tensor,
    ) -> torch.Tensor:
        node_ids = [node.id for node in cig.objective_nodes]
        if not node_ids:
            return global_features
        node_index = {node_id: index for index, node_id in enumerate(node_ids)}
        node_features = torch.stack([
            _objective_node_features(node)
            for node in cig.objective_nodes
        ]).to(global_features.device)
        node_state = torch.tanh(self.node_encoder(node_features))
        messages = torch.zeros_like(node_state)

        for edge in getattr(cig, "edges", []):
            source_idx = node_index.get(edge.source_id)
            target_idx = node_index.get(edge.target_id)
            if source_idx is None or target_idx is None:
                continue
            edge_state = torch.tanh(
                self.edge_encoder(
                    _objective_edge_features(edge, node_index).to(global_features.device)
                )
            )
            directed_message = torch.tanh(node_state[source_idx] + edge_state)
            messages[target_idx] += directed_message * float(edge.strength or 1.0)
            messages[source_idx] -= edge_state * 0.1

        for hyperedge in getattr(cig, "hyperedges", []):
            source_indices = [
                node_index[node_id]
                for node_id in hyperedge.source_ids
                if node_id in node_index
            ]
            target_indices = [
                node_index[node_id]
                for node_id in hyperedge.target_ids
                if node_id in node_index
            ]
            if not source_indices or not target_indices:
                continue
            hyperedge_state = torch.tanh(
                self.hyperedge_encoder(
                    _objective_hyperedge_features(hyperedge, node_index).to(
                        global_features.device
                    )
                )
            )
            source_state = node_state[source_indices].mean(dim=0)
            directed_message = torch.tanh(source_state + hyperedge_state)
            for target_idx in target_indices:
                messages[target_idx] += directed_message * float(hyperedge.strength or 1.0)
            for source_idx in source_indices:
                messages[source_idx] -= hyperedge_state * 0.1

        node_state = torch.tanh(node_state + messages)
        graph_state = node_state.mean(dim=0)
        return self.graph_projection(torch.cat([graph_state, global_features], dim=0))


def _objective_node_features(node, feature_dim: int = 64) -> torch.Tensor:
    features = torch.zeros(feature_dim)
    features[0] = float(getattr(node, "weight", 1.0) or 0.0)
    features[1] = float(getattr(node, "target_value", 0.0) or 0.0)
    features[2] = float(getattr(node, "target_min", 0.0) or 0.0)
    features[3] = float(getattr(node, "target_max", 0.0) or 0.0)
    features[4] = float(getattr(node, "pareto_tier", 1) or 1) / 10.0
    _bucket_one_hot(features, 8, 8, str(getattr(node, "type", "")))
    _bucket_one_hot(features, 16, 16, str(getattr(node, "oracle", "")))
    _bucket_one_hot(features, 32, 16, str(getattr(node, "id", "")))
    _bucket_one_hot(features, 48, 8, str(getattr(node, "property", "")))
    _bucket_one_hot(features, 56, 8, str(getattr(node, "name", "")))
    return features


def _objective_edge_features(
    edge,
    node_index: dict[str, int],
    feature_dim: int = 64,
) -> torch.Tensor:
    features = torch.zeros(feature_dim)
    n_nodes = max(1, len(node_index) - 1)
    source_idx = node_index.get(edge.source_id, 0)
    target_idx = node_index.get(edge.target_id, 0)
    features[0] = float(source_idx) / n_nodes
    features[1] = float(target_idx) / n_nodes
    features[2] = float(edge.strength)
    _bucket_one_hot(features, 8, 16, str(edge.relation))
    _bucket_one_hot(features, 24, 16, str(edge.source_id))
    _bucket_one_hot(features, 40, 16, str(edge.target_id))
    _bucket_one_hot(features, 56, 8, f"{edge.source_id}->{edge.target_id}:{edge.relation}")
    return features


def _objective_hyperedge_features(
    edge,
    node_index: dict[str, int],
    feature_dim: int = 64,
) -> torch.Tensor:
    features = torch.zeros(feature_dim)
    n_nodes = max(1, len(node_index))
    features[0] = float(len(edge.source_ids)) / n_nodes
    features[1] = float(len(edge.target_ids)) / n_nodes
    features[2] = float(edge.strength)
    _bucket_one_hot(features, 8, 16, str(edge.relation))
    _bucket_one_hot(features, 24, 16, "|".join(edge.source_ids))
    _bucket_one_hot(features, 40, 16, "|".join(edge.target_ids))
    _bucket_one_hot(
        features,
        56,
        8,
        f"{','.join(edge.source_ids)}->{','.join(edge.target_ids)}:{edge.relation}",
    )
    return features


def _bucket_one_hot(
    features: torch.Tensor,
    start: int,
    width: int,
    value: str,
) -> None:
    if not value:
        return
    features[start + _stable_bucket(value, width)] = 1.0


def _stable_bucket(value: str, width: int) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % width


def load_hciv_encoder_checkpoint(
    checkpoint_path: str,
    dim: int,
    curvature: float = 1.0,
) -> HCIVEncoder:
    path = Path(checkpoint_path)
    if not path.exists():
        raise RuntimeError(f"HCIV_CHECKPOINT_PATH does not exist: {checkpoint_path}")

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise RuntimeError("HCIV checkpoint must be a state_dict-compatible dict")

    state_dict = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state_dict, dict):
        raise RuntimeError("HCIV checkpoint state_dict must be a dict")

    encoder = HCIVEncoder(dim=dim, curvature=curvature)
    encoder.load_state_dict(state_dict)
    encoder.eval()
    return encoder


def hash_encode_hciv(
    cig: ChemicalIntentGraph,
    dim: int = 128,
    seed: int = 42,
) -> HCIV:
    canonical = "|".join([
        cig.intent_id,
        str(sorted(o.id for o in cig.objective_nodes)),
        str(sorted([(o.id, o.oracle, o.type.value) for o in cig.objective_nodes])),
        str(
            sorted(
                (
                    edge.source_id,
                    edge.target_id,
                    edge.relation,
                    edge.strength,
                )
                for edge in getattr(cig, "edges", [])
            )
        ),
        str(
            sorted(
                (
                    tuple(edge.source_ids),
                    tuple(edge.target_ids),
                    edge.relation,
                    edge.strength,
                )
                for edge in getattr(cig, "hyperedges", [])
            )
        ),
        cig.source_user_input,
        str(seed),
    ])

    digest = hashlib.sha256(canonical.encode("utf-8")).digest()
    values = bytearray()
    counter = 0
    while len(values) < dim:
        values.extend(hashlib.sha256(digest + counter.to_bytes(4, "big")).digest())
        counter += 1
    spatial = torch.tensor(
        [(byte / 255.0 - 0.5) * 0.6 for byte in values[:dim]],
        dtype=torch.float32,
    )
    time = torch.sqrt(1.0 + (spatial ** 2).sum())
    coords = torch.cat([time.unsqueeze(0), spatial], dim=0)

    return HCIV(
        coordinates=coords.tolist(),
        dim=dim,
        curvature=1.0,
    )


def canonical_encode_hciv(
    cig: ChemicalIntentGraph,
    dim: int = 128,
    curvature: float = 1.0,
) -> tuple[HCIV, IntentCone]:
    if dim <= 0:
        raise RuntimeError("HCIV dim must be positive")
    canonical = _canonical_cig_payload(cig)
    digest = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).digest()
    values = bytearray()
    counter = 0
    while len(values) < dim * 2:
        values.extend(hashlib.sha256(digest + counter.to_bytes(4, "big")).digest())
        counter += 1
    raw = torch.tensor(
        [
            (int.from_bytes(values[index * 2 : index * 2 + 2], "big") / 65535.0) - 0.5
            for index in range(dim)
        ],
        dtype=torch.float32,
    )
    norm = torch.linalg.vector_norm(raw).clamp_min(1e-12)
    spatial = raw / norm
    time = torch.sqrt(torch.tensor(1.0 / curvature, dtype=torch.float32) + (spatial**2).sum())
    hciv = HCIV(
        coordinates=torch.cat([time.unsqueeze(0), spatial], dim=0).tolist(),
        dim=dim,
        curvature=curvature,
    )
    cone = IntentCone(
        apex=hciv,
        axis_direction=hciv,
        axis=list(hciv.coordinates),
        half_angle=0.5,
        angle_radians=0.5,
        length=1.0,
        curvature=curvature,
    )
    return hciv, cone


def _canonical_cig_payload(cig: ChemicalIntentGraph) -> dict[str, Any]:
    payload = cig.model_dump(mode="json", by_alias=True)
    payload.pop("intent_id", None)
    payload.pop("created_at", None)
    return payload
