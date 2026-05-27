"""HCIV encoder (hash and learned)."""
import hashlib
from pathlib import Path

import torch
import torch.nn as nn
from mf_core.types.cig import ChemicalIntentGraph
from mf_core.types.humu import HCIV, IntentCone


def cig_to_features(cig: ChemicalIntentGraph, feature_dim: int = 64) -> torch.Tensor:
    features = torch.zeros(feature_dim)
    obj_ids = [o.id for o in cig.objective_nodes]
    if any("fto" in oid for oid in obj_ids):
        features[28] = 1.0
    if any("admet" in oid for oid in obj_ids):
        features[29] = 1.0
    if any("affinity" in oid for oid in obj_ids):
        features[30] = 1.0
    features[0] = float(len(cig.objective_nodes))
    return features


class HCIVEncoder(nn.Module):
    def __init__(self, dim: int = 32, curvature: float = 1.0):
        super().__init__()
        self.dim = dim
        self.curvature = curvature
        self.fc = nn.Linear(64, dim * 2)

    def encode(self, cig: ChemicalIntentGraph) -> tuple[HCIV, IntentCone]:
        features = cig_to_features(cig)
        raw = self.fc(features)
        spatial = raw[: self.dim] * 0.3
        time = torch.sqrt(1.0 + (spatial**2).sum())
        coords = torch.cat([time.unsqueeze(0), spatial], dim=0)

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
