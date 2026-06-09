"""Supervised HCIV encoder training helpers."""
from __future__ import annotations

import json
import math
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch
from mf_core.geometry import normalize_lorentz_embedding
from mf_core.types.cig import ChemicalIntentGraph

from cig_compiler_svc.domain.hciv_encoder import HCIVEncoder


@dataclass(frozen=True)
class HCIVTrainingExample:
    example_id: str
    cig: ChemicalIntentGraph
    target_coordinates: torch.Tensor
    weight: float = 1.0


def load_hciv_training_examples(
    path: str | Path,
    dim: int = 128,
    curvature: float = 1.0,
) -> list[HCIVTrainingExample]:
    records = _load_json_or_jsonl(Path(path))
    examples = [
        _example_from_record(index, record, dim=dim, curvature=curvature)
        for index, record in enumerate(records)
    ]
    if not examples:
        raise ValueError("HCIV training data requires at least one example")
    return examples


def _load_json_or_jsonl(path: Path) -> list[object]:
    if not path.exists():
        raise FileNotFoundError(f"HCIV training data not found: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    payload = json.loads(text)
    if isinstance(payload, Mapping):
        records = payload.get("examples", payload.get("records"))
        if records is None:
            records = [payload]
    else:
        records = payload
    if not isinstance(records, list):
        raise ValueError("HCIV training data must be a JSON list or JSONL records")
    return records


def _example_from_record(
    index: int,
    record: object,
    dim: int,
    curvature: float,
) -> HCIVTrainingExample:
    if not isinstance(record, Mapping):
        raise ValueError("HCIV training record must be a JSON object")
    cig_payload = record.get("cig")
    if not isinstance(cig_payload, Mapping):
        raise ValueError("HCIV training record requires cig object")
    target = _target_coordinates(record, dim=dim, curvature=curvature)
    weight = float(record.get("weight", 1.0))
    if not math.isfinite(weight) or weight <= 0.0:
        raise ValueError("HCIV training record weight must be positive and finite")
    return HCIVTrainingExample(
        example_id=str(record.get("id", index)),
        cig=ChemicalIntentGraph.model_validate(cig_payload),
        target_coordinates=target,
        weight=weight,
    )


def _target_coordinates(record: Mapping, dim: int, curvature: float) -> torch.Tensor:
    target = record.get("target_hciv", record.get("hciv"))
    if isinstance(target, Mapping):
        target = target.get("coordinates")
    if not isinstance(target, list):
        raise ValueError("HCIV training record requires target_hciv coordinates")
    if len(target) != dim + 1:
        raise ValueError("HCIV target_hciv must have dim + 1 coordinates")
    normalized_target = normalize_lorentz_embedding(
        target,
        expected_dim=dim + 1,
        curvature=curvature,
    )
    if normalized_target is None:
        raise ValueError("HCIV target_hciv coordinates must be valid Lorentz coordinates")
    return torch.tensor(normalized_target, dtype=torch.float32)


def train_hciv_encoder_checkpoint(
    data_path: str | Path,
    output_checkpoint: str | Path,
    *,
    manifest_path: str | Path | None = None,
    dim: int = 128,
    curvature: float = 1.0,
    epochs: int = 5,
    batch_size: int = 32,
    device: str | torch.device = "cpu",
    learning_rate: float = 1e-3,
) -> dict[str, object]:
    examples = load_hciv_training_examples(data_path, dim=dim, curvature=curvature)
    torch.manual_seed(0)
    device = torch.device(device)
    encoder = HCIVEncoder(dim=dim, curvature=curvature).to(device)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=float(learning_rate))
    batch_size = max(1, int(batch_size))
    started = time.time()
    losses: list[float] = []

    for _epoch in range(max(1, int(epochs))):
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, len(examples), batch_size):
            batch = examples[start : start + batch_size]
            batch_loss = torch.zeros((), dtype=torch.float32, device=device)
            batch_weight = 0.0
            for example in batch:
                predicted = encoder.forward_coordinates(example.cig)
                target = example.target_coordinates.to(device)
                weight = float(example.weight)
                batch_loss = batch_loss + weight * torch.mean((predicted - target) ** 2)
                batch_weight += weight
            batch_loss = batch_loss / max(batch_weight, 1e-12)
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            epoch_loss += float(batch_loss.detach().cpu())
            n_batches += 1
        losses.append(epoch_loss / max(n_batches, 1))

    output_path = Path(output_checkpoint)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "schema": "moleculeforge.cig_compiler.hciv_encoder.v1",
        "state_dict": encoder.cpu().state_dict(),
        "dim": int(dim),
        "curvature": float(curvature),
    }
    torch.save(checkpoint, output_path)

    manifest = {
        "schema": "moleculeforge.cig_compiler.hciv_encoder.v1",
        "checkpoint": str(output_path),
        "source_data": str(Path(data_path)),
        "dim": int(dim),
        "curvature": float(curvature),
        "example_count": len(examples),
        "epochs": max(1, int(epochs)),
        "batch_size": batch_size,
        "final_loss": losses[-1],
        "elapsed_ms": int((time.time() - started) * 1000),
    }
    if manifest_path is not None:
        manifest_output = Path(manifest_path)
        manifest_output.parent.mkdir(parents=True, exist_ok=True)
        manifest_output.write_text(
            json.dumps(manifest, sort_keys=True),
            encoding="utf-8",
        )
    return manifest
