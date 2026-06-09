#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from mf_core.geometry import normalize_lorentz_embedding


class MoleculeEncoder(Protocol):
    def encode(self, smiles: str) -> list[float]:
        ...


def score_humu_records(
    smiles_list: list[str],
    *,
    encoder: MoleculeEncoder,
    intent_cone: Mapping[str, object] | None = None,
    expected_dim: int = 129,
    curvature: float = 1.0,
) -> dict[str, dict[str, object]]:
    if not smiles_list:
        raise ValueError("CReM HUMU scorer requires at least one SMILES")
    records: dict[str, dict[str, object]] = {}
    for smiles in smiles_list:
        embedding = normalize_lorentz_embedding(
            encoder.encode(smiles),
            expected_dim=expected_dim,
            curvature=curvature,
        )
        if embedding is None:
            raise RuntimeError(f"CReM HUMU scorer produced invalid embedding for {smiles}")
        record: dict[str, object] = {
            "humu_embedding": embedding,
            "humu_embedding_dim": len(embedding),
        }
        alignment = _alignment_score(embedding, intent_cone)
        if alignment is not None:
            record["humu_alignment_score"] = alignment
        records[str(smiles)] = record
    return records


def _alignment_score(
    embedding: list[float],
    intent_cone: Mapping[str, object] | None,
) -> float | None:
    if not intent_cone:
        return None
    axis = intent_cone.get("axis")
    if isinstance(axis, str | bytes | bytearray) or not isinstance(axis, Sequence):
        return None
    try:
        axis_values = [float(value) for value in axis]
    except (TypeError, ValueError):
        return None
    if len(axis_values) == len(embedding) + 1:
        axis_values = axis_values[1:]
    dim = min(len(axis_values), len(embedding))
    if dim == 0:
        return None
    left = axis_values[:dim]
    right = embedding[:dim]
    axis_norm = math.sqrt(math.fsum(value * value for value in left))
    embedding_norm = math.sqrt(math.fsum(value * value for value in right))
    if axis_norm == 0.0 or embedding_norm == 0.0:
        return 0.0
    dot = math.fsum(a * b for a, b in zip(left, right, strict=True))
    return float(dot / (axis_norm * embedding_norm))


def _read_request() -> dict[str, Any]:
    payload = json.load(sys.stdin)
    if not isinstance(payload, dict):
        raise RuntimeError("CReM HUMU scorer request must be a JSON object")
    return payload


def _smiles_from_request(payload: Mapping[str, object]) -> list[str]:
    raw = payload.get("smiles")
    if not isinstance(raw, list) or not raw:
        raise RuntimeError("CReM HUMU scorer request requires non-empty smiles list")
    smiles = [str(item) for item in raw]
    if not all(item for item in smiles):
        raise RuntimeError("CReM HUMU scorer request contains empty SMILES")
    return smiles


def _load_encoder(checkpoint_path: str, device: str) -> MoleculeEncoder:
    from mf_generators.fragfm.humu_labeling import LocalHUMUMoleculeEncoder

    checkpoint = Path(checkpoint_path)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"HUMU checkpoint not found: {checkpoint}")
    return LocalHUMUMoleculeEncoder(checkpoint, device=device)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CReM HUMU JSON scorer")
    parser.add_argument(
        "--checkpoint",
        default=os.environ.get("HUMU_CHECKPOINT_PATH", ""),
    )
    parser.add_argument("--device", default=os.environ.get("HUMU_DEVICE", "cpu"))
    args = parser.parse_args(argv)
    try:
        if not args.checkpoint:
            raise RuntimeError("HUMU checkpoint is required via --checkpoint or HUMU_CHECKPOINT_PATH")
        payload = _read_request()
        encoder = _load_encoder(args.checkpoint, args.device)
        records = score_humu_records(
            _smiles_from_request(payload),
            encoder=encoder,
            intent_cone=payload.get("intent_cone") if isinstance(payload.get("intent_cone"), Mapping) else None,
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    json.dump({"records": records}, sys.stdout, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
