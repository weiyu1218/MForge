"""Derive HUMU-labeled FragFM JSONL records from product SMILES."""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

from mf_core.geometry import normalize_lorentz_embedding


class MoleculeEncoder(Protocol):
    def encode(self, smiles: str) -> list[float]:
        """Return a HUMU molecule embedding for one SMILES string."""
        ...


class LocalHUMUMoleculeEncoder:
    """Small adapter around the frozen local HUMU molecule encoder checkpoint."""

    def __init__(self, checkpoint_path: str | Path, device: str = "cpu"):
        import torch
        from mf_encoders.humu_mol.encoder import HUMUMoleculeEncoder

        self.device = torch.device(device)
        self.encoder = HUMUMoleculeEncoder(dim=128, curvature=1.0).to(self.device)
        _load_humu_molecule_encoder_checkpoint(self.encoder, checkpoint_path, self.device)
        self.encoder.eval()

    def encode(self, smiles: str) -> list[float]:
        import torch

        with torch.no_grad():
            embedding = self.encoder(smiles)
        return [float(value) for value in embedding.detach().cpu().reshape(-1).tolist()]


def label_fragfm_records(
    *,
    input_path: str | Path,
    output_path: str | Path,
    encoder: MoleculeEncoder | None = None,
    checkpoint_path: str | Path | None = None,
    device: str = "cpu",
    expected_humu_dim: int = 129,
    curvature: float = 1.0,
    min_coverage: float = 0.0,
    strict: bool = False,
) -> dict[str, Any]:
    """Write a derived JSONL with valid product-level HUMU embeddings."""

    if expected_humu_dim <= 1:
        raise ValueError("expected_humu_dim must be greater than 1")
    if curvature <= 0.0:
        raise ValueError("curvature must be positive")
    if not 0.0 <= min_coverage <= 1.0:
        raise ValueError("min_coverage must be in [0, 1]")

    input_file = Path(input_path)
    output_file = Path(output_path)
    if input_file.resolve() == output_file.resolve():
        raise ValueError("FragFM HUMU labeling output path must differ from input path")
    if not input_file.is_file():
        raise FileNotFoundError(f"FragFM records file not found: {input_file}")
    if encoder is None:
        checkpoint = checkpoint_path or os.environ.get("HUMU_CHECKPOINT_PATH")
        if not checkpoint:
            raise RuntimeError("HUMU checkpoint is required via --checkpoint or HUMU_CHECKPOINT_PATH")
        encoder = LocalHUMUMoleculeEncoder(checkpoint, device=device)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    messages: list[str] = []
    total_records = 0
    encoded_records = 0
    invalid_smiles = 0
    invalid_embeddings = 0
    missing_product = 0

    with input_file.open(encoding="utf-8") as source, output_file.open(
        "w",
        encoding="utf-8",
    ) as sink:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            total_records += 1
            row = _load_jsonl_row(line, line_number=line_number)
            product = row.get("product")
            if not isinstance(product, str) or not product.strip():
                missing_product += 1
                messages.append(f"record {line_number} missing product")
                continue
            try:
                embedding = encoder.encode(product)
            except ValueError as exc:
                invalid_smiles += 1
                messages.append(f"record {line_number} could not be encoded: {exc}")
                continue
            except Exception as exc:  # pragma: no cover - depends on encoder backend
                invalid_smiles += 1
                messages.append(f"record {line_number} could not be encoded: {exc}")
                continue
            normalized = normalize_lorentz_embedding(
                embedding,
                expected_dim=expected_humu_dim,
                curvature=curvature,
            )
            if normalized is None:
                invalid_embeddings += 1
                messages.append(f"record {line_number} produced invalid HUMU embedding")
                continue
            labeled = dict(row)
            labeled["humu_embedding"] = normalized
            sink.write(json.dumps(labeled, sort_keys=True) + "\n")
            encoded_records += 1

    skipped_records = total_records - encoded_records
    coverage = encoded_records / max(total_records, 1)
    if total_records == 0:
        messages.append("input contained no records")
    if encoded_records == 0:
        messages.append("no FragFM records were HUMU-labeled")
    if coverage < min_coverage:
        messages.append(
            f"HUMU labeling coverage {coverage:.4f} is below required {min_coverage:.4f}"
        )
    status = "pass"
    if encoded_records == 0 or coverage < min_coverage:
        status = "fail"
    if strict and skipped_records:
        status = "fail"

    return {
        "schema_version": "fragfm_humu_labeling_report.v1",
        "status": status,
        "input_path": str(input_file),
        "output_path": str(output_file),
        "total_records": total_records,
        "encoded_records": encoded_records,
        "skipped_records": skipped_records,
        "invalid_smiles": invalid_smiles,
        "invalid_embeddings": invalid_embeddings,
        "missing_product": missing_product,
        "humu_embedding_coverage": coverage,
        "expected_humu_dim": int(expected_humu_dim),
        "curvature": float(curvature),
        "messages": messages,
    }


def _load_jsonl_row(line: str, *, line_number: int) -> dict[str, Any]:
    payload = json.loads(line)
    if not isinstance(payload, Mapping):
        raise ValueError(f"FragFM JSONL record {line_number} must be a JSON object")
    return dict(payload)


def _load_humu_molecule_encoder_checkpoint(
    encoder,
    checkpoint_path: str | Path,
    device,
) -> None:
    import torch

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("HUMU checkpoint must be a state dictionary")
    state_dict = _extract_humu_molecule_state_dict(checkpoint, encoder)
    expected_keys = set(encoder.state_dict())
    missing_keys = sorted(expected_keys.difference(state_dict))
    if missing_keys:
        raise RuntimeError(
            "HUMU checkpoint is missing HUMUMoleculeEncoder keys: "
            + ", ".join(missing_keys[:10])
        )
    try:
        encoder.load_state_dict(
            {key: value for key, value in state_dict.items() if key in expected_keys},
            strict=True,
        )
    except RuntimeError as exc:
        raise RuntimeError(
            f"HUMU checkpoint is incompatible with HUMUMoleculeEncoder: {checkpoint_path}"
        ) from exc


def _extract_humu_molecule_state_dict(checkpoint: dict, encoder) -> dict:
    for key in ("encoder_mol", "state_dict", "model"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return _strip_state_prefixes(value)
    expected_keys = set(encoder.state_dict())
    if expected_keys.intersection(checkpoint):
        return _strip_state_prefixes(checkpoint)
    raise ValueError("HUMU checkpoint does not contain encoder_mol state")


def _strip_state_prefixes(state_dict: dict) -> dict:
    prefixes = ("module.", "_orig_mod.", "inner.")
    normalized = {}
    for key, value in state_dict.items():
        normalized_key = key
        if isinstance(normalized_key, str):
            changed = True
            while changed:
                changed = False
                for prefix in prefixes:
                    if normalized_key.startswith(prefix):
                        normalized_key = normalized_key.removeprefix(prefix)
                        changed = True
        normalized[normalized_key] = value
    return normalized


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Label FragFM JSONL records with HUMU embeddings")
    parser.add_argument("--input", required=True, help="Input FragFM JSONL records")
    parser.add_argument("--output", required=True, help="Derived HUMU-labeled JSONL output")
    parser.add_argument(
        "--checkpoint",
        default=os.environ.get("HUMU_CHECKPOINT_PATH", ""),
        help="Frozen HUMU checkpoint path; defaults to HUMU_CHECKPOINT_PATH",
    )
    parser.add_argument("--device", default=os.environ.get("HUMU_DEVICE", "cpu"))
    parser.add_argument("--humu-dim", type=int, default=129)
    parser.add_argument("--curvature", type=float, default=1.0)
    parser.add_argument("--min-coverage", type=float, default=0.0)
    parser.add_argument("--report", default="", help="Optional JSON report output")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero when coverage is incomplete or below threshold",
    )
    args = parser.parse_args(argv)

    report = label_fragfm_records(
        input_path=args.input,
        output_path=args.output,
        checkpoint_path=args.checkpoint,
        device=args.device,
        expected_humu_dim=args.humu_dim,
        curvature=args.curvature,
        min_coverage=args.min_coverage,
        strict=args.strict,
    )
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.report:
        Path(args.report).write_text(encoded + "\n", encoding="utf-8")
    else:
        print(encoded)
    if args.strict and report["status"] != "pass":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
