#!/usr/bin/env python3
"""Build a CReM MMP mutation database artifact from real mutation records."""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from collections.abc import Iterable
from pathlib import Path

LOGGER = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="CReM-3D MMP database builder")
    parser.add_argument("--data", required=True, help="JSON/JSONL/TSV file or directory")
    parser.add_argument("--output", required=True, help="Output CReM MMP JSON artifact")
    parser.add_argument(
        "--kd-teacher-embeddings",
        default="",
        help="JSON artifact containing teacher embedding targets for KD loss",
    )
    parser.add_argument(
        "--kd-weight",
        type=float,
        default=0.0,
        help="Weight for teacher embedding distillation loss",
    )
    parser.add_argument(
        "--kd-generator-idx",
        type=int,
        default=0,
        help="Generator index used for KD teacher target lookup",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if not math.isfinite(args.kd_weight):
        raise ValueError("--kd-weight must be finite")
    if args.kd_weight < 0.0:
        raise ValueError("--kd-weight must be >= 0")
    if args.kd_generator_idx < 0:
        raise ValueError("--kd-generator-idx must be non-negative")
    if args.kd_weight > 0.0 and not args.kd_teacher_embeddings:
        raise ValueError("--kd-teacher-embeddings is required when --kd-weight > 0")
    if args.kd_teacher_embeddings and not Path(args.kd_teacher_embeddings).is_file():
        raise FileNotFoundError(
            f"KD teacher embedding artifact not found: {args.kd_teacher_embeddings}"
        )

    _add_project_paths()
    records = [
        _normalize_record(index, record)
        for index, record in enumerate(_load_records(args.data))
    ]
    if not records:
        raise ValueError("CReM training data contains no mutation records")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    kd_metrics, kd_projection = _compute_kd_metrics(records, args)
    payload = {
        "schema_version": "crem_mmp_database.v1",
        "mutations": records,
    }
    if kd_projection is not None:
        payload["kd_projection"] = kd_projection
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    manifest_path = output_path.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "crem_training_manifest.v1",
                "records": len(records),
                "artifact_path": str(output_path),
                **kd_metrics,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    LOGGER.info("Wrote %s CReM mutations to %s", len(records), output_path)


def _add_project_paths() -> None:
    project_root = Path(__file__).resolve().parents[3]
    for rel_path in (
        ("libs", "mf-core", "src"),
        ("models", "mf-generators", "crem_3d", "src"),
    ):
        sys.path.insert(0, str(project_root.joinpath(*rel_path)))


def _load_records(path_value: str) -> list[dict]:
    path = Path(path_value)
    if not path.exists():
        raise FileNotFoundError(f"CReM training data not found: {path}")
    records: list[dict] = []
    files = [path] if path.is_file() else sorted(path.rglob("*"))
    for file_path in files:
        if not file_path.is_file() or file_path.suffix not in {".json", ".jsonl", ".tsv"}:
            continue
        records.extend(_load_file(file_path))
    return records


def _load_file(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and isinstance(payload.get("mutations"), list):
            return list(payload["mutations"])
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            return [payload]
        raise ValueError(f"Unsupported CReM JSON payload: {path}")
    return list(_load_tsv(path))


def _load_tsv(path: Path) -> Iterable[dict]:
    with path.open(encoding="utf-8") as handle:
        header = None
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if header is None:
                header = [part.strip() for part in parts]
                continue
            if not any(parts):
                continue
            yield {key: value for key, value in zip(header, parts, strict=False)}


def _normalize_record(index: int, record: object) -> dict:
    if not isinstance(record, dict):
        raise TypeError("CReM mutation record must be a JSON object")
    product = str(record.get("product", "") or "")
    fragment_smiles = str(record.get("fragment_smiles", "") or "")
    if not product and not fragment_smiles:
        raise ValueError("CReM mutation record requires product or fragment_smiles")
    normalized = {
        "id": str(record.get("id", index)),
        "seed_smiles": _canonical_smiles(str(record.get("seed_smiles", "") or ""))
        if record.get("seed_smiles")
        else "",
        "fragment_smiles": fragment_smiles,
        "attachment_index": _optional_int(record.get("attachment_index")),
        "product": _canonical_smiles(product) if product else "",
    }
    return normalized


def _canonical_smiles(smiles: str) -> str:
    try:
        from rdkit import Chem
    except ImportError as exc:
        raise ImportError("RDKit is required for CReM training data validation") from exc
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid CReM SMILES: {smiles}")
    return Chem.MolToSmiles(mol)


def _optional_int(value: object) -> int | None:
    if value in {None, ""}:
        return None
    return int(value)


def _compute_kd_metrics(
    records: list[dict],
    args: argparse.Namespace,
) -> tuple[dict, dict | None]:
    metrics = {
        "kd_teacher_embeddings": str(args.kd_teacher_embeddings or ""),
        "kd_weight": float(args.kd_weight),
        "kd_generator_idx": int(args.kd_generator_idx),
        "kd_loss": 0.0,
    }
    if not args.kd_teacher_embeddings:
        return metrics, None

    import torch
    from mf_core.routing.cross_paradigm_kd import load_teacher_embeddings_artifact

    device = torch.device("cpu")
    teacher_embeddings = load_teacher_embeddings_artifact(
        args.kd_teacher_embeddings,
        device=device,
    ).to(dtype=torch.float64)
    if teacher_embeddings.ndim != 2:
        raise ValueError("CReM KD teacher embeddings must be a 2D matrix")
    if teacher_embeddings.shape[0] != len(records):
        raise ValueError(
            "CReM KD teacher embedding row count must match CReM mutation records"
        )
    if args.kd_weight == 0.0:
        return metrics, None

    features = torch.tensor(
        [_kd_features_from_record(record) for record in records],
        dtype=torch.float64,
        device=device,
    )
    if not torch.isfinite(features).all():
        raise ValueError("CReM KD structural features must contain finite values")
    projection, predictions = _fit_linear_kd_projection(
        features,
        teacher_embeddings,
        kd_weight=float(args.kd_weight),
        generator_idx=int(args.kd_generator_idx),
    )
    squared_distances = torch.mean(
        (predictions - teacher_embeddings) ** 2,
        dim=1,
    )
    projection_parameters = torch.tensor(
        [
            [*row, bias]
            for row, bias in zip(
                projection["weights"],
                projection["bias"],
                strict=True,
            )
        ],
        dtype=torch.float64,
    )
    regularization_loss = float(projection["regularization"]) * torch.mean(
        projection_parameters**2
    )
    weighted_loss = float(args.kd_weight) * (
        torch.mean(squared_distances) + regularization_loss
    )
    if not torch.isfinite(weighted_loss):
        raise ValueError("CReM KD loss must be finite")
    metrics["kd_loss"] = float(weighted_loss.detach().cpu().item())
    for record, squared_distance in zip(records, squared_distances, strict=True):
        score = float(1.0 / (1.0 + squared_distance.detach().cpu().item()))
        if not math.isfinite(score):
            raise ValueError("CReM KD alignment score must be finite")
        record["kd_alignment_score"] = score
        record["kd_weight"] = float(args.kd_weight)
    return metrics, projection


def _fit_linear_kd_projection(
    features,
    teacher_embeddings,
    *,
    kd_weight: float,
    generator_idx: int,
) -> tuple[dict, object]:
    import torch

    regularization = 1.0
    feature_mean = features.mean(dim=0)
    feature_scale = features.std(dim=0, unbiased=False)
    feature_scale = torch.where(
        feature_scale > torch.finfo(features.dtype).eps,
        feature_scale,
        torch.ones_like(feature_scale),
    )
    normalized_features = (features - feature_mean) / feature_scale
    design = torch.cat(
        [
            normalized_features,
            torch.ones(
                (normalized_features.shape[0], 1),
                dtype=features.dtype,
                device=features.device,
            ),
        ],
        dim=1,
    )
    identity = torch.eye(
        design.shape[1],
        dtype=features.dtype,
        device=features.device,
    )
    ridge_strength = regularization * design.shape[0] / design.shape[1]
    coefficients = torch.linalg.solve(
        design.T @ design + ridge_strength * identity,
        design.T @ teacher_embeddings,
    )
    predictions = design @ coefficients
    if not torch.isfinite(coefficients).all() or not torch.isfinite(predictions).all():
        raise ValueError("CReM KD projection must contain finite values")
    weights = coefficients[:-1].T.contiguous()
    bias = coefficients[-1]
    projection = {
        "schema_version": "linear_kd_projection.v1",
        "input_features": [
            "seed_smiles_length",
            "fragment_smiles_length",
            "attachment_index",
            "product_smiles_length",
        ],
        "input_dim": int(features.shape[1]),
        "teacher_dim": int(teacher_embeddings.shape[1]),
        "feature_mean": feature_mean.detach().cpu().tolist(),
        "feature_scale": feature_scale.detach().cpu().tolist(),
        "weights": weights.detach().cpu().tolist(),
        "bias": bias.detach().cpu().tolist(),
        "regularization": regularization,
        "kd_weight": kd_weight,
        "generator_idx": generator_idx,
    }
    return projection, predictions


def _kd_features_from_record(record: dict) -> list[float]:
    return [
        float(len(str(record.get("seed_smiles") or ""))),
        float(len(str(record.get("fragment_smiles") or ""))),
        float(record.get("attachment_index") or 0),
        float(len(str(record.get("product") or ""))),
    ]


if __name__ == "__main__":
    main()
