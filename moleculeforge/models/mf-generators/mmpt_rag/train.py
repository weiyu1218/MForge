#!/usr/bin/env python3
"""Build an MMPT-RAG matched molecular pair transform index."""
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
    parser = argparse.ArgumentParser(description="MMPT-RAG MMP transform index builder")
    parser.add_argument("--data", required=True, help="JSON/JSONL/TSV file or directory")
    parser.add_argument("--output", required=True, help="Output MMPT index JSON artifact")
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
    pairs = [_normalize_pair(record) for record in _load_records(args.data)]
    transforms = [_pair_to_transform(pair) for pair in pairs]
    transforms = _deduplicate_transforms(transforms)
    if not transforms:
        raise ValueError("MMPT training data produced no matched-pair transforms")
    kd_metrics, kd_projection = _compute_kd_metrics(transforms, args)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "mmpt_mmp_index.v1",
        "pairs": pairs,
        "transforms": transforms,
    }
    if kd_projection is not None:
        payload["kd_projection"] = kd_projection
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    output_path.with_suffix(".manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "mmpt_training_manifest.v1",
                "pairs": len(pairs),
                "transforms": len(transforms),
                "artifact_path": str(output_path),
                **kd_metrics,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    LOGGER.info("Wrote %s MMPT transforms to %s", len(transforms), output_path)


def _add_project_paths() -> None:
    project_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(project_root.joinpath("libs", "mf-core", "src")))


def _load_records(path_value: str) -> list[dict]:
    path = Path(path_value)
    if not path.exists():
        raise FileNotFoundError(f"MMPT training data not found: {path}")
    records: list[dict] = []
    files = [path] if path.is_file() else sorted(path.rglob("*"))
    for file_path in files:
        if not file_path.is_file() or file_path.suffix not in {".json", ".jsonl", ".tsv"}:
            continue
        records.extend(_load_file(file_path))
    if not records:
        raise ValueError(f"MMPT training data contains no records: {path}")
    return records


def _load_file(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and isinstance(payload.get("pairs"), list):
            return list(payload["pairs"])
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            return [payload]
        raise ValueError(f"Unsupported MMPT JSON payload: {path}")
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


def _normalize_pair(record: object) -> dict:
    if not isinstance(record, dict):
        raise TypeError("MMPT pair record must be a JSON object")
    seed = record.get("seed_smiles") or record.get("source_smiles")
    product = record.get("product_smiles") or record.get("target_smiles") or record.get("product")
    if not seed or not product:
        raise ValueError("MMPT pair record requires seed_smiles and product_smiles")
    return {
        "id": str(record.get("id", f"{seed}>{product}")),
        "seed_smiles": _canonical_smiles(str(seed)),
        "product_smiles": _canonical_smiles(str(product)),
    }


def _canonical_smiles(smiles: str) -> str:
    try:
        from rdkit import Chem
    except ImportError as exc:
        raise ImportError("RDKit is required for MMPT training data validation") from exc
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid MMPT SMILES: {smiles}")
    return Chem.MolToSmiles(mol)


def _pair_to_transform(pair: dict) -> dict:
    seed = str(pair["seed_smiles"])
    product = str(pair["product_smiles"])
    prefix = 0
    max_prefix = min(len(seed), len(product))
    while prefix < max_prefix and seed[prefix] == product[prefix]:
        prefix += 1
    suffix = 0
    max_suffix = min(len(seed) - prefix, len(product) - prefix)
    while suffix < max_suffix and seed[-suffix - 1] == product[-suffix - 1]:
        suffix += 1
    seed_end = len(seed) - suffix if suffix else len(seed)
    product_end = len(product) - suffix if suffix else len(product)
    pattern = seed[prefix:seed_end]
    replacement = product[prefix:product_end]
    if not pattern or pattern == replacement:
        raise ValueError(f"Cannot derive transform from pair: {seed} -> {product}")
    return {
        "id": str(pair["id"]),
        "pattern": pattern,
        "replacement": replacement,
        "seed_smiles": seed,
        "product_smiles": product,
    }


def _deduplicate_transforms(transforms: list[dict]) -> list[dict]:
    seen: set[tuple[str, str]] = set()
    unique = []
    for transform in transforms:
        key = (str(transform["pattern"]), str(transform["replacement"]))
        if key in seen:
            continue
        seen.add(key)
        unique.append(transform)
    return unique


def _compute_kd_metrics(
    transforms: list[dict],
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
        raise ValueError("MMPT KD teacher embeddings must be a 2D matrix")
    if teacher_embeddings.shape[0] != len(transforms):
        raise ValueError(
            "MMPT KD teacher embedding row count must match MMPT transform records"
        )
    if args.kd_weight == 0.0:
        return metrics, None

    features = torch.tensor(
        [_kd_features_from_transform(transform) for transform in transforms],
        dtype=torch.float64,
        device=device,
    )
    if not torch.isfinite(features).all():
        raise ValueError("MMPT KD structural features must contain finite values")
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
        raise ValueError("MMPT KD loss must be finite")
    metrics["kd_loss"] = float(weighted_loss.detach().cpu().item())
    for transform, squared_distance in zip(
        transforms,
        squared_distances,
        strict=True,
    ):
        score = float(1.0 / (1.0 + squared_distance.detach().cpu().item()))
        if not math.isfinite(score):
            raise ValueError("MMPT KD alignment score must be finite")
        transform["kd_alignment_score"] = score
        transform["kd_weight"] = float(args.kd_weight)
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
        raise ValueError("MMPT KD projection must contain finite values")
    weights = coefficients[:-1].T.contiguous()
    bias = coefficients[-1]
    projection = {
        "schema_version": "linear_kd_projection.v1",
        "input_features": [
            "seed_smiles_length",
            "product_smiles_length",
            "pattern_length",
            "replacement_length",
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


def _kd_features_from_transform(transform: dict) -> list[float]:
    return [
        float(len(str(transform.get("seed_smiles") or ""))),
        float(len(str(transform.get("product_smiles") or ""))),
        float(len(str(transform.get("pattern") or ""))),
        float(len(str(transform.get("replacement") or ""))),
    ]


if __name__ == "__main__":
    main()
