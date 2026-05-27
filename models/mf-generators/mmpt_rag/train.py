#!/usr/bin/env python3
"""Build an MMPT-RAG matched molecular pair transform index."""
from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Iterable
from pathlib import Path

LOGGER = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="MMPT-RAG MMP transform index builder")
    parser.add_argument("--data", required=True, help="JSON/JSONL/TSV file or directory")
    parser.add_argument("--output", required=True, help="Output MMPT index JSON artifact")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    pairs = [_normalize_pair(record) for record in _load_records(args.data)]
    transforms = [_pair_to_transform(pair) for pair in pairs]
    transforms = _deduplicate_transforms(transforms)
    if not transforms:
        raise ValueError("MMPT training data produced no matched-pair transforms")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "mmpt_mmp_index.v1",
        "pairs": pairs,
        "transforms": transforms,
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    output_path.with_suffix(".manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "mmpt_training_manifest.v1",
                "pairs": len(pairs),
                "transforms": len(transforms),
                "artifact_path": str(output_path),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    LOGGER.info("Wrote %s MMPT transforms to %s", len(transforms), output_path)


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


if __name__ == "__main__":
    main()
