#!/usr/bin/env python3
"""Build a CReM MMP mutation database artifact from real mutation records."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Iterable
from pathlib import Path

LOGGER = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="CReM-3D MMP database builder")
    parser.add_argument("--data", required=True, help="JSON/JSONL/TSV file or directory")
    parser.add_argument("--output", required=True, help="Output CReM MMP JSON artifact")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    _add_project_paths()
    records = [
        _normalize_record(index, record)
        for index, record in enumerate(_load_records(args.data))
    ]
    if not records:
        raise ValueError("CReM training data contains no mutation records")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "crem_mmp_database.v1",
        "mutations": records,
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    manifest_path = output_path.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "crem_training_manifest.v1",
                "records": len(records),
                "artifact_path": str(output_path),
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


if __name__ == "__main__":
    main()
