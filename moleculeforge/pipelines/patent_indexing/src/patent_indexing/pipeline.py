"""Patent indexing pipeline: SureChEMBL -> vector store + Dead Zone update.

Indexes patent molecules into the HUMU vector space for FTO dead zone detection.
"""
from __future__ import annotations

import gzip
import inspect
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any, TextIO


async def run(config: dict) -> dict:
    """Execute the full patent indexing pipeline."""
    cfg = _validate_config(config)
    surechembl_result = await index_surechembl_to_vector_store(cfg)
    uspto_result = await index_uspto_patents(cfg)
    deadzone_result = await update_dead_zone_potential(cfg)
    return {
        "pipeline": "patent_indexing",
        "status": "completed",
        "surechembl": surechembl_result,
        "uspto": uspto_result,
        "dead_zone": deadzone_result,
    }


async def index_surechembl_to_vector_store(cfg: dict) -> dict:
    """Index SureChEMBL patent molecules into the DKI vector database.

    Extracts chemical structures from SureChEMBL dump, encodes them via HUMU,
    and inserts into the patent vector index.
    """
    source_path = _required_existing_path(cfg, "surechembl_path")
    records = _load_patent_records(source_path)
    if not records:
        raise ValueError(f"No patent molecule records found in surechembl_path: {source_path}")
    records = await _prepare_records_for_indexing(records, cfg)

    collection_name = cfg.get("vector_collection", "patents_embedding")
    batch_size = int(cfg.get("batch_size", 1000))
    inserted, batches = await _insert_records(
        _required_index_client(cfg),
        collection_name,
        records,
        batch_size,
    )
    return {
        "source": "surechembl",
        "collection": collection_name,
        "molecules_indexed": inserted,
        "batches": batches,
        "status": "completed",
    }


async def index_uspto_patents(cfg: dict) -> dict:
    """Index USPTO patent grants and applications into the DKI vector store.

    Processes USPTO bulk data XML, extracts Markush structures,
    and indexes for FTO searching.
    """
    source_path = _required_existing_path(cfg, "uspto_path")
    records = _load_patent_records(source_path)
    if not records:
        raise ValueError(f"No patent molecule records found in uspto_path: {source_path}")
    records = await _prepare_records_for_indexing(records, cfg)

    collection_name = cfg.get("vector_collection", "patents_embedding")
    batch_size = int(cfg.get("batch_size", 1000))
    inserted, batches = await _insert_records(
        _required_index_client(cfg),
        collection_name,
        records,
        batch_size,
    )
    return {
        "source": "uspto",
        "grants_processed": len(records),
        "applications_processed": 0,
        "structures_extracted": inserted,
        "batches": batches,
        "status": "completed",
    }


async def update_dead_zone_potential(cfg: dict) -> dict:
    """Update the patent dead zone potential field in HUMU space.

    After indexing new patents, recalculate the dead zone repulsive
    potential that guides generators away from patented regions.
    """
    updater = cfg.get("dead_zone_updater")
    if updater is None:
        raise RuntimeError("dead_zone_updater is required to update patent dead zone")
    result = updater.refresh(cfg.get("dead_zone_config", {}))
    if inspect.isawaitable(result):
        result = await result
    return {
        "dead_zone_updated": True,
        "zones_affected": int(result.get("zones_affected", 0)),
        "status": "completed",
    }


async def search_patent_similarity(smiles: str, cfg: dict | None = None) -> dict:
    """Search a molecule against the patent index for FTO assessment."""
    cfg = cfg or {}
    client = _required_search_client(cfg)
    collection_name = cfg.get("vector_collection", "patents_embedding")
    limit = int(cfg.get("top_k", 1))
    query_vector = await _encode_smiles(smiles, cfg)
    result = _call_search(client, collection_name, query_vector, limit)
    if inspect.isawaitable(result):
        result = await result
    hits = list(result or [])
    if not hits:
        return {
            "smiles": smiles,
            "nearest_patent_distance": None,
            "nearest_patent_id": None,
            "claim_evidence": None,
            "fto_risk": "no_index_hit",
        }
    nearest = hits[0]
    entity = nearest.get("entity", {}) if isinstance(nearest, dict) else {}
    patent_id = nearest.get("patent_id") or entity.get("patent_id") or nearest.get("id")
    distance = nearest.get("distance")
    return {
        "smiles": smiles,
        "nearest_patent_distance": distance,
        "nearest_patent_id": patent_id,
        "claim_evidence": nearest.get("claim_evidence") or entity.get("claim_evidence"),
        "source": nearest.get("source") or entity.get("source"),
        "fto_risk": nearest.get("fto_risk") or "requires_review",
    }


def _validate_config(config: dict) -> dict:
    """Validate and set defaults for patent indexing configuration."""
    defaults = {
        "vector_collection": "patents_embedding",
        "batch_size": 1000,
        "dead_zone_config": {},
    }
    return {**defaults, **config}


def _required_existing_path(cfg: dict, key: str) -> Path:
    value = cfg.get(key)
    if not value:
        raise FileNotFoundError(f"{key} is required and must point to existing data")
    path = Path(value)
    if not path.exists():
        raise FileNotFoundError(f"{key} does not exist: {path}")
    return path


def _required_index_client(cfg: dict) -> object:
    client = cfg.get("vector_client")
    if client is None:
        raise RuntimeError("vector_client is required to index patent molecules")
    if not hasattr(client, "insert") and not hasattr(client, "upsert"):
        raise TypeError(
            "vector_client must expose insert(collection, records) or upsert(data)"
        )
    return client


def _required_search_client(cfg: dict) -> object:
    client = cfg.get("vector_client")
    if client is None:
        raise RuntimeError("vector_client is required to search patent molecules")
    if not hasattr(client, "search"):
        raise TypeError("vector_client must expose a search method")
    return client


def _iter_record_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
        return
    for item in sorted(path.rglob("*")):
        if item.is_file() and _is_supported_record_file(item):
            yield item


def _is_supported_record_file(path: Path) -> bool:
    suffixes = path.suffixes
    if suffixes and suffixes[-1] == ".gz":
        suffixes = suffixes[:-1]
    return bool(suffixes) and suffixes[-1] in {".smi", ".smiles", ".txt", ".csv", ".tsv"}


def _open_text(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _load_patent_records(path: Path) -> list[dict]:
    records: list[dict] = []
    for file_path in _iter_record_files(path):
        with _open_text(file_path) as handle:
            header: list[str] | None = None
            for line_number, line in enumerate(handle, start=1):
                parts = _split_record_line(line)
                if parts is None:
                    continue
                if header is None and _looks_like_header(parts):
                    header = [_normalise_header(part) for part in parts]
                    continue
                parsed = _parse_patent_parts(parts, header, file_path, line_number)
                if parsed is not None:
                    records.append(parsed)
    return records


def _split_record_line(line: str) -> list[str] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    separator = "\t" if "\t" in stripped else "," if "," in stripped else None
    return [part.strip() for part in stripped.split(separator)] if separator else stripped.split()


def _looks_like_header(parts: list[str]) -> bool:
    return bool(parts) and _normalise_header(parts[0]) in {"smiles", "canonical_smiles"}


def _normalise_header(value: str) -> str:
    return value.strip().lower().replace(" ", "_")


def _parse_patent_parts(
    parts: list[str],
    header: list[str] | None,
    file_path: Path,
    line_number: int,
) -> dict | None:
    if not parts:
        return None
    row = _parts_to_row(parts, header)
    smiles = row.get("smiles") or row.get("canonical_smiles")
    if not smiles:
        return None
    patent_id = (
        row.get("patent_id")
        or row.get("publication_number")
        or row.get("patent")
        or file_path.stem
    )
    claim_evidence = (
        row.get("claim_evidence")
        or row.get("claim")
        or row.get("claim_text")
        or row.get("evidence")
    )
    source = row.get("source") or file_path.parent.name or file_path.stem
    return {
        "smiles": smiles,
        "patent_id": patent_id,
        "claim_evidence": claim_evidence,
        "source": source,
        "source_file": str(file_path),
        "source_line": line_number,
    }


def _parts_to_row(parts: list[str], header: list[str] | None) -> dict[str, str]:
    if header:
        return {
            header[index]: parts[index]
            for index in range(min(len(header), len(parts)))
            if parts[index] != ""
        }
    row = {"smiles": parts[0]}
    if len(parts) > 1:
        row["patent_id"] = parts[1]
    if len(parts) > 2:
        row["claim_evidence"] = parts[2]
    if len(parts) > 3:
        row["source"] = parts[3]
    return row


async def _prepare_records_for_indexing(records: list[dict], cfg: dict) -> list[dict]:
    prepared: list[dict] = []
    for record in records:
        item = dict(record)
        if "z_humu" not in item:
            item["z_humu"] = await _encode_smiles(str(item["smiles"]), cfg)
        prepared.append(item)
    return prepared


async def _encode_smiles(smiles: str, cfg: dict) -> list[float]:
    encoder = cfg.get("humu_encoder")
    if encoder is None:
        raise RuntimeError("humu_encoder is required to produce z_humu patent vectors")
    if hasattr(encoder, "encode_smiles"):
        result = encoder.encode_smiles(smiles)
    elif hasattr(encoder, "encode"):
        result = encoder.encode(smiles)
    elif callable(encoder):
        result = encoder(smiles)
    else:
        raise TypeError("humu_encoder must expose encode_smiles, encode, or be callable")
    if inspect.isawaitable(result):
        result = await result
    return [float(value) for value in result]


def _call_search(client: object, collection_name: str, vector: list[float], limit: int) -> Any:
    search = getattr(client, "search")
    parameters = list(inspect.signature(search).parameters)
    if parameters and parameters[0] in {"collection", "collection_name"}:
        return search(collection_name, vector, limit=limit)
    return search(vector, top_k=limit, output_fields=["patent_id", "claim_evidence", "source"])


async def _insert_records(
    client: object,
    collection_name: str,
    records: list[dict],
    batch_size: int,
) -> tuple[int, int]:
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")
    inserted = 0
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        result = _call_insert(client, collection_name, batch)
        if inspect.isawaitable(result):
            result = await result
        inserted += len(batch) if result is None else int(result)
    return inserted, math.ceil(len(records) / batch_size)


def _call_insert(client: object, collection_name: str, batch: list[dict]) -> Any:
    if hasattr(client, "insert"):
        return client.insert(collection_name, batch)
    return client.upsert(_records_to_columns(batch))


def _records_to_columns(records: list[dict]) -> dict[str, list]:
    keys: list[str] = []
    for record in records:
        for key in record:
            if key not in keys:
                keys.append(key)
    return {key: [record.get(key) for record in records] for key in keys}
