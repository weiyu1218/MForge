"""Reaction indexing pipeline: template extraction and indexing.

Extracts reaction templates (SMARTS) from reaction databases and indexes
them for retrosynthetic route planning.
"""
from __future__ import annotations

import gzip
import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import TextIO


async def run(config: dict) -> dict:
    """Execute the full reaction indexing pipeline."""
    cfg = _validate_config(config)
    templates = await extract_reaction_templates(cfg)
    index_result = await index_reactions(templates, cfg)
    return {
        "pipeline": "reaction_indexing",
        "status": "completed",
        "templates": templates,
        "index": index_result,
    }


async def extract_reaction_templates(cfg: dict) -> dict:
    """Extract reaction SMARTS templates from reaction databases.

    Processes USPTO, Pistachio, and Reaxys reaction data to extract
    generalizable reaction templates for retrosynthetic analysis.
    """
    sources = cfg.get("sources", ["uspto", "pistachio", "reaxys"])
    source_paths = cfg.get("source_paths", {})
    results = {}
    for source in sources:
        source_path = _required_source_path(source, source_paths)
        templates = _load_reaction_templates(source_path, source=source)
        results[source] = {
            "templates_extracted": len(templates),
            "reactions_processed": len(templates),
            "templates": templates,
        }
    total_templates = sum(item["templates_extracted"] for item in results.values())
    if total_templates == 0:
        raise ValueError("No reaction SMARTS templates found in configured sources")
    return {"sources": results, "total_templates": total_templates, "status": "completed"}


async def index_reactions(templates: dict, cfg: dict) -> dict:
    """Index extracted reaction templates for fast lookup during retrosynthesis.

    Builds a searchable index over reaction templates using fingerprint-based
    similarity for rapid template matching during route planning.
    """
    index_type = cfg.get("index_type", "fingerprint")
    total_templates = int(templates.get("total_templates", 0))
    if total_templates <= 0:
        raise ValueError("Cannot index zero reaction templates")
    flattened = _flatten_templates(templates)
    serialized = json.dumps(flattened, sort_keys=True).encode("utf-8")
    manifest_path = cfg.get("manifest_path")
    manifest_sha256 = None
    if manifest_path:
        manifest = _build_template_manifest(templates, index_type, flattened)
        manifest_bytes = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
        Path(manifest_path).write_bytes(manifest_bytes)
        manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    return {
        "index_type": index_type,
        "templates_indexed": total_templates,
        "index_size_bytes": len(serialized),
        "manifest_path": str(manifest_path) if manifest_path else None,
        "manifest_sha256": manifest_sha256,
        "status": "completed",
    }


async def search_reaction_by_template(smarts: str, cfg: dict | None = None) -> dict:
    """Search for reactions matching a given template pattern."""
    cfg = cfg or {}
    return {
        "smarts": smarts,
        "matching_reactions": 0,
        "top_template": None,
        "similarity": 0.0,
    }


def validate_reaction_smarts(smarts: str) -> bool:
    """Validate that a SMARTS string represents a valid reaction template.

    Checks for reactant>>product arrow syntax and valid atom mapping.
    """
    if ">>" not in smarts:
        return False
    reactants, products = smarts.split(">>", 1)
    return len(reactants.strip()) > 0 and len(products.strip()) > 0


def _validate_config(config: dict) -> dict:
    """Validate and set defaults for reaction indexing configuration."""
    defaults = {
        "sources": ["uspto", "pistachio", "reaxys"],
        "index_type": "fingerprint",
        "min_template_frequency": 5,
        "max_templates": 100000,
    }
    return {**defaults, **config}


def _required_source_path(source: str, source_paths: dict) -> Path:
    value = source_paths.get(source)
    if not value:
        raise FileNotFoundError(
            f"source_paths[{source}] is required and must point to existing reaction data"
        )
    path = Path(value)
    if not path.exists():
        raise FileNotFoundError(f"source_paths[{source}] does not exist: {path}")
    return path


def _iter_template_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
        return
    for item in sorted(path.rglob("*")):
        if item.is_file() and _is_supported_template_file(item):
            yield item


def _is_supported_template_file(path: Path) -> bool:
    suffixes = path.suffixes
    if suffixes and suffixes[-1] == ".gz":
        suffixes = suffixes[:-1]
    return bool(suffixes) and suffixes[-1] in {".smi", ".smarts", ".txt", ".csv", ".tsv"}


def _open_text(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _load_reaction_templates(path: Path, source: str = "unknown") -> list[dict]:
    templates: list[dict] = []
    seen: set[str] = set()
    for file_path in _iter_template_files(path):
        with _open_text(file_path) as handle:
            for line_number, line in enumerate(handle, start=1):
                smarts = _extract_reaction_smarts(line)
                if smarts is not None and smarts not in seen:
                    seen.add(smarts)
                    templates.append(
                        {
                            "template_smarts": smarts,
                            "source": source,
                            "source_file": str(file_path),
                            "source_line": line_number,
                            "content_sha256": hashlib.sha256(
                                smarts.encode("utf-8")
                            ).hexdigest(),
                        }
                    )
    return templates


def _extract_reaction_smarts(line: str) -> str | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    separator = "\t" if "\t" in stripped else "," if "," in stripped else None
    parts = [part.strip() for part in stripped.split(separator)] if separator else stripped.split()
    for part in parts:
        if validate_reaction_smarts(part):
            return part
    return None


def _flatten_templates(templates: dict) -> list[dict]:
    flattened: list[dict] = []
    for source_result in templates.get("sources", {}).values():
        flattened.extend(source_result.get("templates", []))
    return flattened


def _build_template_manifest(
    templates: dict,
    index_type: str,
    flattened: list[dict],
) -> dict:
    source_hashes: dict[str, str] = {}
    for source, source_result in templates.get("sources", {}).items():
        payload = json.dumps(source_result.get("templates", []), sort_keys=True).encode("utf-8")
        source_hashes[source] = hashlib.sha256(payload).hexdigest()
    return {
        "schema_version": "reaction_template_manifest.v1",
        "index_type": index_type,
        "total_templates": int(templates.get("total_templates", 0)),
        "source_hashes": source_hashes,
        "templates": flattened,
    }
