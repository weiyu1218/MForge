"""Stage 1b: Knowledge grounding via external chemistry sources."""
from __future__ import annotations

from typing import Literal

from cig_compiler_svc.domain.tools import chembl_tool, pdb_tool, uniprot_tool

GroundingSource = Literal["uniprot", "pdb", "chembl"]
ALL_GROUNDING_SOURCES: tuple[GroundingSource, ...] = (
    "uniprot",
    "pdb",
    "chembl",
)


async def ground_knowledge(
    extracted: dict,
    enable_pdb: bool = True,
    sources: tuple[GroundingSource, ...] | None = None,
) -> dict:
    enriched = dict(extracted)

    targets = extracted.get("targets", [])
    selected_sources = sources or (("uniprot", "pdb") if enable_pdb else ("uniprot",))
    uniprot_ids: list[str] = []
    pdb_ids: list[str] = []
    chembl_target_ids: list[str] = []
    grounding_evidence: list[dict] = []

    for target in targets:
        query = target["name"] if isinstance(target, dict) else target
        if "uniprot" in selected_sources:
            results = await uniprot_tool.query_uniprot_entry(query)
            for entry in results:
                accession = entry.get("accession", "")
                if accession:
                    uniprot_ids.append(accession)
                entry_pdb_ids = list(entry.get("pdb_ids", []))
                pdb_ids.extend(entry_pdb_ids)
                grounding_evidence.append(_uniprot_evidence(query, entry, entry_pdb_ids))

        unique_uniprot_ids = list(dict.fromkeys(uniprot_ids))
        if "pdb" in selected_sources:
            for entry in await pdb_tool.query_pdb_entries(query, uniprot_ids=unique_uniprot_ids):
                pdb_id = entry.get("pdb_id", "")
                if pdb_id:
                    pdb_ids.append(pdb_id)
                grounding_evidence.append(_pdb_evidence(query, entry))

        if "chembl" in selected_sources:
            chembl_results = await chembl_tool.query_chembl_targets(
                query,
                uniprot_ids=unique_uniprot_ids,
            )
            for entry in chembl_results:
                target_id = entry.get("target_chembl_id", "")
                if target_id:
                    chembl_target_ids.append(target_id)
                grounding_evidence.append(_chembl_evidence(query, entry))

    enriched["_grounded_uniprot_ids"] = list(dict.fromkeys(uniprot_ids))
    enriched["_grounded_pdb_ids"] = list(dict.fromkeys(pdb_ids))
    enriched["_grounded_chembl_target_ids"] = list(dict.fromkeys(chembl_target_ids))
    enriched["_grounding_evidence"] = grounding_evidence
    return enriched


def _require_source_timestamp(source: str, entry: dict) -> str:
    timestamp = entry.get("source_timestamp")
    if not timestamp:
        raise RuntimeError(f"{source} grounding evidence missing source_timestamp")
    return timestamp


def _uniprot_evidence(query: str, entry: dict, pdb_ids: list[str]) -> dict:
    return {
        "source": "uniprot",
        "query": query,
        "accession": entry.get("accession", ""),
        "pdb_ids": pdb_ids,
        "confidence": entry.get("confidence"),
        "source_timestamp": _require_source_timestamp("uniprot", entry),
    }


def _pdb_evidence(query: str, entry: dict) -> dict:
    return {
        "source": "pdb",
        "query": query,
        "pdb_id": entry.get("pdb_id", ""),
        "title": entry.get("title"),
        "score": entry.get("score"),
        "source_timestamp": _require_source_timestamp("pdb", entry),
    }


def _chembl_evidence(query: str, entry: dict) -> dict:
    return {
        "source": "chembl",
        "query": query,
        "target_chembl_id": entry.get("target_chembl_id", ""),
        "pref_name": entry.get("pref_name"),
        "organism": entry.get("organism"),
        "source_timestamp": _require_source_timestamp("chembl", entry),
    }
