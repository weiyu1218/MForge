"""ChEMBL target grounding tool."""
from __future__ import annotations

import os

from cig_compiler_svc.domain.tools.http_json import get_json

CHEMBL_TARGET_SEARCH_URL = "https://www.ebi.ac.uk/chembl/api/data/target/search.json"
CHEMBL_TARGET_URL = "https://www.ebi.ac.uk/chembl/api/data/target.json"


async def query_chembl_targets(
    query: str,
    uniprot_ids: list[str] | None = None,
    limit: int = 5,
) -> list[dict]:
    records: list[dict] = []
    if uniprot_ids:
        for accession in uniprot_ids:
            data, timestamp = await get_json(
                os.environ.get("CHEMBL_TARGET_URL", CHEMBL_TARGET_URL),
                params={
                    "target_components__accession": accession,
                    "limit": limit,
                    "format": "json",
                },
            )
            records.extend(_extract_targets(data, timestamp))
    else:
        data, timestamp = await get_json(
            os.environ.get("CHEMBL_TARGET_SEARCH_URL", CHEMBL_TARGET_SEARCH_URL),
            params={"q": query, "limit": limit},
        )
        records.extend(_extract_targets(data, timestamp))

    deduped: dict[str, dict] = {}
    for record in records:
        target_id = record.get("target_chembl_id")
        if target_id and target_id not in deduped:
            deduped[target_id] = record
    return list(deduped.values())[:limit]


def _extract_targets(data: dict, timestamp: str | None) -> list[dict]:
    targets = data.get("targets", [])
    if not isinstance(targets, list):
        return []

    records = []
    for target in targets:
        target_id = target.get("target_chembl_id")
        if target_id:
            records.append(
                {
                    "target_chembl_id": target_id,
                    "pref_name": target.get("pref_name"),
                    "organism": target.get("organism"),
                    "source_timestamp": timestamp,
                }
            )
    return records
