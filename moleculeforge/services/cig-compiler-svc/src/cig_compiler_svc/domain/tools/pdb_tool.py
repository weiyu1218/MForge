"""RCSB PDB grounding tool."""
from __future__ import annotations

import os
from typing import Any

from cig_compiler_svc.domain.tools.http_json import post_json

RCSB_SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"


async def query_pdb_entries(
    query: str,
    uniprot_ids: list[str] | None = None,
    limit: int = 5,
) -> list[dict]:
    search_terms = [query, *(uniprot_ids or [])]
    payload: dict[str, Any] = {
        "query": {
            "type": "terminal",
            "service": "full_text",
            "parameters": {"value": " ".join(term for term in search_terms if term)},
        },
        "return_type": "entry",
        "request_options": {"paginate": {"start": 0, "rows": limit}},
    }
    data, timestamp = await post_json(
        os.environ.get("RCSB_SEARCH_URL", RCSB_SEARCH_URL),
        payload=payload,
    )

    entries = []
    for result in data.get("result_set", []):
        pdb_id = result.get("identifier")
        if pdb_id:
            entries.append(
                {
                    "pdb_id": pdb_id,
                    "score": result.get("score"),
                    "source_timestamp": timestamp,
                }
            )
    return entries
