"""SureChEMBL patent grounding tool."""
from __future__ import annotations

import os
from typing import Any

from cig_compiler_svc.domain.tools.http_json import post_json

SURECHEMBL_CONTENT_SEARCH_URL = "https://www.surechembl.org/api/search/content"


async def query_surechembl_patents(
    query: str,
    patent_ids: list[str] | None = None,
    limit: int = 5,
) -> list[dict]:
    search_query = " OR ".join(patent_ids) if patent_ids else query
    data, timestamp = await post_json(
        os.environ.get("SURECHEMBL_CONTENT_SEARCH_URL", SURECHEMBL_CONTENT_SEARCH_URL),
        payload={},
        params={"query": search_query, "page": 0, "itemsPerPage": limit},
    )
    return _extract_patents(data, timestamp, limit=limit)


def _extract_patents(data: dict[str, Any], timestamp: str | None, limit: int) -> list[dict]:
    records = _iter_records(data)
    patents: dict[str, dict] = {}
    for record in records:
        patent_id = (
            record.get("patent_id")
            or record.get("patentId")
            or record.get("doc_id")
            or record.get("docId")
            or record.get("documentId")
            or record.get("document_id")
        )
        if not patent_id or patent_id in patents:
            continue
        patents[patent_id] = {
            "patent_id": patent_id,
            "title": record.get("title") or record.get("documentTitle"),
            "source_timestamp": timestamp,
        }
    return list(patents.values())[:limit]


def _iter_records(value: Any) -> list[dict]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if not isinstance(value, dict):
        return []

    records: list[dict] = []
    for key in ("results", "items", "data", "documents", "content", "payload"):
        child = value.get(key)
        if isinstance(child, list):
            records.extend(item for item in child if isinstance(item, dict))
        elif isinstance(child, dict):
            records.extend(_iter_records(child))
    if not records and any(
        key in value
        for key in (
            "patent_id",
            "patentId",
            "doc_id",
            "docId",
            "documentId",
            "document_id",
        )
    ):
        records.append(value)
    return records
