"""Integration tests for patent indexing against DKI Qdrant."""

from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


class DeterministicPatentEncoder:
    def encode_smiles(self, smiles: str) -> list[float]:
        vector = [0.0] * 768
        vector[0] = float(len(smiles))
        vector[1] = 1.0
        return vector


async def test_patent_indexing_round_trips_qdrant(tmp_path: Path) -> None:
    if not (os.environ.get("QDRANT_URL") or os.environ.get("QDRANT_HOST")):
        pytest.skip("QDRANT_URL or QDRANT_HOST is required for patent Qdrant integration tests")

    from mf_core.db.qdrant_client import QdrantCollectionClient
    from patent_indexing.pipeline import (
        index_surechembl_to_vector_store,
        search_patent_similarity,
    )

    patent_id = f"US-INTEGRATION-{uuid4().hex}"
    source_dir = tmp_path / "surechembl"
    source_dir.mkdir()
    (source_dir / "patents.tsv").write_text(
        "smiles\tpatent_id\tclaim_evidence\tsource\n"
        f"CCO\t{patent_id}\tclaim 1 covers ethanol analogs\tsurechembl\n",
        encoding="utf-8",
    )

    client = QdrantCollectionClient(
        host=os.environ.get("QDRANT_HOST", "127.0.0.1"),
        http_port=int(os.environ.get("QDRANT_HTTP_PORT", "16333")),
        url=os.environ.get("QDRANT_URL"),
        collection_name="patents_embedding",
        vector_size=768,
        vector_field="z_humu",
        primary_field="patent_id",
    )
    await client.connect()
    try:
        result = await index_surechembl_to_vector_store(
            {
                "surechembl_path": str(source_dir),
                "vector_client": client,
                "humu_encoder": DeterministicPatentEncoder(),
                "batch_size": 1,
            }
        )
        hit = await search_patent_similarity(
            "CCO",
            {
                "vector_client": client,
                "humu_encoder": DeterministicPatentEncoder(),
                "top_k": 1,
            },
        )

        assert result["molecules_indexed"] == 1
        assert hit["nearest_patent_id"] == patent_id
        assert hit["claim_evidence"] == "claim 1 covers ethanol analogs"
        assert hit["source"] == "surechembl"
    finally:
        await client.delete([patent_id])
        await client.disconnect()
