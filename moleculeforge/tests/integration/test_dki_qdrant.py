"""Integration tests for DKI Qdrant layer."""

from __future__ import annotations

import os
from uuid import uuid4

import numpy as np
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


@pytest.fixture
async def qdrant_client():
    pytest.importorskip("qdrant_client")
    host = os.environ.get("QDRANT_HOST", "127.0.0.1")
    http_port = int(os.environ.get("QDRANT_HTTP_PORT", "16333"))
    if not os.environ.get("QDRANT_HOST") and not os.environ.get("QDRANT_URL"):
        pytest.skip("QDRANT_HOST or QDRANT_URL is required for Qdrant integration tests")
    from mf_core.db.qdrant_client import QdrantCollectionClient

    client = QdrantCollectionClient(
        host=host,
        http_port=http_port,
        collection_name=f"test_molecules_humu_{uuid4().hex}",
        vector_size=129,
        vector_field="z_humu",
        primary_field="mol_id",
    )
    await client.connect()
    yield client
    await client.drop_collection()
    await client.disconnect()


async def test_insert_search_and_delete(qdrant_client) -> None:
    vectors = np.zeros((5, 129), dtype=np.float32)
    vectors[:, 0] = 1.0
    vectors[:, 1] = np.arange(5, dtype=np.float32)

    data = {
        "mol_id": [f"test-mol-{i}" for i in range(5)],
        "smiles": [f"C{'C' * i}O" for i in range(5)],
        "z_humu": vectors.tolist(),
        "oracle_scores": [{"L0": float(i) / 5} for i in range(5)],
        "source": ["integration"] * 5,
    }

    count = await qdrant_client.upsert(data)
    assert count == 5
    await qdrant_client.flush()

    hits = await qdrant_client.search(
        vector=vectors[0].tolist(),
        top_k=3,
        output_fields=["smiles", "oracle_scores"],
    )
    assert len(hits) <= 3
    assert all("id" in hit for hit in hits)
    assert all("distance" in hit for hit in hits)

    deleted = await qdrant_client.delete(["test-mol-0"])
    assert deleted >= 0
