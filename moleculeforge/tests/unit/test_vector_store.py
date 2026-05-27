"""Unit tests for QdrantCollectionClient."""

from __future__ import annotations

from types import SimpleNamespace

import pytest


class _FakePoint:
    def __init__(self, point_id: str, score: float, payload: dict):
        self.id = point_id
        self.score = score
        self.payload = payload


class _FakeQdrantClient:
    def __init__(self) -> None:
        self.created: list[dict] = []
        self.upserts: list[dict] = []
        self.deletes: list[dict] = []
        self.closed = False
        self.collections: set[str] = set()

    def get_collections(self):
        return SimpleNamespace(
            collections=[SimpleNamespace(name=name) for name in sorted(self.collections)]
        )

    def create_collection(self, **kwargs):
        self.created.append(kwargs)
        self.collections.add(kwargs["collection_name"])

    def upsert(self, **kwargs):
        self.upserts.append(kwargs)

    def query_points(self, **kwargs):
        return SimpleNamespace(
            points=[
                _FakePoint(
                    point_id="generated-id",
                    score=0.9,
                    payload={
                        "_id_str": "mol-001",
                        "smiles": "CCO",
                        "oracle_scores": {"L0": 0.8},
                    },
                )
            ]
        )

    def count(self, collection_name: str, exact: bool):
        return SimpleNamespace(count=3)

    def delete(self, **kwargs):
        self.deletes.append(kwargs)
        return SimpleNamespace(operation_id=1)

    def close(self):
        self.closed = True


@pytest.mark.unit
def test_qdrant_client_init() -> None:
    from mf_core.db.qdrant_client import QdrantCollectionClient

    client = QdrantCollectionClient(
        host="localhost",
        http_port=16333,
        collection_name="molecules_humu",
    )

    assert client.host == "localhost"
    assert client.http_port == 16333
    assert client.collection_name == "molecules_humu"
    assert client._client is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_connect_creates_qdrant_collection() -> None:
    from mf_core.db.qdrant_client import QdrantCollectionClient

    fake = _FakeQdrantClient()
    client = QdrantCollectionClient(
        collection_name="test_col",
        vector_size=129,
        qdrant_client=fake,
    )

    await client.connect()

    assert fake.created[0]["collection_name"] == "test_col"
    assert client._client is fake


@pytest.mark.unit
@pytest.mark.asyncio
async def test_qdrant_upsert_preserves_original_ids_in_payload() -> None:
    from mf_core.db.qdrant_client import QdrantCollectionClient

    fake = _FakeQdrantClient()
    fake.collections.add("test_col")
    client = QdrantCollectionClient(collection_name="test_col", qdrant_client=fake)
    await client.connect()

    count = await client.upsert(
        {
            "id": ["mol-001"],
            "vector": [[0.1, 0.2, 0.3]],
            "smiles": ["CCO"],
        }
    )

    point = fake.upserts[0]["points"][0]
    assert count == 1
    assert point.payload["_id_str"] == "mol-001"
    assert point.payload["smiles"] == "CCO"
    assert point.vector == [0.1, 0.2, 0.3]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_qdrant_search_response_handling() -> None:
    from mf_core.db.qdrant_client import QdrantCollectionClient

    fake = _FakeQdrantClient()
    fake.collections.add("test_col")
    client = QdrantCollectionClient(collection_name="test_col", qdrant_client=fake)
    await client.connect()

    hits = await client.search(
        vector=[0.1] * 129,
        top_k=1,
        output_fields=["smiles", "oracle_scores"],
    )

    assert hits == [
        {
            "id": "mol-001",
            "distance": 0.9,
            "entity": {"smiles": "CCO", "oracle_scores": {"L0": 0.8}},
        }
    ]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_qdrant_delete_and_stats_delegate_to_client() -> None:
    from mf_core.db.qdrant_client import QdrantCollectionClient

    fake = _FakeQdrantClient()
    fake.collections.add("test_col")
    client = QdrantCollectionClient(collection_name="test_col", qdrant_client=fake)
    await client.connect()

    deleted = await client.delete(["mol-001"])
    stats = await client.get_stats("test_col")
    await client.disconnect()

    assert deleted == 1
    assert stats == {"collection": "test_col", "row_count": 3}
    assert fake.deletes[0]["collection_name"] == "test_col"
    assert fake.closed is True
