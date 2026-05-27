"""Qdrant collection client for vector search."""
from __future__ import annotations

import uuid
from types import SimpleNamespace
from typing import Any

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qm

    _QDRANT_AVAILABLE = True
except ImportError:
    QdrantClient = None
    qm = None
    _QDRANT_AVAILABLE = False


class QdrantCollectionClient:
    """Wrapper around a Qdrant collection for molecule vector search."""

    def __init__(
        self,
        host: str = "localhost",
        http_port: int = 16333,
        url: str | None = None,
        collection_name: str = "molecules_humu",
        vector_size: int = 128,
        vector_field: str = "vector",
        primary_field: str = "id",
        qdrant_client: Any | None = None,
        timeout_s: float = 30.0,
    ) -> None:
        self.host = host
        self.http_port = http_port
        self.url = url
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.vector_field = vector_field
        self.primary_field = primary_field
        self.timeout_s = timeout_s
        self._client = qdrant_client

    async def connect(self) -> None:
        if self._client is None:
            if not _QDRANT_AVAILABLE:
                raise RuntimeError("qdrant-client is required for QdrantCollectionClient")
            endpoint = self.url or f"http://{self.host}:{self.http_port}"
            self._client = QdrantClient(url=endpoint, timeout=self.timeout_s)
        if self.collection_name not in self._collection_names():
            self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config=_vector_params(self.vector_size),
            )

    async def upsert(self, data: dict[str, list]) -> int:
        client = self._connected_client()
        if not data:
            return 0
        if self.primary_field not in data:
            raise ValueError(f"{self.primary_field} is required for Qdrant upsert")
        if self.vector_field not in data:
            raise ValueError(f"{self.vector_field} is required for Qdrant upsert")
        count = len(data[self.primary_field])
        for key, values in data.items():
            if len(values) != count:
                raise ValueError(f"{key} length does not match {self.primary_field}")
        points = [
            _point_struct(
                point_id=_qdrant_point_id(data[self.primary_field][index]),
                vector=data[self.vector_field][index],
                payload=_payload_from_columns(data, index, self.primary_field, self.vector_field),
            )
            for index in range(count)
        ]
        client.upsert(collection_name=self.collection_name, points=points, wait=True)
        return count

    async def flush(self) -> None:
        self._connected_client()

    async def search(
        self,
        vector: list[float],
        top_k: int = 10,
        output_fields: list[str] | None = None,
    ) -> list[dict]:
        client = self._connected_client()
        fields = output_fields or ["smiles"]
        if hasattr(client, "query_points"):
            result = client.query_points(
                collection_name=self.collection_name,
                query=vector,
                limit=top_k,
                with_payload=True,
            ).points
        else:
            result = client.search(
                collection_name=self.collection_name,
                query_vector=vector,
                limit=top_k,
                with_payload=True,
            )
        hits = []
        for point in result:
            payload = dict(getattr(point, "payload", {}) or {})
            original_id = payload.pop("_id_str", str(getattr(point, "id", "")))
            hits.append(
                {
                    "id": original_id,
                    "distance": float(getattr(point, "score", 0.0)),
                    "entity": {field: payload.get(field) for field in fields},
                }
            )
        return hits

    async def delete(self, ids: list[str]) -> int:
        client = self._connected_client()
        if not ids:
            return 0
        client.delete(
            collection_name=self.collection_name,
            points_selector=_point_selector([_qdrant_point_id(value) for value in ids]),
            wait=True,
        )
        return len(ids)

    async def get_stats(self, collection: str | None = None) -> dict:
        client = self._connected_client()
        collection_name = collection or self.collection_name
        result = client.count(collection_name=collection_name, exact=False)
        return {"collection": collection_name, "row_count": int(result.count)}

    async def drop_collection(self) -> None:
        client = self._connected_client()
        if self.collection_name in self._collection_names():
            client.delete_collection(collection_name=self.collection_name)

    async def disconnect(self) -> None:
        client = self._client
        if client is not None and hasattr(client, "close"):
            client.close()
        self._client = None

    def _connected_client(self):
        if self._client is None:
            raise RuntimeError("Qdrant collection is not connected")
        return self._client

    def _collection_names(self) -> set[str]:
        collections = self._connected_client().get_collections().collections
        return {collection.name for collection in collections}


def _qdrant_point_id(value: object) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_OID, str(value)))


def _payload_from_columns(
    data: dict[str, list],
    index: int,
    primary_field: str,
    vector_field: str,
) -> dict:
    payload = {
        key: values[index]
        for key, values in data.items()
        if key not in {primary_field, vector_field}
    }
    payload["_id_str"] = str(data[primary_field][index])
    return payload


def _vector_params(vector_size: int):
    if qm is not None:
        return qm.VectorParams(size=vector_size, distance=qm.Distance.COSINE)
    return SimpleNamespace(size=vector_size, distance="Cosine")


def _point_struct(point_id: str, vector: list[float], payload: dict):
    if qm is not None:
        return qm.PointStruct(id=point_id, vector=vector, payload=payload)
    return SimpleNamespace(id=point_id, vector=vector, payload=payload)


def _point_selector(point_ids: list[str]):
    if qm is not None:
        return qm.PointIdsList(points=point_ids)
    return SimpleNamespace(points=point_ids)
