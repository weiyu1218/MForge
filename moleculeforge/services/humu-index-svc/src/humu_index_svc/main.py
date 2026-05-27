"""HUMU Index Service - FastAPI server for Qdrant vector store ANN retrieval."""
import inspect
import os
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="HUMU Index Service", version="0.1.0")


class IndexRequest(BaseModel):
    ids: list[str] | None = None
    vectors: list[list[float]]
    collection: str = "default"
    metadata: dict | None = None


class SearchRequest(BaseModel):
    query_vector: list[float]
    collection: str = "default"
    top_k: int = 10
    metric_type: str = "L2"


class DeleteRequest(BaseModel):
    ids: list[str]
    collection: str = "default"


@app.get("/health")
async def health():
    _require_qdrant_config()
    client = getattr(app.state, "qdrant_client", None)
    return {
        "status": "healthy",
        "backend": "qdrant",
        "uri": _qdrant_url(),
        "artifact_status": runtime_status(),
        "client_configured": client is not None,
    }


@app.post("/v1/index/insert")
async def insert_vectors(req: IndexRequest):
    """Insert vectors into Qdrant collection."""
    _require_qdrant_config()
    if not req.ids:
        raise HTTPException(status_code=400, detail="ids are required for Qdrant insert")
    if len(req.ids) != len(req.vectors):
        raise HTTPException(status_code=400, detail="ids and vectors must have equal length")
    client = await _qdrant_client(req.collection)
    data = _columnar_payload(req.ids, req.vectors, req.metadata or {})
    result = _call_insert(client, req.collection, data)
    if inspect.isawaitable(result):
        result = await result
    return {"collection": req.collection, "inserted": int(result)}


@app.post("/v1/index/search")
async def search_vectors(req: SearchRequest):
    """ANN search in Qdrant collection."""
    _require_qdrant_config()
    client = await _qdrant_client(req.collection)
    result = _call_search(client, req.collection, req.query_vector, req.top_k)
    if inspect.isawaitable(result):
        result = await result
    return {"collection": req.collection, "results": list(result or [])}


@app.delete("/v1/index/delete")
async def delete_vectors(req: DeleteRequest):
    """Delete vectors from Qdrant collection."""
    _require_qdrant_config()
    client = await _qdrant_client(req.collection)
    result = _call_delete(client, req.collection, req.ids)
    if inspect.isawaitable(result):
        result = await result
    return {"collection": req.collection, "deleted": int(result)}


@app.get("/v1/index/stats/{collection}")
async def collection_stats(collection: str):
    """Get collection statistics."""
    _require_qdrant_config()
    client = await _qdrant_client(collection)
    result = _call_stats(client, collection)
    if inspect.isawaitable(result):
        result = await result
    stats = dict(result or {})
    stats.setdefault("collection", collection)
    stats["backend"] = "qdrant"
    return stats


def _require_qdrant_config() -> None:
    statuses = runtime_status()
    if any(status["required"] and not status["available"] for status in statuses):
        missing = "; ".join(status["message"] for status in statuses if not status["available"])
        raise HTTPException(
            status_code=503,
            detail={
                "message": f"Required vector backend is unavailable: {missing}",
                "artifact_status": statuses,
            },
        )


def runtime_status() -> list[dict]:
    configured = bool(os.environ.get("QDRANT_URL") or os.environ.get("QDRANT_HOST"))
    return [
        {
            "name": "qdrant_endpoint",
            "configured": configured,
            "available": configured,
            "required": True,
            "path": _qdrant_url() if configured else None,
            "source": "QDRANT_URL or QDRANT_HOST",
            "message": (
                "qdrant endpoint is configured"
                if configured
                else "QDRANT_URL or QDRANT_HOST is required for qdrant_endpoint"
            ),
        }
    ]


async def _qdrant_client(collection: str):
    client = getattr(app.state, "qdrant_client", None)
    if client is not None:
        return client
    from mf_core.db.qdrant_client import QdrantCollectionClient

    parsed = urlparse(_qdrant_url())
    client = QdrantCollectionClient(
        url=_qdrant_url() if parsed.scheme else None,
        host=parsed.hostname or "localhost",
        http_port=parsed.port or int(os.environ.get("QDRANT_HTTP_PORT", "16333")),
        collection_name=collection,
        vector_field="vector",
        primary_field="id",
    )
    await client.connect()
    app.state.qdrant_client = client
    return client


def _qdrant_url() -> str:
    if os.environ.get("QDRANT_URL"):
        return os.environ["QDRANT_URL"]
    host = os.environ.get("QDRANT_HOST", "localhost")
    port = os.environ.get("QDRANT_HTTP_PORT", "16333")
    return f"http://{host}:{port}"


def _columnar_payload(
    ids: list[str],
    vectors: list[list[float]],
    metadata: dict,
) -> dict[str, list]:
    data: dict[str, list] = {"id": ids, "vector": vectors}
    for key, value in metadata.items():
        if not isinstance(value, list) or len(value) != len(ids):
            raise HTTPException(
                status_code=400,
                detail=f"metadata.{key} must be a list with one value per vector",
            )
        data[key] = value
    return data


def _records_from_columns(data: dict[str, list]) -> list[dict]:
    keys = list(data)
    size = len(data["id"])
    return [{key: data[key][index] for key in keys} for index in range(size)]


def _call_insert(client, collection: str, data: dict[str, list]):
    if hasattr(client, "upsert"):
        return client.upsert(data)
    if hasattr(client, "insert"):
        return client.insert(collection, _records_from_columns(data))
    raise HTTPException(status_code=501, detail="Qdrant insert client is not configured")


def _call_search(client, collection: str, vector: list[float], top_k: int):
    if not hasattr(client, "search"):
        raise HTTPException(status_code=501, detail="Qdrant search client is not configured")
    search = getattr(client, "search")
    parameters = list(inspect.signature(search).parameters)
    if parameters and parameters[0] in {"collection", "collection_name"}:
        return search(collection, vector, limit=top_k)
    return search(vector, top_k=top_k, output_fields=["smiles"])


def _call_delete(client, collection: str, ids: list[str]):
    if not hasattr(client, "delete"):
        raise HTTPException(status_code=501, detail="Qdrant delete client is not configured")
    delete = getattr(client, "delete")
    parameters = list(inspect.signature(delete).parameters)
    if parameters and parameters[0] in {"collection", "collection_name"}:
        return delete(collection, ids)
    return delete(ids)


def _call_stats(client, collection: str):
    if hasattr(client, "get_stats"):
        return client.get_stats(collection)
    if hasattr(client, "collection_stats"):
        return client.collection_stats(collection)
    raise HTTPException(status_code=501, detail="Qdrant stats client is not configured")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8009, log_level="info")
