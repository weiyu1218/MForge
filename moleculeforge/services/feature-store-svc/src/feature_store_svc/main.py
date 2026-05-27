"""Feature Store Service - FastAPI server for Feast-based ML feature retrieval."""
import inspect
import os
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from mf_core.artifacts import ArtifactRequirement, check_artifact, require_available
from pydantic import BaseModel

app = FastAPI(title="Feature Store Service", version="0.1.0")
_REQUIREMENTS = (ArtifactRequirement("feast_repo", "FEAST_REPO_PATH", kind="directory"),)


class FeatureRequest(BaseModel):
    entities: list[str]
    features: list[str]
    feature_view: str = "molecule_features"


class BatchFeatureRequest(BaseModel):
    entity_ids: list[str]
    feature_names: list[str]
    feature_views: list[str] = ["molecule_features", "protein_features"]


@app.get("/health")
async def health():
    _require_feast_config()
    store = getattr(app.state, "feast_store", None)
    return {
        "status": "healthy",
        "backend": "feast",
        "repo_path": os.environ["FEAST_REPO_PATH"],
        "artifact_status": runtime_status(),
        "client_configured": store is not None,
    }


@app.post("/v1/features/online")
async def get_online_features(req: FeatureRequest):
    """Retrieve online features for given entities from Feast online store."""
    _require_feast_config()
    store = _feast_store()
    result = store.get_online_features(
        features=_feature_refs(req.feature_view, req.features),
        entity_rows=_entity_rows(req.entities),
    )
    if inspect.isawaitable(result):
        result = await result
    return _result_to_dict(result)


@app.post("/v1/features/batch")
async def get_batch_features(req: BatchFeatureRequest):
    """Retrieve batch features for offline training."""
    _require_feast_config()
    store = _feast_store()
    if not hasattr(store, "get_historical_features"):
        raise HTTPException(status_code=501, detail="Feast offline store client is not configured")
    result = store.get_historical_features(
        features=[
            feature
            if ":" in feature
            else f"{view}:{feature}"
            for view in req.feature_views
            for feature in req.feature_names
        ],
        entity_df=_entity_rows(req.entity_ids),
    )
    if inspect.isawaitable(result):
        result = await result
    return _result_to_dict(result)


@app.get("/v1/features/views")
async def list_feature_views():
    """List available feature views."""
    _require_feast_config()
    store = _feast_store()
    if not hasattr(store, "list_feature_views"):
        raise HTTPException(status_code=501, detail="Feast feature view listing is not configured")
    result = store.list_feature_views()
    if inspect.isawaitable(result):
        result = await result
    return {"feature_views": _normalise_feature_views(result)}


@app.post("/v1/features/materialize")
async def materialize_features(feature_view: str = "molecule_features"):
    """Trigger feature materialization from offline to online store."""
    _require_feast_config()
    store = _feast_store()
    if not hasattr(store, "materialize_incremental"):
        raise HTTPException(status_code=501, detail="Feast materialization is not configured")
    result = store.materialize_incremental(datetime.now(timezone.utc))
    if inspect.isawaitable(result):
        result = await result
    return {"feature_view": feature_view, "materialized": True, "result": _result_to_dict(result)}


def _require_feast_config() -> None:
    statuses = [check_artifact(requirement) for requirement in _REQUIREMENTS]
    try:
        require_available(statuses)
    except RuntimeError as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "message": str(exc),
                "artifact_status": [status.to_dict() for status in statuses],
            },
        ) from exc


def runtime_status() -> list[dict]:
    return [check_artifact(requirement).to_dict() for requirement in _REQUIREMENTS]


def _feast_store():
    store = getattr(app.state, "feast_store", None)
    if store is not None:
        return store
    try:
        from feast import FeatureStore
    except ImportError as exc:
        raise HTTPException(status_code=503, detail="feast package is not installed") from exc
    store = FeatureStore(repo_path=os.environ["FEAST_REPO_PATH"])
    app.state.feast_store = store
    return store


def _feature_refs(feature_view: str, features: list[str]) -> list[str]:
    return [feature if ":" in feature else f"{feature_view}:{feature}" for feature in features]


def _entity_rows(entities: list[str]) -> list[dict]:
    return [{"entity_id": entity} for entity in entities]


def _result_to_dict(result):
    if result is None:
        return {}
    if isinstance(result, dict):
        return result
    if hasattr(result, "to_dict"):
        return result.to_dict()
    if hasattr(result, "to_df"):
        frame = result.to_df()
        if hasattr(frame, "to_dict"):
            return {"rows": frame.to_dict(orient="records")}
    return {"result": result}


def _normalise_feature_views(feature_views) -> list[dict]:
    rows = []
    for view in feature_views:
        if isinstance(view, dict):
            rows.append(view)
            continue
        rows.append(
            {
                "name": getattr(view, "name", ""),
                "features": [
                    getattr(feature, "name", str(feature))
                    for feature in getattr(view, "features", [])
                ],
            }
        )
    return rows


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8008, log_level="info")
