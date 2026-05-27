"""FTO Patent Service - gRPC server for Freedom-to-Operate patent checking."""
import asyncio
import json
import os
from concurrent import futures
from pathlib import Path
from urllib.parse import urlparse

import grpc
from mf_core.artifacts import (
    ArtifactRequirement,
    RequirementStatus,
    check_artifact,
    require_available,
)

_PATENT_INDEX = ArtifactRequirement("patent_index", "PATENT_INDEX_URI", kind="uri")
_DEAD_ZONE_INDEX = ArtifactRequirement(
    "patent_dead_zone_index",
    "PATENT_DEAD_ZONE_INDEX",
    kind="uri",
)


def _status_objects() -> list[RequirementStatus]:
    return [check_artifact(_PATENT_INDEX), check_artifact(_DEAD_ZONE_INDEX)]


def _require_runtime(*requirements: ArtifactRequirement) -> list[RequirementStatus]:
    statuses = [check_artifact(requirement) for requirement in requirements]
    require_available(statuses)
    return statuses


def runtime_status() -> list[dict]:
    return [status.to_dict() for status in _status_objects()]


async def _abort_unavailable(context, *requirements: ArtifactRequirement):
    statuses = [check_artifact(requirement) for requirement in requirements]
    try:
        require_available(statuses)
    except RuntimeError as exc:
        message = str(exc)
    else:
        message = "FTO patent backend is not configured"
    if context is not None and hasattr(context, "abort"):
        await context.abort(grpc.StatusCode.FAILED_PRECONDITION, message)
    raise RuntimeError(message)


def _abort_client_unavailable(context, message: str):
    if context is not None and hasattr(context, "abort"):
        return context.abort(grpc.StatusCode.FAILED_PRECONDITION, message)
    raise RuntimeError(message)


class FilePatentIndex:
    def __init__(self, index_uri: str):
        path = _file_uri_to_path(index_uri)
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        records = payload.get("records", payload) if isinstance(payload, dict) else payload
        if not isinstance(records, list):
            raise ValueError("Patent index file must contain a list of records")
        self.records = [_validate_patent_record(record) for record in records]

    async def search(self, smiles: str, top_k: int = 10) -> list[dict]:
        hits = [record for record in self.records if record["smiles"] == smiles]
        return sorted(
            hits,
            key=lambda item: float(item.get("similarity", 0.0)),
            reverse=True,
        )[:top_k]


def _file_uri_to_path(uri: str) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme != "file":
        raise ValueError("Only file:// patent index URIs are supported by local service mode")
    path = Path(parsed.path)
    if not path.is_file():
        raise FileNotFoundError(f"Patent index file does not exist: {path}")
    return path


def _validate_patent_record(record: object) -> dict:
    if not isinstance(record, dict):
        raise TypeError("Patent index records must be dictionaries")
    required = ("smiles", "patent_id", "claim_evidence", "source")
    missing = [field for field in required if not record.get(field)]
    if missing:
        raise ValueError(f"Patent index record missing fields: {', '.join(missing)}")
    return dict(record)


def _build_patent_index_client():
    patent_index_uri = os.environ.get("PATENT_INDEX_URI")
    if patent_index_uri and patent_index_uri.startswith("file://"):
        return FilePatentIndex(patent_index_uri)
    return None


def _request_smiles(request) -> str:
    smiles = (
        getattr(request, "smiles", None)
        or getattr(request, "query_smiles", None)
        or getattr(request, "canonical_smiles", None)
    )
    if not smiles:
        raise ValueError("request.smiles is required")
    return str(smiles)


def _search_response(smiles: str, hits: list[dict]):
    return type(
        "SearchPatentsResponse",
        (),
        {
            "smiles": smiles,
            "verdict": "requires_review" if hits else "clear",
            "patent_hits": len(hits),
            "hits": hits,
            "patent_evidence": [
                {
                    "patent_id": hit["patent_id"],
                    "claim_evidence": hit["claim_evidence"],
                    "source": hit["source"],
                }
                for hit in hits
            ],
        },
    )()


class FTOPatentServicer:
    def __init__(self, patent_index_client=None, dead_zone_client=None):
        self.patent_index_client = patent_index_client
        self.dead_zone_client = dead_zone_client

    def _patent_client(self):
        return self.patent_index_client or _build_patent_index_client()

    async def SearchPatents(self, request, context):
        """Search patent databases for molecule FTO clearance."""
        try:
            _require_runtime(_PATENT_INDEX)
        except RuntimeError:
            return await _abort_unavailable(context, _PATENT_INDEX)
        client = self._patent_client()
        if client is None:
            return await _abort_client_unavailable(
                context,
                "Patent index search client is not configured",
            )
        smiles = _request_smiles(request)
        hits = await client.search(smiles, top_k=int(getattr(request, "top_k", 10) or 10))
        return _search_response(smiles, hits)

    async def CheckDeadZone(self, request, context):
        """Check if molecule falls in a patent dead zone."""
        try:
            _require_runtime(_DEAD_ZONE_INDEX)
        except RuntimeError:
            return await _abort_unavailable(context, _DEAD_ZONE_INDEX)
        raise RuntimeError("Patent dead-zone index client is not configured")

    async def AnalyzeMolecule(self, request, context):
        """Full FTO analysis for a molecule across all sources."""
        try:
            _require_runtime(_PATENT_INDEX, _DEAD_ZONE_INDEX)
        except RuntimeError:
            return await _abort_unavailable(context, _PATENT_INDEX, _DEAD_ZONE_INDEX)
        raise RuntimeError("FTO analysis backend is not configured")


async def serve():
    _require_runtime(_PATENT_INDEX, _DEAD_ZONE_INDEX)
    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=10))
    server.add_insecure_port("[::]:50058")
    await server.start()
    print("FTO Patent Service running on :50058")
    await server.wait_for_termination()


if __name__ == "__main__":
    asyncio.run(serve())
