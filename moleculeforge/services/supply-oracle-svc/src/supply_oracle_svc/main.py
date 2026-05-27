"""Supply Oracle Service - gRPC server for building block availability queries."""
import asyncio
import json
import os
import time
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

_REQUIREMENTS = (ArtifactRequirement("supply_catalog", "SUPPLY_CATALOG_URI", kind="uri"),)


def _require_runtime() -> list[RequirementStatus]:
    statuses = [check_artifact(requirement) for requirement in _REQUIREMENTS]
    require_available(statuses)
    return statuses


def runtime_status() -> list[dict]:
    return [check_artifact(requirement).to_dict() for requirement in _REQUIREMENTS]


async def _abort_unavailable(context):
    statuses = [check_artifact(requirement) for requirement in _REQUIREMENTS]
    try:
        require_available(statuses)
    except RuntimeError as exc:
        message = str(exc)
    else:
        message = "Supply catalog client is not configured"
    if context is not None and hasattr(context, "abort"):
        await context.abort(grpc.StatusCode.FAILED_PRECONDITION, message)
    raise RuntimeError(message)


def _abort_client_unavailable(context, message: str):
    if context is not None and hasattr(context, "abort"):
        return context.abort(grpc.StatusCode.FAILED_PRECONDITION, message)
    raise RuntimeError(message)


class FileSupplyCatalog:
    def __init__(self, catalog_uri: str):
        path = _file_uri_to_path(catalog_uri)
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        records = payload.get("records", payload) if isinstance(payload, dict) else payload
        if not isinstance(records, list):
            raise ValueError("Supply catalog file must contain a list of records")
        self.records = [_validate_catalog_record(record) for record in records]

    async def check_availability(self, smiles: str) -> dict:
        for record in self.records:
            if record["smiles"] == smiles:
                return dict(record)
        return {
            "smiles": smiles,
            "available": False,
            "catalog_id": None,
            "source": None,
            "source_timestamp": None,
            "price": None,
            "currency": None,
            "lead_time_days": None,
        }

    async def get_price(self, smiles: str | None = None, catalog_id: str | None = None) -> dict:
        for record in self.records:
            if (catalog_id and record.get("catalog_id") == catalog_id) or (
                smiles and record["smiles"] == smiles
            ):
                return dict(record)
        raise KeyError("catalog entry was not found")


def _file_uri_to_path(catalog_uri: str) -> Path:
    parsed = urlparse(catalog_uri)
    if parsed.scheme != "file":
        raise ValueError("Only file:// supply catalog URIs are supported by local service mode")
    path = Path(parsed.path)
    if not path.is_file():
        raise FileNotFoundError(f"Supply catalog file does not exist: {path}")
    return path


def _validate_catalog_record(record: object) -> dict:
    if not isinstance(record, dict):
        raise TypeError("Supply catalog records must be dictionaries")
    required = ("smiles", "catalog_id", "source", "source_timestamp")
    missing = [field for field in required if not record.get(field)]
    if missing:
        raise ValueError(f"Supply catalog record missing fields: {', '.join(missing)}")
    return dict(record)


def _build_catalog_client():
    catalog_uri = os.environ.get("SUPPLY_CATALOG_URI")
    if catalog_uri and catalog_uri.startswith("file://"):
        return FileSupplyCatalog(catalog_uri)
    return None


def _request_smiles(request) -> str:
    smiles = (
        getattr(request, "smiles", None)
        or getattr(request, "building_block_smiles", None)
        or getattr(request, "query_smiles", None)
    )
    if not smiles:
        raise ValueError("request.smiles is required")
    return str(smiles)


def _availability_response(record: dict):
    return type(
        "AvailabilityResponse",
        (),
        {
            "smiles": record["smiles"],
            "available": bool(record.get("available", False)),
            "catalog_id": record.get("catalog_id"),
            "catalog_source": record.get("source"),
            "source_timestamp": record.get("source_timestamp"),
            "price": record.get("price"),
            "currency": record.get("currency"),
            "lead_time_days": record.get("lead_time_days"),
        },
    )()


class SupplyOracleServicer:
    def __init__(self, catalog_client=None):
        self.catalog_client = catalog_client

    def _client(self):
        return self.catalog_client or _build_catalog_client()

    async def CheckAvailability(self, request, context):
        """Check building block availability across suppliers."""
        try:
            _require_runtime()
        except RuntimeError:
            return await _abort_unavailable(context)
        client = self._client()
        if client is None:
            return await _abort_client_unavailable(
                context,
                "Supply catalog client is not configured",
            )
        record = await client.check_availability(_request_smiles(request))
        return _availability_response(record)

    async def BatchCheck(self, request, context):
        """Batch availability check for multiple building blocks."""
        start_time = time.perf_counter()
        results = []
        for req in getattr(request, "requests", []):
            results.append(await self.CheckAvailability(req, context))
        elapsed_ms = int((time.perf_counter() - start_time) * 1000)
        return type(
            "BatchAvailabilityResponse",
            (),
            {"results": results, "total_elapsed_ms": elapsed_ms},
        )()

    async def GetCatalogPrice(self, request, context):
        """Get pricing info for a specific catalog entry."""
        try:
            _require_runtime()
        except RuntimeError:
            return await _abort_unavailable(context)
        client = self._client()
        if client is None:
            return await _abort_client_unavailable(
                context,
                "Supply catalog pricing client is not configured",
            )
        record = await client.get_price(
            smiles=getattr(request, "smiles", None),
            catalog_id=getattr(request, "catalog_id", None),
        )
        return _availability_response(record)


async def serve():
    _require_runtime()
    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=10))
    server.add_insecure_port("[::]:50059")
    await server.start()
    print("Supply Oracle Service running on :50059")
    await server.wait_for_termination()


if __name__ == "__main__":
    asyncio.run(serve())
