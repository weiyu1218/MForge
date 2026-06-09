"""Supply Oracle Service - gRPC server for building block availability queries."""
import asyncio
import json
import os
import time
from collections.abc import Mapping
from concurrent import futures
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import grpc
from mf_core.artifacts import (
    ArtifactRequirement,
    RequirementStatus,
    check_artifact,
    require_available,
)
from mf_core.proto_gen.moleculeforge.v1.oracle import supply_pb2, supply_pb2_grpc

_REQUIREMENTS = (ArtifactRequirement("supply_catalog", "SUPPLY_CATALOG_URI", kind="uri"),)


def _require_runtime() -> list[RequirementStatus]:
    statuses = _runtime_statuses()
    unavailable_configured = [
        status for status in statuses if status.configured and not status.available
    ]
    if unavailable_configured:
        details = "; ".join(
            f"{status.name}: {status.message}" for status in unavailable_configured
        )
        raise RuntimeError(f"Required artifacts or tools are unavailable: {details}")
    if any(status.available for status in statuses):
        return statuses
    require_available(statuses)
    return statuses


def runtime_status() -> list[dict]:
    return [status.to_dict() for status in _runtime_statuses()]


def _runtime_statuses(env: Mapping[str, str] | None = None) -> list[RequirementStatus]:
    env = os.environ if env is None else env
    return [_catalog_status(env)]


def _catalog_status(env: Mapping[str, str]) -> RequirementStatus:
    catalog_uri = str(env.get("SUPPLY_CATALOG_URI", "")).strip()
    if not catalog_uri:
        return check_artifact(_REQUIREMENTS[0], env=env)
    parsed = urlparse(catalog_uri)
    if parsed.scheme != "file":
        return RequirementStatus(
            name="supply_catalog",
            configured=True,
            available=False,
            required=True,
            path=catalog_uri,
            source="SUPPLY_CATALOG_URI",
            message="SUPPLY_CATALOG_URI must use file:// for local supply catalog mode",
        )
    return check_artifact(_REQUIREMENTS[0], env=env)


async def _abort_unavailable(context, message: str | None = None):
    if message is None:
        try:
            _require_runtime()
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


class AiZynthStockCatalog:
    def __init__(self, catalog_uri: str):
        self.path = _file_uri_to_path(catalog_uri)
        self.source_timestamp = datetime.fromtimestamp(
            self.path.stat().st_mtime,
            UTC,
        ).isoformat().replace("+00:00", "Z")
        self._inchi_keys: set[str] | None = None

    async def check_availability(self, smiles: str) -> dict:
        inchi_key = _inchi_key_from_smiles(smiles)
        available = inchi_key in self._stock_inchi_keys()
        return {
            "smiles": smiles,
            "available": available,
            "catalog_id": inchi_key if available else None,
            "source": "aizynth_stock" if available else None,
            "source_timestamp": self.source_timestamp if available else None,
            "price": None,
            "currency": None,
            "lead_time_days": None,
        }

    async def get_price(
        self,
        smiles: str | None = None,
        catalog_id: str | None = None,
    ) -> dict:
        if catalog_id:
            available = catalog_id in self._stock_inchi_keys()
            inchi_key = catalog_id
            query_smiles = smiles or ""
        elif smiles:
            inchi_key = _inchi_key_from_smiles(smiles)
            available = inchi_key in self._stock_inchi_keys()
            query_smiles = smiles
        else:
            raise KeyError("catalog entry was not found")
        if not available:
            raise KeyError("catalog entry was not found")
        return {
            "smiles": query_smiles,
            "available": True,
            "catalog_id": inchi_key,
            "source": "aizynth_stock",
            "source_timestamp": self.source_timestamp,
            "price": None,
            "currency": None,
            "lead_time_days": None,
        }

    def _stock_inchi_keys(self) -> set[str]:
        if self._inchi_keys is None:
            try:
                import pandas as pd
            except ImportError as exc:
                raise RuntimeError("pandas is required to read AiZynthFinder stock HDF5") from exc
            frame = pd.read_hdf(self.path, key="table")
            if "inchi_key" not in frame.columns:
                raise ValueError("AiZynthFinder stock HDF5 requires an inchi_key column")
            self._inchi_keys = {str(value) for value in frame["inchi_key"].dropna().values}
        return self._inchi_keys


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


def _normalize_supply_record(record: Any, smiles: str) -> dict:
    if not isinstance(record, dict):
        raise TypeError("supply provider records must be dictionaries")
    available = bool(record.get("available", False))
    if available:
        required = ("catalog_id", "source", "source_timestamp")
        missing = [field for field in required if not record.get(field)]
        if missing:
            raise ValueError(
                "available supply provider record missing fields: "
                + ", ".join(missing)
            )
    return {
        "smiles": str(record.get("smiles") or smiles),
        "available": available,
        "catalog_id": record.get("catalog_id"),
        "source": record.get("source"),
        "source_timestamp": record.get("source_timestamp"),
        "price": record.get("price"),
        "currency": record.get("currency"),
        "lead_time_days": record.get("lead_time_days"),
    }


def _best_supply_record(smiles: str, records: list[dict]) -> dict:
    available = [record for record in records if record.get("available")]
    if not available:
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
    return min(available, key=_supply_rank_key)


def _supply_rank_key(record: dict) -> tuple[float, float, str]:
    price = record.get("price")
    lead_time = record.get("lead_time_days")
    return (
        float(price) if isinstance(price, int | float) else float("inf"),
        float(lead_time) if isinstance(lead_time, int | float) else float("inf"),
        str(record.get("source") or ""),
    )


def _build_catalog_client():
    catalog_uri = os.environ.get("SUPPLY_CATALOG_URI")
    if catalog_uri and catalog_uri.startswith("file://"):
        path = _file_uri_to_path(catalog_uri)
        if path.suffix.lower() in {".h5", ".hdf5"}:
            return AiZynthStockCatalog(catalog_uri)
        return FileSupplyCatalog(catalog_uri)
    return None


def _inchi_key_from_smiles(smiles: str) -> str:
    try:
        from rdkit import Chem
    except ImportError as exc:
        raise RuntimeError("RDKit is required for supply stock lookup") from exc
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid supply query SMILES: {smiles}")
    return Chem.MolToInchiKey(mol)


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
        if self.catalog_client is not None:
            return self.catalog_client
        _require_runtime()
        return _build_catalog_client()

    async def CheckAvailability(self, request, context):
        """Check building block availability across suppliers."""
        try:
            client = self._client()
        except RuntimeError as exc:
            return await _abort_unavailable(context, str(exc))
        if client is None:
            try:
                _require_runtime()
            except RuntimeError:
                return await _abort_unavailable(context)
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
            client = self._client()
        except RuntimeError as exc:
            return await _abort_unavailable(context, str(exc))
        if client is None:
            try:
                _require_runtime()
            except RuntimeError:
                return await _abort_unavailable(context)
            return await _abort_client_unavailable(
                context,
                "Supply catalog pricing client is not configured",
            )
        record = await client.get_price(
            smiles=getattr(request, "smiles", None),
            catalog_id=getattr(request, "catalog_id", None),
        )
        return _availability_response(record)


class SupplyOracleGrpcServicer(supply_pb2_grpc.SupplyOracleServiceServicer):
    def __init__(self, service: SupplyOracleServicer | None = None):
        self.service = service or SupplyOracleServicer()

    async def CheckAvailability(self, request, context):
        response = await self.service.CheckAvailability(request, context)
        return _availability_proto(response)

    async def BatchCheck(self, request, context):
        start_time = time.perf_counter()
        results = []
        for item in getattr(request, "requests", []):
            results.append(await self.CheckAvailability(item, context))
        return supply_pb2.BatchAvailabilityResponse(
            results=results,
            total_elapsed_ms=int((time.perf_counter() - start_time) * 1000),
        )

    async def GetCatalogPrice(self, request, context):
        response = await self.service.GetCatalogPrice(request, context)
        return _availability_proto(response)


def _availability_proto(response) -> supply_pb2.AvailabilityResponse:
    payload = {
        "smiles": str(getattr(response, "smiles", "")),
        "available": bool(getattr(response, "available", False)),
        "catalog_id": str(getattr(response, "catalog_id", "") or ""),
        "catalog_source": str(getattr(response, "catalog_source", "") or ""),
        "source_timestamp": str(getattr(response, "source_timestamp", "") or ""),
        "currency": str(getattr(response, "currency", "") or ""),
    }
    price = getattr(response, "price", None)
    if price is not None:
        payload["price"] = float(price)
    lead_time_days = getattr(response, "lead_time_days", None)
    if lead_time_days is not None:
        payload["lead_time_days"] = int(lead_time_days)
    return supply_pb2.AvailabilityResponse(**payload)


def register_grpc_services(server) -> None:
    supply_pb2_grpc.add_SupplyOracleServiceServicer_to_server(
        SupplyOracleGrpcServicer(),
        server,
    )


async def serve():
    _require_runtime()
    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=10))
    register_grpc_services(server)
    server.add_insecure_port("[::]:50059")
    await server.start()
    print("Supply Oracle Service running on :50059")
    await server.wait_for_termination()


if __name__ == "__main__":
    asyncio.run(serve())
