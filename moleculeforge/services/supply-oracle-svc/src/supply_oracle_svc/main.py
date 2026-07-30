"""Supply Oracle Service - gRPC server for building block availability queries."""

import asyncio
import hashlib
import json
import logging
import math
import os
import re
import sys
import tempfile
import time
from collections.abc import Mapping
from concurrent import futures
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
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
_SHA256_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_VALIDATION_GATE_ENV = "MF_ALLOW_SYNTHETIC_VALIDATION"
_VALIDATION_MARKER = "synthetic_pipeline_validation_only"
_LOGGER = logging.getLogger(__name__)


class SupplyRequestError(ValueError):
    pass


class SupplyCatalogDataError(RuntimeError):
    pass


class SupplyNotFoundError(LookupError):
    pass


class SupplyUnavailableError(RuntimeError):
    pass


def _require_runtime() -> list[RequirementStatus]:
    statuses = _runtime_statuses()
    unavailable_configured = [
        status for status in statuses if status.configured and not status.available
    ]
    if unavailable_configured:
        details = "; ".join(f"{status.name}: {status.message}" for status in unavailable_configured)
        raise SupplyUnavailableError(f"Required artifacts or tools are unavailable: {details}")
    if any(status.available for status in statuses):
        return statuses
    try:
        require_available(statuses)
    except RuntimeError as exc:
        raise SupplyUnavailableError(str(exc)) from exc
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
    status = check_artifact(_REQUIREMENTS[0], env=env)
    if not status.available:
        return status
    try:
        _validate_catalog_schema(catalog_uri)
    except Exception as exc:
        return RequirementStatus(
            name=status.name,
            configured=True,
            available=False,
            required=status.required,
            path=status.path,
            source=status.source,
            message=f"Supply catalog schema is invalid: {exc}",
        )
    return RequirementStatus(
        name=status.name,
        configured=True,
        available=True,
        required=status.required,
        path=status.path,
        source=status.source,
        message="supply_catalog is available and schema-valid",
    )


class FileSupplyCatalog:
    def __init__(self, catalog_uri: str):
        self.path = _file_uri_to_path(catalog_uri)
        self.catalog_checksum = _catalog_checksum(self.path)
        raw_version, records = _load_json_catalog(self.path)
        self.catalog_version = (
            raw_version.strip() if isinstance(raw_version, str) else self.catalog_checksum
        )
        self.records = records

    def validate_schema(self) -> None:
        return None

    async def check_availability(self, smiles: str, **identity: Any) -> dict:
        for record in self.records:
            if record["smiles"] == smiles:
                return self._with_evidence(dict(record), smiles, identity)
        return self._with_evidence(
            {
                "smiles": smiles,
                "available": False,
                "catalog_id": None,
                "source": None,
                "source_timestamp": None,
                "price": None,
                "currency": None,
                "lead_time_days": None,
            },
            smiles,
            identity,
        )

    async def get_price(
        self,
        smiles: str | None = None,
        catalog_id: str | None = None,
        **identity: Any,
    ) -> dict:
        for record in self.records:
            if (catalog_id and record.get("catalog_id") == catalog_id) or (
                smiles and record["smiles"] == smiles
            ):
                return self._with_evidence(
                    dict(record),
                    str(smiles or record["smiles"]),
                    identity,
                )
        raise KeyError("catalog entry was not found")

    def _with_evidence(
        self,
        record: dict,
        smiles: str,
        identity: Mapping[str, Any],
    ) -> dict:
        record.update(identity)
        record["evidence_id"] = _evidence_id(
            self.catalog_checksum,
            smiles,
            bool(record.get("available", False)),
            record.get("catalog_id"),
        )
        record["catalog_version"] = self.catalog_version
        record["catalog_checksum"] = self.catalog_checksum
        return record


class AiZynthStockCatalog:
    def __init__(self, catalog_uri: str):
        self.path = _file_uri_to_path(catalog_uri)
        self.source_timestamp = (
            datetime.fromtimestamp(
                self.path.stat().st_mtime,
                UTC,
            )
            .isoformat()
            .replace("+00:00", "Z")
        )
        self.catalog_checksum = _catalog_checksum(self.path)
        self.catalog_version = self.catalog_checksum
        self._inchi_keys: set[str] | None = None

    def validate_schema(self) -> None:
        _validate_hdf5_schema(self.path)

    async def check_availability(self, smiles: str, **identity: Any) -> dict:
        inchi_key = _inchi_key_from_smiles(smiles)
        available = inchi_key in self._stock_inchi_keys()
        record = {
            "smiles": smiles,
            "available": available,
            "catalog_id": inchi_key if available else None,
            "source": "aizynth_stock" if available else None,
            "source_timestamp": self.source_timestamp if available else None,
            "price": None,
            "currency": None,
            "lead_time_days": None,
        }
        return self._with_evidence(record, identity)

    async def get_price(
        self,
        smiles: str | None = None,
        catalog_id: str | None = None,
        **identity: Any,
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
        return self._with_evidence(
            {
                "smiles": query_smiles,
                "available": True,
                "catalog_id": inchi_key,
                "source": "aizynth_stock",
                "source_timestamp": self.source_timestamp,
                "price": None,
                "currency": None,
                "lead_time_days": None,
            },
            identity,
        )

    def _stock_inchi_keys(self) -> set[str]:
        if self._inchi_keys is None:
            pd = _pandas_for_hdf5()
            try:
                with pd.HDFStore(self.path, mode="r") as store:
                    storer = store.get_storer("table")
                    raw_version = getattr(storer.attrs, "catalog_version", None)
                    frame = store["table"]
            except SupplyCatalogDataError:
                raise
            except Exception as exc:
                raise SupplyCatalogDataError(
                    f"AiZynthFinder stock HDF5 cannot be read: {exc}"
                ) from exc
            if "inchi_key" not in frame.columns:
                raise SupplyCatalogDataError(
                    "AiZynthFinder stock HDF5 requires an inchi_key column"
                )
            values = frame["inchi_key"].tolist()
            if any(not isinstance(value, str) or not value.strip() for value in values):
                raise SupplyCatalogDataError(
                    "AiZynthFinder stock HDF5 inchi_key values must be non-empty strings"
                )
            if raw_version is not None and (
                not isinstance(raw_version, str) or not raw_version.strip()
            ):
                raise SupplyCatalogDataError("AiZynthFinder stock catalog_version must be a string")
            if isinstance(raw_version, str):
                self.catalog_version = raw_version.strip()
            self._inchi_keys = {value.strip() for value in values}
        return self._inchi_keys

    def _with_evidence(self, record: dict, identity: Mapping[str, Any]) -> dict:
        record.update(identity)
        record["evidence_id"] = _evidence_id(
            self.catalog_checksum,
            str(record["smiles"]),
            bool(record["available"]),
            record.get("catalog_id"),
        )
        record["catalog_version"] = self.catalog_version
        record["catalog_checksum"] = self.catalog_checksum
        return record


def _file_uri_to_path(catalog_uri: str) -> Path:
    parsed = urlparse(catalog_uri)
    if parsed.scheme != "file":
        raise ValueError("Only file:// supply catalog URIs are supported by local service mode")
    path = Path(parsed.path)
    if not path.is_file():
        raise FileNotFoundError(f"Supply catalog file does not exist: {path}")
    return path


def _validate_catalog_schema(catalog_uri: str) -> None:
    path = _file_uri_to_path(catalog_uri)
    if path.suffix.lower() in {".h5", ".hdf5"}:
        _validate_hdf5_schema(path)
    else:
        _load_json_catalog(path)


def _load_json_catalog(path: Path) -> tuple[str | None, list[dict]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SupplyCatalogDataError(f"Supply catalog JSON cannot be read: {exc}") from exc
    records = payload.get("records", payload) if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise SupplyCatalogDataError("Supply catalog file must contain a list of records")
    raw_version = payload.get("catalog_version") if isinstance(payload, dict) else None
    if raw_version is not None and (not isinstance(raw_version, str) or not raw_version.strip()):
        raise SupplyCatalogDataError("Supply catalog catalog_version must be a string")
    return raw_version, [_validate_catalog_record(record) for record in records]


def _pandas_for_hdf5():
    try:
        import pandas as pd

        __import__("tables")
    except ImportError as exc:
        raise SupplyUnavailableError(
            "pandas and tables are required to read AiZynthFinder stock HDF5"
        ) from exc
    return pd


def _validate_hdf5_schema(path: Path) -> None:
    pd = _pandas_for_hdf5()
    try:
        with pd.HDFStore(path, mode="r") as store:
            if "/table" not in store.keys():
                raise SupplyCatalogDataError("AiZynthFinder stock HDF5 requires a table dataset")
            storer = store.get_storer("table")
            raw_version = getattr(storer.attrs, "catalog_version", None)
            sample = store.select("table", start=0, stop=1)
    except SupplyCatalogDataError:
        raise
    except Exception as exc:
        raise SupplyCatalogDataError(f"AiZynthFinder stock HDF5 cannot be read: {exc}") from exc
    if "inchi_key" not in sample.columns:
        raise SupplyCatalogDataError("AiZynthFinder stock HDF5 requires an inchi_key column")
    sample_values = sample["inchi_key"].tolist()
    if any(not isinstance(value, str) or not value.strip() for value in sample_values):
        raise SupplyCatalogDataError(
            "AiZynthFinder stock HDF5 inchi_key values must be non-empty strings"
        )
    if raw_version is not None and (not isinstance(raw_version, str) or not raw_version.strip()):
        raise SupplyCatalogDataError("AiZynthFinder stock catalog_version must be a string")


def _catalog_checksum(path: Path) -> str:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise SupplyUnavailableError(f"Supply catalog cannot be read: {exc}") from exc
    return f"sha256:{digest.hexdigest()}"


def _evidence_id(
    catalog_checksum: str,
    smiles: str,
    available: bool,
    catalog_id: object,
) -> str:
    payload = json.dumps(
        {
            "catalog_checksum": catalog_checksum,
            "smiles": smiles,
            "available": available,
            "catalog_id": str(catalog_id or ""),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _validate_catalog_record(record: object) -> dict:
    if not isinstance(record, dict):
        raise SupplyCatalogDataError("Supply catalog records must be dictionaries")
    if not isinstance(record.get("smiles"), str) or not record["smiles"].strip():
        raise SupplyCatalogDataError("Supply catalog record smiles must be a non-empty string")
    if "available" not in record or not isinstance(record["available"], bool):
        raise SupplyCatalogDataError("Supply catalog record available must be a boolean")
    if record["available"]:
        required = ("catalog_id", "source", "source_timestamp")
        missing = [
            field
            for field in required
            if not isinstance(record.get(field), str) or not record[field].strip()
        ]
        if missing:
            raise SupplyCatalogDataError(
                f"Supply catalog record missing fields: {', '.join(missing)}"
            )
    _validate_optional_commercial_fields(record)
    return dict(record)


def _normalize_supply_record(record: Any, smiles: str) -> dict:
    if not isinstance(record, dict):
        raise SupplyCatalogDataError("supply provider records must be dictionaries")
    if record.get("smiles") != smiles:
        raise SupplyCatalogDataError("supply provider response smiles does not match request")
    if not isinstance(record.get("available"), bool):
        raise SupplyCatalogDataError("supply provider available must be a boolean")
    available = record["available"]
    if available:
        required = ("catalog_id", "source", "source_timestamp")
        missing = [
            field
            for field in required
            if not isinstance(record.get(field), str) or not record[field].strip()
        ]
        if missing:
            raise SupplyCatalogDataError(
                "available supply provider record missing fields: " + ", ".join(missing)
            )
    _validate_optional_commercial_fields(record)
    metadata = ("evidence_id", "catalog_version", "catalog_checksum")
    missing_metadata = [
        field
        for field in metadata
        if not isinstance(record.get(field), str) or not record[field].strip()
    ]
    if missing_metadata:
        raise SupplyCatalogDataError(
            "supply provider record missing evidence metadata: " + ", ".join(missing_metadata)
        )
    if not _SHA256_PATTERN.fullmatch(record["catalog_checksum"]):
        raise SupplyCatalogDataError(
            "supply provider catalog_checksum must be sha256:<64 lowercase hex>"
        )
    return {
        "smiles": smiles,
        "available": available,
        "catalog_id": record.get("catalog_id"),
        "source": record.get("source"),
        "source_timestamp": record.get("source_timestamp"),
        "price": record.get("price"),
        "currency": record.get("currency"),
        "lead_time_days": record.get("lead_time_days"),
        "evidence_id": record["evidence_id"].strip(),
        "catalog_version": record["catalog_version"].strip(),
        "catalog_checksum": record["catalog_checksum"],
    }


def _validate_optional_commercial_fields(record: Mapping[str, Any]) -> None:
    price = record.get("price")
    if price is not None and (
        isinstance(price, bool)
        or not isinstance(price, int | float)
        or not math.isfinite(float(price))
        or price < 0
    ):
        raise SupplyCatalogDataError("supply provider price must be a finite non-negative number")
    currency = record.get("currency")
    if price is not None and (not isinstance(currency, str) or not currency.strip()):
        raise SupplyCatalogDataError("supply provider currency is required when price is present")
    lead_time_days = record.get("lead_time_days")
    if lead_time_days is not None and (
        isinstance(lead_time_days, bool)
        or not isinstance(lead_time_days, int)
        or lead_time_days < 0
    ):
        raise SupplyCatalogDataError(
            "supply provider lead_time_days must be a non-negative integer"
        )


def _build_catalog_client():
    catalog_uri = os.environ.get("SUPPLY_CATALOG_URI")
    if catalog_uri and catalog_uri.startswith("file://"):
        return _catalog_client_from_uri(catalog_uri)
    return None


def _catalog_client_from_uri(catalog_uri: str):
    path = _file_uri_to_path(catalog_uri)
    if path.suffix.lower() in {".h5", ".hdf5"}:
        return AiZynthStockCatalog(catalog_uri)
    return FileSupplyCatalog(catalog_uri)


def _inchi_key_from_smiles(smiles: str) -> str:
    Chem, mol = _rdkit_molecule(smiles)
    return Chem.MolToInchiKey(mol)


def _rdkit_molecule(smiles: str):
    try:
        from rdkit import Chem
    except ImportError as exc:
        raise SupplyUnavailableError("RDKit is required for supply stock lookup") from exc
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise SupplyRequestError(f"Invalid supply query SMILES: {smiles}")
    return Chem, mol


def _request_smiles(request) -> str:
    smiles = (
        getattr(request, "smiles", None)
        or getattr(request, "building_block_smiles", None)
        or getattr(request, "query_smiles", None)
    )
    if not isinstance(smiles, str) or not smiles.strip():
        raise SupplyRequestError("request.smiles is required")
    normalized = smiles.strip()
    _rdkit_molecule(normalized)
    return normalized


def _required_request_id(request) -> str:
    request_id = getattr(request, "request_id", None)
    if not isinstance(request_id, str) or not request_id.strip():
        raise SupplyRequestError("request.request_id is required")
    return request_id.strip()


def _request_identity(request, *, require_request_id: bool = True) -> dict[str, Any]:
    request_id = _required_request_id(request) if require_request_id else ""
    candidate_index = _optional_candidate_index(request)
    return {
        "request_id": request_id,
        "project_id": _optional_string(request, "project_id"),
        "candidate_id": _optional_string(request, "candidate_id"),
        "candidate_index": candidate_index,
        "canonical_smiles": _optional_string(request, "canonical_smiles"),
    }


def _optional_string(request, field: str) -> str:
    value = getattr(request, field, "")
    if value in (None, ""):
        return ""
    if not isinstance(value, str) or not value.strip():
        raise SupplyRequestError(f"request.{field} must be a non-empty string")
    return value.strip()


def _optional_candidate_index(request) -> int | None:
    has_field = getattr(request, "HasField", None)
    if callable(has_field):
        try:
            has_candidate_index = has_field("candidate_index")
        except ValueError:
            has_candidate_index = True
        if not has_candidate_index:
            return None
    elif not hasattr(request, "candidate_index"):
        return None
    value = getattr(request, "candidate_index", None)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SupplyRequestError("request.candidate_index must be a non-negative integer")
    return value


def _availability_response(record: dict, identity: Mapping[str, Any]):
    return SimpleNamespace(
        smiles=record["smiles"],
        available=record["available"],
        catalog_id=record.get("catalog_id"),
        catalog_source=record.get("source"),
        source_timestamp=record.get("source_timestamp"),
        price=record.get("price"),
        currency=record.get("currency"),
        lead_time_days=record.get("lead_time_days"),
        evidence_id=record["evidence_id"],
        catalog_version=record["catalog_version"],
        catalog_checksum=record["catalog_checksum"],
        **identity,
    )


def _require_matching_batch_identity(
    request,
    batch_identity: Mapping[str, Any],
) -> None:
    item_identity = _request_identity(request)
    mismatched = [
        field
        for field in ("project_id", "candidate_id", "candidate_index", "canonical_smiles")
        if item_identity[field] != batch_identity[field]
    ]
    if mismatched:
        raise SupplyRequestError(
            "batch item identity does not match batch request: " + ", ".join(mismatched)
        )


class SupplyOracleServicer:
    def __init__(self, catalog_client=None):
        self.catalog_client = catalog_client
        self._resolved_client = None

    def _client(self):
        if self.catalog_client is not None:
            return self.catalog_client
        if self._resolved_client is not None:
            return self._resolved_client
        _require_runtime()
        self._resolved_client = _build_catalog_client()
        if self._resolved_client is None:
            raise SupplyUnavailableError("Supply catalog client is not configured")
        return self._resolved_client

    async def CheckAvailability(self, request, context):
        """Check building block availability across suppliers."""
        identity = _request_identity(request)
        smiles = _request_smiles(request)
        record = await self._client().check_availability(smiles)
        normalized = _normalize_supply_record(record, smiles)
        return _availability_response(normalized, identity)

    async def BatchCheck(self, request, context):
        """Batch availability check for multiple building blocks."""
        identity = _request_identity(request)
        requests = list(getattr(request, "requests", []))
        if not requests:
            raise SupplyRequestError("batch requests must not be empty")
        for item in requests:
            _require_matching_batch_identity(item, identity)
        start_time = time.perf_counter()
        results = []
        for req in requests:
            results.append(await self.CheckAvailability(req, context))
        elapsed_ms = int((time.perf_counter() - start_time) * 1000)
        return SimpleNamespace(
            results=results,
            total_elapsed_ms=elapsed_ms,
            **identity,
        )

    async def GetCatalogPrice(self, request, context):
        """Get pricing info for a specific catalog entry."""
        identity = _request_identity(request)
        smiles = _optional_string(request, "smiles")
        catalog_id = _optional_string(request, "catalog_id")
        if not smiles and not catalog_id:
            raise SupplyRequestError("request.smiles or request.catalog_id is required")
        if smiles:
            _rdkit_molecule(smiles)
        try:
            record = await self._client().get_price(
                smiles=smiles or None,
                catalog_id=catalog_id or None,
            )
        except KeyError as exc:
            raise SupplyNotFoundError(str(exc)) from exc
        if not isinstance(record, dict):
            raise SupplyCatalogDataError("supply provider records must be dictionaries")
        record_smiles = record.get("smiles") or smiles
        if not isinstance(record_smiles, str) or not record_smiles:
            raise SupplyCatalogDataError("supply provider response smiles is required")
        normalized = _normalize_supply_record(record, record_smiles)
        return _availability_response(normalized, identity)


class SupplyOracleGrpcServicer(supply_pb2_grpc.SupplyOracleServiceServicer):
    def __init__(self, service: SupplyOracleServicer | None = None):
        self.service = service or SupplyOracleServicer()

    async def CheckAvailability(self, request, context):
        try:
            response = await self.service.CheckAvailability(request, context)
            return _availability_proto(response)
        except Exception as exc:
            return await _abort_grpc(context, exc)

    async def BatchCheck(self, request, context):
        try:
            response = await self.service.BatchCheck(request, context)
            payload = {
                "results": [_availability_proto(item) for item in response.results],
                "total_elapsed_ms": response.total_elapsed_ms,
                "request_id": response.request_id,
                "project_id": response.project_id,
                "candidate_id": response.candidate_id,
                "canonical_smiles": response.canonical_smiles,
            }
            if response.candidate_index is not None:
                payload["candidate_index"] = response.candidate_index
            return supply_pb2.BatchAvailabilityResponse(**payload)
        except Exception as exc:
            return await _abort_grpc(context, exc)

    async def GetCatalogPrice(self, request, context):
        try:
            response = await self.service.GetCatalogPrice(request, context)
            return _availability_proto(response)
        except Exception as exc:
            return await _abort_grpc(context, exc)


async def _abort_grpc(context, error: Exception):
    if isinstance(error, SupplyRequestError):
        code = grpc.StatusCode.INVALID_ARGUMENT
    elif isinstance(error, SupplyNotFoundError):
        code = grpc.StatusCode.NOT_FOUND
    elif isinstance(error, SupplyUnavailableError):
        code = grpc.StatusCode.FAILED_PRECONDITION
    elif isinstance(error, TimeoutError | asyncio.TimeoutError):
        code = grpc.StatusCode.DEADLINE_EXCEEDED
    elif isinstance(error, SupplyCatalogDataError):
        code = grpc.StatusCode.DATA_LOSS
    else:
        code = grpc.StatusCode.INTERNAL
    if context is not None and hasattr(context, "abort"):
        await context.abort(code, str(error))
    raise error


def _availability_proto(response) -> supply_pb2.AvailabilityResponse:
    payload = {
        "smiles": str(getattr(response, "smiles", "")),
        "available": bool(getattr(response, "available", False)),
        "catalog_id": str(getattr(response, "catalog_id", "") or ""),
        "catalog_source": str(getattr(response, "catalog_source", "") or ""),
        "source_timestamp": str(getattr(response, "source_timestamp", "") or ""),
        "currency": str(getattr(response, "currency", "") or ""),
        "evidence_id": str(getattr(response, "evidence_id", "") or ""),
        "catalog_version": str(getattr(response, "catalog_version", "") or ""),
        "catalog_checksum": str(getattr(response, "catalog_checksum", "") or ""),
        "request_id": str(getattr(response, "request_id", "") or ""),
        "project_id": str(getattr(response, "project_id", "") or ""),
        "candidate_id": str(getattr(response, "candidate_id", "") or ""),
        "canonical_smiles": str(getattr(response, "canonical_smiles", "") or ""),
    }
    price = getattr(response, "price", None)
    if price is not None:
        payload["price"] = float(price)
    lead_time_days = getattr(response, "lead_time_days", None)
    if lead_time_days is not None:
        payload["lead_time_days"] = int(lead_time_days)
    candidate_index = getattr(response, "candidate_index", None)
    if candidate_index is not None:
        payload["candidate_index"] = int(candidate_index)
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
    _LOGGER.info("Supply Oracle Service running on :50059")
    await server.wait_for_termination()


def _validation_catalog_payload() -> dict:
    source_timestamp = "2000-01-01T00:00:00Z"
    records = []
    for smiles, catalog_id in (
        ("C", "validation-carbon"),
        ("CO", "validation-methanol"),
        ("CN", "validation-methylamine"),
        ("CCO", "validation-ethanol"),
        ("CCN", "validation-ethylamine"),
    ):
        records.append(
            {
                "smiles": smiles,
                "available": True,
                "catalog_id": catalog_id,
                "source": _VALIDATION_MARKER,
                "source_timestamp": source_timestamp,
                "price": 1.0,
                "currency": "USD",
                "lead_time_days": 1,
                "validation_marker": _VALIDATION_MARKER,
            }
        )
    return {
        "catalog_version": _VALIDATION_MARKER,
        "validation_marker": _VALIDATION_MARKER,
        "records": records,
    }


def _is_validation_catalog(path: Path) -> bool:
    if not path.is_file() or path.is_symlink():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    if (
        not isinstance(payload, dict)
        or payload.get("catalog_version") != _VALIDATION_MARKER
        or payload.get("validation_marker") != _VALIDATION_MARKER
        or not isinstance(payload.get("records"), list)
    ):
        return False
    records = payload["records"]
    if not records or any(
        not isinstance(record, dict)
        or record.get("source") != _VALIDATION_MARKER
        or record.get("validation_marker") != _VALIDATION_MARKER
        for record in records
    ):
        return False
    _load_json_catalog(path)
    return True


def _bootstrap_validation_catalog(path: str | os.PathLike[str]) -> Path:
    _require_synthetic_validation_enabled()
    target = Path(path).expanduser().resolve()
    if target.exists():
        if _is_validation_catalog(target):
            return target
        raise FileExistsError(f"Supply validation catalog already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    temporary_path = Path(temporary_name)
    linked = False
    try:
        os.fchmod(descriptor, 0o644)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                _validation_catalog_payload(),
                handle,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        _load_json_catalog(temporary_path)
        try:
            os.link(temporary_path, target)
        except FileExistsError as exc:
            if _is_validation_catalog(target):
                return target
            raise FileExistsError(
                f"Supply validation catalog already exists: {target}"
            ) from exc
        linked = True
    finally:
        temporary_path.unlink(missing_ok=True)
        if linked:
            directory_descriptor = os.open(target.parent, os.O_RDONLY)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
    return target


def _require_synthetic_validation_enabled() -> None:
    if os.environ.get(_VALIDATION_GATE_ENV) != "true":
        raise RuntimeError(f"{_VALIDATION_GATE_ENV}=true is required")


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments:
        asyncio.run(serve())
        return 0
    if len(arguments) != 2 or arguments[0] != "--bootstrap-validation-catalog":
        sys.stderr.write(
            "Supply Oracle service has unexpected command line arguments\n"
        )
        return 2
    try:
        target = _bootstrap_validation_catalog(arguments[1])
    except (RuntimeError, OSError) as exc:
        sys.stderr.write(f"{exc}\n")
        return 1
    except ValueError as exc:
        sys.stderr.write(f"{exc}\n")
        return 2
    json.dump(
        {
            "catalog_uri": target.as_uri(),
            "validation_marker": _VALIDATION_MARKER,
        },
        sys.stdout,
        separators=(",", ":"),
        sort_keys=True,
    )
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
