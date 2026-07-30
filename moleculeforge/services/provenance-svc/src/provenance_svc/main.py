"""Provenance Service - gRPC + REST server for audit trail and provenance tracking.

Provides full NL-to-SSP (Natural Language to Scientific Software Provenance)
traceability via Neo4j graph with Sigstore cryptographic signatures
for each artifact in the molecular design pipeline.
"""

import asyncio
import base64
import binascii
import hashlib
import hmac
import json
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

from provenance_svc.domain.sigstore_integration import SigstoreIntegration

# FastAPI REST app
rest_app = FastAPI(title="Provenance Service", version="0.1.0")


class ProvenanceRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    artifact_type: str
    artifact_id: str
    payload_base64: str
    parent_ids: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)

    @field_validator("artifact_type", "artifact_id")
    @classmethod
    def _validate_identity(cls, value: str) -> str:
        if not value or value != value.strip():
            raise ValueError("must be a non-empty trimmed string")
        return value

    @field_validator("parent_ids")
    @classmethod
    def _validate_parent_ids(cls, values: list[str]) -> list[str]:
        if any(not value or value != value.strip() for value in values):
            raise ValueError("parent_ids must contain non-empty trimmed strings")
        if len(values) != len(set(values)):
            raise ValueError("parent_ids must not contain duplicates")
        return values

    @field_validator("metadata")
    @classmethod
    def _validate_metadata(cls, value: dict) -> dict:
        try:
            _canonical_json_bytes(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("metadata must be strict JSON") from exc
        return value

    @field_validator("payload_base64")
    @classmethod
    def _validate_payload_base64(cls, value: str) -> str:
        _decode_payload_base64(value)
        return value


class VerifyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    artifact_id: str
    signature: str


sigstore = SigstoreIntegration()


def _require_internal_service_token(
    token: Annotated[
        str | None,
        Header(alias="X-MoleculeForge-Service-Token"),
    ] = None,
) -> None:
    expected = os.environ.get("INTERNAL_SERVICE_TOKEN", "")
    if not expected:
        return
    if token is None or not hmac.compare_digest(
        token.encode("utf-8"),
        expected.encode("utf-8"),
    ):
        raise HTTPException(status_code=401, detail="Invalid internal service token")


_INTERNAL_SERVICE_DEPENDENCIES = [Depends(_require_internal_service_token)]


class InMemoryProvenanceStore:
    store_name = "in_memory"

    def __init__(self) -> None:
        self.records: dict[str, dict] = {}

    async def find_record(self, artifact_id: str) -> dict | None:
        return self.records.get(artifact_id)

    async def record(self, record: ProvenanceRecord, signed: dict, recorded_at: str) -> dict:
        stored = _stored_record(record, signed, recorded_at)
        existing = self.records.get(record.artifact_id)
        if existing is not None:
            if _immutable_record_content(existing) != _immutable_record_content(stored):
                raise HTTPException(
                    status_code=409,
                    detail=(
                        f"artifact_id already exists with different content: {record.artifact_id}"
                    ),
                )
            return existing
        self.records[record.artifact_id] = stored
        return stored

    async def get_record(self, artifact_id: str) -> dict:
        record = self.records.get(artifact_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"Unknown artifact_id: {artifact_id}")
        return record

    async def get_chain(self, artifact_id: str) -> list[dict]:
        if artifact_id not in self.records:
            raise HTTPException(status_code=404, detail=f"Unknown artifact_id: {artifact_id}")
        visited: set[str] = set()
        ordered: list[dict] = []

        def visit(current_id: str) -> None:
            if current_id in visited:
                return
            record = self.records.get(current_id)
            if record is None:
                raise HTTPException(status_code=404, detail=f"Unknown artifact_id: {current_id}")
            visited.add(current_id)
            for parent_id in record["parent_ids"]:
                visit(parent_id)
            ordered.append(record)

        visit(artifact_id)
        return ordered

    async def audit(self, project_id: str) -> list[dict]:
        return [
            record
            for record in self.records.values()
            if record.get("metadata", {}).get("project_id") == project_id
        ]

    async def child_count(self, artifact_id: str) -> int:
        return sum(
            1 for record in self.records.values() if artifact_id in record.get("parent_ids", [])
        )

    async def count(self) -> int:
        return len(self.records)


class ProductionProvenanceStore:
    store_name = "production_real"

    def __init__(self, graph_repo: Any, audit_writer: Any, object_store: Any) -> None:
        self.graph_repo = graph_repo
        self.audit_writer = audit_writer
        self.object_store = object_store
        self._write_through = InMemoryProvenanceStore()

    async def initialize(self) -> None:
        ensure_bucket = getattr(self.object_store, "ensure_bucket", None)
        if callable(ensure_bucket):
            await ensure_bucket()
        ensure_schema = getattr(self.audit_writer, "_ensure_schema", None)
        if callable(ensure_schema):
            await ensure_schema()

    async def record(self, record: ProvenanceRecord, signed: dict, recorded_at: str) -> dict:
        if signed.get("signature_type") != "sigstore_rekor":
            raise HTTPException(
                status_code=503,
                detail=(
                    "production_real requires sigstore_rekor; "
                    f"received {signed.get('signature_type') or 'unconfigured'}"
                ),
            )
        ensure_bucket = getattr(self.object_store, "ensure_bucket", None)
        if callable(ensure_bucket):
            await ensure_bucket()
        object_name = f"provenance/{record.artifact_id}.json"
        object_exists = getattr(self.object_store, "object_exists", None)
        if callable(object_exists) and await object_exists(object_name):
            existing = await self._read_record_object(record.artifact_id)
            proposed = _stored_record(record, signed, recorded_at)
            if _immutable_record_content(existing) != _immutable_record_content(proposed):
                raise HTTPException(
                    status_code=409,
                    detail=(
                        f"artifact_id already exists with different content: {record.artifact_id}"
                    ),
                )
            if not await _verify_stored_record(existing):
                raise HTTPException(
                    status_code=500,
                    detail=f"persisted provenance record failed verification: {record.artifact_id}",
                )
            self._write_through.records[record.artifact_id] = existing
            await self.reconcile_record(existing)
            return existing
        stored = await self._write_through.record(record, signed, recorded_at)
        encoded = json.dumps(stored, sort_keys=True).encode("utf-8")
        created = await self.object_store.put_object_if_absent(
            object_name,
            encoded,
            content_type="application/json",
        )
        if not created:
            existing = await self._read_record_object(record.artifact_id)
            if _immutable_record_content(existing) != _immutable_record_content(stored):
                raise HTTPException(
                    status_code=409,
                    detail=(
                        f"artifact_id already exists with different content: {record.artifact_id}"
                    ),
                )
            if not await _verify_stored_record(existing):
                raise HTTPException(
                    status_code=500,
                    detail=f"persisted provenance record failed verification: {record.artifact_id}",
                )
            self._write_through.records[record.artifact_id] = existing
            await self.reconcile_record(existing)
            return existing
        await self.reconcile_record(stored)
        return stored

    async def reconcile_record(self, stored: dict) -> None:
        project_id = str(stored.get("metadata", {}).get("project_id", ""))
        run_id = str(stored.get("metadata", {}).get("run_id", ""))
        trace_id = str(stored.get("metadata", {}).get("trace_id", ""))
        await _write_artifact_to_graph(
            self.graph_repo,
            artifact_id=stored["artifact_id"],
            artifact_type=stored["artifact_type"],
            project_id=project_id,
            run_id=run_id,
            trace_id=trace_id,
            recorded_at=str(stored["recorded_at"]),
            signature_type=str(stored.get("signature_type") or ""),
        )
        for parent_id in stored["parent_ids"]:
            await _write_artifact_parent_to_graph(
                self.graph_repo,
                parent_id,
                stored["artifact_id"],
            )
        await _write_crg_to_graph(
            self.graph_repo,
            stored.get("metadata", {}).get("crg"),
            project_id=project_id,
            run_id=run_id,
        )
        await self.audit_writer.write_event(stored)

    async def find_record(self, artifact_id: str) -> dict | None:
        ensure_bucket = getattr(self.object_store, "ensure_bucket", None)
        if callable(ensure_bucket):
            await ensure_bucket()
        object_name = f"provenance/{artifact_id}.json"
        object_exists = getattr(self.object_store, "object_exists", None)
        if callable(object_exists):
            if not await object_exists(object_name):
                return None
            return await self._read_record_object(artifact_id)
        try:
            return await self._read_record_object(artifact_id)
        except (FileNotFoundError, KeyError):
            return None

    async def get_chain(self, artifact_id: str) -> list[dict]:
        if hasattr(self.graph_repo, "get_artifact_chain_ids"):
            artifact_ids = await self.graph_repo.get_artifact_chain_ids(artifact_id)
            if not artifact_ids:
                raise HTTPException(status_code=404, detail=f"Unknown artifact_id: {artifact_id}")
            return [await self._read_record_object(current_id) for current_id in artifact_ids]
        return await self._write_through.get_chain(artifact_id)

    async def audit(self, project_id: str) -> list[dict]:
        if hasattr(self.audit_writer, "read_project"):
            return await self.audit_writer.read_project(project_id)
        return await self._write_through.audit(project_id)

    async def child_count(self, artifact_id: str) -> int:
        if hasattr(self.graph_repo, "count_artifact_children"):
            return await self.graph_repo.count_artifact_children(artifact_id)
        return await self._write_through.child_count(artifact_id)

    async def count(self) -> int:
        if hasattr(self.graph_repo, "count_artifacts"):
            return await self.graph_repo.count_artifacts()
        if hasattr(self.audit_writer, "count_events"):
            return await self.audit_writer.count_events()
        return await self._write_through.count()

    async def _read_record_object(self, artifact_id: str) -> dict:
        data = await self.object_store.get_object(f"provenance/{artifact_id}.json")
        return json.loads(data.decode("utf-8"))

    async def get_record(self, artifact_id: str) -> dict:
        object_exists = getattr(self.object_store, "object_exists", None)
        object_name = f"provenance/{artifact_id}.json"
        if callable(object_exists) and not await object_exists(object_name):
            raise HTTPException(
                status_code=404,
                detail=f"Unknown artifact_id: {artifact_id}",
            )
        try:
            return await self._read_record_object(artifact_id)
        except (FileNotFoundError, KeyError) as exc:
            raise HTTPException(
                status_code=404,
                detail=f"Unknown artifact_id: {artifact_id}",
            ) from exc


class PostgresAuditWriter:
    def __init__(self, database_url: str) -> None:
        from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

        self.engine = create_async_engine(database_url)
        self.sessionmaker = async_sessionmaker(self.engine, expire_on_commit=False)
        self._schema_ready = False

    async def _ensure_schema(self) -> None:
        from sqlalchemy import text

        if not self._schema_ready:
            async with self.engine.begin() as conn:
                await conn.execute(
                    text(
                        """
                        CREATE TABLE IF NOT EXISTS audit_events (
                            id VARCHAR(36) PRIMARY KEY,
                            project_id VARCHAR(128) NOT NULL,
                            run_id VARCHAR(128) NOT NULL,
                            trace_id VARCHAR(128) NOT NULL,
                            artifact_id VARCHAR(256) NOT NULL,
                            event_type VARCHAR(128) NOT NULL,
                            payload JSONB NOT NULL,
                            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                        )
                        """
                    )
                )
                await conn.execute(
                    text(
                        """
                        ALTER TABLE audit_events
                        ADD COLUMN IF NOT EXISTS run_id VARCHAR(128) NOT NULL DEFAULT ''
                        """
                    )
                )
                await conn.execute(
                    text(
                        """
                        CREATE UNIQUE INDEX IF NOT EXISTS
                        audit_events_artifact_id_unique
                        ON audit_events (artifact_id)
                        """
                    )
                )
            self._schema_ready = True

    async def write_event(self, stored: dict) -> None:
        from sqlalchemy import text

        await self._ensure_schema()
        async with self.sessionmaker() as session:
            result = await session.execute(
                text(
                    """
                    INSERT INTO audit_events
                    (id, project_id, run_id, trace_id, artifact_id, event_type, payload)
                    VALUES (
                        :id,
                        :project_id,
                        :run_id,
                        :trace_id,
                        :artifact_id,
                        :event_type,
                        CAST(:payload AS JSONB)
                    )
                    ON CONFLICT (artifact_id) DO UPDATE
                    SET artifact_id = EXCLUDED.artifact_id
                    WHERE audit_events.payload = EXCLUDED.payload
                    RETURNING audit_events.artifact_id
                    """
                ),
                {
                    "id": str(uuid.uuid4()),
                    "project_id": stored.get("metadata", {}).get("project_id", ""),
                    "run_id": stored.get("metadata", {}).get("run_id", ""),
                    "trace_id": stored.get("metadata", {}).get("trace_id", ""),
                    "artifact_id": stored["artifact_id"],
                    "event_type": "provenance.record",
                    "payload": json.dumps(stored, sort_keys=True),
                },
            )
            if result.scalar_one_or_none() is None:
                raise RuntimeError(
                    "audit event artifact_id already exists with different content: "
                    f"{stored['artifact_id']}"
                )
            await session.commit()

    async def read_project(self, project_id: str) -> list[dict]:
        from sqlalchemy import text

        await self._ensure_schema()
        async with self.sessionmaker() as session:
            result = await session.execute(
                text(
                    """
                    SELECT payload
                    FROM audit_events
                    WHERE project_id = :project_id
                    ORDER BY created_at, id
                    """
                ),
                {"project_id": project_id},
            )
            rows = result.fetchall()
        records: list[dict] = []
        for row in rows:
            payload = row[0]
            records.append(json.loads(payload) if isinstance(payload, str) else dict(payload))
        return records

    async def count_events(self) -> int:
        from sqlalchemy import text

        await self._ensure_schema()
        async with self.sessionmaker() as session:
            result = await session.execute(text("SELECT count(*) FROM audit_events"))
            count = result.scalar_one()
        return int(count)


_IN_MEMORY_STORE = InMemoryProvenanceStore()


@rest_app.get("/health")
async def health():
    store = _get_store()
    initialize = getattr(store, "initialize", None)
    if callable(initialize):
        await initialize()
    return {
        "status": "healthy",
        "components": {
            "provenance_store": store.store_name,
            "signature_type": sigstore.signature_type,
        },
        "records": await store.count() if hasattr(store, "count") else 0,
    }


@rest_app.post(
    "/v1/provenance/record",
    dependencies=_INTERNAL_SERVICE_DEPENDENCIES,
)
async def create_record(record: ProvenanceRecord):
    """Record a provenance entry linking inputs to outputs."""
    raw_payload = _decode_payload_base64(record.payload_base64)
    checksum = f"sha256:{hashlib.sha256(raw_payload).hexdigest()}"
    store = _get_store()
    find_record = getattr(store, "find_record", None)
    if callable(find_record):
        existing = await find_record(record.artifact_id)
        if existing is not None:
            if _immutable_record_content(existing) != _record_content(record, checksum):
                raise HTTPException(
                    status_code=409,
                    detail=(
                        f"artifact_id already exists with different content: {record.artifact_id}"
                    ),
                )
            if (
                getattr(store, "store_name", "") == "production_real"
                and existing.get("signature_type") != "sigstore_rekor"
            ):
                raise HTTPException(
                    status_code=503,
                    detail=(
                        "production_real requires sigstore_rekor; "
                        f"received {existing.get('signature_type') or 'unconfigured'}"
                    ),
                )
            if not await _verify_stored_record(existing):
                raise HTTPException(
                    status_code=500,
                    detail=(
                        f"persisted provenance record failed verification: {record.artifact_id}"
                    ),
                )
            reconcile_record = getattr(store, "reconcile_record", None)
            if callable(reconcile_record):
                await reconcile_record(existing)
            return _create_record_response(existing)
        for parent_id in record.parent_ids:
            if await find_record(parent_id) is None:
                raise HTTPException(
                    status_code=409,
                    detail=f"Unknown parent artifact_id: {parent_id}",
                )
    recorded_at = datetime.now(UTC).isoformat()
    signed = await asyncio.to_thread(
        sigstore.sign_artifact,
        record.artifact_id,
        record.artifact_type,
        record.metadata,
        checksum=checksum,
        parent_ids=record.parent_ids,
        recorded_at=recorded_at,
    )
    stored = await store.record(record, signed, recorded_at)
    return _create_record_response(stored)


def _create_record_response(stored: dict) -> dict:
    return {
        "artifact_id": stored["artifact_id"],
        "artifact_type": stored["artifact_type"],
        "parent_ids": list(stored["parent_ids"]),
        "signature": stored["signature"],
        "certificate": stored.get("certificate"),
        "checksum": stored["checksum"],
        "payload_hash": stored["payload_hash"],
        "signature_type": stored["signature_type"],
        "recorded_at": stored["recorded_at"],
    }


@rest_app.get(
    "/v1/provenance/record/{artifact_id}",
    dependencies=_INTERNAL_SERVICE_DEPENDENCIES,
)
async def get_provenance_record(artifact_id: str) -> dict:
    record = await _get_store().get_record(artifact_id)
    return {
        "artifact_id": record["artifact_id"],
        "artifact_type": record["artifact_type"],
        "parent_ids": list(record["parent_ids"]),
        "metadata": dict(record["metadata"]),
        "payload_base64": record["payload_base64"],
        "checksum": record["checksum"],
        "signature": record["signature"],
        "signature_type": record["signature_type"],
        "recorded_at": record["recorded_at"],
        "verified": await _verify_stored_record(record),
    }


@rest_app.get(
    "/v1/provenance/{artifact_id}",
    dependencies=_INTERNAL_SERVICE_DEPENDENCIES,
)
async def get_provenance(artifact_id: str):
    """Retrieve provenance chain for an artifact from Neo4j."""
    chain = await _get_store().get_chain(artifact_id)
    verification = await asyncio.gather(*(_verify_stored_record(record) for record in chain))
    return {
        "artifact_id": artifact_id,
        "chain": [
            {
                "step": index,
                "artifact_type": record["artifact_type"],
                "artifact_id": record["artifact_id"],
                "timestamp": record["recorded_at"],
                "parent_ids": record["parent_ids"],
                "signature_type": record["signature_type"],
                "checksum": record["checksum"],
                "verified": verification[index],
            }
            for index, record in enumerate(chain)
        ],
        "leaf_artifacts": [],
        "verified": all(verification),
    }


@rest_app.post(
    "/v1/provenance/verify",
    dependencies=_INTERNAL_SERVICE_DEPENDENCIES,
)
async def verify_provenance(req: VerifyRequest):
    """Verify a Sigstore signature for an artifact."""
    record = await _get_store().get_record(req.artifact_id)
    is_valid = await _verify_stored_record(record, signature=req.signature)
    return {
        "artifact_id": req.artifact_id,
        "signature_valid": is_valid,
        "verified_at": datetime.now(UTC).isoformat(),
    }


@rest_app.get(
    "/v1/provenance/audit/{project_id}",
    dependencies=_INTERNAL_SERVICE_DEPENDENCIES,
)
async def audit_project(project_id: str):
    """Full audit trail for a project."""
    store = _get_store()
    artifacts = await store.audit(project_id)
    verification = await asyncio.gather(*(_verify_stored_record(record) for record in artifacts))
    verified_count = sum(verification)
    return {
        "project_id": project_id,
        "total_artifacts": len(artifacts),
        "verified_count": verified_count,
        "unverified_count": len(artifacts) - verified_count,
        "artifacts": [
            {
                "artifact_id": record["artifact_id"],
                "artifact_type": record["artifact_type"],
                "verified": verification[index],
                "children": await store.child_count(record["artifact_id"]),
            }
            for index, record in enumerate(artifacts)
        ],
    }


def _stored_record(record: ProvenanceRecord, signed: dict, recorded_at: str) -> dict:
    signature_bundle = dict(signed)
    signed_payload = signature_bundle.pop("signed_payload")
    if signed_payload.get("recorded_at") != recorded_at:
        raise ValueError("signed payload recorded_at does not match stored record")
    return {
        "artifact_id": record.artifact_id,
        "artifact_type": record.artifact_type,
        "parent_ids": list(record.parent_ids),
        "metadata": dict(record.metadata),
        "payload_base64": record.payload_base64,
        "checksum": signed_payload["checksum"],
        "signed_payload": signed_payload,
        "payload_hash": signature_bundle["payload_hash"],
        "signature_bundle": signature_bundle,
        "signature": signed["signature"],
        "certificate": signed.get("certificate"),
        "recorded_at": recorded_at,
        "signature_type": signed.get("signature_type"),
    }


async def _verify_stored_record(
    record: dict,
    *,
    signature: str | None = None,
) -> bool:
    return await asyncio.to_thread(
        sigstore.verify_record,
        record,
        signature,
    )


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _decode_payload_base64(value: str) -> bytes:
    if not isinstance(value, str) or not value:
        raise ValueError("payload_base64 must be a non-empty base64 string")
    try:
        payload = base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("payload_base64 must be valid base64") from exc
    if not payload:
        raise ValueError("payload_base64 must decode to non-empty bytes")
    if base64.b64encode(payload).decode("ascii") != value:
        raise ValueError("payload_base64 must use canonical base64 encoding")
    return payload


def _immutable_record_content(record: dict) -> dict:
    return {
        "artifact_id": record.get("artifact_id"),
        "artifact_type": record.get("artifact_type"),
        "parent_ids": record.get("parent_ids"),
        "metadata": record.get("metadata"),
        "payload_base64": record.get("payload_base64"),
        "checksum": record.get("checksum"),
    }


def _record_content(record: ProvenanceRecord, checksum: str) -> dict:
    return {
        "artifact_id": record.artifact_id,
        "artifact_type": record.artifact_type,
        "parent_ids": list(record.parent_ids),
        "metadata": dict(record.metadata),
        "payload_base64": record.payload_base64,
        "checksum": checksum,
    }


async def _write_artifact_to_graph(
    graph_repo: Any,
    artifact_id: str,
    artifact_type: str,
    project_id: str,
    run_id: str,
    trace_id: str,
    recorded_at: str,
    signature_type: str,
) -> None:
    if hasattr(graph_repo, "write_artifact"):
        await graph_repo.write_artifact(
            artifact_id=artifact_id,
            artifact_type=artifact_type,
            project_id=project_id,
            run_id=run_id,
            trace_id=trace_id,
            recorded_at=recorded_at,
            signature_type=signature_type,
        )
        return
    query = (
        "MERGE (a:Artifact {id: $artifact_id}) "
        "SET a.type = $artifact_type, "
        "a.project_id = $project_id, "
        "a.run_id = $run_id, "
        "a.trace_id = $trace_id, "
        "a.recorded_at = $recorded_at, "
        "a.signature_type = $signature_type "
        "FOREACH (_ IN CASE WHEN $run_id = '' THEN [] ELSE [1] END | "
        "MERGE (r:Run {id: $run_id}) "
        "MERGE (r)-[:PRODUCED_ARTIFACT]->(a))"
    )
    async with graph_repo.driver.session() as session:
        await session.run(
            query,
            artifact_id=artifact_id,
            artifact_type=artifact_type,
            project_id=project_id,
            run_id=run_id,
            trace_id=trace_id,
            recorded_at=recorded_at,
            signature_type=signature_type,
        )


async def _write_artifact_parent_to_graph(
    graph_repo: Any,
    parent_id: str,
    child_id: str,
) -> None:
    if hasattr(graph_repo, "write_artifact_parent"):
        await graph_repo.write_artifact_parent(parent_id, child_id)
        return
    query = (
        "MERGE (p:Artifact {id: $parent_id}) "
        "MERGE (c:Artifact {id: $child_id}) "
        "MERGE (p)-[:PARENT_OF]->(c)"
    )
    async with graph_repo.driver.session() as session:
        await session.run(query, parent_id=parent_id, child_id=child_id)


async def _write_crg_to_graph(
    graph_repo: Any,
    crg: Any,
    *,
    project_id: str,
    run_id: str,
) -> None:
    if not isinstance(crg, dict):
        return

    write_workflow_belief = getattr(graph_repo, "write_workflow_belief", None)
    if callable(write_workflow_belief):
        crg_project_id = str(crg.get("project_id") or project_id)
        for belief in crg.get("beliefs", []) or []:
            if not isinstance(belief, dict):
                continue
            belief_id = str(belief.get("id") or belief.get("belief_id") or "")
            if not belief_id:
                continue
            subject = str(belief.get("subject") or "")
            await write_workflow_belief(
                project_id=crg_project_id,
                run_id=str(belief.get("run_id") or run_id or subject),
                belief_id=belief_id,
                subject=subject,
                predicate=str(belief.get("predicate") or ""),
                object_value=str(belief.get("object") or ""),
                confidence=float(belief.get("confidence", 0.0)),
                source_agent=str(belief.get("source_agent") or ""),
                timestamp_ns=int(belief.get("timestamp_ns", 0)),
                evidence_ids=list(belief.get("evidence_ids", []) or []),
            )

    write_crg_edge = getattr(graph_repo, "write_crg_edge", None)
    if callable(write_crg_edge):
        for edge in crg.get("edges", []) or []:
            if not isinstance(edge, dict):
                continue
            source_belief_id = str(edge.get("source_belief_id") or "")
            target_belief_id = str(edge.get("target_belief_id") or "")
            if not source_belief_id or not target_belief_id:
                continue
            await write_crg_edge(
                source_belief_id=source_belief_id,
                target_belief_id=target_belief_id,
                relation=str(edge.get("relation") or ""),
                weight=float(edge.get("weight", 0.0)),
            )


def _get_store():
    configured = getattr(rest_app.state, "provenance_store", None)
    if configured is not None:
        return configured
    mode = os.environ.get("PROVENANCE_STORE_MODE", "local_demo")
    if mode not in {"local_demo", "production_real"}:
        raise HTTPException(
            status_code=500,
            detail=f"Unsupported PROVENANCE_STORE_MODE: {mode}",
        )
    if mode == "local_demo":
        return _IN_MEMORY_STORE

    missing = _missing_production_config()
    if missing:
        raise HTTPException(
            status_code=503,
            detail={
                "provenance_store": "production_real",
                "missing_config": missing,
            },
        )
    production_store = getattr(rest_app.state, "_production_store", None)
    if production_store is None:
        production_store = _build_production_store()
        rest_app.state._production_store = production_store
    return production_store


def _missing_production_config() -> list[str]:
    missing = [
        name for name in ("NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD") if not os.environ.get(name)
    ]
    if not (os.environ.get("PROVENANCE_DATABASE_URL") or os.environ.get("TEST_DATABASE_URL")):
        missing.append("PROVENANCE_DATABASE_URL or TEST_DATABASE_URL")
    missing.extend(
        name
        for name in (
            "MINIO_ENDPOINT_URL",
            "MINIO_ACCESS_KEY",
            "MINIO_SECRET_KEY",
            "MINIO_BUCKET",
            "SIGSTORE_SIGN_COMMAND",
            "SIGSTORE_VERIFY_COMMAND",
            "SIGSTORE_EXPECTED_IDENTITY",
            "SIGSTORE_REKOR_URL",
        )
        if not os.environ.get(name)
    )
    if not _sigstore_identity_configured():
        missing.append("SIGSTORE_IDENTITY_TOKEN")
    return missing


def _sigstore_identity_configured() -> bool:
    if os.environ.get("SIGSTORE_IDENTITY_TOKEN", "").strip():
        return True
    token_file = os.environ.get("SIGSTORE_IDENTITY_TOKEN_FILE", "").strip()
    if not token_file:
        return False
    path = Path(token_file)
    return path.is_file() and bool(path.read_text(encoding="utf-8").strip())


def _build_production_store() -> ProductionProvenanceStore:
    from mf_core.db.minio_client import MinIOStorageClient
    from mf_core.db.repositories.graph_repo import GraphRepository
    from neo4j import AsyncGraphDatabase

    driver = AsyncGraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"]),
    )
    graph_repo = GraphRepository(driver)
    audit_writer = PostgresAuditWriter(
        os.environ.get("PROVENANCE_DATABASE_URL") or os.environ["TEST_DATABASE_URL"]
    )
    object_store = MinIOStorageClient(
        endpoint_url=os.environ["MINIO_ENDPOINT_URL"],
        access_key=os.environ["MINIO_ACCESS_KEY"],
        secret_key=os.environ["MINIO_SECRET_KEY"],
        bucket=os.environ["MINIO_BUCKET"],
    )
    return ProductionProvenanceStore(graph_repo, audit_writer, object_store)


def create_app():
    """Entry point for production ASGI servers."""
    return rest_app


if __name__ == "__main__":
    import uvicorn

    async def main():
        config = uvicorn.Config(rest_app, host="0.0.0.0", port=8010, log_level="info")
        server = uvicorn.Server(config)
        await server.serve()

    asyncio.run(main())
