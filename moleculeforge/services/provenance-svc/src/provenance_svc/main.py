"""Provenance Service - gRPC + REST server for audit trail and provenance tracking.

Provides full NL-to-SSP (Natural Language to Scientific Software Provenance)
traceability via Neo4j graph with Sigstore cryptographic signatures
for each artifact in the molecular design pipeline.
"""
import asyncio
import json
import os
import uuid
from concurrent import futures
from datetime import UTC, datetime
from typing import Any

import grpc
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from provenance_svc.domain.sigstore_integration import SigstoreIntegration

# FastAPI REST app
rest_app = FastAPI(title="Provenance Service", version="0.1.0")


class ProvenanceRecord(BaseModel):
    artifact_type: str
    artifact_id: str
    parent_ids: list[str] = []
    metadata: dict = {}


class VerifyRequest(BaseModel):
    artifact_id: str
    signature: str


sigstore = SigstoreIntegration()


class InMemoryProvenanceStore:
    store_name = "in_memory"

    def __init__(self) -> None:
        self.records: dict[str, dict] = {}

    async def record(self, record: ProvenanceRecord, signed: dict, recorded_at: str) -> dict:
        stored = _stored_record(record, signed, recorded_at)
        self.records[record.artifact_id] = stored
        return stored

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
            1
            for record in self.records.values()
            if artifact_id in record.get("parent_ids", [])
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

    async def record(self, record: ProvenanceRecord, signed: dict, recorded_at: str) -> dict:
        stored = await self._write_through.record(record, signed, recorded_at)
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
            recorded_at=recorded_at,
            signature_type=str(stored.get("signature_type") or ""),
        )
        for parent_id in stored["parent_ids"]:
            await _write_artifact_parent_to_graph(
                self.graph_repo,
                parent_id,
                stored["artifact_id"],
            )
        await self.audit_writer.write_event(stored)
        await self.object_store.put_object(
            f"provenance/{stored['artifact_id']}.json",
            json.dumps(stored, sort_keys=True).encode("utf-8"),
            content_type="application/json",
        )
        return stored

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
            self._schema_ready = True

    async def write_event(self, stored: dict) -> None:
        from sqlalchemy import text

        await self._ensure_schema()
        async with self.sessionmaker() as session:
            await session.execute(
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
    return {
        "status": "healthy",
        "components": {
            "provenance_store": store.store_name,
            "signature_type": sigstore.signature_type,
        },
        "records": await store.count() if hasattr(store, "count") else 0,
    }


@rest_app.post("/v1/provenance/record")
async def create_record(record: ProvenanceRecord):
    """Record a provenance entry linking inputs to outputs."""
    recorded_at = datetime.now(UTC).isoformat()
    signed = sigstore.sign_artifact(
        record.artifact_id, record.artifact_type, record.metadata
    )
    stored = await _get_store().record(record, signed, recorded_at)
    return {
        "artifact_id": record.artifact_id,
        "artifact_type": record.artifact_type,
        "parent_ids": record.parent_ids,
        "signature": stored["signature"],
        "certificate": stored.get("certificate"),
        "recorded_at": recorded_at,
    }


@rest_app.get("/v1/provenance/{artifact_id}")
async def get_provenance(artifact_id: str):
    """Retrieve provenance chain for an artifact from Neo4j."""
    chain = await _get_store().get_chain(artifact_id)
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
            }
            for index, record in enumerate(chain)
        ],
        "leaf_artifacts": [],
        "verified": all(bool(record.get("signature")) for record in chain),
    }


@rest_app.post("/v1/provenance/verify")
async def verify_provenance(req: VerifyRequest):
    """Verify a Sigstore signature for an artifact."""
    is_valid = sigstore.verify_signature(req.artifact_id, req.signature)
    return {
        "artifact_id": req.artifact_id,
        "signature_valid": is_valid,
        "verified_at": datetime.now(UTC).isoformat(),
    }


@rest_app.get("/v1/provenance/audit/{project_id}")
async def audit_project(project_id: str):
    """Full audit trail for a project."""
    store = _get_store()
    artifacts = await store.audit(project_id)
    verified = [record for record in artifacts if record.get("signature")]
    return {
        "project_id": project_id,
        "total_artifacts": len(artifacts),
        "verified_count": len(verified),
        "unverified_count": len(artifacts) - len(verified),
        "artifacts": [
            {
                "artifact_id": record["artifact_id"],
                "artifact_type": record["artifact_type"],
                "verified": bool(record.get("signature")),
                "children": await store.child_count(record["artifact_id"]),
            }
            for record in artifacts
        ],
    }


# gRPC Servicer
class ProvenanceServicer:
    async def RecordArtifact(self, request, context):
        """gRPC endpoint for recording provenance."""
        artifact_id = getattr(request, "artifact_id", "")
        artifact_type = getattr(request, "artifact_type", "")
        metadata = getattr(request, "metadata", {})
        response = await create_record(
            ProvenanceRecord(
                artifact_id=artifact_id,
                artifact_type=artifact_type,
                metadata=metadata,
            )
        )
        return type(
            "ProvenanceResponse",
            (),
            {
                "artifact_id": artifact_id,
                "signature": response["signature"],
                "recorded": True,
            },
        )()

    async def GetChain(self, request, context):
        """gRPC endpoint for retrieving provenance chain."""
        artifact_id = getattr(request, "artifact_id", "")
        chain = await get_provenance(artifact_id)
        return type(
            "ChainResponse",
            (),
            {
                "artifact_id": artifact_id,
                "chain_json": str(chain["chain"]),
                "depth": len(chain["chain"]),
            },
        )()


def _stored_record(record: ProvenanceRecord, signed: dict, recorded_at: str) -> dict:
    return {
        "artifact_id": record.artifact_id,
        "artifact_type": record.artifact_type,
        "parent_ids": list(record.parent_ids),
        "metadata": dict(record.metadata),
        "signature": signed["signature"],
        "certificate": signed.get("certificate"),
        "recorded_at": recorded_at,
        "signature_type": signed.get("signature_type"),
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
        name
        for name in ("NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD")
        if not os.environ.get(name)
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
        )
        if not os.environ.get(name)
    )
    return missing


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


async def serve_grpc():
    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=10))
    server.add_insecure_port("[::]:50060")
    await server.start()
    print("Provenance gRPC Service running on :50060")
    await server.wait_for_termination()


def create_app():
    """Entry point for production ASGI servers."""
    return rest_app


if __name__ == "__main__":
    import uvicorn

    async def main():
        asyncio.create_task(serve_grpc())
        config = uvicorn.Config(rest_app, host="0.0.0.0", port=8010, log_level="info")
        server = uvicorn.Server(config)
        await server.serve()

    asyncio.run(main())
