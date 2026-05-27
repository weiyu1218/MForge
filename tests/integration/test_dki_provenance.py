"""Integration tests for production provenance persistence on DKI."""

from __future__ import annotations

import os
from uuid import uuid4

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


def _missing_env() -> list[str]:
    missing = [
        name
        for name in (
            "NEO4J_URI",
            "NEO4J_USER",
            "NEO4J_PASSWORD",
            "MINIO_ENDPOINT_URL",
            "MINIO_ACCESS_KEY",
            "MINIO_SECRET_KEY",
            "MINIO_BUCKET",
        )
        if not os.environ.get(name)
    ]
    if not (os.environ.get("PROVENANCE_DATABASE_URL") or os.environ.get("TEST_DATABASE_URL")):
        missing.append("PROVENANCE_DATABASE_URL or TEST_DATABASE_URL")
    return missing


async def test_production_provenance_store_round_trips_real_dki() -> None:
    missing = _missing_env()
    if missing:
        pytest.skip(", ".join(missing) + " required for provenance DKI integration tests")

    from mf_core.db.minio_client import MinIOStorageClient
    from mf_core.db.repositories.graph_repo import GraphRepository
    from neo4j import AsyncGraphDatabase
    from provenance_svc.main import (
        PostgresAuditWriter,
        ProductionProvenanceStore,
        ProvenanceRecord,
    )

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
    store = ProductionProvenanceStore(graph_repo, audit_writer, object_store)

    suffix = uuid4().hex
    project_id = f"project-{suffix}"
    run_id = f"run-{suffix}"
    trace_id = f"trace-{suffix}"
    parent_id = f"artifact-parent-{suffix}"
    child_id = f"artifact-child-{suffix}"
    metadata = {"project_id": project_id, "run_id": run_id, "trace_id": trace_id}

    try:
        await store.record(
            ProvenanceRecord(
                artifact_type="nl_query",
                artifact_id=parent_id,
                metadata=metadata,
            ),
            {
                "signature": f"sig-{parent_id}",
                "certificate": None,
                "signature_type": "local_dev_signature",
            },
            "2026-05-19T00:00:00Z",
        )
        await store.record(
            ProvenanceRecord(
                artifact_type="candidate",
                artifact_id=child_id,
                parent_ids=[parent_id],
                metadata=metadata,
            ),
            {
                "signature": f"sig-{child_id}",
                "certificate": None,
                "signature_type": "local_dev_signature",
            },
            "2026-05-19T00:01:00Z",
        )

        chain = await store.get_chain(child_id)
        audit = await store.audit(project_id)
        child_count = await store.child_count(parent_id)

        assert [record["artifact_id"] for record in chain] == [parent_id, child_id]
        assert {record["artifact_id"] for record in audit} == {parent_id, child_id}
        assert child_count == 1
        assert await object_store.object_exists(f"provenance/{child_id}.json") is True
    finally:
        await driver.close()
        await audit_writer.engine.dispose()
