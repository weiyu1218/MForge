"""Integration tests for DKI Neo4j layer."""

from __future__ import annotations

import os

import pytest

try:
    from mf_core.db.repositories.graph_repo import GraphRepository
    from neo4j import AsyncGraphDatabase

    _HAS_NEO4J = True
except ImportError:
    _HAS_NEO4J = False

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


@pytest.fixture
async def neo4j_driver():
    if not _HAS_NEO4J:
        pytest.skip("neo4j driver is not installed")
    uri = os.environ.get("NEO4J_URI")
    user = os.environ.get("NEO4J_USER")
    password = os.environ.get("NEO4J_PASSWORD")
    if not uri or not user or not password:
        pytest.skip("NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD are required")
    driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
    try:
        async with driver.session() as session:
            await session.run("RETURN 1")
        yield driver
    finally:
        await driver.close()


async def test_write_and_query_transforms_to(neo4j_driver) -> None:
    repo = GraphRepository(neo4j_driver)
    await repo.write_transforms_to("IK-A", "IK-B", "MMPT", 0.8)

    async with neo4j_driver.session() as session:
        result = await session.run(
            "MATCH (:Molecule {inchikey: $a})-[r:TRANSFORMS_TO]->"
            "(:Molecule {inchikey: $b}) RETURN r.via AS via, r.confidence AS confidence",
            a="IK-A",
            b="IK-B",
        )
        row = await result.single()

    assert row["via"] == "MMPT"
    assert row["confidence"] == 0.8


async def test_write_and_query_fto(neo4j_driver) -> None:
    repo = GraphRepository(neo4j_driver)
    await repo.write_covered_by("IK-FTO", "PAT-1", claim_id="1", similarity=0.9)

    results = await repo.query_fto("IK-FTO", threshold=0.8)

    assert results[0]["patent_id"] == "PAT-1"
    assert results[0]["claim_id"] == "1"


async def test_write_produced_and_audit(neo4j_driver) -> None:
    repo = GraphRepository(neo4j_driver)
    await repo.write_produced("run-1", "IK-P", "validation_agent", "2026-05-15")
    await repo.write_has_belief("IK-P", "belief-1", "rdkit_l0", 0.8, 0.05)

    async with neo4j_driver.session() as session:
        result = await session.run(
            "MATCH (:Run {id: $run_id})-[:PRODUCED]->(m:Molecule)"
            "-[:HAS_BELIEF]->(b:Belief) "
            "RETURN m.inchikey AS inchikey, b.oracle AS oracle",
            run_id="run-1",
        )
        row = await result.single()

    assert row["inchikey"] == "IK-P"
    assert row["oracle"] == "rdkit_l0"
