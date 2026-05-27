"""Unit tests for GraphRepository (Mock Neo4j session)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.fixture
def mock_driver():
    driver = MagicMock()
    mock_session = AsyncMock()
    mock_session.run = AsyncMock()
    cm = AsyncMock()
    cm.__aenter__ = AsyncMock(return_value=mock_session)
    cm.__aexit__ = AsyncMock(return_value=False)
    driver.session.return_value = cm
    return driver


@pytest.mark.unit
@pytest.mark.asyncio
async def test_write_transforms_to(mock_driver) -> None:
    from mf_core.db.repositories.graph_repo import GraphRepository

    repo = GraphRepository(mock_driver)
    await repo.write_transforms_to(
        from_inchikey="AAA",
        to_inchikey="BBB",
        via="MMPT",
        confidence=0.95,
    )

    mock_driver.session.assert_called_once()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_write_binds_to(mock_driver) -> None:
    from mf_core.db.repositories.graph_repo import GraphRepository

    repo = GraphRepository(mock_driver)
    await repo.write_binds_to(
        inchikey="AAA",
        uniprot_id="P12345",
        source="PDBbind",
        affinity=-9.5,
        method="IC50",
    )

    mock_driver.session.assert_called_once()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_query_fto(mock_driver) -> None:
    from mf_core.db.repositories.graph_repo import GraphRepository

    # Mock result data
    mock_result = AsyncMock()
    mock_result.data = AsyncMock(
        return_value=[
            {"patent_id": "US123456", "claim_id": "claim_1", "similarity": 0.85}
        ]
    )
    mock_session = mock_driver.session.return_value.__aenter__.return_value
    mock_session.run.return_value = mock_result

    repo = GraphRepository(mock_driver)
    records = await repo.query_fto("AAA", threshold=0.6)

    assert len(records) == 1
    assert records[0]["patent_id"] == "US123456"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_write_covered_by(mock_driver) -> None:
    from mf_core.db.repositories.graph_repo import GraphRepository

    repo = GraphRepository(mock_driver)
    await repo.write_covered_by(
        inchikey="AAA",
        patent_id="US123456",
        claim_id="claim_1",
        similarity=0.85,
    )

    mock_driver.session.assert_called_once()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_write_produced(mock_driver) -> None:
    from mf_core.db.repositories.graph_repo import GraphRepository

    repo = GraphRepository(mock_driver)
    await repo.write_produced(
        run_id="run-001",
        inchikey="AAA",
        agent="generator",
        timestamp="2026-05-04T12:00:00Z",
    )

    mock_driver.session.assert_called_once()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_write_has_belief(mock_driver) -> None:
    from mf_core.db.repositories.graph_repo import GraphRepository

    repo = GraphRepository(mock_driver)
    await repo.write_has_belief(
        inchikey="AAA",
        belief_id="belief-001",
        oracle="rdkit_oracle_l0",
        value=0.8,
        uncertainty=0.1,
        created_at="2026-05-04T12:00:00Z",
    )

    mock_driver.session.assert_called_once()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_write_artifact_includes_run_and_trace(mock_driver) -> None:
    from mf_core.db.repositories.graph_repo import GraphRepository

    repo = GraphRepository(mock_driver)
    await repo.write_artifact(
        artifact_id="artifact-1",
        artifact_type="candidate",
        project_id="project-1",
        run_id="run-1",
        trace_id="trace-1",
        recorded_at="2026-05-19T00:00:00Z",
        signature_type="local_dev_signature",
    )

    mock_session = mock_driver.session.return_value.__aenter__.return_value
    kwargs = mock_session.run.await_args.kwargs
    assert kwargs["artifact_id"] == "artifact-1"
    assert kwargs["run_id"] == "run-1"
    assert kwargs["trace_id"] == "trace-1"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_write_artifact_parent(mock_driver) -> None:
    from mf_core.db.repositories.graph_repo import GraphRepository

    repo = GraphRepository(mock_driver)
    await repo.write_artifact_parent("artifact-parent", "artifact-child")

    mock_session = mock_driver.session.return_value.__aenter__.return_value
    kwargs = mock_session.run.await_args.kwargs
    assert kwargs["parent_id"] == "artifact-parent"
    assert kwargs["child_id"] == "artifact-child"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_read_artifact_chain_ids_from_graph(mock_driver) -> None:
    from mf_core.db.repositories.graph_repo import GraphRepository

    mock_result = AsyncMock()
    mock_result.data = AsyncMock(
        return_value=[
            {"artifact_id": "artifact-parent", "recorded_at": "2026-05-19T00:00:00Z"},
            {"artifact_id": "artifact-child", "recorded_at": "2026-05-19T00:01:00Z"},
        ]
    )
    mock_session = mock_driver.session.return_value.__aenter__.return_value
    mock_session.run.return_value = mock_result

    repo = GraphRepository(mock_driver)
    chain_ids = await repo.get_artifact_chain_ids("artifact-child")

    assert chain_ids == ["artifact-parent", "artifact-child"]
    assert mock_session.run.await_args.kwargs["artifact_id"] == "artifact-child"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_count_artifact_children_from_graph(mock_driver) -> None:
    from mf_core.db.repositories.graph_repo import GraphRepository

    mock_result = AsyncMock()
    mock_result.single = AsyncMock(return_value={"children": 2})
    mock_session = mock_driver.session.return_value.__aenter__.return_value
    mock_session.run.return_value = mock_result

    repo = GraphRepository(mock_driver)
    children = await repo.count_artifact_children("artifact-parent")

    assert children == 2
    assert mock_session.run.await_args.kwargs["artifact_id"] == "artifact-parent"
