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
async def test_write_workflow_belief(mock_driver) -> None:
    from mf_core.db.repositories.graph_repo import GraphRepository

    repo = GraphRepository(mock_driver)
    await repo.write_workflow_belief(
        project_id="project-1",
        run_id="run-1",
        belief_id="belief-1",
        subject="run-1",
        predicate="workflow_stage",
        object_value="PLANNING",
        confidence=1.0,
        source_agent="orchestrator",
        timestamp_ns=123,
        evidence_ids=["artifact-input"],
    )

    mock_session = mock_driver.session.return_value.__aenter__.return_value
    kwargs = mock_session.run.await_args.kwargs
    assert kwargs["project_id"] == "project-1"
    assert kwargs["run_id"] == "run-1"
    assert kwargs["belief_id"] == "belief-1"
    assert kwargs["evidence_ids"] == ["artifact-input"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_write_crg_edge(mock_driver) -> None:
    from mf_core.db.repositories.graph_repo import GraphRepository

    repo = GraphRepository(mock_driver)
    await repo.write_crg_edge(
        source_belief_id="belief-1",
        target_belief_id="belief-2",
        relation="derives_from",
        weight=1.0,
    )

    mock_session = mock_driver.session.return_value.__aenter__.return_value
    kwargs = mock_session.run.await_args.kwargs
    assert kwargs["source_belief_id"] == "belief-1"
    assert kwargs["target_belief_id"] == "belief-2"
    assert kwargs["relation"] == "derives_from"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_run_crg_reads_workflow_beliefs_and_edges(mock_driver) -> None:
    from mf_core.db.repositories.graph_repo import GraphRepository

    mock_result = AsyncMock()
    mock_result.single = AsyncMock(
        return_value={
            "project_id": "project-1",
            "beliefs": [
                {
                    "id": "belief-1",
                    "subject": "run-1",
                    "predicate": "workflow_stage",
                    "object": "PLANNING",
                    "confidence": 1.0,
                    "source_agent": "orchestrator",
                    "timestamp_ns": 123,
                    "evidence_ids": ["artifact-input"],
                },
                None,
            ],
            "edges": [
                {
                    "source_belief_id": "belief-1",
                    "target_belief_id": "belief-2",
                    "relation": "derives_from",
                    "weight": 1.0,
                },
                {"source_belief_id": None, "target_belief_id": None},
            ],
        }
    )
    mock_session = mock_driver.session.return_value.__aenter__.return_value
    mock_session.run.return_value = mock_result

    repo = GraphRepository(mock_driver)
    crg = await repo.get_run_crg("run-1")

    assert crg == {
        "project_id": "project-1",
        "beliefs": [
            {
                "id": "belief-1",
                "subject": "run-1",
                "predicate": "workflow_stage",
                "object": "PLANNING",
                "confidence": 1.0,
                "source_agent": "orchestrator",
                "timestamp_ns": 123,
                "evidence_ids": ["artifact-input"],
            }
        ],
        "edges": [
            {
                "source_belief_id": "belief-1",
                "target_belief_id": "belief-2",
                "relation": "derives_from",
                "weight": 1.0,
            }
        ],
        "version": 2,
    }
    assert mock_session.run.await_args.kwargs["run_id"] == "run-1"


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


@pytest.mark.unit
@pytest.mark.asyncio
async def test_merge_agent_beliefs_merges_shared_crg_into_final_state(
    mock_driver, monkeypatch: pytest.MonkeyPatch
) -> None:
    from unittest.mock import AsyncMock, patch

    from mf_core.db.repositories.graph_repo import GraphRepository

    mock_result = AsyncMock()
    mock_result.single = AsyncMock(
        return_value={
            "project_id": "project-1",
            "beliefs": [
                {
                    "id": "belief-agent-1",
                    "subject": "run-1",
                    "predicate": "validation_status",
                    "object": "validated",
                    "confidence": 0.9,
                    "source_agent": "validation_agent",
                    "timestamp_ns": 456,
                    "evidence_ids": [],
                }
            ],
            "edges": [],
        }
    )
    mock_session = mock_driver.session.return_value.__aenter__.return_value
    mock_session.run.return_value = mock_result

    repo = GraphRepository(mock_driver)

    with patch(
        "orchestrator_svc.main.build_shared_crg_repository_from_env",
        return_value=repo,
    ):
        from orchestrator_svc.main import _merge_agent_beliefs_into_crg

        final_state = {
            "run_id": "run-1",
            "crg": {
                "beliefs": [
                    {
                        "id": "belief-orch-1",
                        "subject": "run-1",
                        "predicate": "workflow_status",
                        "object": "completed",
                        "confidence": 1.0,
                        "source_agent": "orchestrator",
                        "timestamp_ns": 123,
                        "evidence_ids": [],
                    }
                ],
                "edges": [],
                "version": 1,
            },
        }
        merged = await _merge_agent_beliefs_into_crg(final_state, "run-1")

    assert len(merged["beliefs"]) == 2
    assert merged["version"] == 2
    belief_ids = {b["id"] for b in merged["beliefs"]}
    assert "belief-orch-1" in belief_ids
    assert "belief-agent-1" in belief_ids


@pytest.mark.unit
@pytest.mark.asyncio
async def test_merge_agent_beliefs_deduplicates_existing_beliefs(
    mock_driver,
) -> None:
    from unittest.mock import patch

    from mf_core.db.repositories.graph_repo import GraphRepository

    mock_result = AsyncMock()
    mock_result.single = AsyncMock(
        return_value={
            "project_id": "project-1",
            "beliefs": [
                {
                    "id": "belief-1",
                    "subject": "run-1",
                    "predicate": "workflow_status",
                    "object": "completed",
                    "confidence": 1.0,
                    "source_agent": "orchestrator",
                    "timestamp_ns": 123,
                    "evidence_ids": [],
                }
            ],
            "edges": [],
        }
    )
    mock_session = mock_driver.session.return_value.__aenter__.return_value
    mock_session.run.return_value = mock_result

    repo = GraphRepository(mock_driver)

    with patch(
        "orchestrator_svc.main.build_shared_crg_repository_from_env",
        return_value=repo,
    ):
        from orchestrator_svc.main import _merge_agent_beliefs_into_crg

        final_state = {
            "run_id": "run-1",
            "crg": {
                "beliefs": [
                    {
                        "id": "belief-1",
                        "subject": "run-1",
                        "predicate": "workflow_status",
                        "object": "completed",
                        "confidence": 1.0,
                        "source_agent": "orchestrator",
                        "timestamp_ns": 123,
                        "evidence_ids": [],
                    }
                ],
                "edges": [],
                "version": 1,
            },
        }
        merged = await _merge_agent_beliefs_into_crg(final_state, "run-1")

    assert len(merged["beliefs"]) == 1
    assert merged["version"] == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_merge_agent_beliefs_falls_through_when_no_repository() -> None:
    from unittest.mock import patch

    from orchestrator_svc.main import _merge_agent_beliefs_into_crg

    with patch(
        "orchestrator_svc.main.build_shared_crg_repository_from_env",
        return_value=None,
    ):
        final_state = {
            "run_id": "run-1",
            "crg": {"beliefs": [{"id": "b1"}], "edges": [], "version": 1},
        }
        merged = await _merge_agent_beliefs_into_crg(final_state, "run-1")

    assert merged == final_state["crg"]
