"""Unit tests for GraphRepository (Mock Neo4j session)."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.fixture
def mock_driver():
    driver = MagicMock()
    driver.close = AsyncMock()
    mock_session = AsyncMock()
    mock_session.run = AsyncMock()
    cm = AsyncMock()
    cm.__aenter__ = AsyncMock(return_value=mock_session)
    cm.__aexit__ = AsyncMock(return_value=False)
    driver.session.return_value = cm
    return driver


@pytest.mark.unit
@pytest.mark.asyncio
async def test_graph_repository_health_check_verifies_driver_connectivity() -> None:
    from mf_core.db.repositories.graph_repo import GraphRepository

    class Driver:
        def __init__(self) -> None:
            self.connectivity_checks = 0

        async def verify_connectivity(self) -> None:
            self.connectivity_checks += 1

    driver = Driver()
    repository = GraphRepository(driver)
    health_check = getattr(repository, "health_check", None)

    assert callable(health_check)
    assert await health_check() == {"healthy": True}
    assert driver.connectivity_checks == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_graph_repository_health_check_reports_connectivity_failure() -> None:
    from mf_core.db.repositories.graph_repo import GraphRepository

    class Driver:
        async def verify_connectivity(self) -> None:
            raise RuntimeError("neo4j unavailable")

    repository = GraphRepository(Driver())
    health_check = getattr(repository, "health_check", None)

    assert callable(health_check)
    assert await health_check() == {"healthy": False}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_graph_repository_close_is_idempotent_after_success() -> None:
    from mf_core.db.repositories.graph_repo import GraphRepository

    class Driver:
        def __init__(self) -> None:
            self.close_calls = 0

        async def close(self) -> None:
            self.close_calls += 1

    driver = Driver()
    repository = GraphRepository(driver)
    close = getattr(repository, "close", None)

    assert callable(close)
    await close()
    await close()

    assert driver.close_calls == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_graph_repository_concurrent_close_closes_driver_once() -> None:
    from mf_core.db.repositories.graph_repo import GraphRepository

    class Driver:
        def __init__(self) -> None:
            self.close_calls = 0
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def close(self) -> None:
            self.close_calls += 1
            self.started.set()
            await self.release.wait()

    driver = Driver()
    repository = GraphRepository(driver)
    close = getattr(repository, "close", None)

    assert callable(close)
    first = asyncio.create_task(close())
    await asyncio.wait_for(driver.started.wait(), timeout=0.1)
    second = asyncio.create_task(close())
    await asyncio.sleep(0)

    assert driver.close_calls == 1

    driver.release.set()
    await asyncio.gather(first, second)

    assert driver.close_calls == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_graph_repository_close_retries_after_failure() -> None:
    from mf_core.db.repositories.graph_repo import GraphRepository

    class Driver:
        def __init__(self) -> None:
            self.close_calls = 0

        async def close(self) -> None:
            self.close_calls += 1
            if self.close_calls == 1:
                raise RuntimeError("neo4j close failed")

    driver = Driver()
    repository = GraphRepository(driver)
    close = getattr(repository, "close", None)

    assert callable(close)
    with pytest.raises(RuntimeError, match="neo4j close failed"):
        await close()

    await close()
    await close()

    assert driver.close_calls == 2


@pytest.mark.unit
@pytest.mark.asyncio
async def test_graph_repository_close_retries_after_cancellation() -> None:
    from mf_core.db.repositories.graph_repo import GraphRepository

    class Driver:
        def __init__(self) -> None:
            self.close_calls = 0
            self.started = asyncio.Event()

        async def close(self) -> None:
            self.close_calls += 1
            if self.close_calls == 1:
                self.started.set()
                await asyncio.Event().wait()

    driver = Driver()
    repository = GraphRepository(driver)
    close = getattr(repository, "close", None)

    assert callable(close)
    first = asyncio.create_task(close())
    await asyncio.wait_for(driver.started.wait(), timeout=0.1)
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first

    await close()
    await close()

    assert driver.close_calls == 2


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
    mock_driver.close.assert_awaited_once_with()


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
    mock_driver.close.assert_awaited_once_with()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_merge_agent_beliefs_closes_repository_after_read_failure(
    mock_driver,
) -> None:
    from unittest.mock import patch

    from mf_core.db.repositories.graph_repo import GraphRepository
    from orchestrator_svc.main import _merge_agent_beliefs_into_crg

    mock_session = mock_driver.session.return_value.__aenter__.return_value
    mock_session.run.side_effect = RuntimeError("neo4j read failed")
    repository = GraphRepository(mock_driver)
    final_state = {
        "crg": {
            "beliefs": [{"id": "belief-existing"}],
            "edges": [],
            "version": 1,
        }
    }

    with patch(
        "orchestrator_svc.main.build_shared_crg_repository_from_env",
        return_value=repository,
    ):
        merged = await _merge_agent_beliefs_into_crg(final_state, "run-1")

    assert merged == final_state["crg"]
    mock_driver.close.assert_awaited_once_with()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_merge_agent_beliefs_closes_repository_when_cancelled() -> None:
    from unittest.mock import patch

    from mf_core.db.repositories.graph_repo import GraphRepository
    from orchestrator_svc.main import _merge_agent_beliefs_into_crg

    class Session:
        def __init__(self) -> None:
            self.read_started = asyncio.Event()

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_value, traceback) -> bool:
            return False

        async def run(self, query: str, **parameters):
            self.read_started.set()
            await asyncio.Event().wait()

    class Driver:
        def __init__(self) -> None:
            self.session_instance = Session()
            self.close_calls = 0

        def session(self) -> Session:
            return self.session_instance

        async def close(self) -> None:
            self.close_calls += 1

    driver = Driver()
    repository = GraphRepository(driver)
    with patch(
        "orchestrator_svc.main.build_shared_crg_repository_from_env",
        return_value=repository,
    ):
        task = asyncio.create_task(
            _merge_agent_beliefs_into_crg(
                {"crg": {"beliefs": [], "edges": [], "version": 0}},
                "run-1",
            )
        )
        await asyncio.wait_for(driver.session_instance.read_started.wait(), timeout=0.1)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

    assert driver.close_calls == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_merge_agent_beliefs_preserves_read_fallback_when_close_fails() -> None:
    from unittest.mock import patch

    from orchestrator_svc.main import _merge_agent_beliefs_into_crg

    class Repository:
        def __init__(self) -> None:
            self.close_calls = 0

        async def get_run_crg(self, run_id: str) -> dict:
            raise RuntimeError("neo4j read failed")

        async def close(self) -> None:
            self.close_calls += 1
            raise RuntimeError("neo4j close failed")

    repository = Repository()
    final_state = {
        "crg": {
            "beliefs": [{"id": "belief-existing"}],
            "edges": [],
            "version": 1,
        }
    }
    with patch(
        "orchestrator_svc.main.build_shared_crg_repository_from_env",
        return_value=repository,
    ):
        merged = await _merge_agent_beliefs_into_crg(final_state, "run-1")

    assert merged == final_state["crg"]
    assert repository.close_calls == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_merge_agent_beliefs_preserves_cancellation_when_close_fails() -> None:
    from unittest.mock import patch

    from orchestrator_svc.main import _merge_agent_beliefs_into_crg

    class Repository:
        def __init__(self) -> None:
            self.read_started = asyncio.Event()
            self.close_calls = 0

        async def get_run_crg(self, run_id: str) -> dict:
            self.read_started.set()
            await asyncio.Event().wait()

        async def close(self) -> None:
            self.close_calls += 1
            raise RuntimeError("neo4j close failed")

    repository = Repository()
    with patch(
        "orchestrator_svc.main.build_shared_crg_repository_from_env",
        return_value=repository,
    ):
        task = asyncio.create_task(
            _merge_agent_beliefs_into_crg(
                {"crg": {"beliefs": [], "edges": [], "version": 0}},
                "run-1",
            )
        )
        await asyncio.wait_for(repository.read_started.wait(), timeout=0.1)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

    assert repository.close_calls == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_merge_agent_beliefs_finishes_close_when_cancelled_during_cleanup() -> None:
    from unittest.mock import patch

    from orchestrator_svc.main import _merge_agent_beliefs_into_crg

    class Repository:
        def __init__(self) -> None:
            self.close_started = asyncio.Event()
            self.close_release = asyncio.Event()
            self.close_cancelled = False
            self.close_finished = False

        async def get_run_crg(self, run_id: str) -> dict:
            return {"beliefs": [], "edges": []}

        async def close(self) -> None:
            self.close_started.set()
            try:
                await self.close_release.wait()
            except asyncio.CancelledError:
                self.close_cancelled = True
                raise
            self.close_finished = True

    repository = Repository()
    with patch(
        "orchestrator_svc.main.build_shared_crg_repository_from_env",
        return_value=repository,
    ):
        task = asyncio.create_task(
            _merge_agent_beliefs_into_crg(
                {"crg": {"beliefs": [], "edges": [], "version": 0}},
                "run-1",
            )
        )
        await asyncio.wait_for(repository.close_started.wait(), timeout=0.1)
        task.cancel()
        await asyncio.sleep(0)

        assert repository.close_cancelled is False

        repository.close_release.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert repository.close_finished is True


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
