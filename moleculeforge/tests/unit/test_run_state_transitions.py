from __future__ import annotations

import asyncio
import sqlite3
import threading
from collections.abc import AsyncGenerator
from pathlib import Path

import httpx
import pytest
from mf_core.db.store import RunAlreadyExistsError, RunStatus, RunStore
from orchestrator_svc import main as orchestrator_main
from orchestrator_svc.main import RunControl


async def _create_run(
    store: RunStore,
    run_id: str,
    *,
    created_at: str = "2026-07-27T10:00:00+00:00",
) -> None:
    await store.create_run(
        run_id,
        intent=f"intent for {run_id}",
        policy={"workflow_scope": "engineering"},
        created_at=created_at,
    )


async def _wait_for_thread_event(event: threading.Event) -> None:
    assert await asyncio.to_thread(event.wait, 5)


@pytest.mark.parametrize(
    ("source", "target"),
    [
        (RunStatus.QUEUED, RunStatus.RUNNING),
        (RunStatus.QUEUED, RunStatus.REJECTED),
        (RunStatus.RUNNING, RunStatus.PAUSED),
        (RunStatus.RUNNING, RunStatus.AWAITING_EVIDENCE),
        (RunStatus.RUNNING, RunStatus.COMPLETED),
        (RunStatus.RUNNING, RunStatus.FAILED),
        (RunStatus.PAUSED, RunStatus.RUNNING),
        (RunStatus.AWAITING_EVIDENCE, RunStatus.RUNNING),
    ],
)
async def test_legal_run_state_transitions_are_persisted(
    tmp_path: Path,
    source: RunStatus,
    target: RunStatus,
) -> None:
    store = RunStore(tmp_path / f"{source}-{target}.db")
    await store.initialize()
    await _create_run(store, "run-1")
    if source is not RunStatus.QUEUED:
        await store.transition_run(
            "run-1",
            {RunStatus.QUEUED},
            RunStatus.RUNNING,
            current_stage="setup",
        )
    if source not in {RunStatus.QUEUED, RunStatus.RUNNING}:
        await store.transition_run(
            "run-1",
            {RunStatus.RUNNING},
            source,
            current_stage="setup",
        )

    await store.transition_run(
        "run-1",
        {source},
        target,
        current_stage="test-stage",
        state={"cursor": 3},
    )

    snapshot = await store.get_run("run-1")
    assert snapshot is not None
    assert snapshot["status"] == target.value
    assert snapshot["current_stage"] == "test-stage"
    assert snapshot["state"] == {"cursor": 3}


@pytest.mark.parametrize(
    "terminal",
    [
        RunStatus.COMPLETED,
        RunStatus.REJECTED,
        RunStatus.FAILED,
        RunStatus.INTERRUPTED,
    ],
)
async def test_terminal_runs_reject_further_transitions(
    tmp_path: Path,
    terminal: RunStatus,
) -> None:
    store = RunStore(tmp_path / f"{terminal}.db")
    await store.initialize()
    await _create_run(store, "run-terminal")
    if terminal is RunStatus.INTERRUPTED:
        await store.interrupt_active_runs()
    else:
        await store.transition_run(
            "run-terminal",
            {RunStatus.QUEUED},
            RunStatus.RUNNING,
            current_stage="planning",
        )
        await store.transition_run(
            "run-terminal",
            {RunStatus.RUNNING},
            terminal,
            current_stage="finished",
        )

    with pytest.raises(ValueError, match="illegal run transition"):
        await store.transition_run(
            "run-terminal",
            {terminal},
            RunStatus.RUNNING,
            current_stage="restarted",
        )


async def test_compare_and_update_rejects_stale_expected_status(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runs.db")
    await store.initialize()
    await _create_run(store, "run-stale")

    with pytest.raises(ValueError, match="expected"):
        await store.transition_run(
            "run-stale",
            {RunStatus.RUNNING},
            RunStatus.COMPLETED,
            current_stage="finished",
        )


async def test_cancelled_failure_compensation_replaces_matching_failure(
    tmp_path: Path,
) -> None:
    store = RunStore(tmp_path / "runs.db")
    await store.initialize()
    await _create_run(store, "run-matching-failure")
    await store.transition_run(
        "run-matching-failure",
        {RunStatus.QUEUED},
        RunStatus.RUNNING,
        current_stage="planning",
    )
    await store.transition_run(
        "run-matching-failure",
        {RunStatus.RUNNING},
        RunStatus.FAILED,
        current_stage="generation",
        error_type="RuntimeError",
        error_message="workflow failed",
    )
    compensate = getattr(store, "compensate_cancelled_failure", None)
    assert callable(compensate)

    changed = await compensate(
        "run-matching-failure",
        expected_error_type="RuntimeError",
        expected_error_message="workflow failed",
        cancellation_message="request cancelled",
    )

    snapshot = await store.get_run("run-matching-failure")
    assert changed is True
    assert snapshot is not None
    assert snapshot["status"] == RunStatus.INTERRUPTED.value
    assert snapshot["current_stage"] == "generation"
    assert snapshot["error_type"] == "CancelledError"
    assert snapshot["error_message"] == "request cancelled"


@pytest.mark.parametrize(
    "active",
    [
        RunStatus.QUEUED,
        RunStatus.RUNNING,
        RunStatus.PAUSED,
        RunStatus.AWAITING_EVIDENCE,
    ],
)
async def test_cancelled_failure_compensation_interrupts_nonterminal_run(
    tmp_path: Path,
    active: RunStatus,
) -> None:
    store = RunStore(tmp_path / "runs.db")
    await store.initialize()
    await _create_run(store, "run-active")
    if active is not RunStatus.QUEUED:
        await store.transition_run(
            "run-active",
            {RunStatus.QUEUED},
            RunStatus.RUNNING,
            current_stage=RunStatus.RUNNING.value,
        )
    if active not in {RunStatus.QUEUED, RunStatus.RUNNING}:
        await store.transition_run(
            "run-active",
            {RunStatus.RUNNING},
            active,
            current_stage=active.value,
        )
    compensate = getattr(store, "compensate_cancelled_failure", None)
    assert callable(compensate)

    changed = await compensate(
        "run-active",
        expected_error_type="RuntimeError",
        expected_error_message="workflow failed",
        cancellation_message="request cancelled",
    )

    snapshot = await store.get_run("run-active")
    assert changed is True
    assert snapshot is not None
    assert snapshot["status"] == RunStatus.INTERRUPTED.value
    assert snapshot["current_stage"] == active.value
    assert snapshot["error_type"] == "CancelledError"
    assert snapshot["error_message"] == "request cancelled"


@pytest.mark.parametrize(
    "terminal",
    [
        RunStatus.FAILED,
        RunStatus.COMPLETED,
        RunStatus.REJECTED,
        RunStatus.INTERRUPTED,
    ],
)
async def test_cancelled_failure_compensation_preserves_unrelated_terminal_state(
    tmp_path: Path,
    terminal: RunStatus,
) -> None:
    store = RunStore(tmp_path / f"{terminal.value}.db")
    await store.initialize()
    await _create_run(store, "run-terminal-compensation")
    await store.transition_run(
        "run-terminal-compensation",
        {RunStatus.QUEUED},
        RunStatus.RUNNING,
        current_stage="planning",
    )
    await store.transition_run(
        "run-terminal-compensation",
        {RunStatus.RUNNING},
        terminal,
        current_stage="finished",
        error_type="OtherError" if terminal is RunStatus.FAILED else None,
        error_message="another owner failed" if terminal is RunStatus.FAILED else None,
    )
    compensate = getattr(store, "compensate_cancelled_failure", None)
    assert callable(compensate)

    changed = await compensate(
        "run-terminal-compensation",
        expected_error_type="RuntimeError",
        expected_error_message="workflow failed",
        cancellation_message="request cancelled",
    )

    snapshot = await store.get_run("run-terminal-compensation")
    assert changed is False
    assert snapshot is not None
    assert snapshot["status"] == terminal.value
    if terminal is RunStatus.FAILED:
        assert snapshot["error_type"] == "OtherError"
        assert snapshot["error_message"] == "another owner failed"


async def test_upsert_does_not_delete_existing_events(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runs.db")
    await store.initialize()
    await _create_run(store, "run-upsert")
    await store.append_event(
        "run-upsert",
        0,
        stage="planning",
        payload={"message": "started"},
        timestamp="2026-07-27T10:00:01+00:00",
    )

    await store.create_run(
        "run-upsert",
        intent="updated intent",
        policy={"workflow_scope": "full"},
        created_at="2026-07-27T10:00:00+00:00",
    )

    events = await store.list_events("run-upsert")
    assert [(event["step_index"], event["payload"]) for event in events] == [
        (0, {"message": "started"})
    ]


async def test_concurrent_run_claim_allows_only_one_creator(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runs.db")
    await store.initialize()
    start = asyncio.Event()

    async def claim(intent: str) -> None:
        await start.wait()
        await store.create_run(
            "run-claimed",
            intent=intent,
            policy={"workflow_scope": "engineering"},
            created_at="2026-07-27T10:00:00+00:00",
            require_new=True,
        )

    first = asyncio.create_task(claim("first"))
    second = asyncio.create_task(claim("second"))
    start.set()
    results = await asyncio.gather(first, second, return_exceptions=True)

    assert sum(result is None for result in results) == 1
    errors = [result for result in results if isinstance(result, Exception)]
    assert len(errors) == 1
    assert isinstance(errors[0], ValueError)
    assert "run_id already exists" in str(errors[0])


async def test_event_step_index_is_unique_per_run(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runs.db")
    await store.initialize()
    await _create_run(store, "run-events")
    await store.append_event(
        "run-events",
        0,
        stage="planning",
        payload={},
        timestamp="2026-07-27T10:00:01+00:00",
    )

    with pytest.raises(sqlite3.IntegrityError):
        await store.append_event(
            "run-events",
            0,
            stage="planning-again",
            payload={},
            timestamp="2026-07-27T10:00:02+00:00",
        )


async def test_store_restart_preserves_snapshot_and_events(tmp_path: Path) -> None:
    db_file = tmp_path / "runs.db"
    first = RunStore(db_file)
    await first.initialize()
    await _create_run(first, "run-durable")
    await first.transition_run(
        "run-durable",
        {RunStatus.QUEUED},
        RunStatus.RUNNING,
        current_stage="planning",
        state={"step": 1},
    )
    await first.append_event(
        "run-durable",
        1,
        stage="planning",
        payload={"done": True},
        timestamp="2026-07-27T10:00:01+00:00",
    )

    restarted = RunStore(db_file)
    await restarted.initialize()

    snapshot = await restarted.get_run("run-durable")
    events = await restarted.list_events("run-durable")
    assert snapshot is not None
    assert snapshot["status"] == "running"
    assert snapshot["state"] == {"step": 1}
    assert [(event["step_index"], event["stage"]) for event in events] == [(1, "planning")]


async def test_append_event_updates_running_stage_snapshot(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runs.db")
    await store.initialize()
    await _create_run(store, "run-progress")
    await store.transition_run(
        "run-progress",
        {RunStatus.QUEUED},
        RunStatus.RUNNING,
        current_stage="planning",
    )

    await store.append_event(
        "run-progress",
        2,
        stage="validating",
        payload={"passed": True},
        timestamp="2026-07-27T10:00:02+00:00",
        state={"status": "VALIDATING", "cursor": 2},
    )

    snapshot = await store.get_run("run-progress")
    assert snapshot is not None
    assert snapshot["status"] == "running"
    assert snapshot["current_stage"] == "validating"
    assert snapshot["state"] == {"status": "VALIDATING", "cursor": 2}


async def test_create_run_rejects_unknown_project_id(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runs.db")
    await store.initialize()

    with pytest.raises(ValueError, match="unknown project_id"):
        await store.create_run(
            "run-project",
            intent="intent",
            policy={"workflow_scope": "engineering"},
            created_at="2026-07-27T10:00:00+00:00",
            project_id="project-missing",
        )


async def test_migration_is_versioned_and_transactional(tmp_path: Path) -> None:
    db_file = tmp_path / "legacy.db"
    connection = sqlite3.connect(db_file)
    connection.executescript(
        """
        CREATE TABLE runs (
            run_id TEXT PRIMARY KEY,
            project_id TEXT,
            intent TEXT NOT NULL,
            objectives TEXT NOT NULL,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            finished_at TEXT,
            summary TEXT,
            devices_used TEXT,
            n_candidates INTEGER DEFAULT 0,
            n_novel INTEGER DEFAULT 0,
            n_known INTEGER DEFAULT 0
        );
        CREATE TABLE reasoning_steps (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            step_index INTEGER NOT NULL,
            stage TEXT NOT NULL,
            title TEXT NOT NULL,
            detail TEXT,
            payload TEXT,
            timestamp TEXT NOT NULL,
            FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
        );
        """
    )
    connection.close()

    store = RunStore(db_file)
    await store.initialize()

    connection = sqlite3.connect(db_file)
    try:
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        columns = {row[1] for row in connection.execute("PRAGMA table_info(runs)").fetchall()}
    finally:
        connection.close()
    assert version == 1
    assert {"current_stage", "state", "error_type", "error_message"} <= columns


async def test_migration_preserves_legacy_duplicate_events_with_unique_cursors(
    tmp_path: Path,
) -> None:
    db_file = tmp_path / "legacy-duplicates.db"
    connection = sqlite3.connect(db_file)
    connection.executescript(
        """
        CREATE TABLE runs (
            run_id TEXT PRIMARY KEY,
            project_id TEXT,
            intent TEXT NOT NULL,
            objectives TEXT NOT NULL,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            finished_at TEXT,
            summary TEXT,
            devices_used TEXT,
            n_candidates INTEGER DEFAULT 0,
            n_novel INTEGER DEFAULT 0,
            n_known INTEGER DEFAULT 0
        );
        CREATE TABLE reasoning_steps (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            step_index INTEGER NOT NULL,
            stage TEXT NOT NULL,
            title TEXT NOT NULL,
            detail TEXT,
            payload TEXT,
            timestamp TEXT NOT NULL
        );
        INSERT INTO runs
            (run_id, intent, objectives, status, created_at, devices_used)
        VALUES
            ('run-legacy', 'intent', '{}', 'running', '2026-07-27T10:00:00Z', '[]');
        INSERT INTO reasoning_steps
            (run_id, step_index, stage, title, payload, timestamp)
        VALUES
            ('run-legacy', 0, 'planning', 'planning', '{}', '2026-07-27T10:00:01Z'),
            ('run-legacy', 0, 'planning-retry', 'planning-retry', '{}',
             '2026-07-27T10:00:02Z');
        """
    )
    connection.close()

    store = RunStore(db_file)
    await store.initialize()
    events = await store.list_events("run-legacy")

    assert len(events) == 2
    assert len({event["step_index"] for event in events}) == 2


async def test_run_pagination_is_stable_when_created_at_ties(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runs.db")
    await store.initialize()
    for run_id in ["run-e", "run-a", "run-d", "run-b", "run-c"]:
        await _create_run(store, run_id)

    seen: list[str] = []
    token: str | None = None
    while True:
        page = await store.list_runs(
            page_size=2,
            page_token=token,
            context={"project_id": "project-1"},
        )
        seen.extend(item["run_id"] for item in page["items"])
        token = page["next_page_token"]
        if token is None:
            break

    assert seen == ["run-a", "run-b", "run-c", "run-d", "run-e"]
    assert len(seen) == len(set(seen))


@pytest.mark.parametrize("page_size", [0, -1, 101])
async def test_run_pagination_rejects_out_of_bounds_page_size(
    tmp_path: Path,
    page_size: int,
) -> None:
    store = RunStore(tmp_path / "runs.db")
    await store.initialize()

    with pytest.raises(ValueError, match="page_size"):
        await store.list_runs(page_size=page_size)


async def test_run_pagination_rejects_malformed_or_context_mismatched_tokens(
    tmp_path: Path,
) -> None:
    store = RunStore(tmp_path / "runs.db")
    await store.initialize()
    for run_id in ["run-a", "run-b"]:
        await _create_run(store, run_id)
    first = await store.list_runs(
        page_size=1,
        context={"project_id": "project-1"},
    )
    token = first["next_page_token"]
    assert token is not None

    with pytest.raises(ValueError, match="page_token"):
        await store.list_runs(page_size=1, page_token=str(object()))
    with pytest.raises(ValueError, match="page_token"):
        await store.list_runs(
            page_size=1,
            page_token=token,
            context={"project_id": "project-2"},
        )
    for malformed in (f"{token}!", f"{token}\n", f"{token}A"):
        with pytest.raises(ValueError, match="page_token"):
            await store.list_runs(page_size=1, page_token=malformed)


async def test_pause_waits_for_inflight_stage_and_resume_releases_same_task(
    tmp_path: Path,
) -> None:
    store = RunStore(tmp_path / "runs.db")
    await store.initialize()
    await _create_run(store, "run-controlled")
    await store.transition_run(
        "run-controlled",
        {RunStatus.QUEUED},
        RunStatus.RUNNING,
        current_stage="planning",
    )
    control = RunControl(store)
    entered = asyncio.Event()
    release = asyncio.Event()

    async def domain_call() -> str:
        entered.set()
        await release.wait()
        return "result"

    stage_task = asyncio.create_task(
        control.execute_stage("run-controlled", "planning", domain_call)
    )
    await entered.wait()
    pause_task = asyncio.create_task(control.pause("run-controlled"))
    await asyncio.sleep(0)

    assert not pause_task.done()
    assert (await store.get_run("run-controlled"))["status"] == "running"

    release.set()
    await pause_task
    assert not stage_task.done()
    assert (await store.get_run("run-controlled"))["status"] == "paused"

    resumed_task = stage_task
    await control.resume("run-controlled")
    assert resumed_task is stage_task
    assert (await store.get_run("run-controlled"))["status"] == "running"
    assert await stage_task == "result"
    assert (await store.get_run("run-controlled"))["status"] == "running"


async def test_workflow_pause_gate_stops_between_graph_stages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = RunStore(tmp_path / "runs.db")
    await store.initialize()
    await _create_run(store, "run-graph")
    await store.transition_run(
        "run-graph",
        {RunStatus.QUEUED},
        RunStatus.RUNNING,
        current_stage="planning",
    )
    control = RunControl(store)
    first_stage_entered = asyncio.Event()
    finish_first_stage = asyncio.Event()
    second_stage_entered = asyncio.Event()

    class _Compiled:
        async def ainvoke(self, state: dict) -> dict:
            first_stage_entered.set()
            await finish_first_stage.wait()
            second_stage_entered.set()
            return {**state, "status": "GENERATING"}

        async def astream(
            self,
            state: dict,
            *,
            stream_mode: str,
        ) -> AsyncGenerator[dict, None]:
            assert stream_mode == "values"
            first_stage_entered.set()
            await finish_first_stage.wait()
            yield {
                **state,
                "status": "PLANNING",
                "history": ["PLANNING"],
                "events": [
                    {
                        "event_index": 0,
                        "stage": "PLANNING",
                        "timestamp": "2026-07-27T10:00:01+00:00",
                    }
                ],
            }
            second_stage_entered.set()
            yield {
                **state,
                "status": "GENERATING",
                "history": ["PLANNING", "GENERATING"],
                "events": [
                    {
                        "event_index": 0,
                        "stage": "PLANNING",
                        "timestamp": "2026-07-27T10:00:01+00:00",
                    },
                    {
                        "event_index": 1,
                        "stage": "GENERATING",
                        "timestamp": "2026-07-27T10:00:02+00:00",
                    },
                ],
            }

    class _WorkflowGraph:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

        def build(self) -> _Compiled:
            return _Compiled()

    monkeypatch.setattr(orchestrator_main, "WorkflowGraph", _WorkflowGraph)
    state = {
        "run_id": "run-graph",
        "trace_id": "trace-graph",
        "artifact_ids": [],
        "events": [],
    }
    workflow_task = asyncio.create_task(
        orchestrator_main._invoke_workflow(
            {
                "workflow_scope": "state_only",
                "validation_passed": True,
                "max_refinements": 1,
            },
            state,
            run_control=control,
        )
    )
    await first_stage_entered.wait()
    pause_task = asyncio.create_task(control.pause("run-graph"))
    finish_first_stage.set()
    try:
        await asyncio.sleep(0)
        assert not second_stage_entered.is_set()
        await asyncio.wait_for(pause_task, timeout=1)
        assert (await store.get_run("run-graph"))["status"] == "paused"

        await control.resume("run-graph")
        result = await workflow_task
        assert second_stage_entered.is_set()
        assert result["status"] == "GENERATING"
    finally:
        for task in (pause_task, workflow_task):
            if not task.done():
                task.cancel()


async def test_pause_waiter_is_released_when_run_has_no_more_boundaries(
    tmp_path: Path,
) -> None:
    store = RunStore(tmp_path / "runs.db")
    await store.initialize()
    await _create_run(store, "run-ending")
    await store.transition_run(
        "run-ending",
        {RunStatus.QUEUED},
        RunStatus.RUNNING,
        current_stage="critic",
    )
    control = RunControl(store)

    pause_task = asyncio.create_task(control.pause("run-ending"))
    await asyncio.sleep(0)
    control.close("run-ending")

    with pytest.raises(ValueError, match="no remaining stage boundary"):
        await asyncio.wait_for(pause_task, timeout=1)


async def test_resume_waiter_is_released_when_paused_task_closes(
    tmp_path: Path,
) -> None:
    store = RunStore(tmp_path / "runs.db")
    await store.initialize()
    await _create_run(store, "run-paused-ending")
    await store.transition_run(
        "run-paused-ending",
        {RunStatus.QUEUED},
        RunStatus.RUNNING,
        current_stage="planning",
    )
    await store.transition_run(
        "run-paused-ending",
        {RunStatus.RUNNING},
        RunStatus.PAUSED,
        current_stage="planning",
    )
    control = RunControl(store)

    resume_task = asyncio.create_task(control.resume("run-paused-ending"))
    await asyncio.sleep(0)
    control.close("run-paused-ending")

    with pytest.raises(ValueError, match="closed before resume"):
        await asyncio.wait_for(resume_task, timeout=1)


async def test_awaiting_evidence_uses_dedicated_resume_path(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runs.db")
    await store.initialize()
    await _create_run(store, "run-evidence")
    await store.transition_run(
        "run-evidence",
        {RunStatus.QUEUED},
        RunStatus.RUNNING,
        current_stage="l4",
    )
    control = RunControl(store)

    evidence_task = asyncio.create_task(control.wait_for_evidence("run-evidence", "l4"))
    for _ in range(100):
        snapshot = await store.get_run("run-evidence")
        if snapshot is not None and snapshot["status"] == "awaiting_evidence":
            break
        await asyncio.sleep(0)
    assert snapshot is not None
    assert snapshot["status"] == "awaiting_evidence"

    with pytest.raises(ValueError, match="cannot resume from status awaiting_evidence"):
        await control.resume("run-evidence")

    await control.resume_evidence("run-evidence")
    await evidence_task
    assert (await store.get_run("run-evidence"))["status"] == "running"


async def test_evidence_resume_reenters_pause_gate_before_next_stage(
    tmp_path: Path,
) -> None:
    store = RunStore(tmp_path / "runs.db")
    await store.initialize()
    await _create_run(store, "run-evidence-race")
    await store.transition_run(
        "run-evidence-race",
        {RunStatus.QUEUED},
        RunStatus.RUNNING,
        current_stage="planning",
    )
    control = RunControl(store)
    evidence_stage_ready = asyncio.Event()
    release_evidence_stage = asyncio.Event()
    next_stage_entered = asyncio.Event()

    class _Compiled:
        async def astream(
            self,
            state: dict,
            *,
            stream_mode: str,
        ) -> AsyncGenerator[dict, None]:
            assert stream_mode == "values"
            evidence_stage_ready.set()
            await release_evidence_stage.wait()
            yield {
                **state,
                "status": "awaiting_evidence",
                "events": [
                    {
                        "event_index": 0,
                        "stage": "l4",
                        "timestamp": "2026-07-27T10:00:01+00:00",
                    }
                ],
            }
            next_stage_entered.set()
            yield {
                **state,
                "status": "critic",
                "events": [
                    {
                        "event_index": 0,
                        "stage": "l4",
                        "timestamp": "2026-07-27T10:00:01+00:00",
                    },
                    {
                        "event_index": 1,
                        "stage": "critic",
                        "timestamp": "2026-07-27T10:00:02+00:00",
                    },
                ],
            }

    workflow_task = asyncio.create_task(
        orchestrator_main._stream_workflow_stages(
            _Compiled(),
            {
                "run_id": "run-evidence-race",
                "trace_id": "trace-evidence-race",
                "events": [],
            },
            control,
        )
    )
    await evidence_stage_ready.wait()
    pause_task = asyncio.create_task(control.pause("run-evidence-race"))
    release_evidence_stage.set()
    for _ in range(100):
        snapshot = await store.get_run("run-evidence-race")
        if snapshot is not None and snapshot["status"] == "awaiting_evidence":
            break
        await asyncio.sleep(0)
    assert snapshot is not None
    assert snapshot["status"] == "awaiting_evidence"

    await control.resume_evidence("run-evidence-race")
    await asyncio.wait_for(pause_task, timeout=1)
    assert (await store.get_run("run-evidence-race"))["status"] == "paused"
    assert not next_stage_entered.is_set()

    await control.resume("run-evidence-race")
    await workflow_task
    assert next_stage_entered.is_set()


async def test_real_workflow_validation_outcome_enters_evidence_gate(
    tmp_path: Path,
) -> None:
    store = RunStore(tmp_path / "runs.db")
    await store.initialize()
    await _create_run(store, "run-real-evidence")
    await store.transition_run(
        "run-real-evidence",
        {RunStatus.QUEUED},
        RunStatus.RUNNING,
        current_stage="planning",
    )
    control = RunControl(store)

    class _Clients:
        async def compile_intent(self, state: dict) -> dict:
            return {"cig": {"source": state["nl_input"]}}

        async def generate_candidates(self, state: dict) -> list[dict]:
            return [{"candidate_id": "candidate-1", "canonical_smiles": "CCO"}]

        async def validate_candidates(self, state: dict) -> dict:
            return {
                "passed": True,
                "outcome": "AWAITING_EVIDENCE",
                "results": [{"candidate_id": "candidate-1", "valid": True}],
            }

        async def plan_routes(self, state: dict) -> dict:
            return {"skipped": True}

        async def review_candidates(self, state: dict) -> dict:
            return {"verdict": "pass"}

    compiled = orchestrator_main.WorkflowGraph(
        clients=_Clients(),
        workflow_scope="engineering",
    ).build()
    workflow_task = asyncio.create_task(
        orchestrator_main._stream_workflow_stages(
            compiled,
            {
                "nl_input": "Design evidence-gated molecules",
                "run_id": "run-real-evidence",
                "trace_id": "trace-real-evidence",
                "artifact_ids": [],
                "events": [],
                "history": [],
                "validation_passed": True,
                "max_refinements": 1,
            },
            control,
        )
    )
    for _ in range(100):
        snapshot = await store.get_run("run-real-evidence")
        if snapshot is not None and snapshot["status"] == "awaiting_evidence":
            break
        await asyncio.sleep(0)
    assert snapshot is not None
    assert snapshot["status"] == "awaiting_evidence"
    assert snapshot["state"]["status"] == "awaiting_evidence"
    assert not workflow_task.done()

    await control.resume_evidence("run-real-evidence")
    final_state = await workflow_task

    assert final_state["status"] == "CRITIC"


async def test_runtime_initializes_each_store_once_under_concurrency(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = RunStore(tmp_path / "runtime.db")
    initialize_calls = 0
    original_initialize = store.initialize

    async def counted_initialize() -> None:
        nonlocal initialize_calls
        initialize_calls += 1
        await original_initialize()

    monkeypatch.setattr(store, "initialize", counted_initialize)
    monkeypatch.setattr(orchestrator_main, "_RUN_STORE", store)
    monkeypatch.setattr(orchestrator_main, "_RUN_CONTROL", None)
    monkeypatch.setattr(orchestrator_main, "_RUN_INITIALIZED_STORE", None)
    monkeypatch.setattr(orchestrator_main, "_RUNTIME_INIT_LOCK", None)

    runtimes = await asyncio.gather(*(orchestrator_main._runtime() for _ in range(10)))

    assert initialize_calls == 1
    assert all(runtime[0] is store for runtime in runtimes)


async def test_restart_marks_queued_running_and_paused_runs_interrupted(
    tmp_path: Path,
) -> None:
    store = RunStore(tmp_path / "runs.db")
    await store.initialize()
    for run_id in ["queued", "running", "paused"]:
        await _create_run(store, run_id)
    await store.transition_run(
        "running",
        {RunStatus.QUEUED},
        RunStatus.RUNNING,
        current_stage="planning",
    )
    await store.transition_run(
        "paused",
        {RunStatus.QUEUED},
        RunStatus.RUNNING,
        current_stage="planning",
    )
    await store.transition_run(
        "paused",
        {RunStatus.RUNNING},
        RunStatus.PAUSED,
        current_stage="planning",
    )

    restarted = RunStore(tmp_path / "runs.db")
    await restarted.initialize()
    count = await restarted.interrupt_active_runs()

    assert count == 3
    assert [
        (await restarted.get_run(run_id))["status"] for run_id in ["queued", "running", "paused"]
    ] == ["interrupted", "interrupted", "interrupted"]


async def test_orchestrator_submission_is_async_and_uses_one_canonical_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = RunStore(tmp_path / "runs.db")
    await store.initialize()
    entered = asyncio.Event()
    release = asyncio.Event()

    class _Compiled:
        async def ainvoke(self, state: dict) -> dict:
            entered.set()
            await release.wait()
            return {
                **state,
                "status": "CRITIC",
                "history": ["PLANNING", "CRITIC"],
                "events": [
                    {
                        "event_index": 0,
                        "stage": "PLANNING",
                        "timestamp": "2026-07-27T10:00:01+00:00",
                    },
                    {
                        "event_index": 1,
                        "stage": "CRITIC",
                        "timestamp": "2026-07-27T10:00:02+00:00",
                    },
                ],
            }

    class _WorkflowGraph:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

        def build(self) -> _Compiled:
            return _Compiled()

    monkeypatch.setattr(orchestrator_main, "WorkflowGraph", _WorkflowGraph)
    monkeypatch.setattr(
        orchestrator_main,
        "_shared_agent_request_client",
        lambda: object(),
    )
    monkeypatch.setattr(orchestrator_main, "_RUN_STORE", store, raising=False)
    monkeypatch.setattr(
        orchestrator_main,
        "_RUN_CONTROL",
        RunControl(store),
        raising=False,
    )

    transport = httpx.ASGITransport(app=orchestrator_main.rest_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/orchestrator/design",
            json={
                "nl_input": "Design KRAS G12C inhibitors",
                "workflow_scope": "engineering",
                "validation_passed": True,
                "max_refinements": 1,
                "run_id": "run-api-1",
            },
        )
        await entered.wait()

        assert response.status_code == 202
        assert response.json() == {
            "design_id": "run-api-1",
            "run_id": "run-api-1",
            "status": "queued",
        }
        running = await client.get("/v1/orchestrator/runs/run-api-1")
        assert running.status_code == 200
        assert running.json()["run_id"] == "run-api-1"
        assert running.json()["status"] == "running"

        release.set()
        for _ in range(100):
            completed = await client.get("/v1/orchestrator/runs/run-api-1")
            if completed.json()["status"] == "completed":
                break
            await asyncio.sleep(0)
        assert completed.json()["status"] == "completed"

        events = await client.get(
            "/v1/orchestrator/runs/run-api-1/events",
            params={"after_step": 0},
        )
        assert events.status_code == 200
        assert [event["step_index"] for event in events.json()["events"]] == [1]


async def test_cancelled_rest_creation_registers_persisted_run_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = RunStore(tmp_path / "runs.db")
    await store.initialize()
    control = RunControl(store)
    create_started = threading.Event()
    release_create = threading.Event()
    create_finished = threading.Event()
    execution_started = asyncio.Event()
    release_execution = asyncio.Event()
    original_create_run = store._create_run

    def controlled_create_run(*args: object, **kwargs: object) -> None:
        create_started.set()
        if not release_create.wait(timeout=5):
            raise TimeoutError("timed out waiting to release run creation")
        try:
            original_create_run(*args, **kwargs)
        finally:
            create_finished.set()

    async def controlled_execution(
        run_id: str,
        request: dict,
        state: dict,
    ) -> None:
        execution_started.set()
        await release_execution.wait()

    monkeypatch.setattr(store, "_create_run", controlled_create_run)
    monkeypatch.setattr(orchestrator_main, "_execute_design_run", controlled_execution)
    monkeypatch.setattr(orchestrator_main, "_RUN_STORE", store)
    monkeypatch.setattr(orchestrator_main, "_RUN_CONTROL", control)
    monkeypatch.setattr(orchestrator_main, "_RUN_INITIALIZED_STORE", store)
    monkeypatch.setattr(orchestrator_main, "_RUN_TASKS", {})
    request_task = asyncio.create_task(
        orchestrator_main.create_design_run(
            {
                "nl_input": "Create a cancellation-safe REST run",
                "workflow_scope": "state_only",
                "validation_passed": True,
                "max_refinements": 0,
                "run_id": "run-rest-create-cancel",
            }
        )
    )

    try:
        await _wait_for_thread_event(create_started)
        request_task.cancel("REST request disconnected")
        await asyncio.sleep(0)
        release_create.set()
        await _wait_for_thread_event(create_finished)
        with pytest.raises(asyncio.CancelledError):
            await request_task

        snapshot = await store.get_run("run-rest-create-cancel")
        assert snapshot is not None
        assert snapshot["status"] in orchestrator_main._NONTERMINAL_RUN_STATUS_VALUES
        owner_task = orchestrator_main._RUN_TASKS.get("run-rest-create-cancel")
        assert owner_task is not None
        await asyncio.wait_for(execution_started.wait(), timeout=5)
        assert not owner_task.done()
    finally:
        release_create.set()
        release_execution.set()
        await asyncio.gather(request_task, return_exceptions=True)
        owner_task = orchestrator_main._RUN_TASKS.get("run-rest-create-cancel")
        if owner_task is not None:
            await asyncio.gather(owner_task, return_exceptions=True)


async def test_cancel_endpoint_interrupts_active_rest_run_and_is_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = RunStore(tmp_path / "runs.db")
    await store.initialize()
    control = RunControl(store)
    entered = asyncio.Event()
    release = asyncio.Event()

    class _Compiled:
        async def ainvoke(self, state: dict) -> dict:
            entered.set()
            await release.wait()
            return {**state, "status": "CRITIC", "history": [], "events": []}

    class _WorkflowGraph:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

        def build(self) -> _Compiled:
            return _Compiled()

    monkeypatch.setattr(orchestrator_main, "WorkflowGraph", _WorkflowGraph)
    monkeypatch.setattr(orchestrator_main, "_shared_agent_request_client", lambda: object())
    monkeypatch.setattr(orchestrator_main, "_RUN_STORE", store)
    monkeypatch.setattr(orchestrator_main, "_RUN_CONTROL", control)
    monkeypatch.setattr(orchestrator_main, "_RUN_INITIALIZED_STORE", store)
    monkeypatch.setattr(orchestrator_main, "_RUN_TASKS", {})
    await _create_run(store, "run-completed")
    await store.transition_run(
        "run-completed",
        {RunStatus.QUEUED},
        RunStatus.RUNNING,
        current_stage="planning",
    )
    await store.transition_run(
        "run-completed",
        {RunStatus.RUNNING},
        RunStatus.COMPLETED,
        current_stage="completed",
    )

    transport = httpx.ASGITransport(app=orchestrator_main.rest_app)
    pause_task: asyncio.Task[None] | None = None
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        submitted = await client.post(
            "/v1/orchestrator/design",
            json={
                "nl_input": "Design cancellable molecules",
                "workflow_scope": "engineering",
                "validation_passed": True,
                "max_refinements": 1,
                "run_id": "run-cancel-rest",
            },
        )
        await entered.wait()
        pause_task = asyncio.create_task(control.pause("run-cancel-rest"))
        await asyncio.sleep(0)
        try:
            cancelled = await client.post("/v1/orchestrator/runs/run-cancel-rest/cancel")

            assert submitted.status_code == 202
            assert cancelled.status_code == 200
            snapshot = cancelled.json()
            assert snapshot["status"] == "interrupted"
            assert snapshot["error_type"] == "CancelledError"
            assert snapshot["error_message"]
            with pytest.raises(ValueError, match="no remaining stage boundary"):
                await asyncio.wait_for(pause_task, timeout=1)

            repeated = await client.post("/v1/orchestrator/runs/run-cancel-rest/cancel")
            unknown = await client.post("/v1/orchestrator/runs/run-unknown/cancel")
            terminal = await client.post("/v1/orchestrator/runs/run-completed/cancel")

            assert repeated.status_code == 200
            assert repeated.json()["status"] == "interrupted"
            assert unknown.status_code == 404
            assert terminal.status_code == 409
        finally:
            release.set()
            task = orchestrator_main._RUN_TASKS.get("run-cancel-rest")
            if task is not None and not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            control.close("run-cancel-rest")
            if pause_task is not None and not pause_task.done():
                await asyncio.gather(pause_task, return_exceptions=True)


async def test_execute_design_run_cancellation_during_failure_persistence_interrupts_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = RunStore(tmp_path / "runs.db")
    await store.initialize()
    await _create_run(store, "run-cancel-rest-failure")
    control = RunControl(store)
    failure_transition_committed = threading.Event()
    release_failure_worker = threading.Event()
    original_transition_run = store._transition_run

    async def fail_workflow(
        request: dict,
        state: dict,
        *,
        run_control: RunControl | None = None,
    ) -> dict:
        raise RuntimeError("workflow failed before persistence")

    def controlled_transition(*args: object, **kwargs: object) -> None:
        original_transition_run(*args, **kwargs)
        target = RunStatus(args[2])
        if target == RunStatus.FAILED:
            failure_transition_committed.set()
            if not release_failure_worker.wait(timeout=5):
                raise TimeoutError("timed out waiting to release failure persistence")

    monkeypatch.setattr(store, "_transition_run", controlled_transition)
    monkeypatch.setattr(orchestrator_main, "_invoke_workflow", fail_workflow)
    monkeypatch.setattr(orchestrator_main, "_RUN_STORE", store)
    monkeypatch.setattr(orchestrator_main, "_RUN_CONTROL", control)
    monkeypatch.setattr(orchestrator_main, "_RUN_INITIALIZED_STORE", store)
    run_task = asyncio.create_task(
        orchestrator_main._execute_design_run(
            "run-cancel-rest-failure",
            {"workflow_scope": "state_only"},
            {
                "run_id": "run-cancel-rest-failure",
                "trace_id": "trace-cancel-rest-failure",
                "status": "PLANNING",
            },
        )
    )

    try:
        await _wait_for_thread_event(failure_transition_committed)
        run_task.cancel("cancel during failure persistence")
        await asyncio.sleep(0)
        release_failure_worker.set()
        with pytest.raises(asyncio.CancelledError):
            await run_task

        snapshot = await store.get_run("run-cancel-rest-failure")
        assert snapshot is not None
        assert snapshot["status"] == "interrupted"
        assert snapshot["error_type"] == "CancelledError"
        assert snapshot["error_message"] == "cancel during failure persistence"
    finally:
        release_failure_worker.set()
        await asyncio.gather(run_task, return_exceptions=True)


async def test_start_design_cancellation_during_failure_persistence_interrupts_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = RunStore(tmp_path / "runs.db")
    await store.initialize()
    control = RunControl(store)
    failure_transition_committed = threading.Event()
    release_failure_worker = threading.Event()
    original_transition_run = store._transition_run

    async def fail_workflow(
        request: dict,
        state: dict,
        *,
        run_control: RunControl | None = None,
    ) -> dict:
        raise RuntimeError("direct workflow failed before persistence")

    def controlled_transition(*args: object, **kwargs: object) -> None:
        original_transition_run(*args, **kwargs)
        target = RunStatus(args[2])
        if target == RunStatus.FAILED:
            failure_transition_committed.set()
            if not release_failure_worker.wait(timeout=5):
                raise TimeoutError("timed out waiting to release failure persistence")

    monkeypatch.setattr(store, "_transition_run", controlled_transition)
    monkeypatch.setattr(orchestrator_main, "_invoke_workflow", fail_workflow)
    monkeypatch.setattr(orchestrator_main, "_RUN_STORE", store)
    monkeypatch.setattr(orchestrator_main, "_RUN_CONTROL", control)
    monkeypatch.setattr(orchestrator_main, "_RUN_INITIALIZED_STORE", store)
    run_task = asyncio.create_task(
        orchestrator_main.start_design(
            {
                "nl_input": "Cancel failed direct persistence",
                "workflow_scope": "state_only",
                "validation_passed": True,
                "max_refinements": 0,
                "run_id": "run-cancel-direct-failure",
            }
        )
    )

    try:
        await _wait_for_thread_event(failure_transition_committed)
        run_task.cancel("cancel direct failure persistence")
        await asyncio.sleep(0)
        release_failure_worker.set()
        with pytest.raises(asyncio.CancelledError):
            await run_task

        snapshot = await store.get_run("run-cancel-direct-failure")
        assert snapshot is not None
        assert snapshot["status"] == "interrupted"
        assert snapshot["error_type"] == "CancelledError"
        assert snapshot["error_message"] == "cancel direct failure persistence"
    finally:
        release_failure_worker.set()
        await asyncio.gather(run_task, return_exceptions=True)


async def test_start_design_cancellation_persists_interrupted_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = RunStore(tmp_path / "runs.db")
    await store.initialize()
    control = RunControl(store)
    entered = asyncio.Event()
    release = asyncio.Event()

    class _Compiled:
        async def ainvoke(self, state: dict) -> dict:
            entered.set()
            await release.wait()
            return {**state, "status": "CRITIC", "history": [], "events": []}

    class _WorkflowGraph:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

        def build(self) -> _Compiled:
            return _Compiled()

    monkeypatch.setattr(orchestrator_main, "WorkflowGraph", _WorkflowGraph)
    monkeypatch.setattr(orchestrator_main, "_RUN_STORE", store)
    monkeypatch.setattr(orchestrator_main, "_RUN_CONTROL", control)
    monkeypatch.setattr(orchestrator_main, "_RUN_INITIALIZED_STORE", store)
    direct_task = asyncio.create_task(
        orchestrator_main.start_design(
            {
                "nl_input": "Design cancellable direct molecules",
                "workflow_scope": "state_only",
                "validation_passed": True,
                "max_refinements": 1,
                "run_id": "run-cancel-direct",
            }
        )
    )

    try:
        await entered.wait()
        direct_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await direct_task

        snapshot = await store.get_run("run-cancel-direct")
        assert snapshot is not None
        assert snapshot["status"] == "interrupted"
        assert snapshot["error_type"] == "CancelledError"
        assert snapshot["error_message"]
    finally:
        release.set()
        if not direct_task.done():
            direct_task.cancel()
            await asyncio.gather(direct_task, return_exceptions=True)


async def test_start_design_cancellation_during_create_interrupts_owned_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = RunStore(tmp_path / "runs.db")
    await store.initialize()
    control = RunControl(store)
    created = asyncio.Event()
    release_create = asyncio.Event()
    original_create_run = store.create_run

    async def create_then_wait(*args: object, **kwargs: object) -> None:
        await original_create_run(*args, **kwargs)
        created.set()
        await release_create.wait()

    monkeypatch.setattr(store, "create_run", create_then_wait)
    monkeypatch.setattr(orchestrator_main, "_RUN_STORE", store)
    monkeypatch.setattr(orchestrator_main, "_RUN_CONTROL", control)
    monkeypatch.setattr(orchestrator_main, "_RUN_INITIALIZED_STORE", store)
    direct_task = asyncio.create_task(
        orchestrator_main.start_design(
            {
                "nl_input": "Design cancellation-safe initialization",
                "workflow_scope": "state_only",
                "validation_passed": True,
                "max_refinements": 1,
                "run_id": "run-cancel-create",
            }
        )
    )

    await created.wait()
    direct_task.cancel()
    release_create.set()
    with pytest.raises(asyncio.CancelledError):
        await direct_task

    snapshot = await store.get_run("run-cancel-create")
    assert snapshot is not None
    assert snapshot["status"] == "interrupted"
    assert snapshot["error_type"] == "CancelledError"
    assert snapshot["error_message"]


async def test_start_design_repeated_cancellation_during_create_interrupts_owned_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = RunStore(tmp_path / "runs.db")
    await store.initialize()
    control = RunControl(store)
    created = asyncio.Event()
    release_create = asyncio.Event()
    original_create_run = store.create_run

    async def create_then_wait(*args: object, **kwargs: object) -> None:
        await original_create_run(*args, **kwargs)
        created.set()
        await release_create.wait()

    monkeypatch.setattr(store, "create_run", create_then_wait)
    monkeypatch.setattr(orchestrator_main, "_RUN_STORE", store)
    monkeypatch.setattr(orchestrator_main, "_RUN_CONTROL", control)
    monkeypatch.setattr(orchestrator_main, "_RUN_INITIALIZED_STORE", store)
    direct_task = asyncio.create_task(
        orchestrator_main.start_design(
            {
                "nl_input": "Design repeated-cancellation-safe initialization",
                "workflow_scope": "state_only",
                "validation_passed": True,
                "max_refinements": 1,
                "run_id": "run-repeat-cancel-create",
            }
        )
    )

    await created.wait()
    direct_task.cancel("first cancellation")
    await asyncio.sleep(0)
    assert not direct_task.done()
    direct_task.cancel("second cancellation")
    release_create.set()
    with pytest.raises(asyncio.CancelledError):
        await direct_task

    snapshot = await store.get_run("run-repeat-cancel-create")
    assert snapshot is not None
    assert snapshot["status"] == "interrupted"
    assert snapshot["error_type"] == "CancelledError"
    assert snapshot["error_message"]


async def test_start_design_repeated_cancellation_during_interrupt_persists_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = RunStore(tmp_path / "runs.db")
    await store.initialize()
    control = RunControl(store)
    workflow_entered = asyncio.Event()
    interruption_started = asyncio.Event()
    release_interruption = asyncio.Event()
    original_transition_run = store.transition_run

    class _Compiled:
        async def ainvoke(self, state: dict) -> dict:
            workflow_entered.set()
            await asyncio.Event().wait()
            return state

    class _WorkflowGraph:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

        def build(self) -> _Compiled:
            return _Compiled()

    async def controlled_transition(
        run_id: str,
        expected: set[RunStatus],
        target: RunStatus,
        **kwargs: object,
    ) -> None:
        if target == RunStatus.INTERRUPTED:
            interruption_started.set()
            await release_interruption.wait()
        await original_transition_run(run_id, expected, target, **kwargs)

    monkeypatch.setattr(orchestrator_main, "WorkflowGraph", _WorkflowGraph)
    monkeypatch.setattr(store, "transition_run", controlled_transition)
    monkeypatch.setattr(orchestrator_main, "_RUN_STORE", store)
    monkeypatch.setattr(orchestrator_main, "_RUN_CONTROL", control)
    monkeypatch.setattr(orchestrator_main, "_RUN_INITIALIZED_STORE", store)
    direct_task = asyncio.create_task(
        orchestrator_main.start_design(
            {
                "nl_input": "Persist repeated cancellation",
                "workflow_scope": "state_only",
                "validation_passed": True,
                "max_refinements": 1,
                "run_id": "run-repeat-cancel-interrupt",
            }
        )
    )

    await workflow_entered.wait()
    direct_task.cancel("first cancellation")
    await interruption_started.wait()
    direct_task.cancel("second cancellation")
    release_interruption.set()
    with pytest.raises(asyncio.CancelledError):
        await direct_task

    snapshot = await store.get_run("run-repeat-cancel-interrupt")
    assert snapshot is not None
    assert snapshot["status"] == "interrupted"
    assert snapshot["error_type"] == "CancelledError"
    assert snapshot["error_message"] == "first cancellation"


async def test_start_design_cancelled_duplicate_does_not_interrupt_existing_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = RunStore(tmp_path / "runs.db")
    await store.initialize()
    await _create_run(store, "run-owned-by-other")
    await store.transition_run(
        "run-owned-by-other",
        {RunStatus.QUEUED},
        RunStatus.RUNNING,
        current_stage="existing-work",
    )
    control = RunControl(store)
    existing_control_state = control._state("run-owned-by-other")
    duplicate_detected = asyncio.Event()
    release_duplicate = asyncio.Event()
    original_create_run = store.create_run

    async def detect_duplicate_then_wait(*args: object, **kwargs: object) -> None:
        try:
            await original_create_run(*args, **kwargs)
        except RunAlreadyExistsError:
            duplicate_detected.set()
            await release_duplicate.wait()
            raise

    monkeypatch.setattr(store, "create_run", detect_duplicate_then_wait)
    monkeypatch.setattr(orchestrator_main, "_RUN_STORE", store)
    monkeypatch.setattr(orchestrator_main, "_RUN_CONTROL", control)
    monkeypatch.setattr(orchestrator_main, "_RUN_INITIALIZED_STORE", store)
    direct_task = asyncio.create_task(
        orchestrator_main.start_design(
            {
                "nl_input": "Do not take ownership of another run",
                "workflow_scope": "state_only",
                "validation_passed": True,
                "max_refinements": 1,
                "run_id": "run-owned-by-other",
            }
        )
    )

    await duplicate_detected.wait()
    direct_task.cancel()
    release_duplicate.set()
    with pytest.raises(asyncio.CancelledError):
        await direct_task

    snapshot = await store.get_run("run-owned-by-other")
    assert snapshot is not None
    assert snapshot["status"] == "running"
    assert snapshot["current_stage"] == "existing-work"
    assert snapshot["error_type"] is None
    assert snapshot["error_message"] is None
    assert control._states["run-owned-by-other"] is existing_control_state
    assert not existing_control_state.closed.is_set()


async def test_concurrent_cancel_requests_return_same_interrupted_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = RunStore(tmp_path / "runs.db")
    await store.initialize()
    await _create_run(store, "run-double-cancel")
    await store.transition_run(
        "run-double-cancel",
        {RunStatus.QUEUED},
        RunStatus.RUNNING,
        current_stage="generating",
    )
    control = RunControl(store)
    original_get_run = store.get_run
    first_snapshot_captured = asyncio.Event()
    second_snapshot_captured = asyncio.Event()
    release_first_snapshot = asyncio.Event()
    release_second_snapshot = asyncio.Event()
    captured_snapshots = 0

    async def staged_get_run(run_id: str) -> dict | None:
        nonlocal captured_snapshots
        snapshot = await original_get_run(run_id)
        if run_id != "run-double-cancel" or captured_snapshots >= 2:
            return snapshot
        captured_snapshots += 1
        if captured_snapshots == 1:
            first_snapshot_captured.set()
            await release_first_snapshot.wait()
        else:
            second_snapshot_captured.set()
            await release_second_snapshot.wait()
        return snapshot

    async def active_run() -> None:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError as exc:
            await orchestrator_main._interrupt_cancelled_run(
                store,
                control,
                "run-double-cancel",
                str(exc) or "workflow task cancelled",
            )
            raise

    monkeypatch.setattr(store, "get_run", staged_get_run)
    monkeypatch.setattr(orchestrator_main, "_RUN_STORE", store)
    monkeypatch.setattr(orchestrator_main, "_RUN_CONTROL", control)
    monkeypatch.setattr(orchestrator_main, "_RUN_INITIALIZED_STORE", store)
    monkeypatch.setattr(orchestrator_main, "_RUN_TASKS", {})
    run_task = asyncio.create_task(active_run())
    orchestrator_main._RUN_TASKS["run-double-cancel"] = run_task
    run_task.add_done_callback(
        lambda completed: orchestrator_main._finish_run_task(
            "run-double-cancel",
            completed,
        )
    )

    transport = httpx.ASGITransport(app=orchestrator_main.rest_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        first_request = asyncio.create_task(
            client.post("/v1/orchestrator/runs/run-double-cancel/cancel")
        )
        await first_snapshot_captured.wait()
        second_request = asyncio.create_task(
            client.post("/v1/orchestrator/runs/run-double-cancel/cancel")
        )
        await second_snapshot_captured.wait()
        try:
            release_first_snapshot.set()
            first_response = await first_request
            release_second_snapshot.set()
            second_response = await second_request
        finally:
            release_first_snapshot.set()
            release_second_snapshot.set()
            if not run_task.done():
                run_task.cancel()
            await asyncio.gather(
                first_request,
                second_request,
                run_task,
                return_exceptions=True,
            )

    assert first_response.status_code == 200
    assert second_response.status_code == 200
    assert first_response.json()["status"] == "interrupted"
    assert second_response.json() == first_response.json()


@pytest.mark.parametrize(
    ("payload", "detail"),
    [
        (
            {
                "nl_input": "Design KRAS G12C inhibitors",
                "validation_passed": True,
                "max_refinements": 1,
            },
            "workflow_scope is required",
        ),
        (
            {
                "nl_input": "Design KRAS G12C inhibitors",
                "workflow_scope": "engineering",
                "max_refinements": 1,
            },
            "validation_passed is required",
        ),
        (
            {
                "nl_input": "Design KRAS G12C inhibitors",
                "workflow_scope": "engineering",
                "validation_passed": True,
            },
            "max_refinements is required",
        ),
    ],
)
async def test_orchestrator_requires_explicit_workflow_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    payload: dict,
    detail: str,
) -> None:
    store = RunStore(tmp_path / "runs.db")
    await store.initialize()
    monkeypatch.setattr(orchestrator_main, "_RUN_STORE", store, raising=False)
    monkeypatch.setattr(
        orchestrator_main,
        "_RUN_CONTROL",
        RunControl(store),
        raising=False,
    )
    transport = httpx.ASGITransport(app=orchestrator_main.rest_app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/orchestrator/design",
            json=payload,
        )

    assert response.status_code == 400
    assert response.json()["detail"] == detail


async def test_orchestrator_lists_runs_with_opaque_pagination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = RunStore(tmp_path / "runs.db")
    await store.initialize()
    for run_id in ["run-a", "run-b", "run-c"]:
        await _create_run(store, run_id)
    monkeypatch.setattr(orchestrator_main, "_RUN_STORE", store, raising=False)
    monkeypatch.setattr(
        orchestrator_main,
        "_RUN_CONTROL",
        RunControl(store),
        raising=False,
    )
    transport = httpx.ASGITransport(app=orchestrator_main.rest_app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        first = await client.get("/v1/orchestrator/runs", params={"page_size": 2})
        assert first.status_code == 200
        assert [item["run_id"] for item in first.json()["runs"]] == ["run-a", "run-b"]
        token = first.json()["next_page_token"]
        assert isinstance(token, str)

        second = await client.get(
            "/v1/orchestrator/runs",
            params={"page_size": 2, "page_token": token},
        )
        assert second.status_code == 200
        assert [item["run_id"] for item in second.json()["runs"]] == ["run-c"]

        malformed = await client.get(
            "/v1/orchestrator/runs",
            params={"page_size": 2, "page_token": "invalid"},
        )
        assert malformed.status_code == 400


async def test_orchestrator_rejects_duplicate_run_id_without_scheduling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = RunStore(tmp_path / "runs.db")
    await store.initialize()
    await _create_run(store, "run-existing")
    monkeypatch.setattr(orchestrator_main, "_RUN_STORE", store, raising=False)
    monkeypatch.setattr(
        orchestrator_main,
        "_RUN_CONTROL",
        RunControl(store),
        raising=False,
    )
    transport = httpx.ASGITransport(app=orchestrator_main.rest_app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/orchestrator/design",
            json={
                "nl_input": "Design KRAS G12C inhibitors",
                "workflow_scope": "engineering",
                "validation_passed": True,
                "max_refinements": 1,
                "run_id": "run-existing",
            },
        )

    assert response.status_code == 409
    assert "run-existing" not in orchestrator_main._RUN_TASKS


async def test_inline_orchestrator_does_not_default_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = RunStore(tmp_path / "runs.db")
    await store.initialize()
    monkeypatch.setattr(orchestrator_main, "_RUN_STORE", store, raising=False)
    monkeypatch.setattr(
        orchestrator_main,
        "_RUN_CONTROL",
        RunControl(store),
        raising=False,
    )

    with pytest.raises(orchestrator_main.HTTPException) as error:
        await orchestrator_main.start_design({"nl_input": "Design KRAS G12C inhibitors"})

    assert error.value.status_code == 400
    assert error.value.detail == "workflow_scope is required"


async def test_grpc_pipeline_policy_uses_proto_presence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mf_core.proto_gen.moleculeforge.v1.agent import orchestrator_pb2

    captured: list[dict] = []

    async def fake_start_design(request: dict) -> dict:
        captured.append(request)
        return {
            "design_id": "run-grpc",
            "run_id": "run-grpc",
            "trace_id": "trace-grpc",
            "status": "queued",
        }

    monkeypatch.setattr(orchestrator_main, "start_design", fake_start_design)
    request = orchestrator_pb2.StartPipelineRequest(
        nl_input="Design KRAS G12C inhibitors",
        workflow_scope="engineering",
        validation_passed=False,
        max_refinements=0,
        run_id="run-grpc",
        trace_id="trace-grpc",
    )

    response = await orchestrator_main.OrchestratorServicer().StartPipeline(
        request,
        None,
    )

    assert request.HasField("validation_passed")
    assert request.HasField("max_refinements")
    assert response.status == "queued"
    assert captured == [
        {
            "nl_input": "Design KRAS G12C inhibitors",
            "workflow_scope": "engineering",
            "validation_passed": False,
            "max_refinements": 0,
            "run_id": "run-grpc",
            "trace_id": "trace-grpc",
        }
    ]


async def test_grpc_pipeline_rejects_absent_policy_presence() -> None:
    from mf_core.proto_gen.moleculeforge.v1.agent import orchestrator_pb2

    request = orchestrator_pb2.StartPipelineRequest(
        nl_input="Design KRAS G12C inhibitors",
        workflow_scope="engineering",
    )

    with pytest.raises(orchestrator_main.HTTPException) as error:
        await orchestrator_main.OrchestratorServicer().StartPipeline(request, None)

    assert error.value.status_code == 400
    assert error.value.detail == "validation_passed is required"


def test_legacy_store_run_write_bypasses_are_not_exposed() -> None:
    from mf_core.db import store

    assert not hasattr(store, "insert_run")
    assert not hasattr(store, "update_run")
