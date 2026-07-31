from __future__ import annotations

import asyncio
import base64
import hashlib
import json
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


def _attach_l4_awaiting_validation(state: dict) -> None:
    candidates = state["candidates"]
    project_id = state["request"]["project_id"]
    request_id = state["request"].setdefault("request_id", f"request-{state['run_id']}")
    thresholds = [
        {
            "level": 0,
            "oracle": "rdkit",
            "metric": "qed",
            "direction": "maximize",
            "value": 0.5,
        },
        {
            "level": 1,
            "oracle": "admet",
            "metric": "clearance",
            "direction": "minimize",
            "value": 1.0,
        },
        {
            "level": 2,
            "oracle": "dock",
            "metric": "docking_score",
            "direction": "minimize",
            "value": -6.0,
        },
        {
            "level": 3,
            "oracle": "fep",
            "metric": "rbfe",
            "direction": "minimize",
            "value": 1.0,
        },
        {
            "level": 4,
            "oracle": "external",
            "metric": "activity",
            "direction": "maximize",
            "value": 0.75,
        },
    ]
    policy = {
        "oracle_level": 4,
        "batch_size": 2,
        "max_concurrency": 2,
        "thresholds": thresholds,
        "oracle_inputs": {},
    }
    state["request"]["validation_policy"] = policy
    records = []
    for candidate in candidates:
        metrics = []
        evidence = []
        levels = []
        values = (0.8, 0.2, -7.0, 0.1)
        for level, threshold in enumerate(thresholds[:4]):
            evidence_id = f"{candidate['candidate_id']}:{threshold['oracle']}"
            metric = {
                "level": level,
                "oracle": threshold["oracle"],
                "metric": threshold["metric"],
                "value": values[level],
                "direction": threshold["direction"],
                "threshold": threshold["value"],
                "passed": True,
            }
            metrics.append(metric)
            evidence.append(
                {
                    "evidence_id": evidence_id,
                    "level": level,
                    "oracle": threshold["oracle"],
                }
            )
            levels.append(
                {
                    "level": level,
                    "outcome": "PASS",
                    "oracles": [
                        {
                            "oracle": threshold["oracle"],
                            "outcome": "PASS",
                            "metrics": [metric],
                            "evidence_ids": [evidence_id],
                        }
                    ],
                }
            )
        levels.append(
            {
                "level": 4,
                "outcome": "AWAITING_EVIDENCE",
                "oracles": [
                    {
                        "oracle": "external",
                        "outcome": "AWAITING_EVIDENCE",
                        "metrics": [],
                        "evidence_ids": [],
                        "reason": "external evidence is required",
                    }
                ],
            }
        )
        evidence.append(
            {
                "evidence_id": f"{candidate['candidate_id']}:validation",
                "level": 4,
                "oracle": "validation_agent",
            }
        )
        records.append(
            {
                "schema_version": "validation.record.v1",
                "candidate_id": candidate["candidate_id"],
                "canonical_smiles": candidate["canonical_smiles"],
                "outcome": "AWAITING_EVIDENCE",
                "metrics": metrics,
                "evidence": evidence,
                "levels": levels,
            }
        )
    state["validation"] = {
        "validation_schema_version": "validation.batch.v1",
        "agent": "validation_agent",
        "project_id": project_id,
        "outcome": "AWAITING_EVIDENCE",
        "validation_policy": policy,
        "records": records,
        "results": records,
    }
    state["request"]["request_id"] = request_id


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


async def test_terminal_transition_projects_workflow_summary_into_run_snapshot(
    tmp_path: Path,
) -> None:
    store = RunStore(tmp_path / "runs.db")
    await store.initialize()
    await _create_run(store, "run-summary")
    await store.transition_run(
        "run-summary",
        {RunStatus.QUEUED},
        RunStatus.RUNNING,
        current_stage="planning",
    )
    final_state = {
        "objectives": ["qed", "sa_score"],
        "summary": "Two validated candidates",
        "devices_used": ["cpu", "cuda:0"],
        "candidates": [
            {"candidate_id": "candidate-1"},
            {"candidate_id": "candidate-2"},
            {"candidate_id": "candidate-3"},
        ],
        "validation": {
            "results": [
                {"candidate_id": "candidate-1", "valid": True, "is_novel": True},
                {"candidate_id": "candidate-2", "valid": True, "is_novel": False},
            ]
        },
    }

    await store.transition_run(
        "run-summary",
        {RunStatus.RUNNING},
        RunStatus.COMPLETED,
        current_stage="completed",
        state=final_state,
    )

    snapshot = await store.get_run("run-summary")
    assert snapshot is not None
    assert snapshot["state"] == final_state
    assert snapshot["objectives"] == ["qed", "sa_score"]
    assert snapshot["summary"] == "Two validated candidates"
    assert snapshot["devices_used"] == ["cpu", "cuda:0"]
    assert snapshot["n_candidates"] == 3
    assert snapshot["n_novel"] == 1
    assert snapshot["n_known"] == 1


async def test_engineering_validation_records_predictor_devices() -> None:
    validation = await orchestrator_main.EngineeringWorkflowClients().validate_candidates(
        {
            "candidates": [
                {
                    "candidate_id": "candidate-1",
                    "canonical_smiles": "CCO",
                }
            ],
            "request": {"l0_threshold": 0.0},
        }
    )

    assert validation["devices_used"] == ["cpu"]
    assert validation["results"]


async def test_real_workflow_projects_semantic_metadata_and_actual_devices(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = RunStore(tmp_path / "runs.db")
    await store.initialize()
    control = RunControl(store)
    cig_objectives = [
        {
            "name": "qed",
            "type": "MAXIMIZE",
            "weight": 1.0,
        }
    ]

    class _Clients:
        async def compile_intent(self, state: dict) -> dict:
            return {
                "cig": {
                    "metadata": {"intent_summary": "Prioritize drug-like candidates"},
                    "objectives": cig_objectives,
                }
            }

        async def generate_candidates(self, state: dict) -> list[dict]:
            return [
                {
                    "candidate_id": "candidate-1",
                    "canonical_smiles": "CCO",
                    "devices_used": ["cuda:0"],
                }
            ]

        async def validate_candidates(self, state: dict) -> dict:
            return {
                "passed": True,
                "devices_used": ["cpu"],
                "results": [
                    {
                        "candidate_id": "candidate-1",
                        "canonical_smiles": "CCO",
                        "valid": True,
                        "is_novel": True,
                        "devices_used": ["cuda:1"],
                    }
                ],
            }

        async def plan_routes(self, state: dict) -> dict:
            return {"skipped": True}

        async def review_candidates(self, state: dict) -> dict:
            return {"verdict": "pass"}

    monkeypatch.setattr(orchestrator_main, "_RUN_STORE", store)
    monkeypatch.setattr(orchestrator_main, "_RUN_CONTROL", control)
    monkeypatch.setattr(orchestrator_main, "_RUN_INITIALIZED_STORE", store)

    response = await orchestrator_main.start_design(
        {
            "nl_input": "Design a drug-like candidate",
            "objectives": {"fallback": "request objective"},
            "workflow_scope": "engineering",
            "validation_passed": True,
            "max_refinements": 0,
            "run_id": "run-real-terminal-metadata",
            "clients": _Clients(),
        }
    )

    snapshot = await store.get_run("run-real-terminal-metadata")
    assert snapshot is not None
    assert response["state"]["summary"] == "Prioritize drug-like candidates"
    assert response["state"]["objectives"] == cig_objectives
    assert response["state"]["devices_used"] == ["cuda:0", "cpu", "cuda:1"]
    assert snapshot["summary"] == "Prioritize drug-like candidates"
    assert snapshot["objectives"] == cig_objectives
    assert snapshot["devices_used"] == ["cuda:0", "cpu", "cuda:1"]
    assert snapshot["n_candidates"] == 1
    assert snapshot["n_novel"] == 1
    assert snapshot["n_known"] == 0


async def test_real_engineering_workflow_projects_canonical_cig_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = RunStore(tmp_path / "runs.db")
    await store.initialize()
    control = RunControl(store)
    intent = "Design a drug-like molecule with high QED"

    class _EngineeringClients(orchestrator_main.EngineeringWorkflowClients):
        async def review_candidates(self, state: dict) -> dict:
            return {"verdict": "pass"}

    monkeypatch.delenv("AIZYNTH_CONFIG_PATH", raising=False)
    monkeypatch.setattr(orchestrator_main, "_RUN_STORE", store)
    monkeypatch.setattr(orchestrator_main, "_RUN_CONTROL", control)
    monkeypatch.setattr(orchestrator_main, "_RUN_INITIALIZED_STORE", store)

    response = await orchestrator_main.start_design(
        {
            "nl_input": intent,
            "workflow_scope": "engineering",
            "validation_passed": True,
            "max_refinements": 0,
            "run_id": "run-real-canonical-cig-metadata",
            "seed_smiles": ["CCO"],
            "n_samples": 1,
            "clients": _EngineeringClients(),
        }
    )

    snapshot = await store.get_run("run-real-canonical-cig-metadata")
    assert snapshot is not None
    cig_objectives = response["state"]["cig"]["objective_nodes"]
    assert cig_objectives == [
        {
            "id": "obj_qed",
            "name": "qed",
            "type": "continuous_maximize",
            "oracle": "rdkit",
            "target_value": 0.0,
            "target_min": None,
            "target_max": None,
            "property": "",
            "weight": 1.0,
            "pareto_tier": 1,
            "constraints": None,
        }
    ]
    assert response["state"]["objectives"] == cig_objectives
    assert response["state"]["summary"] == intent
    assert response["state"]["devices_used"] == ["cpu"]
    assert snapshot["objectives"] == cig_objectives
    assert snapshot["summary"] == intent
    assert snapshot["devices_used"] == ["cpu"]


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
    assert version == 2
    assert {
        "current_stage",
        "state",
        "error_type",
        "error_message",
        "owner_principal_id",
    } <= columns


async def test_run_store_persists_immutable_owner_principal(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runs.db")
    await store.initialize()

    await store.create_run(
        "run-owned",
        intent="Design an evidence-backed molecule",
        policy={"workflow_scope": "full"},
        created_at="2026-07-30T00:00:00+00:00",
        owner_principal_id="scientist-1",
    )
    await store.create_run(
        "run-owned",
        intent="Updated intent",
        policy={"workflow_scope": "full"},
        created_at="2026-07-30T00:01:00+00:00",
        owner_principal_id="scientist-2",
    )

    snapshot = await store.get_run("run-owned")
    assert snapshot is not None
    assert snapshot["owner_principal_id"] == "scientist-1"


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

    assert seen == ["run-e", "run-d", "run-c", "run-b", "run-a"]
    assert len(seen) == len(set(seen))


async def test_run_pagination_is_newest_first_without_repeating_after_new_writes(
    tmp_path: Path,
) -> None:
    store = RunStore(tmp_path / "runs.db")
    await store.initialize()
    await _create_run(store, "run-oldest", created_at="2026-07-27T10:00:00+00:00")
    await _create_run(store, "run-middle", created_at="2026-07-27T11:00:00+00:00")
    await _create_run(store, "run-newest", created_at="2026-07-27T12:00:00+00:00")

    first = await store.list_runs(page_size=2)
    assert [item["run_id"] for item in first["items"]] == [
        "run-newest",
        "run-middle",
    ]
    assert first["next_page_token"] is not None

    await _create_run(store, "run-later-write", created_at="2026-07-27T13:00:00+00:00")
    await _create_run(store, "run-a-late-tie", created_at="2026-07-27T11:00:00+00:00")
    second = await store.list_runs(
        page_size=2,
        page_token=str(first["next_page_token"]),
    )

    assert [item["run_id"] for item in second["items"]] == ["run-oldest"]
    assert second["next_page_token"] is None
    fresh = await store.list_runs(page_size=2)
    assert [item["run_id"] for item in fresh["items"]] == [
        "run-later-write",
        "run-newest",
    ]


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


async def test_full_evidence_resume_persists_evidence_and_reenters_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = RunStore(tmp_path / "runs.db")
    await store.initialize()
    state = {
        "nl_input": "Design evidence-backed molecules",
        "run_id": "run-full-evidence",
        "trace_id": "trace-full-evidence",
        "artifact_ids": [],
        "events": [],
        "history": ["PLANNING", "GENERATING", "VALIDATING", "AWAITING_EVIDENCE"],
        "workflow_scope": "full",
        "status": "AWAITING_EVIDENCE",
        "validation_outcome": "AWAITING_EVIDENCE",
        "validation_passed": False,
        "refinement_count": 0,
        "max_refinements": 1,
        "candidates": [
            {
                "candidate_id": "candidate-1",
                "canonical_smiles": "CCO",
                "generator_name": "hfm_3d",
            }
        ],
        "request": {
            "project_id": "project-full-evidence",
            "nl_input": "Design evidence-backed molecules",
            "workflow_scope": "full",
            "validation_passed": False,
            "max_refinements": 1,
            "validation_policy": {"oracle_level": 4},
            "teacher_policy": {"teacher_source": "hypseek"},
            "selection_policy": {"criteria": []},
        },
    }
    _attach_l4_awaiting_validation(state)
    await store.create_run(
        "run-full-evidence",
        intent="Design evidence-backed molecules",
        policy={"workflow_scope": "full"},
        created_at="2026-07-27T10:00:00+00:00",
        state=state,
    )
    await store.transition_run(
        "run-full-evidence",
        {RunStatus.QUEUED},
        RunStatus.RUNNING,
        current_stage="validating",
        state=state,
    )
    await store.transition_run(
        "run-full-evidence",
        {RunStatus.RUNNING},
        RunStatus.AWAITING_EVIDENCE,
        current_stage="awaiting_evidence",
        state=state,
    )
    evidence = [
        {
            "candidate_id": "candidate-1",
            "canonical_smiles": "CCO",
            "metrics": {"activity": 0.81},
            "uncertainties": {"activity": 0.03},
            "evidence_ids": ["artifact:measurement-1"],
        }
    ]
    invocations: list[dict] = []
    provenance_calls: list[str] = []

    async def resumed_workflow(
        request: dict,
        resumed_state: dict,
        *,
        clients: object | None = None,
        run_control: RunControl | None = None,
        entry_point: str = "planning",
    ) -> dict:
        invocations.append(
            {
                "request": dict(request),
                "state": dict(resumed_state),
                "entry_point": entry_point,
            }
        )
        return {
            **resumed_state,
            "request": dict(request),
            "status": "EXECUTING",
            "validation_outcome": "PASS",
            "validation_passed": True,
            "events": list(resumed_state["events"]),
        }

    async def record_provenance(final_state: dict) -> None:
        provenance_calls.append(str(final_state["run_id"]))

    evidence_payload = {
        "schema_version": "external_validation_evidence.v1",
        "project_id": "project-full-evidence",
        "run_id": "run-full-evidence",
        "candidate_id": "candidate-1",
        "canonical_smiles": "CCO",
        "metrics": {"activity": 0.81},
        "uncertainties": {"activity": 0.03},
    }
    evidence_bytes = json.dumps(
        evidence_payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")

    async def fetch_record(artifact_id: str) -> dict:
        assert artifact_id == "artifact:measurement-1"
        return {
            "artifact_id": artifact_id,
            "artifact_type": "external_validation_evidence",
            "metadata": {
                "project_id": "project-full-evidence",
                "run_id": "run-full-evidence",
                "candidate_id": "candidate-1",
                "canonical_smiles": "CCO",
            },
            "payload_base64": base64.b64encode(evidence_bytes).decode("ascii"),
            "checksum": f"sha256:{hashlib.sha256(evidence_bytes).hexdigest()}",
            "signature": "sig-evidence",
            "signature_type": "sigstore_rekor",
            "recorded_at": "2026-07-29T00:00:00+00:00",
            "verified": True,
        }

    monkeypatch.setattr(orchestrator_main, "_invoke_workflow", resumed_workflow)
    monkeypatch.setattr(orchestrator_main, "_record_workflow_provenance", record_provenance)
    monkeypatch.setattr(orchestrator_main, "_fetch_provenance_record", fetch_record)
    monkeypatch.setattr(orchestrator_main, "_RUN_STORE", store)
    monkeypatch.setattr(orchestrator_main, "_RUN_CONTROL", RunControl(store))
    monkeypatch.setattr(orchestrator_main, "_RUN_INITIALIZED_STORE", store)
    monkeypatch.setattr(orchestrator_main, "_RUN_TASKS", {})

    transport = httpx.ASGITransport(app=orchestrator_main.rest_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/orchestrator/runs/run-full-evidence/evidence/resume",
            json={"external_evidence": evidence},
        )
        assert response.status_code == 202
        assert response.json() == {
            "design_id": "run-full-evidence",
            "run_id": "run-full-evidence",
            "status": "running",
        }
        for _ in range(100):
            snapshot = await store.get_run("run-full-evidence")
            if snapshot is not None and snapshot["status"] == "completed":
                break
            await asyncio.sleep(0)

    assert snapshot is not None
    assert snapshot["status"] == "completed"
    assert snapshot["state"]["request"]["external_evidence"] == evidence
    assert invocations[0]["entry_point"] == "validating"
    assert invocations[0]["request"]["external_evidence"] == evidence
    assert invocations[0]["state"]["candidates"] == state["candidates"]
    assert snapshot["state"]["external_evidence_artifacts"][0]["artifact_id"] == (
        "artifact:measurement-1"
    )
    assert len(snapshot["state"]["external_evidence_submissions"]) == 1
    assert provenance_calls == ["run-full-evidence"]


async def test_cancelled_full_evidence_resume_registers_persisted_run_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = RunStore(tmp_path / "runs.db")
    await store.initialize()
    state = {
        "run_id": "run-resume-cancel",
        "workflow_scope": "full",
        "status": "AWAITING_EVIDENCE",
        "candidates": [
            {
                "candidate_id": "candidate-1",
                "canonical_smiles": "CCO",
            }
        ],
        "request": {
            "project_id": "project-resume-cancel",
            "workflow_scope": "full",
        },
    }
    _attach_l4_awaiting_validation(state)
    await store.create_run(
        "run-resume-cancel",
        intent="Design evidence-backed molecules",
        policy={"workflow_scope": "full"},
        created_at="2026-07-29T00:00:00+00:00",
        state=state,
    )
    await store.transition_run(
        "run-resume-cancel",
        {RunStatus.QUEUED},
        RunStatus.RUNNING,
        current_stage="validating",
        state=state,
    )
    await store.transition_run(
        "run-resume-cancel",
        {RunStatus.RUNNING},
        RunStatus.AWAITING_EVIDENCE,
        current_stage="awaiting_evidence",
        state=state,
    )
    transition_started = threading.Event()
    release_transition = threading.Event()
    transition_finished = threading.Event()
    execution_started = asyncio.Event()
    release_execution = asyncio.Event()
    original_transition = store._transition_run

    def controlled_transition(*args: object, **kwargs: object) -> None:
        expected = args[1]
        target = args[2]
        if expected == {RunStatus.AWAITING_EVIDENCE} and target is RunStatus.RUNNING:
            transition_started.set()
            if not release_transition.wait(timeout=5):
                raise TimeoutError("timed out waiting to release evidence resume")
            try:
                original_transition(*args, **kwargs)
            finally:
                transition_finished.set()
            return
        original_transition(*args, **kwargs)

    async def verify_evidence(evidence: list[dict], current_state: dict) -> list[dict]:
        return [
            {
                "artifact_id": "artifact-measurement-1",
                "candidate_id": "candidate-1",
                "checksum": "sha256:" + "a" * 64,
                "signature": "sig-evidence",
                "signature_type": "sigstore_rekor",
                "recorded_at": "2026-07-29T00:00:00+00:00",
            }
        ]

    async def controlled_execution(
        run_id: str,
        request: dict,
        resumed_state: dict,
    ) -> None:
        execution_started.set()
        await release_execution.wait()

    monkeypatch.setattr(store, "_transition_run", controlled_transition)
    monkeypatch.setattr(
        orchestrator_main,
        "_verify_resume_external_evidence",
        verify_evidence,
    )
    monkeypatch.setattr(
        orchestrator_main,
        "_execute_evidence_resume_run",
        controlled_execution,
    )
    monkeypatch.setattr(orchestrator_main, "_RUN_STORE", store)
    monkeypatch.setattr(orchestrator_main, "_RUN_CONTROL", RunControl(store))
    monkeypatch.setattr(orchestrator_main, "_RUN_INITIALIZED_STORE", store)
    monkeypatch.setattr(orchestrator_main, "_RUN_TASKS", {})
    request_task = asyncio.create_task(
        orchestrator_main.resume_evidence_run(
            "run-resume-cancel",
            {
                "external_evidence": [
                    {
                        "candidate_id": "candidate-1",
                        "canonical_smiles": "CCO",
                        "metrics": {"activity": 0.81},
                        "uncertainties": {},
                        "evidence_ids": ["artifact-measurement-1"],
                    }
                ]
            },
        )
    )

    try:
        await _wait_for_thread_event(transition_started)
        request_task.cancel("request disconnected")
        await asyncio.sleep(0)
        release_transition.set()
        await _wait_for_thread_event(transition_finished)
        with pytest.raises(asyncio.CancelledError):
            await request_task

        snapshot = await store.get_run("run-resume-cancel")
        assert snapshot is not None
        assert snapshot["status"] == "running"
        owner_task = orchestrator_main._RUN_TASKS.get("run-resume-cancel")
        assert owner_task is not None
        await asyncio.wait_for(execution_started.wait(), timeout=5)
        assert not owner_task.done()
    finally:
        release_transition.set()
        release_execution.set()
        await asyncio.gather(request_task, return_exceptions=True)
        owner_task = orchestrator_main._RUN_TASKS.get("run-resume-cancel")
        if owner_task is not None:
            await asyncio.gather(owner_task, return_exceptions=True)


async def test_full_evidence_resume_rejects_unknown_candidate_without_state_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = RunStore(tmp_path / "runs.db")
    await store.initialize()
    state = {
        "run_id": "run-invalid-evidence",
        "workflow_scope": "full",
        "status": "AWAITING_EVIDENCE",
        "candidates": [
            {
                "candidate_id": "candidate-1",
                "canonical_smiles": "CCO",
            }
        ],
        "request": {
            "workflow_scope": "full",
            "validation_passed": False,
            "max_refinements": 1,
        },
    }
    await store.create_run(
        "run-invalid-evidence",
        intent="Design evidence-backed molecules",
        policy={"workflow_scope": "full"},
        created_at="2026-07-27T10:00:00+00:00",
        state=state,
    )
    await store.transition_run(
        "run-invalid-evidence",
        {RunStatus.QUEUED},
        RunStatus.RUNNING,
        current_stage="validating",
        state=state,
    )
    await store.transition_run(
        "run-invalid-evidence",
        {RunStatus.RUNNING},
        RunStatus.AWAITING_EVIDENCE,
        current_stage="awaiting_evidence",
        state=state,
    )
    monkeypatch.setattr(orchestrator_main, "_RUN_STORE", store)
    monkeypatch.setattr(orchestrator_main, "_RUN_CONTROL", RunControl(store))
    monkeypatch.setattr(orchestrator_main, "_RUN_INITIALIZED_STORE", store)
    monkeypatch.setattr(orchestrator_main, "_RUN_TASKS", {})

    transport = httpx.ASGITransport(app=orchestrator_main.rest_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/orchestrator/runs/run-invalid-evidence/evidence/resume",
            json={
                "external_evidence": [
                    {
                        "candidate_id": "candidate-other",
                        "metrics": {"activity": 0.81},
                        "evidence_ids": ["artifact:measurement-1"],
                    }
                ]
            },
        )

    snapshot = await store.get_run("run-invalid-evidence")
    assert response.status_code == 422
    assert response.json()["detail"] == (
        "external_evidence references unknown candidate_id: candidate-other"
    )
    assert snapshot is not None
    assert snapshot["status"] == "awaiting_evidence"
    assert "external_evidence" not in snapshot["state"]["request"]


def test_full_evidence_resume_merges_new_candidates_without_replacing_history() -> None:
    first = {
        "candidate_id": "candidate-1",
        "canonical_smiles": "CCO",
        "metrics": {"activity": 0.81},
        "uncertainties": {},
        "evidence_ids": ["artifact-measurement-1"],
    }
    second = {
        "candidate_id": "candidate-2",
        "canonical_smiles": "CCN",
        "metrics": {"activity": 0.84},
        "uncertainties": {},
        "evidence_ids": ["artifact-measurement-2"],
    }
    state = {
        "run_id": "run-merge-evidence",
        "workflow_scope": "full",
        "candidates": [
            {"candidate_id": "candidate-1", "canonical_smiles": "CCO"},
            {"candidate_id": "candidate-2", "canonical_smiles": "CCN"},
        ],
        "request": {
            "project_id": "project-merge-evidence",
            "workflow_scope": "full",
            "external_evidence": [first],
        },
    }
    _attach_l4_awaiting_validation(state)

    request, resumed_state = orchestrator_main._prepare_full_evidence_resume(
        {"external_evidence": [second]},
        state,
    )

    assert request["external_evidence"] == [first, second]
    assert resumed_state["request"]["external_evidence"] == [first, second]


def test_full_evidence_resume_rejects_changes_to_accepted_candidate_evidence() -> None:
    accepted = {
        "candidate_id": "candidate-1",
        "canonical_smiles": "CCO",
        "metrics": {"activity": 0.81},
        "uncertainties": {},
        "evidence_ids": ["artifact-measurement-1"],
    }
    state = {
        "run_id": "run-immutable-evidence",
        "workflow_scope": "full",
        "candidates": [
            {"candidate_id": "candidate-1", "canonical_smiles": "CCO"},
        ],
        "request": {
            "project_id": "project-immutable-evidence",
            "workflow_scope": "full",
            "external_evidence": [accepted],
        },
    }
    replacement = {
        **accepted,
        "metrics": {"activity": 0.99},
        "evidence_ids": ["artifact-measurement-replacement"],
    }

    with pytest.raises(ValueError, match="already accepted with different content"):
        orchestrator_main._prepare_full_evidence_resume(
            {"external_evidence": [replacement]},
            state,
        )


def test_external_evidence_metric_names_reject_normalized_collisions() -> None:
    with pytest.raises(ValueError, match="duplicate normalized metric"):
        orchestrator_main._validated_evidence_number_map(
            {"activity": 0.81, " activity ": 0.99},
            "external_evidence[0].metrics",
            require_nonempty=True,
            nonnegative=False,
        )


async def test_full_evidence_resume_rejects_artifact_bound_to_another_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = RunStore(tmp_path / "runs.db")
    await store.initialize()
    state = {
        "run_id": "run-bound-evidence",
        "workflow_scope": "full",
        "status": "AWAITING_EVIDENCE",
        "candidates": [
            {
                "candidate_id": "candidate-1",
                "canonical_smiles": "CCO",
            }
        ],
        "request": {
            "project_id": "project-bound-evidence",
            "workflow_scope": "full",
            "validation_passed": False,
            "max_refinements": 1,
        },
    }
    _attach_l4_awaiting_validation(state)
    await store.create_run(
        "run-bound-evidence",
        intent="Design evidence-backed molecules",
        policy={"workflow_scope": "full"},
        created_at="2026-07-27T10:00:00+00:00",
        state=state,
    )
    await store.transition_run(
        "run-bound-evidence",
        {RunStatus.QUEUED},
        RunStatus.RUNNING,
        current_stage="validating",
        state=state,
    )
    await store.transition_run(
        "run-bound-evidence",
        {RunStatus.RUNNING},
        RunStatus.AWAITING_EVIDENCE,
        current_stage="awaiting_evidence",
        state=state,
    )
    artifact_payload = {
        "schema_version": "external_validation_evidence.v1",
        "project_id": "project-bound-evidence",
        "run_id": "run-bound-evidence",
        "candidate_id": "candidate-other",
        "canonical_smiles": "CCO",
        "metrics": {"activity": 0.81},
        "uncertainties": {},
    }
    payload_bytes = json.dumps(
        artifact_payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")

    async def fetch_record(artifact_id: str) -> dict:
        assert artifact_id == "artifact-measurement-1"
        return {
            "artifact_id": artifact_id,
            "artifact_type": "external_validation_evidence",
            "metadata": {
                "project_id": "project-bound-evidence",
                "run_id": "run-bound-evidence",
                "candidate_id": "candidate-other",
                "canonical_smiles": "CCO",
            },
            "payload_base64": base64.b64encode(payload_bytes).decode("ascii"),
            "checksum": f"sha256:{hashlib.sha256(payload_bytes).hexdigest()}",
            "signature": "sig-evidence",
            "signature_type": "sigstore_rekor",
            "recorded_at": "2026-07-29T00:00:00+00:00",
            "verified": True,
        }

    monkeypatch.setattr(
        orchestrator_main,
        "_fetch_provenance_record",
        fetch_record,
        raising=False,
    )
    monkeypatch.setattr(orchestrator_main, "_RUN_STORE", store)
    monkeypatch.setattr(orchestrator_main, "_RUN_CONTROL", RunControl(store))
    monkeypatch.setattr(orchestrator_main, "_RUN_INITIALIZED_STORE", store)
    monkeypatch.setattr(orchestrator_main, "_RUN_TASKS", {})
    monkeypatch.setattr(
        orchestrator_main,
        "_register_evidence_resume_task",
        lambda *_args: None,
    )

    transport = httpx.ASGITransport(app=orchestrator_main.rest_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/orchestrator/runs/run-bound-evidence/evidence/resume",
            json={
                "external_evidence": [
                    {
                        "candidate_id": "candidate-1",
                        "canonical_smiles": "CCO",
                        "metrics": {"activity": 0.81},
                        "uncertainties": {},
                        "evidence_ids": ["artifact-measurement-1"],
                    }
                ]
            },
        )

    snapshot = await store.get_run("run-bound-evidence")
    assert response.status_code == 422
    assert "candidate_id mismatch" in response.json()["detail"]
    assert snapshot is not None
    assert snapshot["status"] == "awaiting_evidence"


async def test_full_evidence_resume_stream_appends_after_persisted_checkpoint(
    tmp_path: Path,
) -> None:
    store = RunStore(tmp_path / "runs.db")
    await store.initialize()
    state = {
        "nl_input": "Design evidence-backed molecules",
        "run_id": "run-resume-stream",
        "trace_id": "trace-resume-stream",
        "artifact_ids": [],
        "events": [
            {
                "event_index": index,
                "stage": stage,
                "timestamp": f"2026-07-27T10:00:0{index}+00:00",
            }
            for index, stage in enumerate(
                ["PLANNING", "GENERATING", "VALIDATING", "AWAITING_EVIDENCE"]
            )
        ],
        "history": ["PLANNING", "GENERATING", "VALIDATING", "AWAITING_EVIDENCE"],
        "workflow_scope": "full",
        "status": "AWAITING_EVIDENCE",
        "validation_passed": False,
        "refinement_count": 0,
        "max_refinements": 1,
        "candidates": [
            {
                "candidate_id": "candidate-1",
                "canonical_smiles": "CCO",
                "generator_name": "hfm_3d",
            }
        ],
    }
    request = {
        "workflow_scope": "full",
        "validation_passed": False,
        "max_refinements": 1,
        "external_evidence": [
            {
                "candidate_id": "candidate-1",
                "metrics": {"activity": 0.8},
                "evidence_ids": ["artifact:measurement-1"],
            }
        ],
    }
    state["request"] = request
    await store.create_run(
        "run-resume-stream",
        intent="Design evidence-backed molecules",
        policy={"workflow_scope": "full"},
        created_at="2026-07-27T10:00:00+00:00",
        state=state,
    )
    await store.transition_run(
        "run-resume-stream",
        {RunStatus.QUEUED},
        RunStatus.RUNNING,
        current_stage="validating",
        state=state,
    )
    for event in state["events"]:
        await store.append_event(
            "run-resume-stream",
            event["event_index"],
            stage=event["stage"],
            payload=event,
            timestamp=event["timestamp"],
        )
    calls: list[str] = []

    class _Clients:
        async def compile_intent(self, current_state: dict) -> dict:
            raise AssertionError("resume must not recompile intent")

        async def generate_candidates(self, current_state: dict) -> list[dict]:
            raise AssertionError("resume must not regenerate candidates")

        async def validate_candidates(self, current_state: dict) -> dict:
            calls.append("validate")
            return {
                "outcome": "PASS",
                "passed": True,
                "records": [],
                "results": [],
            }

        async def plan_routes(self, current_state: dict) -> dict:
            calls.append("retrosyn")
            return {"routes": [{"route_id": "route-1"}]}

        async def assess_supply(self, current_state: dict) -> dict:
            calls.append("supply")
            return {
                "route_id": "route-1",
                "supply_assessment": {"overall_feasibility": "available"},
            }

        async def compile_synthesis(self, current_state: dict) -> dict:
            calls.append("srb")
            return {
                "route_id": "route-1",
                "protocols": [{"route_id": "route-1", "ssp_id": "ssp-1"}],
            }

        async def review_candidates(self, current_state: dict) -> dict:
            calls.append("critic")
            return {"verdict": "pass"}

        async def execute_synthesis(self, current_state: dict) -> dict:
            calls.append("execute")
            return {
                "status": "executed",
                "route_id": "route-1",
                "protocols": current_state["srb"]["protocols"],
            }

    final_state = await orchestrator_main._invoke_workflow(
        request,
        state,
        clients=_Clients(),
        run_control=RunControl(store),
        entry_point="validating",
    )

    events = await store.list_events("run-resume-stream")
    assert calls == ["validate", "retrosyn", "supply", "srb", "critic", "execute"]
    assert final_state["status"] == "EXECUTING"
    assert [event["step_index"] for event in events] == list(
        range(len(final_state["events"]))
    )
    assert [event["stage"] for event in events[:4]] == [
        "PLANNING",
        "GENERATING",
        "VALIDATING",
        "AWAITING_EVIDENCE",
    ]


async def test_workflow_provenance_is_recorded_through_the_configured_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict] = []

    class RecordingClient:
        def __init__(self, *, timeout: float) -> None:
            assert timeout == 30.0

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback) -> None:
            return None

        async def post(self, url: str, *, json: dict) -> httpx.Response:
            calls.append({"url": url, "json": json})
            return httpx.Response(
                200,
                request=httpx.Request("POST", url),
                json={
                    "artifact_id": json["artifact_id"],
                    "signature": "sig-service",
                    "recorded_at": "2026-07-29T00:00:00+00:00",
                },
            )

    monkeypatch.setenv("PROVENANCE_SVC_URL", "http://provenance-svc:8010")
    monkeypatch.setattr(httpx, "AsyncClient", RecordingClient)
    monkeypatch.setattr(
        orchestrator_main,
        "build_shared_crg_repository_from_env",
        lambda: None,
    )
    state = {
        "run_id": "run-http-provenance",
        "trace_id": "trace-http-provenance",
        "workflow_scope": "full",
        "status": "CRITIC",
        "request": {"project_id": "project-http-provenance"},
        "artifact_ids": ["artifact-input"],
        "history": ["PLANNING", "CRITIC"],
        "candidates": [{"candidate_id": "candidate-1"}],
        "validation_passed": True,
        "retrosyn": {"routes": []},
        "supply": {},
        "srb": {},
        "critic": {"verdict": "pass"},
    }

    await orchestrator_main._record_workflow_provenance(state)

    assert len(calls) == 1
    assert calls[0]["url"] == "http://provenance-svc:8010/v1/provenance/record"
    assert calls[0]["json"]["artifact_id"] == (
        "artifact-run-http-provenance-workflow-state"
    )
    persisted_payload = json.loads(
        base64.b64decode(
            calls[0]["json"]["payload_base64"],
            validate=True,
        ).decode("utf-8")
    )
    assert persisted_payload["artifact_ids"] == ["artifact-input"]
    assert state["provenance"]["signature"] == "sig-service"


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


async def test_restart_preserves_runs_awaiting_external_evidence(
    tmp_path: Path,
) -> None:
    store = RunStore(tmp_path / "runs.db")
    await store.initialize()
    await _create_run(store, "awaiting")
    await store.transition_run(
        "awaiting",
        {RunStatus.QUEUED},
        RunStatus.RUNNING,
        current_stage="validating",
        state={
            "run_id": "awaiting",
            "workflow_scope": "full",
            "status": "AWAITING_EVIDENCE",
        },
    )
    await store.transition_run(
        "awaiting",
        {RunStatus.RUNNING},
        RunStatus.AWAITING_EVIDENCE,
        current_stage="awaiting_evidence",
    )

    restarted = RunStore(tmp_path / "runs.db")
    await restarted.initialize()
    count = await restarted.interrupt_active_runs()

    snapshot = await restarted.get_run("awaiting")
    assert count == 0
    assert snapshot is not None
    assert snapshot["status"] == "awaiting_evidence"
    assert snapshot["state"]["status"] == "AWAITING_EVIDENCE"


async def test_restart_interrupts_non_full_run_awaiting_in_memory_evidence(
    tmp_path: Path,
) -> None:
    store = RunStore(tmp_path / "runs.db")
    await store.initialize()
    await _create_run(store, "awaiting-engineering")
    await store.transition_run(
        "awaiting-engineering",
        {RunStatus.QUEUED},
        RunStatus.RUNNING,
        current_stage="validating",
        state={
            "run_id": "awaiting-engineering",
            "workflow_scope": "engineering",
            "status": "awaiting_evidence",
        },
    )
    await store.transition_run(
        "awaiting-engineering",
        {RunStatus.RUNNING},
        RunStatus.AWAITING_EVIDENCE,
        current_stage="awaiting_evidence",
    )

    restarted = RunStore(tmp_path / "runs.db")
    await restarted.initialize()
    count = await restarted.interrupt_active_runs()

    snapshot = await restarted.get_run("awaiting-engineering")
    assert count == 1
    assert snapshot is not None
    assert snapshot["status"] == "interrupted"
    assert snapshot["error_type"] == "ServiceRestart"


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


async def test_background_run_persists_real_started_at_and_terminal_devices(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = RunStore(tmp_path / "runs.db")
    await store.initialize()
    control = RunControl(store)
    await _create_run(store, "run-runtime-fields")

    async def completed_workflow(
        request: dict,
        state: dict,
        *,
        run_control: RunControl | None = None,
    ) -> dict:
        assert isinstance(state.get("started_at"), str)
        return {
            **state,
            "status": "CRITIC",
            "history": ["PLANNING", "CRITIC"],
            "events": [],
            "devices_used": ["cuda:0"],
            "summary": "Workflow completed",
        }

    monkeypatch.setattr(orchestrator_main, "_invoke_workflow", completed_workflow)
    monkeypatch.setattr(orchestrator_main, "_RUN_STORE", store)
    monkeypatch.setattr(orchestrator_main, "_RUN_CONTROL", control)
    monkeypatch.setattr(orchestrator_main, "_RUN_INITIALIZED_STORE", store)

    await orchestrator_main._execute_design_run(
        "run-runtime-fields",
        {"workflow_scope": "engineering"},
        {
            "run_id": "run-runtime-fields",
            "trace_id": "trace-runtime-fields",
            "status": "PLANNING",
        },
    )

    snapshot = await store.get_run("run-runtime-fields")
    assert snapshot is not None
    assert snapshot["status"] == "completed"
    assert isinstance(snapshot["state"]["started_at"], str)
    assert snapshot["devices_used"] == ["cuda:0"]
    assert snapshot["summary"] == "Workflow completed"


@pytest.mark.parametrize(
    ("final_status", "expected_stage"),
    [
        pytest.param(None, "<empty>", id="missing"),
        pytest.param("VALIDATING", "VALIDATING", id="nonterminal"),
    ],
)
async def test_execute_design_run_rejects_nonterminal_workflow_result_and_persists_failure(
    final_status: str | None,
    expected_stage: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = f"run-invalid-terminal-{expected_stage.lower().strip('<>')}"
    store = RunStore(tmp_path / "runs.db")
    await store.initialize()
    control = RunControl(store)
    await _create_run(store, run_id)

    async def incomplete_workflow(
        request: dict,
        state: dict,
        *,
        run_control: RunControl | None = None,
    ) -> dict:
        result = {**state, "history": [], "events": []}
        if final_status is None:
            result.pop("status", None)
        else:
            result["status"] = final_status
        return result

    monkeypatch.setattr(orchestrator_main, "_invoke_workflow", incomplete_workflow)
    monkeypatch.setattr(orchestrator_main, "_RUN_STORE", store)
    monkeypatch.setattr(orchestrator_main, "_RUN_CONTROL", control)
    monkeypatch.setattr(orchestrator_main, "_RUN_INITIALIZED_STORE", store)

    error_message = f"WorkflowGraph returned non-terminal stage: {expected_stage}"
    with pytest.raises(RuntimeError, match=error_message):
        await orchestrator_main._execute_design_run(
            run_id,
            {"workflow_scope": "state_only"},
            {
                "run_id": run_id,
                "trace_id": f"trace-{run_id}",
                "status": "PLANNING",
            },
        )

    snapshot = await store.get_run(run_id)
    assert snapshot is not None
    assert snapshot["status"] == "failed"
    assert snapshot["error_type"] == "RuntimeError"
    assert snapshot["error_message"] == error_message


async def test_legacy_successful_workflow_completes_even_when_critic_escalates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = RunStore(tmp_path / "runs.db")
    await store.initialize()
    control = RunControl(store)

    async def critic_failed_workflow(
        request: dict,
        state: dict,
        *,
        clients: object | None = None,
        run_control: RunControl | None = None,
    ) -> dict:
        return {
            **state,
            "status": "ESCALATING",
            "history": ["PLANNING", "CRITIC", "ESCALATING"],
            "events": [],
            "critic": {"verdict": "fail", "reason": "quality gate"},
            "validation": {
                "passed": False,
                "results": [{"candidate_id": "candidate-1", "valid": False}],
                "reason": "quality gate failed",
            },
            "validation_passed": False,
        }

    monkeypatch.setattr(orchestrator_main, "_invoke_workflow", critic_failed_workflow)
    monkeypatch.setattr(orchestrator_main, "_RUN_STORE", store)
    monkeypatch.setattr(orchestrator_main, "_RUN_CONTROL", control)
    monkeypatch.setattr(orchestrator_main, "_RUN_INITIALIZED_STORE", store)

    response = await orchestrator_main.start_design(
        {
            "intent": 'Legacy molecular design: {"constraints":{},"objectives":["qed"]}',
            "workflow_scope": "engineering",
            "validation_passed": True,
            "max_refinements": 0,
            "run_id": "design-1234567890",
            "_mforge_internal_legacy_design_request": True,
            "clients": object(),
        }
    )

    snapshot = await store.get_run("design-1234567890")
    assert response["status"] == "completed"
    assert snapshot is not None
    assert snapshot["status"] == "completed"


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


async def test_cancelled_rest_creation_after_persistence_registers_run_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = RunStore(tmp_path / "runs.db")
    await store.initialize()
    control = RunControl(store)
    snapshot_read_started = asyncio.Event()
    release_execution = asyncio.Event()
    original_get_run = store.get_run

    async def controlled_get_run(run_id: str) -> dict | None:
        if run_id == "run-rest-post-create-cancel":
            snapshot_read_started.set()
            await asyncio.Event().wait()
        return await original_get_run(run_id)

    async def controlled_execution(
        run_id: str,
        request: dict,
        state: dict,
    ) -> None:
        await release_execution.wait()

    monkeypatch.setattr(store, "get_run", controlled_get_run)
    monkeypatch.setattr(orchestrator_main, "_execute_design_run", controlled_execution)
    monkeypatch.setattr(orchestrator_main, "_RUN_STORE", store)
    monkeypatch.setattr(orchestrator_main, "_RUN_CONTROL", control)
    monkeypatch.setattr(orchestrator_main, "_RUN_INITIALIZED_STORE", store)
    monkeypatch.setattr(orchestrator_main, "_RUN_TASKS", {})
    request_task = asyncio.create_task(
        orchestrator_main.create_design_run(
            {
                "nl_input": "Create an owned run after persistence",
                "workflow_scope": "state_only",
                "validation_passed": True,
                "max_refinements": 0,
                "run_id": "run-rest-post-create-cancel",
            }
        )
    )

    try:
        await asyncio.wait_for(snapshot_read_started.wait(), timeout=5)
        request_task.cancel("REST request disconnected after persistence")
        with pytest.raises(asyncio.CancelledError):
            await request_task

        snapshot = await original_get_run("run-rest-post-create-cancel")
        assert snapshot is not None
        assert snapshot["status"] == "queued"
        owner_task = orchestrator_main._RUN_TASKS.get("run-rest-post-create-cancel")
        assert owner_task is not None
        assert not owner_task.done()
    finally:
        release_execution.set()
        await asyncio.gather(request_task, return_exceptions=True)
        owner_task = orchestrator_main._RUN_TASKS.get("run-rest-post-create-cancel")
        if owner_task is not None:
            await asyncio.gather(owner_task, return_exceptions=True)


async def test_rest_creation_registers_owner_before_snapshot_read_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = RunStore(tmp_path / "runs.db")
    await store.initialize()
    control = RunControl(store)
    original_get_run = store.get_run
    execution_started = asyncio.Event()
    release_execution = asyncio.Event()
    execution_count = 0

    async def failing_get_run(run_id: str) -> dict | None:
        if run_id == "run-rest-snapshot-failure":
            raise RuntimeError("snapshot read failed")
        return await original_get_run(run_id)

    async def controlled_execution(
        run_id: str,
        request: dict,
        state: dict,
    ) -> None:
        nonlocal execution_count
        execution_count += 1
        execution_started.set()
        await release_execution.wait()

    monkeypatch.setattr(store, "get_run", failing_get_run)
    monkeypatch.setattr(orchestrator_main, "_execute_design_run", controlled_execution)
    monkeypatch.setattr(orchestrator_main, "_RUN_STORE", store)
    monkeypatch.setattr(orchestrator_main, "_RUN_CONTROL", control)
    monkeypatch.setattr(orchestrator_main, "_RUN_INITIALIZED_STORE", store)
    monkeypatch.setattr(orchestrator_main, "_RUN_TASKS", {})

    try:
        with pytest.raises(RuntimeError, match="snapshot read failed"):
            await orchestrator_main.create_design_run(
                {
                    "nl_input": "Create an owned run before snapshot failure",
                    "workflow_scope": "state_only",
                    "validation_passed": True,
                    "max_refinements": 0,
                    "run_id": "run-rest-snapshot-failure",
                }
            )

        snapshot = await original_get_run("run-rest-snapshot-failure")
        assert snapshot is not None
        owner_task = orchestrator_main._RUN_TASKS.get("run-rest-snapshot-failure")
        assert owner_task is not None
        await asyncio.wait_for(execution_started.wait(), timeout=5)
        assert execution_count == 1
        assert not owner_task.done()
    finally:
        release_execution.set()
        owner_task = orchestrator_main._RUN_TASKS.get("run-rest-snapshot-failure")
        if owner_task is not None:
            await asyncio.gather(owner_task, return_exceptions=True)


async def test_rest_json_clients_are_not_forwarded_to_workflow_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = RunStore(tmp_path / "runs.db")
    await store.initialize()
    control = RunControl(store)
    execution_started = asyncio.Event()
    release_execution = asyncio.Event()
    execution_requests: list[dict] = []

    async def controlled_execution(
        run_id: str,
        request: dict,
        state: dict,
    ) -> None:
        execution_requests.append(dict(request))
        execution_started.set()
        await release_execution.wait()

    monkeypatch.setattr(orchestrator_main, "_execute_design_run", controlled_execution)
    monkeypatch.setattr(orchestrator_main, "_RUN_STORE", store)
    monkeypatch.setattr(orchestrator_main, "_RUN_CONTROL", control)
    monkeypatch.setattr(orchestrator_main, "_RUN_INITIALIZED_STORE", store)
    monkeypatch.setattr(orchestrator_main, "_RUN_TASKS", {})
    transport = httpx.ASGITransport(app=orchestrator_main.rest_app)

    try:
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/v1/orchestrator/design",
                json={
                    "nl_input": "Ignore REST client injection",
                    "workflow_scope": "state_only",
                    "validation_passed": True,
                    "max_refinements": 0,
                    "run_id": "run-rest-clients",
                    "clients": {},
                },
            )

        assert response.status_code == 202
        await asyncio.wait_for(execution_started.wait(), timeout=5)
        assert execution_requests == [
            {
                "nl_input": "Ignore REST client injection",
                "workflow_scope": "state_only",
                "validation_passed": True,
                "max_refinements": 0,
                "run_id": "run-rest-clients",
            }
        ]
        snapshot = await store.get_run("run-rest-clients")
        assert snapshot is not None
        assert "clients" not in snapshot["state"]["request"]
    finally:
        release_execution.set()
        owner_task = orchestrator_main._RUN_TASKS.get("run-rest-clients")
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


async def test_cancel_endpoint_interrupts_active_inline_run(
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
    monkeypatch.setattr(orchestrator_main, "_RUN_TASKS", {})
    direct_task = asyncio.create_task(
        orchestrator_main.start_design(
            {
                "nl_input": "Design an inline run cancellable through REST",
                "workflow_scope": "state_only",
                "validation_passed": True,
                "max_refinements": 0,
                "run_id": "run-inline-rest-cancel",
            }
        )
    )
    transport = httpx.ASGITransport(app=orchestrator_main.rest_app)

    try:
        await asyncio.wait_for(entered.wait(), timeout=5)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/v1/orchestrator/runs/run-inline-rest-cancel/cancel")

        assert response.status_code == 200
        assert response.json()["status"] == "interrupted"
        with pytest.raises(asyncio.CancelledError):
            await direct_task
        snapshot = await store.get_run("run-inline-rest-cancel")
        assert snapshot is not None
        assert snapshot["status"] == "interrupted"
        assert snapshot["error_type"] == "CancelledError"
        assert "run-inline-rest-cancel" not in orchestrator_main._RUN_TASKS
    finally:
        release.set()
        if not direct_task.done():
            direct_task.cancel()
        await asyncio.gather(direct_task, return_exceptions=True)


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


@pytest.mark.parametrize(
    "run_id",
    [
        "run/child",
        "run?query",
        "run#fragment",
        "run%2Fchild",
        ".",
        "..",
        "run\\child",
    ],
)
async def test_orchestrator_rejects_unaddressable_caller_run_ids_without_persisting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    run_id: str,
) -> None:
    store = RunStore(tmp_path / "runs.db")
    await store.initialize()
    monkeypatch.setattr(orchestrator_main, "_RUN_STORE", store)
    monkeypatch.setattr(orchestrator_main, "_RUN_CONTROL", RunControl(store))
    monkeypatch.setattr(orchestrator_main, "_RUN_INITIALIZED_STORE", store)
    monkeypatch.setattr(orchestrator_main, "_RUN_TASKS", {})
    transport = httpx.ASGITransport(app=orchestrator_main.rest_app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/orchestrator/design",
            json={
                "nl_input": "Design an addressable molecule",
                "workflow_scope": "state_only",
                "validation_passed": True,
                "max_refinements": 0,
                "run_id": run_id,
            },
        )

    assert response.status_code == 400
    assert "run_id" in response.json()["detail"]
    assert (await store.list_runs(page_size=10))["items"] == []
    assert orchestrator_main._RUN_TASKS == {}


@pytest.mark.parametrize("project_id", [None, "", {}, [], False, 7])
async def test_orchestrator_rejects_invalid_run_project_id_before_create(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    project_id: object,
) -> None:
    store = RunStore(tmp_path / "runs.db")
    await store.initialize()
    create_calls = 0
    original_create_run = store._create_run

    def tracked_create_run(*args: object, **kwargs: object) -> None:
        nonlocal create_calls
        create_calls += 1
        original_create_run(*args, **kwargs)

    monkeypatch.setattr(store, "_create_run", tracked_create_run)
    monkeypatch.setattr(orchestrator_main, "_RUN_STORE", store)
    monkeypatch.setattr(orchestrator_main, "_RUN_CONTROL", RunControl(store))
    monkeypatch.setattr(orchestrator_main, "_RUN_INITIALIZED_STORE", store)
    monkeypatch.setattr(orchestrator_main, "_RUN_TASKS", {})
    transport = httpx.ASGITransport(
        app=orchestrator_main.rest_app,
        raise_app_exceptions=False,
    )

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/orchestrator/design",
            json={
                "nl_input": "Design a molecule",
                "workflow_scope": "state_only",
                "validation_passed": True,
                "max_refinements": 0,
                "run_id": "run-invalid-project",
                "project_id": project_id,
            },
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "project_id must be a non-empty string"
    assert create_calls == 0
    assert (await store.list_runs(page_size=10))["items"] == []
    assert orchestrator_main._RUN_TASKS == {}


@pytest.mark.parametrize("project_id", [".", "..", "team/../secret", "team/./member"])
async def test_orchestrator_rejects_project_dot_segments_without_persisting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    project_id: str,
) -> None:
    store = RunStore(tmp_path / "projects.db")
    await store.initialize()
    monkeypatch.setattr(orchestrator_main, "_RUN_STORE", store)
    monkeypatch.setattr(orchestrator_main, "_RUN_CONTROL", RunControl(store))
    monkeypatch.setattr(orchestrator_main, "_RUN_INITIALIZED_STORE", store)
    transport = httpx.ASGITransport(app=orchestrator_main.rest_app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/orchestrator/projects",
            json={"name": project_id, "description": "invalid path identity"},
        )

    assert response.status_code == 400
    assert "project_id" in response.json()["detail"]
    assert await store.list_projects() == []


@pytest.mark.parametrize("workflow_scope", ["enginering", "unknown", "FULL", 42])
async def test_orchestrator_rejects_unsupported_workflow_scope_before_persisting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    workflow_scope: object,
) -> None:
    store = RunStore(tmp_path / "runs.db")
    await store.initialize()
    monkeypatch.setattr(orchestrator_main, "_RUN_STORE", store)
    monkeypatch.setattr(orchestrator_main, "_RUN_CONTROL", RunControl(store))
    monkeypatch.setattr(orchestrator_main, "_RUN_INITIALIZED_STORE", store)
    monkeypatch.setattr(orchestrator_main, "_RUN_TASKS", {})
    transport = httpx.ASGITransport(app=orchestrator_main.rest_app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/orchestrator/design",
            json={
                "nl_input": "Design a molecule",
                "workflow_scope": workflow_scope,
                "validation_passed": True,
                "max_refinements": 0,
                "run_id": "run-invalid-scope",
            },
        )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "workflow_scope must be one of: state_only, engineering, full"
    )
    assert await store.get_run("run-invalid-scope") is None
    assert orchestrator_main._RUN_TASKS == {}


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
        assert [item["run_id"] for item in first.json()["runs"]] == ["run-c", "run-b"]
        token = first.json()["next_page_token"]
        assert isinstance(token, str)

        second = await client.get(
            "/v1/orchestrator/runs",
            params={"page_size": 2, "page_token": token},
        )
        assert second.status_code == 200
        assert [item["run_id"] for item in second.json()["runs"]] == ["run-a"]

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


async def test_grpc_full_pipeline_forwards_explicit_json_policies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mf_core.proto_gen.moleculeforge.v1.agent import orchestrator_pb2

    validation_policy = {
        "oracle_level": 0,
        "batch_size": 8,
        "max_concurrency": 2,
        "thresholds": [
            {
                "level": 0,
                "oracle": "rdkit",
                "metric": "qed",
                "direction": "maximize",
                "value": 0.5,
            }
        ],
        "oracle_inputs": {},
    }
    teacher_policy = {
        "teacher_source": "hypseek",
        "teacher_version": "2026-07-29",
        "allow_synthetic": False,
        "kd_weight": 0.25,
    }
    selection_policy = {"criteria": [{"metric": "qed", "direction": "maximize"}]}
    external_evidence = [
        {
            "candidate_id": "candidate-1",
            "metrics": {"experimental_activity": 0.9},
            "evidence_ids": ["evidence-1"],
        }
    ]
    captured: list[dict] = []

    async def fake_start_design(request: dict) -> dict:
        captured.append(request)
        return {
            "design_id": "run-grpc-full",
            "run_id": "run-grpc-full",
            "trace_id": "trace-grpc-full",
            "status": "queued",
        }

    monkeypatch.setattr(orchestrator_main, "start_design", fake_start_design)
    request = orchestrator_pb2.StartPipelineRequest(
        project_id="project-grpc-full",
        nl_input="Design a validated molecule",
        workflow_scope="full",
        run_id="run-grpc-full",
        trace_id="trace-grpc-full",
        max_refinements=1,
        validation_policy_json=json.dumps(validation_policy),
        teacher_policy_json=json.dumps(teacher_policy),
        selection_policy_json=json.dumps(selection_policy),
        external_evidence_json=json.dumps(external_evidence),
    )

    response = await orchestrator_main.OrchestratorServicer().StartPipeline(
        request,
        None,
    )

    assert response.status == "queued"
    assert captured == [
        {
            "project_id": "project-grpc-full",
            "nl_input": "Design a validated molecule",
            "workflow_scope": "full",
            "run_id": "run-grpc-full",
            "trace_id": "trace-grpc-full",
            "max_refinements": 1,
            "validation_policy": validation_policy,
            "teacher_policy": teacher_policy,
            "selection_policy": selection_policy,
            "external_evidence": external_evidence,
        }
    ]


async def test_grpc_full_pipeline_rejects_invalid_policy_json_before_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mf_core.proto_gen.moleculeforge.v1.agent import orchestrator_pb2

    called = False

    async def fake_start_design(request: dict) -> dict:
        nonlocal called
        called = True
        raise AssertionError("invalid gRPC policy JSON must not start a run")

    monkeypatch.setattr(orchestrator_main, "start_design", fake_start_design)
    request = orchestrator_pb2.StartPipelineRequest(
        nl_input="Design a validated molecule",
        workflow_scope="full",
        max_refinements=1,
        validation_policy_json="{",
        teacher_policy_json="{}",
        selection_policy_json="{}",
    )

    with pytest.raises(orchestrator_main.HTTPException) as error:
        await orchestrator_main.OrchestratorServicer().StartPipeline(request, None)

    assert error.value.status_code == 422
    assert error.value.detail == "validation_policy_json must contain valid JSON"
    assert called is False


async def test_grpc_resume_evidence_forwards_typed_external_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mf_core.proto_gen.moleculeforge.v1.agent import orchestrator_pb2

    evidence = [
        {
            "candidate_id": "candidate-1",
            "metrics": {"activity": 0.8},
            "evidence_ids": ["artifact:measurement-1"],
        }
    ]
    captured: list[tuple[str, dict]] = []

    async def fake_resume(run_id: str, request: dict | None = None) -> dict:
        assert request is not None
        captured.append((run_id, request))
        return {
            "design_id": run_id,
            "run_id": run_id,
            "status": "running",
        }

    monkeypatch.setattr(orchestrator_main, "resume_evidence_run", fake_resume)
    request = orchestrator_pb2.ResumeEvidenceRequest(
        run_id="run-resume-grpc",
        external_evidence_json=json.dumps(evidence),
    )

    response = await orchestrator_main.OrchestratorServicer().ResumeEvidence(
        request,
        None,
    )

    assert response.design_id == "run-resume-grpc"
    assert response.run_id == "run-resume-grpc"
    assert response.status == "running"
    assert captured == [
        (
            "run-resume-grpc",
            {"external_evidence": evidence},
        )
    ]


def test_legacy_store_run_write_bypasses_are_not_exposed() -> None:
    from mf_core.db import store

    assert not hasattr(store, "insert_run")
    assert not hasattr(store, "update_run")


async def test_cancelled_failure_recording_interrupts_evidence_resume_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = RunStore(tmp_path / "runs.db")
    await store.initialize()
    state = {
        "run_id": "run-resume-failure-cancel",
        "workflow_scope": "full",
        "request": {"workflow_scope": "full"},
    }
    await store.create_run(
        "run-resume-failure-cancel",
        intent="Resume a persisted evidence workflow",
        policy={"workflow_scope": "full"},
        created_at="2026-07-30T00:00:00+00:00",
        state=state,
    )
    await store.transition_run(
        "run-resume-failure-cancel",
        {RunStatus.QUEUED},
        RunStatus.RUNNING,
        current_stage="validating",
        state=state,
    )
    original_get_run = store.get_run
    snapshot_reads = 0

    async def failed_workflow(*args: object, **kwargs: object) -> dict:
        raise RuntimeError("validation failed")

    async def cancelled_snapshot_read(run_id: str) -> dict | None:
        nonlocal snapshot_reads
        snapshot_reads += 1
        if snapshot_reads == 1:
            raise asyncio.CancelledError("resume owner cancelled")
        return await original_get_run(run_id)

    monkeypatch.setattr(orchestrator_main, "_invoke_workflow", failed_workflow)
    monkeypatch.setattr(store, "get_run", cancelled_snapshot_read)
    monkeypatch.setattr(orchestrator_main, "_RUN_STORE", store)
    monkeypatch.setattr(orchestrator_main, "_RUN_CONTROL", RunControl(store))
    monkeypatch.setattr(orchestrator_main, "_RUN_INITIALIZED_STORE", store)

    with pytest.raises(asyncio.CancelledError, match="resume owner cancelled"):
        await orchestrator_main._execute_evidence_resume_run(
            "run-resume-failure-cancel",
            dict(state["request"]),
            dict(state),
        )

    snapshot = await original_get_run("run-resume-failure-cancel")
    assert snapshot is not None
    assert snapshot["status"] == RunStatus.INTERRUPTED.value
    assert snapshot["error_type"] == asyncio.CancelledError.__name__


async def test_orchestrator_rejects_unauthenticated_and_cross_owner_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = RunStore(tmp_path / "runs.db")
    await store.initialize()
    state = {
        "run_id": "run-owned-resume",
        "workflow_scope": "full",
        "status": "AWAITING_EVIDENCE",
        "candidates": [{"candidate_id": "candidate-1", "canonical_smiles": "CCO"}],
        "request": {
            "project_id": "project-owned",
            "workflow_scope": "full",
        },
    }
    await store.create_run(
        "run-owned-resume",
        intent="Resume an owned run",
        policy={"workflow_scope": "full"},
        created_at="2026-07-30T00:00:00+00:00",
        state=state,
        owner_principal_id="scientist-1",
    )
    await store.transition_run(
        "run-owned-resume",
        {RunStatus.QUEUED},
        RunStatus.RUNNING,
        current_stage="validating",
        state=state,
    )
    await store.transition_run(
        "run-owned-resume",
        {RunStatus.RUNNING},
        RunStatus.AWAITING_EVIDENCE,
        current_stage="awaiting_evidence",
        state=state,
    )
    monkeypatch.setenv("INTERNAL_SERVICE_TOKEN", "orchestrator-service-token")
    monkeypatch.setattr(orchestrator_main, "_RUN_STORE", store)
    monkeypatch.setattr(orchestrator_main, "_RUN_CONTROL", RunControl(store))
    monkeypatch.setattr(orchestrator_main, "_RUN_INITIALIZED_STORE", store)
    transport = httpx.ASGITransport(app=orchestrator_main.rest_app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        unauthenticated = await client.post(
            "/v1/orchestrator/runs/run-owned-resume/evidence/resume",
            json={"external_evidence": []},
        )
        cross_owner = await client.post(
            "/v1/orchestrator/runs/run-owned-resume/evidence/resume",
            json={"external_evidence": []},
            headers={
                "X-MoleculeForge-Service-Token": "orchestrator-service-token",
                "X-MoleculeForge-Principal": "scientist-2",
            },
        )

    assert unauthenticated.status_code == 401
    assert cross_owner.status_code == 403
    assert cross_owner.json() == {"detail": "Run owner does not match authenticated principal"}


async def test_orchestrator_grpc_requires_service_token_and_propagates_principal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import grpc

    monkeypatch.setenv("INTERNAL_SERVICE_TOKEN", "grpc-service-token")
    calls: list[str | None] = []

    class _Abort(Exception):
        def __init__(self, code: grpc.StatusCode, detail: str) -> None:
            super().__init__(detail)
            self.code = code

    class _Context:
        def __init__(self, metadata: tuple[tuple[str, str], ...]) -> None:
            self._metadata = metadata

        def invocation_metadata(self) -> tuple[tuple[str, str], ...]:
            return self._metadata

        async def abort(self, code: grpc.StatusCode, detail: str) -> None:
            raise _Abort(code, detail)

    class _Service:
        async def ResumeEvidence(self, request: object, context: object):
            calls.append(orchestrator_main._current_service_principal())
            return type(
                "Response",
                (),
                {"design_id": "run-1", "run_id": "run-1", "status": "running"},
            )()

    servicer = orchestrator_main.OrchestratorGrpcServicer(service=_Service())

    with pytest.raises(_Abort) as unauthenticated:
        await servicer.ResumeEvidence(object(), _Context(()))
    response = await servicer.ResumeEvidence(
        object(),
        _Context(
            (
                ("x-moleculeforge-service-token", "grpc-service-token"),
                ("x-moleculeforge-principal", "scientist-1"),
            )
        ),
    )

    assert unauthenticated.value.code is grpc.StatusCode.UNAUTHENTICATED
    assert response.run_id == "run-1"
    assert calls == ["scientist-1"]


def test_external_evidence_rejects_artifact_count_above_configured_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EXTERNAL_EVIDENCE_MAX_ARTIFACTS", "2")

    with pytest.raises(ValueError, match="at most 2 provenance artifacts"):
        orchestrator_main._validate_resume_external_evidence(
            [
                {
                    "candidate_id": "candidate-1",
                    "metrics": {"activity": 0.8},
                    "evidence_ids": ["artifact-1", "artifact-2", "artifact-3"],
                }
            ],
            [{"candidate_id": "candidate-1", "canonical_smiles": "CCO"}],
            {"request": {"validation_policy": {"thresholds": []}}},
        )


def _full_downstream_state() -> dict:
    candidate = {
        "candidate_id": "candidate-1",
        "canonical_smiles": "CCO",
        "generator_name": "hfm_3d",
    }
    metric = {
        "level": 0,
        "oracle": "rdkit",
        "metric": "qed",
        "value": 0.8,
        "direction": "maximize",
        "threshold": 0.5,
        "passed": True,
    }
    validation = {
        "schema_version": "validation.record.v1",
        "candidate_id": "candidate-1",
        "canonical_smiles": "CCO",
        "outcome": "PASS",
        "metrics": [metric],
        "evidence": [
            {
                "evidence_id": "evidence-downstream:L0:rdkit",
                "level": 0,
                "oracle": "rdkit",
            }
        ],
        "levels": [
            {
                "level": 0,
                "outcome": "PASS",
                "oracles": [
                    {
                        "oracle": "rdkit",
                        "outcome": "PASS",
                        "metrics": [metric],
                        "evidence_ids": ["evidence-downstream:L0:rdkit"],
                    }
                ],
            }
        ],
    }
    return {
        "run_id": "run-downstream",
        "trace_id": "trace-downstream",
        "request": {
            "project_id": "project-1",
            "retrosyn_engine": "rsgpt",
            "validation_policy": {
                "oracle_level": 0,
                "batch_size": 8,
                "max_concurrency": 2,
                "thresholds": [
                    {
                        "level": 0,
                        "oracle": "rdkit",
                        "metric": "qed",
                        "direction": "maximize",
                        "value": 0.5,
                    }
                ],
                "oracle_inputs": {},
            },
            "selection_policy": {
                "criteria": [{"metric": "qed", "direction": "maximize"}],
            },
        },
        "candidates": [candidate],
        "validation": {
            "outcome": "PASS",
            "records": [validation],
            "results": [validation],
        },
        "retrosyn": {
            "routes": [
                {
                    "route_id": "route-1",
                    "building_blocks": [{"smiles": "CC"}],
                    "steps": [],
                }
            ]
        },
    }


class _FullDownstreamRequestClient:
    def __init__(self, responder) -> None:
        self.responder = responder
        self.calls: list[dict] = []

    async def request(self, subject, payload, *, payload_type_url, timeout):
        self.calls.append(
            {
                "subject": subject,
                "payload": dict(payload),
                "payload_type_url": payload_type_url,
                "timeout": timeout,
            }
        )
        response = dict(self.responder(subject, dict(payload)))
        for field in (
            "project_id",
            "candidate_id",
            "candidate_index",
            "canonical_smiles",
            "run_id",
            "request_id",
            "schema_version",
        ):
            if field in payload:
                response.setdefault(field, payload[field])
        return response


async def test_full_workflow_service_binds_retrosyn_and_supply_to_selected_route() -> None:
    def respond(subject: str, payload: dict) -> dict:
        if subject == "agent.retrosyn.request":
            return {"status": "planned", "routes": [{"route_id": "route-1"}]}
        if subject == "agent.supply.request":
            return {
                "status": "assessed",
                "route_id": payload["route_id"],
                "supply_assessment": {"overall_feasibility": "available"},
            }
        raise AssertionError(f"unexpected subject: {subject}")

    request_client = _FullDownstreamRequestClient(respond)
    clients = orchestrator_main.FullWorkflowClients(request_client)
    state = _full_downstream_state()

    retrosyn = await clients.plan_routes(state)
    supply = await clients.assess_supply(state)

    assert retrosyn["routes"][0]["route_id"] == "route-1"
    assert supply["route_id"] == "route-1"
    assert request_client.calls[0]["payload"]["engine"] == "rsgpt"
    assert request_client.calls[1]["payload"]["workflow_scope"] == "full"
    assert request_client.calls[1]["payload"]["route_id"] == "route-1"


async def test_full_workflow_service_selects_available_route_after_supply_checks() -> None:
    def respond(subject: str, payload: dict) -> dict:
        assert subject == "agent.supply.request"
        feasibility = "unavailable" if payload["route_id"] == "route-1" else "available"
        return {
            "status": "assessed",
            "route_id": payload["route_id"],
            "supply_assessment": {"overall_feasibility": feasibility},
        }

    request_client = _FullDownstreamRequestClient(respond)
    state = _full_downstream_state()
    state["retrosyn"]["routes"].append(
        {
            "route_id": "route-2",
            "steps": [
                {
                    "building_blocks": [
                        {"smiles": "CO"},
                        {"smiles": "CN"},
                    ]
                }
            ],
        }
    )

    supply = await orchestrator_main.FullWorkflowClients(
        request_client
    ).assess_supply(state)

    assert supply["route_id"] == "route-2"
    assert [
        call["payload"]["route_id"] for call in request_client.calls
    ] == ["route-1", "route-2"]
    assert supply["route_assessments"] == [
        {
            "route_id": "route-1",
            "supply_assessment": {"overall_feasibility": "unavailable"},
            "status": "assessed",
        },
        {
            "route_id": "route-2",
            "supply_assessment": {"overall_feasibility": "available"},
            "status": "assessed",
        },
    ]


async def test_full_workflow_service_rejects_supply_route_mismatch() -> None:
    request_client = _FullDownstreamRequestClient(
        lambda _subject, _payload: {
            "status": "assessed",
            "route_id": "route-other",
            "supply_assessment": {"overall_feasibility": "available"},
        }
    )

    with pytest.raises(RuntimeError, match="supply response route_id"):
        await orchestrator_main.FullWorkflowClients(request_client).assess_supply(
            _full_downstream_state()
        )


async def test_full_workflow_service_blocks_compilation_without_available_supply() -> None:
    request_client = _FullDownstreamRequestClient(
        lambda subject, payload: (_ for _ in ()).throw(
            AssertionError(f"unexpected request: {subject} {payload}")
        )
    )
    state = _full_downstream_state()
    state["supply"] = {
        "route_id": "route-1",
        "supply_assessment": {"overall_feasibility": "partial"},
    }

    result = await orchestrator_main.FullWorkflowClients(request_client).compile_synthesis(state)

    assert result["status"] == "not_compiled"
    assert result["route_id"] == "route-1"
    assert result["protocols"] == []
    assert result["blocking_evidence"] == [
        {
            "rule_id": "workflow_supply_feasibility",
            "reason": "selected route supply feasibility is partial",
        }
    ]
    assert request_client.calls == []


async def test_full_workflow_service_compiles_and_executes_one_bound_protocol() -> None:
    def respond(subject: str, payload: dict) -> dict:
        assert subject == "agent.srb.request"
        status = "executed" if payload.get("action") == "execute" else "compiled"
        return {
            "status": status,
            "route_id": payload["route_id"],
            "protocols": [{"route_id": payload["route_id"], "ssp_id": "ssp-1"}],
        }

    request_client = _FullDownstreamRequestClient(respond)
    clients = orchestrator_main.FullWorkflowClients(request_client)
    state = _full_downstream_state()
    state["supply"] = {
        "route_id": "route-1",
        "supply_assessment": {"overall_feasibility": "available"},
    }

    state["srb"] = await clients.compile_synthesis(state)
    executed = await clients.execute_synthesis(state)

    assert executed["status"] == "executed"
    assert executed["route_id"] == "route-1"
    assert request_client.calls[0]["payload"]["pathways"] == [
        state["retrosyn"]["routes"][0]
    ]
    assert request_client.calls[1]["payload"]["action"] == "execute"
    assert request_client.calls[1]["payload"]["protocols"] == state["srb"]["protocols"]
    assert request_client.calls[0]["payload"]["request_id"] != (
        request_client.calls[1]["payload"]["request_id"]
    )
    assert request_client.calls[1]["payload"]["request_id"].endswith(":execute")


def test_full_workflow_rejects_critic_pass_without_executable_protocol() -> None:
    graph = orchestrator_main.WorkflowGraph(clients=None)

    with pytest.raises(RuntimeError, match="executable selected-route protocol"):
        graph._route_after_critic(
            {
                "critic": {"verdict": "pass"},
                "retrosyn": {"routes": [{"route_id": "route-1"}]},
                "supply": {
                    "route_id": "route-1",
                    "supply_assessment": {"overall_feasibility": "unavailable"},
                },
                "srb": {
                    "status": "not_compiled",
                    "route_id": "route-1",
                    "protocols": [],
                },
            }
        )
