"""SQLite-backed persistence for runs, designs and known-molecule catalog.

Single-file DB at $MF_DB_PATH (default ./data/moleculeforge.db) so the app
survives restarts without docker. Schema is intentionally small and readable.
"""
from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import json
import os
import sqlite3
import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

_DB_PATH_ENV = "MF_DB_PATH"
_DEFAULT_DB = "/workspace/MForge/moleculeforge/data/moleculeforge.db"
_SCHEMA_VERSION = 1
_MAX_PAGE_SIZE = 100

_lock = threading.RLock()


class RunStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    PAUSED = "paused"
    AWAITING_EVIDENCE = "awaiting_evidence"
    COMPLETED = "completed"
    REJECTED = "rejected"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


class RunAlreadyExistsError(ValueError):
    pass


_LEGAL_TRANSITIONS: dict[RunStatus, frozenset[RunStatus]] = {
    RunStatus.QUEUED: frozenset(
        {
            RunStatus.RUNNING,
            RunStatus.REJECTED,
            RunStatus.FAILED,
            RunStatus.INTERRUPTED,
        }
    ),
    RunStatus.RUNNING: frozenset(
        {
            RunStatus.PAUSED,
            RunStatus.AWAITING_EVIDENCE,
            RunStatus.COMPLETED,
            RunStatus.REJECTED,
            RunStatus.FAILED,
            RunStatus.INTERRUPTED,
        }
    ),
    RunStatus.PAUSED: frozenset(
        {
            RunStatus.RUNNING,
            RunStatus.FAILED,
            RunStatus.INTERRUPTED,
        }
    ),
    RunStatus.AWAITING_EVIDENCE: frozenset(
        {
            RunStatus.RUNNING,
            RunStatus.REJECTED,
            RunStatus.FAILED,
            RunStatus.INTERRUPTED,
        }
    ),
    RunStatus.COMPLETED: frozenset(),
    RunStatus.REJECTED: frozenset(),
    RunStatus.FAILED: frozenset(),
    RunStatus.INTERRUPTED: frozenset(),
}


def db_path() -> str:
    return os.environ.get(_DB_PATH_ENV, _DEFAULT_DB)


def _connect_path(path: str | Path) -> sqlite3.Connection:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30, isolation_level=None, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def _connect() -> sqlite3.Connection:
    return _connect_path(db_path())


@contextmanager
def cursor() -> Iterator[sqlite3.Cursor]:
    with _lock:
        conn = _connect()
        try:
            cur = conn.cursor()
            yield cur
        finally:
            conn.close()


SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
    project_id      TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    description     TEXT DEFAULT '',
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    run_id          TEXT PRIMARY KEY,
    project_id      TEXT,
    intent          TEXT NOT NULL,
    objectives      TEXT NOT NULL,
    policy          TEXT NOT NULL DEFAULT '{}',
    status          TEXT NOT NULL,
    current_stage   TEXT NOT NULL DEFAULT 'queued',
    state           TEXT,
    error_type      TEXT,
    error_message   TEXT,
    created_at      TEXT NOT NULL,
    updated_at      TEXT,
    finished_at     TEXT,
    summary         TEXT,
    devices_used    TEXT,
    n_candidates    INTEGER DEFAULT 0,
    n_novel         INTEGER DEFAULT 0,
    n_known         INTEGER DEFAULT 0,
    FOREIGN KEY (project_id) REFERENCES projects(project_id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS reasoning_steps (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT NOT NULL,
    step_index      INTEGER NOT NULL,
    stage           TEXT NOT NULL,
    title           TEXT NOT NULL,
    detail          TEXT,
    payload         TEXT,
    timestamp       TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS run_results (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT NOT NULL,
    rank            INTEGER NOT NULL,
    smiles          TEXT NOT NULL,
    canonical_smiles TEXT NOT NULL,
    inchi_key       TEXT,
    is_novel        INTEGER NOT NULL,
    known_match     TEXT,
    properties      TEXT NOT NULL,
    pareto_optimal  INTEGER DEFAULT 0,
    composite_score REAL,
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS known_molecules (
    inchi_key       TEXT PRIMARY KEY,
    canonical_smiles TEXT NOT NULL,
    name            TEXT NOT NULL,
    drugbank_id     TEXT,
    indications     TEXT,
    target          TEXT,
    note            TEXT
);

CREATE INDEX IF NOT EXISTS idx_runs_project ON runs(project_id);
CREATE INDEX IF NOT EXISTS idx_steps_run    ON reasoning_steps(run_id, step_index);
CREATE INDEX IF NOT EXISTS idx_results_run  ON run_results(run_id, rank);
CREATE INDEX IF NOT EXISTS idx_known_smiles ON known_molecules(canonical_smiles);
"""


class RunStore:
    def __init__(self, path: str | Path) -> None:
        self.path = str(path)

    async def initialize(self) -> None:
        await asyncio.to_thread(self._initialize)

    def _initialize(self) -> None:
        with _lock:
            connection = _connect_path(self.path)
            try:
                connection.execute("BEGIN IMMEDIATE")
                for statement in SCHEMA.split(";"):
                    statement = statement.strip()
                    if statement:
                        connection.execute(statement + ";")
                version = int(connection.execute("PRAGMA user_version").fetchone()[0])
                if version > _SCHEMA_VERSION:
                    raise RuntimeError(
                        f"database schema version {version} is newer than {_SCHEMA_VERSION}"
                    )
                if version < 1:
                    columns = {
                        str(row["name"])
                        for row in connection.execute("PRAGMA table_info(runs)").fetchall()
                    }
                    additions = {
                        "policy": "TEXT NOT NULL DEFAULT '{}'",
                        "current_stage": "TEXT NOT NULL DEFAULT 'queued'",
                        "state": "TEXT",
                        "error_type": "TEXT",
                        "error_message": "TEXT",
                        "updated_at": "TEXT",
                    }
                    for name, declaration in additions.items():
                        if name not in columns:
                            connection.execute(
                                f"ALTER TABLE runs ADD COLUMN {name} {declaration}"
                            )
                    connection.execute(
                        """
                        WITH duplicate_rows AS (
                            SELECT
                                id,
                                run_id,
                                ROW_NUMBER() OVER (
                                    PARTITION BY run_id
                                    ORDER BY id
                                ) AS duplicate_offset
                            FROM reasoning_steps
                            WHERE id IN (
                                SELECT id
                                FROM (
                                    SELECT
                                        id,
                                        ROW_NUMBER() OVER (
                                            PARTITION BY run_id, step_index
                                            ORDER BY id
                                        ) AS occurrence
                                    FROM reasoning_steps
                                )
                                WHERE occurrence > 1
                            )
                        ),
                        maxima AS (
                            SELECT run_id, MAX(step_index) AS max_step
                            FROM reasoning_steps
                            GROUP BY run_id
                        )
                        UPDATE reasoning_steps
                        SET step_index = (
                            SELECT maxima.max_step + duplicate_rows.duplicate_offset
                            FROM duplicate_rows
                            JOIN maxima ON maxima.run_id = duplicate_rows.run_id
                            WHERE duplicate_rows.id = reasoning_steps.id
                        )
                        WHERE id IN (SELECT id FROM duplicate_rows)
                        """
                    )
                    connection.execute(
                        "CREATE UNIQUE INDEX IF NOT EXISTS "
                        "idx_steps_run_unique ON reasoning_steps(run_id, step_index)"
                    )
                    connection.execute(f"PRAGMA user_version={_SCHEMA_VERSION}")
                connection.commit()
            except Exception:
                connection.rollback()
                raise
            finally:
                connection.close()

    async def create_project(
        self,
        project_id: str,
        *,
        name: str,
        description: str,
        created_at: str,
    ) -> dict[str, object]:
        if not project_id:
            raise ValueError("project_id is required")
        if not name:
            raise ValueError("name is required")
        if not created_at:
            raise ValueError("created_at is required")
        return await asyncio.to_thread(
            self._create_project,
            project_id,
            name,
            description,
            created_at,
        )

    def _create_project(
        self,
        project_id: str,
        name: str,
        description: str,
        created_at: str,
    ) -> dict[str, object]:
        with _lock:
            connection = _connect_path(self.path)
            try:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    """
                    INSERT INTO projects (project_id, name, description, created_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(project_id) DO UPDATE SET
                        name=excluded.name,
                        description=excluded.description
                    """,
                    (project_id, name, description, created_at),
                )
                row = connection.execute(
                    "SELECT * FROM projects WHERE project_id=?",
                    (project_id,),
                ).fetchone()
                connection.commit()
            except Exception:
                connection.rollback()
                raise
            finally:
                connection.close()
        if row is None:
            raise RuntimeError(f"project was not persisted: {project_id}")
        return dict(row)

    async def list_projects(self) -> list[dict[str, object]]:
        return await asyncio.to_thread(self._list_projects)

    def _list_projects(self) -> list[dict[str, object]]:
        with _lock:
            connection = _connect_path(self.path)
            try:
                rows = connection.execute(
                    "SELECT * FROM projects ORDER BY created_at DESC LIMIT 200"
                ).fetchall()
            finally:
                connection.close()
        return [dict(row) for row in rows]

    async def get_project(self, project_id: str) -> dict[str, object] | None:
        return await asyncio.to_thread(self._get_project, project_id)

    def _get_project(self, project_id: str) -> dict[str, object] | None:
        with _lock:
            connection = _connect_path(self.path)
            try:
                row = connection.execute(
                    "SELECT * FROM projects WHERE project_id=?",
                    (project_id,),
                ).fetchone()
            finally:
                connection.close()
        return dict(row) if row is not None else None

    async def delete_project(self, project_id: str) -> bool:
        return await asyncio.to_thread(self._delete_project, project_id)

    def _delete_project(self, project_id: str) -> bool:
        with _lock:
            connection = _connect_path(self.path)
            try:
                connection.execute("BEGIN IMMEDIATE")
                deleted = connection.execute(
                    "DELETE FROM projects WHERE project_id=?",
                    (project_id,),
                )
                connection.commit()
                return deleted.rowcount == 1
            except Exception:
                connection.rollback()
                raise
            finally:
                connection.close()

    async def create_run(
        self,
        run_id: str,
        *,
        intent: str,
        policy: Mapping[str, object],
        created_at: str,
        project_id: str | None = None,
        state: Mapping[str, object] | None = None,
        require_new: bool = False,
    ) -> None:
        if not run_id:
            raise ValueError("run_id is required")
        if not intent:
            raise ValueError("intent is required")
        if not policy:
            raise ValueError("policy is required")
        await asyncio.to_thread(
            self._create_run,
            run_id,
            intent,
            dict(policy),
            created_at,
            project_id,
            dict(state) if state is not None else None,
            require_new,
        )

    def _create_run(
        self,
        run_id: str,
        intent: str,
        policy: dict[str, object],
        created_at: str,
        project_id: str | None,
        state: dict[str, object] | None,
        require_new: bool,
    ) -> None:
        with _lock:
            connection = _connect_path(self.path)
            try:
                connection.execute("BEGIN IMMEDIATE")
                if require_new:
                    existing = connection.execute(
                        "SELECT 1 FROM runs WHERE run_id=?",
                        (run_id,),
                    ).fetchone()
                    if existing is not None:
                        raise RunAlreadyExistsError(
                            f"run_id already exists: {run_id}"
                        )
                if project_id is not None:
                    project = connection.execute(
                        "SELECT 1 FROM projects WHERE project_id=?",
                        (project_id,),
                    ).fetchone()
                    if project is None:
                        raise ValueError(f"unknown project_id: {project_id}")
                connection.execute(
                    """
                    INSERT INTO runs (
                        run_id, project_id, intent, objectives, policy, status,
                        current_stage, state, created_at, updated_at, devices_used
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(run_id) DO UPDATE SET
                        project_id=excluded.project_id,
                        intent=excluded.intent,
                        policy=excluded.policy,
                        updated_at=excluded.updated_at
                    """,
                    (
                        run_id,
                        project_id,
                        intent,
                        "{}",
                        json.dumps(policy, sort_keys=True),
                        RunStatus.QUEUED.value,
                        RunStatus.QUEUED.value,
                        json.dumps(state, sort_keys=True) if state is not None else None,
                        created_at,
                        created_at,
                        "[]",
                    ),
                )
                connection.commit()
            except Exception:
                connection.rollback()
                raise
            finally:
                connection.close()

    async def transition_run(
        self,
        run_id: str,
        expected: set[RunStatus],
        target: RunStatus,
        *,
        current_stage: str,
        state: Mapping[str, object] | None = None,
        error_type: str | None = None,
        error_message: str | None = None,
    ) -> None:
        expected_statuses = {RunStatus(item) for item in expected}
        target_status = RunStatus(target)
        if not expected_statuses:
            raise ValueError("expected status set must not be empty")
        await asyncio.to_thread(
            self._transition_run,
            run_id,
            expected_statuses,
            target_status,
            current_stage,
            dict(state) if state is not None else None,
            error_type,
            error_message,
        )

    def _transition_run(
        self,
        run_id: str,
        expected: set[RunStatus],
        target: RunStatus,
        current_stage: str,
        state: dict[str, object] | None,
        error_type: str | None,
        error_message: str | None,
    ) -> None:
        with _lock:
            connection = _connect_path(self.path)
            try:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute(
                    "SELECT status FROM runs WHERE run_id=?",
                    (run_id,),
                ).fetchone()
                if row is None:
                    raise ValueError(f"unknown run_id: {run_id}")
                current = RunStatus(str(row["status"]))
                if current not in expected:
                    expected_values = sorted(item.value for item in expected)
                    raise ValueError(
                        f"run {run_id} status {current.value} does not match expected "
                        f"{expected_values}"
                    )
                if target not in _LEGAL_TRANSITIONS[current]:
                    raise ValueError(
                        f"illegal run transition: {current.value} -> {target.value}"
                    )
                finished_at = None
                if target in {
                    RunStatus.COMPLETED,
                    RunStatus.REJECTED,
                    RunStatus.FAILED,
                    RunStatus.INTERRUPTED,
                }:
                    finished_at = datetime.now(UTC).isoformat()
                changed = connection.execute(
                    """
                    UPDATE runs
                    SET status=?,
                        current_stage=?,
                        updated_at=?,
                        error_type=?,
                        error_message=?,
                        state=COALESCE(?, state),
                        finished_at=COALESCE(?, finished_at)
                    WHERE run_id=? AND status=?
                    """,
                    (
                        target.value,
                        current_stage,
                        datetime.now(UTC).isoformat(),
                        error_type,
                        error_message,
                        json.dumps(state, sort_keys=True) if state is not None else None,
                        finished_at,
                        run_id,
                        current.value,
                    ),
                )
                if changed.rowcount != 1:
                    raise ValueError(f"run {run_id} changed during transition")
                connection.commit()
            except Exception:
                connection.rollback()
                raise
            finally:
                connection.close()

    async def compensate_cancelled_failure(
        self,
        run_id: str,
        *,
        expected_error_type: str,
        expected_error_message: str,
        cancellation_message: str,
    ) -> bool:
        return await asyncio.to_thread(
            self._compensate_cancelled_failure,
            run_id,
            expected_error_type,
            expected_error_message,
            cancellation_message,
        )

    def _compensate_cancelled_failure(
        self,
        run_id: str,
        expected_error_type: str,
        expected_error_message: str,
        cancellation_message: str,
    ) -> bool:
        active = (
            RunStatus.QUEUED.value,
            RunStatus.RUNNING.value,
            RunStatus.PAUSED.value,
            RunStatus.AWAITING_EVIDENCE.value,
        )
        with _lock:
            connection = _connect_path(self.path)
            try:
                connection.execute("BEGIN IMMEDIATE")
                now = datetime.now(UTC).isoformat()
                changed = connection.execute(
                    """
                    UPDATE runs
                    SET status=?,
                        updated_at=?,
                        finished_at=?,
                        error_type=?,
                        error_message=?
                    WHERE run_id=?
                      AND (
                          status IN (?, ?, ?, ?)
                          OR (
                              status=?
                              AND error_type=?
                              AND error_message=?
                          )
                      )
                    """,
                    (
                        RunStatus.INTERRUPTED.value,
                        now,
                        now,
                        asyncio.CancelledError.__name__,
                        cancellation_message,
                        run_id,
                        *active,
                        RunStatus.FAILED.value,
                        expected_error_type,
                        expected_error_message,
                    ),
                )
                connection.commit()
                return changed.rowcount == 1
            except Exception:
                connection.rollback()
                raise
            finally:
                connection.close()

    async def append_event(
        self,
        run_id: str,
        step_index: int,
        *,
        stage: str,
        payload: Mapping[str, object],
        timestamp: str,
        state: Mapping[str, object] | None = None,
    ) -> None:
        await asyncio.to_thread(
            self._append_event,
            run_id,
            step_index,
            stage,
            dict(payload),
            timestamp,
            dict(state) if state is not None else None,
        )

    def _append_event(
        self,
        run_id: str,
        step_index: int,
        stage: str,
        payload: dict[str, object],
        timestamp: str,
        state: dict[str, object] | None,
    ) -> None:
        with _lock:
            connection = _connect_path(self.path)
            try:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    """
                    INSERT INTO reasoning_steps
                    (run_id, step_index, stage, title, detail, payload, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        step_index,
                        stage,
                        stage,
                        None,
                        json.dumps(payload, sort_keys=True),
                        timestamp,
                    ),
                )
                connection.execute(
                    """
                    UPDATE runs
                    SET current_stage=?, updated_at=?, state=COALESCE(?, state)
                    WHERE run_id=? AND status=?
                    """,
                    (
                        stage,
                        timestamp,
                        json.dumps(state, sort_keys=True) if state is not None else None,
                        run_id,
                        RunStatus.RUNNING.value,
                    ),
                )
                connection.commit()
            except Exception:
                connection.rollback()
                raise
            finally:
                connection.close()

    async def get_run(self, run_id: str) -> dict[str, object] | None:
        return await asyncio.to_thread(self._get_run, run_id)

    def _get_run(self, run_id: str) -> dict[str, object] | None:
        with _lock:
            connection = _connect_path(self.path)
            try:
                row = connection.execute(
                    "SELECT * FROM runs WHERE run_id=?",
                    (run_id,),
                ).fetchone()
            finally:
                connection.close()
        return _decode_run(row) if row is not None else None

    async def list_runs(
        self,
        *,
        page_size: int,
        page_token: str | None = None,
        context: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        if page_size < 1 or page_size > _MAX_PAGE_SIZE:
            raise ValueError(f"page_size must be between 1 and {_MAX_PAGE_SIZE}")
        listing_context = dict(context or {})
        cursor_value: tuple[str, str] | None = None
        if page_token is not None:
            cursor_value = _decode_page_token(page_token, page_size, listing_context)
        return await asyncio.to_thread(
            self._list_runs,
            page_size,
            cursor_value,
            listing_context,
        )

    def _list_runs(
        self,
        page_size: int,
        cursor_value: tuple[str, str] | None,
        context: dict[str, object],
    ) -> dict[str, object]:
        query = "SELECT * FROM runs"
        values: list[object] = []
        if cursor_value is not None:
            query += " WHERE (created_at, run_id) > (?, ?)"
            values.extend(cursor_value)
        query += " ORDER BY created_at ASC, run_id ASC LIMIT ?"
        values.append(page_size + 1)
        with _lock:
            connection = _connect_path(self.path)
            try:
                rows = connection.execute(query, values).fetchall()
            finally:
                connection.close()
        items = [_decode_run(row) for row in rows[:page_size]]
        next_page_token = None
        if len(rows) > page_size:
            last = rows[page_size - 1]
            next_page_token = _encode_page_token(
                (str(last["created_at"]), str(last["run_id"])),
                page_size,
                context,
            )
        return {"items": items, "next_page_token": next_page_token}

    async def list_events(
        self,
        run_id: str,
        *,
        after_step: int = -1,
    ) -> list[dict[str, object]]:
        if after_step < -1:
            raise ValueError("after_step must be at least -1")
        return await asyncio.to_thread(self._list_events, run_id, after_step)

    def _list_events(self, run_id: str, after_step: int) -> list[dict[str, object]]:
        with _lock:
            connection = _connect_path(self.path)
            try:
                rows = connection.execute(
                    "SELECT * FROM reasoning_steps "
                    "WHERE run_id=? AND step_index>? ORDER BY step_index",
                    (run_id, after_step),
                ).fetchall()
            finally:
                connection.close()
        return [_decode_event(row) for row in rows]

    async def interrupt_active_runs(self) -> int:
        return await asyncio.to_thread(self._interrupt_active_runs)

    def _interrupt_active_runs(self) -> int:
        active = (
            RunStatus.QUEUED.value,
            RunStatus.RUNNING.value,
            RunStatus.PAUSED.value,
            RunStatus.AWAITING_EVIDENCE.value,
        )
        with _lock:
            connection = _connect_path(self.path)
            try:
                connection.execute("BEGIN IMMEDIATE")
                changed = connection.execute(
                    "UPDATE runs SET status=?, current_stage=?, updated_at=?, finished_at=?, "
                    "error_type=?, error_message=? "
                    "WHERE status IN (?, ?, ?, ?)",
                    (
                        RunStatus.INTERRUPTED.value,
                        RunStatus.INTERRUPTED.value,
                        datetime.now(UTC).isoformat(),
                        datetime.now(UTC).isoformat(),
                        "ServiceRestart",
                        "run interrupted by orchestrator restart",
                        *active,
                    ),
                )
                connection.commit()
                return int(changed.rowcount)
            except Exception:
                connection.rollback()
                raise
            finally:
                connection.close()


def _decode_run(row: sqlite3.Row) -> dict[str, object]:
    result: dict[str, object] = dict(row)
    for key, fallback in (
        ("objectives", {}),
        ("policy", {}),
        ("state", None),
        ("devices_used", []),
    ):
        value = result.get(key)
        result[key] = json.loads(str(value)) if value else fallback
    return result


def _decode_event(row: sqlite3.Row) -> dict[str, object]:
    result: dict[str, object] = dict(row)
    result["payload"] = json.loads(str(result.get("payload") or "{}"))
    return result


def _encode_page_token(
    cursor_value: tuple[str, str],
    page_size: int,
    context: Mapping[str, object],
) -> str:
    body = json.dumps(
        {
            "version": 1,
            "cursor": list(cursor_value),
            "page_size": page_size,
            "context": context,
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    digest = hashlib.sha256(body).digest()
    return base64.urlsafe_b64encode(digest + body).decode().rstrip("=")


def _decode_page_token(
    token: str,
    page_size: int,
    context: Mapping[str, object],
) -> tuple[str, str]:
    try:
        if not token or any(character.isspace() for character in token):
            raise ValueError
        padded = token + "=" * (-len(token) % 4)
        encoded = base64.b64decode(
            padded.encode("ascii"),
            altchars=b"-_",
            validate=True,
        )
        if base64.urlsafe_b64encode(encoded).decode().rstrip("=") != token:
            raise ValueError
        digest, body = encoded[:32], encoded[32:]
        if len(digest) != 32 or hashlib.sha256(body).digest() != digest:
            raise ValueError
        payload = json.loads(body)
        cursor_value = payload["cursor"]
        if (
            payload.get("version") != 1
            or payload.get("page_size") != page_size
            or payload.get("context") != dict(context)
            or not isinstance(cursor_value, list)
            or len(cursor_value) != 2
            or not all(isinstance(item, str) for item in cursor_value)
        ):
            raise ValueError
    except (
        ValueError,
        KeyError,
        TypeError,
        UnicodeEncodeError,
        binascii.Error,
        json.JSONDecodeError,
    ) as exc:
        raise ValueError("invalid page_token") from exc
    return str(cursor_value[0]), str(cursor_value[1])


def init_db() -> None:
    """Create schema and seed the known-molecule catalog if empty."""
    with cursor() as cur:
        for stmt in SCHEMA.split(";"):
            stmt = stmt.strip()
            if stmt:
                cur.execute(stmt + ";")
        n = cur.execute("SELECT COUNT(*) FROM known_molecules").fetchone()[0]
    if n == 0:
        _seed_known_molecules()


def insert_project(project_id: str, name: str, description: str, created_at: str) -> None:
    with cursor() as cur:
        cur.execute(
            """
            INSERT INTO projects (project_id, name, description, created_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(project_id) DO UPDATE SET
                name=excluded.name,
                description=excluded.description
            """,
            (project_id, name, description, created_at),
        )


def list_projects() -> list[dict]:
    with cursor() as cur:
        rows = cur.execute(
            "SELECT * FROM projects ORDER BY created_at DESC LIMIT 200"
        ).fetchall()
    return [dict(r) for r in rows]


def get_run(run_id: str) -> dict | None:
    with cursor() as cur:
        row = cur.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
    if not row:
        return None
    out = dict(row)
    out["objectives"] = json.loads(out.get("objectives") or "{}")
    out["devices_used"] = json.loads(out.get("devices_used") or "[]")
    return out


def list_runs(limit: int = 100) -> list[dict]:
    with cursor() as cur:
        rows = cur.execute(
            "SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["objectives"] = json.loads(d.get("objectives") or "{}")
        d["devices_used"] = json.loads(d.get("devices_used") or "[]")
        out.append(d)
    return out


def insert_reasoning_step(
    run_id: str, step_index: int, stage: str, title: str,
    detail: str | None, payload: dict | None, timestamp: str,
) -> None:
    with cursor() as cur:
        cur.execute(
            """
            INSERT INTO reasoning_steps
            (run_id, step_index, stage, title, detail, payload, timestamp)
            VALUES (?,?,?,?,?,?,?)
            """,
            (run_id, step_index, stage, title, detail,
             json.dumps(payload or {}), timestamp),
        )


def get_reasoning_steps(run_id: str) -> list[dict]:
    with cursor() as cur:
        rows = cur.execute(
            "SELECT * FROM reasoning_steps WHERE run_id=? ORDER BY step_index",
            (run_id,),
        ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["payload"] = json.loads(d.get("payload") or "{}")
        out.append(d)
    return out


def insert_run_results(run_id: str, results: list[dict]) -> None:
    if not results:
        return
    with cursor() as cur:
        cur.executemany(
            """
            INSERT INTO run_results
            (run_id, rank, smiles, canonical_smiles, inchi_key,
             is_novel, known_match, properties, pareto_optimal, composite_score)
            VALUES (?,?,?,?,?,?,?,?,?,?)
            """,
            [
                (
                    run_id, r["rank"], r["smiles"], r["canonical_smiles"],
                    r.get("inchi_key"),
                    1 if r.get("is_novel") else 0,
                    json.dumps(r.get("known_match")) if r.get("known_match") else None,
                    json.dumps(r.get("properties") or {}),
                    1 if r.get("pareto_optimal") else 0,
                    r.get("composite_score"),
                )
                for r in results
            ],
        )


def get_run_results(run_id: str) -> list[dict]:
    with cursor() as cur:
        rows = cur.execute(
            "SELECT * FROM run_results WHERE run_id=? ORDER BY rank",
            (run_id,),
        ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["is_novel"] = bool(d["is_novel"])
        d["pareto_optimal"] = bool(d["pareto_optimal"])
        d["properties"] = json.loads(d.get("properties") or "{}")
        if d.get("known_match"):
            d["known_match"] = json.loads(d["known_match"])
        out.append(d)
    return out


# ---------------------------------------------------------------------------
# Known-molecule catalog
# ---------------------------------------------------------------------------


def _seed_known_molecules() -> None:
    """Seed with 80+ well-known drug molecules — used for novelty labelling."""
    try:
        from rdkit import Chem
    except ImportError:
        return

    seeds: list[tuple[str, str, str | None, str | None, str | None]] = [
        # (name, smiles, drugbank_id, indications, target)
        ("Aspirin", "CC(=O)Oc1ccccc1C(=O)O", "DB00945", "analgesic, anti-inflammatory", "COX-1/2"),
        ("Ibuprofen", "CC(C)Cc1ccc(C(C)C(=O)O)cc1", "DB01050", "anti-inflammatory", "COX-1/2"),
        ("Paracetamol", "CC(=O)Nc1ccc(O)cc1", "DB00316", "analgesic", "COX-2"),
        ("Caffeine", "Cn1cnc2c1c(=O)n(C)c(=O)n2C", "DB00201", "stimulant", "Adenosine receptor"),
        ("Nicotine", "CN1CCC[C@H]1c1cccnc1", "DB00184", "stimulant", "nAChR"),
        ("Morphine", "CN1CC[C@]23c4c5ccc(O)c4O[C@H]2[C@@H](O)C=C[C@H]3[C@H]1C5", "DB00295", "analgesic", "Mu opioid"),
        ("Codeine", "COc1ccc2c3c1O[C@H]1[C@@H](O)C=C[C@H]4[C@@H]3CC([C@@H]2C4)N(C)C1", "DB00318", "analgesic", "Mu opioid"),
        ("Ethanol", "CCO", "DB00898", "antiseptic", None),
        ("Methanol", "CO", "DB02327", "industrial solvent", None),
        ("Penicillin G", "CC1(C)S[C@@H]2[C@H](NC(=O)Cc3ccccc3)C(=O)N2[C@H]1C(=O)O", "DB01053", "antibacterial", "PBP"),
        ("Amoxicillin", "CC1(C)S[C@@H]2[C@H](NC(=O)[C@H](N)c3ccc(O)cc3)C(=O)N2[C@H]1C(=O)O", "DB01060", "antibacterial", "PBP"),
        ("Cephalexin", "CC1=C(C(=O)O)N2C(=O)[C@@H](NC(=O)[C@H](N)c3ccccc3)[C@H]2SC1", "DB00567", "antibacterial", "PBP"),
        ("Ciprofloxacin", "O=C(O)c1cn(C2CC2)c2cc(N3CCNCC3)c(F)cc2c1=O", "DB00537", "antibacterial", "DNA gyrase"),
        ("Azithromycin", "CC[C@H]1OC(=O)[C@H](C)[C@@H](O[C@H]2C[C@@](C)(OC)[C@@H](O)[C@H](C)O2)[C@H](C)[C@@H](O[C@@H]2O[C@H](C)C[C@@H]([C@H]2O)N(C)C)[C@@](C)(O)C[C@@H](C)CN(C)[C@H](C)[C@@H](O)[C@]1(C)O", "DB00207", "antibacterial", "50S ribosome"),
        ("Doxycycline", "CC1c2cccc(O)c2C(=O)C2=C(O)[C@]3(O)C(=O)C(C(N)=O)=C(O)[C@@H](N(C)C)[C@@H]3C[C@@H]12", "DB00254", "antibacterial", "30S ribosome"),
        ("Metformin", "CN(C)C(=N)NC(N)=N", "DB00331", "antidiabetic", "AMPK"),
        ("Glipizide", "Cc1ncc(C(=O)NCCc2ccc(S(=O)(=O)NC(=O)NC3CCCCC3)cc2)cn1", "DB01067", "antidiabetic", "K-ATP channel"),
        ("Atorvastatin", "CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CC[C@@H](O)C[C@@H](O)CC(=O)O", "DB01076", "hypolipidemic", "HMG-CoA reductase"),
        ("Simvastatin", "CCC(C)(C)C(=O)O[C@H]1C[C@H](C)C=C2C=C[C@H](C)[C@H](CC[C@@H]3C[C@@H](O)CC(=O)O3)[C@@H]12", "DB00641", "hypolipidemic", "HMG-CoA reductase"),
        ("Lisinopril", "NCCCC[C@@H](N[C@@H](CCc1ccccc1)C(=O)O)C(=O)N1CCC[C@H]1C(=O)O", "DB00722", "antihypertensive", "ACE"),
        ("Losartan", "CCCCc1nc(Cl)c(CO)n1Cc1ccc(-c2ccccc2-c2nnn[nH]2)cc1", "DB00678", "antihypertensive", "AT1 receptor"),
        ("Amlodipine", "CCOC(=O)C1=C(COCCN)NC(C)=C(C(=O)OC)C1c1ccccc1Cl", "DB00381", "antihypertensive", "L-type Ca channel"),
        ("Hydrochlorothiazide", "NS(=O)(=O)c1cc2c(cc1Cl)NCNS2(=O)=O", "DB00999", "antihypertensive", "NCC"),
        ("Furosemide", "NS(=O)(=O)c1cc(C(=O)O)c(NCc2ccco2)cc1Cl", "DB00695", "diuretic", "NKCC2"),
        ("Warfarin", "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O", "DB00682", "anticoagulant", "VKORC1"),
        ("Heparin (frag)", "OCC1OC(O)C(O)C1OC1OC(C(=O)O)C(O)C1OS(=O)(=O)O", "DB01109", "anticoagulant", "antithrombin"),
        ("Omeprazole", "COc1ccc2[nH]c(S(=O)Cc3ncc(C)c(OC)c3C)nc2c1", "DB00338", "PPI", "H+/K+-ATPase"),
        ("Ranitidine", "CNC(=C[N+](=O)[O-])NCCSCc1ccc(CN(C)C)o1", "DB00863", "H2 blocker", "H2 receptor"),
        ("Loratadine", "CCOC(=O)N1CCC(=C2c3ccc(Cl)cc3CCc3cccnc32)CC1", "DB00455", "antihistamine", "H1 receptor"),
        ("Diphenhydramine", "CN(C)CCOC(c1ccccc1)c1ccccc1", "DB01075", "antihistamine", "H1 receptor"),
        ("Sildenafil", "CCCc1nn(C)c2c1NC(=NC2=O)c1cc(S(=O)(=O)N2CCN(C)CC2)ccc1OCC", "DB00203", "PDE5 inhibitor", "PDE5"),
        ("Tadalafil", "CN1CC(=O)N2[C@@H](Cc3c2[nH]c2ccccc32)[C@@H]1c1ccc2c(c1)OCO2", "DB00820", "PDE5 inhibitor", "PDE5"),
        ("Diazepam", "CN1C(=O)CN=C(c2ccccc2)c2cc(Cl)ccc21", "DB00829", "anxiolytic", "GABA-A"),
        ("Alprazolam", "Cc1nnc2n1-c1ccc(Cl)cc1C(c1ccccc1)=NC2", "DB00404", "anxiolytic", "GABA-A"),
        ("Fluoxetine", "CNCCC(c1ccc(C(F)(F)F)cc1)Oc1ccccc1", "DB00472", "antidepressant", "SERT"),
        ("Sertraline", "CN[C@H]1CC[C@H](c2ccc(Cl)c(Cl)c2)c2ccccc21", "DB01104", "antidepressant", "SERT"),
        ("Paroxetine", "Fc1ccc([C@@H]2CCNC[C@H]2COc2ccc3OCOc3c2)cc1", "DB00715", "antidepressant", "SERT"),
        ("Citalopram", "N#CC1=CC=C(C2(c3ccc(F)cc3)OCC2CCN(C)C)C=C1", "DB00215", "antidepressant", "SERT"),
        ("Risperidone", "Cc1c(CCN2CCC(c3noc4cc(F)ccc34)CC2)c(=O)n2CCCCc2n1", "DB00734", "antipsychotic", "D2/5HT2A"),
        ("Olanzapine", "CN1CCN(C2=Nc3cc(C)sc3Nc3ccccc32)CC1", "DB00334", "antipsychotic", "D2/5HT2A"),
        ("Quetiapine", "OCCOCCN1CCN(C2=Nc3ccccc3Sc3ccccc32)CC1", "DB01224", "antipsychotic", "D2/5HT2A"),
        ("Levodopa", "N[C@@H](Cc1ccc(O)c(O)c1)C(=O)O", "DB01235", "antiparkinson", "DDC"),
        ("Donepezil", "COc1cc2c(cc1OC)C(=O)C(CC1CCN(Cc3ccccc3)CC1)C2", "DB00843", "antialzheimer", "AChE"),
        ("Memantine", "CC12CC3CC(C1)CC(N)(C3)C2", "DB01043", "antialzheimer", "NMDAR"),
        ("Insulin (frag)", "CC[C@H](C)[C@H](NC(=O)CN)C(=O)O", "DB00030", "antidiabetic", "Insulin receptor"),
        ("Acetaminophen", "CC(=O)Nc1ccc(O)cc1", "DB00316", "analgesic", "COX-2"),
        ("Naproxen", "COc1ccc2cc([C@@H](C)C(=O)O)ccc2c1", "DB00788", "anti-inflammatory", "COX-1/2"),
        ("Diclofenac", "OC(=O)Cc1ccccc1Nc1c(Cl)cccc1Cl", "DB00586", "anti-inflammatory", "COX-1/2"),
        ("Celecoxib", "Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1", "DB00482", "anti-inflammatory", "COX-2"),
        ("Salbutamol", "CC(C)(C)NCC(O)c1ccc(O)c(CO)c1", "DB01001", "bronchodilator", "Beta-2 receptor"),
        ("Albuterol", "CC(C)(C)NCC(O)c1ccc(O)c(CO)c1", "DB01001", "bronchodilator", "Beta-2 receptor"),
        ("Theophylline", "Cn1c(=O)c2[nH]cnc2n(C)c1=O", "DB00277", "bronchodilator", "PDE/AdR"),
        ("Prednisone", "O=C1C=CC2(C)C3CCC4(C)C(C(=O)CO)(O)CCC4C3CCC2C1", "DB00635", "anti-inflammatory", "GR"),
        ("Hydrocortisone", "CC12CCC3C(CCC4=CC(=O)CCC34C)C1CCC2(O)C(=O)CO", "DB00741", "anti-inflammatory", "GR"),
        ("Dexamethasone", "CC1CC2C3CCC4=CC(=O)C=CC4(C)C3(F)C(O)CC2(C)C1(O)C(=O)CO", "DB01234", "anti-inflammatory", "GR"),
        ("Methotrexate", "CN(Cc1cnc2nc(N)nc(N)c2n1)c1ccc(C(=O)N[C@@H](CCC(=O)O)C(=O)O)cc1", "DB00563", "antineoplastic", "DHFR"),
        ("Doxorubicin", "COc1cccc2c1C(=O)c1c(O)c3c(c(O)c1C2=O)C[C@@](O)(C(=O)CO)C[C@@H]3O[C@H]1C[C@H](N)[C@H](O)[C@H](C)O1", "DB00997", "antineoplastic", "Topo II"),
        ("Imatinib", "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1", "DB00619", "antineoplastic", "Bcr-Abl"),
        ("Erlotinib", "COCCOc1cc2ncnc(Nc3cccc(C#C)c3)c2cc1OCCOC", "DB00530", "antineoplastic", "EGFR"),
        ("Gefitinib", "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1", "DB00317", "antineoplastic", "EGFR"),
        ("Sorafenib", "CNC(=O)c1cc(Oc2ccc(NC(=O)Nc3ccc(Cl)c(C(F)(F)F)c3)cc2)ccn1", "DB00398", "antineoplastic", "VEGFR/PDGFR"),
        ("Pazopanib", "Cc1ccc(N(C)c2nccc(N(C)c3ccc4[nH]nc(C)c4c3)n2)cc1S(N)(=O)=O", "DB06589", "antineoplastic", "VEGFR"),
        ("Tamoxifen", "CC/C(=C(\\c1ccccc1)c1ccc(OCCN(C)C)cc1)c1ccccc1", "DB00675", "antineoplastic", "ER"),
        ("Olaparib", "O=C1c2ccccc2Cc2cc(CC(=O)N3CCN(C(=O)C4CC4)CC3)ccc21", "DB09074", "antineoplastic", "PARP"),
        ("Adenosine", "Nc1ncnc2c1ncn2[C@@H]1O[C@H](CO)[C@@H](O)[C@H]1O", "DB00640", "antiarrhythmic", "AdR"),
        ("ATP", "Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)(O)OP(=O)(O)OP(=O)(O)O)[C@@H](O)[C@H]1O", "DB00171", "energy", None),
        ("Glucose", "OC[C@H]1O[C@H](O)[C@H](O)[C@@H](O)[C@@H]1O", "DB00131", None, None),
        ("Cholesterol", "C[C@H](CCCC(C)C)[C@H]1CC[C@H]2[C@@H]3CC=C4CC(O)CC[C@]4(C)[C@H]3CC[C@]12C", "DB04540", None, None),
        ("Thiamine", "Cc1ncc(C[n+]2csc(CCO)c2C)c(N)n1", "DB00152", "vitamin", None),
        ("Vitamin C", "OC1=C(O)C(=O)O[C@@H]1[C@@H](O)CO", "DB00126", "vitamin", None),
        ("Folic Acid", "Nc1nc2ncc(CNc3ccc(C(=O)N[C@@H](CCC(=O)O)C(=O)O)cc3)nc2c(=O)[nH]1", "DB00158", "vitamin", "DHFR"),
        ("Riboflavin", "Cc1cc2nc3c(=O)[nH]c(=O)nc-3n(C[C@H](O)[C@H](O)[C@H](O)CO)c2cc1C", "DB00140", "vitamin", None),
        ("Niacin", "O=C(O)c1cccnc1", "DB00627", "vitamin", None),
        ("Histamine", "NCCc1c[nH]cn1", "DB05381", "biogenic amine", "H receptors"),
        ("Serotonin", "NCCc1c[nH]c2ccc(O)cc12", "DB08839", "biogenic amine", "5HT receptors"),
        ("Dopamine", "NCCc1ccc(O)c(O)c1", "DB00988", "biogenic amine", "D receptors"),
        ("Adrenaline", "CNC[C@@H](O)c1ccc(O)c(O)c1", "DB00668", "biogenic amine", "Adrenergic"),
        ("GABA", "NCCCC(=O)O", "DB02530", "neurotransmitter", "GABA receptors"),
        ("Glutamate", "N[C@@H](CCC(=O)O)C(=O)O", "DB00142", "neurotransmitter", "NMDA/AMPA"),
        ("Salicylic Acid", "OC(=O)c1ccccc1O", "DB00936", "topical", "COX"),
        ("Benzene", "c1ccccc1", None, None, None),
        ("Pyridine", "c1ccncc1", None, None, None),
        ("Sulfanilamide", "Nc1ccc(S(N)(=O)=O)cc1", "DB00259", "antibacterial", "DHPS"),
    ]

    rows = []
    for name, smiles, db_id, ind, target in seeds:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        canon = Chem.MolToSmiles(mol, canonical=True)
        ikey = Chem.InchiToInchiKey(Chem.MolToInchi(mol)) if hasattr(Chem, "InchiToInchiKey") else None
        if ikey is None:
            try:
                from rdkit.Chem import inchi as _inchi
                ikey = _inchi.MolToInchiKey(mol)
            except Exception:
                ikey = canon  # fallback unique key
        rows.append((ikey, canon, name, db_id, ind, target, ""))

    with cursor() as cur:
        cur.executemany(
            """
            INSERT OR IGNORE INTO known_molecules
            (inchi_key, canonical_smiles, name, drugbank_id, indications, target, note)
            VALUES (?,?,?,?,?,?,?)
            """,
            rows,
        )


def lookup_known(inchi_key: str | None, canonical_smiles: str) -> dict | None:
    with cursor() as cur:
        if inchi_key:
            row = cur.execute(
                "SELECT * FROM known_molecules WHERE inchi_key=?", (inchi_key,)
            ).fetchone()
            if row:
                return dict(row)
        row = cur.execute(
            "SELECT * FROM known_molecules WHERE canonical_smiles=?",
            (canonical_smiles,),
        ).fetchone()
    return dict(row) if row else None


def known_count() -> int:
    with cursor() as cur:
        return int(cur.execute("SELECT COUNT(*) FROM known_molecules").fetchone()[0])
