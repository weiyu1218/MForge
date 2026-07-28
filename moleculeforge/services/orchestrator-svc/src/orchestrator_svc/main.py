"""Orchestrator Service - FastAPI + gRPC server for LangGraph-driven design loops."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import struct
import uuid
from collections.abc import Awaitable, Callable
from concurrent import futures
from datetime import UTC, datetime
from typing import TypeVar

import grpc
from fastapi import FastAPI, HTTPException
from mf_agents.base.agent import (
    AGENT_PROTOCOLS,
    BaseAgent,
    agent_health_check_timeout_seconds,
)
from mf_agents.messaging.redis_bus import RedisBus
from mf_agents.messaging.request_client import AgentRequestClient
from mf_core.db.store import RunAlreadyExistsError, RunStatus, RunStore, db_path
from mf_core.geometry.lorentz import normalize_lorentz_embedding
from mf_core.proto_gen.moleculeforge.v1.agent import orchestrator_pb2, orchestrator_pb2_grpc
from orchestrator.workflow.graph_builder import WorkflowGraph, create_initial_state

rest_app = FastAPI(title="Orchestrator Service", version="0.1.0")
_RUN_STORE: RunStore | None = None
_RUN_CONTROL: RunControl | None = None
_RUN_INITIALIZED_STORE: RunStore | None = None
_RUNTIME_INIT_LOCK: asyncio.Lock | None = None
_RUNTIME_INIT_LOOP: asyncio.AbstractEventLoop | None = None
_RUN_TASKS: dict[str, asyncio.Task[None]] = {}
_AGENT_BUS: RedisBus | None = None
_AGENT_REQUEST_CLIENT: AgentRequestClient | None = None
_INTERNAL_LEGACY_DESIGN_REQUEST = "_mforge_internal_legacy_design_request"
_AGENT_RUNTIME_LOOP: asyncio.AbstractEventLoop | None = None
_AGENT_INIT_LOCK: asyncio.Lock | None = None
_AGENT_INIT_LOOP: asyncio.AbstractEventLoop | None = None
_DIRECT_AGENT_TASKS: set[asyncio.Task[object]] = set()
_AGENT_SHUTDOWN_COUNT = 0
LOGGER = logging.getLogger(__name__)
T = TypeVar("T")
_CURRENT_HFM_LORENTZ_DIM = 129
_AGENT_PROTOCOLS_BY_ENTRY_POINT = {protocol.entry_point: protocol for protocol in AGENT_PROTOCOLS}
_NONTERMINAL_RUN_STATUSES = frozenset(
    {
        RunStatus.QUEUED,
        RunStatus.RUNNING,
        RunStatus.PAUSED,
        RunStatus.AWAITING_EVIDENCE,
    }
)
_NONTERMINAL_RUN_STATUS_VALUES = frozenset(status.value for status in _NONTERMINAL_RUN_STATUSES)
_FULL_WORKFLOW_BLOCKING_CRITIC_RULE_IDS = [
    "rule_001",
    "rule_004",
    "rule_005",
    "rule_014",
    "rule_015",
    "rule_016",
    "rule_017",
    "rule_018",
    "rule_019",
    "rule_020",
    "rule_021",
    "rule_022",
    "rule_024",
    "rule_025",
    "rule_026",
    "rule_027",
    "rule_028",
    "rule_029",
    "rule_030",
    "rule_045",
    "rule_046",
    "rule_049",
    "rule_050",
    "rule_051",
    "rule_052",
    "rule_053",
    "rule_054",
    "rule_055",
    "rule_056",
    "rule_057",
    "rule_058",
    "rule_059",
    "rule_070",
    "rule_074",
    "rule_076",
    "rule_087",
    "rule_088",
    "rule_089",
    "rule_090",
    "rule_091",
    "rule_092",
    "rule_098",
    "rule_099",
    "rule_100",
    "crg_validation_status",
    "crg_retrosyn_routes",
]


class _RunControlState:
    def __init__(self) -> None:
        self.pause_requested = False
        self.paused = asyncio.Event()
        self.resume_requested = asyncio.Event()
        self.resumed = asyncio.Event()
        self.evidence_resume_requested = asyncio.Event()
        self.evidence_resumed = asyncio.Event()
        self.closed = asyncio.Event()


class RunControl:
    def __init__(self, run_store: RunStore) -> None:
        self.store = run_store
        self._states: dict[str, _RunControlState] = {}

    def _state(self, run_id: str) -> _RunControlState:
        return self._states.setdefault(run_id, _RunControlState())

    async def pause(self, run_id: str) -> None:
        snapshot = await self.store.get_run(run_id)
        if snapshot is None:
            raise ValueError(f"unknown run_id: {run_id}")
        if snapshot["status"] != RunStatus.RUNNING.value:
            raise ValueError(f"run {run_id} cannot pause from status {snapshot['status']}")
        control_state = self._state(run_id)
        control_state.pause_requested = True
        control_state.paused.clear()
        control_state.resume_requested.clear()
        control_state.resumed.clear()
        paused_waiter = asyncio.create_task(control_state.paused.wait())
        closed_waiter = asyncio.create_task(control_state.closed.wait())
        done, pending = await asyncio.wait(
            {paused_waiter, closed_waiter},
            return_when=asyncio.FIRST_COMPLETED,
        )
        for waiter in pending:
            waiter.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        if control_state.closed.is_set():
            control_state.pause_requested = False
            raise ValueError(f"run {run_id} has no remaining stage boundary")
        if paused_waiter in done and control_state.paused.is_set():
            return
        control_state.pause_requested = False
        raise ValueError(f"run {run_id} has no remaining stage boundary")

    async def resume(self, run_id: str) -> None:
        snapshot = await self.store.get_run(run_id)
        if snapshot is None:
            raise ValueError(f"unknown run_id: {run_id}")
        if snapshot["status"] != RunStatus.PAUSED.value:
            raise ValueError(f"run {run_id} cannot resume from status {snapshot['status']}")
        control_state = self._state(run_id)
        control_state.resume_requested.set()
        resumed_waiter = asyncio.create_task(control_state.resumed.wait())
        closed_waiter = asyncio.create_task(control_state.closed.wait())
        done, pending = await asyncio.wait(
            {resumed_waiter, closed_waiter},
            return_when=asyncio.FIRST_COMPLETED,
        )
        for waiter in pending:
            waiter.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        if control_state.closed.is_set() and not control_state.resumed.is_set():
            raise ValueError(f"run {run_id} closed before resume")
        if resumed_waiter not in done:
            raise ValueError(f"run {run_id} closed before resume")

    async def wait_for_evidence(self, run_id: str, current_stage: str) -> None:
        control_state = self._state(run_id)
        control_state.evidence_resume_requested.clear()
        control_state.evidence_resumed.clear()
        await self.store.transition_run(
            run_id,
            {RunStatus.RUNNING},
            RunStatus.AWAITING_EVIDENCE,
            current_stage=current_stage,
        )
        resume_waiter = asyncio.create_task(control_state.evidence_resume_requested.wait())
        closed_waiter = asyncio.create_task(control_state.closed.wait())
        done, pending = await asyncio.wait(
            {resume_waiter, closed_waiter},
            return_when=asyncio.FIRST_COMPLETED,
        )
        for waiter in pending:
            waiter.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        if control_state.closed.is_set() and not control_state.evidence_resume_requested.is_set():
            raise ValueError(f"run {run_id} closed before evidence resume")
        await self.store.transition_run(
            run_id,
            {RunStatus.AWAITING_EVIDENCE},
            RunStatus.RUNNING,
            current_stage=current_stage,
        )
        control_state.evidence_resume_requested.clear()
        control_state.evidence_resumed.set()

    async def resume_evidence(self, run_id: str) -> None:
        snapshot = await self.store.get_run(run_id)
        if snapshot is None:
            raise ValueError(f"unknown run_id: {run_id}")
        if snapshot["status"] != RunStatus.AWAITING_EVIDENCE.value:
            raise ValueError(
                f"run {run_id} cannot resume evidence from status {snapshot['status']}"
            )
        control_state = self._state(run_id)
        control_state.evidence_resume_requested.set()
        resumed_waiter = asyncio.create_task(control_state.evidence_resumed.wait())
        closed_waiter = asyncio.create_task(control_state.closed.wait())
        done, pending = await asyncio.wait(
            {resumed_waiter, closed_waiter},
            return_when=asyncio.FIRST_COMPLETED,
        )
        for waiter in pending:
            waiter.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        if control_state.closed.is_set() and not control_state.evidence_resumed.is_set():
            raise ValueError(f"run {run_id} closed before evidence resume")
        if resumed_waiter not in done:
            raise ValueError(f"run {run_id} closed before evidence resume")

    async def wait_if_paused(self, run_id: str, current_stage: str) -> None:
        control_state = self._state(run_id)
        if control_state.pause_requested:
            await self._pause_at_boundary(run_id, current_stage, control_state)

    def close(self, run_id: str) -> None:
        self._state(run_id).closed.set()

    def forget(self, run_id: str) -> None:
        self._states.pop(run_id, None)

    async def execute_stage(
        self,
        run_id: str,
        current_stage: str,
        operation: Callable[[], Awaitable[T]],
    ) -> T:
        await self.wait_if_paused(run_id, current_stage)
        result = await operation()
        control_state = self._state(run_id)
        if control_state.pause_requested:
            await self._pause_at_boundary(run_id, current_stage, control_state)
        return result

    async def _pause_at_boundary(
        self,
        run_id: str,
        current_stage: str,
        control_state: _RunControlState,
    ) -> None:
        await self.store.transition_run(
            run_id,
            {RunStatus.RUNNING},
            RunStatus.PAUSED,
            current_stage=current_stage,
        )
        control_state.paused.set()
        await control_state.resume_requested.wait()
        await self.store.transition_run(
            run_id,
            {RunStatus.PAUSED},
            RunStatus.RUNNING,
            current_stage=current_stage,
        )
        control_state.pause_requested = False
        control_state.paused.clear()
        control_state.resume_requested.clear()
        control_state.resumed.set()


async def _runtime() -> tuple[RunStore, RunControl]:
    global _RUN_STORE, _RUN_CONTROL, _RUN_INITIALIZED_STORE
    global _RUNTIME_INIT_LOCK, _RUNTIME_INIT_LOOP
    if _RUN_STORE is None:
        _RUN_STORE = RunStore(db_path())
    loop = asyncio.get_running_loop()
    if _RUNTIME_INIT_LOCK is None or _RUNTIME_INIT_LOOP is not loop:
        _RUNTIME_INIT_LOCK = asyncio.Lock()
        _RUNTIME_INIT_LOOP = loop
    if _RUN_INITIALIZED_STORE is not _RUN_STORE:
        async with _RUNTIME_INIT_LOCK:
            if _RUN_INITIALIZED_STORE is not _RUN_STORE:
                await _RUN_STORE.initialize()
                _RUN_INITIALIZED_STORE = _RUN_STORE
    if _RUN_CONTROL is None or _RUN_CONTROL.store is not _RUN_STORE:
        _RUN_CONTROL = RunControl(_RUN_STORE)
    return _RUN_STORE, _RUN_CONTROL


async def _orchestrator_startup() -> None:
    run_store, _ = await _runtime()
    await run_store.interrupt_active_runs()
    await _agent_control_startup()


rest_app.add_event_handler("startup", _orchestrator_startup)


async def _agent_control_startup() -> AgentRequestClient:
    global _AGENT_BUS, _AGENT_REQUEST_CLIENT, _AGENT_RUNTIME_LOOP
    loop = asyncio.get_running_loop()
    _active_agent_request_client(loop)
    lock = _agent_control_init_lock(loop)
    if _AGENT_SHUTDOWN_COUNT:
        raise RuntimeError("Orchestrator Agent control is shutting down")
    async with lock:
        existing_client = _active_agent_request_client(loop)
        if existing_client is not None:
            return existing_client
        bus, client = await _create_agent_request_client()
        _AGENT_BUS = bus
        _AGENT_REQUEST_CLIENT = client
        _AGENT_RUNTIME_LOOP = loop
        return client


def _agent_control_init_lock(loop: asyncio.AbstractEventLoop) -> asyncio.Lock:
    global _AGENT_INIT_LOCK, _AGENT_INIT_LOOP
    if _AGENT_INIT_LOCK is not None and _AGENT_INIT_LOOP is not loop and _AGENT_INIT_LOCK.locked():
        raise RuntimeError("Orchestrator Agent control is owned by another event loop")
    if _AGENT_INIT_LOCK is None or _AGENT_INIT_LOOP is not loop:
        _AGENT_INIT_LOCK = asyncio.Lock()
        _AGENT_INIT_LOOP = loop
    return _AGENT_INIT_LOCK


def _active_agent_request_client(
    loop: asyncio.AbstractEventLoop,
) -> AgentRequestClient | None:
    if _AGENT_BUS is not None or _AGENT_REQUEST_CLIENT is not None:
        if _AGENT_RUNTIME_LOOP is not loop:
            raise RuntimeError("Orchestrator Agent control is owned by another event loop")
        if _AGENT_BUS is None or _AGENT_REQUEST_CLIENT is None:
            raise RuntimeError("Orchestrator Agent control runtime is incomplete")
        return _AGENT_REQUEST_CLIENT
    return None


async def _create_agent_request_client() -> tuple[RedisBus, AgentRequestClient]:
    bus = RedisBus(allow_fallback=False)
    try:
        try:
            await bus.connect()
        except Exception as exc:
            raise RuntimeError(f"Redis connection failed: {exc}") from exc
        if not bool(bus.is_redis):
            raise RuntimeError("production Orchestrator Agent control requires Redis")
        redis_timeout = min(1.0, agent_health_check_timeout_seconds())
        try:
            redis_ready = bool(
                await asyncio.wait_for(
                    bus.roundtrip(timeout=redis_timeout),
                    timeout=redis_timeout,
                )
            )
        except Exception as exc:
            raise RuntimeError(f"Redis roundtrip failed: {exc}") from exc
        if not redis_ready:
            raise RuntimeError("Redis roundtrip failed")
        if not BaseAgent("orchestrator").production_signing_configured:
            raise RuntimeError(
                "production Agent signing requires AGENT_MESSAGE_HMAC_SECRET or both "
                "SIGSTORE_SIGN_COMMAND and SIGSTORE_VERIFY_COMMAND"
            )
        client = AgentRequestClient(bus)
    except BaseException:
        await bus.close()
        raise
    return bus, client


async def _agent_control_shutdown() -> None:
    global _AGENT_BUS, _AGENT_REQUEST_CLIENT, _AGENT_RUNTIME_LOOP
    bus = _AGENT_BUS
    if bus is None:
        return
    if _AGENT_RUNTIME_LOOP is not asyncio.get_running_loop():
        raise RuntimeError("Orchestrator Agent control is owned by another event loop")
    await bus.close()
    _AGENT_BUS = None
    _AGENT_REQUEST_CLIENT = None
    _AGENT_RUNTIME_LOOP = None


async def _orchestrator_shutdown() -> None:
    global _AGENT_SHUTDOWN_COUNT
    if _AGENT_BUS is not None and _AGENT_RUNTIME_LOOP is not asyncio.get_running_loop():
        raise RuntimeError("Orchestrator Agent control is owned by another event loop")
    lock = _agent_control_init_lock(asyncio.get_running_loop())
    _AGENT_SHUTDOWN_COUNT += 1
    try:
        async with lock:
            current_task = asyncio.current_task()
            tasks = tuple(
                task
                for task in {
                    *_RUN_TASKS.values(),
                    *_DIRECT_AGENT_TASKS,
                }
                if task is not current_task
            )
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            _RUN_TASKS.clear()
            _DIRECT_AGENT_TASKS.clear()
            await _agent_control_shutdown()
    finally:
        _AGENT_SHUTDOWN_COUNT -= 1


rest_app.add_event_handler("shutdown", _orchestrator_shutdown)


def _validated_policy(request: dict) -> dict[str, object]:
    workflow_scope = request.get("workflow_scope")
    if not workflow_scope:
        raise HTTPException(status_code=400, detail="workflow_scope is required")
    if "validation_passed" not in request:
        raise HTTPException(status_code=400, detail="validation_passed is required")
    if "max_refinements" not in request:
        raise HTTPException(status_code=400, detail="max_refinements is required")
    if not isinstance(request["validation_passed"], bool):
        raise HTTPException(status_code=400, detail="validation_passed must be a boolean")
    max_refinements = request["max_refinements"]
    if (
        isinstance(max_refinements, bool)
        or not isinstance(max_refinements, int)
        or max_refinements < 0
    ):
        raise HTTPException(
            status_code=400,
            detail="max_refinements must be a non-negative integer",
        )
    return {
        "workflow_scope": str(workflow_scope),
        "validation_passed": request["validation_passed"],
        "max_refinements": max_refinements,
    }


@rest_app.get("/health")
async def health():
    return {"status": "healthy", "engine": "langgraph", "runs": len(_RUN_TASKS)}


def _register_design_run_task(
    run_id: str,
    request: dict,
    initial_state: dict,
    *,
    legacy_design_request: bool = False,
) -> asyncio.Task[None]:
    if legacy_design_request:
        execution = _execute_design_run(
            run_id,
            request,
            initial_state,
            legacy_design_request=True,
        )
    else:
        execution = _execute_design_run(run_id, request, initial_state)
    task = asyncio.create_task(
        execution,
        name=f"orchestrator-run-{run_id}",
    )
    _RUN_TASKS[run_id] = task
    task.add_done_callback(lambda completed, key=run_id: _finish_run_task(key, completed))
    return task


@rest_app.post("/v1/orchestrator/design", status_code=202)
async def create_design_run(request: dict) -> dict:
    request = dict(request)
    legacy_design_request = request.pop(_INTERNAL_LEGACY_DESIGN_REQUEST, False) is True
    nl_input = request.get("nl_input") or request.get("intent")
    if not nl_input:
        raise HTTPException(status_code=400, detail="nl_input is required")
    policy = _validated_policy(request)
    workflow_scope = policy["workflow_scope"]
    run_store, _ = await _runtime()
    default_run_id = (
        f"design-{uuid.uuid4().hex[:10]}" if legacy_design_request else f"run-{uuid.uuid4().hex}"
    )
    run_id = str(request.get("run_id") or default_run_id)
    created_at = datetime.now(UTC).isoformat()
    trace_id = str(request.get("trace_id") or f"trace-{uuid.uuid4().hex}")
    initial_state = create_initial_state(
        str(nl_input),
        run_id=run_id,
        trace_id=trace_id,
        artifact_ids=request.get("artifact_ids") or [],
        workflow_scope=str(workflow_scope),
    )
    initial_request = dict(request)
    initial_request.pop("clients", None)
    initial_state["request"] = initial_request
    initial_state["validation_passed"] = bool(policy["validation_passed"])
    initial_state["max_refinements"] = int(policy["max_refinements"])
    create_run_task = asyncio.create_task(
        run_store.create_run(
            run_id,
            intent=str(nl_input),
            policy=policy,
            created_at=created_at,
            project_id=request.get("project_id"),
            state=_persistable_state(initial_state),
            require_new=True,
        )
    )
    try:
        await asyncio.shield(create_run_task)
    except asyncio.CancelledError:
        try:
            await _await_task_completion(create_run_task)
        except (asyncio.CancelledError, Exception):
            run_owned = False
        else:
            run_owned = True
        if run_owned:
            _register_design_run_task(
                run_id,
                dict(request),
                initial_state,
                legacy_design_request=legacy_design_request,
            )
        raise
    except RunAlreadyExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    queued_snapshot = await run_store.get_run(run_id)
    if queued_snapshot is None:
        raise RuntimeError(f"run was not persisted: {run_id}")
    _register_design_run_task(
        run_id,
        dict(request),
        initial_state,
        legacy_design_request=legacy_design_request,
    )
    if legacy_design_request:
        return {"design_id": run_id, **queued_snapshot}
    return {"design_id": run_id, "run_id": run_id, "status": RunStatus.QUEUED.value}


async def _execute_design_run(
    run_id: str,
    request: dict,
    state: dict,
    *,
    legacy_design_request: bool = False,
) -> None:
    run_store, run_control = await _runtime()
    try:
        await run_store.transition_run(
            run_id,
            {RunStatus.QUEUED},
            RunStatus.RUNNING,
            current_stage="planning",
            state=state,
        )
        final_state = await _invoke_workflow(request, state, run_control=run_control)
        if str(request["workflow_scope"]) == "full":
            await _record_workflow_provenance(final_state)
        status = _workflow_terminal_status(
            final_state,
            str(request["workflow_scope"]),
            legacy_design_request=legacy_design_request,
        )
        run_control.close(run_id)
        await _persist_workflow_result(run_store, run_id, final_state, status)
    except asyncio.CancelledError as exc:
        interruption_task = asyncio.create_task(
            _interrupt_cancelled_run(
                run_store,
                run_control,
                run_id,
                str(exc) or "workflow task cancelled",
            )
        )
        await _await_task_completion(interruption_task)
        raise
    except Exception as exc:
        try:
            snapshot = await run_store.get_run(run_id)
        except asyncio.CancelledError as cancellation:
            interruption_task = asyncio.create_task(
                _interrupt_cancelled_run(
                    run_store,
                    run_control,
                    run_id,
                    str(cancellation) or "workflow task cancelled",
                )
            )
            await _await_task_completion(interruption_task)
            raise
        if snapshot is not None and snapshot["status"] in _NONTERMINAL_RUN_STATUS_VALUES:
            failure_error_type = type(exc).__name__
            failure_error_message = str(exc)
            failure_task = asyncio.create_task(
                run_store.transition_run(
                    run_id,
                    {RunStatus(str(snapshot["status"]))},
                    RunStatus.FAILED,
                    current_stage=str(snapshot.get("current_stage") or "failed"),
                    error_type=failure_error_type,
                    error_message=failure_error_message,
                )
            )
            try:
                await asyncio.shield(failure_task)
            except asyncio.CancelledError as cancellation:
                await _compensate_cancelled_failure_persistence(
                    run_store,
                    failure_task,
                    run_id,
                    failure_error_type,
                    failure_error_message,
                    str(cancellation) or "workflow task cancelled",
                )
                raise
            except ValueError:
                LOGGER.exception("run %s failed while recording failure", run_id)
        raise
    finally:
        run_control.close(run_id)
        run_control.forget(run_id)


async def _await_task_completion(task: asyncio.Task[T]) -> T:
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            continue
    return task.result()


async def _compensate_cancelled_failure_persistence(
    run_store: RunStore,
    failure_task: asyncio.Task[None],
    run_id: str,
    expected_error_type: str,
    expected_error_message: str,
    cancellation_message: str,
) -> None:
    try:
        await _await_task_completion(failure_task)
    except (asyncio.CancelledError, Exception):
        LOGGER.exception("run %s failed while recording failure", run_id)
    compensation_task = asyncio.create_task(
        run_store.compensate_cancelled_failure(
            run_id,
            expected_error_type=expected_error_type,
            expected_error_message=expected_error_message,
            cancellation_message=cancellation_message,
        )
    )
    await _await_task_completion(compensation_task)


async def _interrupt_cancelled_run(
    run_store: RunStore,
    run_control: RunControl,
    run_id: str,
    error_message: str,
) -> dict | None:
    try:
        snapshot = await run_store.get_run(run_id)
        if snapshot is None:
            return None
        if snapshot["status"] in _NONTERMINAL_RUN_STATUS_VALUES:
            try:
                await run_store.transition_run(
                    run_id,
                    set(_NONTERMINAL_RUN_STATUSES),
                    RunStatus.INTERRUPTED,
                    current_stage=str(snapshot.get("current_stage") or "interrupted"),
                    error_type=asyncio.CancelledError.__name__,
                    error_message=error_message,
                )
            except ValueError:
                current = await run_store.get_run(run_id)
                if current is not None and current["status"] in _NONTERMINAL_RUN_STATUS_VALUES:
                    raise
        return await run_store.get_run(run_id)
    finally:
        run_control.close(run_id)


async def _persist_workflow_result(
    run_store: RunStore,
    run_id: str,
    final_state: dict,
    status: RunStatus,
) -> None:
    existing = {int(event["step_index"]) for event in await run_store.list_events(run_id)}
    for event in final_state.get("events", []):
        step_index = int(event.get("event_index", len(existing)))
        if step_index in existing:
            continue
        await run_store.append_event(
            run_id,
            step_index,
            stage=str(event.get("stage", "")),
            payload=dict(event),
            timestamp=str(event.get("timestamp") or datetime.now(UTC).isoformat()),
        )
        existing.add(step_index)
    await run_store.transition_run(
        run_id,
        {RunStatus.RUNNING},
        status,
        current_stage=str(final_state.get("status", "")).lower(),
        state=_persistable_state(final_state),
    )


def _persistable_state(state: dict) -> dict:
    persisted = dict(state)
    request = persisted.get("request")
    if isinstance(request, dict):
        persisted_request = dict(request)
        persisted_request.pop("clients", None)
        persisted["request"] = persisted_request
    return persisted


def _workflow_terminal_status(
    final_state: dict,
    workflow_scope: str,
    *,
    legacy_design_request: bool = False,
) -> RunStatus:
    if str(final_state.get("status")) != "ESCALATING":
        return RunStatus.COMPLETED
    validation = final_state.get("validation")
    if (
        legacy_design_request
        and workflow_scope == "engineering"
        and isinstance(validation, dict)
        and validation.get("reason") == "no valid candidates"
        and validation.get("results") == []
        and not bool(final_state.get("validation_passed", False))
    ):
        return RunStatus.COMPLETED
    return RunStatus.REJECTED


def _finish_run_task(run_id: str, task: asyncio.Task[None]) -> None:
    _RUN_TASKS.pop(run_id, None)
    if task.cancelled():
        LOGGER.info("orchestrator run %s cancelled", run_id)
        return
    exception = task.exception()
    if exception is not None:
        LOGGER.error(
            "orchestrator run %s failed",
            run_id,
            exc_info=(type(exception), exception, exception.__traceback__),
        )


async def _invoke_workflow(
    request: dict,
    state: dict,
    *,
    clients: object | None = None,
    run_control: RunControl | None = None,
) -> dict:
    business_request = dict(request)
    injected_clients = business_request.pop("clients", None)
    if clients is None:
        clients = injected_clients
    workflow_scope = str(business_request["workflow_scope"])
    state["request"] = business_request
    state["validation_passed"] = bool(business_request["validation_passed"])
    state["max_refinements"] = int(business_request["max_refinements"])
    if clients is None and workflow_scope in {"engineering", "full"}:
        clients = _default_workflow_clients(
            workflow_scope,
            _shared_agent_request_client(),
        )
    compiled = WorkflowGraph(clients=clients, workflow_scope=workflow_scope).build()
    if run_control is None or not hasattr(compiled, "astream"):
        return await compiled.ainvoke(state)
    return await _stream_workflow_stages(compiled, state, run_control)


async def _stream_workflow_stages(
    compiled: object,
    state: dict,
    run_control: RunControl,
) -> dict:
    run_id = str(state["run_id"])
    final_state = state
    persisted_steps: set[int] = set()
    await run_control.wait_if_paused(run_id, "planning")
    async for stage_state in compiled.astream(state, stream_mode="values"):
        if not isinstance(stage_state, dict):
            raise RuntimeError("workflow stage state must be a dict")
        final_state = stage_state
        events = list(stage_state.get("events", []))
        if events:
            event = events[-1]
            step_index = int(event.get("event_index", len(events) - 1))
            if step_index not in persisted_steps:
                await run_control.store.append_event(
                    run_id,
                    step_index,
                    stage=str(event.get("stage", "")),
                    payload=dict(event),
                    timestamp=str(event.get("timestamp") or datetime.now(UTC).isoformat()),
                    state=_persistable_state(stage_state),
                )
                persisted_steps.add(step_index)
        current_stage = str(stage_state.get("status") or "planning").lower()
        if current_stage == RunStatus.AWAITING_EVIDENCE.value:
            await run_control.wait_for_evidence(run_id, current_stage)
            await run_control.wait_if_paused(run_id, current_stage)
        else:
            await run_control.wait_if_paused(run_id, current_stage)
    return final_state


def _shared_agent_request_client() -> AgentRequestClient:
    if _AGENT_SHUTDOWN_COUNT:
        raise RuntimeError("Orchestrator Agent control is shutting down")
    if _AGENT_REQUEST_CLIENT is None:
        raise RuntimeError("Orchestrator Agent control is not initialized")
    if _AGENT_RUNTIME_LOOP is not asyncio.get_running_loop():
        raise RuntimeError("Orchestrator Agent control is owned by another event loop")
    return _AGENT_REQUEST_CLIENT


def _same_loop_shared_agent_request_client() -> AgentRequestClient | None:
    if _AGENT_SHUTDOWN_COUNT:
        return None
    if _AGENT_REQUEST_CLIENT is None:
        return None
    if _AGENT_RUNTIME_LOOP is not asyncio.get_running_loop():
        return None
    return _AGENT_REQUEST_CLIENT


def _default_workflow_clients(
    workflow_scope: str,
    request_client: AgentRequestClient,
) -> EngineeringWorkflowClients | FullWorkflowClients:
    if workflow_scope == "engineering":
        return EngineeringWorkflowClients(request_client=request_client)
    if workflow_scope == "full":
        return FullWorkflowClients(request_client=request_client)
    raise ValueError(f"unsupported default workflow scope: {workflow_scope}")


@rest_app.post("/v1/orchestrator/projects")
async def create_project_record(request: dict) -> dict:
    name = request.get("name")
    description = request.get("description", "")
    if not isinstance(name, str) or not name:
        raise HTTPException(status_code=400, detail="name is required")
    if not isinstance(description, str):
        raise HTTPException(status_code=400, detail="description must be a string")
    run_store, _ = await _runtime()
    return await run_store.create_project(
        name,
        name=name,
        description=description,
        created_at=datetime.now(UTC).isoformat(),
    )


@rest_app.get("/v1/orchestrator/projects")
async def list_project_records() -> dict:
    run_store, _ = await _runtime()
    projects = await run_store.list_projects()
    return {"projects": projects, "n_projects": len(projects)}


@rest_app.get("/v1/orchestrator/projects/{project_id:path}")
async def get_project_record(project_id: str) -> dict:
    run_store, _ = await _runtime()
    project = await run_store.get_project(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")
    return project


@rest_app.delete("/v1/orchestrator/projects/{project_id:path}")
async def delete_project_record(project_id: str) -> dict:
    run_store, _ = await _runtime()
    if not await run_store.delete_project(project_id):
        raise HTTPException(status_code=404, detail="Project not found")
    return {"deleted": True, "project_id": project_id}


@rest_app.get("/v1/orchestrator/runs")
async def get_runs(
    page_size: int = 50,
    page_token: str | None = None,
) -> dict:
    run_store, _ = await _runtime()
    try:
        page = await run_store.list_runs(
            page_size=page_size,
            page_token=page_token,
            context={},
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "runs": page["items"],
        "next_page_token": page["next_page_token"],
    }


@rest_app.get("/v1/orchestrator/runs/{run_id}")
async def get_run_snapshot(run_id: str) -> dict:
    run_store, _ = await _runtime()
    snapshot = await run_store.get_run(run_id)
    if snapshot is None:
        raise HTTPException(status_code=404, detail=f"Unknown run_id: {run_id}")
    return snapshot


@rest_app.get("/v1/orchestrator/runs/{run_id}/events")
async def get_run_events(run_id: str, after_step: int = -1) -> dict:
    run_store, _ = await _runtime()
    if await run_store.get_run(run_id) is None:
        raise HTTPException(status_code=404, detail=f"Unknown run_id: {run_id}")
    try:
        events = await run_store.list_events(run_id, after_step=after_step)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"run_id": run_id, "events": events}


@rest_app.post("/v1/orchestrator/runs/{run_id}/pause")
async def pause_run(run_id: str) -> dict:
    _, run_control = await _runtime()
    try:
        await run_control.pause(run_id)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"design_id": run_id, "run_id": run_id, "status": RunStatus.PAUSED.value}


@rest_app.post("/v1/orchestrator/runs/{run_id}/resume")
async def resume_run(run_id: str) -> dict:
    _, run_control = await _runtime()
    try:
        await run_control.resume(run_id)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"design_id": run_id, "run_id": run_id, "status": RunStatus.RUNNING.value}


@rest_app.post("/v1/orchestrator/runs/{run_id}/evidence/resume")
async def resume_evidence_run(run_id: str) -> dict:
    _, run_control = await _runtime()
    try:
        await run_control.resume_evidence(run_id)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"design_id": run_id, "run_id": run_id, "status": RunStatus.RUNNING.value}


@rest_app.post("/v1/orchestrator/runs/{run_id}/cancel")
async def cancel_run(run_id: str) -> dict:
    run_store, run_control = await _runtime()
    snapshot = await run_store.get_run(run_id)
    if snapshot is None:
        raise HTTPException(status_code=404, detail=f"Unknown run_id: {run_id}")
    if snapshot["status"] == RunStatus.INTERRUPTED.value:
        return snapshot
    if snapshot["status"] not in _NONTERMINAL_RUN_STATUS_VALUES:
        raise HTTPException(
            status_code=409,
            detail=f"run {run_id} cannot cancel from status {snapshot['status']}",
        )
    task = _RUN_TASKS.get(run_id)
    if task is None or task.done():
        snapshot = await run_store.get_run(run_id)
        if snapshot is not None and snapshot["status"] == RunStatus.INTERRUPTED.value:
            return snapshot
        raise HTTPException(
            status_code=409,
            detail=f"run {run_id} has no active in-process task",
        )

    task.cancel("run cancelled by client request")
    await asyncio.gather(task, return_exceptions=True)
    snapshot = await run_store.get_run(run_id)
    if (
        task.cancelled()
        and snapshot is not None
        and snapshot["status"] in _NONTERMINAL_RUN_STATUS_VALUES
    ):
        snapshot = await _interrupt_cancelled_run(
            run_store,
            run_control,
            run_id,
            "run cancelled by client request",
        )
    if snapshot is not None and snapshot["status"] == RunStatus.INTERRUPTED.value:
        return snapshot
    status = snapshot["status"] if snapshot is not None else "unknown"
    raise HTTPException(
        status_code=409,
        detail=f"run {run_id} cancellation did not interrupt status {status}",
    )


async def start_design(request: dict):
    """Run a workflow inline for the gRPC and direct Python compatibility API."""
    request = dict(request)
    legacy_design_request = request.pop(_INTERNAL_LEGACY_DESIGN_REQUEST, False) is True
    nl_input = request.get("nl_input") or request.get("intent")
    if not nl_input:
        raise HTTPException(status_code=400, detail="nl_input is required")
    policy = _validated_policy(request)
    workflow_scope = str(policy["workflow_scope"])
    state = create_initial_state(
        str(nl_input),
        run_id=request.get("run_id"),
        trace_id=request.get("trace_id"),
        artifact_ids=request.get("artifact_ids") or [],
        workflow_scope=workflow_scope,
    )
    inline_request = dict(request)
    inline_request["workflow_scope"] = workflow_scope
    workflow_clients = inline_request.pop("clients", None)
    run_id = str(state["run_id"])
    run_store, run_control = await _runtime()
    run_owned = False
    run_started = False
    local_agent_bus: RedisBus | None = None
    shared_agent_task: asyncio.Task[object] | None = None
    try:
        create_run_task = asyncio.create_task(
            run_store.create_run(
                run_id,
                intent=str(nl_input),
                policy=policy,
                created_at=datetime.now(UTC).isoformat(),
                project_id=request.get("project_id"),
                state={
                    "run_id": run_id,
                    "trace_id": state["trace_id"],
                    "artifact_ids": list(state.get("artifact_ids", [])),
                },
                require_new=True,
            )
        )
        try:
            await asyncio.shield(create_run_task)
        except asyncio.CancelledError:
            try:
                await _await_task_completion(create_run_task)
            except (asyncio.CancelledError, Exception):
                run_owned = False
            else:
                run_owned = True
            raise
        run_owned = True
        await run_store.transition_run(
            run_id,
            {RunStatus.QUEUED},
            RunStatus.RUNNING,
            current_stage="planning",
        )
        run_started = True
        if workflow_clients is None and workflow_scope in {"engineering", "full"}:
            request_client = _same_loop_shared_agent_request_client()
            if request_client is None:
                local_agent_bus, request_client = await _create_agent_request_client()
            else:
                shared_agent_task = asyncio.current_task()
                if shared_agent_task is None:
                    raise RuntimeError("direct Agent workflow requires an asyncio task")
                _DIRECT_AGENT_TASKS.add(shared_agent_task)
            workflow_clients = _default_workflow_clients(
                workflow_scope,
                request_client,
            )
        final_state = await _invoke_workflow(
            inline_request,
            state,
            clients=workflow_clients,
        )
        if workflow_scope == "full":
            await _record_workflow_provenance(final_state)
        terminal_status = _workflow_terminal_status(
            final_state,
            workflow_scope,
            legacy_design_request=legacy_design_request,
        )
        await _persist_workflow_result(
            run_store,
            run_id,
            final_state,
            terminal_status,
        )
    except asyncio.CancelledError as exc:
        if run_owned:
            interruption_task = asyncio.create_task(
                _interrupt_cancelled_run(
                    run_store,
                    run_control,
                    run_id,
                    str(exc) or "workflow task cancelled",
                )
            )
            await _await_task_completion(interruption_task)
        raise
    except Exception as exc:
        if run_started:
            failure_error_type = type(exc).__name__
            failure_error_message = str(exc)
            failure_task = asyncio.create_task(
                run_store.transition_run(
                    run_id,
                    {RunStatus.RUNNING},
                    RunStatus.FAILED,
                    current_stage=str(state.get("status", "failed")).lower(),
                    error_type=failure_error_type,
                    error_message=failure_error_message,
                )
            )
            try:
                await asyncio.shield(failure_task)
            except asyncio.CancelledError as cancellation:
                await _compensate_cancelled_failure_persistence(
                    run_store,
                    failure_task,
                    run_id,
                    failure_error_type,
                    failure_error_message,
                    str(cancellation) or "workflow task cancelled",
                )
                raise
        raise
    finally:
        if shared_agent_task is not None:
            _DIRECT_AGENT_TASKS.discard(shared_agent_task)
        if local_agent_bus is not None:
            await local_agent_bus.close()
        if run_owned:
            run_control.close(run_id)
            run_control.forget(run_id)
    status = terminal_status.value
    return {
        "design_id": run_id,
        "run_id": run_id,
        "trace_id": final_state.get("trace_id"),
        "status": status,
        "current_stage": final_state.get("status"),
        "artifact_ids": final_state.get("artifact_ids", []),
        "history": final_state.get("history", []),
        "pipeline": final_state.get("history", []),
        "events": final_state.get("events", []),
        "state": final_state,
    }


@rest_app.get("/v1/orchestrator/{design_id}")
async def get_design_status(design_id: str):
    """Get design workflow status from the LangGraph state machine."""
    run_store, _ = await _runtime()
    run = await run_store.get_run(design_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Unknown design_id: {design_id}")
    state = run.get("state") or {}
    events = await run_store.list_events(design_id)
    return {
        "design_id": design_id,
        "run_id": design_id,
        "trace_id": state.get("trace_id"),
        "status": run["status"],
        "current_stage": run.get("current_stage"),
        "artifact_ids": state.get("artifact_ids", []),
        "history": state.get("history", []),
        "stages_completed": len(state.get("history", [])),
        "stages_total": len(state.get("history", [])),
        "events": events,
        "state": state,
    }


@rest_app.post("/v1/orchestrator/{design_id}/pause")
async def pause_design(design_id: str):
    """Pause a running design workflow."""
    return await pause_run(design_id)


@rest_app.post("/v1/orchestrator/{design_id}/resume")
async def resume_design(design_id: str):
    """Resume a paused design workflow."""
    return await resume_run(design_id)


class OrchestratorServicer:
    async def StartPipeline(self, request, context):
        """gRPC: Start a new design pipeline."""
        project_id = getattr(request, "project_id", "")
        objectives = getattr(request, "objectives", [])
        pipeline_request = {
            "nl_input": getattr(request, "nl_input", ""),
            "workflow_scope": getattr(request, "workflow_scope", ""),
            "run_id": getattr(request, "run_id", None) or None,
            "trace_id": getattr(request, "trace_id", None) or None,
        }
        if project_id:
            pipeline_request["project_id"] = project_id
        if request.HasField("validation_passed"):
            pipeline_request["validation_passed"] = request.validation_passed
        if request.HasField("max_refinements"):
            pipeline_request["max_refinements"] = request.max_refinements
        response = await start_design(pipeline_request)

        return type(
            "PipelineResponse",
            (),
            {
                "design_id": response["design_id"],
                "run_id": response["run_id"],
                "trace_id": response["trace_id"],
                "project_id": project_id,
                "status": response["status"],
                "n_objectives": len(objectives),
            },
        )()

    async def GetPipelineState(self, request, context):
        """gRPC: Get pipeline state from LangGraph."""
        design_id = getattr(request, "design_id", "")
        state = await get_design_status(design_id)
        return type(
            "PipelineStateResponse",
            (),
            {
                "design_id": design_id,
                "current_stage": state["current_stage"],
                "state_json": str(state["state"]),
            },
        )()


class OrchestratorGrpcServicer(orchestrator_pb2_grpc.OrchestratorServiceServicer):
    def __init__(self, service: OrchestratorServicer | None = None):
        self.service = service or OrchestratorServicer()

    async def StartPipeline(self, request, context):
        response = await self.service.StartPipeline(request, context)
        return orchestrator_pb2.PipelineResponse(
            design_id=str(response.design_id),
            run_id=str(response.run_id),
            trace_id=str(response.trace_id),
            project_id=str(response.project_id),
            status=str(response.status),
            n_objectives=int(response.n_objectives),
        )

    async def GetPipelineState(self, request, context):
        response = await self.service.GetPipelineState(request, context)
        return orchestrator_pb2.PipelineStateResponse(
            design_id=str(response.design_id),
            current_stage=str(response.current_stage),
            state_json=str(response.state_json),
        )


async def _request_agent(
    request_client: AgentRequestClient | None,
    state: dict,
    entry_point: str,
    stage: str,
    business_payload: dict,
    *,
    candidate_index: int | None = None,
) -> dict:
    run_id = str(state.get("run_id") or "")
    trace_id = str(state.get("trace_id") or "")
    if not run_id:
        raise ValueError("run_id is required for Orchestrator Agent requests")
    if not trace_id:
        raise ValueError("trace_id is required for Orchestrator Agent requests")
    protocol = _AGENT_PROTOCOLS_BY_ENTRY_POINT[entry_point]
    refinement_count = int(state.get("refinement_count", 0))
    parent_id = f"{run_id}:{stage}:{refinement_count}"
    request_id = f"{run_id}:{entry_point}:{refinement_count}"
    if candidate_index is not None:
        request_id = f"{request_id}:candidate-{candidate_index}"
    payload = dict(business_payload)
    payload.update(
        {
            "trace_id": trace_id,
            "parent_id": parent_id,
            "run_id": run_id,
            "request_id": request_id,
            "schema_version": protocol.schema_version,
        }
    )
    client = request_client or _shared_agent_request_client()
    result = await client.request(
        protocol.subject,
        payload,
        payload_type_url=protocol.payload_type_url,
        timeout=_agent_request_timeout(state),
    )
    business_result = dict(result)
    for field in ("run_id", "request_id", "schema_version"):
        business_result.pop(field, None)
    return business_result


def _agent_request_timeout(state: dict) -> float:
    request = state.get("request")
    configured = (
        request.get("agent_request_timeout_seconds", 30.0) if isinstance(request, dict) else 30.0
    )
    timeout = float(configured)
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("agent_request_timeout_seconds must be positive")
    return timeout


class EngineeringWorkflowClients:
    """Local, resource-light clients for the reduced engineering workflow."""

    def __init__(self, request_client: AgentRequestClient | None = None) -> None:
        self.request_client = request_client

    async def compile_intent(self, state: dict) -> dict:
        from cig_compiler_svc.domain.compiler import CIGCompiler, CompilerMode, EncodingMode

        compiler = CIGCompiler(
            mode=CompilerMode.LOCAL_DEMO,
            encoding_mode=EncodingMode.HASH,
            enable_grounding=False,
        )
        cig, hciv, cone = await compiler.compile(state["nl_input"])
        return {
            "cig": cig.model_dump(mode="json"),
            "hciv": hciv.model_dump(mode="json"),
            "intent_cone": cone.model_dump(mode="json"),
        }

    async def generate_candidates(self, state: dict) -> list[dict]:
        request = state.get("request", {})
        n_samples = int(request.get("n_samples", request.get("batch_size", 8)) or 8)
        seed = request.get("seed")
        seed_smiles = request.get("seed_smiles")
        if isinstance(seed_smiles, list) and seed_smiles:
            offset = abs(int(seed or 0)) % len(seed_smiles)
            return _engineering_candidate_rows(
                [
                    {"smiles": seed_smiles[(index + offset) % len(seed_smiles)]}
                    for index in range(n_samples)
                ]
            )

        from mf_generators.rdkit_random import RDKitRandomGenerator

        generator = RDKitRandomGenerator(seed=int(seed) if seed is not None else 42)
        candidates = []
        async for molecule in generator.generate(
            state.get("hciv"),
            state.get("intent_cone"),
            state.get("cig"),
            n_samples=n_samples,
            seed=int(seed) if seed is not None else 42,
        ):
            candidates.append(molecule.model_dump(mode="json"))
        return _engineering_candidate_rows(candidates)

    async def validate_candidates(self, state: dict) -> dict:
        from mf_chem.predict.engine import MolPredictEngine
        from mf_oracles.rdkit_oracle.oracle import RDKitOracle

        candidates = list(state.get("candidates", []))
        if not candidates:
            return {"passed": False, "results": [], "reason": "no candidates generated"}
        threshold = float(state.get("request", {}).get("l0_threshold", 0.0))
        predictor = MolPredictEngine(device_ids=[])
        occurrences = []
        for candidate in candidates:
            smiles_item = str(candidate.get("canonical_smiles") or candidate.get("smiles") or "")
            if not smiles_item:
                continue
            properties = _engineering_candidate_properties(predictor, smiles_item)
            if properties.get("valid") is not True:
                continue
            occurrences.append((candidate, smiles_item, properties))
        if not occurrences:
            return {
                "passed": False,
                "threshold": threshold,
                "results": [],
                "reason": "no valid candidates",
            }
        unique_smiles = list(dict.fromkeys(smiles_item for _, smiles_item, _ in occurrences))
        results = await RDKitOracle().evaluate(unique_smiles, ["admet_score"])
        rows = []
        for candidate, smiles_item, properties in occurrences:
            row = {
                **candidate,
                **properties,
                **results[smiles_item],
            }
            rows.append(_normalise_engineering_critic_properties(row))
        ranked_rows = _rank_engineering_results(rows)
        return {
            "passed": any(float(row.get("admet_score", 0.0)) >= threshold for row in ranked_rows),
            "threshold": threshold,
            "results": ranked_rows,
        }

    async def plan_routes(self, state: dict) -> dict:
        if not os.environ.get("AIZYNTH_CONFIG_PATH"):
            return {
                "skipped": True,
                "reason": "AIZYNTH_CONFIG_PATH is not configured",
            }
        candidates = list(state.get("candidates", []))
        if not candidates:
            return {"skipped": True, "reason": "no candidate available for retrosynthesis"}
        from mf_retrosyn.aizynth.retrosyn import AiZynthRetrosyn

        planner = AiZynthRetrosyn.from_env()
        routes = await planner.find_routes(candidates[0]["canonical_smiles"], max_routes=3)
        return {"skipped": False, "routes": routes}

    async def review_candidates(self, state: dict) -> dict:
        candidates = list(state.get("candidates", []))
        if not candidates:
            return {"verdict": "fail", "reason": "no candidate available for critic"}

        properties = {}
        validation_rows = state.get("validation", {}).get("results", [])
        if validation_rows:
            properties = dict(_best_engineering_validation_row(validation_rows))
        return await _request_agent(
            self.request_client,
            state,
            "critic",
            "critic",
            {
                "smiles": _best_engineering_candidate_smiles(state),
                "properties": properties,
            },
        )


class FullWorkflowClients(EngineeringWorkflowClients):
    async def generate_candidates(self, state: dict) -> list[dict]:
        request = state.get("request", {})
        n_samples = int(request.get("n_samples", request.get("batch_size", 4)) or 4)
        generator_params = dict(request.get("generator_params") or {})
        generator_params.setdefault("sampling_seed", int(request.get("seed", 42) or 42))
        await _attach_generation_feedback(generator_params, state)
        generation_strategy = request.get("generation_strategy")
        return await _generate_with_generator_coord(
            self.request_client,
            state,
            request,
            n_samples,
            generator_params,
            (str(generation_strategy) if generation_strategy not in (None, "") else None),
        )

    async def validate_candidates(self, state: dict) -> dict:
        candidates = list(state.get("candidates", []))
        if not candidates:
            return {"passed": False, "results": [], "reason": "no candidates generated"}
        request = state.get("request", {})
        oracle_level = _requested_oracle_level(request)
        return await _validate_with_oracle_cascade(
            self.request_client,
            state,
            candidates,
            oracle_level,
        )

    async def plan_routes(self, state: dict) -> dict:
        request = state.get("request", {})
        return await _request_agent(
            self.request_client,
            state,
            "retrosyn",
            "retrosyn",
            {
                "project_id": str(request.get("project_id") or ""),
                "run_id": str(state.get("run_id", "")),
                "smiles": _first_candidate_smiles(state),
                "max_routes": int(
                    request.get("retrosyn_max_routes", request.get("max_routes", 3)) or 3
                ),
            },
        )

    async def assess_supply(self, state: dict) -> dict:
        route = _first_retrosyn_route_or_none(state)
        if route is None:
            return _unavailable_supply_result(state, "retrosyn.routes is empty")

        return await _request_agent(
            self.request_client,
            state,
            "supply",
            "supply",
            {
                "project_id": str(state.get("request", {}).get("project_id") or ""),
                "run_id": str(state.get("run_id", "")),
                "smiles": _first_candidate_smiles(state),
                "building_blocks": _route_building_blocks(route),
            },
        )

    async def compile_synthesis(self, state: dict) -> dict:
        if _supply_feasibility(state) == "unavailable":
            return {
                "status": "skipped",
                "protocols": [],
                "skip_reason": "supply feasibility is unavailable",
            }
        route = _first_retrosyn_route_or_none(state)
        if route is None:
            return {
                "status": "skipped",
                "protocols": [],
                "skip_reason": "retrosyn.routes is empty",
            }

        return await _request_agent(
            self.request_client,
            state,
            "srb",
            "srb",
            {
                "project_id": str(state.get("request", {}).get("project_id") or ""),
                "run_id": str(state.get("run_id", "")),
                "molecule": {"smiles": _first_candidate_smiles(state)},
                "retrosyn_route": route,
            },
        )

    async def review_candidates(self, state: dict) -> dict:
        candidates = list(state.get("candidates", []))
        if not candidates:
            return {"verdict": "fail", "reason": "no candidate available for critic"}

        request = dict(state.get("request") or {})
        smiles = _best_engineering_candidate_smiles(state)
        properties = _full_workflow_critic_properties(state, smiles)
        return await _request_agent(
            self.request_client,
            state,
            "critic",
            "critic",
            {
                "project_id": str(request.get("project_id") or ""),
                "run_id": str(state.get("run_id", "")),
                "smiles": smiles,
                "properties": properties,
            },
        )


async def _generate_with_generator_coord(
    request_client: AgentRequestClient | None,
    state: dict,
    request: dict,
    n_samples: int,
    generator_params: dict,
    generation_strategy: str | None,
) -> list[dict]:
    run_id = str(state.get("run_id", ""))
    payload = {
        "project_id": str(request.get("project_id") or ""),
        "run_id": run_id,
        "request_id": run_id,
        "objectives": dict(request.get("objectives") or {}),
        "hciv": state.get("hciv"),
        "intent_cone": state.get("intent_cone"),
        "n_samples": n_samples,
        "batch_size": n_samples,
        "generator_params": dict(generator_params),
    }
    if generation_strategy is not None:
        payload["generation_strategy"] = generation_strategy
    result = await _request_agent(
        request_client,
        state,
        "generator_coord",
        "generating",
        payload,
    )
    candidates = result.get("candidates")
    if not isinstance(candidates, list):
        raise RuntimeError("generator_coord Agent must return candidates as a list")
    return _normalise_candidate_rows(candidates)


async def _attach_generation_feedback(generator_params: dict, state: dict) -> None:
    jmcg_feedback = await _jmcg_context_feedback_from_state(state)
    feedback = state.get("generation_feedback")
    if isinstance(feedback, list) and feedback:
        generator_params["generation_feedback"] = json.dumps(feedback, sort_keys=True)
        property_feedback = _property_jmcg_feedback_from_generation_feedback(
            feedback,
            run_id=str(state.get("run_id") or ""),
            project_id=str((state.get("request") or {}).get("project_id") or ""),
        )
        jmcg_feedback = _merge_jmcg_feedback(jmcg_feedback, property_feedback)
    if jmcg_feedback:
        generator_params["jmcg_feedback"] = json.dumps(
            jmcg_feedback,
            sort_keys=True,
        )


async def _jmcg_context_feedback_from_state(state: dict) -> dict | None:
    run_id = str(state.get("run_id") or "")
    request = dict(state.get("request") or {})
    project_id = str(request.get("project_id") or "")
    records = []
    intent_record = _intent_jmcg_feedback_record(state, run_id)
    if intent_record:
        records.append(intent_record)
    pocket_record = await _pocket_jmcg_feedback_record(state, run_id)
    if pocket_record:
        records.append(pocket_record)
    if not records:
        return None
    return {
        "schema": "moleculeforge.jmcg.feedback.v1",
        "run_id": run_id,
        "project_id": project_id,
        "records": records,
    }


def _merge_jmcg_feedback(left: dict | None, right: dict | None) -> dict | None:
    if not left:
        return right
    if not right:
        return left
    merged = dict(left)
    merged["records"] = list(left.get("records") or []) + list(right.get("records") or [])
    return merged


def _intent_jmcg_feedback_record(state: dict, run_id: str) -> dict | None:
    hciv = state.get("hciv")
    intent_cone = state.get("intent_cone")
    if not isinstance(hciv, dict) and not isinstance(intent_cone, dict):
        return None
    metadata = {}
    if isinstance(hciv, dict):
        metadata["has_hciv"] = True
        metadata["hciv_keys"] = sorted(str(key) for key in hciv.keys())
    if isinstance(intent_cone, dict):
        metadata["has_intent_cone"] = True
        metadata["intent_cone_keys"] = sorted(str(key) for key in intent_cone.keys())
        if "half_angle" in intent_cone:
            metadata["half_angle"] = intent_cone["half_angle"]
    record = {
        "kind": "intent",
        "source": "orchestrator_svc",
        "run_id": run_id,
        "subject": {"type": "intent", "id": run_id},
        "weight": 1.0,
        "polarity": "attract",
        "confidence": 1.0,
        "evidence_ids": [],
        "metadata": metadata,
    }
    embedding, embedding_metadata = _intent_feedback_embedding(state)
    if embedding is not None:
        record["humu_embedding"] = embedding
        record["metadata"].update(embedding_metadata)
    return record


async def _pocket_jmcg_feedback_record(state: dict, run_id: str) -> dict | None:
    cig = state.get("cig")
    if not isinstance(cig, dict):
        return None
    target_context = cig.get("target_context")
    if not isinstance(target_context, dict) or not target_context:
        return None
    pocket_metadata = {
        str(key): value for key, value in target_context.items() if _is_pocket_context_key(str(key))
    }
    if not pocket_metadata:
        return None
    pocket_id = str(
        target_context.get("pocket_id")
        or target_context.get("pdb_id")
        or target_context.get("target_id")
        or run_id
    )
    record = {
        "kind": "pocket",
        "source": "orchestrator_svc",
        "run_id": run_id,
        "subject": {"type": "pocket", "id": pocket_id},
        "weight": 1.0,
        "polarity": "attract",
        "confidence": 1.0,
        "evidence_ids": [],
        "metadata": pocket_metadata,
    }
    payload = _pocket_encoder_payload(target_context)
    if payload is not None:
        feedback = await _encode_pocket_humu_feedback(payload)
        if feedback is not None:
            record["humu_embedding"] = feedback["humu_embedding"]
            record["curvature"] = feedback["curvature"]
            record["source"] = feedback["source"]
            record["evidence_ids"] = feedback["evidence_ids"]
    return record


def _intent_feedback_embedding(state: dict) -> tuple[list[float] | None, dict]:
    intent_cone = state.get("intent_cone")
    if not isinstance(intent_cone, dict):
        return None, {}
    embedding = _valid_hfm_feedback_embedding(intent_cone.get("axis"))
    if embedding is None:
        return None, {}
    return embedding, {"embedding_source": "intent_cone.axis"}


def _valid_hfm_feedback_embedding(value: object) -> list[float] | None:
    return normalize_lorentz_embedding(
        value,
        expected_dim=_CURRENT_HFM_LORENTZ_DIM,
        curvature=1.0,
    )


def _pocket_encoder_payload(target_context: dict) -> dict | None:
    coords = target_context.get("coords") or target_context.get("coordinates")
    elements = target_context.get("elements")
    residues = target_context.get("residue_types") or target_context.get("residues")
    if (
        not isinstance(coords, list)
        or not isinstance(elements, list)
        or not isinstance(residues, list)
    ):
        return None
    if len(coords) != len(elements) or len(coords) != len(residues) or not coords:
        return None
    return {
        "coords": coords,
        "elements": elements,
        "residue_types": residues,
    }


async def _encode_pocket_humu_feedback(payload: dict) -> dict | None:
    target = os.environ.get("HUMU_ENCODER_TARGET", "").strip()
    if not target:
        return None
    from mf_core.proto_gen.moleculeforge.v1.humu import encoder_pb2, encoder_pb2_grpc

    channel = grpc.aio.insecure_channel(target)
    try:
        stub = encoder_pb2_grpc.HUMUEncoderServiceStub(channel)
        response = await stub.Encode(
            encoder_pb2.EncodeRequest(
                entity_type="pocket",
                input_data=json.dumps(payload, sort_keys=True).encode("utf-8"),
            )
        )
    except Exception as exc:
        LOGGER.warning("Skipping pocket HUMU feedback enrichment: %s", exc)
        return None
    finally:
        await channel.close()
    try:
        embedding = _valid_hfm_feedback_embedding(
            _float32_embedding_from_bytes(response.humu_embedding)
        )
    except ValueError as exc:
        LOGGER.warning("Skipping pocket HUMU feedback enrichment: %s", exc)
        return None
    if embedding is None:
        return None
    return {
        "humu_embedding": embedding,
        "curvature": float(response.curvature),
        "source": "humu_encoder_svc",
        "evidence_ids": ["humu_encoder:pocket"],
    }


def _float32_embedding_from_bytes(payload: bytes) -> list[float]:
    if len(payload) % 4 != 0:
        raise ValueError("HUMU embedding bytes must contain float32 values")
    return [float(item[0]) for item in struct.iter_unpack("<f", payload)]


def _is_pocket_context_key(key: str) -> bool:
    lowered = key.lower()
    return "pocket" in lowered or lowered in {"pdb_id", "target_id", "binding_mode_prior"}


def _property_jmcg_feedback_from_generation_feedback(
    feedback: list,
    run_id: str,
    project_id: str = "",
) -> dict | None:
    records = []
    for index, item in enumerate(feedback):
        if not isinstance(item, dict):
            continue
        source = str(item.get("source") or "workflow")
        records.append(
            {
                "kind": "property",
                "source": source,
                "run_id": run_id,
                "subject": {
                    "type": "workflow_feedback",
                    "id": f"{source}-{index}",
                },
                "weight": float(item.get("weight", 1.0)),
                "polarity": _property_feedback_polarity(item),
                "confidence": float(item.get("confidence", 1.0)),
                "evidence_ids": _feedback_evidence_ids(item.get("evidence_ids")),
                "metadata": _property_feedback_metadata(item),
            }
        )
    if not records:
        return None
    return {
        "schema": "moleculeforge.jmcg.feedback.v1",
        "run_id": run_id,
        "project_id": project_id,
        "records": records,
    }


def _property_feedback_polarity(feedback: dict) -> str:
    explicit = str(feedback.get("polarity") or "")
    if explicit in {"attract", "repel"}:
        return explicit
    if feedback.get("passed") is False:
        return "repel"
    verdict = str(feedback.get("verdict") or "").lower()
    if verdict in {"fail", "failed", "reject"}:
        return "repel"
    return "attract"


def _property_feedback_metadata(feedback: dict) -> dict:
    excluded = {
        "source",
        "weight",
        "confidence",
        "polarity",
        "evidence_ids",
        "humu_embedding",
        "route_humu_embedding",
    }
    return {str(key): value for key, value in feedback.items() if key not in excluded}


def _feedback_evidence_ids(value: object) -> list[str]:
    if value in (None, "", b""):
        return []
    if isinstance(value, bytes):
        return [value.decode("utf-8")]
    if isinstance(value, str):
        return [value]
    if isinstance(value, list | tuple | set):
        return [str(item) for item in value]
    return [str(value)]


def _normalise_candidate_rows(candidates: list[dict]) -> list[dict]:
    rows = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise RuntimeError("generated candidates must be objects")
        row = dict(candidate)
        smiles = row.get("canonical_smiles") or row.get("smiles")
        if smiles:
            row["canonical_smiles"] = str(smiles)
        rows.append(row)
    return rows


def _engineering_candidate_rows(candidates: list[dict]) -> list[dict]:
    rows = _normalise_candidate_rows(candidates)
    for occurrence, row in enumerate(rows, start=1):
        row["candidate_id"] = f"candidate-{occurrence}"
    return rows


def _engineering_candidate_properties(predictor, smiles: str) -> dict:
    prediction = predictor.predict_one(smiles)
    row = prediction.to_dict()
    admet = dict(row.get("admet") or {})
    row.update(admet)
    molecular_weight = row.get("molecular_weight")
    if molecular_weight is not None:
        row["mw"] = molecular_weight
    row["ring_count"] = row.get("rings", 0) or 0
    row["fsp3"] = row.get("fraction_csp3", 0.0) or 0.0
    row["n_rotatable_bonds"] = row.get("rotatable_bonds", 0) or 0
    row["num_aromatic_rings"] = row.get("aromatic_rings", 0) or 0
    row["num_h_bond_donors"] = row.get("hbd", 0) or 0
    row["num_h_bond_acceptors"] = row.get("hba", 0) or 0
    row["logd"] = row.get("logd", row.get("logp", 0.0) or 0.0)
    row["log_s"] = row.get("solubility_logS", 0.0) or 0.0
    row["clearance"] = row.get("clearance_ml_min_kg", 0.0) or 0.0
    row["oral_bioavailability"] = (row.get("bioavailability_pct", 0.0) or 0.0) / 100.0
    row["ppb"] = (row.get("ppb_pct", 0.0) or 0.0) / 100.0
    row["caco2_papp"] = 10 ** float(row.get("caco2_logPapp", -10.0) or -10.0)
    return _normalise_engineering_critic_properties(row)


def _normalise_engineering_critic_properties(row: dict) -> dict:
    pains_alerts = row.get("pains_alerts", 0)
    if isinstance(pains_alerts, list):
        row["pains_alerts"] = len(pains_alerts)
    row["pains_alert_count"] = int(row.get("pains_alerts", 0) or 0)
    herg_risk = row.get("herg_risk", 0.0)
    if isinstance(herg_risk, str):
        row["herg_risk"] = {"low": 0.1, "medium": 0.5, "high": 0.9}.get(
            herg_risk.lower(),
            0.0,
        )
    return row


def _rank_engineering_results(rows: list[dict]) -> list[dict]:
    valid_rows = [
        (candidate_index, row) for candidate_index, row in enumerate(rows) if row.get("valid")
    ]
    if not valid_rows:
        return []
    objectives = [
        (
            row["qed"] or 0.0,
            -(row["sa_score"] or 10.0),
            -abs((row["logp"] or 5.0) - 2.5),
        )
        for _, row in valid_rows
    ]
    pareto_flags = [True] * len(valid_rows)
    for index, objective in enumerate(objectives):
        for other_index, other_objective in enumerate(objectives):
            if index == other_index:
                continue
            if all(
                other_objective[axis] >= objective[axis] for axis in range(len(objective))
            ) and any(other_objective[axis] > objective[axis] for axis in range(len(objective))):
                pareto_flags[index] = False
                break
    pareto_by_candidate_index = {
        candidate_index: pareto_flags[index]
        for index, (candidate_index, _) in enumerate(valid_rows)
    }
    ranked_rows = sorted(
        valid_rows,
        key=lambda item: -(item[1].get("composite_score") or 0.0),
    )
    return [
        {
            **row,
            "rank": rank,
            "pareto_optimal": bool(pareto_by_candidate_index[candidate_index]),
        }
        for rank, (candidate_index, row) in enumerate(ranked_rows, start=1)
    ]


def _full_workflow_critic_properties(state: dict, smiles: str) -> dict:
    properties = {}
    candidate = _candidate_row_for_smiles(state, smiles)
    if candidate:
        properties.update(_candidate_critic_properties(candidate, smiles))
    validation_rows = state.get("validation", {}).get("results", [])
    validation_row = _validation_row_for_smiles(validation_rows, smiles)
    if validation_row:
        properties.update(validation_row)
    properties.update(_srb_critic_properties(state))
    properties.update(_supply_critic_properties(state))
    properties.update(_request_critic_properties(state))
    properties["_critic_blocking_rule_ids"] = list(_FULL_WORKFLOW_BLOCKING_CRITIC_RULE_IDS)
    return _normalise_engineering_critic_properties(properties)


def _candidate_row_for_smiles(state: dict, smiles: str) -> dict:
    candidates = state.get("candidates")
    if not isinstance(candidates, list):
        return {}
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        candidate_smiles = str(candidate.get("canonical_smiles") or candidate.get("smiles") or "")
        if candidate_smiles == smiles:
            return dict(candidate)
    first = candidates[0] if candidates else {}
    return dict(first) if isinstance(first, dict) else {}


def _candidate_critic_properties(candidate: dict, smiles: str) -> dict:
    row = dict(candidate)
    if not _has_core_critic_properties(row):
        try:
            from mf_chem.predict.engine import MolPredictEngine

            enriched = _engineering_candidate_properties(MolPredictEngine(device_ids=[]), smiles)
            enriched.update(row)
            row = enriched
        except Exception as exc:
            LOGGER.warning("Skipping full workflow critic property enrichment: %s", exc)
    return row


def _has_core_critic_properties(row: dict) -> bool:
    return all(key in row for key in ("mw", "logp", "tpsa", "qed", "sa_score"))


def _validation_row_for_smiles(validation_rows: object, smiles: str) -> dict:
    rows = (
        [row for row in validation_rows if isinstance(row, dict)]
        if isinstance(validation_rows, list)
        else []
    )
    if not rows:
        return {}
    for row in rows:
        row_smiles = str(row.get("canonical_smiles") or row.get("smiles") or "")
        if row_smiles == smiles:
            return dict(row)
    return dict(_best_engineering_validation_row(rows))


def _supply_critic_properties(state: dict) -> dict:
    supply = state.get("supply")
    if not isinstance(supply, dict):
        return {}
    assessment = supply.get("supply_assessment")
    if not isinstance(assessment, dict):
        return {}
    total_blocks = int(assessment.get("total_blocks") or 0)
    available_blocks = int(assessment.get("commercially_available") or 0)
    properties = {
        "critical_material_suppliers": int(assessment.get("supplier_diversity") or 0),
        "estimated_cost_per_gram": float(assessment.get("avg_price_per_gram") or 0.0),
    }
    if total_blocks > 0:
        properties["building_block_availability"] = available_blocks / total_blocks
    return properties


def _srb_critic_properties(state: dict) -> dict:
    srb = state.get("srb")
    if not isinstance(srb, dict):
        return {}
    protocols = srb.get("protocols")
    if not isinstance(protocols, list) or not protocols:
        return {}
    protocol = protocols[0]
    if not isinstance(protocol, dict):
        return {}
    steps = protocol.get("steps")
    properties = {
        "estimated_cost_per_gram": float(protocol.get("total_estimated_cost_usd") or 0.0),
    }
    if isinstance(steps, list):
        properties["synthesis_steps"] = len(steps)
    return properties


def _request_critic_properties(state: dict) -> dict:
    request = state.get("request")
    if not isinstance(request, dict):
        return {}
    properties = {}
    for key in ("isoform_data_count", "kinase_selectivity_ratio", "cns_mpo", "bbb_score"):
        if key in request:
            properties[key] = request[key]
    return properties


def _best_engineering_validation_row(validation_rows: list) -> dict:
    rows = [row for row in validation_rows if isinstance(row, dict)]
    if not rows:
        return {}
    return max(
        rows,
        key=lambda row: (
            float(row.get("composite_score") or 0.0),
            float(row.get("admet_score") or 0.0),
        ),
    )


def _best_engineering_candidate_smiles(state: dict) -> str:
    rows = state.get("validation", {}).get("results", [])
    best = _best_engineering_validation_row(rows if isinstance(rows, list) else [])
    smiles = str(best.get("canonical_smiles") or best.get("smiles") or "")
    if smiles:
        return smiles
    return _first_candidate_smiles(state)


async def _merge_agent_beliefs_into_crg(final_state: dict, run_id: str) -> dict:
    from mf_core.db.repositories import build_shared_crg_repository_from_env

    crg = dict(final_state.get("crg") or {})
    if not run_id:
        return crg
    try:
        repo = build_shared_crg_repository_from_env()
    except Exception as exc:
        LOGGER.debug("Skipping CRG belief merge (repository unavailable): %s", exc)
        return crg
    if repo is None:
        return crg
    try:
        shared_crg = await repo.get_run_crg(run_id)
    except Exception as exc:
        LOGGER.warning("Failed to read shared CRG for run %s: %s", run_id, exc)
        return crg
    shared_beliefs = list(shared_crg.get("beliefs") or [])
    shared_edges = list(shared_crg.get("edges") or [])
    if not shared_beliefs and not shared_edges:
        return crg
    existing_ids = {
        str(b.get("id") or "") for b in (crg.get("beliefs") or []) if isinstance(b, dict)
    }
    merged_beliefs = list(crg.get("beliefs") or [])
    for belief in shared_beliefs:
        if not isinstance(belief, dict):
            continue
        if str(belief.get("id") or "") not in existing_ids:
            merged_beliefs.append(belief)
    existing_edge_keys = {
        (str(e.get("source_belief_id") or ""), str(e.get("target_belief_id") or ""))
        for e in (crg.get("edges") or [])
        if isinstance(e, dict)
    }
    merged_edges = list(crg.get("edges") or [])
    for edge in shared_edges:
        if not isinstance(edge, dict):
            continue
        key = (str(edge.get("source_belief_id") or ""), str(edge.get("target_belief_id") or ""))
        if key not in existing_edge_keys:
            merged_edges.append(edge)
    merged = dict(crg)
    merged["beliefs"] = merged_beliefs
    merged["edges"] = merged_edges
    merged["version"] = len(merged_beliefs) + len(merged_edges)
    return merged


async def _record_workflow_provenance(final_state: dict) -> None:
    from provenance_svc.main import ProvenanceRecord, create_record

    run_id = str(final_state.get("run_id", ""))
    artifact_id = f"artifact-{_safe_id(run_id)}-workflow-state"
    crg = await _merge_agent_beliefs_into_crg(final_state, run_id)
    if crg:
        crg["provenance_id"] = artifact_id
        final_state["crg"] = crg
    supply = final_state.get("supply") if isinstance(final_state.get("supply"), dict) else {}
    supply_assessment = (
        supply.get("supply_assessment") if isinstance(supply.get("supply_assessment"), dict) else {}
    )
    srb = final_state.get("srb") if isinstance(final_state.get("srb"), dict) else {}
    metadata = {
        "project_id": str(final_state.get("request", {}).get("project_id") or run_id),
        "run_id": run_id,
        "trace_id": str(final_state.get("trace_id", "")),
        "workflow_scope": str(final_state.get("workflow_scope", "")),
        "status": str(final_state.get("status", "")),
        "history": list(final_state.get("history", [])),
        "candidate_count": len(final_state.get("candidates", []) or []),
        "validation_passed": bool(final_state.get("validation_passed", False)),
        "retrosyn_route_count": len(final_state.get("retrosyn", {}).get("routes", []) or []),
        "supply_feasibility": str(supply_assessment.get("overall_feasibility", "")),
        "srb_protocol_count": len(srb.get("protocols", []) or []),
        "critic_verdict": str(final_state.get("critic", {}).get("verdict", "")),
        "crg": crg,
        "crg_belief_count": len(crg.get("beliefs", []) or []),
        "crg_edge_count": len(crg.get("edges", []) or []),
    }
    parent_ids = list(final_state.get("artifact_ids", []))
    record = await create_record(
        ProvenanceRecord(
            artifact_type="workflow_state",
            artifact_id=artifact_id,
            parent_ids=parent_ids,
            metadata=metadata,
        )
    )
    artifact_ids = list(final_state.get("artifact_ids", []))
    if artifact_id not in artifact_ids:
        artifact_ids.append(artifact_id)
    final_state["artifact_ids"] = artifact_ids
    final_state["provenance"] = {
        "recorded": True,
        "artifact_id": record["artifact_id"],
        "signature": record["signature"],
        "recorded_at": record.get("recorded_at", ""),
    }


def _requested_oracle_level(request: dict) -> int | None:
    for key in ("oracle_level", "max_oracle_level", "validation_oracle_level"):
        value = request.get(key)
        if value not in (None, ""):
            return int(value)
    return None


async def _validate_with_oracle_cascade(
    request_client: AgentRequestClient | None,
    state: dict,
    candidates: list[dict],
    oracle_level: int | None,
) -> dict:
    request = dict(state.get("request", {}) or {})
    request.pop("clients", None)
    for key in ("oracle_level", "max_oracle_level", "validation_oracle_level"):
        request.pop(key, None)
    default_oracle_level = 0 if oracle_level is None else oracle_level
    rows = []
    for candidate_index, candidate in enumerate(candidates):
        smiles = _candidate_smiles(candidate, purpose="validation")
        payload = dict(request)
        payload["project_id"] = str(request.get("project_id") or "")
        payload["run_id"] = str(state.get("run_id", ""))
        payload["smiles"] = smiles
        if oracle_level is not None:
            payload["oracle_level"] = oracle_level
        result = await _request_agent(
            request_client,
            state,
            "validation",
            "validating",
            payload,
            candidate_index=candidate_index,
        )
        status = str(result.get("status") or "")
        overall_passed = bool(result.get("overall_passed", status == "validated"))
        rows.append(
            {
                "smiles": smiles,
                "status": status,
                "overall_passed": overall_passed,
                "max_oracle_level": result.get(
                    "max_oracle_level",
                    default_oracle_level,
                ),
                "cascade": dict(result.get("cascade") or {}),
                "upgrade_path": list(result.get("upgrade_path") or []),
            }
        )
    return {
        "passed": any(bool(row.get("overall_passed")) for row in rows),
        "results": rows,
        "validation_mode": "adaptive_oracle_cascade",
        "oracle_level": default_oracle_level,
    }


def _candidate_smiles(candidate: dict, purpose: str) -> str:
    if not isinstance(candidate, dict):
        raise RuntimeError("candidate entries must be objects")
    smiles = str(candidate.get("canonical_smiles") or candidate.get("smiles") or "")
    if not smiles:
        raise RuntimeError(f"candidate canonical_smiles is required for full workflow {purpose}")
    return smiles


def _first_candidate_smiles(state: dict) -> str:
    candidates = list(state.get("candidates", []) or [])
    if not candidates:
        raise RuntimeError("candidates are required for full workflow synthesis")
    return _candidate_smiles(candidates[0], purpose="synthesis")


def _supply_feasibility(state: dict) -> str:
    supply = state.get("supply")
    if not isinstance(supply, dict):
        return ""
    assessment = supply.get("supply_assessment")
    if not isinstance(assessment, dict):
        return ""
    return str(assessment.get("overall_feasibility") or "").lower()


def _first_retrosyn_route(state: dict) -> dict:
    route = _first_retrosyn_route_or_none(state)
    if route is None:
        raise RuntimeError("retrosyn.routes is required for full workflow synthesis")
    return route


def _first_retrosyn_route_or_none(state: dict) -> dict | None:
    retrosyn = state.get("retrosyn")
    if not isinstance(retrosyn, dict):
        return None
    routes = retrosyn.get("routes")
    if not isinstance(routes, list) or not routes:
        return None
    route = routes[0]
    if not isinstance(route, dict):
        raise RuntimeError("retrosyn route entries must be objects")
    return route


def _unavailable_supply_result(state: dict, reason: str) -> dict:
    return {
        "agent": "supply_agent",
        "status": "assessed",
        "smiles": _first_candidate_smiles(state),
        "skip_reason": reason,
        "supply_assessment": {
            "total_blocks": 0,
            "commercially_available": 0,
            "avg_price_per_gram": 0.0,
            "avg_lead_time_days": 0.0,
            "supplier_diversity": 0,
            "overall_feasibility": "unavailable",
        },
        "block_assessments": [],
    }


def _route_building_blocks(route: dict) -> list[dict]:
    blocks = route.get("building_blocks")
    if isinstance(blocks, list) and blocks:
        return list(blocks)
    extracted = []
    seen = set()
    for step in route.get("steps", []) or []:
        if not isinstance(step, dict):
            continue
        for key in ("building_blocks", "reactants"):
            values = step.get(key)
            if not isinstance(values, list):
                continue
            for value in values:
                smiles = _block_smiles(value)
                if smiles and smiles not in seen:
                    seen.add(smiles)
                    extracted.append({"smiles": smiles})
    if not extracted:
        raise RuntimeError("retrosyn route building_blocks are required for supply assessment")
    return extracted


def _block_smiles(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return str(value.get("smiles") or value.get("building_block_smiles") or "")
    return ""


def _safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-") or "run"


async def _start_grpc_server():
    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=10))
    register_grpc_services(server)
    server.add_insecure_port("[::]:50071")
    await server.start()
    LOGGER.info("Orchestrator gRPC Service running on :50071")
    return server


async def serve_grpc():
    await _orchestrator_startup()
    server = None
    try:
        server = await _start_grpc_server()
        await server.wait_for_termination()
    finally:
        if server is not None:
            await server.stop(30.0)
            await server.wait_for_termination()
        await _orchestrator_shutdown()


async def _serve_process(rest_server) -> None:
    await _orchestrator_startup()
    grpc_server = None
    try:
        grpc_server = await _start_grpc_server()
        await rest_server.serve()
    finally:
        if grpc_server is not None:
            await grpc_server.stop(30.0)
            await grpc_server.wait_for_termination()
        await _orchestrator_shutdown()


def register_grpc_services(server) -> None:
    orchestrator_pb2_grpc.add_OrchestratorServiceServicer_to_server(
        OrchestratorGrpcServicer(),
        server,
    )


if __name__ == "__main__":
    import uvicorn

    async def main():
        config = uvicorn.Config(
            rest_app,
            host="0.0.0.0",
            port=8011,
            log_level="info",
            lifespan="off",
        )
        server = uvicorn.Server(config)
        await _serve_process(server)

    asyncio.run(main())
