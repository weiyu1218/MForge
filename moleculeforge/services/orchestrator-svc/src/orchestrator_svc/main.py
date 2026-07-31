"""Asynchronous full-workflow orchestration service."""

from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import hmac
import json
import logging
import math
import os
import re
import struct
import uuid
from collections.abc import Awaitable, Callable, Mapping
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any, TypeVar
from urllib.parse import quote

import grpc
import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from mf_agents.base.agent import (
    AGENT_PROTOCOLS,
    BaseAgent,
    agent_health_check_timeout_seconds,
)
from mf_agents.messaging.redis_bus import RedisBus
from mf_agents.messaging.request_client import AgentRequestClient
from mf_core.db.repositories import build_shared_crg_repository_from_env
from mf_core.db.store import RunAlreadyExistsError, RunStatus, RunStore, db_path
from mf_core.geometry.lorentz import normalize_lorentz_embedding
from orchestrator.workflow.graph_builder import (
    WorkflowGraph,
    agent_request_timeout_seconds,
    create_initial_state,
    critic_feedback_groups,
    full_workflow_candidate_identity,
    full_workflow_critic_properties,
    generation_controls,
    require_feedback_acknowledgement,
    require_full_downstream_response_identity,
    require_validation_batch_contract,
    select_full_candidate,
    validate_full_workflow_policies,
    validation_candidate_payload,
    validation_feedback_groups,
)

rest_app = FastAPI(title="Orchestrator Service", version="0.1.0")
_RUN_STORE: RunStore | None = None
_RUN_CONTROL: RunControl | None = None
_RUN_INITIALIZED_STORE: RunStore | None = None
_RUNTIME_INIT_LOCK: asyncio.Lock | None = None
_RUNTIME_INIT_LOOP: asyncio.AbstractEventLoop | None = None
_RUN_TASKS: dict[str, asyncio.Task[Any]] = {}
_AGENT_BUS: RedisBus | None = None
_AGENT_REQUEST_CLIENT: AgentRequestClient | None = None
_SERVICE_TOKEN_HEADER = "X-MoleculeForge-Service-Token"
_PRINCIPAL_HEADER = "X-MoleculeForge-Principal"
_SERVICE_PRINCIPAL: ContextVar[str | None] = ContextVar(
    "orchestrator_service_principal",
    default=None,
)
_AGENT_RUNTIME_LOOP: asyncio.AbstractEventLoop | None = None
_AGENT_INIT_LOCK: asyncio.Lock | None = None
_AGENT_INIT_LOOP: asyncio.AbstractEventLoop | None = None
_AGENT_SHUTDOWN_COUNT = 0
LOGGER = logging.getLogger(__name__)
T = TypeVar("T")
_ROUTE_IDENTIFIER_PATTERN = re.compile(r"[A-Za-z0-9._~!$&'()*+,;=:@-]+")
_CURRENT_HFM_LORENTZ_DIM = 129
_DEFAULT_EXTERNAL_EVIDENCE_MAX_ARTIFACTS = 256
_EXTERNAL_EVIDENCE_FETCH_BATCH_SIZE = 8
_AGENT_PROTOCOLS_BY_ENTRY_POINT = {protocol.entry_point: protocol for protocol in AGENT_PROTOCOLS}
_NL2OBJ_SUBJECT = "agent.nl2obj.request"
_NL2OBJ_PAYLOAD_TYPE_URL = "type.moleculeforge.ai/agent/nl2obj/request.v1"
_NL2OBJ_SCHEMA_VERSION = "nl2obj.request.v1"
_NONTERMINAL_RUN_STATUSES = frozenset(
    {
        RunStatus.QUEUED,
        RunStatus.RUNNING,
        RunStatus.PAUSED,
        RunStatus.AWAITING_EVIDENCE,
    }
)
_NONTERMINAL_RUN_STATUS_VALUES = frozenset(status.value for status in _NONTERMINAL_RUN_STATUSES)


def _internal_service_token() -> str:
    return os.environ.get("INTERNAL_SERVICE_TOKEN", "").strip()


def _current_service_principal() -> str | None:
    return _SERVICE_PRINCIPAL.get()


@rest_app.middleware("http")
async def _authenticate_internal_request(
    request: Request,
    call_next: Callable[[Request], Awaitable[Any]],
):
    expected_token = _internal_service_token()
    if request.url.path == "/health" or not expected_token:
        return await call_next(request)
    supplied_token = request.headers.get(_SERVICE_TOKEN_HEADER, "")
    if not supplied_token or not hmac.compare_digest(
        supplied_token.encode("utf-8"),
        expected_token.encode("utf-8"),
    ):
        return JSONResponse(status_code=401, content={"detail": "Invalid service token"})
    principal = request.headers.get(_PRINCIPAL_HEADER)
    normalized_principal = principal.strip() if principal is not None else None
    if normalized_principal == "":
        return JSONResponse(
            status_code=401,
            content={"detail": "Authenticated principal is required"},
        )
    context_token = _SERVICE_PRINCIPAL.set(normalized_principal)
    try:
        return await call_next(request)
    finally:
        _SERVICE_PRINCIPAL.reset(context_token)


def _run_is_full_workflow(snapshot: Mapping[str, object]) -> bool:
    state = snapshot.get("state")
    return isinstance(state, Mapping) and state.get("workflow_scope") == "full"


def _authorize_run_snapshot(snapshot: Mapping[str, object]) -> None:
    if not _run_is_full_workflow(snapshot):
        raise HTTPException(status_code=404, detail="Run not found")
    if not _internal_service_token():
        return
    principal = _current_service_principal()
    owner = snapshot.get("owner_principal_id")
    if not isinstance(owner, str) or not owner:
        raise HTTPException(
            status_code=403,
            detail="Full workflow is not bound to an authenticated owner",
        )
    if principal != owner:
        raise HTTPException(
            status_code=403,
            detail="Run owner does not match authenticated principal",
        )


def _public_run_snapshot(snapshot: Mapping[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in snapshot.items()
        if key != "owner_principal_id"
    }


class _RunControlState:
    def __init__(self) -> None:
        self.pause_requested = False
        self.paused = asyncio.Event()
        self.resume_requested = asyncio.Event()
        self.resumed = asyncio.Event()
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
                for task in _RUN_TASKS.values()
                if task is not current_task
            )
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            _RUN_TASKS.clear()
            await _agent_control_shutdown()
    finally:
        _AGENT_SHUTDOWN_COUNT -= 1


rest_app.add_event_handler("shutdown", _orchestrator_shutdown)


def _validated_policy(request: dict) -> dict[str, object]:
    workflow_scope = request.get("workflow_scope", "full")
    if workflow_scope != "full":
        raise HTTPException(status_code=400, detail="workflow_scope must be full")
    if "max_refinements" not in request:
        raise HTTPException(status_code=400, detail="max_refinements is required")
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
    policy = {
        "workflow_scope": "full",
        "validation_passed": False,
        "max_refinements": max_refinements,
    }
    policy.update(_validated_full_workflow_policies(request))
    return policy


def _validated_full_workflow_policies(
    request: Mapping[str, object],
) -> dict[str, object]:
    try:
        return validate_full_workflow_policies(request)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _validated_caller_run_id(value: object) -> str | None:
    if value is None or value == "":
        return None
    if (
        not isinstance(value, str)
        or value in {".", ".."}
        or _ROUTE_IDENTIFIER_PATTERN.fullmatch(value) is None
    ):
        raise HTTPException(
            status_code=400,
            detail="run_id must be a URL-safe single path segment",
        )
    return value


def _validate_project_id(project_id: str) -> None:
    if any(segment in {".", ".."} for segment in project_id.split("/")):
        raise HTTPException(
            status_code=400,
            detail="project_id must not contain dot path segments",
        )


def _validated_run_project_id(request: Mapping[str, object]) -> str | None:
    if "project_id" not in request:
        return None
    project_id = request["project_id"]
    if not isinstance(project_id, str) or not project_id.strip():
        raise HTTPException(
            status_code=400,
            detail="project_id must be a non-empty string",
        )
    return project_id


@rest_app.get("/health")
async def health():
    return {"status": "healthy", "engine": "langgraph", "runs": len(_RUN_TASKS)}


def _register_design_run_task(
    run_id: str,
    request: dict,
    initial_state: dict,
) -> asyncio.Task[None]:
    task = asyncio.create_task(
        _execute_design_run(run_id, request, initial_state),
        name=f"orchestrator-run-{run_id}",
    )
    _RUN_TASKS[run_id] = task
    task.add_done_callback(lambda completed, key=run_id: _finish_run_task(key, completed))
    return task


def _register_evidence_resume_task(
    run_id: str,
    request: dict,
    state: dict,
) -> asyncio.Task[None]:
    task = asyncio.create_task(
        _execute_evidence_resume_run(run_id, request, state),
        name=f"orchestrator-evidence-resume-{run_id}",
    )
    _RUN_TASKS[run_id] = task
    task.add_done_callback(lambda completed, key=run_id: _finish_run_task(key, completed))
    return task


@rest_app.post("/v1/orchestrator/design", status_code=202)
async def create_design_run(request: dict) -> dict:
    request = dict(request)
    request.pop("clients", None)
    nl_input = request.get("nl_input") or request.get("intent")
    if not nl_input:
        raise HTTPException(status_code=400, detail="nl_input is required")
    policy = _validated_policy(request)
    owner_principal_id = None
    if _internal_service_token():
        owner_principal_id = _current_service_principal()
        if owner_principal_id is None:
            raise HTTPException(
                status_code=403,
                detail="Authenticated principal is required for full workflows",
            )
    request["workflow_scope"] = "full"
    request["validation_passed"] = False
    for field in ("validation_policy", "teacher_policy", "selection_policy"):
        request[field] = policy[field]
    project_id = _validated_run_project_id(request)
    run_store, _ = await _runtime()
    run_id = _validated_caller_run_id(request.get("run_id")) or f"run-{uuid.uuid4().hex}"
    created_at = datetime.now(UTC).isoformat()
    trace_id = str(request.get("trace_id") or f"trace-{uuid.uuid4().hex}")
    initial_state = create_initial_state(
        str(nl_input),
        run_id=run_id,
        trace_id=trace_id,
        artifact_ids=request.get("artifact_ids") or [],
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
            project_id=project_id,
            owner_principal_id=owner_principal_id,
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
            _register_design_run_task(run_id, dict(request), initial_state)
        raise
    except RunAlreadyExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    _register_design_run_task(run_id, dict(request), initial_state)
    queued_snapshot = await run_store.get_run(run_id)
    if queued_snapshot is None:
        raise RuntimeError(f"run was not persisted: {run_id}")
    return {"design_id": run_id, "run_id": run_id, "status": RunStatus.QUEUED.value}


async def _execute_design_run(
    run_id: str,
    request: dict,
    state: dict,
) -> None:
    run_store, run_control = await _runtime()
    try:
        state["started_at"] = datetime.now(UTC).isoformat()
        await run_store.transition_run(
            run_id,
            {RunStatus.QUEUED},
            RunStatus.RUNNING,
            current_stage="planning",
            state=state,
        )
        final_state = await _invoke_workflow(request, state, run_control=run_control)
        status = _workflow_terminal_status(final_state)
        if str(final_state.get("status")) != "AWAITING_EVIDENCE":
            await _record_workflow_provenance(final_state)
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


async def _execute_evidence_resume_run(
    run_id: str,
    request: dict,
    state: dict,
) -> None:
    run_store, run_control = await _runtime()
    try:
        final_state = await _invoke_workflow(
            request,
            state,
            run_control=run_control,
            entry_point="validating",
        )
        status = _workflow_terminal_status(final_state)
        if str(final_state.get("status")) != "AWAITING_EVIDENCE":
            await _record_workflow_provenance(final_state)
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
                    current_stage=str(snapshot.get("current_stage") or "validating"),
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
                LOGGER.exception("run %s failed while recording resume failure", run_id)
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
    _synchronize_terminal_state(final_state)
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


def _synchronize_terminal_state(final_state: dict) -> None:
    cig = final_state.get("cig")
    cig = cig if isinstance(cig, dict) else {}
    request = final_state.get("request")
    request = request if isinstance(request, dict) else {}

    objectives = final_state.get("objectives")
    if not isinstance(objectives, (dict, list)) or not objectives:
        objectives = cig.get("objective_nodes")
    if not isinstance(objectives, (dict, list)) or not objectives:
        objectives = cig.get("objectives")
    if not isinstance(objectives, (dict, list)) or not objectives:
        objectives = request.get("objectives")
    if isinstance(objectives, (dict, list)):
        final_state["objectives"] = objectives

    metadata = cig.get("metadata")
    intent_summary = metadata.get("intent_summary") if isinstance(metadata, dict) else None
    if not isinstance(intent_summary, str) or not intent_summary.strip():
        intent_summary = cig.get("source_user_input")
    if isinstance(intent_summary, str) and intent_summary.strip():
        final_state["summary"] = intent_summary

    devices_used: list[str] = []

    def append_devices(value: object) -> None:
        if not isinstance(value, list):
            return
        for device in value:
            if isinstance(device, str) and device and device not in devices_used:
                devices_used.append(device)

    append_devices(final_state.get("devices_used"))
    candidates = final_state.get("candidates")
    if isinstance(candidates, list):
        for candidate in candidates:
            if isinstance(candidate, dict):
                append_devices(candidate.get("devices_used"))
    validation = final_state.get("validation")
    if isinstance(validation, dict):
        append_devices(validation.get("devices_used"))
        validation_rows = validation.get("results")
        if isinstance(validation_rows, list):
            for row in validation_rows:
                if isinstance(row, dict):
                    append_devices(row.get("devices_used"))
    if devices_used:
        final_state["devices_used"] = devices_used


def _persistable_state(state: dict) -> dict:
    persisted = dict(state)
    request = persisted.get("request")
    if isinstance(request, dict):
        persisted_request = dict(request)
        persisted_request.pop("clients", None)
        persisted["request"] = persisted_request
    return persisted


def _workflow_terminal_status(final_state: dict) -> RunStatus:
    current_stage = str(final_state.get("status") or "")
    if current_stage == "ERROR":
        return RunStatus.FAILED
    if current_stage == "AWAITING_EVIDENCE":
        return RunStatus.AWAITING_EVIDENCE
    if current_stage == "ESCALATING":
        return RunStatus.REJECTED
    if current_stage == "EXECUTING":
        return RunStatus.COMPLETED
    raise RuntimeError(f"WorkflowGraph returned non-terminal stage: {current_stage or '<empty>'}")


def _finish_run_task(run_id: str, task: asyncio.Task[Any]) -> None:
    if _RUN_TASKS.get(run_id) is task:
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
    entry_point: str = "planning",
) -> dict:
    business_request = dict(request)
    injected_clients = business_request.pop("clients", None)
    if clients is None:
        clients = injected_clients
    if business_request.get("workflow_scope", "full") != "full":
        raise ValueError("workflow_scope must be full")
    business_request["workflow_scope"] = "full"
    state["request"] = business_request
    state["validation_passed"] = False
    state["max_refinements"] = int(business_request["max_refinements"])
    if clients is None:
        clients = _default_workflow_clients(_shared_agent_request_client())
    workflow_graph = WorkflowGraph(clients=clients)
    compiled = (
        workflow_graph.build()
        if entry_point == "planning"
        else workflow_graph.build(entry_point=entry_point)
    )
    if run_control is None or not hasattr(compiled, "astream"):
        return await compiled.ainvoke(state)
    return await _stream_workflow_stages(
        compiled,
        state,
        run_control,
        initial_stage=entry_point,
    )


async def _stream_workflow_stages(
    compiled: object,
    state: dict,
    run_control: RunControl,
    *,
    initial_stage: str = "planning",
) -> dict:
    run_id = str(state["run_id"])
    final_state = state
    persisted_steps = {
        int(event["step_index"])
        for event in await run_control.store.list_events(run_id)
    }
    await run_control.wait_if_paused(run_id, initial_stage)
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


def _default_workflow_clients(
    request_client: AgentRequestClient,
) -> FullWorkflowClients:
    return FullWorkflowClients(request_client=request_client)


@rest_app.post("/v1/orchestrator/projects")
async def create_project_record(request: dict) -> dict:
    name = request.get("name")
    description = request.get("description", "")
    if not isinstance(name, str) or not name:
        raise HTTPException(status_code=400, detail="name is required")
    if not isinstance(description, str):
        raise HTTPException(status_code=400, detail="description must be a string")
    _validate_project_id(name)
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
    visible_runs = []
    for snapshot in page["items"]:
        if not _run_is_full_workflow(snapshot):
            continue
        try:
            _authorize_run_snapshot(snapshot)
        except HTTPException:
            continue
        visible_runs.append(_public_run_snapshot(snapshot))
    return {
        "runs": visible_runs,
        "next_page_token": page["next_page_token"],
    }


@rest_app.get("/v1/orchestrator/runs/{run_id}")
async def get_run_snapshot(run_id: str) -> dict:
    run_store, _ = await _runtime()
    snapshot = await run_store.get_run(run_id)
    if snapshot is None:
        raise HTTPException(status_code=404, detail=f"Unknown run_id: {run_id}")
    _authorize_run_snapshot(snapshot)
    return _public_run_snapshot(snapshot)


@rest_app.get("/v1/orchestrator/runs/{run_id}/events")
async def get_run_events(run_id: str, after_step: int = -1) -> dict:
    run_store, _ = await _runtime()
    snapshot = await run_store.get_run(run_id)
    if snapshot is None:
        raise HTTPException(status_code=404, detail=f"Unknown run_id: {run_id}")
    _authorize_run_snapshot(snapshot)
    try:
        events = await run_store.list_events(run_id, after_step=after_step)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"run_id": run_id, "events": events}


@rest_app.post("/v1/orchestrator/runs/{run_id}/pause")
async def pause_run(run_id: str) -> dict:
    run_store, run_control = await _runtime()
    snapshot = await run_store.get_run(run_id)
    if snapshot is None:
        raise HTTPException(status_code=404, detail=f"Unknown run_id: {run_id}")
    _authorize_run_snapshot(snapshot)
    try:
        await run_control.pause(run_id)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"design_id": run_id, "run_id": run_id, "status": RunStatus.PAUSED.value}


@rest_app.post("/v1/orchestrator/runs/{run_id}/resume")
async def resume_run(run_id: str) -> dict:
    run_store, run_control = await _runtime()
    snapshot = await run_store.get_run(run_id)
    if snapshot is None:
        raise HTTPException(status_code=404, detail=f"Unknown run_id: {run_id}")
    _authorize_run_snapshot(snapshot)
    try:
        await run_control.resume(run_id)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"design_id": run_id, "run_id": run_id, "status": RunStatus.RUNNING.value}


@rest_app.post("/v1/orchestrator/runs/{run_id}/evidence/resume", status_code=202)
async def resume_evidence_run(
    run_id: str,
    request: dict | None = None,
) -> dict:
    run_store, run_control = await _runtime()
    snapshot = await run_store.get_run(run_id)
    if snapshot is None:
        raise HTTPException(status_code=404, detail=f"Unknown run_id: {run_id}")
    _authorize_run_snapshot(snapshot)
    state = snapshot.get("state")
    if snapshot["status"] != RunStatus.AWAITING_EVIDENCE.value:
        raise HTTPException(
            status_code=409,
            detail=(
                f"run {run_id} cannot resume evidence from status "
                f"{snapshot['status']}"
            ),
        )
    try:
        resumed_request, resumed_state = _prepare_full_evidence_resume(
            request,
            state,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    try:
        evidence_artifacts = await _verify_resume_external_evidence(
            resumed_request["external_evidence"],
            resumed_state,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    resumed_state["external_evidence_artifacts"] = evidence_artifacts
    resumed_state["external_evidence_resume_verified"] = True
    submissions = list(state.get("external_evidence_submissions", []) or [])
    submissions.append(
        {
            "submitted_at": datetime.now(UTC).isoformat(),
            "submitted_by": _current_service_principal(),
            "artifacts": evidence_artifacts,
        }
    )
    resumed_state["external_evidence_submissions"] = submissions
    active_task = _RUN_TASKS.get(run_id)
    if active_task is not None:
        if active_task.done():
            _RUN_TASKS.pop(run_id, None)
        else:
            raise HTTPException(
                status_code=409,
                detail=f"run {run_id} is still finalizing its evidence checkpoint",
            )
    run_control.forget(run_id)
    transition_task = asyncio.create_task(
        run_store.transition_run(
            run_id,
            {RunStatus.AWAITING_EVIDENCE},
            RunStatus.RUNNING,
            current_stage="validating",
            state=_persistable_state(resumed_state),
        )
    )
    try:
        await asyncio.shield(transition_task)
    except asyncio.CancelledError:
        try:
            await _await_task_completion(transition_task)
        except (asyncio.CancelledError, Exception):
            transition_owned = False
        else:
            transition_owned = True
        if transition_owned:
            _register_evidence_resume_task(
                run_id,
                resumed_request,
                resumed_state,
            )
        raise
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    try:
        _register_evidence_resume_task(
            run_id,
            resumed_request,
            resumed_state,
        )
    except Exception as exc:
        await run_store.transition_run(
            run_id,
            {RunStatus.RUNNING},
            RunStatus.FAILED,
            current_stage="validating",
            error_type=type(exc).__name__,
            error_message=str(exc),
        )
        raise
    return {"design_id": run_id, "run_id": run_id, "status": RunStatus.RUNNING.value}


def _prepare_full_evidence_resume(
    payload: object,
    state: Mapping[str, object],
) -> tuple[dict, dict]:
    if not isinstance(payload, Mapping):
        raise ValueError("request body must be an object")
    if set(payload) != {"external_evidence"}:
        raise ValueError("request body must contain only external_evidence")
    candidates = state.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("persisted full workflow candidates are required")
    normalized_evidence = _validate_resume_external_evidence(
        payload["external_evidence"],
        candidates,
        state,
    )
    persisted_request = state.get("request")
    if not isinstance(persisted_request, Mapping):
        raise ValueError("persisted full workflow request is required")
    business_request = dict(persisted_request)
    if str(business_request.get("workflow_scope")) != "full":
        raise ValueError("persisted request workflow_scope must be full")
    persisted_evidence = business_request.get("external_evidence")
    if persisted_evidence is None:
        existing_evidence: list[dict] = []
    else:
        existing_evidence = _validate_resume_external_evidence(
            persisted_evidence,
            candidates,
            state,
        )
    evidence_by_candidate = {
        item["candidate_id"]: item for item in existing_evidence
    }
    merged_evidence = list(existing_evidence)
    for item in normalized_evidence:
        existing = evidence_by_candidate.get(item["candidate_id"])
        if existing is None:
            evidence_by_candidate[item["candidate_id"]] = item
            merged_evidence.append(item)
            continue
        if existing != item:
            raise ValueError(
                "external evidence was already accepted with different content "
                f"for {item['candidate_id']}"
            )
    business_request["external_evidence"] = merged_evidence
    resumed_state = dict(state)
    resumed_state["request"] = business_request
    _validated_evidence_resume_records(
        resumed_state,
        business_request,
        candidates,
    )
    for field in (
        "retrosyn",
        "supply",
        "srb",
        "critic",
        "critic_feedback",
        "provenance",
    ):
        resumed_state.pop(field, None)
    resumed_state["validation_passed"] = False
    return business_request, resumed_state


def _validated_evidence_resume_records(
    state: Mapping[str, object],
    request: Mapping[str, object],
    candidates: list[Mapping[str, object]],
) -> list[dict]:
    prior_validation = state.get("validation")
    validation_policy = request.get("validation_policy")
    outer_request_id = request.get("request_id")
    run_id = state.get("run_id")
    if not isinstance(prior_validation, Mapping):
        raise ValueError("persisted AWAITING_EVIDENCE validation is required")
    if not isinstance(validation_policy, Mapping):
        raise ValueError("persisted validation_policy is required")
    if not isinstance(outer_request_id, str) or not outer_request_id:
        raise ValueError("persisted request_id is required")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("persisted run_id is required")
    refinement_count = int(state.get("refinement_count", 0))
    validation_request_id = f"{run_id}:validation:{refinement_count}"
    contract_payload = {
        **dict(prior_validation),
        "run_id": run_id,
        "request_id": validation_request_id,
    }
    try:
        outcome, records = require_validation_batch_contract(
            contract_payload,
            project_id=str(request.get("project_id") or ""),
            run_id=run_id,
            request_id=validation_request_id,
            validation_policy=validation_policy,
            candidates=candidates,
        )
    except RuntimeError as exc:
        raise ValueError(f"persisted validation is invalid: {exc}") from exc
    if outcome != "AWAITING_EVIDENCE":
        raise ValueError("persisted validation must have outcome AWAITING_EVIDENCE")
    return records


async def _fetch_provenance_record(artifact_id: str) -> dict:
    service_url = os.environ.get("PROVENANCE_SVC_URL", "").strip().rstrip("/")
    if not service_url:
        raise RuntimeError("PROVENANCE_SVC_URL is required for external evidence")
    url = f"{service_url}/v1/provenance/record/{quote(artifact_id, safe='')}"
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            headers = _downstream_service_headers()
            if headers:
                response = await client.get(url, headers=headers)
            else:
                response = await client.get(url)
            response.raise_for_status()
            record = response.json()
    except httpx.HTTPError as exc:
        raise RuntimeError(
            f"provenance service request failed for {artifact_id}: {exc}"
        ) from exc
    if not isinstance(record, dict):
        raise RuntimeError(
            f"provenance service returned an invalid record for {artifact_id}"
        )
    return record


def _downstream_service_headers() -> dict[str, str]:
    service_token = _internal_service_token()
    if not service_token:
        return {}
    return {_SERVICE_TOKEN_HEADER: service_token}


async def _verify_resume_external_evidence(
    evidence: list[dict],
    state: Mapping[str, object],
) -> list[dict]:
    request = state.get("request")
    if not isinstance(request, Mapping):
        raise ValueError("persisted full workflow request is required")
    project_id = request.get("project_id")
    run_id = state.get("run_id")
    if not isinstance(project_id, str) or not project_id.strip():
        raise ValueError("persisted project_id is required for external evidence")
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("persisted run_id is required for external evidence")
    artifact_ids = [
        artifact_id
        for item in evidence
        for artifact_id in item["evidence_ids"]
    ]
    records: list[dict] = []
    for offset in range(0, len(artifact_ids), _EXTERNAL_EVIDENCE_FETCH_BATCH_SIZE):
        batch = artifact_ids[offset : offset + _EXTERNAL_EVIDENCE_FETCH_BATCH_SIZE]
        records.extend(
            await asyncio.gather(
                *(_fetch_provenance_record(artifact_id) for artifact_id in batch)
            )
        )
    records_by_id = dict(zip(artifact_ids, records, strict=True))
    artifact_summaries: list[dict] = []
    for item in evidence:
        candidate_id = item["candidate_id"]
        canonical_smiles = item.get("canonical_smiles")
        merged_metrics: dict[str, float | int] = {}
        merged_uncertainties: dict[str, float | int] = {}
        for artifact_id in item["evidence_ids"]:
            record = records_by_id[artifact_id]
            payload = _validated_external_evidence_record(
                record,
                artifact_id=artifact_id,
                project_id=project_id,
                run_id=run_id,
                candidate_id=candidate_id,
                canonical_smiles=canonical_smiles,
            )
            _merge_evidence_numbers(
                merged_metrics,
                payload["metrics"],
                f"{artifact_id}.metrics",
            )
            _merge_evidence_numbers(
                merged_uncertainties,
                payload["uncertainties"],
                f"{artifact_id}.uncertainties",
            )
            artifact_summaries.append(
                {
                    "artifact_id": artifact_id,
                    "candidate_id": candidate_id,
                    "checksum": record["checksum"],
                    "signature": record["signature"],
                    "signature_type": record["signature_type"],
                    "recorded_at": record["recorded_at"],
                }
            )
        if merged_metrics != item["metrics"]:
            raise ValueError(
                f"external evidence metrics do not match provenance for {candidate_id}"
            )
        if merged_uncertainties != item["uncertainties"]:
            raise ValueError(
                f"external evidence uncertainties do not match provenance for {candidate_id}"
            )
    return artifact_summaries


def _validated_external_evidence_record(
    record: Mapping[str, object],
    *,
    artifact_id: str,
    project_id: str,
    run_id: str,
    candidate_id: str,
    canonical_smiles: object,
) -> dict:
    if record.get("artifact_id") != artifact_id:
        raise ValueError(f"provenance artifact_id mismatch for {artifact_id}")
    if record.get("artifact_type") != "external_validation_evidence":
        raise ValueError(
            f"provenance artifact_type mismatch for {artifact_id}"
        )
    if record.get("verified") is not True:
        raise ValueError(f"provenance verification failed for {artifact_id}")
    required_signature_type = os.environ.get(
        "PROVENANCE_REQUIRED_SIGNATURE_TYPE",
        "",
    ).strip()
    signature_type = record.get("signature_type")
    if required_signature_type and signature_type != required_signature_type:
        raise ValueError(
            f"provenance signature_type mismatch for {artifact_id}"
        )
    for field in ("signature", "recorded_at", "checksum"):
        value = record.get(field)
        if not isinstance(value, str) or not value:
            raise ValueError(f"provenance {field} is required for {artifact_id}")
    payload_base64 = record.get("payload_base64")
    if not isinstance(payload_base64, str) or not payload_base64:
        raise ValueError(f"provenance payload_base64 is required for {artifact_id}")
    try:
        payload_bytes = base64.b64decode(payload_base64, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError(
            f"provenance payload_base64 is invalid for {artifact_id}"
        ) from exc
    if (
        not payload_bytes
        or base64.b64encode(payload_bytes).decode("ascii") != payload_base64
    ):
        raise ValueError(
            f"provenance payload_base64 is not canonical for {artifact_id}"
        )
    checksum = f"sha256:{hashlib.sha256(payload_bytes).hexdigest()}"
    if record["checksum"] != checksum:
        raise ValueError(f"provenance checksum mismatch for {artifact_id}")
    try:
        payload = json.loads(payload_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"provenance payload is not valid JSON for {artifact_id}"
        ) from exc
    required_fields = {
        "schema_version",
        "project_id",
        "run_id",
        "candidate_id",
        "canonical_smiles",
        "metrics",
        "uncertainties",
    }
    if not isinstance(payload, dict) or set(payload) != required_fields:
        raise ValueError(
            f"provenance evidence payload contract mismatch for {artifact_id}"
        )
    expected_identity = {
        "project_id": project_id,
        "run_id": run_id,
        "candidate_id": candidate_id,
        "canonical_smiles": canonical_smiles,
    }
    if payload.get("schema_version") != "external_validation_evidence.v1":
        raise ValueError(
            f"provenance schema_version mismatch for {artifact_id}"
        )
    for field, expected in expected_identity.items():
        if payload.get(field) != expected:
            raise ValueError(
                f"provenance {field} mismatch for {artifact_id}"
            )
    metadata = record.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError(f"provenance metadata is required for {artifact_id}")
    for field, expected in expected_identity.items():
        if metadata.get(field) != expected:
            raise ValueError(
                f"provenance metadata {field} mismatch for {artifact_id}"
            )
    return {
        **payload,
        "metrics": _validated_evidence_number_map(
            payload["metrics"],
            f"{artifact_id}.metrics",
            require_nonempty=True,
            nonnegative=False,
        ),
        "uncertainties": _validated_evidence_number_map(
            payload["uncertainties"],
            f"{artifact_id}.uncertainties",
            require_nonempty=False,
            nonnegative=True,
        ),
    }


def _merge_evidence_numbers(
    target: dict[str, float | int],
    values: Mapping[str, float | int],
    field: str,
) -> None:
    for metric, value in values.items():
        if metric in target and target[metric] != value:
            raise ValueError(f"{field} conflicts for metric {metric}")
        target[metric] = value


def _validate_resume_external_evidence(
    value: object,
    candidates: list[object],
    state: Mapping[str, object],
) -> list[dict]:
    if not isinstance(value, list) or not value:
        raise ValueError("external_evidence must be a non-empty list")
    candidates_by_id: dict[str, str] = {}
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, Mapping):
            raise ValueError(f"persisted candidates[{index}] must be an object")
        candidate_id = candidate.get("candidate_id")
        canonical_smiles = candidate.get("canonical_smiles")
        if not isinstance(candidate_id, str) or not candidate_id.strip():
            raise ValueError(f"persisted candidates[{index}].candidate_id is required")
        if not isinstance(canonical_smiles, str) or not canonical_smiles.strip():
            raise ValueError(
                f"persisted candidates[{index}].canonical_smiles is required"
            )
        if candidate_id in candidates_by_id:
            raise ValueError(f"duplicate persisted candidate_id: {candidate_id}")
        candidates_by_id[candidate_id] = canonical_smiles

    allowed_fields = {
        "candidate_id",
        "canonical_smiles",
        "metrics",
        "uncertainties",
        "evidence_ids",
    }
    normalized: list[dict] = []
    seen_candidate_ids: set[str] = set()
    artifact_count = 0
    artifact_limit = _external_evidence_artifact_limit()
    required_external_metrics = _required_external_evidence_metrics(state)
    for index, item in enumerate(value):
        field = f"external_evidence[{index}]"
        if not isinstance(item, Mapping):
            raise ValueError(f"{field} must be an object")
        unknown_fields = set(item) - allowed_fields
        if unknown_fields:
            unknown = sorted(str(name) for name in unknown_fields)[0]
            raise ValueError(f"{field} contains unsupported field: {unknown}")
        candidate_id = item.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id.strip():
            raise ValueError(f"{field}.candidate_id must be a non-empty string")
        candidate_id = candidate_id.strip()
        if candidate_id not in candidates_by_id:
            raise ValueError(
                f"external_evidence references unknown candidate_id: {candidate_id}"
            )
        if candidate_id in seen_candidate_ids:
            raise ValueError(f"duplicate external_evidence candidate_id: {candidate_id}")
        seen_candidate_ids.add(candidate_id)
        canonical_smiles = item.get("canonical_smiles")
        if canonical_smiles is not None:
            if (
                not isinstance(canonical_smiles, str)
                or canonical_smiles.strip() != candidates_by_id[candidate_id]
            ):
                raise ValueError(
                    f"external_evidence canonical_smiles mismatch for {candidate_id}"
                )
            canonical_smiles = canonical_smiles.strip()
        else:
            canonical_smiles = candidates_by_id[candidate_id]
        metrics = _validated_evidence_number_map(
            item.get("metrics"),
            f"{field}.metrics",
            require_nonempty=True,
            nonnegative=False,
        )
        uncertainties = _validated_evidence_number_map(
            item.get("uncertainties", {}),
            f"{field}.uncertainties",
            require_nonempty=False,
            nonnegative=True,
        )
        evidence_ids = item.get("evidence_ids")
        if (
            not isinstance(evidence_ids, list)
            or not evidence_ids
            or any(
                not isinstance(evidence_id, str) or not evidence_id.strip()
                for evidence_id in evidence_ids
            )
        ):
            raise ValueError(f"{field}.evidence_ids must be a non-empty string list")
        normalized_evidence_ids = [evidence_id.strip() for evidence_id in evidence_ids]
        if len(set(normalized_evidence_ids)) != len(normalized_evidence_ids):
            raise ValueError(f"{field}.evidence_ids must be unique")
        artifact_count += len(normalized_evidence_ids)
        if artifact_count > artifact_limit:
            raise ValueError(
                "external_evidence must reference at most "
                f"{artifact_limit} provenance artifacts"
            )
        for metric, requires_uncertainty in required_external_metrics.items():
            if metric not in metrics:
                raise ValueError(f"{field}.metrics requires {metric}")
            if requires_uncertainty and metric not in uncertainties:
                raise ValueError(f"{field}.uncertainties requires {metric}")
        normalized_item = {
            "candidate_id": candidate_id,
            "canonical_smiles": canonical_smiles,
            "metrics": metrics,
            "uncertainties": uncertainties,
            "evidence_ids": normalized_evidence_ids,
        }
        normalized.append(normalized_item)
    return normalized


def _external_evidence_artifact_limit() -> int:
    raw_limit = os.environ.get(
        "EXTERNAL_EVIDENCE_MAX_ARTIFACTS",
        str(_DEFAULT_EXTERNAL_EVIDENCE_MAX_ARTIFACTS),
    ).strip()
    try:
        limit = int(raw_limit)
    except ValueError as exc:
        raise RuntimeError(
            "EXTERNAL_EVIDENCE_MAX_ARTIFACTS must be a positive integer"
        ) from exc
    if limit < 1:
        raise RuntimeError(
            "EXTERNAL_EVIDENCE_MAX_ARTIFACTS must be a positive integer"
        )
    return limit


def _validated_evidence_number_map(
    value: object,
    field: str,
    *,
    require_nonempty: bool,
    nonnegative: bool,
) -> dict[str, float | int]:
    if not isinstance(value, Mapping) or (require_nonempty and not value):
        qualifier = "non-empty " if require_nonempty else ""
        raise ValueError(f"{field} must be a {qualifier}object")
    normalized: dict[str, float | int] = {}
    for metric, raw_number in value.items():
        if not isinstance(metric, str) or not metric.strip():
            raise ValueError(f"{field} keys must be non-empty strings")
        if (
            isinstance(raw_number, bool)
            or not isinstance(raw_number, (int, float))
            or not math.isfinite(float(raw_number))
        ):
            raise ValueError(f"{field}.{metric} must be a finite number")
        if nonnegative and float(raw_number) < 0:
            raise ValueError(f"{field}.{metric} must be non-negative")
        normalized_metric = metric.strip()
        if normalized_metric in normalized:
            raise ValueError(
                f"{field} contains duplicate normalized metric: {normalized_metric}"
            )
        normalized[normalized_metric] = raw_number
    return normalized


def _required_external_evidence_metrics(
    state: Mapping[str, object],
) -> dict[str, bool]:
    request = state.get("request")
    if not isinstance(request, Mapping):
        return {}
    validation_policy = request.get("validation_policy")
    if not isinstance(validation_policy, Mapping):
        return {}
    thresholds = validation_policy.get("thresholds")
    if not isinstance(thresholds, list):
        return {}
    required: dict[str, bool] = {}
    for threshold in thresholds:
        if (
            isinstance(threshold, Mapping)
            and threshold.get("level") == 4
            and threshold.get("oracle") == "external"
            and isinstance(threshold.get("metric"), str)
        ):
            metric = str(threshold["metric"])
            required[metric] = "max_uncertainty" in threshold
    return required


@rest_app.post("/v1/orchestrator/runs/{run_id}/cancel")
async def cancel_run(run_id: str) -> dict:
    run_store, run_control = await _runtime()
    snapshot = await run_store.get_run(run_id)
    if snapshot is None:
        raise HTTPException(status_code=404, detail=f"Unknown run_id: {run_id}")
    _authorize_run_snapshot(snapshot)
    if snapshot["status"] == RunStatus.INTERRUPTED.value:
        return _public_run_snapshot(snapshot)
    if snapshot["status"] not in _NONTERMINAL_RUN_STATUS_VALUES:
        raise HTTPException(
            status_code=409,
            detail=f"run {run_id} cannot cancel from status {snapshot['status']}",
        )
    task = _RUN_TASKS.get(run_id)
    if task is None or task.done():
        if snapshot["status"] == RunStatus.AWAITING_EVIDENCE.value:
            interrupted = await _interrupt_cancelled_run(
                run_store,
                run_control,
                run_id,
                "run cancelled by client request",
            )
            if (
                interrupted is not None
                and interrupted["status"] == RunStatus.INTERRUPTED.value
            ):
                return _public_run_snapshot(interrupted)
        snapshot = await run_store.get_run(run_id)
        if snapshot is not None and snapshot["status"] == RunStatus.INTERRUPTED.value:
            return _public_run_snapshot(snapshot)
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
        return _public_run_snapshot(snapshot)
    status = snapshot["status"] if snapshot is not None else "unknown"
    raise HTTPException(
        status_code=409,
        detail=f"run {run_id} cancellation did not interrupt status {status}",
    )



async def _request_agent(
    request_client: AgentRequestClient | None,
    state: dict,
    entry_point: str,
    stage: str,
    business_payload: dict,
    *,
    candidate_index: int | None = None,
    preserve_correlation: bool = False,
    request_id_suffix: str | None = None,
) -> dict:
    run_id = str(state.get("run_id") or "")
    trace_id = str(state.get("trace_id") or "")
    if not run_id:
        raise ValueError("run_id is required for Orchestrator Agent requests")
    if not trace_id:
        raise ValueError("trace_id is required for Orchestrator Agent requests")
    if entry_point == "nl2obj":
        subject = _NL2OBJ_SUBJECT
        payload_type_url = _NL2OBJ_PAYLOAD_TYPE_URL
        schema_version = _NL2OBJ_SCHEMA_VERSION
    else:
        protocol = _AGENT_PROTOCOLS_BY_ENTRY_POINT[entry_point]
        subject = protocol.subject
        payload_type_url = protocol.payload_type_url
        schema_version = protocol.schema_version
    refinement_count = int(state.get("refinement_count", 0))
    parent_id = f"{run_id}:{stage}:{refinement_count}"
    request_kind = entry_point
    if business_payload.get("action") == "generator_coord/feedback/v1":
        request_kind = "generator_coord_feedback"
    request_id = f"{run_id}:{request_kind}:{refinement_count}"
    if candidate_index is not None:
        request_id = f"{request_id}:candidate-{candidate_index}"
    if request_id_suffix is not None:
        if not request_id_suffix or request_id_suffix != request_id_suffix.strip():
            raise ValueError("request_id_suffix must be a non-empty trimmed string")
        request_id = f"{request_id}:{request_id_suffix}"
    payload = dict(business_payload)
    payload.update(
        {
            "trace_id": trace_id,
            "parent_id": parent_id,
            "run_id": run_id,
            "request_id": request_id,
            "schema_version": schema_version,
        }
    )
    client = request_client or _shared_agent_request_client()
    result = await client.request(
        subject,
        payload,
        payload_type_url=payload_type_url,
        timeout=_agent_request_timeout(state, entry_point, payload),
    )
    business_result = dict(result)
    if not preserve_correlation:
        for field in ("run_id", "request_id", "schema_version"):
            business_result.pop(field, None)
    return business_result


def _agent_request_timeout(
    state: Mapping[str, object],
    entry_point: str,
    payload: Mapping[str, object],
) -> float:
    request = state.get("request")
    return agent_request_timeout_seconds(
        request if isinstance(request, Mapping) else {},
        entry_point,
        payload,
    )


class FullWorkflowClients:
    def __init__(self, request_client: AgentRequestClient | None = None) -> None:
        self.request_client = request_client

    async def compile_intent(self, state: dict) -> dict:
        request = dict(state.get("request") or {})
        excluded = {
            "artifact_ids",
            "clients",
            "parent_id",
            "request_id",
            "run_id",
            "schema_version",
            "trace_id",
            "workflow_scope",
        }
        business_request = {key: value for key, value in request.items() if key not in excluded}
        result = await _request_agent(
            self.request_client,
            state,
            "nl2obj",
            "planning",
            {
                **business_request,
                "project_id": str(request.get("project_id") or ""),
                "intent": str(state["nl_input"]),
            },
        )
        return {
            key: result[key]
            for key in ("cig", "hciv", "intent_cone", "objectives")
            if key in result
        }

    async def generate_candidates(self, state: dict) -> list[dict]:
        request = state.get("request", {})
        n_samples, generator_params = generation_controls(request)
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
            return {
                "outcome": "FAIL",
                "passed": False,
                "records": [],
                "results": [],
                "reason": "no candidates generated",
            }
        return await _validate_candidate_batch(
            self.request_client,
            state,
            candidates,
        )

    async def plan_routes(self, state: dict) -> dict:
        request = state.get("request", {})
        if not isinstance(request, Mapping):
            raise RuntimeError("full workflow request is required for retrosynthesis")
        candidate, _validation, candidate_index = _selected_full_candidate(state)
        identity = full_workflow_candidate_identity(state)
        retrosyn_engine = _required_runtime_text(request, "retrosyn_engine")
        result = await _request_agent(
            self.request_client,
            state,
            "retrosyn",
            "retrosyn",
            {
                **identity,
                "run_id": str(state.get("run_id", "")),
                "smiles": _candidate_smiles(candidate, purpose="retrosynthesis"),
                "engine": retrosyn_engine,
                "max_routes": int(
                    request.get("retrosyn_max_routes", request.get("max_routes", 3)) or 3
                ),
            },
            candidate_index=candidate_index,
        )
        return require_full_downstream_response_identity(
            result,
            state,
            stage="retrosyn",
        )

    async def assess_supply(self, state: dict) -> dict:
        candidate, _validation, candidate_index = _selected_full_candidate(state)
        routes = _retrosyn_routes(state)
        if not routes:
            return _unavailable_supply_result(state, "retrosyn.routes is empty")

        identity = full_workflow_candidate_identity(state)
        assessed_routes: list[dict] = []
        selected: dict | None = None
        for route in routes:
            route_id = _route_id(route)
            result = await _request_agent(
                self.request_client,
                state,
                "supply",
                "supply",
                {
                    **identity,
                    "run_id": str(state.get("run_id", "")),
                    "workflow_scope": "full",
                    "route_id": route_id,
                    "smiles": _candidate_smiles(candidate, purpose="supply assessment"),
                    "building_blocks": _route_building_blocks(route),
                },
                candidate_index=candidate_index,
                request_id_suffix=f"route-{_safe_id(route_id)}",
            )
            result = require_full_downstream_response_identity(
                result,
                state,
                stage="supply",
            )
            _require_route_id(result, route_id, stage="supply")
            assessed_routes.append(_route_assessment_summary(result))
            if selected is None or (
                _supply_feasibility(selected) != "available"
                and _supply_feasibility(result) == "available"
            ):
                selected = result
        if selected is None:
            raise RuntimeError("retrosyn routes were not assessed")
        return {
            **selected,
            "route_assessments": assessed_routes,
        }

    async def compile_synthesis(self, state: dict) -> dict:
        candidate, _validation, candidate_index = _selected_full_candidate(state)
        identity = full_workflow_candidate_identity(state)
        route = _selected_retrosyn_route_or_none(state)
        if route is None:
            return {
                "status": "not_compiled",
                "protocols": [],
                "blocking_evidence": [
                    {
                        "rule_id": "workflow_retrosyn_routes",
                        "reason": "retrosyn.routes is empty",
                    }
                ],
                **identity,
            }
        route_id = _route_id(route)
        supply = state.get("supply")
        if not isinstance(supply, Mapping):
            raise RuntimeError("selected route requires supply assessment before compilation")
        _require_route_id(supply, route_id, stage="supply")
        supply_feasibility = _supply_feasibility(supply)
        if supply_feasibility != "available":
            return {
                "status": "not_compiled",
                "route_id": route_id,
                "protocols": [],
                "blocking_evidence": [
                    {
                        "rule_id": "workflow_supply_feasibility",
                        "reason": (
                            f"selected route supply feasibility is {supply_feasibility}"
                        ),
                    }
                ],
                **identity,
            }

        result = await _request_agent(
            self.request_client,
            state,
            "srb",
            "srb",
            {
                **identity,
                "run_id": str(state.get("run_id", "")),
                "workflow_scope": "full",
                "route_id": route_id,
                "molecule": {
                    "smiles": _candidate_smiles(candidate, purpose="synthesis"),
                },
                "pathways": [route],
            },
            candidate_index=candidate_index,
        )
        result = require_full_downstream_response_identity(
            result,
            state,
            stage="srb",
        )
        _require_compiled_protocol_binding(result, route_id)
        return result

    async def execute_synthesis(self, state: dict) -> dict:
        candidate, _validation, candidate_index = _selected_full_candidate(state)
        identity = full_workflow_candidate_identity(state)
        route = _selected_retrosyn_route_or_none(state)
        if route is None:
            raise RuntimeError("selected route is required for synthesis execution")
        route_id = _route_id(route)
        srb = state.get("srb")
        if not isinstance(srb, Mapping):
            raise RuntimeError("compiled synthesis protocol is required for execution")
        protocols = _require_compiled_protocol_binding(srb, route_id)
        result = await _request_agent(
            self.request_client,
            state,
            "srb",
            "srb",
            {
                **identity,
                "action": "execute",
                "workflow_scope": "full",
                "route_id": route_id,
                "molecule": {
                    "smiles": _candidate_smiles(candidate, purpose="synthesis execution"),
                },
                "protocols": protocols,
            },
            candidate_index=candidate_index,
            request_id_suffix="execute",
        )
        result = require_full_downstream_response_identity(
            result,
            state,
            stage="srb execution",
        )
        if result.get("status") != "executed":
            raise RuntimeError("srb execution status must be executed")
        _require_compiled_protocol_binding(result, route_id)
        return result

    async def review_candidates(self, state: dict) -> dict:
        candidates = list(state.get("candidates", []))
        if not candidates:
            return {"verdict": "fail", "reason": "no candidate available for critic"}

        candidate, validation, candidate_index = _selected_full_candidate(state)
        smiles = _candidate_smiles(candidate, purpose="critic")
        properties = _full_workflow_critic_properties(state, candidate, validation)
        identity = full_workflow_candidate_identity(state)
        result = await _request_agent(
            self.request_client,
            state,
            "critic",
            "critic",
            {
                **identity,
                "run_id": str(state.get("run_id", "")),
                "smiles": smiles,
                "workflow_scope": "full",
                "properties": properties,
            },
            candidate_index=candidate_index,
        )
        return require_full_downstream_response_identity(
            result,
            state,
            stage="critic",
        )

    async def submit_critic_feedback(self, state: dict) -> dict:
        critic = state.get("critic")
        if not isinstance(critic, Mapping):
            raise RuntimeError("critic result is required before feedback submission")
        groups = critic_feedback_groups(state, critic)
        refinement_count = int(state.get("refinement_count", 0))
        run_id = str(state.get("run_id") or "")
        feedback = await _request_agent(
            self.request_client,
            state,
            "generator_coord",
            "critic",
            {
                "action": "generator_coord/feedback/v1",
                "route_request_id": f"{run_id}:generator_coord:{refinement_count}",
                "iteration": refinement_count,
                "groups": groups,
            },
        )
        require_feedback_acknowledgement(feedback, expected_groups=len(groups))
        return feedback


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
        "cig": state.get("cig"),
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


def _predict_candidate_properties(predictor, smiles: str) -> dict:
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
    return _normalise_critic_properties(row)


def _normalise_critic_properties(row: dict) -> dict:
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


def _full_workflow_critic_properties(
    state: dict,
    candidate: dict,
    validation_row: dict,
) -> dict:
    smiles = _candidate_smiles(candidate, purpose="critic")
    candidate_properties = _candidate_critic_properties(candidate, smiles)
    return full_workflow_critic_properties(
        state,
        candidate_properties,
        validation_row or None,
    )


def _candidate_critic_properties(candidate: dict, smiles: str) -> dict:
    row = dict(candidate)
    if not _has_core_critic_properties(row):
        try:
            from mf_chem.predict.engine import MolPredictEngine

            enriched = _predict_candidate_properties(MolPredictEngine(device_ids=[]), smiles)
            enriched.update(row)
            row = enriched
        except Exception as exc:
            LOGGER.warning("Skipping full workflow critic property enrichment: %s", exc)
    return row


def _has_core_critic_properties(row: dict) -> bool:
    return all(key in row for key in ("mw", "logp", "tpsa", "qed", "sa_score"))


async def _close_owned_crg_repository(repository: Any) -> None:
    close_task = asyncio.create_task(repository.close())
    try:
        await asyncio.shield(close_task)
    except asyncio.CancelledError as cancellation:
        try:
            await _await_task_completion(close_task)
        except BaseException as close_error:
            raise cancellation from close_error
        raise


async def _merge_agent_beliefs_into_crg(final_state: dict, run_id: str) -> dict:
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
    suppress_close_error = False
    primary_error: BaseException | None = None
    try:
        try:
            shared_crg = await repo.get_run_crg(run_id)
        except Exception as exc:
            LOGGER.warning("Failed to read shared CRG for run %s: %s", run_id, exc)
            suppress_close_error = True
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
            key = (
                str(edge.get("source_belief_id") or ""),
                str(edge.get("target_belief_id") or ""),
            )
            if key not in existing_edge_keys:
                merged_edges.append(edge)
        merged = dict(crg)
        merged["beliefs"] = merged_beliefs
        merged["edges"] = merged_edges
        merged["version"] = len(merged_beliefs) + len(merged_edges)
        return merged
    except BaseException as error:
        primary_error = error
        raise
    finally:
        try:
            await _close_owned_crg_repository(repo)
        except BaseException as close_error:
            if primary_error is None and (
                isinstance(close_error, asyncio.CancelledError) or not suppress_close_error
            ):
                raise
            LOGGER.warning(
                "Failed to close shared CRG repository: %s",
                close_error,
                exc_info=(type(close_error), close_error, close_error.__traceback__),
            )


async def _record_workflow_provenance(final_state: dict) -> None:
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
    payload_state = _persistable_state(final_state)
    payload_state.pop("provenance", None)
    payload_state["artifact_ids"] = parent_ids
    payload_bytes = json.dumps(
        payload_state,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    service_url = os.environ.get("PROVENANCE_SVC_URL", "").strip().rstrip("/")
    if not service_url:
        raise RuntimeError("PROVENANCE_SVC_URL is required for workflow provenance")
    record_payload = {
        "artifact_type": "workflow_state",
        "artifact_id": artifact_id,
        "parent_ids": parent_ids,
        "metadata": metadata,
        "payload_base64": base64.b64encode(payload_bytes).decode("ascii"),
    }
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            request_kwargs: dict[str, object] = {"json": record_payload}
            headers = _downstream_service_headers()
            if headers:
                request_kwargs["headers"] = headers
            response = await client.post(
                f"{service_url}/v1/provenance/record",
                **request_kwargs,
            )
            response.raise_for_status()
            record = response.json()
    except httpx.HTTPError as exc:
        raise RuntimeError(f"provenance service request failed: {exc}") from exc
    if not isinstance(record, dict):
        raise RuntimeError("provenance service response must be an object")
    if record.get("artifact_id") != artifact_id:
        raise RuntimeError("provenance service returned a mismatched artifact_id")
    signature = record.get("signature")
    recorded_at = record.get("recorded_at")
    if not isinstance(signature, str) or not signature:
        raise RuntimeError("provenance service response signature is required")
    if not isinstance(recorded_at, str) or not recorded_at:
        raise RuntimeError("provenance service response recorded_at is required")
    artifact_ids = list(final_state.get("artifact_ids", []))
    if artifact_id not in artifact_ids:
        artifact_ids.append(artifact_id)
    final_state["artifact_ids"] = artifact_ids
    final_state["provenance"] = {
        "recorded": True,
        "artifact_id": record["artifact_id"],
        "signature": signature,
        "recorded_at": recorded_at,
    }


async def _validate_candidate_batch(
    request_client: AgentRequestClient | None,
    state: dict,
    candidates: list[dict],
) -> dict:
    request = state.get("request")
    if not isinstance(request, Mapping):
        raise RuntimeError("full workflow request is required for validation")
    validation_policy = _required_runtime_policy(request, "validation_policy")
    teacher_policy = _required_runtime_policy(request, "teacher_policy")
    selection_policy = _required_runtime_policy(request, "selection_policy")
    candidate_payloads = [validation_candidate_payload(candidate) for candidate in candidates]
    run_id = str(state.get("run_id") or "")
    refinement_count = int(state.get("refinement_count", 0))
    validation_request_id = f"{run_id}:validation:{refinement_count}"
    resume_verified = state.get("external_evidence_resume_verified") is True
    prior_records = None
    if resume_verified:
        prior_records = _validated_evidence_resume_records(
            state,
            request,
            candidates,
        )
    elif request.get("external_evidence"):
        raise RuntimeError(
            "external_evidence requires a verified evidence resume checkpoint"
        )
    validation_payload = {
        "project_id": str(request.get("project_id") or ""),
        "run_id": str(state.get("run_id") or ""),
        "validation_policy": dict(validation_policy),
        "teacher_policy": dict(teacher_policy),
        "selection_policy": dict(selection_policy),
        "candidates": candidate_payloads,
        "external_evidence": request.get("external_evidence"),
    }
    if prior_records is not None:
        validation_payload.update(
            {
                "resume_external_evidence": True,
                "prior_validation_records": prior_records,
            }
        )
    result = await _request_agent(
        request_client,
        state,
        "validation",
        "validating",
        validation_payload,
        preserve_correlation=True,
    )
    outcome, raw_records = require_validation_batch_contract(
        result,
        project_id=str(request.get("project_id") or ""),
        run_id=run_id,
        request_id=validation_request_id,
        validation_policy=validation_policy,
        candidates=candidates,
    )
    business_result = dict(result)
    for field in ("run_id", "request_id", "schema_version"):
        business_result.pop(field, None)
    validation_result = {
        **business_result,
        "schema_version": "validation.batch.v1",
        "outcome": outcome,
        "passed": outcome == "PASS",
        "records": raw_records,
        "results": raw_records,
    }
    groups = validation_feedback_groups(
        candidates,
        raw_records,
        teacher_policy=teacher_policy,
        validation_policy=validation_policy,
    )
    if not groups:
        return validation_result
    feedback = await _request_agent(
        request_client,
        state,
        "generator_coord",
        "validating",
        {
            "action": "generator_coord/feedback/v1",
            "route_request_id": (f"{run_id}:generator_coord:{refinement_count}"),
            "iteration": refinement_count,
            "groups": groups,
        },
    )
    require_feedback_acknowledgement(feedback, expected_groups=len(groups))
    validation_result["feedback"] = feedback
    return validation_result


def _required_runtime_policy(
    request: Mapping[str, object],
    field: str,
) -> Mapping[str, object]:
    policy = request.get(field)
    if not isinstance(policy, Mapping):
        raise RuntimeError(f"{field} is required for full workflow")
    return policy


def _required_runtime_text(value: Mapping[str, object], field: str) -> str:
    item = value.get(field)
    if not isinstance(item, str) or not item or item != item.strip():
        raise RuntimeError(f"{field} must be a non-empty trimmed string")
    return item


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


def _selected_full_candidate(state: dict) -> tuple[dict, dict, int]:
    candidates = list(state.get("candidates", []) or [])
    if not candidates:
        raise RuntimeError("candidates are required for full workflow synthesis")
    if not all(isinstance(candidate, dict) for candidate in candidates):
        raise RuntimeError("candidate entries must be objects")
    validation = state.get("validation")
    if not isinstance(validation, Mapping):
        raise RuntimeError("full workflow requires validation records")
    request = state.get("request")
    selection_policy = request.get("selection_policy") if isinstance(request, Mapping) else None
    if not isinstance(selection_policy, Mapping):
        raise RuntimeError("selection_policy is required for full workflow")
    return select_full_candidate(
        candidates,
        validation,
        selection_policy,
        validation_policy=request.get("validation_policy"),
    )


def _first_retrosyn_route(state: dict) -> dict:
    route = _first_retrosyn_route_or_none(state)
    if route is None:
        raise RuntimeError("retrosyn.routes is required for full workflow synthesis")
    return route


def _first_retrosyn_route_or_none(state: dict) -> dict | None:
    routes = _retrosyn_routes(state)
    return dict(routes[0]) if routes else None


def _selected_retrosyn_route_or_none(state: dict) -> dict | None:
    routes = _retrosyn_routes(state)
    if not routes:
        return None
    supply = state.get("supply")
    selected_route_id = supply.get("route_id") if isinstance(supply, Mapping) else None
    if selected_route_id is None:
        return dict(routes[0])
    if not isinstance(selected_route_id, str) or not selected_route_id.strip():
        raise RuntimeError("supply route_id must be a non-empty string")
    for route in routes:
        if route.get("route_id") == selected_route_id:
            return dict(route)
    raise RuntimeError("supply route_id must reference a retrosyn route")


def _route_id(route: Mapping[str, object]) -> str:
    return _required_runtime_text(route, "route_id")


def _require_route_id(
    response: Mapping[str, object],
    expected_route_id: str,
    *,
    stage: str,
) -> None:
    if response.get("route_id") != expected_route_id:
        raise RuntimeError(f"{stage} response route_id must match selected route")


def _require_compiled_protocol_binding(
    response: Mapping[str, object],
    expected_route_id: str,
) -> list[dict]:
    _require_route_id(response, expected_route_id, stage="srb")
    protocols = response.get("protocols")
    if not isinstance(protocols, list) or len(protocols) != 1:
        raise RuntimeError("srb response must contain exactly one selected-route protocol")
    protocol = protocols[0]
    if not isinstance(protocol, dict):
        raise RuntimeError("srb protocol must be an object")
    if protocol.get("route_id") != expected_route_id:
        raise RuntimeError("srb protocol route_id must match selected route")
    return [dict(protocol)]


def _supply_feasibility(supply: Mapping[str, object]) -> str:
    assessment = supply.get("supply_assessment")
    if not isinstance(assessment, Mapping):
        raise RuntimeError("selected route supply_assessment is required before compilation")
    value = assessment.get("overall_feasibility")
    if not isinstance(value, str) or not value or value != value.strip():
        raise RuntimeError("selected route supply feasibility must be a non-empty string")
    return value.lower()


def _route_assessment_summary(supply: Mapping[str, object]) -> dict:
    assessment = supply.get("supply_assessment")
    if not isinstance(assessment, Mapping):
        raise RuntimeError("route supply_assessment must be an object")
    return {
        "route_id": _required_runtime_text(supply, "route_id"),
        "status": str(supply.get("status") or "assessed"),
        "supply_assessment": dict(assessment),
    }


def _retrosyn_routes(state: dict) -> list[dict]:
    retrosyn = state.get("retrosyn")
    if not isinstance(retrosyn, dict):
        return []
    routes = retrosyn.get("routes")
    if routes is None:
        return []
    if not isinstance(routes, list):
        raise RuntimeError("retrosyn routes must be a list")
    if not all(isinstance(route, dict) for route in routes):
        raise RuntimeError("retrosyn route entries must be objects")
    return [dict(route) for route in routes]


def _unavailable_supply_result(state: dict, reason: str) -> dict:
    candidate, _validation, _candidate_index = _selected_full_candidate(state)
    return {
        "agent": "supply_agent",
        "status": "assessed",
        "smiles": _candidate_smiles(candidate, purpose="supply assessment"),
        **full_workflow_candidate_identity(state),
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(rest_app, host="0.0.0.0", port=8011, log_level="info")
