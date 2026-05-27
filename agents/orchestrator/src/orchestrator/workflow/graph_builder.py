"""LangGraph workflow builder for orchestrator state transitions."""

from __future__ import annotations

import inspect
import uuid
from datetime import datetime, timezone
from typing import TypedDict

from langgraph.graph import END, StateGraph


class WorkflowState(TypedDict, total=False):
    nl_input: str
    status: str
    history: list[str]
    events: list[dict]
    run_id: str
    trace_id: str
    artifact_ids: list[str]
    workflow_scope: str
    request: dict
    cig: dict
    hciv: dict
    intent_cone: dict
    candidates: list[dict]
    validation: dict
    retrosyn: dict
    critic: dict
    validation_passed: bool
    refinement_count: int
    max_refinements: int


class WorkflowGraph:
    def __init__(self, clients=None, workflow_scope: str = "state_only") -> None:
        self.clients = clients
        self.workflow_scope = workflow_scope

    def build(self):
        graph = StateGraph(WorkflowState)
        graph.add_node("planning", self._planning)
        graph.add_node("generating", self._generating)
        graph.add_node("validating", self._validating)
        graph.add_node("retrosyn", self._retrosyn)
        graph.add_node("critic", self._critic)
        graph.add_node("refining", self._refining)
        graph.add_node("escalating", self._escalating)
        graph.set_entry_point("planning")
        graph.add_edge("planning", "generating")
        graph.add_edge("generating", "validating")
        graph.add_conditional_edges(
            "validating",
            self._route_after_validation,
            {
                "done": "retrosyn",
                "refine": "refining",
                "escalate": "escalating",
            },
        )
        graph.add_edge("retrosyn", "critic")
        graph.add_edge("critic", END)
        graph.add_edge("refining", "generating")
        graph.add_edge("escalating", END)
        return graph.compile()

    async def _planning(self, state: WorkflowState) -> WorkflowState:
        next_state = self._with_status(state, "PLANNING")
        if self.clients is not None and hasattr(self.clients, "compile_intent"):
            result = await _maybe_await(self.clients.compile_intent(next_state))
            if not isinstance(result, dict):
                raise RuntimeError("compile_intent must return a dict")
            next_state.update(result)
        return next_state

    async def _generating(self, state: WorkflowState) -> WorkflowState:
        next_state = self._with_status(state, "GENERATING")
        if self.clients is not None and hasattr(self.clients, "generate_candidates"):
            result = await _maybe_await(self.clients.generate_candidates(next_state))
            if not isinstance(result, list):
                raise RuntimeError("generate_candidates must return a list")
            next_state["candidates"] = result
        return next_state

    async def _validating(self, state: WorkflowState) -> WorkflowState:
        next_state = self._with_status(state, "VALIDATING")
        if self.clients is not None and hasattr(self.clients, "validate_candidates"):
            result = await _maybe_await(self.clients.validate_candidates(next_state))
            if not isinstance(result, dict):
                raise RuntimeError("validate_candidates must return a dict")
            next_state["validation"] = result
            next_state["validation_passed"] = bool(result.get("passed", False))
        return next_state

    async def _retrosyn(self, state: WorkflowState) -> WorkflowState:
        next_state = self._with_status(state, "RETROSYN")
        if self.clients is not None and hasattr(self.clients, "plan_routes"):
            result = await _maybe_await(self.clients.plan_routes(next_state))
            if not isinstance(result, dict):
                raise RuntimeError("plan_routes must return a dict")
            next_state["retrosyn"] = result
        return next_state

    async def _critic(self, state: WorkflowState) -> WorkflowState:
        next_state = self._with_status(state, "CRITIC")
        if self.clients is not None and hasattr(self.clients, "review_candidates"):
            result = await _maybe_await(self.clients.review_candidates(next_state))
            if not isinstance(result, dict):
                raise RuntimeError("review_candidates must return a dict")
            next_state["critic"] = result
        return next_state

    async def _refining(self, state: WorkflowState) -> WorkflowState:
        next_state = self._with_status(state, "REFINING")
        next_state["refinement_count"] = int(next_state.get("refinement_count", 0)) + 1
        return next_state

    async def _escalating(self, state: WorkflowState) -> WorkflowState:
        return self._with_status(state, "ESCALATING")

    def _route_after_validation(self, state: WorkflowState) -> str:
        if bool(state.get("validation_passed", True)):
            return "done"
        if int(state.get("refinement_count", 0)) < int(state.get("max_refinements", 1)):
            return "refine"
        return "escalate"

    def _with_status(self, state: WorkflowState, status: str) -> WorkflowState:
        next_state = dict(state)
        history = list(next_state.get("history", []))
        history.append(status)
        events = list(next_state.get("events", []))
        events.append(
            {
                "event_index": len(events),
                "stage": status,
                "run_id": str(next_state.get("run_id", "")),
                "trace_id": str(next_state.get("trace_id", "")),
                "artifact_ids": list(next_state.get("artifact_ids", [])),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
        next_state["status"] = status
        next_state["history"] = history
        next_state["events"] = events
        return next_state


def build_graph():
    return WorkflowGraph()


async def _maybe_await(value):
    if inspect.isawaitable(value):
        return await value
    return value


def create_initial_state(
    nl_input: str,
    run_id: str | None = None,
    trace_id: str | None = None,
    artifact_ids: list[str] | None = None,
    workflow_scope: str = "state_only",
) -> WorkflowState:
    resolved_run_id = run_id or f"run-{uuid.uuid4().hex}"
    return {
        "nl_input": nl_input,
        "status": "PLANNING",
        "history": [],
        "events": [],
        "run_id": resolved_run_id,
        "trace_id": trace_id or f"trace-{uuid.uuid4().hex}",
        "artifact_ids": list(artifact_ids or []),
        "workflow_scope": workflow_scope,
        "validation_passed": True,
        "refinement_count": 0,
        "max_refinements": 1,
    }
