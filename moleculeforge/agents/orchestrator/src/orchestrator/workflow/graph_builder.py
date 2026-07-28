"""LangGraph workflow builder for orchestrator state transitions."""

from __future__ import annotations

import inspect
import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import TypedDict

FULL_WORKFLOW_BLOCKING_CRITIC_RULE_IDS = (
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
)


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
    crg: dict
    cig: dict
    hciv: dict
    intent_cone: dict
    candidates: list[dict]
    generation_feedback: list[dict]
    validation: dict
    retrosyn: dict
    supply: dict
    srb: dict
    critic: dict
    validation_passed: bool
    refinement_count: int
    max_refinements: int


class WorkflowGraph:
    def __init__(self, clients=None, workflow_scope: str = "state_only") -> None:
        self.clients = clients
        self.workflow_scope = workflow_scope

    def build(self):
        end, state_graph = _langgraph_symbols()
        graph = state_graph(WorkflowState)
        graph.add_node("planning", self._planning)
        graph.set_entry_point("planning")
        if self.workflow_scope == "state_only":
            graph.add_edge("planning", end)
            return graph.compile()

        graph.add_node("generating", self._generating)
        graph.add_node("validating", self._validating)
        graph.add_node("critic", self._critic)
        graph.add_node("refining", self._refining)
        graph.add_node("escalating", self._escalating)
        graph.add_edge("planning", "generating")
        graph.add_edge("generating", "validating")
        validation_done_node = "critic"
        if self.workflow_scope == "full":
            graph.add_node("retrosyn", self._retrosyn)
            graph.add_edge("retrosyn", "critic")
            validation_done_node = "retrosyn"
        graph.add_conditional_edges(
            "validating",
            self._route_after_validation,
            {
                "done": validation_done_node,
                "refine": "refining",
                "escalate": "escalating",
            },
        )
        graph.add_conditional_edges(
            "critic",
            self._route_after_critic,
            {
                "done": end,
                "refine": "refining",
                "escalate": "escalating",
            },
        )
        graph.add_edge("refining", "generating")
        graph.add_edge("escalating", end)
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
        if int(next_state.get("refinement_count", 0)) > 0:
            for key in ("validation", "critic", "retrosyn", "supply", "srb"):
                next_state[key] = None
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
            outcome = str(result.get("outcome") or result.get("status") or "").upper()
            if outcome == "AWAITING_EVIDENCE":
                next_state["status"] = "awaiting_evidence"
        return next_state

    async def _retrosyn(self, state: WorkflowState) -> WorkflowState:
        next_state = self._with_status(state, "RETROSYN")
        if self.clients is not None and hasattr(self.clients, "plan_routes"):
            result = await _maybe_await(self.clients.plan_routes(next_state))
            if not isinstance(result, dict):
                raise RuntimeError("plan_routes must return a dict")
            next_state["retrosyn"] = result
        if self.clients is not None and hasattr(self.clients, "assess_supply"):
            result = await _maybe_await(self.clients.assess_supply(next_state))
            if not isinstance(result, dict):
                raise RuntimeError("assess_supply must return a dict")
            next_state["supply"] = result
        if self.clients is not None and hasattr(self.clients, "compile_synthesis"):
            result = await _maybe_await(self.clients.compile_synthesis(next_state))
            if not isinstance(result, dict):
                raise RuntimeError("compile_synthesis must return a dict")
            next_state["srb"] = result
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
        refinement_count = int(next_state.get("refinement_count", 0)) + 1
        next_state["refinement_count"] = refinement_count
        critic = next_state.get("critic")
        validation = next_state.get("validation")
        if isinstance(critic, dict) and str(critic.get("verdict") or "").lower() == "fail":
            feedback = list(next_state.get("generation_feedback", []))
            entry = {
                "source": "critic",
                "refinement_count": refinement_count,
                "verdict": "fail",
            }
            if "reason" in critic:
                entry["reason"] = str(critic["reason"])
            if isinstance(critic.get("rule_results"), list):
                entry["rule_results"] = list(critic["rule_results"])
            feedback.append(entry)
            next_state["generation_feedback"] = feedback
        elif isinstance(validation, dict) and validation:
            feedback = list(next_state.get("generation_feedback", []))
            entry = {
                "source": "validation",
                "refinement_count": refinement_count,
                "passed": bool(validation.get("passed", False)),
            }
            if "reason" in validation:
                entry["reason"] = str(validation["reason"])
            if isinstance(validation.get("results"), list):
                entry["results"] = list(validation["results"])
            feedback.append(entry)
            next_state["generation_feedback"] = feedback
        return next_state

    async def _escalating(self, state: WorkflowState) -> WorkflowState:
        return self._with_status(state, "ESCALATING")

    def _route_after_validation(self, state: WorkflowState) -> str:
        if bool(state.get("validation_passed", True)):
            return "done"
        if int(state.get("refinement_count", 0)) < int(state.get("max_refinements", 1)):
            return "refine"
        return "escalate"

    def _route_after_critic(self, state: WorkflowState) -> str:
        critic = state.get("critic")
        if not isinstance(critic, dict):
            return "done"
        if str(critic.get("verdict") or "").lower() != "fail":
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
                "timestamp": datetime.now(UTC).isoformat(),
            }
        )
        next_state["status"] = status
        next_state["history"] = history
        next_state["events"] = events
        next_state["crg"] = _record_workflow_stage_belief(next_state, status)
        return next_state


def build_graph():
    return WorkflowGraph()


def _langgraph_symbols():
    from langgraph.graph import END, StateGraph

    return END, StateGraph


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


def full_workflow_critic_properties(
    state: Mapping[str, object],
    candidate: Mapping[str, object],
    validation: Mapping[str, object] | None = None,
) -> dict:
    candidate_properties = candidate.get("properties")
    properties = dict(candidate_properties) if isinstance(candidate_properties, Mapping) else {}
    properties.update({str(key): value for key, value in candidate.items() if key != "properties"})
    if validation is not None:
        validation_properties = validation.get("properties")
        if isinstance(validation_properties, Mapping):
            properties.update(validation_properties)
        properties.update(validation)
        cascade = validation.get("cascade")
        if isinstance(cascade, Mapping):
            for level_result in cascade.values():
                if not isinstance(level_result, Mapping):
                    continue
                scores = level_result.get("result")
                if isinstance(scores, Mapping):
                    properties.update(scores)
    properties.update(_srb_critic_properties(state))
    properties.update(_supply_critic_properties(state))
    properties.update(_request_critic_properties(state))
    properties["_critic_blocking_rule_ids"] = list(FULL_WORKFLOW_BLOCKING_CRITIC_RULE_IDS)
    return _normalise_critic_properties(properties)


def _supply_critic_properties(state: Mapping[str, object]) -> dict:
    supply = state.get("supply")
    if not isinstance(supply, Mapping):
        return {}
    assessment = supply.get("supply_assessment")
    if not isinstance(assessment, Mapping):
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


def _srb_critic_properties(state: Mapping[str, object]) -> dict:
    srb = state.get("srb")
    if not isinstance(srb, Mapping):
        return {}
    protocols = srb.get("protocols")
    if not isinstance(protocols, list) or not protocols:
        return {}
    protocol = protocols[0]
    if not isinstance(protocol, Mapping):
        return {}
    properties = {
        "estimated_cost_per_gram": float(protocol.get("total_estimated_cost_usd") or 0.0),
    }
    steps = protocol.get("steps")
    if isinstance(steps, list):
        properties["synthesis_steps"] = len(steps)
    return properties


def _request_critic_properties(state: Mapping[str, object]) -> dict:
    request = state.get("request")
    if not isinstance(request, Mapping):
        return {}
    return {
        key: request[key]
        for key in (
            "isoform_data_count",
            "kinase_selectivity_ratio",
            "cns_mpo",
            "bbb_score",
        )
        if key in request
    }


def _normalise_critic_properties(properties: dict) -> dict:
    pains_alerts = properties.get("pains_alerts", 0)
    if isinstance(pains_alerts, list):
        properties["pains_alerts"] = len(pains_alerts)
    properties["pains_alert_count"] = int(properties.get("pains_alerts", 0) or 0)
    herg_risk = properties.get("herg_risk", 0.0)
    if isinstance(herg_risk, str):
        properties["herg_risk"] = {
            "low": 0.1,
            "medium": 0.5,
            "high": 0.9,
        }.get(herg_risk.lower(), 0.0)
    return properties


def _record_workflow_stage_belief(state: WorkflowState, status: str) -> dict:
    crg = dict(state.get("crg") or {})
    request = state.get("request") or {}
    run_id = str(state.get("run_id", ""))
    events = list(state.get("events", []))
    event_index = max(0, len(events) - 1)
    project_id = str(request.get("project_id") or crg.get("project_id") or run_id)

    beliefs = list(crg.get("beliefs", []))
    edges = list(crg.get("edges", []))
    belief_id = f"belief-{_safe_fragment(run_id)}-{event_index}-{status.lower()}"
    previous_belief_id = str(beliefs[-1]["id"]) if beliefs else ""
    belief = {
        "id": belief_id,
        "subject": run_id,
        "predicate": "workflow_stage",
        "object": status,
        "confidence": 1.0,
        "evidence_ids": list(state.get("artifact_ids", [])),
        "source_agent": "orchestrator",
        "timestamp_ns": int(datetime.now(UTC).timestamp() * 1e9),
    }
    beliefs.append(belief)
    if previous_belief_id:
        edges.append(
            {
                "source_belief_id": previous_belief_id,
                "target_belief_id": belief_id,
                "relation": "derives_from",
                "weight": 1.0,
            }
        )

    return {
        "project_id": project_id,
        "beliefs": beliefs,
        "edges": edges,
        "version": len(beliefs),
        "provenance_id": str(crg.get("provenance_id", "")),
    }


def _safe_fragment(value: str) -> str:
    return "".join(ch if ch.isalnum() else "-" for ch in value).strip("-")
