"""LangGraph workflow builder for orchestrator state transitions."""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any, TypedDict

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
    "rule_081",
    "crg_validation_status",
    "crg_retrosyn_routes",
    "crg_supply_feasibility",
    "workflow_retrosyn_routes",
    "workflow_supply_feasibility",
    "workflow_srb_protocols",
)

_DEFAULT_AGENT_REQUEST_TIMEOUT_SECONDS = 60.0
_DEFAULT_VALIDATION_ORACLE_TIMEOUT_SECONDS = 300.0
_DEFAULT_ICLM_MODEL_UPDATE_TIMEOUT_SECONDS = 330.0
_DEFAULT_TEACHER_TIMEOUT_SECONDS = 60.0
_AGENT_REQUEST_TIMEOUT_BUFFER_SECONDS = 60.0
_DEFAULT_OPENFE_QUICKRUN_TIMEOUT_SECONDS = 3600.0
_DEFAULT_OPENFE_GATHER_TIMEOUT_SECONDS = 600.0
_DEFAULT_OPENFE_MAX_TRANSFORMATIONS_PER_PAIR = 2
_FEP_RUNNER_TIMEOUT_BUFFER_SECONDS = 60.0
_VALIDATION_CHUNK_TIMEOUT_BUFFER_SECONDS = 30.0


def agent_request_timeout_seconds(
    request: Mapping[str, object],
    entry_point: str,
    payload: Mapping[str, object],
) -> float:
    if "agent_request_timeout_seconds" in request:
        return _positive_timeout_value(
            request["agent_request_timeout_seconds"],
            "agent_request_timeout_seconds",
        )
    if entry_point == "validation":
        return _validation_request_timeout_seconds(payload)
    if (
        entry_point == "generator_coord"
        and payload.get("action") == "generator_coord/feedback/v1"
    ):
        groups = payload.get("groups")
        group_count = len(groups) if isinstance(groups, list) and groups else 1
        teacher_timeout = _positive_environment_timeout(
            "HYPSEEK_TEACHER_TIMEOUT_SECONDS",
            _DEFAULT_TEACHER_TIMEOUT_SECONDS,
        )
        model_timeout = _positive_environment_timeout(
            "ICLM_MODEL_UPDATE_TIMEOUT_SECONDS",
            _DEFAULT_ICLM_MODEL_UPDATE_TIMEOUT_SECONDS,
        )
        return (
            group_count * (teacher_timeout + model_timeout + 30.0)
            + _AGENT_REQUEST_TIMEOUT_BUFFER_SECONDS
        )
    return _DEFAULT_AGENT_REQUEST_TIMEOUT_SECONDS


def _validation_request_timeout_seconds(payload: Mapping[str, object]) -> float:
    if payload.get("resume_external_evidence") is True:
        return _DEFAULT_AGENT_REQUEST_TIMEOUT_SECONDS
    policy = payload.get("validation_policy")
    candidates = payload.get("candidates")
    if not isinstance(policy, Mapping) or not isinstance(candidates, list):
        return _DEFAULT_AGENT_REQUEST_TIMEOUT_SECONDS
    oracle_level = policy.get("oracle_level")
    batch_size = policy.get("batch_size")
    max_concurrency = policy.get("max_concurrency")
    thresholds = policy.get("thresholds")
    if (
        isinstance(oracle_level, bool)
        or not isinstance(oracle_level, int)
        or oracle_level not in range(5)
        or isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size <= 0
        or isinstance(max_concurrency, bool)
        or not isinstance(max_concurrency, int)
        or max_concurrency <= 0
        or not isinstance(thresholds, list)
    ):
        return _DEFAULT_AGENT_REQUEST_TIMEOUT_SECONDS
    chunks = max(1, math.ceil(len(candidates) / batch_size))
    generic_rounds = 0
    for level in range(min(oracle_level, 3) + 1):
        if level == 3:
            continue
        oracle_count = 1
        if level == 1 and any(
            isinstance(threshold, Mapping) and threshold.get("oracle") == "boltz2"
            for threshold in thresholds
        ):
            oracle_count += 1
        generic_rounds += math.ceil((chunks * oracle_count) / max_concurrency)
    oracle_timeout = _positive_environment_timeout(
        "VALIDATION_ORACLE_TIMEOUT_SECONDS",
        _DEFAULT_VALIDATION_ORACLE_TIMEOUT_SECONDS,
    )
    fep_budget = 0.0
    if oracle_level >= 3:
        n_repeats = _validation_fep_repeats(policy)
        if n_repeats is None:
            generic_rounds += math.ceil(chunks / max_concurrency)
        else:
            chunk_size = min(batch_size, max(1, len(candidates)))
            fep_chunk_timeout = max(
                oracle_timeout,
                _fep_chunk_timeout_seconds(n_repeats, chunk_size),
            )
            fep_budget = (
                math.ceil(chunks / max_concurrency) * fep_chunk_timeout
            )
    return (
        generic_rounds * oracle_timeout
        + fep_budget
        + _AGENT_REQUEST_TIMEOUT_BUFFER_SECONDS
    )


def _validation_fep_repeats(policy: Mapping[str, object]) -> int | None:
    oracle_inputs = policy.get("oracle_inputs")
    if not isinstance(oracle_inputs, Mapping):
        return None
    fep_inputs = oracle_inputs.get("fep")
    if not isinstance(fep_inputs, Mapping):
        return None
    parameters = fep_inputs.get("oracle_parameters")
    if not isinstance(parameters, Mapping):
        return None
    value = parameters.get("n_repeats")
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def _fep_chunk_timeout_seconds(n_repeats: int, chunk_size: int) -> float:
    max_transformations = _positive_environment_integer(
        "OPENFE_MAX_TRANSFORMATIONS_PER_PAIR",
        _DEFAULT_OPENFE_MAX_TRANSFORMATIONS_PER_PAIR,
    )
    quickrun_timeout = _positive_environment_timeout(
        "OPENFE_QUICKRUN_TIMEOUT_SECONDS",
        _DEFAULT_OPENFE_QUICKRUN_TIMEOUT_SECONDS,
    )
    gather_timeout = _positive_environment_timeout(
        "OPENFE_GATHER_TIMEOUT_SECONDS",
        _DEFAULT_OPENFE_GATHER_TIMEOUT_SECONDS,
    )
    return (
        n_repeats
        * chunk_size
        * (max_transformations * quickrun_timeout + gather_timeout)
        + _FEP_RUNNER_TIMEOUT_BUFFER_SECONDS
        + _VALIDATION_CHUNK_TIMEOUT_BUFFER_SECONDS
    )


def _positive_environment_timeout(name: str, default: float) -> float:
    return _positive_timeout_value(os.environ.get(name, str(default)), name)


def _positive_environment_integer(name: str, default: int) -> int:
    raw_value = os.environ.get(name, str(default))
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if value <= 0 or str(value) != raw_value:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _positive_timeout_value(value: object, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be positive")
    try:
        timeout = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be positive") from exc
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError(f"{field} must be positive")
    return timeout


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
    validation_outcome: str
    invalid_policy: dict
    retrosyn: dict
    supply: dict
    srb: dict
    srb_execution: dict
    critic: dict
    critic_feedback: dict
    validation_passed: bool
    refinement_count: int
    max_refinements: int


class WorkflowGraph:
    def __init__(self, clients=None) -> None:
        self.clients = clients

    def build(self, *, entry_point: str = "planning"):
        if entry_point not in {"planning", "validating"}:
            raise ValueError(f"unsupported workflow entry point: {entry_point}")
        end, state_graph = _langgraph_symbols()
        graph = state_graph(WorkflowState)
        graph.add_node("planning", self._planning)
        graph.add_node("generating", self._generating)
        graph.add_node("validating", self._validating)
        graph.add_node("retrosyn", self._retrosyn)
        graph.add_node("critic", self._critic)
        graph.add_node("executing", self._executing)
        graph.add_node("refining", self._refining)
        graph.add_node("escalating", self._escalating)
        graph.add_node("validation_error", self._validation_error)
        graph.add_node("awaiting_evidence", self._awaiting_evidence)
        graph.add_conditional_edges(
            "planning",
            self._route_after_planning,
            {
                "generate": "generating",
                "escalate": "escalating",
            },
        )
        graph.add_edge("generating", "validating")
        graph.add_edge("retrosyn", "critic")
        graph.add_conditional_edges(
            "validating",
            self._route_after_validation,
            {
                "done": "retrosyn",
                "refine": "refining",
                "escalate": "escalating",
                "await": "awaiting_evidence",
                "error": "validation_error",
            },
        )
        critic_routes = {
            "done": end,
            "refine": "refining",
            "escalate": "escalating",
            "execute": "executing",
        }
        graph.add_edge("executing", end)
        graph.add_conditional_edges(
            "critic",
            self._route_after_critic,
            critic_routes,
        )
        graph.add_edge("refining", "generating")
        graph.add_edge("escalating", end)
        graph.add_edge("validation_error", end)
        graph.add_edge("awaiting_evidence", end)
        graph.set_entry_point(entry_point)
        return graph.compile()

    async def _planning(self, state: WorkflowState) -> WorkflowState:
        next_state = self._with_status(state, "PLANNING")
        if self.clients is not None and hasattr(self.clients, "compile_intent"):
            result = await _maybe_await(self.clients.compile_intent(next_state))
            if not isinstance(result, dict):
                raise RuntimeError("compile_intent must return a dict")
            next_state.update(result)
        conflicts = policy_direction_conflicts(next_state)
        if conflicts:
            next_state["invalid_policy"] = {
                "reason": "CIG objective direction conflicts with workflow policy",
                "conflicts": conflicts,
            }
        return next_state

    async def _generating(self, state: WorkflowState) -> WorkflowState:
        next_state = self._with_status(state, "GENERATING")
        if int(next_state.get("refinement_count", 0)) > 0:
            for key in (
                "validation",
                "critic",
                "critic_feedback",
                "retrosyn",
                "supply",
                "srb",
                "srb_execution",
            ):
                next_state[key] = None
            request = next_state.get("request")
            if isinstance(request, Mapping) and request.get("external_evidence"):
                next_state["request"] = {
                    key: value
                    for key, value in request.items()
                    if key != "external_evidence"
                }
        if self.clients is not None and hasattr(self.clients, "generate_candidates"):
            result = await _maybe_await(self.clients.generate_candidates(next_state))
            if not isinstance(result, list):
                raise RuntimeError("generate_candidates must return a list")
            next_state["candidates"] = ensure_candidate_identities(result)
        return next_state

    async def _validating(self, state: WorkflowState) -> WorkflowState:
        next_state = self._with_status(state, "VALIDATING")
        if self.clients is not None and hasattr(self.clients, "validate_candidates"):
            result = await _maybe_await(self.clients.validate_candidates(next_state))
            if not isinstance(result, dict):
                raise RuntimeError("validate_candidates must return a dict")
            next_state["validation"] = result
            next_state.pop("external_evidence_resume_verified", None)
            outcome = _validation_outcome(result, strict=True)
            next_state["validation_outcome"] = outcome
            next_state["validation_passed"] = outcome == "PASS"
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
            verdict = str(result.get("verdict") or "").lower()
            if verdict not in {"pass", "fail"}:
                raise RuntimeError("full workflow critic verdict must be pass or fail")
            if verdict == "fail":
                submit_feedback = getattr(self.clients, "submit_critic_feedback", None)
                if not callable(submit_feedback):
                    raise RuntimeError(
                        "full workflow clients must expose submit_critic_feedback(state)"
                    )
                acknowledgement = await _maybe_await(submit_feedback(next_state))
                if not isinstance(acknowledgement, dict):
                    raise RuntimeError("submit_critic_feedback must return a dict")
                require_feedback_acknowledgement(
                    acknowledgement,
                    expected_groups=1,
                )
                next_state["critic_feedback"] = acknowledgement
        return next_state

    async def _executing(self, state: WorkflowState) -> WorkflowState:
        next_state = self._with_status(state, "EXECUTING")
        execute_synthesis = getattr(self.clients, "execute_synthesis", None)
        if not callable(execute_synthesis):
            raise RuntimeError("full workflow clients must expose execute_synthesis(state)")
        result = await _maybe_await(execute_synthesis(next_state))
        if not isinstance(result, dict):
            raise RuntimeError("execute_synthesis must return a dict")
        next_state["srb_execution"] = result
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

    async def _validation_error(self, state: WorkflowState) -> WorkflowState:
        return self._with_status(state, "ERROR")

    async def _awaiting_evidence(self, state: WorkflowState) -> WorkflowState:
        return self._with_status(state, "AWAITING_EVIDENCE")

    def _route_after_planning(self, state: WorkflowState) -> str:
        return "escalate" if state.get("invalid_policy") else "generate"

    def _route_after_validation(self, state: WorkflowState) -> str:
        validation = state.get("validation")
        outcome = _validation_outcome(
            validation if isinstance(validation, dict) else {},
            strict=False,
        )
        if outcome == "PASS":
            return "done"
        if outcome == "AWAITING_EVIDENCE":
            return "await"
        if outcome == "ERROR":
            return "error"
        if int(state.get("refinement_count", 0)) < int(state.get("max_refinements", 1)):
            return "refine"
        return "escalate"

    def _route_after_critic(self, state: WorkflowState) -> str:
        critic = state.get("critic")
        if not isinstance(critic, dict):
            return "done"
        verdict = str(critic.get("verdict") or "").lower()
        if verdict == "pass":
            if not _selected_protocol_ready(state):
                raise RuntimeError(
                    "full workflow critic cannot pass without an executable "
                    "selected-route protocol"
                )
            return "execute"
        if verdict != "fail":
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


def validate_full_workflow_policies(
    request: Mapping[str, object],
) -> dict[str, object]:
    _validate_full_workflow_context(request)
    validation_policy = _validate_validation_policy(
        _required_policy_object(request, "validation_policy")
    )
    teacher_policy = _validate_teacher_policy(_required_policy_object(request, "teacher_policy"))
    selection_policy = _validate_selection_policy(
        _required_policy_object(request, "selection_policy")
    )
    _validate_selection_threshold_alignment(
        validation_policy,
        selection_policy,
    )
    return {
        "validation_policy": validation_policy,
        "teacher_policy": teacher_policy,
        "selection_policy": selection_policy,
    }


def _validate_full_workflow_context(request: Mapping[str, object]) -> None:
    project_id = request.get("project_id")
    if not isinstance(project_id, str) or not project_id.strip():
        raise ValueError("project_id is required and must be a non-empty string for full workflow")
    if "external_evidence" not in request:
        return
    external_evidence = request["external_evidence"]
    if not isinstance(external_evidence, list):
        raise ValueError("external_evidence must be a list of objects")
    if external_evidence:
        raise ValueError(
            "external_evidence is accepted only by the evidence resume endpoint"
        )


def _required_policy_object(
    request: Mapping[str, object],
    field: str,
) -> Mapping[str, object]:
    value = request.get(field)
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} is required and must be an object for full workflow")
    return value


def _validate_validation_policy(
    value: Mapping[str, object],
) -> dict[str, object]:
    required = {
        "oracle_level",
        "batch_size",
        "max_concurrency",
        "thresholds",
        "oracle_inputs",
    }
    _require_exact_fields(value, required, "validation_policy")
    oracle_level = _bounded_integer(
        value["oracle_level"],
        "validation_policy.oracle_level",
        minimum=0,
        maximum=4,
    )
    batch_size = _bounded_integer(
        value["batch_size"],
        "validation_policy.batch_size",
        minimum=1,
    )
    max_concurrency = _bounded_integer(
        value["max_concurrency"],
        "validation_policy.max_concurrency",
        minimum=1,
    )
    raw_thresholds = value["thresholds"]
    if not isinstance(raw_thresholds, list):
        raise ValueError("validation_policy.thresholds must be a list")
    thresholds: list[dict[str, object]] = []
    threshold_identities: set[tuple[int, str, str]] = set()
    allowed_oracles = {
        0: {"rdkit"},
        1: {"admet", "boltz2"},
        2: {"dock"},
        3: {"fep"},
        4: {"external"},
    }
    fixed_oracle_metrics = {
        "boltz2": {"affinity"},
        "dock": {"docking_score"},
        "fep": {"rbfe"},
    }
    for index, raw_threshold in enumerate(raw_thresholds):
        field = f"validation_policy.thresholds[{index}]"
        if not isinstance(raw_threshold, Mapping):
            raise ValueError(f"{field} must be an object")
        required_threshold_fields = {
            "level",
            "oracle",
            "metric",
            "direction",
            "value",
        }
        _require_fields(
            raw_threshold,
            required_threshold_fields,
            required_threshold_fields | {"max_uncertainty"},
            field,
        )
        level = _bounded_integer(
            raw_threshold["level"],
            f"{field}.level",
            minimum=0,
            maximum=oracle_level,
        )
        oracle = _non_empty_text(raw_threshold["oracle"], f"{field}.oracle")
        if oracle not in allowed_oracles[level]:
            raise ValueError(f"{field}.oracle is invalid for level {level}: {oracle}")
        metric = _non_empty_text(raw_threshold["metric"], f"{field}.metric")
        if oracle == "rdkit" and metric not in {
            "qed",
            "sa_score",
            "logp",
            "lipinski_violations",
            "admet_score",
        }:
            raise ValueError(f"{field}.metric is unsupported for rdkit")
        if oracle in fixed_oracle_metrics and metric not in fixed_oracle_metrics[oracle]:
            raise ValueError(f"{field}.metric is unsupported for {oracle}")
        direction = _non_empty_text(
            raw_threshold["direction"],
            f"{field}.direction",
        )
        if direction not in {"maximize", "minimize"}:
            raise ValueError(f"{field}.direction must be maximize or minimize")
        threshold = {
            "level": level,
            "oracle": oracle,
            "metric": metric,
            "direction": direction,
            "value": _finite_number(
                raw_threshold["value"],
                f"{field}.value",
            ),
        }
        if "max_uncertainty" in raw_threshold:
            max_uncertainty = _finite_number(
                raw_threshold["max_uncertainty"],
                f"{field}.max_uncertainty",
            )
            if max_uncertainty < 0:
                raise ValueError(f"{field}.max_uncertainty must be non-negative")
            threshold["max_uncertainty"] = max_uncertainty
        identity = (level, oracle, metric)
        if identity in threshold_identities:
            raise ValueError("validation_policy.thresholds identities must be unique")
        threshold_identities.add(identity)
        thresholds.append(threshold)

    required_oracle_by_level = {
        0: "rdkit",
        1: "admet",
        2: "dock",
        3: "fep",
        4: "external",
    }
    configured_oracles = {(level, oracle) for level, oracle, _metric in threshold_identities}
    for level in range(oracle_level + 1):
        oracle = required_oracle_by_level[level]
        if (level, oracle) not in configured_oracles:
            raise ValueError(f"validation_policy.thresholds requires {oracle} at level {level}")

    oracle_inputs = _validate_oracle_inputs(
        value["oracle_inputs"],
        oracle_level=oracle_level,
        configured_oracles=configured_oracles,
    )
    return {
        "oracle_level": oracle_level,
        "batch_size": batch_size,
        "max_concurrency": max_concurrency,
        "thresholds": thresholds,
        "oracle_inputs": oracle_inputs,
    }


def _validate_oracle_inputs(
    value: object,
    *,
    oracle_level: int,
    configured_oracles: set[tuple[int, str]],
) -> dict[str, dict[str, object]]:
    if not isinstance(value, Mapping):
        raise ValueError("validation_policy.oracle_inputs must be an object")
    unknown = set(value) - {"boltz2", "dock", "fep"}
    if unknown:
        raise ValueError(
            "validation_policy.oracle_inputs contains unsupported oracle: "
            f"{sorted(str(item) for item in unknown)[0]}"
        )
    result: dict[str, dict[str, object]] = {}
    for oracle, raw_inputs in value.items():
        field = f"validation_policy.oracle_inputs.{oracle}"
        if not isinstance(raw_inputs, Mapping):
            raise ValueError(f"{field} must be an object")
        if oracle == "boltz2":
            result[oracle] = _validate_boltz2_inputs(raw_inputs, field)
        elif oracle == "dock":
            result[oracle] = _validate_dock_inputs(raw_inputs, field)
        else:
            result[oracle] = _validate_fep_inputs(raw_inputs, field)

    if (1, "boltz2") in configured_oracles and "boltz2" not in result:
        raise ValueError("validation_policy.oracle_inputs.boltz2.protein_pdb_id is required")
    if oracle_level >= 2 and "dock" not in result:
        raise ValueError("validation_policy.oracle_inputs.dock.receptor_uri is required")
    if oracle_level >= 3 and "fep" not in result:
        raise ValueError("validation_policy.oracle_inputs.fep.protein_pdb_id is required")
    return result


def _validate_boltz2_inputs(
    value: Mapping[str, object],
    field: str,
) -> dict[str, object]:
    _require_fields(
        value,
        {"protein_pdb_id"},
        {"protein_pdb_id", "oracle_parameters"},
        field,
    )
    result: dict[str, object] = {
        "protein_pdb_id": _non_empty_text(
            value["protein_pdb_id"],
            f"{field}.protein_pdb_id",
        )
    }
    if "oracle_parameters" in value:
        parameters = value["oracle_parameters"]
        if not isinstance(parameters, Mapping):
            raise ValueError(f"{field}.oracle_parameters must be an object")
        _require_exact_fields(
            parameters,
            {"ensemble_size"},
            f"{field}.oracle_parameters",
        )
        result["oracle_parameters"] = {
            "ensemble_size": _bounded_integer(
                parameters["ensemble_size"],
                f"{field}.oracle_parameters.ensemble_size",
                minimum=1,
            )
        }
    return result


def _validate_dock_inputs(
    value: Mapping[str, object],
    field: str,
) -> dict[str, object]:
    _require_exact_fields(
        value,
        {"receptor_uri", "oracle_parameters"},
        field,
    )
    parameters = value["oracle_parameters"]
    if not isinstance(parameters, Mapping):
        raise ValueError(f"{field}.oracle_parameters must be an object")
    _require_exact_fields(
        parameters,
        {"engine"},
        f"{field}.oracle_parameters",
    )
    engine = _non_empty_text(
        parameters["engine"],
        f"{field}.oracle_parameters.engine",
    )
    if engine not in {"gnina", "diffdock"}:
        raise ValueError(f"{field}.oracle_parameters.engine must be gnina or diffdock")
    return {
        "receptor_uri": _non_empty_text(
            value["receptor_uri"],
            f"{field}.receptor_uri",
        ),
        "oracle_parameters": {"engine": engine},
    }


def _validate_fep_inputs(
    value: Mapping[str, object],
    field: str,
) -> dict[str, object]:
    _require_exact_fields(
        value,
        {
            "protein_pdb_id",
            "reference_ligand_smiles",
            "oracle_parameters",
        },
        field,
    )
    parameters = value["oracle_parameters"]
    if not isinstance(parameters, Mapping):
        raise ValueError(f"{field}.oracle_parameters must be an object")
    _require_exact_fields(
        parameters,
        {"method", "n_repeats"},
        f"{field}.oracle_parameters",
    )
    return {
        "protein_pdb_id": _non_empty_text(
            value["protein_pdb_id"],
            f"{field}.protein_pdb_id",
        ),
        "reference_ligand_smiles": _non_empty_text(
            value["reference_ligand_smiles"],
            f"{field}.reference_ligand_smiles",
        ),
        "oracle_parameters": {
            "method": _non_empty_text(
                parameters["method"],
                f"{field}.oracle_parameters.method",
            ),
            "n_repeats": _bounded_integer(
                parameters["n_repeats"],
                f"{field}.oracle_parameters.n_repeats",
                minimum=1,
            ),
        },
    }


def _validate_teacher_policy(
    value: Mapping[str, object],
) -> dict[str, object]:
    required = {"teacher_source", "teacher_version", "allow_synthetic", "kd_weight"}
    _require_exact_fields(value, required, "teacher_policy")
    allow_synthetic = value["allow_synthetic"]
    if not isinstance(allow_synthetic, bool):
        raise ValueError("teacher_policy.allow_synthetic must be a boolean")
    teacher_source = _non_empty_text(
        value["teacher_source"],
        "teacher_policy.teacher_source",
    )
    if teacher_source != "hypseek":
        raise ValueError("teacher_policy.teacher_source must be hypseek")
    policy: dict[str, object] = {
        "teacher_source": teacher_source,
        "teacher_version": _non_empty_text(
            value["teacher_version"],
            "teacher_policy.teacher_version",
        ),
        "allow_synthetic": allow_synthetic,
    }
    kd_weight = _finite_number(
        value["kd_weight"],
        "teacher_policy.kd_weight",
    )
    if kd_weight <= 0.0:
        raise ValueError("teacher_policy.kd_weight must be positive")
    policy["kd_weight"] = kd_weight
    return policy


def _validate_selection_policy(
    value: Mapping[str, object],
) -> dict[str, object]:
    _require_exact_fields(value, {"criteria"}, "selection_policy")
    raw_criteria = value["criteria"]
    if not isinstance(raw_criteria, list) or not raw_criteria:
        raise ValueError("selection_policy.criteria must be a non-empty list")
    criteria: list[dict[str, str]] = []
    metrics: set[str] = set()
    for index, raw_criterion in enumerate(raw_criteria):
        field = f"selection_policy.criteria[{index}]"
        if not isinstance(raw_criterion, Mapping):
            raise ValueError(f"{field} must be an object")
        _require_exact_fields(
            raw_criterion,
            {"metric", "direction"},
            field,
        )
        metric = _non_empty_text(
            raw_criterion["metric"],
            f"{field}.metric",
        )
        metric_identity = metric.lower()
        if metric_identity in metrics:
            raise ValueError("selection_policy criteria metrics must be unique")
        metrics.add(metric_identity)
        direction = _non_empty_text(
            raw_criterion["direction"],
            f"{field}.direction",
        )
        if direction not in {"maximize", "minimize"}:
            raise ValueError(f"{field}.direction must be maximize or minimize")
        criteria.append({"metric": metric, "direction": direction})
    return {"criteria": criteria}


def _validate_selection_threshold_alignment(
    validation_policy: Mapping[str, object],
    selection_policy: Mapping[str, object],
) -> None:
    thresholds = validation_policy["thresholds"]
    criteria = selection_policy["criteria"]
    if not isinstance(thresholds, list) or not isinstance(criteria, list):
        raise RuntimeError("validated full-workflow policies are malformed")
    for index, criterion in enumerate(criteria):
        if not isinstance(criterion, Mapping):
            raise RuntimeError("validated selection policy is malformed")
        metric = criterion["metric"]
        matches = [
            threshold
            for threshold in thresholds
            if isinstance(threshold, Mapping) and threshold.get("metric") == metric
        ]
        if len(matches) != 1:
            raise ValueError(
                f"selection_policy.criteria[{index}].metric must match exactly one "
                "validation_policy threshold"
            )
        if matches[0].get("direction") != criterion["direction"]:
            raise ValueError(
                f"selection_policy.criteria[{index}].direction must match its "
                "validation_policy threshold direction"
            )


def _require_exact_fields(
    value: Mapping[str, object],
    required: set[str],
    field: str,
) -> None:
    _require_fields(value, required, required, field)


def _require_fields(
    value: Mapping[str, object],
    required: set[str],
    allowed: set[str],
    field: str,
) -> None:
    missing = required - set(value)
    if missing:
        raise ValueError(f"{field}.{sorted(missing)[0]} is required")
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"{field}.{sorted(str(item) for item in unknown)[0]} is unsupported")


def _bounded_integer(
    value: object,
    field: str,
    *,
    minimum: int,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    if value < minimum or (maximum is not None and value > maximum):
        if maximum is None:
            raise ValueError(f"{field} must be at least {minimum}")
        raise ValueError(f"{field} must be between {minimum} and {maximum}")
    return value


def _finite_number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _non_empty_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def generation_controls(
    request: Mapping[str, object],
) -> tuple[int, dict[str, object]]:
    raw_generator_params = request.get("generator_params", {})
    if not isinstance(raw_generator_params, Mapping):
        raise ValueError("generator_params must be an object")
    generator_params = dict(raw_generator_params)

    for field in ("n_samples", "batch_size"):
        if field in request:
            _bounded_integer(
                request[field],
                field,
                minimum=1,
            )
    if "n_samples" in request:
        n_samples = int(request["n_samples"])
    elif "batch_size" in request:
        n_samples = int(request["batch_size"])
    else:
        n_samples = 4

    for field in ("sampling_seed", "seed"):
        if field in request:
            _bounded_integer(
                request[field],
                field,
                minimum=0,
            )
    if "sampling_seed" in generator_params:
        _bounded_integer(
            generator_params["sampling_seed"],
            "generator_params.sampling_seed",
            minimum=0,
        )
    if "sampling_seed" in request:
        sampling_seed = int(request["sampling_seed"])
    elif "seed" in request:
        sampling_seed = int(request["seed"])
    elif "sampling_seed" in generator_params:
        sampling_seed = int(generator_params["sampling_seed"])
    else:
        sampling_seed = 42
    generator_params["sampling_seed"] = sampling_seed
    return n_samples, generator_params


def ensure_candidate_identities(candidates: list[dict]) -> list[dict]:
    if not all(isinstance(candidate, dict) for candidate in candidates):
        raise RuntimeError("generated candidates must be objects")
    rows = [dict(candidate) for candidate in candidates]
    provided_ids: list[str] = []
    for candidate in rows:
        candidate_id = candidate.get("candidate_id")
        if candidate_id in (None, ""):
            continue
        if (
            not isinstance(candidate_id, str)
            or not candidate_id.strip()
            or candidate_id != candidate_id.strip()
        ):
            raise RuntimeError("generated candidate_id must be a non-empty trimmed string")
        provided_ids.append(candidate_id)
    if len(provided_ids) != len(set(provided_ids)):
        raise RuntimeError("generated candidate_id values must be unique")
    used_ids = set(provided_ids)
    occurrences: dict[str, int] = {}
    for candidate in rows:
        candidate_id = candidate.get("candidate_id")
        if candidate_id not in (None, ""):
            continue
        try:
            identity = json.dumps(
                candidate,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        except (TypeError, ValueError) as exc:
            raise RuntimeError("generated candidate must be canonical JSON") from exc
        fingerprint = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        occurrence = occurrences.get(fingerprint, 0)
        while True:
            digest = hashlib.sha256(f"{fingerprint}:{occurrence}".encode("ascii")).hexdigest()
            generated_id = f"candidate-{digest[:24]}"
            occurrence += 1
            if generated_id not in used_ids:
                break
        occurrences[fingerprint] = occurrence
        used_ids.add(generated_id)
        candidate["candidate_id"] = generated_id
    return rows


def validation_candidate_payload(candidate: Mapping[str, Any]) -> dict[str, str]:
    candidate_id = _native_trimmed_string(
        candidate.get("candidate_id"),
        "candidate_id",
    )
    generator_name = candidate.get("generator_name") or candidate.get("generator")
    if not isinstance(generator_name, str) or not generator_name:
        raise RuntimeError("candidate generator_name is required for batch validation")
    return {
        "candidate_id": candidate_id,
        "canonical_smiles": _candidate_canonical_smiles(candidate),
        "generator_name": generator_name,
    }


def validation_feedback_groups(
    candidates: list[dict],
    records: list[dict],
    *,
    teacher_policy: Mapping[str, object],
    validation_policy: Mapping[str, object],
) -> list[dict]:
    require_validation_records_contract(records, candidates, validation_policy)
    candidate_by_identity: dict[tuple[str, str], dict] = {}
    for candidate in candidates:
        candidate_id = _native_trimmed_string(
            candidate.get("candidate_id"),
            "generated candidate candidate_id",
        )
        canonical_smiles = _candidate_canonical_smiles(candidate)
        identity = (candidate_id, canonical_smiles)
        if identity in candidate_by_identity:
            raise RuntimeError("validation feedback requires unique candidate identities")
        candidate_by_identity[identity] = candidate

    grouped: dict[tuple[str, str], dict] = {}
    matched_candidates: set[tuple[str, str]] = set()
    for record in records:
        candidate_id = _native_trimmed_string(
            record.get("candidate_id"),
            "ValidationAgent record candidate_id",
        )
        canonical_smiles = _native_trimmed_string(
            record.get("canonical_smiles"),
            "ValidationAgent record canonical_smiles",
        )
        identity = (candidate_id, canonical_smiles)
        candidate = candidate_by_identity.get(identity)
        if candidate is None or identity in matched_candidates:
            raise RuntimeError("ValidationAgent records must match generated candidates one-to-one")
        matched_candidates.add(identity)
        record_outcome = strict_validation_outcome(record.get("outcome"))
        if record_outcome not in {"PASS", "FAIL"}:
            continue
        generator_name = str(candidate.get("generator_name") or candidate.get("generator") or "")
        if not generator_name:
            raise RuntimeError("validation feedback requires generator_name")
        evidence_ids = validation_record_evidence_ids(record)
        if not evidence_ids:
            raise RuntimeError("validation feedback requires evidence_ids")
        group_key = (generator_name, canonical_smiles)
        group = grouped.setdefault(
            group_key,
            {
                "phase": "validation",
                "generator_name": generator_name,
                "canonical_smiles": canonical_smiles,
                "candidate_ids": [],
                "evidence_ids": [],
                "records": [],
                "teacher_policy": dict(teacher_policy),
            },
        )
        group["candidate_ids"].append(candidate_id)
        passed = record_outcome == "PASS"
        if "passed" in record and record.get("passed") is not passed:
            raise RuntimeError("ValidationAgent record passed does not match outcome")
        group["records"].append({**dict(record), "passed": passed})
        for evidence_id in evidence_ids:
            if evidence_id not in group["evidence_ids"]:
                group["evidence_ids"].append(evidence_id)
    if matched_candidates != set(candidate_by_identity):
        raise RuntimeError("ValidationAgent response is missing candidate records")
    return list(grouped.values())


def validation_record_evidence_ids(record: Mapping[str, object]) -> list[str]:
    evidence_ids: list[str] = []
    raw_evidence_ids = record.get("evidence_ids")
    if raw_evidence_ids is not None:
        if not isinstance(raw_evidence_ids, list):
            raise RuntimeError("ValidationAgent record evidence_ids must be a list")
        for value in raw_evidence_ids:
            evidence_id = _native_trimmed_string(
                value,
                "ValidationAgent record evidence_id",
            )
            if evidence_id not in evidence_ids:
                evidence_ids.append(evidence_id)
    evidence = record.get("evidence")
    if evidence is not None:
        if not isinstance(evidence, list):
            raise RuntimeError("ValidationAgent record evidence must be a list")
        for item in evidence:
            if not isinstance(item, Mapping):
                raise RuntimeError("ValidationAgent record evidence entries must be objects")
            evidence_id = _native_trimmed_string(
                item.get("evidence_id"),
                "ValidationAgent record evidence_id",
            )
            if evidence_id not in evidence_ids:
                evidence_ids.append(evidence_id)
    return evidence_ids


def full_workflow_candidate_identity(
    state: Mapping[str, Any],
) -> dict[str, object]:
    candidates = _workflow_candidate_objects(state)
    request = state.get("request")
    validation = state.get("validation")
    selection_policy = request.get("selection_policy") if isinstance(request, Mapping) else None
    if not isinstance(validation, Mapping):
        raise RuntimeError("full workflow requires validation records")
    if not isinstance(selection_policy, Mapping):
        raise RuntimeError("selection_policy is required for full workflow")
    candidate, _record, candidate_index = select_full_candidate(
        candidates,
        validation,
        selection_policy,
        validation_policy=request.get("validation_policy"),
    )
    if not isinstance(request, Mapping):
        raise RuntimeError("full workflow request is required")
    return {
        "project_id": _native_trimmed_string(request.get("project_id"), "project_id"),
        "candidate_id": _native_trimmed_string(
            candidate.get("candidate_id"),
            "candidate_id",
        ),
        "candidate_index": candidate_index,
        "canonical_smiles": _candidate_canonical_smiles(candidate),
    }


def require_full_downstream_response_identity(
    response: Mapping[str, object],
    state: Mapping[str, Any],
    *,
    stage: str,
) -> dict[str, object]:
    if not isinstance(response, Mapping):
        raise RuntimeError(f"{stage} response must be an object")
    expected = full_workflow_candidate_identity(state)
    for field, expected_value in expected.items():
        actual_value = response.get(field)
        if type(actual_value) is not type(expected_value) or actual_value != expected_value:
            raise RuntimeError(f"{stage} response {field} must match selected candidate identity")
    return dict(response)


def critic_feedback_groups(
    state: Mapping[str, Any],
    critic: Mapping[str, object],
) -> list[dict]:
    identity = full_workflow_candidate_identity(state)
    candidates = _workflow_candidate_objects(state)
    candidate = candidates[int(identity["candidate_index"])]
    generator_name = candidate.get("generator_name") or candidate.get("generator")
    if not isinstance(generator_name, str) or not generator_name:
        raise RuntimeError("critic feedback requires generator_name")
    rule_results = critic.get("rule_results")
    if not isinstance(rule_results, list) or not rule_results:
        raise RuntimeError("failed critic response requires rule_results")
    evidence_ids = []
    for result in rule_results:
        if not isinstance(result, Mapping):
            raise RuntimeError("critic rule_results must contain objects")
        rule_id = _native_trimmed_string(
            result.get("rule_id"),
            "critic rule_result rule_id",
        )
        if rule_id not in evidence_ids:
            evidence_ids.append(rule_id)
    request = state.get("request")
    teacher_policy = request.get("teacher_policy") if isinstance(request, Mapping) else None
    if not isinstance(teacher_policy, Mapping):
        raise RuntimeError("teacher_policy is required for critic feedback")
    return [
        {
            "phase": "critic",
            "generator_name": generator_name,
            "canonical_smiles": identity["canonical_smiles"],
            "candidate_ids": [identity["candidate_id"]],
            "evidence_ids": evidence_ids,
            "records": [dict(critic)],
            "teacher_policy": dict(teacher_policy),
        }
    ]


def require_feedback_acknowledgement(
    feedback: Mapping[str, object],
    *,
    expected_groups: int,
) -> None:
    submitted = feedback.get("submitted")
    duplicates = feedback.get("duplicates")
    if (
        feedback.get("action") == "generator_coord/feedback/v1"
        and feedback.get("status") == "feedback_submitted"
        and isinstance(expected_groups, int)
        and not isinstance(expected_groups, bool)
        and expected_groups >= 0
        and isinstance(submitted, int)
        and not isinstance(submitted, bool)
        and submitted >= 0
        and isinstance(duplicates, int)
        and not isinstance(duplicates, bool)
        and duplicates >= 0
        and submitted + duplicates == expected_groups
    ):
        return
    raise RuntimeError("GeneratorCoord did not acknowledge validation feedback")


def validation_records_outcome(records: list[Mapping[str, object]]) -> str:
    if not records:
        raise RuntimeError("ValidationAgent batch response must contain candidate records")
    outcomes = [strict_validation_outcome(record.get("outcome")) for record in records]
    if "ERROR" in outcomes:
        return "ERROR"
    if "PASS" in outcomes:
        return "PASS"
    if "AWAITING_EVIDENCE" in outcomes:
        return "AWAITING_EVIDENCE"
    return "FAIL"


def require_validation_batch_consistency(
    outcome: str,
    records: list[Mapping[str, object]],
) -> None:
    aggregate = validation_records_outcome(records)
    if outcome != aggregate:
        raise RuntimeError("ValidationAgent batch outcome does not match record aggregation")


def require_validation_records_contract(
    records: list[Mapping[str, object]],
    candidates: list[Mapping[str, object]],
    validation_policy: Mapping[str, object],
) -> None:
    candidate_identities = {
        (
            _native_trimmed_string(candidate.get("candidate_id"), "candidate candidate_id"),
            _candidate_canonical_smiles(candidate),
        )
        for candidate in candidates
    }
    if len(candidate_identities) != len(candidates):
        raise RuntimeError("candidate identities must be unique for validation")
    if len(records) != len(candidates):
        raise RuntimeError("ValidationAgent records must match candidates one-to-one")

    oracle_level = validation_policy.get("oracle_level")
    thresholds = validation_policy.get("thresholds")
    if (
        isinstance(oracle_level, bool)
        or not isinstance(oracle_level, int)
        or oracle_level not in range(5)
        or not isinstance(thresholds, list)
        or not all(isinstance(threshold, Mapping) for threshold in thresholds)
    ):
        raise RuntimeError("validation_policy is invalid for record validation")
    thresholds_by_oracle: dict[tuple[int, str], list[Mapping[str, object]]] = {}
    for threshold in thresholds:
        level = threshold.get("level")
        oracle = threshold.get("oracle")
        if isinstance(level, bool) or not isinstance(level, int) or level not in range(5):
            raise RuntimeError("validation_policy threshold level is invalid")
        oracle_name = _native_trimmed_string(oracle, "validation_policy threshold oracle")
        thresholds_by_oracle.setdefault((level, oracle_name), []).append(threshold)

    seen_identities: set[tuple[str, str]] = set()
    for record in records:
        if not {
            "schema_version",
            "candidate_id",
            "canonical_smiles",
            "outcome",
            "metrics",
            "evidence",
            "levels",
        }.issubset(record):
            raise RuntimeError("ValidationAgent record schema fields do not match")
        if record.get("schema_version") != "validation.record.v1":
            raise RuntimeError("ValidationAgent record schema_version does not match")
        identity = (
            _native_trimmed_string(
                record.get("candidate_id"),
                "ValidationAgent record candidate_id",
            ),
            _native_trimmed_string(
                record.get("canonical_smiles"),
                "ValidationAgent record canonical_smiles",
            ),
        )
        if identity not in candidate_identities or identity in seen_identities:
            raise RuntimeError("ValidationAgent record candidate binding does not match")
        seen_identities.add(identity)
        _require_validation_record_contract(
            record,
            oracle_level=oracle_level,
            thresholds_by_oracle=thresholds_by_oracle,
        )
    if seen_identities != candidate_identities:
        raise RuntimeError("ValidationAgent records must match candidates one-to-one")


def _require_validation_record_contract(
    record: Mapping[str, object],
    *,
    oracle_level: int,
    thresholds_by_oracle: Mapping[tuple[int, str], list[Mapping[str, object]]],
) -> None:
    record_outcome = strict_validation_outcome(record.get("outcome"))
    levels = record.get("levels")
    if not isinstance(levels, list) or not levels or not all(
        isinstance(level_record, Mapping) for level_record in levels
    ):
        raise RuntimeError("ValidationAgent record levels must be a non-empty list")
    level_numbers = [level_record.get("level") for level_record in levels]
    if (
        any(isinstance(level, bool) or not isinstance(level, int) for level in level_numbers)
        or level_numbers != list(range(len(levels)))
        or level_numbers[-1] > oracle_level
    ):
        raise RuntimeError("ValidationAgent record levels must be continuous from L0")
    if record_outcome == "PASS" and level_numbers[-1] != oracle_level:
        raise RuntimeError("PASS validation records must reach the configured oracle_level")
    if record_outcome == "AWAITING_EVIDENCE" and (
        oracle_level != 4 or level_numbers[-1] != 4
    ):
        raise RuntimeError("AWAITING_EVIDENCE validation records must stop at L4")

    flattened_metrics: list[dict] = []
    oracle_evidence: dict[str, tuple[int, str]] = {}
    for level_index, level_record in enumerate(levels):
        if set(level_record) != {"level", "outcome", "oracles"}:
            raise RuntimeError("ValidationAgent level schema fields do not match")
        level_outcome = strict_validation_outcome(level_record.get("outcome"))
        if level_index < len(levels) - 1 and level_outcome != "PASS":
            raise RuntimeError("only the final validation level may be non-PASS")
        raw_oracles = level_record.get("oracles")
        if not isinstance(raw_oracles, list) or not raw_oracles or not all(
            isinstance(oracle_record, Mapping) for oracle_record in raw_oracles
        ):
            raise RuntimeError("ValidationAgent level oracles must be a non-empty list")
        expected_oracles = {
            oracle
            for level, oracle in thresholds_by_oracle
            if level == level_index
        }
        actual_oracles = [
            _native_trimmed_string(
                oracle_record.get("oracle"),
                "ValidationAgent level oracle",
            )
            for oracle_record in raw_oracles
        ]
        if len(actual_oracles) != len(set(actual_oracles)) or set(actual_oracles) != (
            expected_oracles
        ):
            raise RuntimeError("ValidationAgent level required oracle set does not match policy")

        oracle_outcomes = []
        for oracle_record, oracle_name in zip(raw_oracles, actual_oracles, strict=True):
            oracle_outcome = strict_validation_outcome(oracle_record.get("outcome"))
            oracle_outcomes.append(oracle_outcome)
            oracle_metrics = _require_validation_oracle_record(
                oracle_record,
                level=level_index,
                oracle_name=oracle_name,
                outcome=oracle_outcome,
                thresholds=thresholds_by_oracle[(level_index, oracle_name)],
            )
            flattened_metrics.extend(oracle_metrics)
            for evidence_id in oracle_record["evidence_ids"]:
                if evidence_id in oracle_evidence:
                    raise RuntimeError("ValidationAgent oracle evidence_ids must be unique")
                oracle_evidence[evidence_id] = (level_index, oracle_name)
        if _validation_level_outcome(oracle_outcomes) != level_outcome:
            raise RuntimeError("ValidationAgent level outcome does not match oracle outcomes")

    if strict_validation_outcome(levels[-1].get("outcome")) != record_outcome:
        raise RuntimeError("ValidationAgent record outcome does not match final level")
    raw_metrics = record.get("metrics")
    if not isinstance(raw_metrics, list) or raw_metrics != flattened_metrics:
        raise RuntimeError("ValidationAgent record metrics do not match level oracle metrics")
    _require_validation_record_evidence(
        record.get("evidence"),
        levels=levels,
        oracle_evidence=oracle_evidence,
    )


def _require_validation_oracle_record(
    oracle_record: Mapping[str, object],
    *,
    level: int,
    oracle_name: str,
    outcome: str,
    thresholds: list[Mapping[str, object]],
) -> list[dict]:
    allowed_fields = {"oracle", "outcome", "metrics", "evidence_ids"}
    if outcome == "ERROR":
        allowed_fields.add("error")
    elif outcome == "AWAITING_EVIDENCE":
        allowed_fields.add("reason")
    elif outcome == "FAIL" and "failure" in oracle_record:
        allowed_fields.add("failure")
    if set(oracle_record) != allowed_fields:
        raise RuntimeError("ValidationAgent oracle schema fields do not match")
    raw_evidence_ids = oracle_record.get("evidence_ids")
    if not isinstance(raw_evidence_ids, list):
        raise RuntimeError("ValidationAgent oracle evidence_ids must be a list")
    evidence_ids = [
        _native_trimmed_string(value, "ValidationAgent oracle evidence_id")
        for value in raw_evidence_ids
    ]
    if len(evidence_ids) != len(set(evidence_ids)):
        raise RuntimeError("ValidationAgent oracle evidence_ids must be unique")

    raw_metrics = oracle_record.get("metrics")
    if not isinstance(raw_metrics, list) or not all(
        isinstance(metric, Mapping) for metric in raw_metrics
    ):
        raise RuntimeError("ValidationAgent oracle metrics must be a list")
    if outcome in {"ERROR", "AWAITING_EVIDENCE"}:
        if raw_metrics:
            raise RuntimeError(f"ValidationAgent {outcome} oracle metrics must be empty")
        diagnostic_field = "error" if outcome == "ERROR" else "reason"
        if not oracle_record.get(diagnostic_field):
            raise RuntimeError(f"ValidationAgent {outcome} oracle requires {diagnostic_field}")
        return []
    if outcome == "FAIL" and "failure" in oracle_record:
        failure = oracle_record["failure"]
        if not isinstance(failure, Mapping):
            raise RuntimeError("ValidationAgent failure diagnostic is malformed")
        failure_code = failure.get("code")
        failure_message = failure.get("message")
        if not isinstance(failure_message, str) or not failure_message:
            raise RuntimeError("ValidationAgent failure diagnostic is malformed")
        if failure_code == "INVALID_SMILES":
            if level != 0 or oracle_name != "rdkit" or raw_metrics:
                raise RuntimeError("ValidationAgent invalid candidate failure is malformed")
            return []
        if failure_code != "ORACLE_OUTCOME_FAIL" or not raw_metrics:
            raise RuntimeError("ValidationAgent failure diagnostic is malformed")
    if len(raw_metrics) != len(thresholds):
        raise RuntimeError("ValidationAgent oracle metrics do not match policy thresholds")
    metrics = []
    for raw_metric, threshold in zip(raw_metrics, thresholds, strict=True):
        metrics.append(
            _require_validation_metric(
                raw_metric,
                threshold,
                level=level,
                oracle_name=oracle_name,
            )
        )
    if outcome == "PASS" and not all(metric["passed"] for metric in metrics):
        raise RuntimeError("ValidationAgent PASS oracle contains a failed metric")
    if outcome == "FAIL":
        failure = oracle_record.get("failure")
        if failure is None and all(metric["passed"] for metric in metrics):
            raise RuntimeError("ValidationAgent FAIL oracle requires a failed metric")
        if failure is not None and not all(metric["passed"] for metric in metrics):
            raise RuntimeError(
                "ValidationAgent ORACLE_OUTCOME_FAIL diagnostic requires passing metrics"
            )
    return metrics


def _require_validation_metric(
    raw_metric: Mapping[str, object],
    threshold: Mapping[str, object],
    *,
    level: int,
    oracle_name: str,
) -> dict:
    expected_fields = {
        "level",
        "oracle",
        "metric",
        "value",
        "direction",
        "threshold",
        "passed",
    }
    if "max_uncertainty" in threshold:
        expected_fields.update({"uncertainty", "max_uncertainty"})
    if set(raw_metric) != expected_fields:
        raise RuntimeError("ValidationAgent metric schema fields do not match")
    if raw_metric.get("level") != level or raw_metric.get("oracle") != oracle_name:
        raise RuntimeError("ValidationAgent metric level or oracle does not match")
    if raw_metric.get("metric") != threshold.get("metric"):
        raise RuntimeError("ValidationAgent metric identity does not match policy")
    if raw_metric.get("direction") != threshold.get("direction"):
        raise RuntimeError("ValidationAgent metric direction does not match policy")
    if raw_metric.get("threshold") != threshold.get("value"):
        raise RuntimeError("ValidationAgent metric threshold does not match policy")
    value = _runtime_finite_number(raw_metric.get("value"), "ValidationAgent metric value")
    passed = (
        value >= float(threshold["value"])
        if threshold["direction"] == "maximize"
        else value <= float(threshold["value"])
    )
    if "max_uncertainty" in threshold:
        uncertainty = _runtime_finite_number(
            raw_metric.get("uncertainty"),
            "ValidationAgent metric uncertainty",
        )
        if uncertainty < 0:
            raise RuntimeError("ValidationAgent metric uncertainty must be non-negative")
        if raw_metric.get("max_uncertainty") != threshold.get("max_uncertainty"):
            raise RuntimeError("ValidationAgent metric max_uncertainty does not match policy")
        passed = passed and uncertainty <= float(threshold["max_uncertainty"])
    if not isinstance(raw_metric.get("passed"), bool) or raw_metric["passed"] is not passed:
        raise RuntimeError("ValidationAgent metric passed does not match threshold evaluation")
    return dict(raw_metric)


def _require_validation_record_evidence(
    raw_evidence: object,
    *,
    levels: list[Mapping[str, object]],
    oracle_evidence: Mapping[str, tuple[int, str]],
) -> None:
    if not isinstance(raw_evidence, list) or not raw_evidence or not all(
        isinstance(item, Mapping) for item in raw_evidence
    ):
        raise RuntimeError("ValidationAgent record evidence must be a non-empty list")
    seen_ids: set[str] = set()
    referenced_oracle_ids: set[str] = set()
    for item in raw_evidence:
        evidence_id = _native_trimmed_string(
            item.get("evidence_id"),
            "ValidationAgent record evidence_id",
        )
        if evidence_id in seen_ids:
            raise RuntimeError("ValidationAgent record evidence_ids must be unique")
        seen_ids.add(evidence_id)
        level = item.get("level")
        oracle = item.get("oracle")
        if (
            isinstance(level, bool)
            or not isinstance(level, int)
            or level not in range(len(levels))
        ):
            raise RuntimeError("ValidationAgent record evidence level is invalid")
        oracle_name = _native_trimmed_string(oracle, "ValidationAgent record evidence oracle")
        if oracle_name == "validation_agent":
            if level != len(levels) - 1:
                raise RuntimeError("ValidationAgent belief evidence must bind the final level")
            continue
        if oracle_evidence.get(evidence_id) != (level, oracle_name):
            raise RuntimeError("ValidationAgent record evidence does not match oracle evidence")
        referenced_oracle_ids.add(evidence_id)
    if referenced_oracle_ids != set(oracle_evidence):
        raise RuntimeError("ValidationAgent oracle evidence is missing from the record")


def _validation_level_outcome(outcomes: list[str]) -> str:
    if "ERROR" in outcomes:
        return "ERROR"
    if "FAIL" in outcomes:
        return "FAIL"
    if "AWAITING_EVIDENCE" in outcomes:
        return "AWAITING_EVIDENCE"
    return "PASS"


def _runtime_finite_number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise RuntimeError(f"{field} must be finite")
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError(f"{field} must be finite")
    return number


def require_validation_batch_contract(
    response: Mapping[str, object],
    *,
    project_id: str,
    run_id: str,
    request_id: str,
    validation_policy: Mapping[str, object],
    candidates: list[Mapping[str, object]],
) -> tuple[str, list[dict]]:
    expected_text = {
        "validation_schema_version": "validation.batch.v1",
        "agent": "validation_agent",
        "project_id": project_id,
        "run_id": run_id,
        "request_id": request_id,
    }
    for field, expected in expected_text.items():
        actual = response.get(field)
        if (
            not isinstance(actual, str)
            or not actual
            or actual != actual.strip()
            or actual != expected
        ):
            raise RuntimeError(f"ValidationAgent batch response {field} does not match the request")
    echoed_policy = response.get("validation_policy")
    if not isinstance(echoed_policy, Mapping):
        raise RuntimeError(
            "ValidationAgent batch response validation_policy does not match the request"
        )
    try:
        expected_policy_json = json.dumps(
            dict(validation_policy),
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        echoed_policy_json = json.dumps(
            dict(echoed_policy),
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "ValidationAgent batch response validation_policy must be canonical JSON"
        ) from exc
    if echoed_policy_json != expected_policy_json:
        raise RuntimeError(
            "ValidationAgent batch response validation_policy does not match the request"
        )
    outcome = strict_validation_outcome(response.get("outcome"))
    records = response.get("records")
    if not isinstance(records, list) or not all(isinstance(record, Mapping) for record in records):
        raise RuntimeError("ValidationAgent batch response records must be a list of objects")
    raw_records = [dict(record) for record in records]
    require_validation_records_contract(
        raw_records,
        candidates,
        validation_policy,
    )
    require_validation_batch_consistency(outcome, raw_records)
    return outcome, raw_records


def select_full_candidate(
    candidates: list[dict],
    validation: Mapping[str, object],
    selection_policy: Mapping[str, object],
    *,
    validation_policy: Mapping[str, object] | None = None,
) -> tuple[dict, dict, int]:
    if validation.get("outcome") != "PASS":
        raise RuntimeError("full workflow requires a PASS validation batch")
    rows = validation.get("records", validation.get("results"))
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("full workflow requires validation records")
    if validation_policy is None:
        validation_policy = validation.get("validation_policy")
    if not isinstance(validation_policy, Mapping):
        raise RuntimeError("full workflow validation_policy is required for selection")
    if not all(isinstance(row, Mapping) for row in rows):
        raise RuntimeError("full workflow validation records must be objects")
    require_validation_records_contract(rows, candidates, validation_policy)
    require_validation_batch_consistency(
        strict_validation_outcome(validation.get("outcome")),
        rows,
    )
    criteria = selection_policy.get("criteria")
    if not isinstance(criteria, list) or not criteria:
        raise RuntimeError("selection_policy.criteria is required for full workflow")

    candidate_by_identity: dict[tuple[str, str], tuple[dict, int]] = {}
    for candidate_index, candidate in enumerate(candidates):
        candidate_id = _native_trimmed_string(
            candidate.get("candidate_id"),
            "candidate candidate_id",
        )
        canonical_smiles = _candidate_canonical_smiles(candidate)
        identity = (candidate_id, canonical_smiles)
        if identity in candidate_by_identity:
            raise RuntimeError("candidate identities must be unique for selection")
        candidate_by_identity[identity] = (candidate, candidate_index)

    selectable: list[tuple[tuple[object, ...], dict, dict, int]] = []
    for row in rows:
        if not isinstance(row, dict) or row.get("outcome") != "PASS":
            continue
        identity = (
            _native_trimmed_string(
                row.get("candidate_id"),
                "ValidationAgent record candidate_id",
            ),
            _native_trimmed_string(
                row.get("canonical_smiles"),
                "ValidationAgent record canonical_smiles",
            ),
        )
        matched = candidate_by_identity.get(identity)
        if matched is None:
            continue
        values = selection_metric_values(row)
        score_key: list[float] = []
        complete = True
        for criterion in criteria:
            if not isinstance(criterion, Mapping):
                complete = False
                break
            metric = str(criterion.get("metric") or "")
            direction = str(criterion.get("direction") or "").lower()
            value = values.get(metric)
            if value is None or direction not in {"maximize", "minimize"}:
                complete = False
                break
            score_key.append(-value if direction == "maximize" else value)
        if not complete:
            continue
        candidate, candidate_index = matched
        canonical_smiles = identity[1]
        candidate_id = identity[0]
        key: tuple[object, ...] = (*score_key, canonical_smiles, candidate_id)
        selectable.append((key, candidate, row, candidate_index))
    if not selectable:
        raise RuntimeError(
            "full workflow requires a PASS candidate with every finite selection metric"
        )
    _key, candidate, row, candidate_index = min(selectable, key=lambda item: item[0])
    return candidate, row, candidate_index


def selection_metric_values(record: Mapping[str, object]) -> dict[str, float]:
    raw_metrics = record.get("metrics")
    if not isinstance(raw_metrics, list):
        return {}
    values: dict[str, float] = {}
    duplicates: set[str] = set()
    for raw_metric in raw_metrics:
        if not isinstance(raw_metric, Mapping):
            continue
        metric = str(raw_metric.get("metric") or "")
        value = raw_metric.get("value")
        if (
            not metric
            or isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(float(value))
        ):
            continue
        if metric in values:
            duplicates.add(metric)
            continue
        values[metric] = float(value)
    for metric in duplicates:
        values.pop(metric, None)
    return values


def strict_validation_outcome(value: object) -> str:
    if not isinstance(value, str) or value not in {
        "PASS",
        "FAIL",
        "ERROR",
        "AWAITING_EVIDENCE",
    }:
        raise RuntimeError(
            "ValidationAgent batch outcome must be PASS, FAIL, ERROR, or AWAITING_EVIDENCE"
        )
    return value


def _candidate_canonical_smiles(candidate: Mapping[str, Any]) -> str:
    value = candidate.get("canonical_smiles") or candidate.get("smiles")
    return _native_trimmed_string(value, "candidate canonical_smiles")


def _workflow_candidate_objects(state: Mapping[str, Any]) -> list[dict]:
    candidates = state.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise RuntimeError("workflow requires at least one generated candidate")
    if not all(isinstance(candidate, dict) for candidate in candidates):
        raise RuntimeError("workflow candidates must be objects")
    return candidates


def _native_trimmed_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or not value.strip() or value != value.strip():
        raise RuntimeError(f"{field} must be a non-empty trimmed string")
    return value


def policy_direction_conflicts(state: Mapping[str, Any]) -> list[dict[str, str]]:
    cig = state.get("cig")
    request = state.get("request")
    if not isinstance(cig, Mapping) or not isinstance(request, Mapping):
        return []
    raw_objectives = cig.get("objectives")
    if not isinstance(raw_objectives, list):
        raw_objectives = cig.get("objective_nodes")
    if not isinstance(raw_objectives, list):
        return []

    policy_directions: dict[str, list[tuple[str, str]]] = {}
    validation_policy = request.get("validation_policy")
    if isinstance(validation_policy, Mapping):
        thresholds = validation_policy.get("thresholds")
        if isinstance(thresholds, list):
            for threshold in thresholds:
                if not isinstance(threshold, Mapping):
                    continue
                metric = _normalised_metric(threshold.get("metric"))
                direction = _normalised_direction(threshold.get("direction"))
                if metric and direction:
                    policy_directions.setdefault(metric, []).append(
                        ("validation_policy", direction)
                    )
    selection_policy = request.get("selection_policy")
    if isinstance(selection_policy, Mapping):
        criteria = selection_policy.get("criteria")
        if isinstance(criteria, list):
            for criterion in criteria:
                if not isinstance(criterion, Mapping):
                    continue
                metric = _normalised_metric(criterion.get("metric"))
                direction = _normalised_direction(criterion.get("direction"))
                if metric and direction:
                    policy_directions.setdefault(metric, []).append(("selection_policy", direction))

    conflicts: list[dict[str, str]] = []
    for objective in raw_objectives:
        if not isinstance(objective, Mapping):
            continue
        metric_value = objective.get("property") or objective.get("name")
        metric = _normalised_metric(metric_value)
        objective_direction = _objective_direction(objective.get("type"))
        if not metric or objective_direction is None:
            continue
        for source, policy_direction in policy_directions.get(metric, []):
            if policy_direction != objective_direction:
                conflicts.append(
                    {
                        "metric": str(metric_value),
                        "cig_direction": objective_direction,
                        "policy_source": source,
                        "policy_direction": policy_direction,
                    }
                )
    return conflicts


def _validation_outcome(result: Mapping[str, Any], *, strict: bool) -> str:
    raw_outcome = result.get("outcome")
    if isinstance(raw_outcome, str):
        try:
            return strict_validation_outcome(raw_outcome)
        except RuntimeError:
            raw_outcome = None
    if strict:
        raise RuntimeError(
            "full workflow validation outcome must be PASS, FAIL, ERROR, or AWAITING_EVIDENCE"
        )
    return "PASS" if bool(result.get("passed", False)) else "FAIL"


def _normalised_metric(value: object) -> str:
    return str(value or "").strip().lower()


def _normalised_direction(value: object) -> str | None:
    direction = str(value or "").strip().lower()
    return direction if direction in {"maximize", "minimize"} else None


def _objective_direction(value: object) -> str | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return {1: "maximize", 2: "minimize"}.get(value)
    normalised = str(value or "").strip().lower()
    if normalised in {"maximize", "continuous_maximize", "ratio_maximize"}:
        return "maximize"
    if normalised in {"minimize", "continuous_minimize"}:
        return "minimize"
    return None


def create_initial_state(
    nl_input: str,
    run_id: str | None = None,
    trace_id: str | None = None,
    artifact_ids: list[str] | None = None,
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
        "workflow_scope": "full",
        "validation_passed": False,
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
    properties.update(_retrosyn_critic_properties(state))
    properties.update(_selected_route_critic_properties(state))
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
        "supply_feasibility": str(assessment.get("overall_feasibility") or "").lower(),
    }
    properties["building_block_availability"] = (
        available_blocks / total_blocks if total_blocks > 0 else 0.0
    )
    return properties


def _srb_critic_properties(state: Mapping[str, object]) -> dict:
    srb = state.get("srb")
    if not isinstance(srb, Mapping):
        return {"srb_protocol_count": 0}
    protocols = srb.get("protocols")
    if not isinstance(protocols, list) or not protocols:
        return {"srb_protocol_count": 0}
    protocol = protocols[0]
    if not isinstance(protocol, Mapping):
        return {"srb_protocol_count": 0}
    properties = {"srb_protocol_count": len(protocols)}
    estimated_cost_per_gram = protocol.get("estimated_cost_usd_per_g")
    if isinstance(estimated_cost_per_gram, int | float) and not isinstance(
        estimated_cost_per_gram,
        bool,
    ):
        properties["estimated_cost_per_gram"] = float(estimated_cost_per_gram)
    steps = protocol.get("steps")
    if isinstance(steps, list):
        properties["synthesis_steps"] = len(steps)
    return properties


def _retrosyn_critic_properties(state: Mapping[str, object]) -> dict:
    retrosyn = state.get("retrosyn")
    routes = retrosyn.get("routes") if isinstance(retrosyn, Mapping) else None
    return {
        "retrosyn_route_count": len(routes) if isinstance(routes, list) else 0,
    }


def _selected_route_critic_properties(state: Mapping[str, object]) -> dict:
    retrosyn = state.get("retrosyn")
    routes = retrosyn.get("routes") if isinstance(retrosyn, Mapping) else None
    if not isinstance(routes, list) or not routes:
        return {}
    supply = state.get("supply")
    if not isinstance(supply, Mapping):
        raise RuntimeError("supply assessment must identify the selected route")
    route_id = _native_trimmed_string(supply.get("route_id"), "selected route_id")
    selected_routes = [
        route
        for route in routes
        if isinstance(route, Mapping) and route.get("route_id") == route_id
    ]
    if len(selected_routes) != 1:
        raise RuntimeError("supply route_id must reference exactly one retrosyn route")
    if any(not isinstance(route, Mapping) for route in routes):
        raise RuntimeError("retrosyn route entries must be objects")
    srb = state.get("srb")
    if not isinstance(srb, Mapping) or srb.get("route_id") != route_id:
        raise RuntimeError("srb route_id must match selected route")
    protocols = srb.get("protocols")
    if not isinstance(protocols, list) or any(
        not isinstance(protocol, Mapping) or protocol.get("route_id") != route_id
        for protocol in protocols
    ):
        raise RuntimeError("srb protocols must bind to selected route")
    return {"selected_route_id": route_id}


def _selected_protocol_ready(state: Mapping[str, object]) -> bool:
    try:
        properties = _selected_route_critic_properties(state)
    except RuntimeError:
        return False
    if "selected_route_id" not in properties:
        return False
    srb = state.get("srb")
    protocols = srb.get("protocols") if isinstance(srb, Mapping) else None
    return isinstance(protocols, list) and len(protocols) == 1


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
