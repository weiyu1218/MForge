"""Orchestrator Agent - the central coordinator using LangGraph state machine."""

import inspect
import json
import math
from collections import deque
from collections.abc import Mapping
from typing import Any

from mf_agents.base.agent import AGENT_PROTOCOLS, BaseAgent
from mf_agents.crg.graph import ChemicalReasoningGraph
from mf_agents.messaging.request_client import AgentRequestClient
from mf_core.db.repositories import build_shared_crg_repository_from_env

from orchestrator.workflow.graph_builder import (
    WorkflowGraph,
    create_initial_state,
    full_workflow_critic_properties,
)

_AGENT_PROTOCOLS_BY_ENTRY_POINT = {protocol.entry_point: protocol for protocol in AGENT_PROTOCOLS}
_WORKFLOW_SCOPES = frozenset({"state_only", "engineering", "full"})
_NL2OBJ_SUBJECT = "agent.nl2obj.request"
_NL2OBJ_PAYLOAD_TYPE_URL = "type.moleculeforge.ai/agent/nl2obj/request.v1"
_NL2OBJ_SCHEMA_VERSION = "nl2obj.request.v1"


class OrchestratorAgent(BaseAgent):
    def __init__(self, message_bus=None, crg_repository: Any = None):
        super().__init__("orchestrator", message_bus)
        self._subscription_subjects = ["orchestrator.design.request", "orchestrator.status"]
        self.crg = ChemicalReasoningGraph()
        self.crg_repository = (
            crg_repository if crg_repository is not None else build_shared_crg_repository_from_env()
        )

    async def handle_message(self, subject, payload, reply_to=""):
        data = json.loads(payload) if isinstance(payload, bytes) else payload
        if "design.request" in subject:
            result = await self.run_design_workflow(data)
            if reply_to:
                await self.publish(reply_to, json.dumps(result).encode())

    async def process(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        return await self.run_design_workflow(dict(payload))

    async def run_design_workflow(self, request: dict) -> dict:
        request = dict(request)
        project_id = _required_text(request, "project_id")
        run_id = _required_text(request, "run_id")
        trace_id = _required_text(request, "trace_id")
        intent = _workflow_intent(request)
        workflow_scope = _workflow_scope(request)
        max_refinements = _max_refinements(request)
        state = create_initial_state(
            intent,
            run_id=run_id,
            trace_id=trace_id,
            artifact_ids=_artifact_ids(request),
            workflow_scope=workflow_scope,
        )
        state["request"] = request
        state["max_refinements"] = max_refinements
        state["validation_passed"] = _initial_validation_state(
            request,
            workflow_scope,
        )
        clients = None
        if workflow_scope != "state_only":
            if self.message_bus is None:
                raise RuntimeError(f"{workflow_scope} workflow requires an Agent message bus")
            request_client = AgentRequestClient(self.message_bus, sender=self.name)
            if workflow_scope == "full":
                clients = _FullAgentWorkflowClients(request_client)
            else:
                clients = _EngineeringAgentWorkflowClients(request_client)
        compiled = WorkflowGraph(
            clients=clients,
            workflow_scope=workflow_scope,
        ).build()
        final_state = await compiled.ainvoke(
            state,
            config={
                "recursion_limit": max(25, 8 + 5 * max_refinements),
            },
        )
        if not isinstance(final_state, dict):
            raise RuntimeError("WorkflowGraph must return a state mapping")
        current_stage = str(final_state.get("status") or "")
        if current_stage == "CRITIC":
            workflow_status = "completed"
        elif current_stage == "ESCALATING":
            workflow_status = "rejected"
        else:
            raise RuntimeError(
                f"WorkflowGraph returned non-terminal stage: {current_stage or '<empty>'}"
            )
        history = [str(item) for item in final_state.get("history", [])]
        result = dict(final_state)
        result.update(
            {
                "project_id": project_id,
                "status": workflow_status,
                "current_stage": current_stage,
                "visited_nodes": [stage.lower() for stage in history],
            }
        )
        belief = self.crg.add_belief(
            subject=str(project_id),
            predicate="workflow_status",
            obj=workflow_status,
            confidence=1.0,
            source_agent=self.name,
            evidence_ids=history,
        )
        await self._persist_belief(
            belief,
            project_id=str(project_id),
            run_id=run_id,
        )
        return result

    async def _persist_belief(self, belief, project_id: str, run_id: str) -> None:
        if self.crg_repository is None:
            return
        write_belief = getattr(self.crg_repository, "write_workflow_belief", None)
        if not callable(write_belief):
            raise TypeError("crg_repository must expose write_workflow_belief(**kwargs)")
        result = write_belief(
            project_id=project_id,
            run_id=run_id or belief.subject,
            belief_id=belief.id,
            subject=belief.subject,
            predicate=belief.predicate,
            object_value=belief.object,
            confidence=belief.confidence,
            source_agent=belief.source_agent,
            timestamp_ns=belief.timestamp_ns,
            evidence_ids=list(belief.evidence_ids),
        )
        if inspect.isawaitable(result):
            await result


class _FullAgentWorkflowClients:
    def __init__(self, request_client: AgentRequestClient) -> None:
        self.request_client = request_client

    async def compile_intent(self, state: dict) -> dict:
        request = dict(state.get("request") or {})
        result = await self._request(
            state,
            "nl2obj",
            {
                **_business_request(request),
                "project_id": str(request["project_id"]),
                "intent": str(state["nl_input"]),
            },
        )
        return {
            key: result[key]
            for key in ("cig", "hciv", "intent_cone", "objectives")
            if key in result
        }

    async def generate_candidates(self, state: dict) -> list[dict]:
        request = dict(state.get("request") or {})
        generator_params = dict(request.get("generator_params") or {})
        generation_feedback = state.get("generation_feedback")
        if isinstance(generation_feedback, list) and generation_feedback:
            generator_params["generation_feedback"] = json.dumps(
                generation_feedback,
                sort_keys=True,
            )
        result = await self._request(
            state,
            "generator_coord",
            {
                "project_id": str(request["project_id"]),
                "objectives": dict(state.get("objectives") or request.get("objectives") or {}),
                "cig": state.get("cig"),
                "hciv": state.get("hciv"),
                "intent_cone": state.get("intent_cone"),
                "n_samples": request.get("n_samples", request.get("batch_size")),
                "batch_size": request.get("batch_size", request.get("n_samples")),
                "generator_params": generator_params,
                **(
                    {"generation_strategy": request["generation_strategy"]}
                    if "generation_strategy" in request
                    else {}
                ),
            },
        )
        candidates = result.get("candidates")
        if not isinstance(candidates, list) or not all(
            isinstance(candidate, dict) for candidate in candidates
        ):
            raise RuntimeError("generator_coord Agent must return candidates as a list")
        return [dict(candidate) for candidate in candidates]

    async def validate_candidates(self, state: dict) -> dict:
        request = dict(state.get("request") or {})
        generated = state.get("candidates")
        if isinstance(generated, list) and not generated:
            return {
                "passed": False,
                "reason": "no valid candidates",
                "results": [],
            }
        candidates = _workflow_candidates(state)
        rows = []
        for candidate_index, candidate in enumerate(candidates):
            smiles = _candidate_smiles(candidate)
            payload = {
                **_validation_request(request),
                "project_id": str(request["project_id"]),
                "smiles": smiles,
                **_candidate_reference(candidate),
            }
            validation = await self._request(
                state,
                "validation",
                payload,
                candidate_index=candidate_index,
            )
            row = dict(validation)
            row["smiles"] = smiles
            row["candidate_index"] = candidate_index
            row.update(_candidate_reference(candidate))
            rows.append(row)
        passed = any(
            bool(row.get("overall_passed", row.get("status") == "validated")) for row in rows
        )
        result = {
            "passed": passed,
            "results": rows,
        }
        if not passed:
            reason = next(
                (
                    row["reason"]
                    for row in rows
                    if isinstance(row.get("reason"), str) and row["reason"]
                ),
                None,
            )
            if reason is not None:
                result["reason"] = reason
        return result

    async def plan_routes(self, state: dict) -> dict:
        request = dict(state.get("request") or {})
        candidate = _selected_candidate(state)
        return await self._request(
            state,
            "retrosyn",
            {
                "project_id": str(request["project_id"]),
                "smiles": _candidate_smiles(candidate),
                **_candidate_reference(candidate),
                "max_routes": request.get(
                    "retrosyn_max_routes",
                    request.get("max_routes", 3),
                ),
            },
        )

    async def assess_supply(self, state: dict) -> dict:
        request = dict(state.get("request") or {})
        candidate = _selected_candidate(state)
        route = _workflow_route_or_none(state)
        if route is None:
            return _unavailable_supply_result(
                candidate,
                "retrosyn.routes is empty",
            )
        return await self._request(
            state,
            "supply",
            {
                "project_id": str(request["project_id"]),
                "smiles": _candidate_smiles(candidate),
                **_candidate_reference(candidate),
                "building_blocks": _route_building_blocks(route),
            },
        )

    async def compile_synthesis(self, state: dict) -> dict:
        request = dict(state.get("request") or {})
        candidate = _selected_candidate(state)
        if _supply_feasibility(state) == "unavailable":
            return {
                "status": "skipped",
                "protocols": [],
                "skip_reason": "supply feasibility is unavailable",
            }
        route = _workflow_route_or_none(state)
        if route is None:
            return {
                "status": "skipped",
                "protocols": [],
                "skip_reason": "retrosyn.routes is empty",
            }
        return await self._request(
            state,
            "srb",
            {
                "project_id": str(request["project_id"]),
                "molecule": {"smiles": _candidate_smiles(candidate)},
                **_candidate_reference(candidate),
                "retrosyn_route": route,
            },
        )

    async def review_candidates(self, state: dict) -> dict:
        return await self._review_candidates(state, full_workflow=True)

    async def review_engineering_candidates(self, state: dict) -> dict:
        return await self._review_candidates(state, full_workflow=False)

    async def _review_candidates(
        self,
        state: dict,
        *,
        full_workflow: bool,
    ) -> dict:
        request = dict(state.get("request") or {})
        candidate = _selected_candidate(state)
        smiles = _candidate_smiles(candidate)
        properties = (
            _full_critic_properties(candidate, state)
            if full_workflow
            else _critic_properties(candidate, state)
        )
        return await self._request(
            state,
            "critic",
            {
                "project_id": str(request["project_id"]),
                "smiles": smiles,
                **_candidate_reference(candidate),
                "properties": properties,
            },
        )

    async def _request(
        self,
        state: dict,
        entry_point: str,
        payload: dict,
        *,
        candidate_index: int | None = None,
    ) -> dict:
        run_id = _required_text(state, "run_id")
        trace_id = _required_text(state, "trace_id")
        request = dict(state.get("request") or {})
        outer_request_id = _required_text(request, "request_id")
        refinement_count = int(state.get("refinement_count", 0))
        request_id = f"{outer_request_id}:{entry_point}:{refinement_count}"
        if candidate_index is not None:
            request_id = f"{request_id}:candidate-{candidate_index}"
        parent_id = f"{outer_request_id}:{entry_point}"
        if entry_point == "nl2obj":
            subject = _NL2OBJ_SUBJECT
            payload_type_url = _NL2OBJ_PAYLOAD_TYPE_URL
            schema_version = _NL2OBJ_SCHEMA_VERSION
        else:
            protocol = _AGENT_PROTOCOLS_BY_ENTRY_POINT[entry_point]
            subject = protocol.subject
            payload_type_url = protocol.payload_type_url
            schema_version = protocol.schema_version
        agent_payload = {
            **payload,
            "trace_id": trace_id,
            "parent_id": parent_id,
            "run_id": run_id,
            "request_id": request_id,
            "schema_version": schema_version,
        }
        result = await self.request_client.request(
            subject,
            agent_payload,
            payload_type_url=payload_type_url,
            timeout=_agent_request_timeout(request),
        )
        business_result = dict(result)
        for field in ("run_id", "request_id", "schema_version"):
            business_result.pop(field, None)
        return business_result


class _EngineeringAgentWorkflowClients:
    def __init__(self, request_client: AgentRequestClient) -> None:
        self._full_clients = _FullAgentWorkflowClients(request_client)

    async def compile_intent(self, state: dict) -> dict:
        return await self._full_clients.compile_intent(state)

    async def generate_candidates(self, state: dict) -> list[dict]:
        return await self._full_clients.generate_candidates(state)

    async def validate_candidates(self, state: dict) -> dict:
        return await self._full_clients.validate_candidates(state)

    async def review_candidates(self, state: dict) -> dict:
        return await self._full_clients.review_engineering_candidates(state)


def _required_text(mapping: Mapping[str, Any], field: str) -> str:
    value = mapping.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} is required")
    return value


def _workflow_intent(request: Mapping[str, Any]) -> str:
    for field in ("nl_input", "intent"):
        value = request.get(field)
        if isinstance(value, str) and value:
            return value
    raise ValueError("nl_input or intent is required")


def _workflow_scope(request: Mapping[str, Any]) -> str:
    workflow_scope = request.get("workflow_scope")
    if not isinstance(workflow_scope, str) or workflow_scope not in _WORKFLOW_SCOPES:
        allowed = ", ".join(sorted(_WORKFLOW_SCOPES))
        raise ValueError(f"workflow_scope must be one of: {allowed}")
    return workflow_scope


def _max_refinements(request: Mapping[str, Any]) -> int:
    value = request.get("max_refinements")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("max_refinements must be a non-negative integer")
    return value


def _artifact_ids(request: Mapping[str, Any]) -> list[str]:
    value = request.get("artifact_ids", [])
    if not isinstance(value, list) or not all(
        isinstance(artifact_id, str) and artifact_id for artifact_id in value
    ):
        raise ValueError("artifact_ids must be a list of non-empty strings")
    return list(value)


def _initial_validation_state(
    request: Mapping[str, Any],
    workflow_scope: str,
) -> bool:
    value = request.get("validation_passed")
    if workflow_scope == "state_only":
        if not isinstance(value, bool):
            raise ValueError("validation_passed must be a boolean")
        return value
    if value is not None and not isinstance(value, bool):
        raise ValueError("validation_passed must be a boolean")
    return True if value is None else value


def _agent_request_timeout(request: Mapping[str, Any]) -> float:
    value = request.get("agent_request_timeout_seconds", 30.0)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError("agent_request_timeout_seconds must be positive")
    timeout = float(value)
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("agent_request_timeout_seconds must be positive")
    return timeout


def _business_request(request: Mapping[str, Any]) -> dict:
    excluded = {
        "artifact_ids",
        "parent_id",
        "request_id",
        "run_id",
        "schema_version",
        "trace_id",
        "workflow_scope",
    }
    return {key: value for key, value in request.items() if key not in excluded}


def _validation_request(request: Mapping[str, Any]) -> dict:
    payload = _business_request(request)
    requested_level = None
    for key in ("oracle_level", "max_oracle_level", "validation_oracle_level"):
        if key in payload and payload[key] is not None:
            requested_level = payload[key]
            break
    for key in ("max_oracle_level", "validation_oracle_level"):
        payload.pop(key, None)
    if requested_level is None:
        payload.pop("oracle_level", None)
    else:
        payload["oracle_level"] = requested_level
    return payload


def _workflow_candidates(state: Mapping[str, Any]) -> list[dict]:
    candidates = state.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise RuntimeError("workflow requires at least one generated candidate")
    if not all(isinstance(candidate, dict) for candidate in candidates):
        raise RuntimeError("workflow candidates must be objects")
    return candidates


def _candidate_smiles(candidate: Mapping[str, Any]) -> str:
    value = candidate.get("canonical_smiles") or candidate.get("smiles")
    if not isinstance(value, str) or not value:
        raise RuntimeError("candidate smiles is required")
    return value


def _candidate_reference(candidate: Mapping[str, Any]) -> dict[str, str]:
    candidate_id = candidate.get("candidate_id")
    if candidate_id in (None, ""):
        return {}
    return {"candidate_id": str(candidate_id)}


def _candidate_validation_pairs(
    state: Mapping[str, Any],
) -> list[tuple[dict, dict]]:
    candidates = _workflow_candidates(state)
    validation = state.get("validation")
    rows = validation.get("results") if isinstance(validation, dict) else None
    if not isinstance(rows, list):
        return []
    by_candidate_id: dict[str, deque[int]] = {}
    by_candidate_id_and_smiles: dict[tuple[str, str], deque[int]] = {}
    by_smiles: dict[str, deque[int]] = {}
    for index, candidate in enumerate(candidates):
        candidate_id = candidate.get("candidate_id")
        smiles = _candidate_smiles(candidate)
        by_smiles.setdefault(smiles, deque()).append(index)
        if candidate_id in (None, ""):
            continue
        candidate_id = str(candidate_id)
        by_candidate_id.setdefault(candidate_id, deque()).append(index)
        by_candidate_id_and_smiles.setdefault(
            (candidate_id, smiles),
            deque(),
        ).append(index)

    explicit_matches: dict[int, int] = {}
    explicitly_matched_indices: set[int] = set()
    for row_index, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        candidate_id = str(row.get("candidate_id") or "")
        smiles = str(row.get("canonical_smiles") or row.get("smiles") or "")
        if not candidate_id or not smiles:
            continue
        matches = by_candidate_id_and_smiles.get(
            (candidate_id, smiles),
            deque(),
        )
        while matches and matches[0] in explicitly_matched_indices:
            matches.popleft()
        if matches:
            match = matches.popleft()
            explicit_matches[row_index] = match
            explicitly_matched_indices.add(match)

    for row_index, row in enumerate(rows):
        if row_index in explicit_matches or not isinstance(row, dict):
            continue
        candidate_id = str(row.get("candidate_id") or "")
        if not candidate_id:
            continue
        matches = by_candidate_id.get(candidate_id, deque())
        while matches and matches[0] in explicitly_matched_indices:
            matches.popleft()
        if matches:
            match = matches.popleft()
            explicit_matches[row_index] = match
            explicitly_matched_indices.add(match)

    reserved_indices = set(explicit_matches.values())
    claimed_indices: set[int] = set()
    pairs = []
    for row_index, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        match = explicit_matches.get(row_index)
        candidate_id = str(row.get("candidate_id") or "")
        if candidate_id and match is None:
            continue
        if match is None:
            smiles = str(row.get("canonical_smiles") or row.get("smiles") or "")
            if not smiles:
                continue
            matches = by_smiles.get(smiles, deque())
            while matches and (matches[0] in reserved_indices or matches[0] in claimed_indices):
                matches.popleft()
            if matches:
                match = matches.popleft()
        if match is None:
            continue
        claimed_indices.add(match)
        pairs.append((candidates[match], row))
    return pairs


def _selected_candidate(state: Mapping[str, Any]) -> dict:
    for candidate, validation in _candidate_validation_pairs(state):
        passed = validation.get(
            "overall_passed",
            validation.get("status") == "validated",
        )
        if passed is True:
            return candidate
    raise RuntimeError("workflow requires a passing validated candidate")


def _workflow_route_or_none(state: Mapping[str, Any]) -> dict | None:
    retrosyn = state.get("retrosyn")
    routes = retrosyn.get("routes") if isinstance(retrosyn, dict) else None
    if not isinstance(routes, list) or not routes:
        return None
    if not isinstance(routes[0], dict):
        raise RuntimeError("retrosyn route entries must be objects")
    return dict(routes[0])


def _supply_feasibility(state: Mapping[str, Any]) -> str:
    supply = state.get("supply")
    assessment = supply.get("supply_assessment") if isinstance(supply, dict) else None
    if not isinstance(assessment, dict):
        return ""
    return str(assessment.get("overall_feasibility") or "").lower()


def _unavailable_supply_result(
    candidate: Mapping[str, Any],
    reason: str,
) -> dict:
    return {
        "agent": "supply_agent",
        "status": "assessed",
        "smiles": _candidate_smiles(candidate),
        **_candidate_reference(candidate),
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


def _route_building_blocks(route: Mapping[str, Any]) -> list:
    for key in ("building_blocks", "starting_materials", "precursors"):
        value = route.get(key)
        if value is not None:
            if not isinstance(value, list):
                raise RuntimeError(f"retrosyn route {key} must be a list")
            return list(value)
    return []


def _critic_properties(
    candidate: Mapping[str, Any],
    state: Mapping[str, Any],
) -> dict:
    properties = dict(candidate.get("properties") or {})
    validation_row = _validation_row_for_candidate(candidate, state)
    if validation_row is None:
        return properties
    cascade = validation_row.get("cascade")
    if not isinstance(cascade, dict):
        return properties
    for level_result in cascade.values():
        if not isinstance(level_result, dict):
            continue
        scores = level_result.get("result")
        if isinstance(scores, dict):
            properties.update(scores)
    return properties


def _full_critic_properties(
    candidate: Mapping[str, Any],
    state: Mapping[str, Any],
) -> dict:
    return full_workflow_critic_properties(
        state,
        candidate,
        _validation_row_for_candidate(candidate, state),
    )


def _validation_row_for_candidate(
    candidate: Mapping[str, Any],
    state: Mapping[str, Any],
) -> dict | None:
    for paired_candidate, row in _candidate_validation_pairs(state):
        if paired_candidate is candidate:
            return row
    return None
