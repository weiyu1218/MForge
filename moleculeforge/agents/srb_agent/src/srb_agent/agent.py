"""SRB Agent - Synthesis Reality Bridge: compiles retrosynthesis into SSPs."""

import asyncio
import copy
import inspect
import json
import os
import shlex
import subprocess
from collections.abc import Mapping
from typing import Any

from mf_agents.base.agent import BaseAgent, agent_health_check_timeout_seconds
from mf_agents.crg.graph import ChemicalReasoningGraph
from mf_core.artifacts import CommandRequirement, check_command, require_available
from mf_core.db.repositories import build_shared_crg_repository_from_env

from srb_agent.compiler import compile_ssp
from srb_agent.xdl_bridge import export_xdl

_SILA2_PLAN_COMMAND = CommandRequirement(
    "sila2_plan_command",
    "SILA2_PLAN_COMMAND",
)


class _Sila2CommandTarget:
    def __init__(self, command: str) -> None:
        self.command = command

    async def health_check(self) -> dict[str, bool]:
        if not self.command:
            return {"healthy": False}
        env = {**os.environ, "SILA2_PLAN_COMMAND": self.command}
        if not check_command(_SILA2_PLAN_COMMAND, env=env).available:
            return {"healthy": False}
        payload = {
            "dry_run": True,
            "health_check": True,
            "route_id": "runtime-health",
            "run_id": "runtime-health",
            "sila2_plan": {
                "endpoint": "",
                "route_id": "runtime-health",
                "run_id": "runtime-health",
                "ssp_id": "runtime-health",
                "steps": [],
                "target_smiles": "C",
            },
            "ssp_id": "runtime-health",
            "target_smiles": "C",
            "xdl_xml": "",
        }
        completed = await asyncio.to_thread(
            subprocess.run,
            shlex.split(self.command),
            input=json.dumps(payload, sort_keys=True),
            capture_output=True,
            check=False,
            text=True,
            timeout=agent_health_check_timeout_seconds(),
        )
        if completed.returncode != 0:
            return {"healthy": False}
        try:
            response = json.loads(completed.stdout)
        except json.JSONDecodeError:
            return {"healthy": False}
        return {"healthy": isinstance(response, dict) and response.get("healthy") is True}


class SRBAgent(BaseAgent):
    def __init__(self, message_bus=None, crg_repository: Any = None):
        super().__init__("srb_agent", message_bus)
        self._subscription_subjects = ["agent.srb.request", "orchestrator.srb.compile"]
        self.crg = ChemicalReasoningGraph()
        if crg_repository is None:
            self.crg_repository = build_shared_crg_repository_from_env()
            self._owns_crg_repository = self.crg_repository is not None
        else:
            self.crg_repository = crg_repository
            self._owns_crg_repository = False

    def runtime_targets(self) -> dict[str, object]:
        targets: dict[str, object] = {
            "sila2": _Sila2CommandTarget(os.environ.get("SILA2_PLAN_COMMAND", "").strip())
        }
        if self._owns_crg_repository:
            targets["crg_repository"] = self.crg_repository
        return targets

    async def process(self, data):
        if not isinstance(data, dict):
            raise TypeError("SRB request must be a dictionary")
        action = data.get("action", "compile")
        if action == "compile":
            return await self._compile(data)
        if action == "execute":
            return await self._execute(data)
        raise ValueError("action must be compile or execute")

    async def _compile(self, data: dict) -> dict:
        """Compile retrosynthesis routes into Structured Synthesis Protocols (SSPs).

        Translates abstract retrosynthetic pathways into executable synthesis
        protocols with specific conditions, yields, and purification steps.
        """
        run_id = str(data.get("run_id", ""))
        molecule = data.get("molecule")
        if molecule is None:
            smiles = data.get("smiles")
            if not isinstance(smiles, str) or not smiles:
                raise ValueError("molecule or smiles is required")
            molecule = {"smiles": smiles}
        target_smiles = _molecule_smiles(molecule)
        identity = _selected_candidate_identity(data, target_smiles)
        route = data.get("retrosyn_route")
        pathways = [route] if route is not None else data.get("pathways", [])
        if not isinstance(pathways, list) or not pathways:
            raise ValueError("retrosyn_route or non-empty pathways is required")
        if not all(isinstance(pathway, dict) for pathway in pathways):
            raise TypeError("retrosynthesis pathways must contain objects")
        selected_route_id = _selected_route_id(data)
        if data.get("workflow_scope") == "full":
            if len(pathways) != 1:
                raise ValueError("full workflow compilation requires exactly one selected route")
            route_id = _required_trimmed_text(pathways[0].get("route_id"), "pathway route_id")
            if route_id != selected_route_id:
                raise ValueError("pathway route_id must match selected route_id")
        protocols = []
        for retrosyn_route in pathways:
            ssp = await compile_ssp(molecule, retrosyn_route, run_id)
            protocol = _ssp_protocol_dict(ssp, retrosyn_route)
            protocols.append(protocol)
            belief = self.crg.add_belief(
                subject=protocol["target_smiles"],
                predicate="ssp_compiled",
                obj=str(protocol.get("route_id") or ""),
                confidence=1.0,
                source_agent=self.name,
                evidence_ids=[str(protocol["ssp_id"])],
            )
            await self._persist_belief(
                belief,
                project_id=str(data.get("project_id") or ""),
                run_id=run_id,
            )
        result = {
            "agent": self.name,
            "status": "compiled",
            "protocols": protocols,
            **identity,
        }
        if selected_route_id:
            result["route_id"] = selected_route_id
        return result

    async def _execute(self, data: dict) -> dict:
        if data.get("workflow_scope") != "full":
            raise ValueError("SRB execution requires workflow_scope=full")
        protocols = data.get("protocols")
        if not isinstance(protocols, list) or len(protocols) != 1:
            raise ValueError("SRB execution requires exactly one compiled protocol")
        if not isinstance(protocols[0], Mapping):
            raise TypeError("compiled protocol must be an object")
        protocol = copy.deepcopy(dict(protocols[0]))
        target_smiles = _required_trimmed_text(
            protocol.get("target_smiles"),
            "protocol target_smiles",
        )
        identity = _selected_candidate_identity(data, target_smiles)
        run_id = _required_trimmed_text(data.get("run_id"), "run_id")
        request_id = _required_trimmed_text(data.get("request_id"), "request_id")
        route_id = _selected_route_id(data)
        _require_protocol_binding(
            protocol,
            run_id=run_id,
            route_id=route_id,
            target_smiles=target_smiles,
        )
        await _attach_sila2_execution(
            protocol,
            {
                **identity,
                "run_id": run_id,
                "request_id": request_id,
                "route_id": route_id,
                "ssp_id": protocol["ssp_id"],
                "target_smiles": target_smiles,
            },
        )
        return {
            "agent": self.name,
            "status": "executed",
            "route_id": route_id,
            "protocols": [protocol],
            **identity,
        }

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


def _ssp_protocol_dict(ssp, retrosyn_route: Mapping[str, object]) -> dict:
    xdl_xml = export_xdl(ssp)
    protocol = {
        "ssp_id": ssp.ssp_id,
        "route_id": ssp.route_id,
        "target_smiles": ssp.target_smiles,
        "materials": [material.model_dump() for material in ssp.materials],
        "steps": [step.model_dump() for step in ssp.steps],
        "total_estimated_yield": ssp.total_estimated_yield,
        "total_estimated_cost_usd": ssp.total_estimated_cost_usd,
        "xdl_version": ssp.xdl_version,
        "xdl_xml": xdl_xml,
        "sila2_endpoint": ssp.sila2_endpoint,
        "sila2_plan": _sila2_plan_dict(ssp),
    }
    estimated_cost_usd_per_g = retrosyn_route.get("estimated_cost_usd_per_g")
    if estimated_cost_usd_per_g is not None:
        if isinstance(estimated_cost_usd_per_g, bool) or not isinstance(
            estimated_cost_usd_per_g,
            int | float,
        ):
            raise RuntimeError("retrosyn route requires numeric estimated_cost_usd_per_g")
        protocol["estimated_cost_usd_per_g"] = float(estimated_cost_usd_per_g)
    return protocol


async def _attach_sila2_execution(
    protocol: dict,
    execution_identity: Mapping[str, object],
) -> None:
    command = os.environ.get("SILA2_PLAN_COMMAND", "").strip()
    if not command:
        raise RuntimeError("SILA2_PLAN_COMMAND is required for SRB execution")
    _require_command_available(_SILA2_PLAN_COMMAND, command)
    payload = {
        **dict(execution_identity),
        "ssp_id": protocol["ssp_id"],
        "run_id": protocol["sila2_plan"]["run_id"],
        "route_id": protocol["route_id"],
        "target_smiles": protocol["target_smiles"],
        "sila2_plan": protocol["sila2_plan"],
        "xdl_xml": protocol["xdl_xml"],
    }
    completed = await asyncio.to_thread(
        subprocess.run,
        shlex.split(command),
        input=json.dumps(payload, sort_keys=True),
        capture_output=True,
        check=False,
        text=True,
        timeout=float(os.environ.get("SILA2_PLAN_TIMEOUT_SECONDS", "120")),
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        raise RuntimeError(f"SILA2_PLAN_COMMAND failed: {stderr}")
    try:
        response = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("SILA2_PLAN_COMMAND returned invalid JSON") from exc
    if not isinstance(response, dict):
        raise RuntimeError("SILA2_PLAN_COMMAND must return a JSON object")
    if response.get("status") != "completed":
        raise RuntimeError("SILA2_PLAN_COMMAND response status must be completed")
    job_id = response.get("job_id")
    if not isinstance(job_id, str) or not job_id or not job_id.strip() or job_id != job_id.strip():
        raise RuntimeError("SILA2_PLAN_COMMAND response job_id must be a non-empty string")
    for field, expected in execution_identity.items():
        actual = response.get(field)
        if type(actual) is not type(expected) or actual != expected:
            raise RuntimeError(f"SILA2_PLAN_COMMAND response {field} does not match request")
    protocol["sila2_execution"] = response
    endpoint = response.get("endpoint", response.get("sila2_endpoint"))
    if endpoint:
        protocol["sila2_endpoint"] = str(endpoint)
        protocol["sila2_plan"]["endpoint"] = str(endpoint)


def _require_command_available(
    requirement: CommandRequirement,
    command: str,
) -> None:
    env = {**os.environ, requirement.env_var: command}
    require_available([check_command(requirement, env=env)])


def _sila2_plan_dict(ssp) -> dict:
    return {
        "ssp_id": ssp.ssp_id,
        "run_id": ssp.run_id,
        "route_id": ssp.route_id,
        "target_smiles": ssp.target_smiles,
        "endpoint": ssp.sila2_endpoint,
        "steps": [_sila2_step_dict(step) for step in ssp.steps],
    }


def _sila2_step_dict(step) -> dict:
    return {
        "command": "execute_reaction_step",
        "ssp_step_id": step.step_id,
        "retrosyn_route_step_id": step.parameters.get("retrosyn_route_step_id", ""),
        "operation": step.operation,
        "reaction_type": step.reaction_type or "",
        "reactants": [reactant.model_dump() for reactant in step.reactants],
        "reagents": list(step.reagents),
        "temperature_C": step.temperature_C,
        "time_h": step.time_h,
        "purification": step.purification or "",
    }


def _molecule_smiles(molecule: Any) -> str:
    if isinstance(molecule, dict):
        return str(molecule.get("smiles") or molecule.get("canonical_smiles") or "")
    return str(getattr(molecule, "smiles", "") or getattr(molecule, "canonical_smiles", ""))


def _selected_candidate_identity(data: Any, target_smiles: str) -> dict[str, object]:
    if not isinstance(data, dict) or data.get("workflow_scope") != "full":
        return {}
    identity: dict[str, object] = {}
    for field in ("project_id", "candidate_id", "canonical_smiles"):
        value = data.get(field)
        if not isinstance(value, str) or not value or value != value.strip():
            raise ValueError(f"{field} must be a non-empty trimmed string")
        identity[field] = value
    candidate_index = data.get("candidate_index")
    if (
        isinstance(candidate_index, bool)
        or not isinstance(candidate_index, int)
        or candidate_index < 0
    ):
        raise ValueError("candidate_index must be a non-negative integer")
    identity["candidate_index"] = candidate_index
    if identity["canonical_smiles"] != target_smiles:
        raise ValueError("canonical_smiles must match molecule smiles")
    return identity


def _selected_route_id(data: Mapping[str, object]) -> str:
    if data.get("workflow_scope") != "full":
        return ""
    return _required_trimmed_text(data.get("route_id"), "route_id")


def _require_protocol_binding(
    protocol: Mapping[str, object],
    *,
    run_id: str,
    route_id: str,
    target_smiles: str,
) -> None:
    expected = {
        "route_id": route_id,
        "target_smiles": target_smiles,
    }
    for field, expected_value in expected.items():
        if protocol.get(field) != expected_value:
            raise ValueError(f"protocol {field} must match selected route")
    ssp_id = _required_trimmed_text(protocol.get("ssp_id"), "protocol ssp_id")
    sila2_plan = protocol.get("sila2_plan")
    if not isinstance(sila2_plan, Mapping):
        raise ValueError("protocol sila2_plan must be an object")
    plan_expected = {
        "ssp_id": ssp_id,
        "run_id": run_id,
        "route_id": route_id,
        "target_smiles": target_smiles,
    }
    for field, expected_value in plan_expected.items():
        if sila2_plan.get(field) != expected_value:
            raise ValueError(f"protocol sila2_plan {field} does not match protocol")
    _required_trimmed_text(protocol.get("xdl_xml"), "protocol xdl_xml")


def _required_trimmed_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or not value.strip() or value != value.strip():
        raise ValueError(f"{field} must be a non-empty trimmed string")
    return value
