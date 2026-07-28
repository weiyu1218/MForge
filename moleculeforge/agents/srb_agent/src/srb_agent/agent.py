"""SRB Agent - Synthesis Reality Bridge: compiles retrosynthesis into SSPs."""

import asyncio
import inspect
import json
import os
import shlex
import subprocess
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
        self.crg_repository = (
            crg_repository if crg_repository is not None else build_shared_crg_repository_from_env()
        )

    def runtime_targets(self) -> dict[str, object]:
        return {"sila2": _Sila2CommandTarget(os.environ.get("SILA2_PLAN_COMMAND", "").strip())}

    async def process(self, data):
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
        if await self._has_unavailable_supply(target_smiles, run_id):
            belief = self.crg.add_belief(
                subject=target_smiles,
                predicate="ssp_compiled",
                obj="skipped",
                confidence=0.0,
                source_agent=self.name,
                evidence_ids=["crg_supply_feasibility"],
            )
            await self._persist_belief(
                belief,
                project_id=str(data.get("project_id") or ""),
                run_id=run_id,
            )
            return {
                "agent": self.name,
                "status": "skipped",
                "protocols": [],
                "skip_reason": "shared CRG contains unavailable supply_feasibility",
            }
        route = data.get("retrosyn_route")
        pathways = [route] if route is not None else data.get("pathways", [])
        protocols = []
        for retrosyn_route in pathways:
            ssp = await compile_ssp(molecule, retrosyn_route, run_id)
            protocol = _ssp_protocol_dict(ssp)
            _attach_sila2_execution(protocol)
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
        return {
            "agent": self.name,
            "status": "compiled",
            "protocols": protocols,
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

    async def _has_unavailable_supply(self, target_smiles: str, run_id: str) -> bool:
        if (
            not target_smiles
            or not run_id
            or self.crg_repository is None
            or not callable(getattr(self.crg_repository, "get_run_crg", None))
        ):
            return False
        crg = await self.read_shared_crg(run_id)
        for belief in crg.get("beliefs", []) or []:
            if not isinstance(belief, dict):
                continue
            if str(belief.get("subject") or "") != target_smiles:
                continue
            if str(belief.get("predicate") or "") != "supply_feasibility":
                continue
            value = str(belief.get("object") or belief.get("object_value") or "")
            if value == "unavailable":
                return True
        return False


def _ssp_protocol_dict(ssp) -> dict:
    xdl_xml = export_xdl(ssp)
    return {
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


def _attach_sila2_execution(protocol: dict) -> None:
    command = os.environ.get("SILA2_PLAN_COMMAND", "").strip()
    if not command:
        return
    _require_command_available(_SILA2_PLAN_COMMAND, command)
    payload = {
        "ssp_id": protocol["ssp_id"],
        "run_id": protocol["sila2_plan"]["run_id"],
        "route_id": protocol["route_id"],
        "target_smiles": protocol["target_smiles"],
        "sila2_plan": protocol["sila2_plan"],
        "xdl_xml": protocol["xdl_xml"],
    }
    completed = subprocess.run(
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
