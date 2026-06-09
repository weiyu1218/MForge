"""Orchestrator Agent - the central coordinator using LangGraph state machine."""
import inspect
import json
from typing import Any

from mf_agents.base.agent import BaseAgent
from mf_agents.crg.graph import ChemicalReasoningGraph
from mf_core.db.repositories import build_shared_crg_repository_from_env


class OrchestratorAgent(BaseAgent):
    def __init__(self, message_bus=None, crg_repository: Any = None):
        super().__init__("orchestrator", message_bus)
        self._subscription_subjects = ["orchestrator.design.request", "orchestrator.status"]
        self.crg = ChemicalReasoningGraph()
        self.crg_repository = (
            crg_repository
            if crg_repository is not None
            else build_shared_crg_repository_from_env()
        )
        self.cycle_count = 0
        self.max_cycles = 20

    async def handle_message(self, subject, payload, reply_to=""):
        data = json.loads(payload) if isinstance(payload, bytes) else payload
        if "design.request" in subject:
            result = await self.run_design_workflow(data)
            if reply_to:
                await self.publish(reply_to, json.dumps(result).encode())

    async def run_design_workflow(self, request: dict) -> dict:
        project_id = request.get("project_id", "unknown")
        run_id = str(request.get("run_id") or request.get("request_id") or project_id)
        cached = await self._completed_workflow_from_shared_crg(str(project_id), run_id)
        if cached is not None:
            belief = self.crg.add_belief(
                subject=str(project_id),
                predicate="workflow_status",
                obj="completed",
                confidence=1.0,
                source_agent=self.name,
                evidence_ids=["crg_workflow_status"],
            )
            await self._persist_belief(
                belief,
                project_id=str(project_id),
                run_id=run_id,
            )
            return {
                "project_id": project_id,
                "visited_nodes": cached,
                "status": "completed",
                "cached": True,
            }
        nodes = [
            "nl2obj", "humu_encode", "generate", "validate",
            "retrosyn", "critic", "orchestrate", "refine",
        ]
        results = {"project_id": project_id, "visited_nodes": [], "status": "running"}
        for node in nodes:
            self.cycle_count += 1
            if self.cycle_count > self.max_cycles:
                results["status"] = "max_cycles_reached"
                break
            results["visited_nodes"].append(node)
        results["status"] = "completed"
        belief = self.crg.add_belief(
            subject=str(project_id),
            predicate="workflow_status",
            obj=str(results["status"]),
            confidence=1.0,
            source_agent=self.name,
            evidence_ids=list(results["visited_nodes"]),
        )
        await self._persist_belief(
            belief,
            project_id=str(project_id),
            run_id=run_id,
        )
        return results

    async def _completed_workflow_from_shared_crg(
        self,
        project_id: str,
        run_id: str,
    ) -> list[str] | None:
        if (
            not run_id
            or self.crg_repository is None
            or not callable(getattr(self.crg_repository, "get_run_crg", None))
        ):
            return None
        crg = await self.read_shared_crg(run_id)
        for belief in crg.get("beliefs", []) or []:
            if not isinstance(belief, dict):
                continue
            if str(belief.get("subject") or "") != project_id:
                continue
            if str(belief.get("predicate") or "") != "workflow_status":
                continue
            status = str(belief.get("object") or belief.get("object_value") or "")
            if status == "completed":
                return [str(item) for item in belief.get("evidence_ids", [])]
        return None

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
