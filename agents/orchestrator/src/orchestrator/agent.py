"""Orchestrator Agent - the central coordinator using LangGraph state machine."""
import json

from mf_agents.base.agent import BaseAgent


class OrchestratorAgent(BaseAgent):
    def __init__(self, message_bus=None):
        super().__init__("orchestrator", message_bus)
        self._subscription_subjects = ["orchestrator.design.request", "orchestrator.status"]
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
        nodes = [
            "nl2obj", "humu_encode", "generate", "validate",
            "fto_check", "retrosyn", "critic", "orchestrate", "refine",
        ]
        results = {"project_id": project_id, "visited_nodes": [], "status": "running"}
        for node in nodes:
            self.cycle_count += 1
            if self.cycle_count > self.max_cycles:
                results["status"] = "max_cycles_reached"
                break
            results["visited_nodes"].append(node)
        results["status"] = "completed"
        return results
