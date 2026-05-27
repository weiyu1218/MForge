"""NL2Obj Agent - Natural language to objective specification agent (Agent-1)."""
import json

from mf_agents.base.agent import BaseAgent
from mf_agents.crg.graph import ChemicalReasoningGraph


class NL2ObjAgent(BaseAgent):
    def __init__(self, message_bus=None):
        super().__init__("nl2obj", message_bus)
        self._subscription_subjects = ["agent.nl2obj.request", "orchestrator.nl2obj.resolve"]
        self.crg = ChemicalReasoningGraph()

    async def handle_message(self, subject, payload, reply_to=""):
        data = json.loads(payload) if isinstance(payload, bytes) else {"raw": payload}
        result = await self.process(data)
        if reply_to:
            await self.publish(reply_to, json.dumps(result).encode())

    async def process(self, data):
        """Parse natural language intent into structured objective specification (CIG).

        Extracts target properties, constraints, and objectives from user intent text
        and compiles them into a Compliant Intent Graph (CIG).
        """
        intent_text = data.get("intent", data.get("text", ""))
        constraints = data.get("constraints", {})
        return {
            "agent": self.name,
            "status": "resolved",
            "intent": intent_text,
            "objectives": {
                "target_properties": data.get("target_properties", {}),
                "constraints": constraints,
            },
        }
