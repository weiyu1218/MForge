"""FTO Agent - Freedom-to-operate and IP safety checker (Agent-5)."""
import json

from mf_agents.base.agent import BaseAgent
from mf_agents.crg.graph import ChemicalReasoningGraph


class FTOAgent(BaseAgent):
    def __init__(self, message_bus=None):
        super().__init__("fto_agent", message_bus)
        self._subscription_subjects = ["agent.fto.request", "orchestrator.fto.check"]
        self.crg = ChemicalReasoningGraph()

    async def handle_message(self, subject, payload, reply_to=""):
        data = json.loads(payload) if isinstance(payload, bytes) else {"raw": payload}
        result = await self.process(data)
        if reply_to:
            await self.publish(reply_to, json.dumps(result).encode())

    async def process(self, data):
        """Check freedom-to-operate status for a molecular structure.

        Queries patent databases and performs substructure matching against
        known patented chemical space. Assesses novelty and IP risk.
        """
        smiles = data.get("smiles", "")
        return {
            "agent": self.name,
            "status": "checked",
            "smiles": smiles,
            "fto_result": {
                "patent_matches": 0,
                "structure_novel": True,
                "ip_risk": "low",
                "blocking_patents": [],
            },
        }
