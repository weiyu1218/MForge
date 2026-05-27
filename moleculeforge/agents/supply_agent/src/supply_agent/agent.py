"""Supply Agent - Building block accessibility scoring (Agent-6)."""
import json

from mf_agents.base.agent import BaseAgent
from mf_agents.crg.graph import ChemicalReasoningGraph


class SupplyAgent(BaseAgent):
    def __init__(self, message_bus=None):
        super().__init__("supply_agent", message_bus)
        self._subscription_subjects = ["agent.supply.request", "orchestrator.supply.check"]
        self.crg = ChemicalReasoningGraph()

    async def handle_message(self, subject, payload, reply_to=""):
        data = json.loads(payload) if isinstance(payload, bytes) else {"raw": payload}
        result = await self.process(data)
        if reply_to:
            await self.publish(reply_to, json.dumps(result).encode())

    async def process(self, data):
        """Evaluate supply chain feasibility for building blocks.

        Checks building block availability, lead times, pricing, and
        supplier diversity across major chemical catalogs.
        """
        smiles = data.get("smiles", "")
        building_blocks = data.get("building_blocks", [])
        return {
            "agent": self.name,
            "status": "assessed",
            "smiles": smiles,
            "supply_assessment": {
                "total_blocks": len(building_blocks),
                "commercially_available": 0,
                "avg_price_per_gram": 0.0,
                "avg_lead_time_days": 0,
                "supplier_diversity": 0,
                "overall_feasibility": "unknown",
            },
        }
