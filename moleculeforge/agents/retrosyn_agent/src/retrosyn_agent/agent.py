"""RetroSyn Agent - 3-layer retrosynthesis planning agent (Agent-3)."""
import json

from mf_agents.base.agent import BaseAgent
from mf_agents.crg.graph import ChemicalReasoningGraph


class RetroSynAgent(BaseAgent):
    def __init__(self, message_bus=None, planner=None):
        super().__init__("retrosyn_agent", message_bus)
        self._subscription_subjects = ["agent.retrosyn.request", "orchestrator.retrosyn.plan"]
        self.crg = ChemicalReasoningGraph()
        self.planner = planner

    async def handle_message(self, subject, payload, reply_to=""):
        data = json.loads(payload) if isinstance(payload, bytes) else {"raw": payload}
        result = await self.process(data)
        if reply_to:
            await self.publish(reply_to, json.dumps(result).encode())

    async def process(self, data):
        """Plan 3-layer retrosynthesis: strategy -> pathway -> reaction.

        Layer 1: Strategic disconnections and route planning
        Layer 2: Pathway enumeration and ranking
        Layer 3: Individual reaction feasibility and condition selection
        """
        target_smiles = data.get("smiles", "")
        if not isinstance(target_smiles, str) or not target_smiles:
            raise ValueError("smiles is required for retrosynthesis planning")
        max_routes = int(data.get("max_routes", 10) or 10)
        planner = self._planner()
        routes = await planner.find_routes(target_smiles, max_routes=max_routes)
        return {
            "agent": self.name,
            "status": "planned",
            "target_smiles": target_smiles,
            "routes": routes,
            "layers": {
                "strategy": {
                    "route_count": len(routes),
                    "engine": planner.__class__.__name__,
                },
                "pathways": routes,
                "reactions": _route_reactions(routes),
            },
        }

    def _planner(self):
        if self.planner is None:
            from mf_retrosyn.aizynth.retrosyn import AiZynthRetrosyn

            self.planner = AiZynthRetrosyn.from_env()
        return self.planner


def _route_reactions(routes: list[dict]) -> list[str]:
    reactions: list[str] = []
    for route in routes:
        for step in route.get("steps") or []:
            reaction = step.get("reaction") if isinstance(step, dict) else None
            if isinstance(reaction, str) and reaction:
                reactions.append(reaction)
    return reactions
