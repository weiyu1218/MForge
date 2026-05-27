"""SRB Agent - Synthesis Reality Bridge: compiles retrosynthesis into SSPs."""
import json

from mf_agents.base.agent import BaseAgent
from mf_agents.crg.graph import ChemicalReasoningGraph
from srb_agent.compiler import compile_ssp


class SRBAgent(BaseAgent):
    def __init__(self, message_bus=None):
        super().__init__("srb_agent", message_bus)
        self._subscription_subjects = ["agent.srb.request", "orchestrator.srb.compile"]
        self.crg = ChemicalReasoningGraph()

    async def handle_message(self, subject, payload, reply_to=""):
        data = json.loads(payload) if isinstance(payload, bytes) else {"raw": payload}
        result = await self.process(data)
        if reply_to:
            await self.publish(reply_to, json.dumps(result).encode())

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
        route = data.get("retrosyn_route")
        pathways = [route] if route is not None else data.get("pathways", [])
        protocols = []
        for retrosyn_route in pathways:
            ssp = await compile_ssp(molecule, retrosyn_route, run_id)
            protocols.append(_ssp_protocol_dict(ssp))
        return {
            "agent": self.name,
            "status": "compiled",
            "protocols": protocols,
        }


def _ssp_protocol_dict(ssp) -> dict:
    return {
        "ssp_id": ssp.ssp_id,
        "route_id": ssp.route_id,
        "target_smiles": ssp.target_smiles,
        "materials": [material.model_dump() for material in ssp.materials],
        "steps": [step.model_dump() for step in ssp.steps],
        "total_estimated_yield": ssp.total_estimated_yield,
        "total_estimated_cost_usd": ssp.total_estimated_cost_usd,
        "xdl_version": ssp.xdl_version,
    }
