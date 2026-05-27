"""Generator Coordinator Agent - Coordinates multiple generators based on routing (Agent-2)."""
import json

from mf_agents.base.agent import BaseAgent
from mf_agents.crg.graph import ChemicalReasoningGraph
from mf_core.routing.task_router import GENERATOR_NAMES


DEFAULT_GENERATORS = ["hfm_3d", "fragfm"]
if not set(DEFAULT_GENERATORS).issubset(GENERATOR_NAMES):
    raise RuntimeError("Default generators must be present in GENERATOR_NAMES")


class GeneratorCoordAgent(BaseAgent):
    def __init__(self, message_bus=None):
        super().__init__("generator_coord", message_bus)
        self._subscription_subjects = [
            "agent.generator_coord.request",
            "orchestrator.generate.request",
        ]
        self.crg = ChemicalReasoningGraph()
        self.generators = list(GENERATOR_NAMES)

    async def handle_message(self, subject, payload, reply_to=""):
        data = json.loads(payload) if isinstance(payload, bytes) else {"raw": payload}
        result = await self.process(data)
        if reply_to:
            await self.publish(reply_to, json.dumps(result).encode())

    async def process(self, data):
        """Route generation request to appropriate generator(s) based on objectives.

        Selects generation strategy based on target properties, complexity,
        and available generators. Dispatches to one or more generator backends.
        """
        strategy = data.get("generation_strategy", "auto")
        objectives = data.get("objectives", {})
        return {
            "agent": self.name,
            "status": "dispatched",
            "strategy": strategy,
            "selected_generators": self._select_generators(strategy, objectives),
            "available_generators": self.generators,
        }

    def _select_generators(self, strategy: str, objectives: dict) -> list:
        if strategy == "auto":
            complexity = objectives.get("complexity", "medium")
            if complexity == "high":
                return ["evomol_rl", "fragfm"]
            elif complexity == "low":
                return ["hfm_3d"]
            return list(DEFAULT_GENERATORS)
        elif strategy == "all":
            return list(self.generators)
        elif strategy in self.generators:
            return [strategy]
        return list(DEFAULT_GENERATORS)
