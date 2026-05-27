"""Scientific Critic Agent - internal adversary for bias prevention."""
import importlib
import json
import pkgutil
from pathlib import Path

from mf_agents.base.agent import BaseAgent
from mf_agents.crg.graph import ChemicalReasoningGraph

from critic_agent.rules.rule_base import CriticRule


class ScientificCriticAgent(BaseAgent):
    def __init__(self, message_bus=None):
        super().__init__("critic_agent", message_bus)
        self._subscription_subjects = ["agent.critic.request", "orchestrator.critic.evaluate"]
        self.crg = ChemicalReasoningGraph()
        self.rules: list[CriticRule] = []
        self._load_rules()

    def _load_rules(self) -> None:
        """Auto-discover and load all rule classes from the rules package."""
        import critic_agent.rules as rules_pkg

        rules_path = Path(rules_pkg.__path__[0])
        for module_info in pkgutil.iter_modules([str(rules_path)]):
            if module_info.name == "rule_base" or module_info.name.startswith("_"):
                continue
            try:
                module = importlib.import_module(f"critic_agent.rules.{module_info.name}")
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (
                        isinstance(attr, type)
                        and issubclass(attr, CriticRule)
                        and attr is not CriticRule
                    ):
                        self.rules.append(attr(self.crg))
                        break
            except Exception as e:
                self.logger.warning(f"Could not load rule module {module_info.name}: {e}")

    async def handle_message(self, subject, payload, reply_to=""):
        data = json.loads(payload) if isinstance(payload, bytes) else payload
        if "evaluate" in subject or "request" in subject:
            result = await self.evaluate_molecule(data)
            if reply_to:
                await self.publish(reply_to, json.dumps(result).encode())

    async def evaluate_molecule(self, data: dict) -> dict:
        smiles = data.get("smiles", "")
        properties = data.get("properties", {})
        results = []
        passed = 0
        failed = 0

        for rule in self.rules:
            try:
                verdict = rule.evaluate(smiles, properties)
                results.append(verdict)
                if verdict.get("verdict") == "pass":
                    passed += 1
                else:
                    failed += 1
            except Exception as e:
                results.append({
                    "rule_id": rule.rule_id,
                    "rule_name": rule.name,
                    "verdict": "error",
                    "score": 0.0,
                    "reasoning": str(e),
                })
                failed += 1

        overall_verdict = "pass" if failed == 0 else "fail"
        return {
            "smiles": smiles,
            "verdict": overall_verdict,
            "passed": passed,
            "failed": failed,
            "total_rules": len(self.rules),
            "rule_results": results,
        }

    async def process(self, data):
        return await self.evaluate_molecule(data)
