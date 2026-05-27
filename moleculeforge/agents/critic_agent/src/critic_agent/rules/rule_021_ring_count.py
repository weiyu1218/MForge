"""Ring count 1-6 rings"""
from critic_agent.rules.rule_base import CriticRule


class RingCountRule(CriticRule):
    def __init__(self, crg=None):
        super().__init__("rule_021", "Ring count 1-6 rings", crg)

    def evaluate(self, molecule_smiles, properties):
        rings = properties.get("ring_count", 0)
        if 1 <= rings <= 6:
            verdict = "pass"
            score = 1.0
            reasoning = f"Ring count={rings} in [1,6]"
        else:
            verdict = "fail"
            score = max(0.0, 1.0 - abs(rings - 3) / 6)
            reasoning = f"Ring count={rings} outside [1,6]"
        if self.crg:
            self.crg.add_belief(molecule_smiles, self.name, verdict, confidence=score, source_agent="critic")
        return {"verdict": verdict, "score": score, "reasoning": reasoning, "rule_id": self.rule_id, "rule_name": self.name}
