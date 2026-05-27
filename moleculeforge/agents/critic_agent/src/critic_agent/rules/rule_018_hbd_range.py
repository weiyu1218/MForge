"""Hydrogen bond donors <= 5"""
from critic_agent.rules.rule_base import CriticRule


class HBDRangeRule(CriticRule):
    def __init__(self, crg=None):
        super().__init__("rule_018", "Hydrogen bond donors <= 5", crg)

    def evaluate(self, molecule_smiles, properties):
        hbd = properties.get("hbd", 0)
        if hbd <= 5:
            verdict = "pass"
            score = 1.0
            reasoning = f"HBD={hbd} <= 5"
        else:
            verdict = "fail"
            score = max(0.0, 1.0 - (hbd - 5) / 5)
            reasoning = f"HBD={hbd} > 5"
        if self.crg:
            self.crg.add_belief(molecule_smiles, self.name, verdict, confidence=score, source_agent="critic")
        return {"verdict": verdict, "score": score, "reasoning": reasoning, "rule_id": self.rule_id, "rule_name": self.name}
