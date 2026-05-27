"""Rotatable bonds <= 10"""
from critic_agent.rules.rule_base import CriticRule


class RotatableBondsRule(CriticRule):
    def __init__(self, crg=None):
        super().__init__("rule_020", "Rotatable bonds <= 10", crg)

    def evaluate(self, molecule_smiles, properties):
        rot = properties.get("rotatable_bonds", 0)
        if rot <= 10:
            verdict = "pass"
            score = 1.0
            reasoning = f"Rotatable bonds={rot} <= 10"
        else:
            verdict = "fail"
            score = max(0.0, 1.0 - (rot - 10) / 10)
            reasoning = f"Rotatable bonds={rot} > 10"
        if self.crg:
            self.crg.add_belief(molecule_smiles, self.name, verdict, confidence=score, source_agent="critic")
        return {"verdict": verdict, "score": score, "reasoning": reasoning, "rule_id": self.rule_id, "rule_name": self.name}
