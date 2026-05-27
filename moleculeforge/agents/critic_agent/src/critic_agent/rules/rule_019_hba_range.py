"""Hydrogen bond acceptors <= 10"""
from critic_agent.rules.rule_base import CriticRule


class HBARangeRule(CriticRule):
    def __init__(self, crg=None):
        super().__init__("rule_019", "Hydrogen bond acceptors <= 10", crg)

    def evaluate(self, molecule_smiles, properties):
        hba = properties.get("hba", 0)
        if hba <= 10:
            verdict = "pass"
            score = 1.0
            reasoning = f"HBA={hba} <= 10"
        else:
            verdict = "fail"
            score = max(0.0, 1.0 - (hba - 10) / 10)
            reasoning = f"HBA={hba} > 10"
        if self.crg:
            self.crg.add_belief(molecule_smiles, self.name, verdict, confidence=score, source_agent="critic")
        return {"verdict": verdict, "score": score, "reasoning": reasoning, "rule_id": self.rule_id, "rule_name": self.name}
