"""Chiral centers <= 8"""
from critic_agent.rules.rule_base import CriticRule


class ChiralCentersRule(CriticRule):
    def __init__(self, crg=None):
        super().__init__("rule_024", "Chiral centers <= 8", crg)

    def evaluate(self, molecule_smiles, properties):
        chiral = properties.get("chiral_centers", 0)
        if chiral <= 8:
            verdict = "pass"
            score = 1.0
            reasoning = f"Chiral centers={chiral} <= 8"
        else:
            verdict = "fail"
            score = max(0.0, 1.0 - (chiral - 8) / 8)
            reasoning = f"Chiral centers={chiral} > 8"
        if self.crg:
            self.crg.add_belief(molecule_smiles, self.name, verdict, confidence=score, source_agent="critic")
        return {"verdict": verdict, "score": score, "reasoning": reasoning, "rule_id": self.rule_id, "rule_name": self.name}
