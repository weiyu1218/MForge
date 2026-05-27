"""Fraction sp3 carbons >= 0.25"""
from critic_agent.rules.rule_base import CriticRule


class Fsp3Rule(CriticRule):
    def __init__(self, crg=None):
        super().__init__("rule_023", "Fraction sp3 carbons >= 0.25", crg)

    def evaluate(self, molecule_smiles, properties):
        fsp3 = properties.get("fsp3", 0.0)
        if fsp3 >= 0.25:
            verdict = "pass"
            score = 1.0
            reasoning = f"Fsp3={fsp3:.2f} >= 0.25"
        else:
            verdict = "fail"
            score = max(0.0, fsp3 / 0.25)
            reasoning = f"Fsp3={fsp3:.2f} < 0.25"
        if self.crg:
            self.crg.add_belief(molecule_smiles, self.name, verdict, confidence=score, source_agent="critic")
        return {"verdict": verdict, "score": score, "reasoning": reasoning, "rule_id": self.rule_id, "rule_name": self.name}
