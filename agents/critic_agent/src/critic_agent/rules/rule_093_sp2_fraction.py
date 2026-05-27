"""Fraction sp2 hybridized carbons <= 0.7"""
from critic_agent.rules.rule_base import CriticRule


class Sp2FractionRule(CriticRule):
    def __init__(self, crg=None):
        super().__init__("rule_093", "Fraction sp2 carbons <= 0.7", crg)

    def evaluate(self, molecule_smiles, properties):
        fsp2 = properties.get("fraction_csp2", 0.0)
        if fsp2 <= 0.7:
            verdict = "pass"
            score = max(0.0, 1.0 - fsp2)
            reasoning = f"Fsp2={fsp2:.2f} <= 0.7 (non-planar)"
        else:
            verdict = "fail"
            score = max(0.0, 1.0 - fsp2) * 0.6
            reasoning = f"Fsp2={fsp2:.2f} > 0.7 — molecule too flat, solubility / selectivity risk"
        if self.crg:
            self.crg.add_belief(molecule_smiles, self.name, verdict, confidence=score, source_agent="critic")
        return {"verdict": verdict, "score": score, "reasoning": reasoning, "rule_id": self.rule_id, "rule_name": self.name}
