"""Fsp3 < 0.25 indicates over-aromatization, poor solubility and hERG risk"""
from critic_agent.rules.rule_base import CriticRule


class Fsp3TooLowRule(CriticRule):
    def __init__(self, crg=None):
        super().__init__("rule_065", "Fsp3 >= 0.25 (aromaticity balance)", crg)

    def evaluate(self, molecule_smiles, properties):
        fsp3 = properties.get("fsp3", 0.0)
        threshold = 0.25
        if fsp3 < threshold:
            verdict = "fail"
            score = max(0.0, fsp3 / threshold)
            reasoning = f"Fsp3={fsp3:.2f} < {threshold}, over-aromatized: solubility and hERG risk"
        else:
            verdict = "pass"
            score = min(1.0, fsp3)
            reasoning = f"Fsp3={fsp3:.2f} >= {threshold}, good aromatic/aliphatic balance"
        if self.crg:
            self.crg.add_belief(molecule_smiles, self.name, verdict, confidence=score, source_agent="critic")
        return {"verdict": verdict, "score": score, "reasoning": reasoning, "rule_id": self.rule_id, "rule_name": self.name}
