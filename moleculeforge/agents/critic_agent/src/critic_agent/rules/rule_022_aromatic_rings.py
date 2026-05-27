"""Aromatic ring count 0-4"""
from critic_agent.rules.rule_base import CriticRule


class AromaticRingsRule(CriticRule):
    def __init__(self, crg=None):
        super().__init__("rule_022", "Aromatic ring count 0-4", crg)

    def evaluate(self, molecule_smiles, properties):
        ar_rings = properties.get("aromatic_rings", 0)
        if ar_rings <= 4:
            verdict = "pass"
            score = 1.0
            reasoning = f"Aromatic rings={ar_rings} <= 4"
        else:
            verdict = "fail"
            score = max(0.0, 1.0 - (ar_rings - 4) / 4)
            reasoning = f"Aromatic rings={ar_rings} > 4"
        if self.crg:
            self.crg.add_belief(molecule_smiles, self.name, verdict, confidence=score, source_agent="critic")
        return {"verdict": verdict, "score": score, "reasoning": reasoning, "rule_id": self.rule_id, "rule_name": self.name}
