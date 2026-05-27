"""Normalized rotatable bond ratio <= 0.35"""
from critic_agent.rules.rule_base import CriticRule


class NormalizedRotatableBondsRule(CriticRule):
    def __init__(self, crg=None):
        super().__init__("rule_095", "Normalized rotatable bond ratio <= 0.35", crg)

    def evaluate(self, molecule_smiles, properties):
        nrb_ratio = properties.get("normalized_rotatable_bonds", 0.0)
        if nrb_ratio <= 0.35:
            verdict = "pass"
            score = max(0.0, 1.0 - nrb_ratio * 1.5)
            reasoning = f"NRB ratio={nrb_ratio:.2f} <= 0.35 (acceptable flexibility)"
        else:
            verdict = "fail"
            score = max(0.0, 1.0 - nrb_ratio)
            reasoning = f"NRB ratio={nrb_ratio:.2f} > 0.35 — excessive molecular flexibility"
        if self.crg:
            self.crg.add_belief(molecule_smiles, self.name, verdict, confidence=score, source_agent="critic")
        return {"verdict": verdict, "score": score, "reasoning": reasoning, "rule_id": self.rule_id, "rule_name": self.name}
