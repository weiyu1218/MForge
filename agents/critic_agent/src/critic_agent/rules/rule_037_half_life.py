"""Plasma half-life 2-24 hours"""
from critic_agent.rules.rule_base import CriticRule


class HalfLifeRule(CriticRule):
    def __init__(self, crg=None):
        super().__init__("rule_037", "Plasma half-life 2-24 hours", crg)

    def evaluate(self, molecule_smiles, properties):
        t12 = properties.get("half_life_h", 0.0)
        if 2 <= t12 <= 24:
            verdict = "pass"
            score = 1.0
            reasoning = f"t1/2={t12:.1f}h in [2,24]"
        else:
            verdict = "fail"
            score = max(0.0, 1.0 - abs(t12 - 12) / 24)
            reasoning = f"t1/2={t12:.1f}h outside [2,24]"
        if self.crg:
            self.crg.add_belief(molecule_smiles, self.name, verdict, confidence=score, source_agent="critic")
        return {"verdict": verdict, "score": score, "reasoning": reasoning, "rule_id": self.rule_id, "rule_name": self.name}
