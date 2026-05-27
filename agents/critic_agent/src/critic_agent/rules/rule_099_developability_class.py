"""Developability Classification Score >= 3 (5-point scale)"""
from critic_agent.rules.rule_base import CriticRule


class DevelopabilityClassRule(CriticRule):
    def __init__(self, crg=None):
        super().__init__("rule_099", "Developability Classification >= 3", crg)

    def evaluate(self, molecule_smiles, properties):
        dc_score = properties.get("developability_class", 3)
        if dc_score >= 3:
            verdict = "pass"
            score = min(1.0, dc_score / 5.0)
            reasoning = f"Developability class={dc_score}/5 (acceptable)"
        else:
            verdict = "fail"
            score = max(0.0, dc_score / 3.0)
            reasoning = f"Developability class={dc_score}/5 < 3 — high attrition risk"
        if self.crg:
            self.crg.add_belief(molecule_smiles, self.name, verdict, confidence=score, source_agent="critic")
        return {"verdict": verdict, "score": score, "reasoning": reasoning, "rule_id": self.rule_id, "rule_name": self.name}
