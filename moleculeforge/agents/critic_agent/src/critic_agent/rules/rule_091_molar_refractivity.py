"""Molar refractivity in [40, 130]"""
from critic_agent.rules.rule_base import CriticRule


class MolarRefractivityRule(CriticRule):
    def __init__(self, crg=None):
        super().__init__("rule_091", "Molar refractivity in [40, 130]", crg)

    def evaluate(self, molecule_smiles, properties):
        mr_val = properties.get("molar_refractivity", 60)
        if 40 <= mr_val <= 130:
            verdict = "pass"
            score = 1.0
            reasoning = f"Molar refractivity={mr_val:.1f} within [40, 130]"
        else:
            verdict = "fail"
            score = max(0.0, min(1.0, 130 / mr_val if mr_val > 130 else mr_val / 40))
            reasoning = f"Molar refractivity={mr_val:.1f} outside [40, 130]"
        if self.crg:
            self.crg.add_belief(molecule_smiles, self.name, verdict, confidence=score, source_agent="critic")
        return {"verdict": verdict, "score": score, "reasoning": reasoning, "rule_id": self.rule_id, "rule_name": self.name}
