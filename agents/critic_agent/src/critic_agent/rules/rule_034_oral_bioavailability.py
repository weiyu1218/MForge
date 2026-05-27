"""Oral bioavailability >= 30%"""
from critic_agent.rules.rule_base import CriticRule


class OralBioavailabilityRule(CriticRule):
    def __init__(self, crg=None):
        super().__init__("rule_034", "Oral bioavailability >= 30%", crg)

    def evaluate(self, molecule_smiles, properties):
        f_oral = properties.get("oral_bioavailability", 0.0)
        if f_oral >= 0.3:
            verdict = "pass"
            score = min(1.0, f_oral)
            reasoning = f"F_oral={f_oral:.1%} >= 30%"
        else:
            verdict = "fail"
            score = f_oral / 0.3
            reasoning = f"F_oral={f_oral:.1%} < 30%"
        if self.crg:
            self.crg.add_belief(molecule_smiles, self.name, verdict, confidence=score, source_agent="critic")
        return {"verdict": verdict, "score": score, "reasoning": reasoning, "rule_id": self.rule_id, "rule_name": self.name}
