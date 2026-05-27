"""Undefined stereocenter count <= 2"""
from critic_agent.rules.rule_base import CriticRule


class UndefinedStereocentersRule(CriticRule):
    def __init__(self, crg=None):
        super().__init__("rule_092", "Undefined stereocenter count <= 2", crg)

    def evaluate(self, molecule_smiles, properties):
        count = properties.get("undefined_stereocenters", 0)
        if count <= 2:
            verdict = "pass"
            score = max(0.0, 1.0 - count * 0.15)
            reasoning = f"Undefined stereocenters={count} <= 2 (acceptable)"
        else:
            verdict = "fail"
            score = max(0.0, 1.0 - count * 0.1)
            reasoning = f"Undefined stereocenters={count} > 2 — synthesis/purification risk"
        if self.crg:
            self.crg.add_belief(molecule_smiles, self.name, verdict, confidence=score, source_agent="critic")
        return {"verdict": verdict, "score": score, "reasoning": reasoning, "rule_id": self.rule_id, "rule_name": self.name}
