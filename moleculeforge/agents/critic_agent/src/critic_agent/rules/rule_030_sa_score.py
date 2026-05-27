"""Synthetic accessibility score <= 6.0"""
from critic_agent.rules.rule_base import CriticRule


class SAScoreRule(CriticRule):
    def __init__(self, crg=None):
        super().__init__("rule_030", "Synthetic accessibility score <= 6.0", crg)

    def evaluate(self, molecule_smiles, properties):
        sa = properties.get("sa_score", 10.0)
        if sa <= 6.0:
            verdict = "pass"
            score = 1.0
            reasoning = f"SA score={sa:.1f} <= 6.0"
        else:
            verdict = "fail"
            score = max(0.0, 1.0 - (sa - 6.0) / 4.0)
            reasoning = f"SA score={sa:.1f} > 6.0"
        if self.crg:
            self.crg.add_belief(molecule_smiles, self.name, verdict, confidence=score, source_agent="critic")
        return {"verdict": verdict, "score": score, "reasoning": reasoning, "rule_id": self.rule_id, "rule_name": self.name}
