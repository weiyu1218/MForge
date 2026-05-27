"""Biodegradability assessment"""
from critic_agent.rules.rule_base import CriticRule


class BiodegradabilityRule(CriticRule):
    def __init__(self, crg=None):
        super().__init__("rule_048", "Biodegradability assessment", crg)

    def evaluate(self, molecule_smiles, properties):
        score_val = properties.get("biodegradability_score", 0.0)
        if score_val > 0.5:
            verdict = "pass"
            score = score_val
            reasoning = f"Biodegradability score={score_val:.2f} > 0.5"
        else:
            verdict = "fail"
            score = max(0.0, score_val)
            reasoning = f"Biodegradability score={score_val:.2f} <= 0.5 (persistent)"
        if self.crg:
            self.crg.add_belief(molecule_smiles, self.name, verdict, confidence=score, source_agent="critic")
        return {"verdict": verdict, "score": score, "reasoning": reasoning, "rule_id": self.rule_id, "rule_name": self.name}
