"""Carcinogenicity structural alert assessment"""
from critic_agent.rules.rule_base import CriticRule


class CarcinogenicityRule(CriticRule):
    def __init__(self, crg=None):
        super().__init__("rule_050", "Carcinogenicity structural alert assessment", crg)

    def evaluate(self, molecule_smiles, properties):
        risk = properties.get("carcinogenicity_risk", 0.0)
        if risk < 0.5:
            verdict = "pass"
            score = 1.0 - risk
            reasoning = f"Carcinogenicity risk={risk:.2f}"
        else:
            verdict = "fail"
            score = max(0.0, 1.0 - risk)
            reasoning = f"Carcinogenicity risk={risk:.2f} (high concern)"
        if self.crg:
            self.crg.add_belief(molecule_smiles, self.name, verdict, confidence=score, source_agent="critic")
        return {"verdict": verdict, "score": score, "reasoning": reasoning, "rule_id": self.rule_id, "rule_name": self.name}
