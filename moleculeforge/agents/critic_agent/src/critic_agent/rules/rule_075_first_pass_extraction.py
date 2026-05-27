"""Hepatic extraction ratio > 0.7 indicates oral bioavailability < 20%"""
from critic_agent.rules.rule_base import CriticRule


class FirstPassExtractionRule(CriticRule):
    def __init__(self, crg=None):
        super().__init__("rule_075", "Hepatic extraction ratio <= 0.7", crg)

    def evaluate(self, molecule_smiles, properties):
        eh = properties.get("hepatic_extraction_ratio", 0.0)
        threshold = 0.7
        if eh > threshold:
            verdict = "fail"
            score = max(0.0, 1.0 - eh)
            reasoning = f"Hepatic extraction ratio Eh={eh:.2f} > {threshold}, oral bioavailability may be < 20%"
        else:
            verdict = "pass"
            score = min(1.0, 1.0 - eh)
            reasoning = f"Hepatic extraction ratio Eh={eh:.2f} <= {threshold}, acceptable first-pass"
        if self.crg:
            self.crg.add_belief(molecule_smiles, self.name, verdict, confidence=score, source_agent="critic")
        return {"verdict": verdict, "score": score, "reasoning": reasoning, "rule_id": self.rule_id, "rule_name": self.name}
