"""Structural alerts for mutagenicity"""
from critic_agent.rules.rule_base import CriticRule


class StructuralMutagenicityRule(CriticRule):
    def __init__(self, crg=None):
        super().__init__("rule_049", "Structural alerts for mutagenicity", crg)

    def evaluate(self, molecule_smiles, properties):
        alerts = properties.get("mutagenic_alerts", 0)
        verdict = "pass" if alerts == 0 else "fail"
        score = max(0.0, 1.0 - alerts * 0.25)
        reasoning = f"Structural mutagenic alerts={alerts}"
        if self.crg:
            self.crg.add_belief(molecule_smiles, self.name, verdict, confidence=score, source_agent="critic")
        return {"verdict": verdict, "score": score, "reasoning": reasoning, "rule_id": self.rule_id, "rule_name": self.name}
