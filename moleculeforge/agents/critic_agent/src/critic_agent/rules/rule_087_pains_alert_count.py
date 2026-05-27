"""PAINS alert count == 0 (pan-assay interference compounds)"""
from critic_agent.rules.rule_base import CriticRule


class PAINSAlertCountRule(CriticRule):
    def __init__(self, crg=None):
        super().__init__("rule_087", "PAINS substructure alert count == 0", crg)

    def evaluate(self, molecule_smiles, properties):
        count = properties.get("pains_alert_count", 0)
        if count == 0:
            verdict = "pass"
            score = 1.0
            reasoning = "No PAINS alerts detected"
        else:
            verdict = "fail"
            score = max(0.0, 1.0 - count * 0.25)
            reasoning = f"{count} PAINS alert(s) detected — assay interference risk"
        if self.crg:
            self.crg.add_belief(molecule_smiles, self.name, verdict, confidence=score, source_agent="critic")
        return {"verdict": verdict, "score": score, "reasoning": reasoning, "rule_id": self.rule_id, "rule_name": self.name}
