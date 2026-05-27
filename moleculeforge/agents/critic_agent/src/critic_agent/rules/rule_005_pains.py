"""PAINS pan-assay interference compounds"""
from critic_agent.rules.rule_base import CriticRule


class PAINSRule(CriticRule):
    def __init__(self, crg=None):
        super().__init__("rule_005", "PAINS pan-assay interference compounds", crg)

    def evaluate(self, molecule_smiles, properties):
        pains_alerts = properties.get("pains_alerts", 0)
        verdict = "pass" if pains_alerts == 0 else "fail"
        score = max(0.0, 1.0 - pains_alerts * 0.33)
        reasoning = f"PAINS alerts detected={pains_alerts}"
        if self.crg:
            self.crg.add_belief(molecule_smiles, self.name, verdict, confidence=score, source_agent="critic")
        return {"verdict": verdict, "score": score, "reasoning": reasoning, "rule_id": self.rule_id, "rule_name": self.name}
