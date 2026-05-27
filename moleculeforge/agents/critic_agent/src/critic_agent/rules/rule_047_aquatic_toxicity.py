"""Aquatic toxicity log(LC50) > -3"""
from critic_agent.rules.rule_base import CriticRule


class AquaticToxicityRule(CriticRule):
    def __init__(self, crg=None):
        super().__init__("rule_047", "Aquatic toxicity log(LC50) > -3", crg)

    def evaluate(self, molecule_smiles, properties):
        loglc50 = properties.get("log_lc50", -10.0)
        if loglc50 > -3:
            verdict = "pass"
            score = min(1.0, (loglc50 + 3) / 3)
            reasoning = f"log(LC50)={loglc50:.1f} > -3"
        else:
            verdict = "fail"
            score = max(0.0, (loglc50 + 10) / 7.0)
            reasoning = f"log(LC50)={loglc50:.1f} <= -3 (toxic to aquatic life)"
        if self.crg:
            self.crg.add_belief(molecule_smiles, self.name, verdict, confidence=score, source_agent="critic")
        return {"verdict": verdict, "score": score, "reasoning": reasoning, "rule_id": self.rule_id, "rule_name": self.name}
