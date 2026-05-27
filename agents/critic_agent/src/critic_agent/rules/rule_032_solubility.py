"""Aqueous solubility logS > -6"""
from critic_agent.rules.rule_base import CriticRule


class SolubilityRule(CriticRule):
    def __init__(self, crg=None):
        super().__init__("rule_032", "Aqueous solubility logS > -6", crg)

    def evaluate(self, molecule_smiles, properties):
        logs = properties.get("log_s", -10.0)
        if logs > -6:
            verdict = "pass"
            score = 1.0
            reasoning = f"logS={logs:.1f} > -6 (soluble)"
        else:
            verdict = "fail"
            score = max(0.0, (logs + 10) / 4.0)
            reasoning = f"logS={logs:.1f} <= -6 (poor solubility)"
        if self.crg:
            self.crg.add_belief(molecule_smiles, self.name, verdict, confidence=score, source_agent="critic")
        return {"verdict": verdict, "score": score, "reasoning": reasoning, "rule_id": self.rule_id, "rule_name": self.name}
