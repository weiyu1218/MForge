"""Quantitative Estimate of Drug-likeness >= 0.5"""
from critic_agent.rules.rule_base import CriticRule


class QEDScoreRule(CriticRule):
    def __init__(self, crg=None):
        super().__init__("rule_029", "Quantitative Estimate of Drug-likeness >= 0.5", crg)

    def evaluate(self, molecule_smiles, properties):
        qed = properties.get("qed", 0.0)
        if qed >= 0.5:
            verdict = "pass"
            score = min(1.0, qed)
            reasoning = f"QED={qed:.2f} >= 0.5"
        else:
            verdict = "fail"
            score = qed
            reasoning = f"QED={qed:.2f} < 0.5"
        if self.crg:
            self.crg.add_belief(molecule_smiles, self.name, verdict, confidence=score, source_agent="critic")
        return {"verdict": verdict, "score": score, "reasoning": reasoning, "rule_id": self.rule_id, "rule_name": self.name}
