"""Reactive functional group count == 0"""
from critic_agent.rules.rule_base import CriticRule


class ReactiveGroupCountRule(CriticRule):
    def __init__(self, crg=None):
        super().__init__("rule_088", "Reactive functional group count == 0", crg)

    def evaluate(self, molecule_smiles, properties):
        count = properties.get("reactive_group_count", 0)
        if count == 0:
            verdict = "pass"
            score = 1.0
            reasoning = "No reactive functional groups detected"
        else:
            verdict = "fail"
            score = max(0.0, 1.0 - count * 0.5)
            reasoning = f"{count} reactive group(s) — covalent binding / HTS interference risk"
        if self.crg:
            self.crg.add_belief(molecule_smiles, self.name, verdict, confidence=score, source_agent="critic")
        return {"verdict": verdict, "score": score, "reasoning": reasoning, "rule_id": self.rule_id, "rule_name": self.name}
