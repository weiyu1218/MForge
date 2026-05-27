"""Undesirable/toxic fragments check"""
from critic_agent.rules.rule_base import CriticRule


class UndesirableFragmentsRule(CriticRule):
    def __init__(self, crg=None):
        super().__init__("rule_028", "Undesirable/toxic fragments check", crg)

    def evaluate(self, molecule_smiles, properties):
        frags = properties.get("undesirable_fragments", 0)
        verdict = "pass" if frags == 0 else "fail"
        score = max(0.0, 1.0 - frags * 0.2)
        reasoning = f"Undesirable fragments={frags}"
        if self.crg:
            self.crg.add_belief(molecule_smiles, self.name, verdict, confidence=score, source_agent="critic")
        return {"verdict": verdict, "score": score, "reasoning": reasoning, "rule_id": self.rule_id, "rule_name": self.name}
