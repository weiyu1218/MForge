"""Brenk fragment-based toxicity alerts"""
from critic_agent.rules.rule_base import CriticRule


class BrenkRule(CriticRule):
    def __init__(self, crg=None):
        super().__init__("rule_004", "Brenk fragment-based toxicity alerts", crg)

    def evaluate(self, molecule_smiles, properties):
        undesirable_frags = properties.get("undesirable_fragments", 0)
        verdict = "pass" if undesirable_frags == 0 else "fail"
        score = max(0.0, 1.0 - undesirable_frags * 0.2)
        reasoning = f"Brenk undesirable fragments={undesirable_frags}"
        if self.crg:
            self.crg.add_belief(molecule_smiles, self.name, verdict, confidence=score, source_agent="critic")
        return {"verdict": verdict, "score": score, "reasoning": reasoning, "rule_id": self.rule_id, "rule_name": self.name}
