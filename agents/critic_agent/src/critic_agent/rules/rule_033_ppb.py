"""Plasma protein binding fraction < 0.99"""
from critic_agent.rules.rule_base import CriticRule


class PlasmaProteinBindingRule(CriticRule):
    def __init__(self, crg=None):
        super().__init__("rule_033", "Plasma protein binding fraction < 0.99", crg)

    def evaluate(self, molecule_smiles, properties):
        ppb = properties.get("ppb", 1.0)
        if ppb < 0.99:
            verdict = "pass"
            score = 1.0
            reasoning = f"PPB={ppb:.3f} < 0.99"
        else:
            verdict = "fail"
            score = max(0.0, 1.0 - ppb)
            reasoning = f"PPB={ppb:.3f} >= 0.99 (excessive binding)"
        if self.crg:
            self.crg.add_belief(molecule_smiles, self.name, verdict, confidence=score, source_agent="critic")
        return {"verdict": verdict, "score": score, "reasoning": reasoning, "rule_id": self.rule_id, "rule_name": self.name}
