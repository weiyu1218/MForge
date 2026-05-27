"""Heavy atom count 10-70"""
from critic_agent.rules.rule_base import CriticRule


class HeavyAtomsRule(CriticRule):
    def __init__(self, crg=None):
        super().__init__("rule_025", "Heavy atom count 10-70", crg)

    def evaluate(self, molecule_smiles, properties):
        heavy = properties.get("heavy_atoms", 0)
        if 10 <= heavy <= 70:
            verdict = "pass"
            score = 1.0
            reasoning = f"Heavy atoms={heavy} in [10,70]"
        else:
            verdict = "fail"
            score = max(0.0, 1.0 - abs(heavy - 40) / 70)
            reasoning = f"Heavy atoms={heavy} outside [10,70]"
        if self.crg:
            self.crg.add_belief(molecule_smiles, self.name, verdict, confidence=score, source_agent="critic")
        return {"verdict": verdict, "score": score, "reasoning": reasoning, "rule_id": self.rule_id, "rule_name": self.name}
