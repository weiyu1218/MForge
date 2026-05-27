"""Molecular complexity (Bertz CT) <= 1500"""
from critic_agent.rules.rule_base import CriticRule


class BertzComplexityRule(CriticRule):
    def __init__(self, crg=None):
        super().__init__("rule_089", "Molecular complexity Bertz CT <= 1500", crg)

    def evaluate(self, molecule_smiles, properties):
        bertz = properties.get("bertz_ct", 0.0)
        if bertz <= 1500:
            verdict = "pass"
            score = max(0.0, 1.0 - bertz / 2000)
            reasoning = f"Bertz CT={bertz:.0f} <= 1500 (acceptable complexity)"
        else:
            verdict = "fail"
            score = max(0.0, 3000 / bertz)
            reasoning = f"Bertz CT={bertz:.0f} > 1500 (excessive topological complexity)"
        if self.crg:
            self.crg.add_belief(molecule_smiles, self.name, verdict, confidence=score, source_agent="critic")
        return {"verdict": verdict, "score": score, "reasoning": reasoning, "rule_id": self.rule_id, "rule_name": self.name}
