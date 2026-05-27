"""Topological polar surface area within 20-140 Ang^2"""
from critic_agent.rules.rule_base import CriticRule


class TPSARangeRule(CriticRule):
    def __init__(self, crg=None):
        super().__init__("rule_017", "Topological polar surface area within 20-140 Ang^2", crg)

    def evaluate(self, molecule_smiles, properties):
        tpsa = properties.get("tpsa", 0)
        if 20 <= tpsa <= 140:
            verdict = "pass"
            score = 1.0
            reasoning = f"TPSA={tpsa:.1f} in [20,140]"
        else:
            verdict = "fail"
            score = max(0.0, 1.0 - abs(tpsa - 80) / 140)
            reasoning = f"TPSA={tpsa:.1f} outside [20,140]"
        if self.crg:
            self.crg.add_belief(molecule_smiles, self.name, verdict, confidence=score, source_agent="critic")
        return {"verdict": verdict, "score": score, "reasoning": reasoning, "rule_id": self.rule_id, "rule_name": self.name}
