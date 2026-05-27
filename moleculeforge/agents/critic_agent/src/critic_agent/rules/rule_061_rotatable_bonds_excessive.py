"""Rotatable bonds > 10 violates Veber oral bioavailability threshold"""
from critic_agent.rules.rule_base import CriticRule


class RotatableBondsExcessiveRule(CriticRule):
    def __init__(self, crg=None):
        super().__init__("rule_061", "Rotatable bonds > 10 (Veber oral bioavailability)", crg)

    def evaluate(self, molecule_smiles, properties):
        n_rotb = properties.get("n_rotatable_bonds", 0)
        if n_rotb > 10:
            verdict = "fail"
            score = max(0.0, 1.0 - (n_rotb - 10) * 0.1)
            reasoning = f"Rotatable bonds={n_rotb} > 10, oral bioavailability may be low"
        else:
            verdict = "pass"
            score = min(1.0, n_rotb / 10)
            reasoning = f"Rotatable bonds={n_rotb} <= 10 (Veber compliant)"
        if self.crg:
            self.crg.add_belief(molecule_smiles, self.name, verdict, confidence=score, source_agent="critic")
        return {"verdict": verdict, "score": score, "reasoning": reasoning, "rule_id": self.rule_id, "rule_name": self.name}
