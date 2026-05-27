"""Net formal charge in [-1, 1]"""
from critic_agent.rules.rule_base import CriticRule


class FormalChargeRangeRule(CriticRule):
    def __init__(self, crg=None):
        super().__init__("rule_090", "Net formal charge in [-1, 1]", crg)

    def evaluate(self, molecule_smiles, properties):
        charge = properties.get("formal_charge", 0)
        if -1 <= charge <= 1:
            verdict = "pass"
            score = 1.0
            reasoning = f"Formal charge={charge} within [-1, 1]"
        else:
            verdict = "fail"
            score = max(0.0, 1.0 - (abs(charge) - 1) * 0.3)
            reasoning = f"Formal charge={charge} outside [-1, 1] — permeability / solubility concerns"
        if self.crg:
            self.crg.add_belief(molecule_smiles, self.name, verdict, confidence=score, source_agent="critic")
        return {"verdict": verdict, "score": score, "reasoning": reasoning, "rule_id": self.rule_id, "rule_name": self.name}
