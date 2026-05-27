"""Non-specific protein binding assessment"""
from critic_agent.rules.rule_base import CriticRule


class ProteinBindingRule(CriticRule):
    def __init__(self, crg=None):
        super().__init__("rule_059", "Non-specific protein binding assessment", crg)

    def evaluate(self, molecule_smiles, properties):
        risk = properties.get("nonspecific_binding_risk", 0.0)
        if risk < 0.7:
            verdict = "pass"
            score = 1.0 - risk * 0.7
            reasoning = f"Non-specific binding={risk:.2f}"
        else:
            verdict = "fail"
            score = max(0.0, 1.0 - risk)
            reasoning = f"Non-specific binding={risk:.2f} (high)"
        if self.crg:
            self.crg.add_belief(molecule_smiles, self.name, verdict, confidence=score, source_agent="critic")
        return {"verdict": verdict, "score": score, "reasoning": reasoning, "rule_id": self.rule_id, "rule_name": self.name}
