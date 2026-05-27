"""Reactive metabolite formation risk (glutathione trapping positive)"""
from critic_agent.rules.rule_base import CriticRule


class ReactiveMetaboliteRule(CriticRule):
    def __init__(self, crg=None):
        super().__init__("rule_074", "No reactive metabolite predicted", crg)

    def evaluate(self, molecule_smiles, properties):
        has_reactive = properties.get("has_reactive_metabolite", False)
        if has_reactive:
            verdict = "fail"
            score = 0.2
            reasoning = "Reactive metabolite predicted (glutathione trapping positive), covalent toxicity risk"
        else:
            verdict = "pass"
            score = 1.0
            reasoning = "No reactive metabolite predicted"
        if self.crg:
            self.crg.add_belief(molecule_smiles, self.name, verdict, confidence=score, source_agent="critic")
        return {"verdict": verdict, "score": score, "reasoning": reasoning, "rule_id": self.rule_id, "rule_name": self.name}
