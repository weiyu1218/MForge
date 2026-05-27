"""Synthesis requires controlled substance precursor (regulatory risk)"""
from critic_agent.rules.rule_base import CriticRule


class ControlledPrecursorRule(CriticRule):
    def __init__(self, crg=None):
        super().__init__("rule_084", "No controlled substance precursor required", crg)

    def evaluate(self, molecule_smiles, properties):
        has_controlled = properties.get("has_controlled_precursor", False)
        precursor = properties.get("controlled_precursor_name", "unknown")
        if has_controlled:
            verdict = "fail"
            score = 0.1
            reasoning = f"Synthesis requires controlled precursor: {precursor}, regulatory compliance challenge"
        else:
            verdict = "pass"
            score = 1.0
            reasoning = "No controlled substance precursors required"
        if self.crg:
            self.crg.add_belief(molecule_smiles, self.name, verdict, confidence=score, source_agent="critic")
        return {"verdict": verdict, "score": score, "reasoning": reasoning, "rule_id": self.rule_id, "rule_name": self.name}
