"""Promiscuity index > 0.3 indicates multi-target binding toxicity risk"""
from critic_agent.rules.rule_base import CriticRule


class PromiscuityIndexRule(CriticRule):
    def __init__(self, crg=None):
        super().__init__("rule_067", "Promiscuity index <= 0.3", crg)

    def evaluate(self, molecule_smiles, properties):
        pmi = properties.get("promiscuity_index", 0.0)
        threshold = 0.3
        if pmi > threshold:
            verdict = "fail"
            score = max(0.0, 1.0 - pmi)
            reasoning = f"PMI={pmi:.3f} > {threshold}, may bind multiple targets and cause toxicity"
        else:
            verdict = "pass"
            score = 1.0 - pmi
            reasoning = f"PMI={pmi:.3f} <= {threshold}, selectivity profile acceptable"
        if self.crg:
            self.crg.add_belief(molecule_smiles, self.name, verdict, confidence=score, source_agent="critic")
        return {"verdict": verdict, "score": score, "reasoning": reasoning, "rule_id": self.rule_id, "rule_name": self.name}
