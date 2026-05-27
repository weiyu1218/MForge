"""Estimated synthesis cost > $10k/g exceeds preclinical budget"""
from critic_agent.rules.rule_base import CriticRule


class SynthesisCostRule(CriticRule):
    def __init__(self, crg=None):
        super().__init__("rule_082", "Synthesis cost <= $10,000/g", crg)

    def evaluate(self, molecule_smiles, properties):
        cost = properties.get("estimated_cost_per_gram", 0.0)
        threshold = 10000.0
        if cost > threshold:
            verdict = "fail"
            score = max(0.0, 1.0 - cost / 20000)
            reasoning = f"Estimated synthesis cost=${cost:.0f}/g > ${threshold:.0f}/g, exceeds preclinical budget"
        else:
            verdict = "pass"
            score = min(1.0, 1.0 - cost / threshold)
            reasoning = f"Synthesis cost=${cost:.0f}/g, within acceptable range"
        if self.crg:
            self.crg.add_belief(molecule_smiles, self.name, verdict, confidence=score, source_agent="critic")
        return {"verdict": verdict, "score": score, "reasoning": reasoning, "rule_id": self.rule_id, "rule_name": self.name}
