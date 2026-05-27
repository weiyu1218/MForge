"""Microsomal metabolic stability t1/2 > 60 min"""
from critic_agent.rules.rule_base import CriticRule


class MetabolicStabilityRule(CriticRule):
    def __init__(self, crg=None):
        super().__init__("rule_060", "Microsomal metabolic stability t1/2 > 60 min", crg)

    def evaluate(self, molecule_smiles, properties):
        t12 = properties.get("microsome_t12_min", 0.0)
        if t12 > 60:
            verdict = "pass"
            score = min(1.0, t12 / 120)
            reasoning = f"Microsomal t1/2={t12:.1f} min > 60 (stable)"
        else:
            verdict = "fail"
            score = max(0.0, t12 / 60)
            reasoning = f"Microsomal t1/2={t12:.1f} min <= 60 (labile)"
        if self.crg:
            self.crg.add_belief(molecule_smiles, self.name, verdict, confidence=score, source_agent="critic")
        return {"verdict": verdict, "score": score, "reasoning": reasoning, "rule_id": self.rule_id, "rule_name": self.name}
