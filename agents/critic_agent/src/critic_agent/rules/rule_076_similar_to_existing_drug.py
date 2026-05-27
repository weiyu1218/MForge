"""Tanimoto similarity > 0.85 to existing FDA drug indicates me-too risk"""
from critic_agent.rules.rule_base import CriticRule


class SimilarToExistingDrugRule(CriticRule):
    def __init__(self, crg=None):
        super().__init__("rule_076", "FDA drug similarity <= 0.85 (me-too risk)", crg)

    def evaluate(self, molecule_smiles, properties):
        sim = properties.get("max_fda_drug_similarity", 0.0)
        drug_name = properties.get("closest_fda_drug", "unknown")
        threshold = 0.85
        if sim > threshold:
            verdict = "fail"
            score = max(0.0, 1.0 - sim)
            reasoning = f"Tanimoto={sim:.2f} to {drug_name} > {threshold}, me-too IP risk"
        else:
            verdict = "pass"
            score = min(1.0, 1.0 - sim)
            reasoning = f"FDA drug similarity={sim:.2f} <= {threshold}"
        if self.crg:
            self.crg.add_belief(molecule_smiles, self.name, verdict, confidence=score, source_agent="critic")
        return {"verdict": verdict, "score": score, "reasoning": reasoning, "rule_id": self.rule_id, "rule_name": self.name}
