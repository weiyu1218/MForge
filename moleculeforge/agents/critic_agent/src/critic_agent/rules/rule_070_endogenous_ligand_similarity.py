"""Too similar to endogenous ligand may be recognized as false substrate"""
from critic_agent.rules.rule_base import CriticRule


class EndogenousLigandSimilarityRule(CriticRule):
    def __init__(self, crg=None):
        super().__init__("rule_070", "Endogenous ligand similarity <= 0.75", crg)

    def evaluate(self, molecule_smiles, properties):
        sim = properties.get("endogenous_ligand_similarity", 0.0)
        threshold = 0.75
        if sim > threshold:
            verdict = "fail"
            score = max(0.0, 1.0 - sim)
            reasoning = f"Endogenous ligand similarity={sim:.2f} > {threshold}, may interfere with normal signaling"
        else:
            verdict = "pass"
            score = 1.0 - sim
            reasoning = f"Endogenous ligand similarity={sim:.2f} <= {threshold}"
        if self.crg:
            self.crg.add_belief(molecule_smiles, self.name, verdict, confidence=score, source_agent="critic")
        return {"verdict": verdict, "score": score, "reasoning": reasoning, "rule_id": self.rule_id, "rule_name": self.name}
