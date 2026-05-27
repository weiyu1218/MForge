"""Number of aromatic rings <= 4"""
from critic_agent.rules.rule_base import CriticRule


class AromaticRingCountExcessiveRule(CriticRule):
    def __init__(self, crg=None):
        super().__init__("rule_097", "Aromatic ring count <= 4", crg)

    def evaluate(self, molecule_smiles, properties):
        n_arom = properties.get("num_aromatic_rings", 0)
        if n_arom <= 4:
            verdict = "pass"
            score = max(0.0, 1.0 - n_arom * 0.1)
            reasoning = f"Aromatic rings={n_arom} <= 4"
        else:
            verdict = "fail"
            score = max(0.0, 1.0 - n_arom * 0.12)
            reasoning = f"Aromatic rings={n_arom} > 4 — developability / solubility concern"
        if self.crg:
            self.crg.add_belief(molecule_smiles, self.name, verdict, confidence=score, source_agent="critic")
        return {"verdict": verdict, "score": score, "reasoning": reasoning, "rule_id": self.rule_id, "rule_name": self.name}
