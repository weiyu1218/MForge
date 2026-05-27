"""Chemical novelty < 0.3 makes strong patent protection unlikely"""
from critic_agent.rules.rule_base import CriticRule


class ChemicalNoveltyLowRule(CriticRule):
    def __init__(self, crg=None):
        super().__init__("rule_078", "Chemical novelty >= 0.3", crg)

    def evaluate(self, molecule_smiles, properties):
        novelty = properties.get("chemical_novelty", 0.5)
        threshold = 0.3
        if novelty < threshold:
            verdict = "fail"
            score = max(0.0, novelty / threshold)
            reasoning = f"Chemical novelty={novelty:.2f} < {threshold}, weak patent protection likely"
        else:
            verdict = "pass"
            score = min(1.0, novelty)
            reasoning = f"Chemical novelty={novelty:.2f} >= {threshold}, acceptable for patent"
        if self.crg:
            self.crg.add_belief(molecule_smiles, self.name, verdict, confidence=score, source_agent="critic")
        return {"verdict": verdict, "score": score, "reasoning": reasoning, "rule_id": self.rule_id, "rule_name": self.name}
