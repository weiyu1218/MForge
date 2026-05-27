"""FTO score in borderline zone [0.6, 0.85) requires manual review"""
from critic_agent.rules.rule_base import CriticRule


class FTOScoreBorderlineRule(CriticRule):
    def __init__(self, crg=None):
        super().__init__("rule_077", "FTO score outside borderline zone", crg)

    def evaluate(self, molecule_smiles, properties):
        fto = properties.get("fto_score", 1.0)
        if 0.6 <= fto < 0.85:
            verdict = "fail"
            score = fto
            reasoning = f"FTO score={fto:.2f} in borderline zone [0.6, 0.85), manual patent attorney review needed"
        else:
            verdict = "pass"
            score = fto
            reasoning = f"FTO score={fto:.2f}, outside borderline zone"
        if self.crg:
            self.crg.add_belief(molecule_smiles, self.name, verdict, confidence=score, source_agent="critic")
        return {"verdict": verdict, "score": score, "reasoning": reasoning, "rule_id": self.rule_id, "rule_name": self.name}
