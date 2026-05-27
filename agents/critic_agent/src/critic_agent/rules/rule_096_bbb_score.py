"""BBB score >= 4 for CNS penetration desirability"""
from critic_agent.rules.rule_base import CriticRule


class BBBScoreRule(CriticRule):
    def __init__(self, crg=None):
        super().__init__("rule_096", "BBB score >= 4.0", crg)

    def evaluate(self, molecule_smiles, properties):
        bbb = properties.get("bbb_score", 0.0)
        if bbb >= 4.0:
            verdict = "pass"
            score = min(1.0, bbb / 6.0)
            reasoning = f"BBB score={bbb:.1f} >= 4.0 (CNS-penetrant)"
        else:
            verdict = "fail"
            score = max(0.0, bbb / 4.0)
            reasoning = f"BBB score={bbb:.1f} < 4.0 (limited CNS penetration)"
        if self.crg:
            self.crg.add_belief(molecule_smiles, self.name, verdict, confidence=score, source_agent="critic")
        return {"verdict": verdict, "score": score, "reasoning": reasoning, "rule_id": self.rule_id, "rule_name": self.name}
