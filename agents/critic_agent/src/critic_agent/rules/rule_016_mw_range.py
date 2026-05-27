"""Molecular weight within drug-like range 150-500 Da"""
from critic_agent.rules.rule_base import CriticRule


class MWRangeRule(CriticRule):
    def __init__(self, crg=None):
        super().__init__("rule_016", "Molecular weight within drug-like range 150-500 Da", crg)

    def evaluate(self, molecule_smiles, properties):
        mw = properties.get("mw", 0)
        if 150 <= mw <= 500:
            verdict = "pass"
            score = 1.0
            reasoning = f"MW={mw:.1f} in [150,500] Da"
        else:
            verdict = "fail"
            score = max(0.0, 1.0 - abs(mw - 325) / 500)
            reasoning = f"MW={mw:.1f} outside [150,500] Da"
        if self.crg:
            self.crg.add_belief(molecule_smiles, self.name, verdict, confidence=score, source_agent="critic")
        return {"verdict": verdict, "score": score, "reasoning": reasoning, "rule_id": self.rule_id, "rule_name": self.name}
