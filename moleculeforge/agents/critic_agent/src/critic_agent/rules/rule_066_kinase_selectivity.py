"""Kinase selectivity ratio < 10 indicates high off-target toxicity risk"""
from critic_agent.rules.rule_base import CriticRule


class KinaseSelectivityRule(CriticRule):
    def __init__(self, crg=None):
        super().__init__("rule_066", "Kinase selectivity ratio >= 10", crg)

    def evaluate(self, molecule_smiles, properties):
        selectivity_ratio = properties.get("kinase_selectivity_ratio", 100.0)
        threshold = 10.0
        if selectivity_ratio < threshold:
            verdict = "fail"
            score = max(0.0, selectivity_ratio / threshold)
            reasoning = f"Kinase selectivity ratio={selectivity_ratio:.1f} < {threshold}, off-target toxicity risk"
        else:
            verdict = "pass"
            score = min(1.0, selectivity_ratio / 100)
            reasoning = f"Kinase selectivity ratio={selectivity_ratio:.1f} >= {threshold}"
        if self.crg:
            self.crg.add_belief(molecule_smiles, self.name, verdict, confidence=score, source_agent="critic")
        return {"verdict": verdict, "score": score, "reasoning": reasoning, "rule_id": self.rule_id, "rule_name": self.name}
