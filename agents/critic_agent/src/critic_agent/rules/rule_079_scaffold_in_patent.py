"""Molecular scaffold already covered by known patent"""
from critic_agent.rules.rule_base import CriticRule


class ScaffoldInPatentRule(CriticRule):
    def __init__(self, crg=None):
        super().__init__("rule_079", "Scaffold not in known patents", crg)

    def evaluate(self, molecule_smiles, properties):
        in_patent = properties.get("scaffold_in_patent", False)
        patent_id = properties.get("overlapping_patent_id", "unknown")
        if in_patent:
            verdict = "fail"
            score = 0.2
            reasoning = f"Molecular scaffold found in patent {patent_id}, freedom to operate compromised"
        else:
            verdict = "pass"
            score = 1.0
            reasoning = "Scaffold not found in known patents"
        if self.crg:
            self.crg.add_belief(molecule_smiles, self.name, verdict, confidence=score, source_agent="critic")
        return {"verdict": verdict, "score": score, "reasoning": reasoning, "rule_id": self.rule_id, "rule_name": self.name}
