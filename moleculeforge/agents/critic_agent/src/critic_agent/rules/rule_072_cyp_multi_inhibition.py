"""Inhibiting >= 2 CYP isoforms simultaneously causes DDI risk"""
from critic_agent.rules.rule_base import CriticRule


class CYPMultiInhibitionRule(CriticRule):
    def __init__(self, crg=None):
        super().__init__("rule_072", "CYP multi-isoform inhibition DDI risk", crg)

    def evaluate(self, molecule_smiles, properties):
        cyp_ic50s = properties.get("cyp_ic50s", {})
        inhibited = {k: v for k, v in cyp_ic50s.items() if v < 1.0}
        if len(inhibited) >= 2:
            verdict = "fail"
            score = max(0.0, 1.0 - len(inhibited) * 0.25)
            reasoning = f"Inhibits {len(inhibited)} CYP isoforms: {list(inhibited.keys())}, DDI risk high"
        else:
            verdict = "pass"
            score = 1.0
            reasoning = f"CYP multi-inhibition risk low ({len(inhibited)} isoforms inhibited)"
        if self.crg:
            self.crg.add_belief(molecule_smiles, self.name, verdict, confidence=score, source_agent="critic")
        return {"verdict": verdict, "score": score, "reasoning": reasoning, "rule_id": self.rule_id, "rule_name": self.name}
