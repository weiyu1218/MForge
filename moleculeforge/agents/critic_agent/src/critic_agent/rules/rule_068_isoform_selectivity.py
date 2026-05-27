"""Missing isoform selectivity data prevents safety margin assessment"""
from critic_agent.rules.rule_base import CriticRule


class IsoformSelectivityRule(CriticRule):
    def __init__(self, crg=None):
        super().__init__("rule_068", "Isoform selectivity data available", crg)

    def evaluate(self, molecule_smiles, properties):
        isoform_count = properties.get("isoform_data_count", 0)
        if isoform_count == 0:
            verdict = "fail"
            score = 0.3
            reasoning = "No isoform selectivity data available, cannot assess safety margin"
        else:
            verdict = "pass"
            score = min(1.0, isoform_count / 5)
            reasoning = f"Isoform selectivity data available ({isoform_count} isoforms)"
        if self.crg:
            self.crg.add_belief(molecule_smiles, self.name, verdict, confidence=score, source_agent="critic")
        return {"verdict": verdict, "score": score, "reasoning": reasoning, "rule_id": self.rule_id, "rule_name": self.name}
