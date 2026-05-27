"""Existing patent with priority date predating the project"""
from critic_agent.rules.rule_base import CriticRule


class PriorArtPriorityDateRule(CriticRule):
    def __init__(self, crg=None):
        super().__init__("rule_080", "No prior art with earlier priority date", crg)

    def evaluate(self, molecule_smiles, properties):
        prior_date = properties.get("earliest_prior_art_date", None)
        if prior_date is not None:
            verdict = "fail"
            score = 0.2
            reasoning = f"Prior art patent with priority date {prior_date} found, FTO at risk"
        else:
            verdict = "pass"
            score = 1.0
            reasoning = "No prior art with earlier priority date found"
        if self.crg:
            self.crg.add_belief(molecule_smiles, self.name, verdict, confidence=score, source_agent="critic")
        return {"verdict": verdict, "score": score, "reasoning": reasoning, "rule_id": self.rule_id, "rule_name": self.name}
