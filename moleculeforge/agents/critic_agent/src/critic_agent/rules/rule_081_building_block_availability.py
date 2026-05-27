"""Key building block commercial availability < 0.5"""
from critic_agent.rules.rule_base import CriticRule


class BuildingBlockAvailabilityRule(CriticRule):
    def __init__(self, crg=None):
        super().__init__("rule_081", "Building block availability >= 0.5", crg)

    def evaluate(self, molecule_smiles, properties):
        availability = properties.get("building_block_availability", 1.0)
        threshold = 0.5
        if availability < threshold:
            verdict = "fail"
            score = max(0.0, availability / threshold)
            reasoning = f"Building block availability={availability:.2f} < {threshold}, commercial sourcing limited"
        else:
            verdict = "pass"
            score = min(1.0, availability)
            reasoning = f"Building block availability={availability:.2f} >= {threshold}, commercially accessible"
        if self.crg:
            self.crg.add_belief(molecule_smiles, self.name, verdict, confidence=score, source_agent="critic")
        return {"verdict": verdict, "score": score, "reasoning": reasoning, "rule_id": self.rule_id, "rule_name": self.name}
