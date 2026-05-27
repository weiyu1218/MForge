"""Ring count > 6 makes synthesis extremely difficult"""
from critic_agent.rules.rule_base import CriticRule


class RingCountExtremeRule(CriticRule):
    def __init__(self, crg=None):
        super().__init__("rule_063", "Ring count <= 6 (synthetic accessibility)", crg)

    def evaluate(self, molecule_smiles, properties):
        n_rings = properties.get("ring_count", 0)
        if n_rings > 6:
            verdict = "fail"
            score = max(0.0, 1.0 - (n_rings - 6) * 0.15)
            reasoning = f"Ring count={n_rings} > 6, synthesis extremely challenging"
        else:
            verdict = "pass"
            score = min(1.0, n_rings / 6)
            reasoning = f"Ring count={n_rings} <= 6, synthetically reasonable"
        if self.crg:
            self.crg.add_belief(molecule_smiles, self.name, verdict, confidence=score, source_agent="critic")
        return {"verdict": verdict, "score": score, "reasoning": reasoning, "rule_id": self.rule_id, "rule_name": self.name}
