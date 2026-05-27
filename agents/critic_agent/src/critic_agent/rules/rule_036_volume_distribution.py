"""Volume of distribution 0.1-10 L/kg"""
from critic_agent.rules.rule_base import CriticRule


class VolumeDistributionRule(CriticRule):
    def __init__(self, crg=None):
        super().__init__("rule_036", "Volume of distribution 0.1-10 L/kg", crg)

    def evaluate(self, molecule_smiles, properties):
        vd = properties.get("vd_ss", 0.0)
        if 0.1 <= vd <= 10.0:
            verdict = "pass"
            score = 1.0
            reasoning = f"Vd_ss={vd:.1f} L/kg in [0.1,10]"
        else:
            verdict = "fail"
            score = max(0.0, 1.0 - abs(vd - 5) / 10)
            reasoning = f"Vd_ss={vd:.1f} L/kg outside [0.1,10]"
        if self.crg:
            self.crg.add_belief(molecule_smiles, self.name, verdict, confidence=score, source_agent="critic")
        return {"verdict": verdict, "score": score, "reasoning": reasoning, "rule_id": self.rule_id, "rule_name": self.name}
