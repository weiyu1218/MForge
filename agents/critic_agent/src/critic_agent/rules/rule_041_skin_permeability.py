"""Skin permeability logKp > -5 cm/s"""
from critic_agent.rules.rule_base import CriticRule


class SkinPermeabilityRule(CriticRule):
    def __init__(self, crg=None):
        super().__init__("rule_041", "Skin permeability logKp > -5 cm/s", crg)

    def evaluate(self, molecule_smiles, properties):
        logkp = properties.get("logkp", -10.0)
        if logkp > -5:
            verdict = "pass"
            score = 1.0
            reasoning = f"logKp={logkp:.1f} > -5 cm/s"
        else:
            verdict = "fail"
            score = max(0.0, (logkp + 10) / 5.0)
            reasoning = f"logKp={logkp:.1f} <= -5 cm/s (poor)"
        if self.crg:
            self.crg.add_belief(molecule_smiles, self.name, verdict, confidence=score, source_agent="critic")
        return {"verdict": verdict, "score": score, "reasoning": reasoning, "rule_id": self.rule_id, "rule_name": self.name}
