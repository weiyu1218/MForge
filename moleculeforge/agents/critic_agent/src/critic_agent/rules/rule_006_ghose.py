"""Ghose filter: MW 160-480, logP -0.4 to 5.6, atoms 20-70"""
from critic_agent.rules.rule_base import CriticRule


class GhoseRule(CriticRule):
    def __init__(self, crg=None):
        super().__init__("rule_006", "Ghose filter: MW 160-480, logP -0.4 to 5.6, atoms 20-70", crg)

    def evaluate(self, molecule_smiles, properties):
        mw = properties.get("mw", 0)
        logp = properties.get("logp", 0)
        heavy_atoms = properties.get("heavy_atoms", 0)
        violations = 0
        if mw < 160 or mw > 480:
            violations += 1
        if logp < -0.4 or logp > 5.6:
            violations += 1
        if heavy_atoms < 20 or heavy_atoms > 70:
            violations += 1
        verdict = "pass" if violations == 0 else "fail"
        score = max(0.0, 1.0 - violations * 0.33)
        reasoning = f"Ghose violations={violations} (MW={mw:.1f}, logP={logp:.1f}, atoms={heavy_atoms})"
        if self.crg:
            self.crg.add_belief(molecule_smiles, self.name, verdict, confidence=score, source_agent="critic")
        return {"verdict": verdict, "score": score, "reasoning": reasoning, "rule_id": self.rule_id, "rule_name": self.name}
