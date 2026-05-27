"""Hepatic clearance < 15 mL/min/kg"""
from critic_agent.rules.rule_base import CriticRule


class ClearanceRule(CriticRule):
    def __init__(self, crg=None):
        super().__init__("rule_035", "Hepatic clearance < 15 mL/min/kg", crg)

    def evaluate(self, molecule_smiles, properties):
        cl = properties.get("clearance", 100.0)
        if cl < 15:
            verdict = "pass"
            score = 1.0
            reasoning = f"Clearance={cl:.1f} mL/min/kg < 15"
        else:
            verdict = "fail"
            score = max(0.0, 1.0 - (cl - 15) / 50)
            reasoning = f"Clearance={cl:.1f} mL/min/kg >= 15 (high clearance)"
        if self.crg:
            self.crg.add_belief(molecule_smiles, self.name, verdict, confidence=score, source_agent="critic")
        return {"verdict": verdict, "score": score, "reasoning": reasoning, "rule_id": self.rule_id, "rule_name": self.name}
