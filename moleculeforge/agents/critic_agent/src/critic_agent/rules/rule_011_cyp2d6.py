"""CYP2D6 inhibition risk assessment"""
from critic_agent.rules.rule_base import CriticRule


class CYP2D6InhibitionRule(CriticRule):
    def __init__(self, crg=None):
        super().__init__("rule_011", "CYP2D6 inhibition risk assessment", crg)

    def evaluate(self, molecule_smiles, properties):
        risk = properties.get("cyp2d6_risk", 0.0)
        threshold = 0.5
        verdict = "pass" if risk < threshold else "fail"
        score = max(0.0, 1.0 - risk)
        reasoning = f"CYP2D6 inhibition risk={risk:.2f}, threshold={threshold}"
        if self.crg:
            self.crg.add_belief(molecule_smiles, self.name, verdict, confidence=score, source_agent="critic")
        return {"verdict": verdict, "score": score, "reasoning": reasoning, "rule_id": self.rule_id, "rule_name": self.name}
