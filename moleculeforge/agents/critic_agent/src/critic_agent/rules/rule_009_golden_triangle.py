"""Golden Triangle: logD 1-3 and MW 200-500"""
from critic_agent.rules.rule_base import CriticRule


class GoldenTriangleRule(CriticRule):
    def __init__(self, crg=None):
        super().__init__("rule_009", "Golden Triangle: logD 1-3 and MW 200-500", crg)

    def evaluate(self, molecule_smiles, properties):
        logd = properties.get("logd", 0)
        mw = properties.get("mw", 0)
        in_triangle = (1 <= logd <= 3) and (200 <= mw <= 500)
        verdict = "pass" if in_triangle else "fail"
        score = 1.0 if in_triangle else max(0.0, 1.0 - abs(logd - 2) / 3 - abs(mw - 350) / 300)
        reasoning = f"GoldenTriangle logD={logd:.1f}, MW={mw:.1f} -> {'inside' if in_triangle else 'outside'} triangle"
        if self.crg:
            self.crg.add_belief(molecule_smiles, self.name, verdict, confidence=score, source_agent="critic")
        return {"verdict": verdict, "score": score, "reasoning": reasoning, "rule_id": self.rule_id, "rule_name": self.name}
