"""Human liver microsome CLint > 30 mL/min/g indicates metabolic instability"""
from critic_agent.rules.rule_base import CriticRule


class HclintTooHighRule(CriticRule):
    def __init__(self, crg=None):
        super().__init__("rule_071", "HLM CLint <= 30 mL/min/g", crg)

    def evaluate(self, molecule_smiles, properties):
        hclint = properties.get("hlm_clint_ml_min_g", 0.0)
        threshold = 30.0
        if hclint > threshold:
            verdict = "fail"
            score = max(0.0, 1.0 - hclint / 60)
            reasoning = f"HLM CLint={hclint:.1f} > {threshold} mL/min/g, metabolically unstable"
        else:
            verdict = "pass"
            score = min(1.0, 1.0 - hclint / threshold)
            reasoning = f"HLM CLint={hclint:.1f} <= {threshold} mL/min/g, acceptable stability"
        if self.crg:
            self.crg.add_belief(molecule_smiles, self.name, verdict, confidence=score, source_agent="critic")
        return {"verdict": verdict, "score": score, "reasoning": reasoning, "rule_id": self.rule_id, "rule_name": self.name}
