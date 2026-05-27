"""3D pharmacophore matching score too low for target binding"""
from critic_agent.rules.rule_base import CriticRule


class PharmacophoreScoreLowRule(CriticRule):
    def __init__(self, crg=None):
        super().__init__("rule_062", "3D pharmacophore matching score >= 0.5", crg)

    def evaluate(self, molecule_smiles, properties):
        pharm_score = properties.get("pharmacophore_score", 1.0)
        threshold = 0.5
        if pharm_score < threshold:
            verdict = "fail"
            score = max(0.0, pharm_score / threshold)
            reasoning = f"3D pharmacophore score={pharm_score:.3f} < {threshold}, binding pose may be suboptimal"
        else:
            verdict = "pass"
            score = min(1.0, pharm_score)
            reasoning = f"3D pharmacophore score={pharm_score:.3f} >= {threshold}"
        if self.crg:
            self.crg.add_belief(molecule_smiles, self.name, verdict, confidence=score, source_agent="critic")
        return {"verdict": verdict, "score": score, "reasoning": reasoning, "rule_id": self.rule_id, "rule_name": self.name}
