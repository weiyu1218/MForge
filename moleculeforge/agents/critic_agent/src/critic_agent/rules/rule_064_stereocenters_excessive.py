"""Stereocenters > 3 causes exponential synthesis/purification cost"""
from critic_agent.rules.rule_base import CriticRule


class StereocentersExcessiveRule(CriticRule):
    def __init__(self, crg=None):
        super().__init__("rule_064", "Stereocenters <= 3 (synthesis cost)", crg)

    def evaluate(self, molecule_smiles, properties):
        n_stereo = properties.get("n_stereocenters", 0)
        if n_stereo > 3:
            verdict = "fail"
            score = max(0.0, 1.0 - (n_stereo - 3) * 0.2)
            reasoning = f"Stereocenters={n_stereo} > 3, separation/purification cost prohibitive"
        else:
            verdict = "pass"
            score = min(1.0, n_stereo / 3)
            reasoning = f"Stereocenters={n_stereo} <= 3, manageable"
        if self.crg:
            self.crg.add_belief(molecule_smiles, self.name, verdict, confidence=score, source_agent="critic")
        return {"verdict": verdict, "score": score, "reasoning": reasoning, "rule_id": self.rule_id, "rule_name": self.name}
