"""Synthesis route > 8 steps exceeds standard CRO capabilities"""
from critic_agent.rules.rule_base import CriticRule


class SynthesisStepsRule(CriticRule):
    def __init__(self, crg=None):
        super().__init__("rule_083", "Synthesis steps <= 8", crg)

    def evaluate(self, molecule_smiles, properties):
        n_steps = properties.get("synthesis_steps", 1)
        threshold = 8
        if n_steps > threshold:
            verdict = "fail"
            score = max(0.0, 1.0 - (n_steps - threshold) * 0.1)
            reasoning = f"Synthesis route {n_steps} steps > {threshold}, exceeds standard CRO capabilities"
        else:
            verdict = "pass"
            score = min(1.0, n_steps / threshold)
            reasoning = f"Synthesis route {n_steps} steps, feasible for CRO"
        if self.crg:
            self.crg.add_belief(molecule_smiles, self.name, verdict, confidence=score, source_agent="critic")
        return {"verdict": verdict, "score": score, "reasoning": reasoning, "rule_id": self.rule_id, "rule_name": self.name}
