"""logP within drug-like range 0-5"""
from critic_agent.rules.rule_base import CriticRule


class LogPRangeRule(CriticRule):
    def __init__(self, crg=None):
        super().__init__("rule_015", "logP within drug-like range 0-5", crg)

    def evaluate(self, molecule_smiles, properties):
        logp = properties.get("logp", 0)
        if 0 <= logp <= 5:
            verdict = "pass"
            score = 1.0
            reasoning = f"logP={logp:.1f} in [0,5] range"
        else:
            verdict = "fail"
            score = max(0.0, 1.0 - abs(logp - 2.5) / 5.0)
            reasoning = f"logP={logp:.1f} outside [0,5] range"
        if self.crg:
            self.crg.add_belief(molecule_smiles, self.name, verdict, confidence=score, source_agent="critic")
        return {"verdict": verdict, "score": score, "reasoning": reasoning, "rule_id": self.rule_id, "rule_name": self.name}
