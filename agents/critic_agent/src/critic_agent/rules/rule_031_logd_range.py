"""logD within drug-like range 1-5 at pH 7.4"""
from critic_agent.rules.rule_base import CriticRule


class LogDRangeRule(CriticRule):
    def __init__(self, crg=None):
        super().__init__("rule_031", "logD within drug-like range 1-5 at pH 7.4", crg)

    def evaluate(self, molecule_smiles, properties):
        logd = properties.get("logd", 0.0)
        if 1 <= logd <= 5:
            verdict = "pass"
            score = 1.0
            reasoning = f"logD={logd:.1f} in [1,5]"
        else:
            verdict = "fail"
            score = max(0.0, 1.0 - abs(logd - 3) / 5)
            reasoning = f"logD={logd:.1f} outside [1,5]"
        if self.crg:
            self.crg.add_belief(molecule_smiles, self.name, verdict, confidence=score, source_agent="critic")
        return {"verdict": verdict, "score": score, "reasoning": reasoning, "rule_id": self.rule_id, "rule_name": self.name}
