"""IC50 shift > 1.5 indicates time-dependent CYP inhibition (irreversible)"""
from critic_agent.rules.rule_base import CriticRule


class TimeDependentInhibitionRule(CriticRule):
    def __init__(self, crg=None):
        super().__init__("rule_073", "Time-dependent CYP inhibition <= 1.5", crg)

    def evaluate(self, molecule_smiles, properties):
        tdi_shift = properties.get("tdi_ic50_shift", 1.0)
        threshold = 1.5
        if tdi_shift > threshold:
            verdict = "fail"
            score = max(0.0, 1.0 - (tdi_shift - 1.0) * 0.5)
            reasoning = f"IC50 shift={tdi_shift:.2f} > {threshold}, time-dependent CYP inhibition suspected"
        else:
            verdict = "pass"
            score = 1.0
            reasoning = f"No TDI signal detected (IC50 shift={tdi_shift:.2f})"
        if self.crg:
            self.crg.add_belief(molecule_smiles, self.name, verdict, confidence=score, source_agent="critic")
        return {"verdict": verdict, "score": score, "reasoning": reasoning, "rule_id": self.rule_id, "rule_name": self.name}
