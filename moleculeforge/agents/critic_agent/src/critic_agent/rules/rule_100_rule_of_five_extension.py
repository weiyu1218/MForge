"""Extended Rule of Five (bRo5) — oral beyond Ro5 space"""
from critic_agent.rules.rule_base import CriticRule


class ExtendedRuleOfFiveRule(CriticRule):
    def __init__(self, crg=None):
        super().__init__("rule_100", "Extended Rule of Five (bRo5)", crg)

    def evaluate(self, molecule_smiles, properties):
        mw = properties.get("molecular_weight", 500)
        logp = properties.get("logp", 5.0)
        hbd = properties.get("num_h_bond_donors", 5)
        hba = properties.get("num_h_bond_acceptors", 10)
        tpsa = properties.get("tpsa", 140)
        violations = 0
        if mw > 500:
            violations += 1
        if logp > 5.0:
            violations += 1
        if hbd > 5:
            violations += 1
        if hba > 10:
            violations += 1
        if tpsa > 200:
            violations += 1
        if violations <= 1:
            verdict = "pass"
            score = max(0.0, 1.0 - violations * 0.3)
            reasoning = f"bRo5 violations={violations}/5 (acceptable beyond-Ro5 space)"
        elif violations <= 2:
            verdict = "fail"
            score = 0.5
            reasoning = f"bRo5 violations={violations}/5 (marginal bRo5 — monitor ADMET closely)"
        else:
            verdict = "fail"
            score = max(0.0, 1.0 - violations * 0.25)
            reasoning = f"bRo5 violations={violations}/5 (far beyond Ro5 — high developability risk)"
        if self.crg:
            self.crg.add_belief(molecule_smiles, self.name, verdict, confidence=score, source_agent="critic")
        return {"verdict": verdict, "score": score, "reasoning": reasoning, "rule_id": self.rule_id, "rule_name": self.name}
