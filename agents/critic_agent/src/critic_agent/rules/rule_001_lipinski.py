"""Lipinski Rule of Five: MW<=500, logP<=5, HBD<=5, HBA<=10"""
from critic_agent.rules.rule_base import CriticRule


class LipinskiRule(CriticRule):
    def __init__(self, crg=None):
        super().__init__("rule_001", "Lipinski Rule of Five: MW<=500, logP<=5, HBD<=5, HBA<=10", crg)

    def evaluate(self, molecule_smiles, properties):
        mw = properties.get("mw", 0)
        logp = properties.get("logp", 0)
        hbd = properties.get("hbd", 0)
        hba = properties.get("hba", 0)
        violations = 0
        reasons = []
        if mw > 500:
            violations += 1
            reasons.append(f"MW={mw:.1f}>500")
        if logp > 5:
            violations += 1
            reasons.append(f"logP={logp:.1f}>5")
        if hbd > 5:
            violations += 1
            reasons.append(f"HBD={hbd}>5")
        if hba > 10:
            violations += 1
            reasons.append(f"HBA={hba}>10")
        if violations <= 1:
            verdict = "pass"
            score = 1.0
            reasoning = f"RO5 violations={violations} (≤1 allowed)"
        else:
            verdict = "fail"
            score = max(0.0, 1.0 - violations * 0.25)
            reasoning = f"RO5 violations={violations}: {', '.join(reasons)}"
        if self.crg:
            self.crg.add_belief(molecule_smiles, self.name, verdict, confidence=score, source_agent="critic")
        return {"verdict": verdict, "score": score, "reasoning": reasoning, "rule_id": self.rule_id, "rule_name": self.name}
