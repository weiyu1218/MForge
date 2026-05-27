"""REOS: Rapid Elimination Of Swill - MW 200-500, logP 0-5, HBD 0-5, HBA 0-10"""
from critic_agent.rules.rule_base import CriticRule


class REOSRule(CriticRule):
    def __init__(self, crg=None):
        super().__init__("rule_007", "REOS: Rapid Elimination Of Swill - MW 200-500, logP 0-5, HBD 0-5, HBA 0-10", crg)

    def evaluate(self, molecule_smiles, properties):
        mw = properties.get("mw", 0)
        logp = properties.get("logp", 0)
        hbd = properties.get("hbd", 0)
        hba = properties.get("hba", 0)
        violations = 0
        if mw < 200 or mw > 500:
            violations += 1
        if logp < 0 or logp > 5:
            violations += 1
        if hbd < 0 or hbd > 5:
            violations += 1
        if hba < 0 or hba > 10:
            violations += 1
        verdict = "pass" if violations == 0 else "fail"
        score = max(0.0, 1.0 - violations * 0.25)
        reasoning = f"REOS violations={violations} (MW={mw:.1f}, logP={logp:.1f}, HBD={hbd}, HBA={hba})"
        if self.crg:
            self.crg.add_belief(molecule_smiles, self.name, verdict, confidence=score, source_agent="critic")
        return {"verdict": verdict, "score": score, "reasoning": reasoning, "rule_id": self.rule_id, "rule_name": self.name}
