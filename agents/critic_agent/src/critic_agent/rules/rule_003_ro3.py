"""Rule of Three: MW<=300, logP<=3, HBD<=3, HBA<=3"""
from critic_agent.rules.rule_base import CriticRule


class RO3Rule(CriticRule):
    def __init__(self, crg=None):
        super().__init__("rule_003", "Rule of Three: MW<=300, logP<=3, HBD<=3, HBA<=3", crg)

    def evaluate(self, molecule_smiles, properties):
        mw = properties.get("mw", 0)
        logp = properties.get("logp", 0)
        hbd = properties.get("hbd", 0)
        hba = properties.get("hba", 0)
        violations = sum([mw > 300, logp > 3, hbd > 3, hba > 3])
        verdict = "pass" if violations == 0 else "fail"
        score = max(0.0, 1.0 - violations * 0.25)
        reasoning = f"RO3 violations={violations} (MW={mw:.1f}, logP={logp:.1f}, HBD={hbd}, HBA={hba})"
        if self.crg:
            self.crg.add_belief(molecule_smiles, self.name, verdict, confidence=score, source_agent="critic")
        return {"verdict": verdict, "score": score, "reasoning": reasoning, "rule_id": self.rule_id, "rule_name": self.name}
