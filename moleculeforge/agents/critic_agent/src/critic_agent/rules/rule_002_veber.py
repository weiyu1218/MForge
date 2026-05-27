"""Veber Rule: rotatable bonds <=10, TPSA <=140"""
from critic_agent.rules.rule_base import CriticRule


class VeberRule(CriticRule):
    def __init__(self, crg=None):
        super().__init__("rule_002", "Veber Rule: rotatable bonds <=10, TPSA <=140", crg)

    def evaluate(self, molecule_smiles, properties):
        rot_bonds = properties.get("rotatable_bonds", 0)
        tpsa = properties.get("tpsa", 0)
        violations = 0
        reasons = []
        if rot_bonds > 10:
            violations += 1
            reasons.append(f"RotBonds={rot_bonds}>10")
        if tpsa > 140:
            violations += 1
            reasons.append(f"TPSA={tpsa:.1f}>140")
        verdict = "pass" if violations == 0 else "fail"
        score = max(0.0, 1.0 - violations * 0.5)
        reasoning = f"Veber violations={violations}" + (f": {', '.join(reasons)}" if reasons else "")
        if self.crg:
            self.crg.add_belief(molecule_smiles, self.name, verdict, confidence=score, source_agent="critic")
        return {"verdict": verdict, "score": score, "reasoning": reasoning, "rule_id": self.rule_id, "rule_name": self.name}
