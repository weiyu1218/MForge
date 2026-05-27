"""Rule of Three for fragment-based screening hits"""
from critic_agent.rules.rule_base import CriticRule


class RuleOfThreeRule(CriticRule):
    def __init__(self, crg=None):
        super().__init__("rule_094", "Rule of Three (fragment-likeness)", crg)

    def evaluate(self, molecule_smiles, properties):
        mw = properties.get("molecular_weight", 300)
        logp = properties.get("logp", 3.0)
        hbd = properties.get("num_h_bond_donors", 3)
        hba = properties.get("num_h_bond_acceptors", 3)
        rot = properties.get("rotatable_bonds", 3)
        violations = 0
        if mw > 300:
            violations += 1
        if logp > 3.0:
            violations += 1
        if hbd > 3:
            violations += 1
        if hba > 3:
            violations += 1
        if rot > 3:
            violations += 1
        if violations <= 1:
            verdict = "pass"
            score = max(0.0, 1.0 - violations * 0.2)
            reasoning = f"Ro3 violations={violations}/5 (fragment-like)"
        else:
            verdict = "fail"
            score = max(0.0, 1.0 - violations * 0.15)
            reasoning = f"Ro3 violations={violations}/5 (not fragment-like)"
        if self.crg:
            self.crg.add_belief(molecule_smiles, self.name, verdict, confidence=score, source_agent="critic")
        return {"verdict": verdict, "score": score, "reasoning": reasoning, "rule_id": self.rule_id, "rule_name": self.name}
