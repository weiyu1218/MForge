"""CNS MPO score >= 4 for CNS-targeting candidates"""
from critic_agent.rules.rule_base import CriticRule


class CNSMPORule(CriticRule):
    def __init__(self, crg=None):
        super().__init__("rule_086", "CNS MPO score >= 4.0", crg)

    def evaluate(self, molecule_smiles, properties):
        cns_mpo = properties.get("cns_mpo", 0.0)
        if cns_mpo >= 4.0:
            verdict = "pass"
            score = min(1.0, cns_mpo / 6.0)
            reasoning = f"CNS MPO={cns_mpo:.1f} >= 4.0 (CNS drug-like)"
        else:
            verdict = "fail"
            score = max(0.0, cns_mpo / 4.0)
            reasoning = f"CNS MPO={cns_mpo:.1f} < 4.0 (poor CNS penetration potential)"
        if self.crg:
            self.crg.add_belief(molecule_smiles, self.name, verdict, confidence=score, source_agent="critic")
        return {"verdict": verdict, "score": score, "reasoning": reasoning, "rule_id": self.rule_id, "rule_name": self.name}
