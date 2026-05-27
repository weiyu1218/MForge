"""P-glycoprotein substrate prediction"""
from critic_agent.rules.rule_base import CriticRule


class PgpSubstrateRule(CriticRule):
    def __init__(self, crg=None):
        super().__init__("rule_038", "P-glycoprotein substrate prediction", crg)

    def evaluate(self, molecule_smiles, properties):
        risk = properties.get("pgp_substrate_risk", 0.0)
        if risk < 0.5:
            verdict = "pass"
            score = 1.0 - risk
            reasoning = f"P-gp substrate risk={risk:.2f} (low efflux)"
        else:
            verdict = "fail"
            score = max(0.0, 1.0 - risk)
            reasoning = f"P-gp substrate risk={risk:.2f} (likely efflux)"
        if self.crg:
            self.crg.add_belief(molecule_smiles, self.name, verdict, confidence=score, source_agent="critic")
        return {"verdict": verdict, "score": score, "reasoning": reasoning, "rule_id": self.rule_id, "rule_name": self.name}
