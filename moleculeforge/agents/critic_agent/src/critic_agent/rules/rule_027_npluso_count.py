"""N+O atom count <= 10"""
from critic_agent.rules.rule_base import CriticRule


class NPlusOCountRule(CriticRule):
    def __init__(self, crg=None):
        super().__init__("rule_027", "N+O atom count <= 10", crg)

    def evaluate(self, molecule_smiles, properties):
        n_plus_o = properties.get("npluso_count", 0)
        if n_plus_o <= 10:
            verdict = "pass"
            score = 1.0
            reasoning = f"N+O count={n_plus_o} <= 10"
        else:
            verdict = "fail"
            score = max(0.0, 1.0 - (n_plus_o - 10) / 10)
            reasoning = f"N+O count={n_plus_o} > 10"
        if self.crg:
            self.crg.add_belief(molecule_smiles, self.name, verdict, confidence=score, source_agent="critic")
        return {"verdict": verdict, "score": score, "reasoning": reasoning, "rule_id": self.rule_id, "rule_name": self.name}
