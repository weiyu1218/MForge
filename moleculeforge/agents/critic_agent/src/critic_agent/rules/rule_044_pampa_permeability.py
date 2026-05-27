"""PAMPA permeability > 1e-6 cm/s"""
from critic_agent.rules.rule_base import CriticRule


class PAMPAPermeabilityRule(CriticRule):
    def __init__(self, crg=None):
        super().__init__("rule_044", "PAMPA permeability > 1e-6 cm/s", crg)

    def evaluate(self, molecule_smiles, properties):
        papp = properties.get("pampa_papp", 0.0)
        threshold = 1e-6
        if papp > threshold:
            verdict = "pass"
            score = min(1.0, papp / 1e-5)
            reasoning = f"PAMPA Papp={papp:.1e} cm/s > 1e-6"
        else:
            verdict = "fail"
            score = max(0.0, papp / threshold)
            reasoning = f"PAMPA Papp={papp:.1e} cm/s <= 1e-6"
        if self.crg:
            self.crg.add_belief(molecule_smiles, self.name, verdict, confidence=score, source_agent="critic")
        return {"verdict": verdict, "score": score, "reasoning": reasoning, "rule_id": self.rule_id, "rule_name": self.name}
