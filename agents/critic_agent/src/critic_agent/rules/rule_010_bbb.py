"""BBB permeability: TPSA<90, MW<400, logP 1-4"""
from critic_agent.rules.rule_base import CriticRule


class BBBPermeabilityRule(CriticRule):
    def __init__(self, crg=None):
        super().__init__("rule_010", "BBB permeability: TPSA<90, MW<400, logP 1-4", crg)

    def evaluate(self, molecule_smiles, properties):
        tpsa = properties.get("tpsa", 0)
        mw = properties.get("mw", 0)
        logp = properties.get("logp", 0)
        violations = 0
        if tpsa >= 90:
            violations += 1
        if mw >= 400:
            violations += 1
        if not (1 <= logp <= 4):
            violations += 1
        verdict = "pass" if violations == 0 else "fail"
        score = max(0.0, 1.0 - violations * 0.33)
        reasoning = f"BBB permeability TPSA={tpsa:.1f}, MW={mw:.1f}, logP={logp:.1f} -> {'permeable' if violations==0 else 'poor'}"
        if self.crg:
            self.crg.add_belief(molecule_smiles, self.name, verdict, confidence=score, source_agent="critic")
        return {"verdict": verdict, "score": score, "reasoning": reasoning, "rule_id": self.rule_id, "rule_name": self.name}
