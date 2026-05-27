"""Peripheral target drug with high CNS penetration risk"""
from critic_agent.rules.rule_base import CriticRule


class CNSPenetrationRule(CriticRule):
    def __init__(self, crg=None):
        super().__init__("rule_069", "CNS penetration risk for peripheral targets", crg)

    def evaluate(self, molecule_smiles, properties):
        is_peripheral = properties.get("target_is_peripheral", True)
        if not is_peripheral:
            return {"verdict": "pass", "score": 1.0, "reasoning": "CNS target, penetration acceptable", "rule_id": self.rule_id, "rule_name": self.name}
        mw = properties.get("mw", 500)
        logp = properties.get("logp", 2.0)
        tpsa = properties.get("tpsa", 100)
        cns_risk = (mw < 450) and (logp > 3.0) and (tpsa < 60)
        if cns_risk:
            verdict = "fail"
            score = 0.3
            reasoning = f"Peripheral target but MW={mw}, logP={logp:.1f}, TPSA={tpsa:.1f} suggest CNS penetration risk"
        else:
            verdict = "pass"
            score = 1.0
            reasoning = f"CNS penetration risk low for peripheral target (logP={logp:.1f}, TPSA={tpsa:.1f})"
        if self.crg:
            self.crg.add_belief(molecule_smiles, self.name, verdict, confidence=score, source_agent="critic")
        return {"verdict": verdict, "score": score, "reasoning": reasoning, "rule_id": self.rule_id, "rule_name": self.name}
