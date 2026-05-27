"""Lead-likeness: MW<=350, logP<=4.5"""
from critic_agent.rules.rule_base import CriticRule


class LeadLikenessRule(CriticRule):
    def __init__(self, crg=None):
        super().__init__("rule_008", "Lead-likeness: MW<=350, logP<=4.5", crg)

    def evaluate(self, molecule_smiles, properties):
        mw = properties.get("mw", 0)
        logp = properties.get("logp", 0)
        violations = 0
        reasons = []
        if mw > 350:
            violations += 1
            reasons.append(f"MW={mw:.1f}>350")
        if logp > 4.5:
            violations += 1
            reasons.append(f"logP={logp:.1f}>4.5")
        verdict = "pass" if violations == 0 else "fail"
        score = max(0.0, 1.0 - violations * 0.5)
        reasoning = f"Lead-likeness violations={violations}" + (f": {', '.join(reasons)}" if reasons else "")
        if self.crg:
            self.crg.add_belief(molecule_smiles, self.name, verdict, confidence=score, source_agent="critic")
        return {"verdict": verdict, "score": score, "reasoning": reasoning, "rule_id": self.rule_id, "rule_name": self.name}
