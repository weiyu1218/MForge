"""BBB penetration category not 'high' for peripheral targets"""
from critic_agent.rules.rule_base import CriticRule


class BBBCategoryRule(CriticRule):
    def __init__(self, crg=None):
        super().__init__("rule_098", "BBB category check for peripheral targets", crg)

    def evaluate(self, molecule_smiles, properties):
        bbb_cat = properties.get("bbb_category", "medium")
        is_peripheral = properties.get("target_is_peripheral", True)
        if not is_peripheral:
            verdict = "pass"
            score = 1.0
            reasoning = "CNS target — BBB penetration acceptable"
        elif bbb_cat in ("low", "medium"):
            verdict = "pass"
            score = 0.8 if bbb_cat == "medium" else 1.0
            reasoning = f"BBB category={bbb_cat} — acceptable for peripheral target"
        else:
            verdict = "fail"
            score = 0.3
            reasoning = f"BBB category={bbb_cat} — high CNS penetration risk for peripheral target"
        if self.crg:
            self.crg.add_belief(molecule_smiles, self.name, verdict, confidence=score, source_agent="critic")
        return {"verdict": verdict, "score": score, "reasoning": reasoning, "rule_id": self.rule_id, "rule_name": self.name}
