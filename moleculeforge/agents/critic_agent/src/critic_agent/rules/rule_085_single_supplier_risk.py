"""Critical building block has only 1 supplier (supply chain fragility)"""
from critic_agent.rules.rule_base import CriticRule


class SingleSupplierRiskRule(CriticRule):
    def __init__(self, crg=None):
        super().__init__("rule_085", "Multiple suppliers for critical materials", crg)

    def evaluate(self, molecule_smiles, properties):
        n_suppliers = properties.get("critical_material_suppliers", 2)
        if n_suppliers == 1:
            verdict = "fail"
            score = 0.4
            reasoning = "Critical building block has only 1 known supplier, supply chain fragile"
        else:
            verdict = "pass"
            score = min(1.0, n_suppliers / 3)
            reasoning = f"Critical material has {n_suppliers} suppliers, supply chain acceptable"
        if self.crg:
            self.crg.add_belief(molecule_smiles, self.name, verdict, confidence=score, source_agent="critic")
        return {"verdict": verdict, "score": score, "reasoning": reasoning, "rule_id": self.rule_id, "rule_name": self.name}
