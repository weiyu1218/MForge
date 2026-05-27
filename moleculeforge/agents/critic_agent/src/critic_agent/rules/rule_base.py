"""Base class for critic rules."""
from mf_agents.crg.graph import ChemicalReasoningGraph


class CriticRule:
    def __init__(self, rule_id: str, name: str, crg: ChemicalReasoningGraph | None = None):
        self.rule_id = rule_id
        self.name = name
        self.crg = crg

    def evaluate(self, molecule_smiles: str, properties: dict) -> dict:
        """Return verdict dict with keys: verdict, score, reasoning."""
        raise NotImplementedError  # ABC - subclasses override
