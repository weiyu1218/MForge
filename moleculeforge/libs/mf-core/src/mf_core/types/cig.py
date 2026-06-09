from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


def default_jsonld_context() -> dict[str, str]:
    return {
        "mf": "https://moleculeforge.io/ontology#",
        "intent_id": "mf:intentId",
        "target_context": "mf:targetContext",
        "objective_nodes": "mf:objectiveNodes",
        "edges": "mf:objectiveEdges",
        "hyperedges": "mf:objectiveHyperedges",
        "generative_priors": "mf:generativePriors",
        "budget_constraints": "mf:budgetConstraints",
        "source_user_input": "mf:sourceUserInput",
    }


class ObjectiveType(str, Enum):
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"
    TARGET_RANGE = "target_range"
    CONSTRAINT = "constraint"
    CONTINUOUS_MAXIMIZE = "continuous_maximize"
    CONTINUOUS_MINIMIZE = "continuous_minimize"
    RATIO_MAXIMIZE = "ratio_maximize"
    MULTI_CONSTRAINT_SATISFY = "multi_constraint_satisfy"


class ObjectiveNode(BaseModel):
    id: str
    name: str = ""
    type: ObjectiveType
    oracle: str = "rdkit"
    target_value: float = 0.0
    target_min: float | None = None
    target_max: float | None = None
    property: str = ""
    weight: float = 1.0
    pareto_tier: int = 1
    constraints: dict | None = None


class ObjectiveEdge(BaseModel):
    source_id: str
    target_id: str
    relation: str = "depends_on"
    strength: float = 0.0


class ObjectiveHyperedge(BaseModel):
    source_ids: list[str] = Field(default_factory=list)
    target_ids: list[str] = Field(default_factory=list)
    relation: str = "depends_on"
    strength: float = 0.0


class TargetContext(dict):
    """Target context for CIG (uniprot IDs, PDB IDs, etc.)."""
    def __init__(self, uniprot_ids=None, pdb_ids=None, **kwargs):
        super().__init__(**kwargs)
        if uniprot_ids is not None:
            self["uniprot_ids"] = uniprot_ids
        if pdb_ids is not None:
            self["pdb_ids"] = pdb_ids


class BudgetConstraints(dict):
    """Budget constraints for CIG (oracle calls, wall time, etc.)."""
    pass


class CIG(BaseModel):
    project_id: str
    objectives: list[ObjectiveNode] = Field(default_factory=list)
    edges: list[ObjectiveEdge] = Field(default_factory=list)
    constraints: dict[str, str] = Field(default_factory=dict)
    created_by: str = ""

    def validate_consistency(self) -> list[str]:
        issues = []
        if not self.objectives:
            issues.append("CIG has no objectives")
        node_ids = {n.id for n in self.objectives}
        for edge in self.edges:
            if edge.source_id not in node_ids:
                issues.append(f"Edge source {edge.source_id} not found in objectives")
            if edge.target_id not in node_ids:
                issues.append(f"Edge target {edge.target_id} not found in objectives")
        for obj in self.objectives:
            if obj.type == ObjectiveType.TARGET_RANGE and (
                obj.target_min is None or obj.target_max is None
            ):
                issues.append(f"Objective {obj.id} is TARGET_RANGE but missing min/max")
        return issues


class ChemicalIntentGraph(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    jsonld_context: dict[str, str] = Field(
        default_factory=default_jsonld_context,
        alias="@context",
    )
    intent_id: str
    version: str = "2.0"
    signature: str | None = None
    target_context: dict = Field(default_factory=dict)
    objective_nodes: list[ObjectiveNode] = Field(default_factory=list)
    edges: list[ObjectiveEdge] = Field(default_factory=list)
    hyperedges: list[ObjectiveHyperedge] = Field(default_factory=list)
    generative_priors: dict = Field(default_factory=dict)
    budget_constraints: dict = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: str | None = None
    source_user_input: str = ""
