from pydantic import BaseModel, Field


class SSPMaterial(BaseModel):
    id: str
    smiles: str
    quantity: float = 1.0
    unit: str = "mmol"
    source: str = ""


class SSPReactant(BaseModel):
    smiles: str
    amount_mmol: float = 1.0
    source: str = ""


class SSPStep(BaseModel):
    step_id: str
    operation: str = "add"
    parameters: dict[str, str] = Field(default_factory=dict)
    reactants: list[SSPReactant] = Field(default_factory=list)
    reagents: list[str] = Field(default_factory=list)
    reaction_type: str | None = None
    temperature_C: float | None = None
    time_h: float | None = None
    yield_estimate: float | None = None
    yield_uncertainty: float | None = None
    purification: str | None = None


class SSP(BaseModel):
    model_config = {"extra": "forbid"}

    ssp_id: str
    run_id: str = ""
    target_smiles: str
    route_id: str | None = None
    materials: list[SSPMaterial] = Field(default_factory=list)
    steps: list[SSPStep] = Field(default_factory=list)
    total_estimated_yield: float | None = None
    total_estimated_cost_usd: float | None = None
    xdl_version: str = "2.0"
    sila2_endpoint: str | None = None
