"""XDL domain models."""
from __future__ import annotations

from pydantic import BaseModel, Field


class XDLHardware(BaseModel):
    id: str
    type: str

    model_config = {"extra": "forbid"}


class XDLReagent(BaseModel):
    id: str
    smiles: str
    quantity: float = 1.0
    unit: str = "mmol"


class XDLStep(BaseModel):
    tag: str
    attributes: dict = Field(default_factory=dict)


class XDLProcedure(BaseModel):
    hardware: list[XDLHardware] = Field(default_factory=list)
    reagents: list[XDLReagent] = Field(default_factory=list)
    steps: list[XDLStep] = Field(default_factory=list)

    model_config = {"extra": "forbid"}
