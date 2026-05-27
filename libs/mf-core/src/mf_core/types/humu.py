from pydantic import BaseModel, Field
import numpy as np


class HCIV(BaseModel):
    coordinates: list[float] = Field(default_factory=lambda: [0.0] * 129)
    dim: int = 128
    curvature: float = 1.0
    manifold_type: str = "lorentz"
    molecule_smiles: str = ""
    parent_hciv_id: str | None = None

    class Config:
        arbitrary_types_allowed = True


class IntentCone(BaseModel):
    axis: list[float] = Field(default_factory=lambda: [0.0] * 129)
    half_angle: float = 0.5
    angle_radians: float = 0.5
    curvature: float = 1.0
    property_weights: dict[str, float] = Field(default_factory=dict)
    apex: HCIV | None = None
    axis_direction: HCIV | None = None
    length: float = 1.0

    class Config:
        arbitrary_types_allowed = True
