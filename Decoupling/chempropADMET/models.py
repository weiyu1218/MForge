"""Data models for request / response validation."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    smiles: list[str] = Field(..., min_length=1, description="List of SMILES strings")
    endpoints: list[str] | None = Field(
        default=None,
        description="ADMET endpoints to predict. Null = all available.",
    )
    batch_size: int = Field(default=64, ge=1, le=512, description="Inference batch size")


class MoleculeResult(BaseModel):
    smiles: str
    predictions: dict[str, float | None]


class PredictResponse(BaseModel):
    results: list[MoleculeResult]
    n_molecules: int
    endpoints_used: list[str]


class HealthResponse(BaseModel):
    status: str
    available_endpoints: list[str]
    device: str
