"""Provenance domain models — nodes and edges for NL-to-SSP traceability."""
from __future__ import annotations

from pydantic import BaseModel, Field


class ProvenanceNode(BaseModel):
    node_id: str
    node_type: str
    run_id: str
    trace_id: str
    content_hash: str
    payload: dict = Field(default_factory=dict)
    signature: str | None = None

    model_config = {"extra": "forbid"}


class ProvenanceEdge(BaseModel):
    from_node_id: str
    to_node_id: str
    relation: str
    agent: str
    metadata: dict = Field(default_factory=dict)
    signature: str | None = None

    model_config = {"extra": "forbid"}
