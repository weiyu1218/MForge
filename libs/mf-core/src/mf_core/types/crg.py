from pydantic import BaseModel, Field
import time


class Belief(BaseModel):
    id: str
    subject: str
    predicate: str
    object: str
    confidence: float = 0.0
    evidence_ids: list[str] = Field(default_factory=list)
    source_agent: str = ""
    timestamp_ns: int = Field(default_factory=lambda: int(time.time() * 1e9))


class CRGEdge(BaseModel):
    source_belief_id: str
    target_belief_id: str
    relation: str = "supports"
    weight: float = 0.0


class CRG(BaseModel):
    project_id: str
    beliefs: list[Belief] = Field(default_factory=list)
    edges: list[CRGEdge] = Field(default_factory=list)
    version: int = 0
    provenance_id: str = ""
