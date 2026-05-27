from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Belief(_message.Message):
    __slots__ = ("id", "subject", "predicate", "object", "confidence", "evidence_ids", "source_agent", "timestamp_ns")
    ID_FIELD_NUMBER: _ClassVar[int]
    SUBJECT_FIELD_NUMBER: _ClassVar[int]
    PREDICATE_FIELD_NUMBER: _ClassVar[int]
    OBJECT_FIELD_NUMBER: _ClassVar[int]
    CONFIDENCE_FIELD_NUMBER: _ClassVar[int]
    EVIDENCE_IDS_FIELD_NUMBER: _ClassVar[int]
    SOURCE_AGENT_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_NS_FIELD_NUMBER: _ClassVar[int]
    id: str
    subject: str
    predicate: str
    object: str
    confidence: float
    evidence_ids: _containers.RepeatedScalarFieldContainer[str]
    source_agent: str
    timestamp_ns: int
    def __init__(self, id: _Optional[str] = ..., subject: _Optional[str] = ..., predicate: _Optional[str] = ..., object: _Optional[str] = ..., confidence: _Optional[float] = ..., evidence_ids: _Optional[_Iterable[str]] = ..., source_agent: _Optional[str] = ..., timestamp_ns: _Optional[int] = ...) -> None: ...

class CRGEdge(_message.Message):
    __slots__ = ("source_belief_id", "target_belief_id", "relation", "weight")
    SOURCE_BELIEF_ID_FIELD_NUMBER: _ClassVar[int]
    TARGET_BELIEF_ID_FIELD_NUMBER: _ClassVar[int]
    RELATION_FIELD_NUMBER: _ClassVar[int]
    WEIGHT_FIELD_NUMBER: _ClassVar[int]
    source_belief_id: str
    target_belief_id: str
    relation: str
    weight: float
    def __init__(self, source_belief_id: _Optional[str] = ..., target_belief_id: _Optional[str] = ..., relation: _Optional[str] = ..., weight: _Optional[float] = ...) -> None: ...

class CRG(_message.Message):
    __slots__ = ("project_id", "beliefs", "edges", "version", "provenance_id")
    PROJECT_ID_FIELD_NUMBER: _ClassVar[int]
    BELIEFS_FIELD_NUMBER: _ClassVar[int]
    EDGES_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    PROVENANCE_ID_FIELD_NUMBER: _ClassVar[int]
    project_id: str
    beliefs: _containers.RepeatedCompositeFieldContainer[Belief]
    edges: _containers.RepeatedCompositeFieldContainer[CRGEdge]
    version: int
    provenance_id: str
    def __init__(self, project_id: _Optional[str] = ..., beliefs: _Optional[_Iterable[_Union[Belief, _Mapping]]] = ..., edges: _Optional[_Iterable[_Union[CRGEdge, _Mapping]]] = ..., version: _Optional[int] = ..., provenance_id: _Optional[str] = ...) -> None: ...
