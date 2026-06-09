from mf_core.proto_gen.moleculeforge.v1.core import humu_pb2 as _humu_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ObjectiveType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    OBJECTIVE_TYPE_UNSPECIFIED: _ClassVar[ObjectiveType]
    MAXIMIZE: _ClassVar[ObjectiveType]
    MINIMIZE: _ClassVar[ObjectiveType]
    TARGET_RANGE: _ClassVar[ObjectiveType]
    CONSTRAINT: _ClassVar[ObjectiveType]
OBJECTIVE_TYPE_UNSPECIFIED: ObjectiveType
MAXIMIZE: ObjectiveType
MINIMIZE: ObjectiveType
TARGET_RANGE: ObjectiveType
CONSTRAINT: ObjectiveType

class ObjectiveNode(_message.Message):
    __slots__ = ("id", "name", "type", "target_value", "target_min", "target_max", "property", "weight", "pareto_tier")
    ID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    TARGET_VALUE_FIELD_NUMBER: _ClassVar[int]
    TARGET_MIN_FIELD_NUMBER: _ClassVar[int]
    TARGET_MAX_FIELD_NUMBER: _ClassVar[int]
    PROPERTY_FIELD_NUMBER: _ClassVar[int]
    WEIGHT_FIELD_NUMBER: _ClassVar[int]
    PARETO_TIER_FIELD_NUMBER: _ClassVar[int]
    id: str
    name: str
    type: ObjectiveType
    target_value: float
    target_min: float
    target_max: float
    property: str
    weight: float
    pareto_tier: int
    def __init__(self, id: _Optional[str] = ..., name: _Optional[str] = ..., type: _Optional[_Union[ObjectiveType, str]] = ..., target_value: _Optional[float] = ..., target_min: _Optional[float] = ..., target_max: _Optional[float] = ..., property: _Optional[str] = ..., weight: _Optional[float] = ..., pareto_tier: _Optional[int] = ...) -> None: ...

class ObjectiveEdge(_message.Message):
    __slots__ = ("source_id", "target_id", "relation", "strength")
    SOURCE_ID_FIELD_NUMBER: _ClassVar[int]
    TARGET_ID_FIELD_NUMBER: _ClassVar[int]
    RELATION_FIELD_NUMBER: _ClassVar[int]
    STRENGTH_FIELD_NUMBER: _ClassVar[int]
    source_id: str
    target_id: str
    relation: str
    strength: float
    def __init__(self, source_id: _Optional[str] = ..., target_id: _Optional[str] = ..., relation: _Optional[str] = ..., strength: _Optional[float] = ...) -> None: ...

class ObjectiveHyperedge(_message.Message):
    __slots__ = ("source_ids", "target_ids", "relation", "strength")
    SOURCE_IDS_FIELD_NUMBER: _ClassVar[int]
    TARGET_IDS_FIELD_NUMBER: _ClassVar[int]
    RELATION_FIELD_NUMBER: _ClassVar[int]
    STRENGTH_FIELD_NUMBER: _ClassVar[int]
    source_ids: _containers.RepeatedScalarFieldContainer[str]
    target_ids: _containers.RepeatedScalarFieldContainer[str]
    relation: str
    strength: float
    def __init__(self, source_ids: _Optional[_Iterable[str]] = ..., target_ids: _Optional[_Iterable[str]] = ..., relation: _Optional[str] = ..., strength: _Optional[float] = ...) -> None: ...

class CIG(_message.Message):
    __slots__ = ("project_id", "objectives", "edges", "constraints", "created_by", "hyperedges")
    class ConstraintsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    PROJECT_ID_FIELD_NUMBER: _ClassVar[int]
    OBJECTIVES_FIELD_NUMBER: _ClassVar[int]
    EDGES_FIELD_NUMBER: _ClassVar[int]
    CONSTRAINTS_FIELD_NUMBER: _ClassVar[int]
    CREATED_BY_FIELD_NUMBER: _ClassVar[int]
    HYPEREDGES_FIELD_NUMBER: _ClassVar[int]
    project_id: str
    objectives: _containers.RepeatedCompositeFieldContainer[ObjectiveNode]
    edges: _containers.RepeatedCompositeFieldContainer[ObjectiveEdge]
    constraints: _containers.ScalarMap[str, str]
    created_by: str
    hyperedges: _containers.RepeatedCompositeFieldContainer[ObjectiveHyperedge]
    def __init__(self, project_id: _Optional[str] = ..., objectives: _Optional[_Iterable[_Union[ObjectiveNode, _Mapping]]] = ..., edges: _Optional[_Iterable[_Union[ObjectiveEdge, _Mapping]]] = ..., constraints: _Optional[_Mapping[str, str]] = ..., created_by: _Optional[str] = ..., hyperedges: _Optional[_Iterable[_Union[ObjectiveHyperedge, _Mapping]]] = ...) -> None: ...

class CIGCompileRequest(_message.Message):
    __slots__ = ("project_id", "nl_query", "seed")
    PROJECT_ID_FIELD_NUMBER: _ClassVar[int]
    NL_QUERY_FIELD_NUMBER: _ClassVar[int]
    SEED_FIELD_NUMBER: _ClassVar[int]
    project_id: str
    nl_query: str
    seed: int
    def __init__(self, project_id: _Optional[str] = ..., nl_query: _Optional[str] = ..., seed: _Optional[int] = ...) -> None: ...

class CIGCompileResponse(_message.Message):
    __slots__ = ("cig", "hciv", "intent_cone", "parse_confidence", "ambiguities", "elapsed_ms")
    CIG_FIELD_NUMBER: _ClassVar[int]
    HCIV_FIELD_NUMBER: _ClassVar[int]
    INTENT_CONE_FIELD_NUMBER: _ClassVar[int]
    PARSE_CONFIDENCE_FIELD_NUMBER: _ClassVar[int]
    AMBIGUITIES_FIELD_NUMBER: _ClassVar[int]
    ELAPSED_MS_FIELD_NUMBER: _ClassVar[int]
    cig: CIG
    hciv: _humu_pb2.HCIV
    intent_cone: _humu_pb2.IntentCone
    parse_confidence: float
    ambiguities: _containers.RepeatedScalarFieldContainer[str]
    elapsed_ms: int
    def __init__(self, cig: _Optional[_Union[CIG, _Mapping]] = ..., hciv: _Optional[_Union[_humu_pb2.HCIV, _Mapping]] = ..., intent_cone: _Optional[_Union[_humu_pb2.IntentCone, _Mapping]] = ..., parse_confidence: _Optional[float] = ..., ambiguities: _Optional[_Iterable[str]] = ..., elapsed_ms: _Optional[int] = ...) -> None: ...

class CIGValidationRequest(_message.Message):
    __slots__ = ("cig",)
    CIG_FIELD_NUMBER: _ClassVar[int]
    cig: CIG
    def __init__(self, cig: _Optional[_Union[CIG, _Mapping]] = ...) -> None: ...

class CIGValidationResponse(_message.Message):
    __slots__ = ("valid", "issues", "warnings", "suggestions")
    VALID_FIELD_NUMBER: _ClassVar[int]
    ISSUES_FIELD_NUMBER: _ClassVar[int]
    WARNINGS_FIELD_NUMBER: _ClassVar[int]
    SUGGESTIONS_FIELD_NUMBER: _ClassVar[int]
    valid: bool
    issues: _containers.RepeatedScalarFieldContainer[str]
    warnings: _containers.RepeatedScalarFieldContainer[str]
    suggestions: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, valid: bool = ..., issues: _Optional[_Iterable[str]] = ..., warnings: _Optional[_Iterable[str]] = ..., suggestions: _Optional[_Iterable[str]] = ...) -> None: ...

class CIGRefineRequest(_message.Message):
    __slots__ = ("cig", "feedback", "context")
    class ContextEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    CIG_FIELD_NUMBER: _ClassVar[int]
    FEEDBACK_FIELD_NUMBER: _ClassVar[int]
    CONTEXT_FIELD_NUMBER: _ClassVar[int]
    cig: CIG
    feedback: str
    context: _containers.ScalarMap[str, str]
    def __init__(self, cig: _Optional[_Union[CIG, _Mapping]] = ..., feedback: _Optional[str] = ..., context: _Optional[_Mapping[str, str]] = ...) -> None: ...
