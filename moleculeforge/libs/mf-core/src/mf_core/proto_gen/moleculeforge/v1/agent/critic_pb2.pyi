from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class CriticFeedback(_message.Message):
    __slots__ = ("molecule_smiles", "rule_id", "rule_name", "verdict", "score", "reasoning", "violated_constraints", "satisfied_constraints", "metric_values")
    class MetricValuesEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: float
        def __init__(self, key: _Optional[str] = ..., value: _Optional[float] = ...) -> None: ...
    MOLECULE_SMILES_FIELD_NUMBER: _ClassVar[int]
    RULE_ID_FIELD_NUMBER: _ClassVar[int]
    RULE_NAME_FIELD_NUMBER: _ClassVar[int]
    VERDICT_FIELD_NUMBER: _ClassVar[int]
    SCORE_FIELD_NUMBER: _ClassVar[int]
    REASONING_FIELD_NUMBER: _ClassVar[int]
    VIOLATED_CONSTRAINTS_FIELD_NUMBER: _ClassVar[int]
    SATISFIED_CONSTRAINTS_FIELD_NUMBER: _ClassVar[int]
    METRIC_VALUES_FIELD_NUMBER: _ClassVar[int]
    molecule_smiles: str
    rule_id: str
    rule_name: str
    verdict: str
    score: float
    reasoning: str
    violated_constraints: _containers.RepeatedScalarFieldContainer[str]
    satisfied_constraints: _containers.RepeatedScalarFieldContainer[str]
    metric_values: _containers.ScalarMap[str, float]
    def __init__(self, molecule_smiles: _Optional[str] = ..., rule_id: _Optional[str] = ..., rule_name: _Optional[str] = ..., verdict: _Optional[str] = ..., score: _Optional[float] = ..., reasoning: _Optional[str] = ..., violated_constraints: _Optional[_Iterable[str]] = ..., satisfied_constraints: _Optional[_Iterable[str]] = ..., metric_values: _Optional[_Mapping[str, float]] = ...) -> None: ...

class CriticBatchResult(_message.Message):
    __slots__ = ("molecule_smiles", "project_id", "rule_results", "all_passed", "rules_evaluated", "rules_passed", "aggregate_score", "candidate_id", "candidate_index", "canonical_smiles", "run_id", "request_id", "schema_version")
    MOLECULE_SMILES_FIELD_NUMBER: _ClassVar[int]
    PROJECT_ID_FIELD_NUMBER: _ClassVar[int]
    RULE_RESULTS_FIELD_NUMBER: _ClassVar[int]
    ALL_PASSED_FIELD_NUMBER: _ClassVar[int]
    RULES_EVALUATED_FIELD_NUMBER: _ClassVar[int]
    RULES_PASSED_FIELD_NUMBER: _ClassVar[int]
    AGGREGATE_SCORE_FIELD_NUMBER: _ClassVar[int]
    CANDIDATE_ID_FIELD_NUMBER: _ClassVar[int]
    CANDIDATE_INDEX_FIELD_NUMBER: _ClassVar[int]
    CANONICAL_SMILES_FIELD_NUMBER: _ClassVar[int]
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    SCHEMA_VERSION_FIELD_NUMBER: _ClassVar[int]
    molecule_smiles: str
    project_id: str
    rule_results: _containers.RepeatedCompositeFieldContainer[CriticFeedback]
    all_passed: bool
    rules_evaluated: int
    rules_passed: int
    aggregate_score: float
    candidate_id: str
    candidate_index: int
    canonical_smiles: str
    run_id: str
    request_id: str
    schema_version: str
    def __init__(self, molecule_smiles: _Optional[str] = ..., project_id: _Optional[str] = ..., rule_results: _Optional[_Iterable[_Union[CriticFeedback, _Mapping]]] = ..., all_passed: bool = ..., rules_evaluated: _Optional[int] = ..., rules_passed: _Optional[int] = ..., aggregate_score: _Optional[float] = ..., candidate_id: _Optional[str] = ..., candidate_index: _Optional[int] = ..., canonical_smiles: _Optional[str] = ..., run_id: _Optional[str] = ..., request_id: _Optional[str] = ..., schema_version: _Optional[str] = ...) -> None: ...
