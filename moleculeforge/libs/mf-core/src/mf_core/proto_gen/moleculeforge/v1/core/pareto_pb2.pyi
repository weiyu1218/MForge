from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ParetoPoint(_message.Message):
    __slots__ = ("molecule_smiles", "objective_values", "pareto_tier", "humu_embedding", "additional_metrics")
    class AdditionalMetricsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: float
        def __init__(self, key: _Optional[str] = ..., value: _Optional[float] = ...) -> None: ...
    MOLECULE_SMILES_FIELD_NUMBER: _ClassVar[int]
    OBJECTIVE_VALUES_FIELD_NUMBER: _ClassVar[int]
    PARETO_TIER_FIELD_NUMBER: _ClassVar[int]
    HUMU_EMBEDDING_FIELD_NUMBER: _ClassVar[int]
    ADDITIONAL_METRICS_FIELD_NUMBER: _ClassVar[int]
    molecule_smiles: str
    objective_values: _containers.RepeatedScalarFieldContainer[float]
    pareto_tier: int
    humu_embedding: bytes
    additional_metrics: _containers.ScalarMap[str, float]
    def __init__(self, molecule_smiles: _Optional[str] = ..., objective_values: _Optional[_Iterable[float]] = ..., pareto_tier: _Optional[int] = ..., humu_embedding: _Optional[bytes] = ..., additional_metrics: _Optional[_Mapping[str, float]] = ...) -> None: ...

class ParetoFrontier(_message.Message):
    __slots__ = ("project_id", "points", "iteration", "hypervolume", "reference_point")
    PROJECT_ID_FIELD_NUMBER: _ClassVar[int]
    POINTS_FIELD_NUMBER: _ClassVar[int]
    ITERATION_FIELD_NUMBER: _ClassVar[int]
    HYPERVOLUME_FIELD_NUMBER: _ClassVar[int]
    REFERENCE_POINT_FIELD_NUMBER: _ClassVar[int]
    project_id: str
    points: _containers.RepeatedCompositeFieldContainer[ParetoPoint]
    iteration: int
    hypervolume: float
    reference_point: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, project_id: _Optional[str] = ..., points: _Optional[_Iterable[_Union[ParetoPoint, _Mapping]]] = ..., iteration: _Optional[int] = ..., hypervolume: _Optional[float] = ..., reference_point: _Optional[_Iterable[float]] = ...) -> None: ...
