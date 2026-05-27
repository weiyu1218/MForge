from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class RouterRequest(_message.Message):
    __slots__ = ("project_id", "cig", "generator_weights", "generator_performance", "n_select")
    PROJECT_ID_FIELD_NUMBER: _ClassVar[int]
    CIG_FIELD_NUMBER: _ClassVar[int]
    GENERATOR_WEIGHTS_FIELD_NUMBER: _ClassVar[int]
    GENERATOR_PERFORMANCE_FIELD_NUMBER: _ClassVar[int]
    N_SELECT_FIELD_NUMBER: _ClassVar[int]
    project_id: str
    cig: bytes
    generator_weights: _containers.RepeatedScalarFieldContainer[float]
    generator_performance: _containers.RepeatedScalarFieldContainer[float]
    n_select: int
    def __init__(self, project_id: _Optional[str] = ..., cig: _Optional[bytes] = ..., generator_weights: _Optional[_Iterable[float]] = ..., generator_performance: _Optional[_Iterable[float]] = ..., n_select: _Optional[int] = ...) -> None: ...

class RouterResponse(_message.Message):
    __slots__ = ("selected_generators", "selection_weights", "strategy", "expected_rewards")
    SELECTED_GENERATORS_FIELD_NUMBER: _ClassVar[int]
    SELECTION_WEIGHTS_FIELD_NUMBER: _ClassVar[int]
    STRATEGY_FIELD_NUMBER: _ClassVar[int]
    EXPECTED_REWARDS_FIELD_NUMBER: _ClassVar[int]
    selected_generators: _containers.RepeatedScalarFieldContainer[str]
    selection_weights: _containers.RepeatedScalarFieldContainer[float]
    strategy: str
    expected_rewards: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, selected_generators: _Optional[_Iterable[str]] = ..., selection_weights: _Optional[_Iterable[float]] = ..., strategy: _Optional[str] = ..., expected_rewards: _Optional[_Iterable[float]] = ...) -> None: ...
