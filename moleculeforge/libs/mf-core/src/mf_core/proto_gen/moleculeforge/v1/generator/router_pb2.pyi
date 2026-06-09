from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class RouterRequest(_message.Message):
    __slots__ = ("project_id", "cig", "generator_weights", "generator_performance", "n_select", "hciv", "target_family", "stage", "data_richness", "novelty_demand", "multi_target", "sa_constraint", "n_samples")
    PROJECT_ID_FIELD_NUMBER: _ClassVar[int]
    CIG_FIELD_NUMBER: _ClassVar[int]
    GENERATOR_WEIGHTS_FIELD_NUMBER: _ClassVar[int]
    GENERATOR_PERFORMANCE_FIELD_NUMBER: _ClassVar[int]
    N_SELECT_FIELD_NUMBER: _ClassVar[int]
    HCIV_FIELD_NUMBER: _ClassVar[int]
    TARGET_FAMILY_FIELD_NUMBER: _ClassVar[int]
    STAGE_FIELD_NUMBER: _ClassVar[int]
    DATA_RICHNESS_FIELD_NUMBER: _ClassVar[int]
    NOVELTY_DEMAND_FIELD_NUMBER: _ClassVar[int]
    MULTI_TARGET_FIELD_NUMBER: _ClassVar[int]
    SA_CONSTRAINT_FIELD_NUMBER: _ClassVar[int]
    N_SAMPLES_FIELD_NUMBER: _ClassVar[int]
    project_id: str
    cig: bytes
    generator_weights: _containers.RepeatedScalarFieldContainer[float]
    generator_performance: _containers.RepeatedScalarFieldContainer[float]
    n_select: int
    hciv: _containers.RepeatedScalarFieldContainer[float]
    target_family: str
    stage: str
    data_richness: float
    novelty_demand: float
    multi_target: bool
    sa_constraint: float
    n_samples: int
    def __init__(self, project_id: _Optional[str] = ..., cig: _Optional[bytes] = ..., generator_weights: _Optional[_Iterable[float]] = ..., generator_performance: _Optional[_Iterable[float]] = ..., n_select: _Optional[int] = ..., hciv: _Optional[_Iterable[float]] = ..., target_family: _Optional[str] = ..., stage: _Optional[str] = ..., data_richness: _Optional[float] = ..., novelty_demand: _Optional[float] = ..., multi_target: bool = ..., sa_constraint: _Optional[float] = ..., n_samples: _Optional[int] = ...) -> None: ...

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

class RouterProxylessSearchRequest(_message.Message):
    __slots__ = ("reward_batches_json", "generator_costs_json", "cost_weight", "learning_rate", "temperature")
    REWARD_BATCHES_JSON_FIELD_NUMBER: _ClassVar[int]
    GENERATOR_COSTS_JSON_FIELD_NUMBER: _ClassVar[int]
    COST_WEIGHT_FIELD_NUMBER: _ClassVar[int]
    LEARNING_RATE_FIELD_NUMBER: _ClassVar[int]
    TEMPERATURE_FIELD_NUMBER: _ClassVar[int]
    reward_batches_json: str
    generator_costs_json: str
    cost_weight: float
    learning_rate: float
    temperature: float
    def __init__(self, reward_batches_json: _Optional[str] = ..., generator_costs_json: _Optional[str] = ..., cost_weight: _Optional[float] = ..., learning_rate: _Optional[float] = ..., temperature: _Optional[float] = ...) -> None: ...

class RouterProxylessSearchResponse(_message.Message):
    __slots__ = ("acknowledged", "result_json", "generator_names", "architecture_probabilities", "round_count")
    ACKNOWLEDGED_FIELD_NUMBER: _ClassVar[int]
    RESULT_JSON_FIELD_NUMBER: _ClassVar[int]
    GENERATOR_NAMES_FIELD_NUMBER: _ClassVar[int]
    ARCHITECTURE_PROBABILITIES_FIELD_NUMBER: _ClassVar[int]
    ROUND_COUNT_FIELD_NUMBER: _ClassVar[int]
    acknowledged: bool
    result_json: str
    generator_names: _containers.RepeatedScalarFieldContainer[str]
    architecture_probabilities: _containers.RepeatedScalarFieldContainer[float]
    round_count: int
    def __init__(self, acknowledged: bool = ..., result_json: _Optional[str] = ..., generator_names: _Optional[_Iterable[str]] = ..., architecture_probabilities: _Optional[_Iterable[float]] = ..., round_count: _Optional[int] = ...) -> None: ...
