from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class TaskComplexity(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    TASK_COMPLEXITY_UNSPECIFIED: _ClassVar[TaskComplexity]
    TASK_COMPLEXITY_LOW: _ClassVar[TaskComplexity]
    TASK_COMPLEXITY_MEDIUM: _ClassVar[TaskComplexity]
    TASK_COMPLEXITY_HIGH: _ClassVar[TaskComplexity]

class RouterFeedbackPhase(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    ROUTER_FEEDBACK_PHASE_UNSPECIFIED: _ClassVar[RouterFeedbackPhase]
    ROUTER_FEEDBACK_PHASE_VALIDATION: _ClassVar[RouterFeedbackPhase]
    ROUTER_FEEDBACK_PHASE_CRITIC: _ClassVar[RouterFeedbackPhase]
TASK_COMPLEXITY_UNSPECIFIED: TaskComplexity
TASK_COMPLEXITY_LOW: TaskComplexity
TASK_COMPLEXITY_MEDIUM: TaskComplexity
TASK_COMPLEXITY_HIGH: TaskComplexity
ROUTER_FEEDBACK_PHASE_UNSPECIFIED: RouterFeedbackPhase
ROUTER_FEEDBACK_PHASE_VALIDATION: RouterFeedbackPhase
ROUTER_FEEDBACK_PHASE_CRITIC: RouterFeedbackPhase

class RouterRequest(_message.Message):
    __slots__ = ("project_id", "cig", "generator_weights", "generator_performance", "n_select", "hciv", "target_family", "stage", "data_richness", "novelty_demand", "multi_target", "sa_constraint", "n_samples", "request_id", "available_generator_names", "task_complexity")
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
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    AVAILABLE_GENERATOR_NAMES_FIELD_NUMBER: _ClassVar[int]
    TASK_COMPLEXITY_FIELD_NUMBER: _ClassVar[int]
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
    request_id: str
    available_generator_names: _containers.RepeatedScalarFieldContainer[str]
    task_complexity: TaskComplexity
    def __init__(self, project_id: _Optional[str] = ..., cig: _Optional[bytes] = ..., generator_weights: _Optional[_Iterable[float]] = ..., generator_performance: _Optional[_Iterable[float]] = ..., n_select: _Optional[int] = ..., hciv: _Optional[_Iterable[float]] = ..., target_family: _Optional[str] = ..., stage: _Optional[str] = ..., data_richness: _Optional[float] = ..., novelty_demand: _Optional[float] = ..., multi_target: bool = ..., sa_constraint: _Optional[float] = ..., n_samples: _Optional[int] = ..., request_id: _Optional[str] = ..., available_generator_names: _Optional[_Iterable[str]] = ..., task_complexity: _Optional[_Union[TaskComplexity, str]] = ...) -> None: ...

class GeneratorAllocation(_message.Message):
    __slots__ = ("generator_name", "n_samples", "normalized_weight", "expected_reward")
    GENERATOR_NAME_FIELD_NUMBER: _ClassVar[int]
    N_SAMPLES_FIELD_NUMBER: _ClassVar[int]
    NORMALIZED_WEIGHT_FIELD_NUMBER: _ClassVar[int]
    EXPECTED_REWARD_FIELD_NUMBER: _ClassVar[int]
    generator_name: str
    n_samples: int
    normalized_weight: float
    expected_reward: float
    def __init__(self, generator_name: _Optional[str] = ..., n_samples: _Optional[int] = ..., normalized_weight: _Optional[float] = ..., expected_reward: _Optional[float] = ...) -> None: ...

class RouterResponse(_message.Message):
    __slots__ = ("selected_generators", "selection_weights", "strategy", "expected_rewards", "allocations", "warnings", "state_version")
    SELECTED_GENERATORS_FIELD_NUMBER: _ClassVar[int]
    SELECTION_WEIGHTS_FIELD_NUMBER: _ClassVar[int]
    STRATEGY_FIELD_NUMBER: _ClassVar[int]
    EXPECTED_REWARDS_FIELD_NUMBER: _ClassVar[int]
    ALLOCATIONS_FIELD_NUMBER: _ClassVar[int]
    WARNINGS_FIELD_NUMBER: _ClassVar[int]
    STATE_VERSION_FIELD_NUMBER: _ClassVar[int]
    selected_generators: _containers.RepeatedScalarFieldContainer[str]
    selection_weights: _containers.RepeatedScalarFieldContainer[float]
    strategy: str
    expected_rewards: _containers.RepeatedScalarFieldContainer[float]
    allocations: _containers.RepeatedCompositeFieldContainer[GeneratorAllocation]
    warnings: _containers.RepeatedScalarFieldContainer[str]
    state_version: int
    def __init__(self, selected_generators: _Optional[_Iterable[str]] = ..., selection_weights: _Optional[_Iterable[float]] = ..., strategy: _Optional[str] = ..., expected_rewards: _Optional[_Iterable[float]] = ..., allocations: _Optional[_Iterable[_Union[GeneratorAllocation, _Mapping]]] = ..., warnings: _Optional[_Iterable[str]] = ..., state_version: _Optional[int] = ...) -> None: ...

class RouterFeedbackRequest(_message.Message):
    __slots__ = ("feedback_id", "run_id", "request_id", "iteration", "phase", "generator_name", "candidate_ids", "canonical_smiles", "evidence_ids", "teacher_score", "teacher_source", "teacher_version", "synthetic")
    FEEDBACK_ID_FIELD_NUMBER: _ClassVar[int]
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    ITERATION_FIELD_NUMBER: _ClassVar[int]
    PHASE_FIELD_NUMBER: _ClassVar[int]
    GENERATOR_NAME_FIELD_NUMBER: _ClassVar[int]
    CANDIDATE_IDS_FIELD_NUMBER: _ClassVar[int]
    CANONICAL_SMILES_FIELD_NUMBER: _ClassVar[int]
    EVIDENCE_IDS_FIELD_NUMBER: _ClassVar[int]
    TEACHER_SCORE_FIELD_NUMBER: _ClassVar[int]
    TEACHER_SOURCE_FIELD_NUMBER: _ClassVar[int]
    TEACHER_VERSION_FIELD_NUMBER: _ClassVar[int]
    SYNTHETIC_FIELD_NUMBER: _ClassVar[int]
    feedback_id: str
    run_id: str
    request_id: str
    iteration: int
    phase: RouterFeedbackPhase
    generator_name: str
    candidate_ids: _containers.RepeatedScalarFieldContainer[str]
    canonical_smiles: str
    evidence_ids: _containers.RepeatedScalarFieldContainer[str]
    teacher_score: float
    teacher_source: str
    teacher_version: str
    synthetic: bool
    def __init__(self, feedback_id: _Optional[str] = ..., run_id: _Optional[str] = ..., request_id: _Optional[str] = ..., iteration: _Optional[int] = ..., phase: _Optional[_Union[RouterFeedbackPhase, str]] = ..., generator_name: _Optional[str] = ..., candidate_ids: _Optional[_Iterable[str]] = ..., canonical_smiles: _Optional[str] = ..., evidence_ids: _Optional[_Iterable[str]] = ..., teacher_score: _Optional[float] = ..., teacher_source: _Optional[str] = ..., teacher_version: _Optional[str] = ..., synthetic: bool = ...) -> None: ...

class RouterFeedbackResponse(_message.Message):
    __slots__ = ("acknowledged", "duplicate", "state_version")
    ACKNOWLEDGED_FIELD_NUMBER: _ClassVar[int]
    DUPLICATE_FIELD_NUMBER: _ClassVar[int]
    STATE_VERSION_FIELD_NUMBER: _ClassVar[int]
    acknowledged: bool
    duplicate: bool
    state_version: int
    def __init__(self, acknowledged: bool = ..., duplicate: bool = ..., state_version: _Optional[int] = ...) -> None: ...

class RouterWeightsResponse(_message.Message):
    __slots__ = ("generator_names", "weights", "state_version")
    GENERATOR_NAMES_FIELD_NUMBER: _ClassVar[int]
    WEIGHTS_FIELD_NUMBER: _ClassVar[int]
    STATE_VERSION_FIELD_NUMBER: _ClassVar[int]
    generator_names: _containers.RepeatedScalarFieldContainer[str]
    weights: _containers.RepeatedScalarFieldContainer[float]
    state_version: int
    def __init__(self, generator_names: _Optional[_Iterable[str]] = ..., weights: _Optional[_Iterable[float]] = ..., state_version: _Optional[int] = ...) -> None: ...

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
