from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class OrchestratorState(_message.Message):
    __slots__ = ("project_id", "run_id", "current_node", "cycle_count", "max_cycles", "visited_nodes", "pending_actions", "status", "started_at_ns", "last_updated_ns")
    PROJECT_ID_FIELD_NUMBER: _ClassVar[int]
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    CURRENT_NODE_FIELD_NUMBER: _ClassVar[int]
    CYCLE_COUNT_FIELD_NUMBER: _ClassVar[int]
    MAX_CYCLES_FIELD_NUMBER: _ClassVar[int]
    VISITED_NODES_FIELD_NUMBER: _ClassVar[int]
    PENDING_ACTIONS_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    STARTED_AT_NS_FIELD_NUMBER: _ClassVar[int]
    LAST_UPDATED_NS_FIELD_NUMBER: _ClassVar[int]
    project_id: str
    run_id: str
    current_node: str
    cycle_count: int
    max_cycles: int
    visited_nodes: _containers.RepeatedScalarFieldContainer[str]
    pending_actions: _containers.RepeatedScalarFieldContainer[str]
    status: str
    started_at_ns: int
    last_updated_ns: int
    def __init__(self, project_id: _Optional[str] = ..., run_id: _Optional[str] = ..., current_node: _Optional[str] = ..., cycle_count: _Optional[int] = ..., max_cycles: _Optional[int] = ..., visited_nodes: _Optional[_Iterable[str]] = ..., pending_actions: _Optional[_Iterable[str]] = ..., status: _Optional[str] = ..., started_at_ns: _Optional[int] = ..., last_updated_ns: _Optional[int] = ...) -> None: ...

class NL2ObjRequest(_message.Message):
    __slots__ = ("project_id", "natural_language_prompt", "context")
    class ContextEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    PROJECT_ID_FIELD_NUMBER: _ClassVar[int]
    NATURAL_LANGUAGE_PROMPT_FIELD_NUMBER: _ClassVar[int]
    CONTEXT_FIELD_NUMBER: _ClassVar[int]
    project_id: str
    natural_language_prompt: str
    context: _containers.ScalarMap[str, str]
    def __init__(self, project_id: _Optional[str] = ..., natural_language_prompt: _Optional[str] = ..., context: _Optional[_Mapping[str, str]] = ...) -> None: ...

class ObjectiveSpec(_message.Message):
    __slots__ = ("name", "description", "direction", "target_value", "unit", "priority")
    NAME_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    DIRECTION_FIELD_NUMBER: _ClassVar[int]
    TARGET_VALUE_FIELD_NUMBER: _ClassVar[int]
    UNIT_FIELD_NUMBER: _ClassVar[int]
    PRIORITY_FIELD_NUMBER: _ClassVar[int]
    name: str
    description: str
    direction: str
    target_value: float
    unit: str
    priority: str
    def __init__(self, name: _Optional[str] = ..., description: _Optional[str] = ..., direction: _Optional[str] = ..., target_value: _Optional[float] = ..., unit: _Optional[str] = ..., priority: _Optional[str] = ...) -> None: ...

class NL2ObjResponse(_message.Message):
    __slots__ = ("project_id", "nl_query", "objectives", "constraints", "confidence", "parsed_intent", "elapsed_ms")
    class ConstraintsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    PROJECT_ID_FIELD_NUMBER: _ClassVar[int]
    NL_QUERY_FIELD_NUMBER: _ClassVar[int]
    OBJECTIVES_FIELD_NUMBER: _ClassVar[int]
    CONSTRAINTS_FIELD_NUMBER: _ClassVar[int]
    CONFIDENCE_FIELD_NUMBER: _ClassVar[int]
    PARSED_INTENT_FIELD_NUMBER: _ClassVar[int]
    ELAPSED_MS_FIELD_NUMBER: _ClassVar[int]
    project_id: str
    nl_query: str
    objectives: _containers.RepeatedCompositeFieldContainer[ObjectiveSpec]
    constraints: _containers.ScalarMap[str, str]
    confidence: float
    parsed_intent: str
    elapsed_ms: int
    def __init__(self, project_id: _Optional[str] = ..., nl_query: _Optional[str] = ..., objectives: _Optional[_Iterable[_Union[ObjectiveSpec, _Mapping]]] = ..., constraints: _Optional[_Mapping[str, str]] = ..., confidence: _Optional[float] = ..., parsed_intent: _Optional[str] = ..., elapsed_ms: _Optional[int] = ...) -> None: ...

class NL2ObjRefineRequest(_message.Message):
    __slots__ = ("project_id", "objectives", "feedback")
    PROJECT_ID_FIELD_NUMBER: _ClassVar[int]
    OBJECTIVES_FIELD_NUMBER: _ClassVar[int]
    FEEDBACK_FIELD_NUMBER: _ClassVar[int]
    project_id: str
    objectives: _containers.RepeatedCompositeFieldContainer[ObjectiveSpec]
    feedback: str
    def __init__(self, project_id: _Optional[str] = ..., objectives: _Optional[_Iterable[_Union[ObjectiveSpec, _Mapping]]] = ..., feedback: _Optional[str] = ...) -> None: ...

class NL2ObjRefineResponse(_message.Message):
    __slots__ = ("objectives", "feedback_applied", "confidence")
    OBJECTIVES_FIELD_NUMBER: _ClassVar[int]
    FEEDBACK_APPLIED_FIELD_NUMBER: _ClassVar[int]
    CONFIDENCE_FIELD_NUMBER: _ClassVar[int]
    objectives: _containers.RepeatedCompositeFieldContainer[ObjectiveSpec]
    feedback_applied: str
    confidence: float
    def __init__(self, objectives: _Optional[_Iterable[_Union[ObjectiveSpec, _Mapping]]] = ..., feedback_applied: _Optional[str] = ..., confidence: _Optional[float] = ...) -> None: ...

class OrchestrationDecision(_message.Message):
    __slots__ = ("decision", "reasoning", "next_agents", "target_molecule_id", "parameters")
    class ParametersEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    DECISION_FIELD_NUMBER: _ClassVar[int]
    REASONING_FIELD_NUMBER: _ClassVar[int]
    NEXT_AGENTS_FIELD_NUMBER: _ClassVar[int]
    TARGET_MOLECULE_ID_FIELD_NUMBER: _ClassVar[int]
    PARAMETERS_FIELD_NUMBER: _ClassVar[int]
    decision: str
    reasoning: str
    next_agents: _containers.RepeatedScalarFieldContainer[str]
    target_molecule_id: str
    parameters: _containers.ScalarMap[str, str]
    def __init__(self, decision: _Optional[str] = ..., reasoning: _Optional[str] = ..., next_agents: _Optional[_Iterable[str]] = ..., target_molecule_id: _Optional[str] = ..., parameters: _Optional[_Mapping[str, str]] = ...) -> None: ...

class StartPipelineRequest(_message.Message):
    __slots__ = ("project_id", "nl_input", "objectives", "workflow_scope", "run_id", "trace_id", "validation_passed", "max_refinements", "validation_policy_json", "teacher_policy_json", "selection_policy_json", "external_evidence_json")
    PROJECT_ID_FIELD_NUMBER: _ClassVar[int]
    NL_INPUT_FIELD_NUMBER: _ClassVar[int]
    OBJECTIVES_FIELD_NUMBER: _ClassVar[int]
    WORKFLOW_SCOPE_FIELD_NUMBER: _ClassVar[int]
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    TRACE_ID_FIELD_NUMBER: _ClassVar[int]
    VALIDATION_PASSED_FIELD_NUMBER: _ClassVar[int]
    MAX_REFINEMENTS_FIELD_NUMBER: _ClassVar[int]
    VALIDATION_POLICY_JSON_FIELD_NUMBER: _ClassVar[int]
    TEACHER_POLICY_JSON_FIELD_NUMBER: _ClassVar[int]
    SELECTION_POLICY_JSON_FIELD_NUMBER: _ClassVar[int]
    EXTERNAL_EVIDENCE_JSON_FIELD_NUMBER: _ClassVar[int]
    project_id: str
    nl_input: str
    objectives: _containers.RepeatedScalarFieldContainer[str]
    workflow_scope: str
    run_id: str
    trace_id: str
    validation_passed: bool
    max_refinements: int
    validation_policy_json: str
    teacher_policy_json: str
    selection_policy_json: str
    external_evidence_json: str
    def __init__(self, project_id: _Optional[str] = ..., nl_input: _Optional[str] = ..., objectives: _Optional[_Iterable[str]] = ..., workflow_scope: _Optional[str] = ..., run_id: _Optional[str] = ..., trace_id: _Optional[str] = ..., validation_passed: bool = ..., max_refinements: _Optional[int] = ..., validation_policy_json: _Optional[str] = ..., teacher_policy_json: _Optional[str] = ..., selection_policy_json: _Optional[str] = ..., external_evidence_json: _Optional[str] = ...) -> None: ...

class PipelineResponse(_message.Message):
    __slots__ = ("design_id", "run_id", "trace_id", "project_id", "status", "n_objectives")
    DESIGN_ID_FIELD_NUMBER: _ClassVar[int]
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    TRACE_ID_FIELD_NUMBER: _ClassVar[int]
    PROJECT_ID_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    N_OBJECTIVES_FIELD_NUMBER: _ClassVar[int]
    design_id: str
    run_id: str
    trace_id: str
    project_id: str
    status: str
    n_objectives: int
    def __init__(self, design_id: _Optional[str] = ..., run_id: _Optional[str] = ..., trace_id: _Optional[str] = ..., project_id: _Optional[str] = ..., status: _Optional[str] = ..., n_objectives: _Optional[int] = ...) -> None: ...

class PipelineStateRequest(_message.Message):
    __slots__ = ("design_id",)
    DESIGN_ID_FIELD_NUMBER: _ClassVar[int]
    design_id: str
    def __init__(self, design_id: _Optional[str] = ...) -> None: ...

class PipelineStateResponse(_message.Message):
    __slots__ = ("design_id", "current_stage", "state_json")
    DESIGN_ID_FIELD_NUMBER: _ClassVar[int]
    CURRENT_STAGE_FIELD_NUMBER: _ClassVar[int]
    STATE_JSON_FIELD_NUMBER: _ClassVar[int]
    design_id: str
    current_stage: str
    state_json: str
    def __init__(self, design_id: _Optional[str] = ..., current_stage: _Optional[str] = ..., state_json: _Optional[str] = ...) -> None: ...
