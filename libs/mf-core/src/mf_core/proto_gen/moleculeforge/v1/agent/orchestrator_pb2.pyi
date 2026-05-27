from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional

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
