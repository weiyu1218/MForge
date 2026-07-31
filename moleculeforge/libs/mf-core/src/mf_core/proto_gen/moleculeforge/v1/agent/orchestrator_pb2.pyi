from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

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
