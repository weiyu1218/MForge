from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class AuditEvent(_message.Message):
    __slots__ = ("trace_id", "event_id", "timestamp", "actor", "action", "target", "outcome", "signature", "signature_uri", "lineage")
    class LineageEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    TRACE_ID_FIELD_NUMBER: _ClassVar[int]
    EVENT_ID_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    ACTOR_FIELD_NUMBER: _ClassVar[int]
    ACTION_FIELD_NUMBER: _ClassVar[int]
    TARGET_FIELD_NUMBER: _ClassVar[int]
    OUTCOME_FIELD_NUMBER: _ClassVar[int]
    SIGNATURE_FIELD_NUMBER: _ClassVar[int]
    SIGNATURE_URI_FIELD_NUMBER: _ClassVar[int]
    LINEAGE_FIELD_NUMBER: _ClassVar[int]
    trace_id: str
    event_id: str
    timestamp: str
    actor: str
    action: str
    target: str
    outcome: str
    signature: bytes
    signature_uri: str
    lineage: _containers.ScalarMap[str, str]
    def __init__(self, trace_id: _Optional[str] = ..., event_id: _Optional[str] = ..., timestamp: _Optional[str] = ..., actor: _Optional[str] = ..., action: _Optional[str] = ..., target: _Optional[str] = ..., outcome: _Optional[str] = ..., signature: _Optional[bytes] = ..., signature_uri: _Optional[str] = ..., lineage: _Optional[_Mapping[str, str]] = ...) -> None: ...

class AuditQuery(_message.Message):
    __slots__ = ("trace_id", "project_id", "start_time_ns", "end_time_ns", "actors", "limit")
    TRACE_ID_FIELD_NUMBER: _ClassVar[int]
    PROJECT_ID_FIELD_NUMBER: _ClassVar[int]
    START_TIME_NS_FIELD_NUMBER: _ClassVar[int]
    END_TIME_NS_FIELD_NUMBER: _ClassVar[int]
    ACTORS_FIELD_NUMBER: _ClassVar[int]
    LIMIT_FIELD_NUMBER: _ClassVar[int]
    trace_id: str
    project_id: str
    start_time_ns: int
    end_time_ns: int
    actors: _containers.RepeatedScalarFieldContainer[str]
    limit: int
    def __init__(self, trace_id: _Optional[str] = ..., project_id: _Optional[str] = ..., start_time_ns: _Optional[int] = ..., end_time_ns: _Optional[int] = ..., actors: _Optional[_Iterable[str]] = ..., limit: _Optional[int] = ...) -> None: ...

class AuditReport(_message.Message):
    __slots__ = ("events", "total_count", "has_more", "next_page_token")
    EVENTS_FIELD_NUMBER: _ClassVar[int]
    TOTAL_COUNT_FIELD_NUMBER: _ClassVar[int]
    HAS_MORE_FIELD_NUMBER: _ClassVar[int]
    NEXT_PAGE_TOKEN_FIELD_NUMBER: _ClassVar[int]
    events: _containers.RepeatedCompositeFieldContainer[AuditEvent]
    total_count: int
    has_more: bool
    next_page_token: str
    def __init__(self, events: _Optional[_Iterable[_Union[AuditEvent, _Mapping]]] = ..., total_count: _Optional[int] = ..., has_more: bool = ..., next_page_token: _Optional[str] = ...) -> None: ...
