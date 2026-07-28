from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class AgentMessage(_message.Message):
    __slots__ = ("trace_id", "message_id", "sender", "recipient", "message_type", "reply_to", "payload", "payload_type_url", "timestamp_ns", "signature", "lineage", "ttl", "run_id", "request_id", "parent_id", "schema_version")
    class LineageEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    TRACE_ID_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_ID_FIELD_NUMBER: _ClassVar[int]
    SENDER_FIELD_NUMBER: _ClassVar[int]
    RECIPIENT_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_TYPE_FIELD_NUMBER: _ClassVar[int]
    REPLY_TO_FIELD_NUMBER: _ClassVar[int]
    PAYLOAD_FIELD_NUMBER: _ClassVar[int]
    PAYLOAD_TYPE_URL_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_NS_FIELD_NUMBER: _ClassVar[int]
    SIGNATURE_FIELD_NUMBER: _ClassVar[int]
    LINEAGE_FIELD_NUMBER: _ClassVar[int]
    TTL_FIELD_NUMBER: _ClassVar[int]
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    PARENT_ID_FIELD_NUMBER: _ClassVar[int]
    SCHEMA_VERSION_FIELD_NUMBER: _ClassVar[int]
    trace_id: str
    message_id: str
    sender: str
    recipient: str
    message_type: str
    reply_to: str
    payload: bytes
    payload_type_url: str
    timestamp_ns: int
    signature: bytes
    lineage: _containers.ScalarMap[str, str]
    ttl: int
    run_id: str
    request_id: str
    parent_id: str
    schema_version: str
    def __init__(self, trace_id: _Optional[str] = ..., message_id: _Optional[str] = ..., sender: _Optional[str] = ..., recipient: _Optional[str] = ..., message_type: _Optional[str] = ..., reply_to: _Optional[str] = ..., payload: _Optional[bytes] = ..., payload_type_url: _Optional[str] = ..., timestamp_ns: _Optional[int] = ..., signature: _Optional[bytes] = ..., lineage: _Optional[_Mapping[str, str]] = ..., ttl: _Optional[int] = ..., run_id: _Optional[str] = ..., request_id: _Optional[str] = ..., parent_id: _Optional[str] = ..., schema_version: _Optional[str] = ...) -> None: ...

class AgentHeartbeat(_message.Message):
    __slots__ = ("agent_name", "status", "cpu_percent", "memory_mb", "active_jobs")
    AGENT_NAME_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    CPU_PERCENT_FIELD_NUMBER: _ClassVar[int]
    MEMORY_MB_FIELD_NUMBER: _ClassVar[int]
    ACTIVE_JOBS_FIELD_NUMBER: _ClassVar[int]
    agent_name: str
    status: str
    cpu_percent: float
    memory_mb: float
    active_jobs: int
    def __init__(self, agent_name: _Optional[str] = ..., status: _Optional[str] = ..., cpu_percent: _Optional[float] = ..., memory_mb: _Optional[float] = ..., active_jobs: _Optional[int] = ...) -> None: ...

class AgentCapability(_message.Message):
    __slots__ = ("agent_name", "supported_actions", "input_types", "output_types")
    AGENT_NAME_FIELD_NUMBER: _ClassVar[int]
    SUPPORTED_ACTIONS_FIELD_NUMBER: _ClassVar[int]
    INPUT_TYPES_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_TYPES_FIELD_NUMBER: _ClassVar[int]
    agent_name: str
    supported_actions: _containers.RepeatedScalarFieldContainer[str]
    input_types: _containers.RepeatedScalarFieldContainer[str]
    output_types: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, agent_name: _Optional[str] = ..., supported_actions: _Optional[_Iterable[str]] = ..., input_types: _Optional[_Iterable[str]] = ..., output_types: _Optional[_Iterable[str]] = ...) -> None: ...
