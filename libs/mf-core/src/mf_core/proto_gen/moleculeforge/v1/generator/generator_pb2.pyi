from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class GenerateRequest(_message.Message):
    __slots__ = ("project_id", "batch_size", "total_molecules", "intent_cone", "target_properties", "property_targets", "checkpoint_version", "generator_params", "timeout_seconds")
    class PropertyTargetsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: float
        def __init__(self, key: _Optional[str] = ..., value: _Optional[float] = ...) -> None: ...
    class GeneratorParamsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    PROJECT_ID_FIELD_NUMBER: _ClassVar[int]
    BATCH_SIZE_FIELD_NUMBER: _ClassVar[int]
    TOTAL_MOLECULES_FIELD_NUMBER: _ClassVar[int]
    INTENT_CONE_FIELD_NUMBER: _ClassVar[int]
    TARGET_PROPERTIES_FIELD_NUMBER: _ClassVar[int]
    PROPERTY_TARGETS_FIELD_NUMBER: _ClassVar[int]
    CHECKPOINT_VERSION_FIELD_NUMBER: _ClassVar[int]
    GENERATOR_PARAMS_FIELD_NUMBER: _ClassVar[int]
    TIMEOUT_SECONDS_FIELD_NUMBER: _ClassVar[int]
    project_id: str
    batch_size: int
    total_molecules: int
    intent_cone: bytes
    target_properties: _containers.RepeatedScalarFieldContainer[str]
    property_targets: _containers.ScalarMap[str, float]
    checkpoint_version: str
    generator_params: _containers.ScalarMap[str, str]
    timeout_seconds: int
    def __init__(self, project_id: _Optional[str] = ..., batch_size: _Optional[int] = ..., total_molecules: _Optional[int] = ..., intent_cone: _Optional[bytes] = ..., target_properties: _Optional[_Iterable[str]] = ..., property_targets: _Optional[_Mapping[str, float]] = ..., checkpoint_version: _Optional[str] = ..., generator_params: _Optional[_Mapping[str, str]] = ..., timeout_seconds: _Optional[int] = ...) -> None: ...

class GenerateResponse(_message.Message):
    __slots__ = ("generator_name", "generation_id", "molecules", "humu_embeddings", "aggregate_stats", "elapsed_ms")
    class AggregateStatsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: float
        def __init__(self, key: _Optional[str] = ..., value: _Optional[float] = ...) -> None: ...
    GENERATOR_NAME_FIELD_NUMBER: _ClassVar[int]
    GENERATION_ID_FIELD_NUMBER: _ClassVar[int]
    MOLECULES_FIELD_NUMBER: _ClassVar[int]
    HUMU_EMBEDDINGS_FIELD_NUMBER: _ClassVar[int]
    AGGREGATE_STATS_FIELD_NUMBER: _ClassVar[int]
    ELAPSED_MS_FIELD_NUMBER: _ClassVar[int]
    generator_name: str
    generation_id: str
    molecules: _containers.RepeatedScalarFieldContainer[bytes]
    humu_embeddings: _containers.RepeatedScalarFieldContainer[bytes]
    aggregate_stats: _containers.ScalarMap[str, float]
    elapsed_ms: int
    def __init__(self, generator_name: _Optional[str] = ..., generation_id: _Optional[str] = ..., molecules: _Optional[_Iterable[bytes]] = ..., humu_embeddings: _Optional[_Iterable[bytes]] = ..., aggregate_stats: _Optional[_Mapping[str, float]] = ..., elapsed_ms: _Optional[int] = ...) -> None: ...

class GeneratorInfo(_message.Message):
    __slots__ = ("name", "version", "description", "supported_properties", "max_batch_size", "supports_streaming", "requires_gpu", "default_params")
    class DefaultParamsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    NAME_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    SUPPORTED_PROPERTIES_FIELD_NUMBER: _ClassVar[int]
    MAX_BATCH_SIZE_FIELD_NUMBER: _ClassVar[int]
    SUPPORTS_STREAMING_FIELD_NUMBER: _ClassVar[int]
    REQUIRES_GPU_FIELD_NUMBER: _ClassVar[int]
    DEFAULT_PARAMS_FIELD_NUMBER: _ClassVar[int]
    name: str
    version: str
    description: str
    supported_properties: _containers.RepeatedScalarFieldContainer[str]
    max_batch_size: int
    supports_streaming: bool
    requires_gpu: bool
    default_params: _containers.ScalarMap[str, str]
    def __init__(self, name: _Optional[str] = ..., version: _Optional[str] = ..., description: _Optional[str] = ..., supported_properties: _Optional[_Iterable[str]] = ..., max_batch_size: _Optional[int] = ..., supports_streaming: bool = ..., requires_gpu: bool = ..., default_params: _Optional[_Mapping[str, str]] = ...) -> None: ...
