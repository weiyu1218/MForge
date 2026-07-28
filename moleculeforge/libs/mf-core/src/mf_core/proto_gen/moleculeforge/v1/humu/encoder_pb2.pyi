from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class EncodeRequest(_message.Message):
    __slots__ = ("entity_type", "input_data", "params", "checkpoint_version")
    class ParamsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    ENTITY_TYPE_FIELD_NUMBER: _ClassVar[int]
    INPUT_DATA_FIELD_NUMBER: _ClassVar[int]
    PARAMS_FIELD_NUMBER: _ClassVar[int]
    CHECKPOINT_VERSION_FIELD_NUMBER: _ClassVar[int]
    entity_type: str
    input_data: bytes
    params: _containers.ScalarMap[str, str]
    checkpoint_version: str
    def __init__(self, entity_type: _Optional[str] = ..., input_data: _Optional[bytes] = ..., params: _Optional[_Mapping[str, str]] = ..., checkpoint_version: _Optional[str] = ...) -> None: ...

class EncodeResponse(_message.Message):
    __slots__ = ("humu_embedding", "curvature", "elapsed_ms", "checkpoint_version", "checkpoint_checksum", "embedding_dimension")
    HUMU_EMBEDDING_FIELD_NUMBER: _ClassVar[int]
    CURVATURE_FIELD_NUMBER: _ClassVar[int]
    ELAPSED_MS_FIELD_NUMBER: _ClassVar[int]
    CHECKPOINT_VERSION_FIELD_NUMBER: _ClassVar[int]
    CHECKPOINT_CHECKSUM_FIELD_NUMBER: _ClassVar[int]
    EMBEDDING_DIMENSION_FIELD_NUMBER: _ClassVar[int]
    humu_embedding: bytes
    curvature: float
    elapsed_ms: int
    checkpoint_version: str
    checkpoint_checksum: str
    embedding_dimension: int
    def __init__(self, humu_embedding: _Optional[bytes] = ..., curvature: _Optional[float] = ..., elapsed_ms: _Optional[int] = ..., checkpoint_version: _Optional[str] = ..., checkpoint_checksum: _Optional[str] = ..., embedding_dimension: _Optional[int] = ...) -> None: ...

class BatchEncodeRequest(_message.Message):
    __slots__ = ("requests", "batch_id")
    REQUESTS_FIELD_NUMBER: _ClassVar[int]
    BATCH_ID_FIELD_NUMBER: _ClassVar[int]
    requests: _containers.RepeatedCompositeFieldContainer[EncodeRequest]
    batch_id: str
    def __init__(self, requests: _Optional[_Iterable[_Union[EncodeRequest, _Mapping]]] = ..., batch_id: _Optional[str] = ...) -> None: ...

class BatchEncodeResponse(_message.Message):
    __slots__ = ("responses", "batch_id", "total_elapsed_ms")
    RESPONSES_FIELD_NUMBER: _ClassVar[int]
    BATCH_ID_FIELD_NUMBER: _ClassVar[int]
    TOTAL_ELAPSED_MS_FIELD_NUMBER: _ClassVar[int]
    responses: _containers.RepeatedCompositeFieldContainer[EncodeResponse]
    batch_id: str
    total_elapsed_ms: int
    def __init__(self, responses: _Optional[_Iterable[_Union[EncodeResponse, _Mapping]]] = ..., batch_id: _Optional[str] = ..., total_elapsed_ms: _Optional[int] = ...) -> None: ...
