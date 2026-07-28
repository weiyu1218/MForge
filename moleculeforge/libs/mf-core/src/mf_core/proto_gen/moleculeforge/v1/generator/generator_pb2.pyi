from mf_core.proto_gen.moleculeforge.v1.core import audit_pb2 as _audit_pb2
from mf_core.proto_gen.moleculeforge.v1.core import cig_pb2 as _cig_pb2
from mf_core.proto_gen.moleculeforge.v1.core import humu_pb2 as _humu_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class GenerateRequest(_message.Message):
    __slots__ = ("project_id", "batch_size", "total_molecules", "intent_cone", "target_properties", "property_targets", "checkpoint_version", "generator_params", "timeout_seconds", "request_id", "cig", "hciv", "context_schema_version")
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
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    CIG_FIELD_NUMBER: _ClassVar[int]
    HCIV_FIELD_NUMBER: _ClassVar[int]
    CONTEXT_SCHEMA_VERSION_FIELD_NUMBER: _ClassVar[int]
    project_id: str
    batch_size: int
    total_molecules: int
    intent_cone: bytes
    target_properties: _containers.RepeatedScalarFieldContainer[str]
    property_targets: _containers.ScalarMap[str, float]
    checkpoint_version: str
    generator_params: _containers.ScalarMap[str, str]
    timeout_seconds: int
    request_id: str
    cig: _cig_pb2.CIG
    hciv: _humu_pb2.HCIV
    context_schema_version: str
    def __init__(self, project_id: _Optional[str] = ..., batch_size: _Optional[int] = ..., total_molecules: _Optional[int] = ..., intent_cone: _Optional[bytes] = ..., target_properties: _Optional[_Iterable[str]] = ..., property_targets: _Optional[_Mapping[str, float]] = ..., checkpoint_version: _Optional[str] = ..., generator_params: _Optional[_Mapping[str, str]] = ..., timeout_seconds: _Optional[int] = ..., request_id: _Optional[str] = ..., cig: _Optional[_Union[_cig_pb2.CIG, _Mapping]] = ..., hciv: _Optional[_Union[_humu_pb2.HCIV, _Mapping]] = ..., context_schema_version: _Optional[str] = ...) -> None: ...

class GenerateResponse(_message.Message):
    __slots__ = ("generator_name", "generation_id", "molecules", "humu_embeddings", "aggregate_stats", "elapsed_ms", "request_id", "artifacts", "molecule_payload_schema", "embedding_payload_schema")
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
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    ARTIFACTS_FIELD_NUMBER: _ClassVar[int]
    MOLECULE_PAYLOAD_SCHEMA_FIELD_NUMBER: _ClassVar[int]
    EMBEDDING_PAYLOAD_SCHEMA_FIELD_NUMBER: _ClassVar[int]
    generator_name: str
    generation_id: str
    molecules: _containers.RepeatedScalarFieldContainer[bytes]
    humu_embeddings: _containers.RepeatedScalarFieldContainer[bytes]
    aggregate_stats: _containers.ScalarMap[str, float]
    elapsed_ms: int
    request_id: str
    artifacts: _containers.RepeatedCompositeFieldContainer[_audit_pb2.ArtifactRef]
    molecule_payload_schema: str
    embedding_payload_schema: str
    def __init__(self, generator_name: _Optional[str] = ..., generation_id: _Optional[str] = ..., molecules: _Optional[_Iterable[bytes]] = ..., humu_embeddings: _Optional[_Iterable[bytes]] = ..., aggregate_stats: _Optional[_Mapping[str, float]] = ..., elapsed_ms: _Optional[int] = ..., request_id: _Optional[str] = ..., artifacts: _Optional[_Iterable[_Union[_audit_pb2.ArtifactRef, _Mapping]]] = ..., molecule_payload_schema: _Optional[str] = ..., embedding_payload_schema: _Optional[str] = ...) -> None: ...

class GeneratorInfo(_message.Message):
    __slots__ = ("name", "version", "description", "supported_properties", "max_batch_size", "supports_streaming", "requires_gpu", "default_params", "runtime_status", "status_message", "artifacts")
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
    RUNTIME_STATUS_FIELD_NUMBER: _ClassVar[int]
    STATUS_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    ARTIFACTS_FIELD_NUMBER: _ClassVar[int]
    name: str
    version: str
    description: str
    supported_properties: _containers.RepeatedScalarFieldContainer[str]
    max_batch_size: int
    supports_streaming: bool
    requires_gpu: bool
    default_params: _containers.ScalarMap[str, str]
    runtime_status: _audit_pb2.GeneratorRuntimeStatus
    status_message: str
    artifacts: _containers.RepeatedCompositeFieldContainer[_audit_pb2.ArtifactRef]
    def __init__(self, name: _Optional[str] = ..., version: _Optional[str] = ..., description: _Optional[str] = ..., supported_properties: _Optional[_Iterable[str]] = ..., max_batch_size: _Optional[int] = ..., supports_streaming: bool = ..., requires_gpu: bool = ..., default_params: _Optional[_Mapping[str, str]] = ..., runtime_status: _Optional[_Union[_audit_pb2.GeneratorRuntimeStatus, str]] = ..., status_message: _Optional[str] = ..., artifacts: _Optional[_Iterable[_Union[_audit_pb2.ArtifactRef, _Mapping]]] = ...) -> None: ...

class ModelUpdateRequest(_message.Message):
    __slots__ = ("run_id", "request_id", "training_batch_json", "teacher_embeddings", "rows", "dim", "teacher_source", "teacher_version", "target_checkpoint_version")
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    TRAINING_BATCH_JSON_FIELD_NUMBER: _ClassVar[int]
    TEACHER_EMBEDDINGS_FIELD_NUMBER: _ClassVar[int]
    ROWS_FIELD_NUMBER: _ClassVar[int]
    DIM_FIELD_NUMBER: _ClassVar[int]
    TEACHER_SOURCE_FIELD_NUMBER: _ClassVar[int]
    TEACHER_VERSION_FIELD_NUMBER: _ClassVar[int]
    TARGET_CHECKPOINT_VERSION_FIELD_NUMBER: _ClassVar[int]
    run_id: str
    request_id: str
    training_batch_json: str
    teacher_embeddings: bytes
    rows: int
    dim: int
    teacher_source: str
    teacher_version: str
    target_checkpoint_version: str
    def __init__(self, run_id: _Optional[str] = ..., request_id: _Optional[str] = ..., training_batch_json: _Optional[str] = ..., teacher_embeddings: _Optional[bytes] = ..., rows: _Optional[int] = ..., dim: _Optional[int] = ..., teacher_source: _Optional[str] = ..., teacher_version: _Optional[str] = ..., target_checkpoint_version: _Optional[str] = ...) -> None: ...

class ModelUpdateResponse(_message.Message):
    __slots__ = ("acknowledged", "active_version", "artifacts", "updated_samples")
    ACKNOWLEDGED_FIELD_NUMBER: _ClassVar[int]
    ACTIVE_VERSION_FIELD_NUMBER: _ClassVar[int]
    ARTIFACTS_FIELD_NUMBER: _ClassVar[int]
    UPDATED_SAMPLES_FIELD_NUMBER: _ClassVar[int]
    acknowledged: bool
    active_version: str
    artifacts: _containers.RepeatedCompositeFieldContainer[_audit_pb2.ArtifactRef]
    updated_samples: int
    def __init__(self, acknowledged: bool = ..., active_version: _Optional[str] = ..., artifacts: _Optional[_Iterable[_Union[_audit_pb2.ArtifactRef, _Mapping]]] = ..., updated_samples: _Optional[int] = ...) -> None: ...
