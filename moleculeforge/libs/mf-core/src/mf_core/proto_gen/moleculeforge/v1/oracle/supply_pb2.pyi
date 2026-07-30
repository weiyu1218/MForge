from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class AvailabilityRequest(_message.Message):
    __slots__ = ("smiles", "request_id", "project_id", "candidate_id", "candidate_index", "canonical_smiles")
    SMILES_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    PROJECT_ID_FIELD_NUMBER: _ClassVar[int]
    CANDIDATE_ID_FIELD_NUMBER: _ClassVar[int]
    CANDIDATE_INDEX_FIELD_NUMBER: _ClassVar[int]
    CANONICAL_SMILES_FIELD_NUMBER: _ClassVar[int]
    smiles: str
    request_id: str
    project_id: str
    candidate_id: str
    candidate_index: int
    canonical_smiles: str
    def __init__(self, smiles: _Optional[str] = ..., request_id: _Optional[str] = ..., project_id: _Optional[str] = ..., candidate_id: _Optional[str] = ..., candidate_index: _Optional[int] = ..., canonical_smiles: _Optional[str] = ...) -> None: ...

class AvailabilityResponse(_message.Message):
    __slots__ = ("smiles", "available", "catalog_id", "catalog_source", "source_timestamp", "price", "currency", "lead_time_days", "evidence_id", "catalog_version", "catalog_checksum", "request_id", "project_id", "candidate_id", "candidate_index", "canonical_smiles")
    SMILES_FIELD_NUMBER: _ClassVar[int]
    AVAILABLE_FIELD_NUMBER: _ClassVar[int]
    CATALOG_ID_FIELD_NUMBER: _ClassVar[int]
    CATALOG_SOURCE_FIELD_NUMBER: _ClassVar[int]
    SOURCE_TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    PRICE_FIELD_NUMBER: _ClassVar[int]
    CURRENCY_FIELD_NUMBER: _ClassVar[int]
    LEAD_TIME_DAYS_FIELD_NUMBER: _ClassVar[int]
    EVIDENCE_ID_FIELD_NUMBER: _ClassVar[int]
    CATALOG_VERSION_FIELD_NUMBER: _ClassVar[int]
    CATALOG_CHECKSUM_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    PROJECT_ID_FIELD_NUMBER: _ClassVar[int]
    CANDIDATE_ID_FIELD_NUMBER: _ClassVar[int]
    CANDIDATE_INDEX_FIELD_NUMBER: _ClassVar[int]
    CANONICAL_SMILES_FIELD_NUMBER: _ClassVar[int]
    smiles: str
    available: bool
    catalog_id: str
    catalog_source: str
    source_timestamp: str
    price: float
    currency: str
    lead_time_days: int
    evidence_id: str
    catalog_version: str
    catalog_checksum: str
    request_id: str
    project_id: str
    candidate_id: str
    candidate_index: int
    canonical_smiles: str
    def __init__(self, smiles: _Optional[str] = ..., available: bool = ..., catalog_id: _Optional[str] = ..., catalog_source: _Optional[str] = ..., source_timestamp: _Optional[str] = ..., price: _Optional[float] = ..., currency: _Optional[str] = ..., lead_time_days: _Optional[int] = ..., evidence_id: _Optional[str] = ..., catalog_version: _Optional[str] = ..., catalog_checksum: _Optional[str] = ..., request_id: _Optional[str] = ..., project_id: _Optional[str] = ..., candidate_id: _Optional[str] = ..., candidate_index: _Optional[int] = ..., canonical_smiles: _Optional[str] = ...) -> None: ...

class BatchAvailabilityRequest(_message.Message):
    __slots__ = ("requests", "request_id", "project_id", "candidate_id", "candidate_index", "canonical_smiles")
    REQUESTS_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    PROJECT_ID_FIELD_NUMBER: _ClassVar[int]
    CANDIDATE_ID_FIELD_NUMBER: _ClassVar[int]
    CANDIDATE_INDEX_FIELD_NUMBER: _ClassVar[int]
    CANONICAL_SMILES_FIELD_NUMBER: _ClassVar[int]
    requests: _containers.RepeatedCompositeFieldContainer[AvailabilityRequest]
    request_id: str
    project_id: str
    candidate_id: str
    candidate_index: int
    canonical_smiles: str
    def __init__(self, requests: _Optional[_Iterable[_Union[AvailabilityRequest, _Mapping]]] = ..., request_id: _Optional[str] = ..., project_id: _Optional[str] = ..., candidate_id: _Optional[str] = ..., candidate_index: _Optional[int] = ..., canonical_smiles: _Optional[str] = ...) -> None: ...

class BatchAvailabilityResponse(_message.Message):
    __slots__ = ("results", "total_elapsed_ms", "request_id", "project_id", "candidate_id", "candidate_index", "canonical_smiles")
    RESULTS_FIELD_NUMBER: _ClassVar[int]
    TOTAL_ELAPSED_MS_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    PROJECT_ID_FIELD_NUMBER: _ClassVar[int]
    CANDIDATE_ID_FIELD_NUMBER: _ClassVar[int]
    CANDIDATE_INDEX_FIELD_NUMBER: _ClassVar[int]
    CANONICAL_SMILES_FIELD_NUMBER: _ClassVar[int]
    results: _containers.RepeatedCompositeFieldContainer[AvailabilityResponse]
    total_elapsed_ms: int
    request_id: str
    project_id: str
    candidate_id: str
    candidate_index: int
    canonical_smiles: str
    def __init__(self, results: _Optional[_Iterable[_Union[AvailabilityResponse, _Mapping]]] = ..., total_elapsed_ms: _Optional[int] = ..., request_id: _Optional[str] = ..., project_id: _Optional[str] = ..., candidate_id: _Optional[str] = ..., candidate_index: _Optional[int] = ..., canonical_smiles: _Optional[str] = ...) -> None: ...

class CatalogPriceRequest(_message.Message):
    __slots__ = ("smiles", "catalog_id", "request_id", "project_id", "candidate_id", "candidate_index", "canonical_smiles")
    SMILES_FIELD_NUMBER: _ClassVar[int]
    CATALOG_ID_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    PROJECT_ID_FIELD_NUMBER: _ClassVar[int]
    CANDIDATE_ID_FIELD_NUMBER: _ClassVar[int]
    CANDIDATE_INDEX_FIELD_NUMBER: _ClassVar[int]
    CANONICAL_SMILES_FIELD_NUMBER: _ClassVar[int]
    smiles: str
    catalog_id: str
    request_id: str
    project_id: str
    candidate_id: str
    candidate_index: int
    canonical_smiles: str
    def __init__(self, smiles: _Optional[str] = ..., catalog_id: _Optional[str] = ..., request_id: _Optional[str] = ..., project_id: _Optional[str] = ..., candidate_id: _Optional[str] = ..., candidate_index: _Optional[int] = ..., canonical_smiles: _Optional[str] = ...) -> None: ...
