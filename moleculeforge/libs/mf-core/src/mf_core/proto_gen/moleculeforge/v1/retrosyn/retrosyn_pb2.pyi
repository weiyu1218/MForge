from google.protobuf import struct_pb2 as _struct_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class RetrosynthesisRequest(_message.Message):
    __slots__ = ("project_id", "molecule_smiles", "max_routes", "max_depth", "engine", "include_building_blocks", "price_threshold_usd", "engine_params", "request_id", "candidate_id", "candidate_index", "canonical_smiles", "run_id")
    class EngineParamsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    PROJECT_ID_FIELD_NUMBER: _ClassVar[int]
    MOLECULE_SMILES_FIELD_NUMBER: _ClassVar[int]
    MAX_ROUTES_FIELD_NUMBER: _ClassVar[int]
    MAX_DEPTH_FIELD_NUMBER: _ClassVar[int]
    ENGINE_FIELD_NUMBER: _ClassVar[int]
    INCLUDE_BUILDING_BLOCKS_FIELD_NUMBER: _ClassVar[int]
    PRICE_THRESHOLD_USD_FIELD_NUMBER: _ClassVar[int]
    ENGINE_PARAMS_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    CANDIDATE_ID_FIELD_NUMBER: _ClassVar[int]
    CANDIDATE_INDEX_FIELD_NUMBER: _ClassVar[int]
    CANONICAL_SMILES_FIELD_NUMBER: _ClassVar[int]
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    project_id: str
    molecule_smiles: str
    max_routes: int
    max_depth: int
    engine: str
    include_building_blocks: bool
    price_threshold_usd: float
    engine_params: _containers.ScalarMap[str, str]
    request_id: str
    candidate_id: str
    candidate_index: int
    canonical_smiles: str
    run_id: str
    def __init__(self, project_id: _Optional[str] = ..., molecule_smiles: _Optional[str] = ..., max_routes: _Optional[int] = ..., max_depth: _Optional[int] = ..., engine: _Optional[str] = ..., include_building_blocks: bool = ..., price_threshold_usd: _Optional[float] = ..., engine_params: _Optional[_Mapping[str, str]] = ..., request_id: _Optional[str] = ..., candidate_id: _Optional[str] = ..., candidate_index: _Optional[int] = ..., canonical_smiles: _Optional[str] = ..., run_id: _Optional[str] = ...) -> None: ...

class RetrosynthesisResponse(_message.Message):
    __slots__ = ("request_id", "routes", "total_routes_found", "elapsed_ms", "project_id", "candidate_id", "candidate_index", "canonical_smiles", "run_id", "assessments")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    ROUTES_FIELD_NUMBER: _ClassVar[int]
    TOTAL_ROUTES_FOUND_FIELD_NUMBER: _ClassVar[int]
    ELAPSED_MS_FIELD_NUMBER: _ClassVar[int]
    PROJECT_ID_FIELD_NUMBER: _ClassVar[int]
    CANDIDATE_ID_FIELD_NUMBER: _ClassVar[int]
    CANDIDATE_INDEX_FIELD_NUMBER: _ClassVar[int]
    CANONICAL_SMILES_FIELD_NUMBER: _ClassVar[int]
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    ASSESSMENTS_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    routes: _containers.RepeatedCompositeFieldContainer[SyntheticRoute]
    total_routes_found: int
    elapsed_ms: int
    project_id: str
    candidate_id: str
    candidate_index: int
    canonical_smiles: str
    run_id: str
    assessments: _containers.RepeatedCompositeFieldContainer[RetrosynthesisAssessment]
    def __init__(self, request_id: _Optional[str] = ..., routes: _Optional[_Iterable[_Union[SyntheticRoute, _Mapping]]] = ..., total_routes_found: _Optional[int] = ..., elapsed_ms: _Optional[int] = ..., project_id: _Optional[str] = ..., candidate_id: _Optional[str] = ..., candidate_index: _Optional[int] = ..., canonical_smiles: _Optional[str] = ..., run_id: _Optional[str] = ..., assessments: _Optional[_Iterable[_Union[RetrosynthesisAssessment, _Mapping]]] = ...) -> None: ...

class SyntheticRouteStep(_message.Message):
    __slots__ = ("step_id", "reaction", "reaction_type", "reactants", "conditions", "reagents", "purification", "operation", "building_blocks", "yield_fraction")
    STEP_ID_FIELD_NUMBER: _ClassVar[int]
    REACTION_FIELD_NUMBER: _ClassVar[int]
    REACTION_TYPE_FIELD_NUMBER: _ClassVar[int]
    REACTANTS_FIELD_NUMBER: _ClassVar[int]
    CONDITIONS_FIELD_NUMBER: _ClassVar[int]
    REAGENTS_FIELD_NUMBER: _ClassVar[int]
    PURIFICATION_FIELD_NUMBER: _ClassVar[int]
    OPERATION_FIELD_NUMBER: _ClassVar[int]
    BUILDING_BLOCKS_FIELD_NUMBER: _ClassVar[int]
    YIELD_FRACTION_FIELD_NUMBER: _ClassVar[int]
    step_id: str
    reaction: str
    reaction_type: str
    reactants: _containers.RepeatedCompositeFieldContainer[_struct_pb2.Struct]
    conditions: _struct_pb2.Struct
    reagents: _containers.RepeatedScalarFieldContainer[str]
    purification: str
    operation: str
    building_blocks: _containers.RepeatedCompositeFieldContainer[_struct_pb2.Struct]
    yield_fraction: float
    def __init__(self, step_id: _Optional[str] = ..., reaction: _Optional[str] = ..., reaction_type: _Optional[str] = ..., reactants: _Optional[_Iterable[_Union[_struct_pb2.Struct, _Mapping]]] = ..., conditions: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ..., reagents: _Optional[_Iterable[str]] = ..., purification: _Optional[str] = ..., operation: _Optional[str] = ..., building_blocks: _Optional[_Iterable[_Union[_struct_pb2.Struct, _Mapping]]] = ..., yield_fraction: _Optional[float] = ...) -> None: ...

class SyntheticRoute(_message.Message):
    __slots__ = ("route_id", "reaction_smiles", "predicted_score", "predicted_yield", "n_steps", "building_blocks", "estimated_cost_usd_per_g", "all_commercially_available", "steps", "source_engine", "route_type", "building_block_records")
    ROUTE_ID_FIELD_NUMBER: _ClassVar[int]
    REACTION_SMILES_FIELD_NUMBER: _ClassVar[int]
    PREDICTED_SCORE_FIELD_NUMBER: _ClassVar[int]
    PREDICTED_YIELD_FIELD_NUMBER: _ClassVar[int]
    N_STEPS_FIELD_NUMBER: _ClassVar[int]
    BUILDING_BLOCKS_FIELD_NUMBER: _ClassVar[int]
    ESTIMATED_COST_USD_PER_G_FIELD_NUMBER: _ClassVar[int]
    ALL_COMMERCIALLY_AVAILABLE_FIELD_NUMBER: _ClassVar[int]
    STEPS_FIELD_NUMBER: _ClassVar[int]
    SOURCE_ENGINE_FIELD_NUMBER: _ClassVar[int]
    ROUTE_TYPE_FIELD_NUMBER: _ClassVar[int]
    BUILDING_BLOCK_RECORDS_FIELD_NUMBER: _ClassVar[int]
    route_id: str
    reaction_smiles: _containers.RepeatedScalarFieldContainer[str]
    predicted_score: float
    predicted_yield: float
    n_steps: int
    building_blocks: _containers.RepeatedScalarFieldContainer[str]
    estimated_cost_usd_per_g: float
    all_commercially_available: bool
    steps: _containers.RepeatedCompositeFieldContainer[SyntheticRouteStep]
    source_engine: str
    route_type: str
    building_block_records: _containers.RepeatedCompositeFieldContainer[_struct_pb2.Struct]
    def __init__(self, route_id: _Optional[str] = ..., reaction_smiles: _Optional[_Iterable[str]] = ..., predicted_score: _Optional[float] = ..., predicted_yield: _Optional[float] = ..., n_steps: _Optional[int] = ..., building_blocks: _Optional[_Iterable[str]] = ..., estimated_cost_usd_per_g: _Optional[float] = ..., all_commercially_available: bool = ..., steps: _Optional[_Iterable[_Union[SyntheticRouteStep, _Mapping]]] = ..., source_engine: _Optional[str] = ..., route_type: _Optional[str] = ..., building_block_records: _Optional[_Iterable[_Union[_struct_pb2.Struct, _Mapping]]] = ...) -> None: ...

class RetrosynthesisAssessment(_message.Message):
    __slots__ = ("assessment_id", "assessment_type", "source_engine", "score", "details")
    ASSESSMENT_ID_FIELD_NUMBER: _ClassVar[int]
    ASSESSMENT_TYPE_FIELD_NUMBER: _ClassVar[int]
    SOURCE_ENGINE_FIELD_NUMBER: _ClassVar[int]
    SCORE_FIELD_NUMBER: _ClassVar[int]
    DETAILS_FIELD_NUMBER: _ClassVar[int]
    assessment_id: str
    assessment_type: str
    source_engine: str
    score: float
    details: _struct_pb2.Struct
    def __init__(self, assessment_id: _Optional[str] = ..., assessment_type: _Optional[str] = ..., source_engine: _Optional[str] = ..., score: _Optional[float] = ..., details: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ...) -> None: ...
