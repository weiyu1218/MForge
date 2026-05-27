from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class RetrosynthesisRequest(_message.Message):
    __slots__ = ("project_id", "molecule_smiles", "max_routes", "max_depth", "engine", "include_building_blocks", "price_threshold_usd", "engine_params")
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
    project_id: str
    molecule_smiles: str
    max_routes: int
    max_depth: int
    engine: str
    include_building_blocks: bool
    price_threshold_usd: float
    engine_params: _containers.ScalarMap[str, str]
    def __init__(self, project_id: _Optional[str] = ..., molecule_smiles: _Optional[str] = ..., max_routes: _Optional[int] = ..., max_depth: _Optional[int] = ..., engine: _Optional[str] = ..., include_building_blocks: bool = ..., price_threshold_usd: _Optional[float] = ..., engine_params: _Optional[_Mapping[str, str]] = ...) -> None: ...

class RetrosynthesisResponse(_message.Message):
    __slots__ = ("request_id", "routes", "total_routes_found", "elapsed_ms")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    ROUTES_FIELD_NUMBER: _ClassVar[int]
    TOTAL_ROUTES_FOUND_FIELD_NUMBER: _ClassVar[int]
    ELAPSED_MS_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    routes: _containers.RepeatedCompositeFieldContainer[SyntheticRoute]
    total_routes_found: int
    elapsed_ms: int
    def __init__(self, request_id: _Optional[str] = ..., routes: _Optional[_Iterable[_Union[SyntheticRoute, _Mapping]]] = ..., total_routes_found: _Optional[int] = ..., elapsed_ms: _Optional[int] = ...) -> None: ...

class SyntheticRoute(_message.Message):
    __slots__ = ("route_id", "reaction_smiles", "predicted_score", "predicted_yield", "n_steps", "building_blocks", "estimated_cost_usd_per_g", "all_commercially_available")
    ROUTE_ID_FIELD_NUMBER: _ClassVar[int]
    REACTION_SMILES_FIELD_NUMBER: _ClassVar[int]
    PREDICTED_SCORE_FIELD_NUMBER: _ClassVar[int]
    PREDICTED_YIELD_FIELD_NUMBER: _ClassVar[int]
    N_STEPS_FIELD_NUMBER: _ClassVar[int]
    BUILDING_BLOCKS_FIELD_NUMBER: _ClassVar[int]
    ESTIMATED_COST_USD_PER_G_FIELD_NUMBER: _ClassVar[int]
    ALL_COMMERCIALLY_AVAILABLE_FIELD_NUMBER: _ClassVar[int]
    route_id: str
    reaction_smiles: _containers.RepeatedScalarFieldContainer[str]
    predicted_score: float
    predicted_yield: float
    n_steps: int
    building_blocks: _containers.RepeatedScalarFieldContainer[str]
    estimated_cost_usd_per_g: float
    all_commercially_available: bool
    def __init__(self, route_id: _Optional[str] = ..., reaction_smiles: _Optional[_Iterable[str]] = ..., predicted_score: _Optional[float] = ..., predicted_yield: _Optional[float] = ..., n_steps: _Optional[int] = ..., building_blocks: _Optional[_Iterable[str]] = ..., estimated_cost_usd_per_g: _Optional[float] = ..., all_commercially_available: bool = ...) -> None: ...
