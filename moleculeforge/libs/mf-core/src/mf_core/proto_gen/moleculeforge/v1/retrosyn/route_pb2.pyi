from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class RouteEncoding(_message.Message):
    __slots__ = ("route_id", "humu_embedding", "curvature", "path_length")
    ROUTE_ID_FIELD_NUMBER: _ClassVar[int]
    HUMU_EMBEDDING_FIELD_NUMBER: _ClassVar[int]
    CURVATURE_FIELD_NUMBER: _ClassVar[int]
    PATH_LENGTH_FIELD_NUMBER: _ClassVar[int]
    route_id: str
    humu_embedding: bytes
    curvature: float
    path_length: int
    def __init__(self, route_id: _Optional[str] = ..., humu_embedding: _Optional[bytes] = ..., curvature: _Optional[float] = ..., path_length: _Optional[int] = ...) -> None: ...

class RouteComparisonResult(_message.Message):
    __slots__ = ("route_a_id", "route_b_id", "geodesic_distance", "similarity", "distance_breakdown")
    ROUTE_A_ID_FIELD_NUMBER: _ClassVar[int]
    ROUTE_B_ID_FIELD_NUMBER: _ClassVar[int]
    GEODESIC_DISTANCE_FIELD_NUMBER: _ClassVar[int]
    SIMILARITY_FIELD_NUMBER: _ClassVar[int]
    DISTANCE_BREAKDOWN_FIELD_NUMBER: _ClassVar[int]
    route_a_id: str
    route_b_id: str
    geodesic_distance: float
    similarity: float
    distance_breakdown: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, route_a_id: _Optional[str] = ..., route_b_id: _Optional[str] = ..., geodesic_distance: _Optional[float] = ..., similarity: _Optional[float] = ..., distance_breakdown: _Optional[_Iterable[float]] = ...) -> None: ...

class RouteCluster(_message.Message):
    __slots__ = ("cluster_id", "route_ids", "centroid_embedding", "intra_cluster_distance")
    CLUSTER_ID_FIELD_NUMBER: _ClassVar[int]
    ROUTE_IDS_FIELD_NUMBER: _ClassVar[int]
    CENTROID_EMBEDDING_FIELD_NUMBER: _ClassVar[int]
    INTRA_CLUSTER_DISTANCE_FIELD_NUMBER: _ClassVar[int]
    cluster_id: str
    route_ids: _containers.RepeatedScalarFieldContainer[str]
    centroid_embedding: bytes
    intra_cluster_distance: float
    def __init__(self, cluster_id: _Optional[str] = ..., route_ids: _Optional[_Iterable[str]] = ..., centroid_embedding: _Optional[bytes] = ..., intra_cluster_distance: _Optional[float] = ...) -> None: ...
