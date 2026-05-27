from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class HCIV(_message.Message):
    __slots__ = ("coordinates", "curvature", "molecule_smiles", "parent_hciv_id")
    COORDINATES_FIELD_NUMBER: _ClassVar[int]
    CURVATURE_FIELD_NUMBER: _ClassVar[int]
    MOLECULE_SMILES_FIELD_NUMBER: _ClassVar[int]
    PARENT_HCIV_ID_FIELD_NUMBER: _ClassVar[int]
    coordinates: _containers.RepeatedScalarFieldContainer[float]
    curvature: float
    molecule_smiles: str
    parent_hciv_id: str
    def __init__(self, coordinates: _Optional[_Iterable[float]] = ..., curvature: _Optional[float] = ..., molecule_smiles: _Optional[str] = ..., parent_hciv_id: _Optional[str] = ...) -> None: ...

class IntentCone(_message.Message):
    __slots__ = ("axis", "half_angle", "curvature", "property_weights")
    class PropertyWeightsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: float
        def __init__(self, key: _Optional[str] = ..., value: _Optional[float] = ...) -> None: ...
    AXIS_FIELD_NUMBER: _ClassVar[int]
    HALF_ANGLE_FIELD_NUMBER: _ClassVar[int]
    CURVATURE_FIELD_NUMBER: _ClassVar[int]
    PROPERTY_WEIGHTS_FIELD_NUMBER: _ClassVar[int]
    axis: _containers.RepeatedScalarFieldContainer[float]
    half_angle: float
    curvature: float
    property_weights: _containers.ScalarMap[str, float]
    def __init__(self, axis: _Optional[_Iterable[float]] = ..., half_angle: _Optional[float] = ..., curvature: _Optional[float] = ..., property_weights: _Optional[_Mapping[str, float]] = ...) -> None: ...

class HCIVBatch(_message.Message):
    __slots__ = ("embeddings", "batch_id")
    EMBEDDINGS_FIELD_NUMBER: _ClassVar[int]
    BATCH_ID_FIELD_NUMBER: _ClassVar[int]
    embeddings: _containers.RepeatedCompositeFieldContainer[HCIV]
    batch_id: str
    def __init__(self, embeddings: _Optional[_Iterable[_Union[HCIV, _Mapping]]] = ..., batch_id: _Optional[str] = ...) -> None: ...
