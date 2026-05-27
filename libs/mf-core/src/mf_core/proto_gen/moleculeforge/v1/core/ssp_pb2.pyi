from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class SynthesisStep(_message.Message):
    __slots__ = ("step_number", "operation", "parameters", "reagents", "solvent", "temperature_c", "duration_min", "atmosphere")
    class ParametersEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    STEP_NUMBER_FIELD_NUMBER: _ClassVar[int]
    OPERATION_FIELD_NUMBER: _ClassVar[int]
    PARAMETERS_FIELD_NUMBER: _ClassVar[int]
    REAGENTS_FIELD_NUMBER: _ClassVar[int]
    SOLVENT_FIELD_NUMBER: _ClassVar[int]
    TEMPERATURE_C_FIELD_NUMBER: _ClassVar[int]
    DURATION_MIN_FIELD_NUMBER: _ClassVar[int]
    ATMOSPHERE_FIELD_NUMBER: _ClassVar[int]
    step_number: int
    operation: str
    parameters: _containers.ScalarMap[str, str]
    reagents: _containers.RepeatedScalarFieldContainer[str]
    solvent: str
    temperature_c: float
    duration_min: int
    atmosphere: str
    def __init__(self, step_number: _Optional[int] = ..., operation: _Optional[str] = ..., parameters: _Optional[_Mapping[str, str]] = ..., reagents: _Optional[_Iterable[str]] = ..., solvent: _Optional[str] = ..., temperature_c: _Optional[float] = ..., duration_min: _Optional[int] = ..., atmosphere: _Optional[str] = ...) -> None: ...

class SSP(_message.Message):
    __slots__ = ("ssp_id", "molecule_smiles", "steps", "generated_by", "predicted_yield", "xdl_output", "metadata")
    class MetadataEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    SSP_ID_FIELD_NUMBER: _ClassVar[int]
    MOLECULE_SMILES_FIELD_NUMBER: _ClassVar[int]
    STEPS_FIELD_NUMBER: _ClassVar[int]
    GENERATED_BY_FIELD_NUMBER: _ClassVar[int]
    PREDICTED_YIELD_FIELD_NUMBER: _ClassVar[int]
    XDL_OUTPUT_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    ssp_id: str
    molecule_smiles: str
    steps: _containers.RepeatedCompositeFieldContainer[SynthesisStep]
    generated_by: str
    predicted_yield: float
    xdl_output: str
    metadata: _containers.ScalarMap[str, str]
    def __init__(self, ssp_id: _Optional[str] = ..., molecule_smiles: _Optional[str] = ..., steps: _Optional[_Iterable[_Union[SynthesisStep, _Mapping]]] = ..., generated_by: _Optional[str] = ..., predicted_yield: _Optional[float] = ..., xdl_output: _Optional[str] = ..., metadata: _Optional[_Mapping[str, str]] = ...) -> None: ...
