from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class OracleLevel(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    ORACLE_LEVEL_UNSPECIFIED: _ClassVar[OracleLevel]
    L0_RDKIT: _ClassVar[OracleLevel]
    L1_ML_SURROGATE: _ClassVar[OracleLevel]
    L2_DOCKING: _ClassVar[OracleLevel]
    L3_FEP: _ClassVar[OracleLevel]
    L4_WETLAB: _ClassVar[OracleLevel]
ORACLE_LEVEL_UNSPECIFIED: OracleLevel
L0_RDKIT: OracleLevel
L1_ML_SURROGATE: OracleLevel
L2_DOCKING: OracleLevel
L3_FEP: OracleLevel
L4_WETLAB: OracleLevel

class OracleEvaluation(_message.Message):
    __slots__ = ("oracle_name", "molecule_smiles", "level", "scores", "uncertainties", "elapsed_ms", "success", "error_message")
    class ScoresEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: float
        def __init__(self, key: _Optional[str] = ..., value: _Optional[float] = ...) -> None: ...
    class UncertaintiesEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: float
        def __init__(self, key: _Optional[str] = ..., value: _Optional[float] = ...) -> None: ...
    ORACLE_NAME_FIELD_NUMBER: _ClassVar[int]
    MOLECULE_SMILES_FIELD_NUMBER: _ClassVar[int]
    LEVEL_FIELD_NUMBER: _ClassVar[int]
    SCORES_FIELD_NUMBER: _ClassVar[int]
    UNCERTAINTIES_FIELD_NUMBER: _ClassVar[int]
    ELAPSED_MS_FIELD_NUMBER: _ClassVar[int]
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    ERROR_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    oracle_name: str
    molecule_smiles: str
    level: OracleLevel
    scores: _containers.ScalarMap[str, float]
    uncertainties: _containers.ScalarMap[str, float]
    elapsed_ms: int
    success: bool
    error_message: str
    def __init__(self, oracle_name: _Optional[str] = ..., molecule_smiles: _Optional[str] = ..., level: _Optional[_Union[OracleLevel, str]] = ..., scores: _Optional[_Mapping[str, float]] = ..., uncertainties: _Optional[_Mapping[str, float]] = ..., elapsed_ms: _Optional[int] = ..., success: bool = ..., error_message: _Optional[str] = ...) -> None: ...

class OracleBatchRequest(_message.Message):
    __slots__ = ("project_id", "molecule_smiles", "level", "requested_properties", "return_uncertainty")
    PROJECT_ID_FIELD_NUMBER: _ClassVar[int]
    MOLECULE_SMILES_FIELD_NUMBER: _ClassVar[int]
    LEVEL_FIELD_NUMBER: _ClassVar[int]
    REQUESTED_PROPERTIES_FIELD_NUMBER: _ClassVar[int]
    RETURN_UNCERTAINTY_FIELD_NUMBER: _ClassVar[int]
    project_id: str
    molecule_smiles: _containers.RepeatedScalarFieldContainer[str]
    level: OracleLevel
    requested_properties: _containers.RepeatedScalarFieldContainer[str]
    return_uncertainty: bool
    def __init__(self, project_id: _Optional[str] = ..., molecule_smiles: _Optional[_Iterable[str]] = ..., level: _Optional[_Union[OracleLevel, str]] = ..., requested_properties: _Optional[_Iterable[str]] = ..., return_uncertainty: bool = ...) -> None: ...

class OracleBatchResponse(_message.Message):
    __slots__ = ("evaluations", "batch_id", "total_elapsed_ms")
    EVALUATIONS_FIELD_NUMBER: _ClassVar[int]
    BATCH_ID_FIELD_NUMBER: _ClassVar[int]
    TOTAL_ELAPSED_MS_FIELD_NUMBER: _ClassVar[int]
    evaluations: _containers.RepeatedCompositeFieldContainer[OracleEvaluation]
    batch_id: str
    total_elapsed_ms: int
    def __init__(self, evaluations: _Optional[_Iterable[_Union[OracleEvaluation, _Mapping]]] = ..., batch_id: _Optional[str] = ..., total_elapsed_ms: _Optional[int] = ...) -> None: ...
