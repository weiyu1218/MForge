from mf_core.proto_gen.moleculeforge.v1.core import audit_pb2 as _audit_pb2
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

class OracleOutcome(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    ORACLE_OUTCOME_UNSPECIFIED: _ClassVar[OracleOutcome]
    ORACLE_OUTCOME_PASS: _ClassVar[OracleOutcome]
    ORACLE_OUTCOME_FAIL: _ClassVar[OracleOutcome]
    ORACLE_OUTCOME_SKIPPED: _ClassVar[OracleOutcome]
    ORACLE_OUTCOME_ERROR: _ClassVar[OracleOutcome]
ORACLE_LEVEL_UNSPECIFIED: OracleLevel
L0_RDKIT: OracleLevel
L1_ML_SURROGATE: OracleLevel
L2_DOCKING: OracleLevel
L3_FEP: OracleLevel
L4_WETLAB: OracleLevel
ORACLE_OUTCOME_UNSPECIFIED: OracleOutcome
ORACLE_OUTCOME_PASS: OracleOutcome
ORACLE_OUTCOME_FAIL: OracleOutcome
ORACLE_OUTCOME_SKIPPED: OracleOutcome
ORACLE_OUTCOME_ERROR: OracleOutcome

class OracleMetric(_message.Message):
    __slots__ = ("property", "value", "unit", "uncertainty")
    PROPERTY_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    UNIT_FIELD_NUMBER: _ClassVar[int]
    UNCERTAINTY_FIELD_NUMBER: _ClassVar[int]
    property: str
    value: float
    unit: str
    uncertainty: float
    def __init__(self, property: _Optional[str] = ..., value: _Optional[float] = ..., unit: _Optional[str] = ..., uncertainty: _Optional[float] = ...) -> None: ...

class OracleEvaluation(_message.Message):
    __slots__ = ("oracle_name", "molecule_smiles", "level", "scores", "uncertainties", "elapsed_ms", "success", "error_message", "outcome", "oracle_version", "model_version", "artifact_refs", "evidence_id", "metrics", "error_code")
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
    OUTCOME_FIELD_NUMBER: _ClassVar[int]
    ORACLE_VERSION_FIELD_NUMBER: _ClassVar[int]
    MODEL_VERSION_FIELD_NUMBER: _ClassVar[int]
    ARTIFACT_REFS_FIELD_NUMBER: _ClassVar[int]
    EVIDENCE_ID_FIELD_NUMBER: _ClassVar[int]
    METRICS_FIELD_NUMBER: _ClassVar[int]
    ERROR_CODE_FIELD_NUMBER: _ClassVar[int]
    oracle_name: str
    molecule_smiles: str
    level: OracleLevel
    scores: _containers.ScalarMap[str, float]
    uncertainties: _containers.ScalarMap[str, float]
    elapsed_ms: int
    success: bool
    error_message: str
    outcome: OracleOutcome
    oracle_version: str
    model_version: str
    artifact_refs: _containers.RepeatedCompositeFieldContainer[_audit_pb2.ArtifactRef]
    evidence_id: str
    metrics: _containers.RepeatedCompositeFieldContainer[OracleMetric]
    error_code: str
    def __init__(self, oracle_name: _Optional[str] = ..., molecule_smiles: _Optional[str] = ..., level: _Optional[_Union[OracleLevel, str]] = ..., scores: _Optional[_Mapping[str, float]] = ..., uncertainties: _Optional[_Mapping[str, float]] = ..., elapsed_ms: _Optional[int] = ..., success: bool = ..., error_message: _Optional[str] = ..., outcome: _Optional[_Union[OracleOutcome, str]] = ..., oracle_version: _Optional[str] = ..., model_version: _Optional[str] = ..., artifact_refs: _Optional[_Iterable[_Union[_audit_pb2.ArtifactRef, _Mapping]]] = ..., evidence_id: _Optional[str] = ..., metrics: _Optional[_Iterable[_Union[OracleMetric, _Mapping]]] = ..., error_code: _Optional[str] = ...) -> None: ...

class OracleBatchRequest(_message.Message):
    __slots__ = ("project_id", "molecule_smiles", "level", "requested_properties", "return_uncertainty", "receptor_uri", "protein_pdb_id", "reference_ligand_smiles", "oracle_parameters", "request_id")
    class OracleParametersEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    PROJECT_ID_FIELD_NUMBER: _ClassVar[int]
    MOLECULE_SMILES_FIELD_NUMBER: _ClassVar[int]
    LEVEL_FIELD_NUMBER: _ClassVar[int]
    REQUESTED_PROPERTIES_FIELD_NUMBER: _ClassVar[int]
    RETURN_UNCERTAINTY_FIELD_NUMBER: _ClassVar[int]
    RECEPTOR_URI_FIELD_NUMBER: _ClassVar[int]
    PROTEIN_PDB_ID_FIELD_NUMBER: _ClassVar[int]
    REFERENCE_LIGAND_SMILES_FIELD_NUMBER: _ClassVar[int]
    ORACLE_PARAMETERS_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    project_id: str
    molecule_smiles: _containers.RepeatedScalarFieldContainer[str]
    level: OracleLevel
    requested_properties: _containers.RepeatedScalarFieldContainer[str]
    return_uncertainty: bool
    receptor_uri: str
    protein_pdb_id: str
    reference_ligand_smiles: str
    oracle_parameters: _containers.ScalarMap[str, str]
    request_id: str
    def __init__(self, project_id: _Optional[str] = ..., molecule_smiles: _Optional[_Iterable[str]] = ..., level: _Optional[_Union[OracleLevel, str]] = ..., requested_properties: _Optional[_Iterable[str]] = ..., return_uncertainty: bool = ..., receptor_uri: _Optional[str] = ..., protein_pdb_id: _Optional[str] = ..., reference_ligand_smiles: _Optional[str] = ..., oracle_parameters: _Optional[_Mapping[str, str]] = ..., request_id: _Optional[str] = ...) -> None: ...

class OracleBatchResponse(_message.Message):
    __slots__ = ("evaluations", "batch_id", "total_elapsed_ms")
    EVALUATIONS_FIELD_NUMBER: _ClassVar[int]
    BATCH_ID_FIELD_NUMBER: _ClassVar[int]
    TOTAL_ELAPSED_MS_FIELD_NUMBER: _ClassVar[int]
    evaluations: _containers.RepeatedCompositeFieldContainer[OracleEvaluation]
    batch_id: str
    total_elapsed_ms: int
    def __init__(self, evaluations: _Optional[_Iterable[_Union[OracleEvaluation, _Mapping]]] = ..., batch_id: _Optional[str] = ..., total_elapsed_ms: _Optional[int] = ...) -> None: ...
