from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class FEPResult(_message.Message):
    __slots__ = ("ligand_a_smiles", "ligand_b_smiles", "ddg_kcal_mol", "ddg_uncertainty", "n_repeats", "method", "per_repeat_ddg", "converged")
    class PerRepeatDdgEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: float
        def __init__(self, key: _Optional[str] = ..., value: _Optional[float] = ...) -> None: ...
    LIGAND_A_SMILES_FIELD_NUMBER: _ClassVar[int]
    LIGAND_B_SMILES_FIELD_NUMBER: _ClassVar[int]
    DDG_KCAL_MOL_FIELD_NUMBER: _ClassVar[int]
    DDG_UNCERTAINTY_FIELD_NUMBER: _ClassVar[int]
    N_REPEATS_FIELD_NUMBER: _ClassVar[int]
    METHOD_FIELD_NUMBER: _ClassVar[int]
    PER_REPEAT_DDG_FIELD_NUMBER: _ClassVar[int]
    CONVERGED_FIELD_NUMBER: _ClassVar[int]
    ligand_a_smiles: str
    ligand_b_smiles: str
    ddg_kcal_mol: float
    ddg_uncertainty: float
    n_repeats: int
    method: str
    per_repeat_ddg: _containers.ScalarMap[str, float]
    converged: bool
    def __init__(self, ligand_a_smiles: _Optional[str] = ..., ligand_b_smiles: _Optional[str] = ..., ddg_kcal_mol: _Optional[float] = ..., ddg_uncertainty: _Optional[float] = ..., n_repeats: _Optional[int] = ..., method: _Optional[str] = ..., per_repeat_ddg: _Optional[_Mapping[str, float]] = ..., converged: bool = ...) -> None: ...

class FEPBatchRequest(_message.Message):
    __slots__ = ("project_id", "protein_pdb_id", "reference_ligand_smiles", "test_ligand_smiles", "method", "n_repeats")
    PROJECT_ID_FIELD_NUMBER: _ClassVar[int]
    PROTEIN_PDB_ID_FIELD_NUMBER: _ClassVar[int]
    REFERENCE_LIGAND_SMILES_FIELD_NUMBER: _ClassVar[int]
    TEST_LIGAND_SMILES_FIELD_NUMBER: _ClassVar[int]
    METHOD_FIELD_NUMBER: _ClassVar[int]
    N_REPEATS_FIELD_NUMBER: _ClassVar[int]
    project_id: str
    protein_pdb_id: str
    reference_ligand_smiles: str
    test_ligand_smiles: _containers.RepeatedScalarFieldContainer[str]
    method: str
    n_repeats: int
    def __init__(self, project_id: _Optional[str] = ..., protein_pdb_id: _Optional[str] = ..., reference_ligand_smiles: _Optional[str] = ..., test_ligand_smiles: _Optional[_Iterable[str]] = ..., method: _Optional[str] = ..., n_repeats: _Optional[int] = ...) -> None: ...

class FEPBatchResponse(_message.Message):
    __slots__ = ("results", "batch_id", "total_elapsed_ms")
    RESULTS_FIELD_NUMBER: _ClassVar[int]
    BATCH_ID_FIELD_NUMBER: _ClassVar[int]
    TOTAL_ELAPSED_MS_FIELD_NUMBER: _ClassVar[int]
    results: _containers.RepeatedCompositeFieldContainer[FEPResult]
    batch_id: str
    total_elapsed_ms: int
    def __init__(self, results: _Optional[_Iterable[_Union[FEPResult, _Mapping]]] = ..., batch_id: _Optional[str] = ..., total_elapsed_ms: _Optional[int] = ...) -> None: ...

class FEPJobStatusRequest(_message.Message):
    __slots__ = ("job_id",)
    JOB_ID_FIELD_NUMBER: _ClassVar[int]
    job_id: str
    def __init__(self, job_id: _Optional[str] = ...) -> None: ...

class FEPJobStatus(_message.Message):
    __slots__ = ("job_id", "state", "response", "error", "submitted_at_ms", "started_at_ms", "completed_at_ms")
    JOB_ID_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    RESPONSE_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    SUBMITTED_AT_MS_FIELD_NUMBER: _ClassVar[int]
    STARTED_AT_MS_FIELD_NUMBER: _ClassVar[int]
    COMPLETED_AT_MS_FIELD_NUMBER: _ClassVar[int]
    job_id: str
    state: str
    response: FEPBatchResponse
    error: str
    submitted_at_ms: int
    started_at_ms: int
    completed_at_ms: int
    def __init__(self, job_id: _Optional[str] = ..., state: _Optional[str] = ..., response: _Optional[_Union[FEPBatchResponse, _Mapping]] = ..., error: _Optional[str] = ..., submitted_at_ms: _Optional[int] = ..., started_at_ms: _Optional[int] = ..., completed_at_ms: _Optional[int] = ...) -> None: ...
