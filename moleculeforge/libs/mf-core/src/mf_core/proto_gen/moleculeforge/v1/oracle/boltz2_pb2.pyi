from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Boltz2BindingAffinity(_message.Message):
    __slots__ = ("protein_pdb_id", "ligand_smiles", "delta_g_kcal_mol", "uncertainty", "ki_nm", "ensemble_size", "per_member_dg")
    PROTEIN_PDB_ID_FIELD_NUMBER: _ClassVar[int]
    LIGAND_SMILES_FIELD_NUMBER: _ClassVar[int]
    DELTA_G_KCAL_MOL_FIELD_NUMBER: _ClassVar[int]
    UNCERTAINTY_FIELD_NUMBER: _ClassVar[int]
    KI_NM_FIELD_NUMBER: _ClassVar[int]
    ENSEMBLE_SIZE_FIELD_NUMBER: _ClassVar[int]
    PER_MEMBER_DG_FIELD_NUMBER: _ClassVar[int]
    protein_pdb_id: str
    ligand_smiles: str
    delta_g_kcal_mol: float
    uncertainty: float
    ki_nm: float
    ensemble_size: int
    per_member_dg: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, protein_pdb_id: _Optional[str] = ..., ligand_smiles: _Optional[str] = ..., delta_g_kcal_mol: _Optional[float] = ..., uncertainty: _Optional[float] = ..., ki_nm: _Optional[float] = ..., ensemble_size: _Optional[int] = ..., per_member_dg: _Optional[_Iterable[float]] = ...) -> None: ...

class Boltz2BatchRequest(_message.Message):
    __slots__ = ("project_id", "protein_pdb_id", "ligand_smiles", "ensemble_size", "use_triton_inference")
    PROJECT_ID_FIELD_NUMBER: _ClassVar[int]
    PROTEIN_PDB_ID_FIELD_NUMBER: _ClassVar[int]
    LIGAND_SMILES_FIELD_NUMBER: _ClassVar[int]
    ENSEMBLE_SIZE_FIELD_NUMBER: _ClassVar[int]
    USE_TRITON_INFERENCE_FIELD_NUMBER: _ClassVar[int]
    project_id: str
    protein_pdb_id: str
    ligand_smiles: _containers.RepeatedScalarFieldContainer[str]
    ensemble_size: int
    use_triton_inference: bool
    def __init__(self, project_id: _Optional[str] = ..., protein_pdb_id: _Optional[str] = ..., ligand_smiles: _Optional[_Iterable[str]] = ..., ensemble_size: _Optional[int] = ..., use_triton_inference: bool = ...) -> None: ...

class Boltz2BatchResponse(_message.Message):
    __slots__ = ("protein_pdb_id", "affinities", "elapsed_ms")
    PROTEIN_PDB_ID_FIELD_NUMBER: _ClassVar[int]
    AFFINITIES_FIELD_NUMBER: _ClassVar[int]
    ELAPSED_MS_FIELD_NUMBER: _ClassVar[int]
    protein_pdb_id: str
    affinities: _containers.RepeatedCompositeFieldContainer[Boltz2BindingAffinity]
    elapsed_ms: int
    def __init__(self, protein_pdb_id: _Optional[str] = ..., affinities: _Optional[_Iterable[_Union[Boltz2BindingAffinity, _Mapping]]] = ..., elapsed_ms: _Optional[int] = ...) -> None: ...
