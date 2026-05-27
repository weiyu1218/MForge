from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class MolecularProperties(_message.Message):
    __slots__ = ("mw", "logp", "hbd", "hba", "tpsa", "qed", "sa_score", "ecfp4")
    MW_FIELD_NUMBER: _ClassVar[int]
    LOGP_FIELD_NUMBER: _ClassVar[int]
    HBD_FIELD_NUMBER: _ClassVar[int]
    HBA_FIELD_NUMBER: _ClassVar[int]
    TPSA_FIELD_NUMBER: _ClassVar[int]
    QED_FIELD_NUMBER: _ClassVar[int]
    SA_SCORE_FIELD_NUMBER: _ClassVar[int]
    ECFP4_FIELD_NUMBER: _ClassVar[int]
    mw: float
    logp: float
    hbd: float
    hba: float
    tpsa: float
    qed: float
    sa_score: float
    ecfp4: bytes
    def __init__(self, mw: _Optional[float] = ..., logp: _Optional[float] = ..., hbd: _Optional[float] = ..., hba: _Optional[float] = ..., tpsa: _Optional[float] = ..., qed: _Optional[float] = ..., sa_score: _Optional[float] = ..., ecfp4: _Optional[bytes] = ...) -> None: ...

class Molecule(_message.Message):
    __slots__ = ("smiles", "properties", "inchi_key", "sdf_bytes", "humu_embedding", "metadata")
    class MetadataEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    SMILES_FIELD_NUMBER: _ClassVar[int]
    PROPERTIES_FIELD_NUMBER: _ClassVar[int]
    INCHI_KEY_FIELD_NUMBER: _ClassVar[int]
    SDF_BYTES_FIELD_NUMBER: _ClassVar[int]
    HUMU_EMBEDDING_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    smiles: str
    properties: MolecularProperties
    inchi_key: str
    sdf_bytes: bytes
    humu_embedding: bytes
    metadata: _containers.ScalarMap[str, str]
    def __init__(self, smiles: _Optional[str] = ..., properties: _Optional[_Union[MolecularProperties, _Mapping]] = ..., inchi_key: _Optional[str] = ..., sdf_bytes: _Optional[bytes] = ..., humu_embedding: _Optional[bytes] = ..., metadata: _Optional[_Mapping[str, str]] = ...) -> None: ...

class MoleculeBatch(_message.Message):
    __slots__ = ("molecules", "batch_id")
    MOLECULES_FIELD_NUMBER: _ClassVar[int]
    BATCH_ID_FIELD_NUMBER: _ClassVar[int]
    molecules: _containers.RepeatedCompositeFieldContainer[Molecule]
    batch_id: str
    def __init__(self, molecules: _Optional[_Iterable[_Union[Molecule, _Mapping]]] = ..., batch_id: _Optional[str] = ...) -> None: ...
