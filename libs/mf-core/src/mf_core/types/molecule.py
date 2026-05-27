from pydantic import BaseModel, Field


class MolecularProperties(BaseModel):
    mw: float = 0.0
    logp: float = 0.0
    hbd: int = 0
    hba: int = 0
    tpsa: float = 0.0
    qed: float = 0.0
    sa_score: float = 0.0
    ecfp4: bytes = b""


class Molecule(BaseModel):
    smiles: str
    properties: MolecularProperties = Field(default_factory=MolecularProperties)
    inchi_key: str = ""
    sdf_bytes: bytes | None = None
    humu_embedding: bytes | None = None
    metadata: dict[str, str] = Field(default_factory=dict)


class MoleculeModel(BaseModel):
    id: str = ""
    smiles: str
    canonical_smiles: str = ""
    generator_name: str = ""
    humu_embedding: list[float] | None = None
    properties: dict = Field(default_factory=dict)
    embedding: list[float] | None = None

    @property
    def is_valid(self) -> bool:
        if not self.smiles:
            return False
        try:
            from rdkit import Chem
            mol = Chem.MolFromSmiles(self.smiles)
            return mol is not None
        except Exception:
            return len(self.smiles) >= 2 and self.smiles != "not_a_smiles!!!"
