"""Adapter layer for RDKit operations."""
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
    _HAS_RDKIT = True
except ImportError:
    _HAS_RDKIT = False
    Chem = None  # type: ignore
    AllChem = None  # type: ignore
    Descriptors = None  # type: ignore
    rdMolDescriptors = None  # type: ignore


class RDKitAdapter:
    """Adapter providing a consistent interface for RDKit molecule operations."""

    @staticmethod
    def mol_from_smiles(smiles: str):
        """Create an RDKit Mol from a SMILES string."""
        if not _HAS_RDKIT:
            return None
        return Chem.MolFromSmiles(smiles) if smiles else None

    @staticmethod
    def canonical_smiles(smiles: str) -> str:
        """Convert a SMILES string to canonical form."""
        if not _HAS_RDKIT:
            return smiles
        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(mol, canonical=True) if mol else ""

    @staticmethod
    def compute_ecfp4(smiles: str, radius: int = 2, n_bits: int = 2048) -> bytes:
        """Compute ECFP4 (Morgan) fingerprint for a molecule."""
        if not _HAS_RDKIT:
            return b"\x00" * (n_bits // 8)
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return b"\x00" * (n_bits // 8)
        fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        return fp.ToBitString().encode()

    @staticmethod
    def generate_conformer(smiles: str, n_conformers: int = 1) -> list:
        """Generate 3D conformers for a molecule."""
        if not _HAS_RDKIT:
            return []
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return []
        mol = Chem.AddHs(mol)
        AllChem.EmbedMultipleConfs(mol, numConfs=n_conformers, randomSeed=42)
        return [mol]
