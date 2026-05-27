"""
分子解析统一入口。
RDKit 为系统级依赖；Docker 环境由 Dockerfile.chem 保证；
本地开发若未安装 RDKit，模块仍可导入，但调用函数时会抛出 ImportError。
"""
from mf_core.types.molecule import Molecule, MolecularProperties
import warnings

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, QED, rdMolDescriptors
    from rdkit.Chem import AllChem
    _RDKIT_AVAILABLE = True
except ImportError:
    _RDKIT_AVAILABLE = False
    warnings.warn(
        "RDKit not installed. Install via: conda install -c conda-forge rdkit\n"
        "Molecule parsing functions will raise ImportError when called.",
        ImportWarning,
        stacklevel=2,
    )


def _require_rdkit():
    if not _RDKIT_AVAILABLE:
        raise ImportError(
            "RDKit is required for this function. "
            "Install: conda install -c conda-forge rdkit"
        )


def parse_smiles(smiles: str, generate_3d: bool = False) -> Molecule | None:
    """Parse a SMILES string into a Molecule object with computed properties."""
    _require_rdkit()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    props = MolecularProperties(
        mw=Descriptors.MolWt(mol),
        logp=Descriptors.MolLogP(mol),
        hbd=Descriptors.NumHDonors(mol),
        hba=Descriptors.NumHAcceptors(mol),
        tpsa=Descriptors.TPSA(mol),
        qed=QED.qed(mol),
        sa_score=_compute_sa_score(mol),
    )

    sdf_bytes = None
    if generate_3d:
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)
        sdf_bytes = Chem.MolToMolBlock(mol).encode()

    return Molecule(
        smiles=Chem.MolToSmiles(mol, canonical=True),
        properties=props,
        inchi_key=Chem.MolToInchiKey(mol) or "",
        sdf_bytes=sdf_bytes,
    )


def canonicalize(smiles: str) -> str:
    """SMILES 规范化。"""
    _require_rdkit()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    return Chem.MolToSmiles(mol, canonical=True)


def mol_to_inchikey(smiles: str) -> str:
    """SMILES → InChIKey。"""
    _require_rdkit()
    from rdkit.Chem.inchi import MolToInchi
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    inchi = MolToInchi(mol)
    return Chem.InchiInfo.InchiToInchiKey(inchi)


def _compute_sa_score(mol) -> float:
    """Compute synthetic accessibility score.

    Uses RDKit Contrib SA_Score if available, otherwise returns a default.
    """
    try:
        from rdkit.Contrib.SA_Score import sascorer
        return sascorer.calculateScore(mol)
    except ImportError:
        return 5.0
