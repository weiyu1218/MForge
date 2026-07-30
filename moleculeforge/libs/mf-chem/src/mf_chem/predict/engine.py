"""Deterministic molecular property prediction from RDKit descriptors."""
from __future__ import annotations

from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field

try:
    from rdkit import Chem
    from rdkit.Chem import (
        QED,
        Crippen,
        Descriptors,
        Lipinski,
        rdMolDescriptors,
    )
    _HAS_RDKIT = True
except ImportError:  # pragma: no cover
    _HAS_RDKIT = False


@dataclass
class PredictionResult:
    """Bundle of properties predicted for a single SMILES."""

    smiles: str
    canonical_smiles: str
    valid: bool

    # Physicochemical (RDKit, real)
    molecular_weight: float | None = None
    exact_mass: float | None = None
    heavy_atoms: int | None = None
    logp: float | None = None
    tpsa: float | None = None
    hbd: int | None = None
    hba: int | None = None
    rotatable_bonds: int | None = None
    aromatic_rings: int | None = None
    rings: int | None = None
    fraction_csp3: float | None = None
    formal_charge: int | None = None
    qed: float | None = None
    sa_score: float | None = None
    lipinski_violations: int | None = None
    formula: str | None = None

    # Drug-likeness flags (derived)
    drug_likeness: dict = field(default_factory=dict)

    # Learned ADMET results are unavailable until an explicit model is wired.
    admet: dict = field(default_factory=dict)

    # Retained wire fields; populated only by an explicit learned-model service
    humu_embedding_norm: float | None = None
    humu_embedding_mean: float | None = None
    humu_embedding_dim: int | None = None

    # Composite oracle score in [0,1]
    composite_score: float | None = None

    # Identity (for novelty lookup)
    inchi_key: str | None = None

    # Diagnostics
    device: str | None = None
    error: str | None = None
    admet_available: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# RDKit feature extraction (real values)
# ---------------------------------------------------------------------------


def _rdkit_descriptors(smiles: str) -> tuple[object | None, dict]:
    """Compute every RDKit descriptor we expose.

    Returns (mol, props). When SMILES is invalid mol=None and the dict is
    empty so the caller can surface ``valid=False``.
    """
    if not _HAS_RDKIT:
        return None, {}
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, {}

    canonical = Chem.MolToSmiles(mol, canonical=True)
    try:
        from rdkit.Chem import inchi as _inchi
        inchi_key = _inchi.MolToInchiKey(mol) or None
    except Exception:
        inchi_key = None
    props: dict = {
        "canonical_smiles": canonical,
        "inchi_key": inchi_key,
        "molecular_weight": float(Descriptors.MolWt(mol)),
        "exact_mass": float(Descriptors.ExactMolWt(mol)),
        "heavy_atoms": int(mol.GetNumHeavyAtoms()),
        "logp": float(Crippen.MolLogP(mol)),
        "tpsa": float(Descriptors.TPSA(mol)),
        "hbd": int(Lipinski.NumHDonors(mol)),
        "hba": int(Lipinski.NumHAcceptors(mol)),
        "rotatable_bonds": int(Descriptors.NumRotatableBonds(mol)),
        "aromatic_rings": int(Lipinski.NumAromaticRings(mol)),
        "rings": int(rdMolDescriptors.CalcNumRings(mol)),
        "fraction_csp3": float(rdMolDescriptors.CalcFractionCSP3(mol)),
        "formal_charge": int(Chem.GetFormalCharge(mol)),
        "qed": float(QED.qed(mol)),
        "formula": rdMolDescriptors.CalcMolFormula(mol),
    }

    # Lipinski violations
    violations = 0
    if props["molecular_weight"] > 500:
        violations += 1
    if props["hbd"] > 5:
        violations += 1
    if props["hba"] > 10:
        violations += 1
    if props["logp"] > 5:
        violations += 1
    props["lipinski_violations"] = violations

    # SA score: prefer Contrib, otherwise heuristic on RDKit features
    try:  # pragma: no cover (Contrib presence depends on install)
        from rdkit.Contrib.SA_Score import sascore as _sa
        props["sa_score"] = round(float(_sa.calculateScore(mol)), 3)
    except ImportError:
        chiral = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
        sa = (
            1.0
            + 0.005 * props["molecular_weight"]
            + 0.4 * props["rotatable_bonds"]
            - 0.3 * props["rings"]
            + 0.8 * chiral
        )
        props["sa_score"] = round(max(1.0, min(10.0, sa)), 3)

    return mol, props


def _drug_likeness_flags(props: dict) -> dict:
    """Derive Lipinski / Veber / Egan style flags from RDKit features."""
    flags: dict = {}
    flags["lipinski_pass"] = props.get("lipinski_violations", 4) == 0

    rb = props.get("rotatable_bonds")
    tpsa = props.get("tpsa")
    if rb is not None and tpsa is not None:
        flags["veber_pass"] = rb <= 10 and tpsa <= 140

    logp = props.get("logp")
    if logp is not None and tpsa is not None:
        flags["egan_pass"] = logp <= 5.88 and tpsa <= 131.6

    qed = props.get("qed")
    if qed is not None:
        if qed >= 0.67:
            label = "excellent"
        elif qed >= 0.5:
            label = "good"
        elif qed >= 0.35:
            label = "moderate"
        else:
            label = "poor"
        flags["qed_label"] = label
    return flags


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class MolPredictEngine:
    """High-throughput predictor based on real RDKit descriptors.

    Args:
        device_ids: retained for API compatibility; descriptor inference uses CPU.
        humu_dim: retained for API compatibility with older callers.
        max_workers: thread pool size for RDKit-only batches.
    """

    def __init__(
        self,
        device_ids: list[int] | None = None,
        humu_dim: int = 128,
        max_workers: int = 8,
    ) -> None:
        self.humu_dim = humu_dim
        self._max_workers = max_workers
        self._requested_device_ids = list(device_ids or [])

    # -- public API -----------------------------------------------------

    @property
    def devices(self) -> list[str]:
        return ["cpu"]

    def predict_one(self, smiles: str) -> PredictionResult:
        if not smiles or not isinstance(smiles, str):
            return PredictionResult(
                smiles=smiles or "", canonical_smiles="", valid=False,
                error="empty_smiles",
            )
        mol, props = _rdkit_descriptors(smiles)
        if mol is None:
            return PredictionResult(
                smiles=smiles, canonical_smiles="", valid=False,
                error="invalid_smiles",
            )

        composite = self._composite(props)
        result = PredictionResult(
            smiles=smiles,
            canonical_smiles=props["canonical_smiles"],
            inchi_key=props.get("inchi_key"),
            valid=True,
            molecular_weight=props["molecular_weight"],
            exact_mass=props["exact_mass"],
            heavy_atoms=props["heavy_atoms"],
            logp=props["logp"],
            tpsa=props["tpsa"],
            hbd=props["hbd"],
            hba=props["hba"],
            rotatable_bonds=props["rotatable_bonds"],
            aromatic_rings=props["aromatic_rings"],
            rings=props["rings"],
            fraction_csp3=props["fraction_csp3"],
            formal_charge=props["formal_charge"],
            qed=props["qed"],
            sa_score=props["sa_score"],
            lipinski_violations=props["lipinski_violations"],
            formula=props["formula"],
            drug_likeness=_drug_likeness_flags(props),
            composite_score=composite,
            device="cpu",
        )
        return result

    def predict_batch(self, smiles_list: Iterable[str]) -> list[PredictionResult]:
        smiles_list = list(smiles_list)
        if not smiles_list:
            return []
        results: list[PredictionResult] = [None] * len(smiles_list)  # type: ignore[list-item]
        with ThreadPoolExecutor(max_workers=self._max_workers) as ex:
            futures = {
                ex.submit(self.predict_one, s): i for i, s in enumerate(smiles_list)
            }
            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    results[idx] = fut.result()
                except Exception as e:  # noqa: BLE001
                    results[idx] = PredictionResult(
                        smiles=smiles_list[idx],
                        canonical_smiles="",
                        valid=False,
                        error=f"{type(e).__name__}: {e}",
                    )
        return results

    def _composite(self, props: dict) -> float:
        sa_norm = max(0.0, min(1.0, (10.0 - (props["sa_score"] or 5.0)) / 9.0))
        violations_pen = max(0.0, 1.0 - props["lipinski_violations"] * 0.25)
        score = props["qed"] * 0.45 + sa_norm * 0.35 + violations_pen * 0.20
        return round(float(score), 4)


_default_engine: MolPredictEngine | None = None


def get_default_engine() -> MolPredictEngine:
    """Return the process-wide deterministic descriptor engine."""
    global _default_engine
    if _default_engine is None:
        _default_engine = MolPredictEngine()
    return _default_engine
