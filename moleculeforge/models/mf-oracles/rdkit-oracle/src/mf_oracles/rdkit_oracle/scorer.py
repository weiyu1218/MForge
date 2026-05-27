"""RDKit scoring functions: SA score, QED, Lipinski, composite."""
from __future__ import annotations

PAINS_COMPOSITE_PENALTY = 0.35


def compute_sa_score(smiles: str) -> float | None:
    """Compute Synthetic Accessibility score (1=easy, 10=hard)."""
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        try:
            from rdkit.Contrib.SA_Score import sascore
            return sascore.calculateScore(mol)
        except ImportError:
            # Fallback: simple heuristic based on molecular complexity
            from rdkit.Chem import Descriptors
            mw = Descriptors.MolWt(mol)
            rot_bonds = Descriptors.NumRotatableBonds(mol)
            rings = Descriptors.RingCount(mol)
            chiral = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
            # Rough SA approximation: 1-10 scale
            sa = 1.0 + 0.02 * mw + 0.5 * rot_bonds - 0.3 * rings + 1.0 * chiral
            return round(max(1.0, min(10.0, sa)), 2)
    except Exception:
        return None


def compute_qed(smiles: str) -> float | None:
    """Compute Quantitative Estimate of Drug-likeness (0-1)."""
    try:
        from rdkit import Chem
        from rdkit.Chem.QED import qed
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return qed(mol)
    except Exception:
        return None


def count_lipinski_violations(smiles: str) -> int:
    """Count Lipinski Rule of Five violations (0-4)."""
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, Lipinski
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0
        violations = 0
        if Descriptors.MolWt(mol) > 500:
            violations += 1
        if Lipinski.NumHDonors(mol) > 5:
            violations += 1
        if Lipinski.NumHAcceptors(mol) > 10:
            violations += 1
        if Descriptors.MolLogP(mol) > 5:
            violations += 1
        return violations
    except Exception:
        return 0


def pains_alerts(smiles: str) -> list[str]:
    """Return RDKit PAINS filter descriptions for a SMILES string."""
    try:
        from rdkit import Chem
    except ImportError as exc:
        raise RuntimeError("RDKit is required for PAINS filtering") from exc
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    catalog = _pains_catalog()
    return [match.GetDescription() for match in catalog.GetMatches(mol)]


def has_pains_alert(smiles: str) -> bool | None:
    """Return whether a molecule matches a PAINS filter; None means invalid SMILES."""
    try:
        from rdkit import Chem
    except ImportError as exc:
        raise RuntimeError("RDKit is required for PAINS filtering") from exc
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    catalog = _pains_catalog()
    return bool(catalog.HasMatch(mol))


def _pains_catalog():
    try:
        from rdkit.Chem import FilterCatalog
    except ImportError as exc:
        raise RuntimeError("RDKit PAINS filter catalog is unavailable") from exc
    catalogs = FilterCatalog.FilterCatalogParams.FilterCatalogs
    if not hasattr(catalogs, "PAINS"):
        raise RuntimeError("RDKit PAINS filter catalog is unavailable")
    params = FilterCatalog.FilterCatalogParams()
    params.AddCatalog(catalogs.PAINS)
    return FilterCatalog.FilterCatalog(params)


def compute_composite_score(smiles: str) -> float:
    """Compute composite oracle score combining SA, QED, and Lipinski."""
    qed_val = compute_qed(smiles) or 0.0
    sa_val = compute_sa_score(smiles)
    if sa_val is None:
        sa_val = 5.0
    sa_norm = max(0.0, min(1.0, (10.0 - sa_val) / 9.0))
    violations = count_lipinski_violations(smiles)
    lipinski_penalty = max(0.0, 1.0 - violations * 0.25)
    score = qed_val * 0.4 + sa_norm * 0.35 + lipinski_penalty * 0.25
    if has_pains_alert(smiles) is True:
        score = max(0.0, score - PAINS_COMPOSITE_PENALTY)
    return round(score, 4)
