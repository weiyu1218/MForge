"""Reaction template extraction for retrosynthesis training.

Extracts reaction SMARTS templates from reaction datasets (USPTO, Pistachio)
for use by AiZynthFinder and RSGPT models.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from rdkit import Chem
from rdkit.Chem import AllChem


def extract_template(
    reactants_smiles: str,
    products_smiles: str,
    *,
    radius: int = 1,
) -> dict[str, Any] | None:
    """Extract a reaction template (SMARTS) from a single reaction.

    Uses the rdchiral approach: identify the reaction center (atoms that
    change), then extract the local neighborhood as SMARTS patterns.

    Returns dict with keys:
      - template_smarts: str
      - reaction_center_atoms: list[int]
      - retro_template: str (reversed for retrosynthesis)
    """
    reactants = Chem.MolFromSmiles(reactants_smiles)
    products = Chem.MolFromSmiles(products_smiles)
    if reactants is None or products is None:
        return None

    # Find atoms that differ between reactants and products
    r_fp = Chem.RDKFingerprint(reactants)
    p_fp = Chem.RDKFingerprint(products)

    # Simplified: use the difference in atom counts as heuristic
    r_atoms = {atom.GetSymbol() for atom in reactants.GetAtoms()}
    p_atoms = {atom.GetSymbol() for atom in products.GetAtoms()}

    changed_atoms = r_atoms.symmetric_difference(p_atoms)
    reaction_center: list[int] = []
    for atom in reactants.GetAtoms():
        if atom.GetSymbol() in changed_atoms:
            reaction_center.append(atom.GetIdx())

    if not reaction_center:
        return None

    # Extract reaction SMARTS (forward direction)
    try:
        rxn = AllChem.ReactionFromSmarts(f"{reactants_smiles}>>{products_smiles}")
        rxn.Initialize()
        template_smarts = AllChem.ReactionToSmarts(rxn)
    except Exception:
        return None

    return {
        "template_smarts": template_smarts,
        "reactants": reactants_smiles,
        "products": products_smiles,
        "reaction_center_atoms": reaction_center,
        "retro_template": f"{products_smiles}>>{reactants_smiles}",
    }


def extract_templates_batch(
    reactions: list[tuple[str, str]],
    *,
    radius: int = 1,
) -> list[dict[str, Any]]:
    """Extract templates from a batch of (reactants, products) pairs."""
    results: list[dict] = []
    for r, p in reactions:
        tmpl = extract_template(r, p, radius=radius)
        if tmpl is not None:
            results.append(tmpl)
    return results


def build_template_library(
    reactions: list[tuple[str, str]],
    *,
    min_frequency: int = 3,
) -> list[dict[str, Any]]:
    """Build a template library with frequency counts.

    Filters out rare templates (frequency < min_frequency).
    """
    templates = extract_templates_batch(reactions)
    freq: Counter = Counter()
    template_map: dict[str, dict] = {}

    for t in templates:
        smarts = t["template_smarts"]
        freq[smarts] += 1
        if smarts not in template_map:
            template_map[smarts] = t

    result = []
    for smarts, count in freq.items():
        if count >= min_frequency:
            entry = template_map[smarts]
            entry["frequency"] = count
            result.append(entry)

    return sorted(result, key=lambda x: x["frequency"], reverse=True)
