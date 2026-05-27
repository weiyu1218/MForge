"""Yield estimator for SSP synthesis steps."""
from __future__ import annotations


KNOWN_YIELDS: dict[str, tuple[float, float]] = {
    "Suzuki_coupling": (0.82, 0.06),
    "Buchwald_Hartwig_amination": (0.78, 0.08),
    "amide_coupling": (0.88, 0.05),
    "reductive_amination": (0.80, 0.07),
    "alkylation": (0.85, 0.04),
    "Boc_deprotection": (0.95, 0.02),
    "Sonogashira_coupling": (0.75, 0.09),
    "Mitsunobu_reaction": (0.70, 0.10),
}


def estimate_step_yield(reaction_type: str) -> tuple[float, float]:
    """Estimate yield and uncertainty for a reaction type.

    Returns (yield_estimate, uncertainty) both in (0, 1].
    """
    if reaction_type in KNOWN_YIELDS:
        return KNOWN_YIELDS[reaction_type]
    return (0.72, 0.12)
