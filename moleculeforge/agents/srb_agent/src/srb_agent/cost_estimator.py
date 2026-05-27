"""Cost estimator for SSP synthesis."""
from __future__ import annotations

from mf_core.types.ssp import SSPMaterial, SSPStep


REAGENT_COSTS: dict[str, float] = {
    "Pd2(dba)3": 45.0,
    "Xantphos": 18.0,
    "NaOtBu": 2.5,
    "Pd(PPh3)4": 35.0,
    "K2CO3": 1.2,
    "HATU": 22.0,
    "DIPEA": 3.0,
    "NaBH3CN": 8.0,
    "AcOH": 1.0,
    "KI": 2.0,
    "TFA": 5.0,
    "Pd(PPh3)2Cl2": 30.0,
    "CuI": 4.0,
    "Et3N": 3.0,
    "DIAD": 15.0,
    "PPh3": 8.0,
}


def _estimate_step_cost(step: SSPStep) -> float:
    """Estimate cost of a single step in USD."""
    cost = 0.0
    for reagent in step.reagents:
        cost += REAGENT_COSTS.get(reagent, 5.0)
    # Add labor and energy cost
    if step.time_h:
        cost += step.time_h * 2.0  # $2/hr for equipment + energy
    return cost


async def estimate_total_cost(materials: list[SSPMaterial], latency_factor: float = 0.0) -> float:
    """Estimate total synthesis cost including materials and overhead."""
    base_cost = 50.0  # fixed setup cost
    for mat in materials:
        base_cost += mat.quantity * 10.0  # generic $10/mmol
    base_cost *= (1.0 + latency_factor)
    return round(base_cost, 2)
