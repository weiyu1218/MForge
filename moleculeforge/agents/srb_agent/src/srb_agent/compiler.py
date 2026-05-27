"""SRB compiler — converts retrosynthetic routes into SynthesisSequencePlans."""
from __future__ import annotations

import uuid

from mf_core.types.ssp import SSP, SSPMaterial, SSPReactant, SSPStep

REACTION_TYPES = [
    "Buchwald_Hartwig_amination",
    "Suzuki_coupling",
    "amide_coupling",
    "reductive_amination",
    "alkylation",
    "Boc_deprotection",
    "Sonogashira_coupling",
    "Mitsunobu_reaction",
]

PURIFICATIONS = ["column_chromatography", "recrystallization", "extraction", "HPLC"]


async def compile_ssp(molecule: dict, retrosyn_route: dict, run_id: str) -> SSP:
    """Compile a molecule + retrosynthetic route into an SSP."""
    smiles = molecule["smiles"]
    route_steps = retrosyn_route.get("steps")
    if not isinstance(route_steps, list) or not route_steps:
        raise RuntimeError("retrosyn_route.steps is required to compile SSP")

    from srb_agent.cost_estimator import _estimate_step_cost
    from srb_agent.yield_estimator import estimate_step_yield as _yield_fn

    steps = _build_steps_from_route(route_steps)
    materials = _build_materials(steps)

    total_yield = 1.0
    total_cost = 0.0
    for step in steps:
        y, _ = _yield_fn(step.reaction_type or "generic")
        total_yield *= y
        total_cost += _estimate_step_cost(step)

    ssp_id = f"ssp-{uuid.uuid4().hex[:12]}"
    route_id = retrosyn_route.get("route_id")
    if not route_id:
        raise RuntimeError("retrosyn_route.route_id is required to compile SSP")

    return SSP(
        ssp_id=ssp_id,
        run_id=run_id,
        target_smiles=smiles,
        route_id=route_id,
        materials=materials,
        steps=steps,
        total_estimated_yield=round(total_yield, 4),
        total_estimated_cost_usd=round(total_cost, 2),
        xdl_version="2.0",
        sila2_endpoint=None,
    )


def _build_steps_from_route(route_steps: list[dict]) -> list[SSPStep]:
    """Build SSP steps from retrosynthetic route."""
    steps = []
    for i, route_step in enumerate(route_steps):
        reaction_type = _required_str(route_step, "reaction_type")
        conditions = route_step.get("conditions") or {}
        reactants = [
            SSPReactant(
                smiles=_required_str(reactant, "smiles"),
                amount_mmol=float(reactant.get("amount_mmol", 1.0)),
                source=str(reactant.get("source", "")),
            )
            for reactant in route_step.get("reactants", [])
        ]
        if not reactants:
            raise RuntimeError("retrosyn route step requires reactants")
        steps.append(
            SSPStep(
                step_id=str(i + 1),
                operation=str(route_step.get("operation", "add")),
                parameters={
                    "retrosyn_route_step_id": _required_str(route_step, "step_id"),
                    "retrosyn_reaction": _required_str(route_step, "reaction"),
                },
                reaction_type=reaction_type,
                reactants=reactants,
                reagents=[str(reagent) for reagent in route_step.get("reagents", [])],
                temperature_C=float(conditions.get("temperature_C", 25.0)),
                time_h=float(conditions.get("time_h", 4.0)),
                yield_estimate=float(route_step.get("yield", 0.0)),
                yield_uncertainty=float(route_step.get("yield_uncertainty", 0.0)),
                purification=str(route_step.get("purification", "")),
            )
        )
    return steps


def _required_str(data: dict, key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"retrosyn route step requires {key}")
    return value


def _build_materials(steps: list[SSPStep]) -> list[SSPMaterial]:
    """Extract unique materials from steps."""
    seen = set()
    materials = []
    for i, step in enumerate(steps):
        for r in step.reactants:
            if r.smiles not in seen:
                seen.add(r.smiles)
                materials.append(SSPMaterial(
                    id=f"M-{i+1:03d}",
                    smiles=r.smiles,
                    quantity=r.amount_mmol,
                    unit="mmol",
                ))
    return materials


def _reagents_for_type(rxn_type: str) -> list[str]:
    return {
        "Buchwald_Hartwig_amination": ["Pd2(dba)3", "Xantphos", "NaOtBu"],
        "Suzuki_coupling": ["Pd(PPh3)4", "K2CO3"],
        "amide_coupling": ["HATU", "DIPEA"],
        "reductive_amination": ["NaBH3CN", "AcOH"],
        "alkylation": ["K2CO3", "KI"],
        "Boc_deprotection": ["TFA"],
        "Sonogashira_coupling": ["Pd(PPh3)2Cl2", "CuI", "Et3N"],
        "Mitsunobu_reaction": ["DIAD", "PPh3"],
    }.get(rxn_type, ["generic_reagent"])


def _default_temp(rxn_type: str) -> float:
    return {
        "Buchwald_Hartwig_amination": 110.0,
        "Suzuki_coupling": 90.0,
        "amide_coupling": 25.0,
        "reductive_amination": 25.0,
        "alkylation": 80.0,
        "Boc_deprotection": 25.0,
        "Sonogashira_coupling": 60.0,
        "Mitsunobu_reaction": 0.0,
    }.get(rxn_type, 25.0)


def _default_time(rxn_type: str) -> float:
    return {
        "Buchwald_Hartwig_amination": 16.0,
        "Suzuki_coupling": 12.0,
        "amide_coupling": 4.0,
        "reductive_amination": 8.0,
        "alkylation": 6.0,
        "Boc_deprotection": 2.0,
        "Sonogashira_coupling": 8.0,
        "Mitsunobu_reaction": 4.0,
    }.get(rxn_type, 4.0)
