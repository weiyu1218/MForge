"""XDL compiler — converts SSP into XDLProcedure."""
from __future__ import annotations

from xdl_compiler.models import XDLHardware, XDLProcedure, XDLReagent, XDLStep

_OP_TAG_MAP: dict[str, str] = {
    "add": "Add",
    "stir": "Stir",
    "heat": "Heat",
    "cool": "Cool",
    "filter": "Filter",
    "wash": "Wash",
    "dry": "Dry",
    "quench": "Quench",
    "evaporate": "Evaporate",
    "extract": "Extract",
    "separate": "Separate",
    "crystallize": "Crystallize",
    "purify": "Purify",
    "analyze": "Analyze",
    "reflux": "Reflux",
    "distill": "Distill",
    "degas": "Degas",
}


def compile_xdl(ssp) -> XDLProcedure:
    hardware = []
    for i, step in enumerate(ssp.steps):
        hw_type = _op_to_hardware(step.operation)
        hardware.append(XDLHardware(id=f"{hw_type}_{i}", type=hw_type))

    reagents = []
    seen_reagents = set()
    for mat in ssp.materials:
        if mat.id not in seen_reagents:
            seen_reagents.add(mat.id)
            reagents.append(XDLReagent(
                id=mat.id, smiles=mat.smiles,
                quantity=mat.quantity, unit=mat.unit,
            ))

    steps = []
    for i, step in enumerate(ssp.steps):
        tag = _OP_TAG_MAP.get(step.operation, "Stir")
        vessel = hardware[i].id if hardware else "reactor_0"
        temp = step.temperature_C or 25.0
        time_h = step.time_h or 2.0
        attrs = {
            "vessel": vessel,
            "temp": f"{temp}",
            "duration_h": f"{time_h}",
            "ssp_step_id": step.step_id,
        }
        retrosyn_step_id = step.parameters.get("retrosyn_route_step_id")
        if retrosyn_step_id:
            attrs["retrosyn_route_step_id"] = retrosyn_step_id
        if step.reagents:
            for j, r in enumerate(step.reagents):
                attrs[f"reagent_{j}"] = r
        steps.append(XDLStep(tag=tag, attributes=attrs))

    return XDLProcedure(hardware=hardware, reagents=reagents, steps=steps)


def _op_to_hardware(operation: str) -> str:
    """Map operation to hardware type needed."""
    return "reactor"
