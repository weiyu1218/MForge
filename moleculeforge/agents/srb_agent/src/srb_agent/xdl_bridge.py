"""XDL bridge — exports SSP to XDL format."""

from __future__ import annotations

import os
from xml.etree.ElementTree import Element, SubElement, tostring

_XDL_COMPILER_MODES = frozenset({"local_demo", "production_real"})


def export_xdl(ssp) -> str:
    """Export an SSP to XDL XML format using the xdl-compiler package."""
    mode = os.environ.get("XDL_COMPILER_MODE", "production_real").strip()
    if mode not in _XDL_COMPILER_MODES:
        allowed = ", ".join(sorted(_XDL_COMPILER_MODES))
        raise ValueError(f"XDL_COMPILER_MODE must be one of: {allowed}")
    try:
        from xdl_compiler.compiler import compile_xdl
        from xdl_compiler.serializer import to_xml

        proc = compile_xdl(ssp)
        return to_xml(proc)
    except ImportError as exc:
        if mode == "production_real":
            raise RuntimeError(
                "production_real XDL compilation requires the xdl-compiler package"
            ) from exc
        return _fallback_xml(ssp)


def _fallback_xml(ssp) -> str:
    """Generate basic XDL XML from SSP without xdl-compiler."""
    root = Element("Synthesis")

    hw_el = SubElement(root, "Hardware")
    SubElement(hw_el, "Component", id="reactor_0", type="reactor")

    reagents_el = SubElement(root, "Reagents")
    seen_r = set()
    for mat in ssp.materials:
        if mat.id not in seen_r:
            seen_r.add(mat.id)
            SubElement(
                reagents_el,
                "Reagent",
                id=mat.id,
                smiles=mat.smiles,
                quantity=str(mat.quantity),
                unit=mat.unit,
            )
    for step in ssp.steps:
        for r in step.reagents:
            if r not in seen_r:
                seen_r.add(r)
                SubElement(reagents_el, "Reagent", name=r)
    if not seen_r:
        SubElement(reagents_el, "Reagent", id="default", smiles="CCO")

    proc_el = SubElement(root, "Procedure")
    op_map = {
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
    for step in ssp.steps:
        tag = op_map.get(step.operation, "Stir")
        temp = step.temperature_C if step.temperature_C is not None else 25.0
        time_h = step.time_h if step.time_h is not None else 2.0
        SubElement(proc_el, tag, vessel="reactor_0", temp=str(temp), duration_h=str(time_h))

    return tostring(root, encoding="unicode")
