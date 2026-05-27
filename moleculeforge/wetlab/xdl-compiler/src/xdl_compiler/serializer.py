"""XDL serializer — converts XDLProcedure to ChemputerXDL-compatible XML."""
from __future__ import annotations

from xml.etree.ElementTree import Element, SubElement, tostring


def to_xml(proc) -> str:
    root = Element("Synthesis")

    hw_el = SubElement(root, "Hardware")
    for hw in proc.hardware:
        comp = SubElement(hw_el, "Component", id=hw.id, type=hw.type)

    reagents_el = SubElement(root, "Reagents")
    for r in proc.reagents:
        SubElement(reagents_el, "Reagent", id=r.id, smiles=r.smiles,
                   quantity=str(r.quantity), unit=r.unit)

    proc_el = SubElement(root, "Procedure")
    for s in proc.steps:
        SubElement(proc_el, s.tag, **s.attributes)

    return tostring(root, encoding="unicode")
