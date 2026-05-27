"""Unit tests for XDL compiler (wetlab/xdl-compiler)."""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "libs", "mf-core", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "wetlab", "xdl-compiler", "src"))

from mf_core.types.ssp import SSP, SSPMaterial, SSPReactant, SSPStep
from xdl_compiler.compiler import compile_xdl, _OP_TAG_MAP
from xdl_compiler.models import XDLProcedure, XDLHardware, XDLReagent, XDLStep
from xdl_compiler.serializer import to_xml


# ── Fixture: minimal SSP ─────────────────────────────────────────────────────

def _make_ssp() -> SSP:
    return SSP(
        ssp_id="ssp-test",
        run_id="run-test",
        target_smiles="CCO",
        materials=[
            SSPMaterial(id="M-001", smiles="CCO", quantity=1.0, unit="mmol"),
        ],
        steps=[
            SSPStep(
                step_id="1",
                operation="add",
                parameters={"solvent": "toluene"},
                reaction_type="Buchwald_Hartwig_amination",
                reactants=[SSPReactant(smiles="CCO", amount_mmol=1.0)],
                reagents=["Pd2(dba)3"],
                temperature_C=110.0,
                time_h=16.0,
            ),
            SSPStep(
                step_id="2",
                operation="stir",
                reaction_type="amide_coupling",
                reactants=[SSPReactant(smiles="C1O", amount_mmol=1.0)],
                temperature_C=25.0,
                time_h=4.0,
            ),
        ],
    )


# ── Test: compile_xdl ────────────────────────────────────────────────────────

def test_compile_xdl_returns_procedure():
    ssp = _make_ssp()
    proc = compile_xdl(ssp)
    assert isinstance(proc, XDLProcedure)
    assert len(proc.hardware) == 2
    assert len(proc.reagents) == 1
    assert len(proc.steps) == 2


def test_compile_xdl_hardware():
    ssp = _make_ssp()
    proc = compile_xdl(ssp)
    assert proc.hardware[0].id == "reactor_0"
    assert proc.hardware[0].type == "reactor"


def test_compile_xdl_reagents():
    ssp = _make_ssp()
    proc = compile_xdl(ssp)
    assert proc.reagents[0].id == "M-001"
    assert proc.reagents[0].smiles == "CCO"


def test_compile_xdl_step_tags():
    ssp = _make_ssp()
    proc = compile_xdl(ssp)
    assert proc.steps[0].tag == "Add"
    assert proc.steps[1].tag == "Stir"


def test_compile_xdl_step_attributes():
    ssp = _make_ssp()
    proc = compile_xdl(ssp)
    attrs = proc.steps[0].attributes
    assert "vessel" in attrs
    assert "temp" in attrs
    assert "110" in attrs["temp"]


def test_compile_xdl_all_operations():
    """Every SSP operation should map to a valid XDL tag."""
    for op, tag in _OP_TAG_MAP.items():
        ssp = SSP(
            ssp_id="t", run_id="r", target_smiles="CCO",
            materials=[],
            steps=[SSPStep(step_id="1", operation=op)],  # type: ignore[arg-type]
        )
        proc = compile_xdl(ssp)
        assert proc.steps[0].tag == tag


# ── Test: serializer ─────────────────────────────────────────────────────────

def test_to_xml_returns_string():
    ssp = _make_ssp()
    proc = compile_xdl(ssp)
    xml = to_xml(proc)
    assert isinstance(xml, str)
    assert "<Synthesis>" in xml
    assert "</Synthesis>" in xml


def test_to_xml_contains_hardware():
    ssp = _make_ssp()
    proc = compile_xdl(ssp)
    xml = to_xml(proc)
    assert "<Hardware>" in xml
    assert '<Component id="reactor_0"' in xml


def test_to_xml_contains_reagents():
    ssp = _make_ssp()
    proc = compile_xdl(ssp)
    xml = to_xml(proc)
    assert "<Reagents>" in xml
    assert '<Reagent id="M-001"' in xml


def test_to_xml_contains_procedure():
    ssp = _make_ssp()
    proc = compile_xdl(ssp)
    xml = to_xml(proc)
    assert "<Procedure>" in xml
    assert "<Add" in xml
    assert "<Stir" in xml


# ── Test: XDL model extra_forbid ─────────────────────────────────────────────

def test_xdl_procedure_extra_forbid():
    with pytest.raises(Exception):
        XDLProcedure(bad_field="x")


def test_xdl_hardware_extra_forbid():
    with pytest.raises(Exception):
        XDLHardware(id="h1", type="reactor", bad_field="x")
