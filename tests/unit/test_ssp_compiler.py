"""Unit tests for SSP compiler (srb_agent.compiler.compile_ssp)."""

from __future__ import annotations

import json
import os
import sys

import pytest

# Ensure project packages are importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "libs", "mf-core", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "agents", "srb_agent", "src"))

from mf_core.types.ssp import SSP, SSPMaterial, SSPReactant, SSPStep
from srb_agent.compiler import compile_ssp, _build_steps_from_route, _build_materials
from srb_agent.yield_estimator import estimate_step_yield
from srb_agent.cost_estimator import estimate_total_cost


# ── Fixture: sample molecule & retrosyn route ───────────────────────────────

SAMPLE_MOLECULE = {
    "smiles": "c1ccc(Nc2cccc(C(=O)O)c2)cc1",
    "objective_values": [0.8, 1.2],
    "id": "mol-001",
}

SAMPLE_RETROSYN_ROUTE = {
    "route_id": "ROUTE-001",
    "smiles": "c1ccc(Nc2cccc(C(=O)O)c2)cc1",
    "route_found": True,
    "estimated_yield": 0.75,
    "steps": [
        {
            "step_id": "route-step-1",
            "reaction": "Buchwald-Hartwig amination",
            "reaction_type": "Buchwald_Hartwig_amination",
            "reactants": [
                {"smiles": "c1ccc(Br)cc1", "amount_mmol": 1.0, "source": "Enamine"},
                {"smiles": "Nc1cccc(C(=O)O)c1", "amount_mmol": 1.1, "source": "Enamine"},
            ],
            "reagents": ["Pd2(dba)3", "Xantphos", "NaOtBu"],
            "conditions": {"temperature_C": 110.0, "time_h": 16.0},
            "yield": 0.72,
            "yield_uncertainty": 0.08,
            "purification": "column_chromatography",
        },
        {
            "step_id": "route-step-2",
            "reaction": "amide coupling workup",
            "reaction_type": "amide_coupling",
            "reactants": [
                {"smiles": "c1ccc(Nc2cccc(C(=O)O)c2)cc1", "amount_mmol": 0.7, "source": "step-1"},
            ],
            "reagents": ["HATU", "DIPEA"],
            "conditions": {"temperature_C": 25.0, "time_h": 4.0},
            "yield": 0.81,
            "yield_uncertainty": 0.05,
            "purification": "recrystallization",
        },
    ],
}


# ── Test: compile_ssp contract ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_compile_ssp_returns_valid_ssp():
    """compile_ssp() should return a valid SSP object."""
    ssp = await compile_ssp(SAMPLE_MOLECULE, SAMPLE_RETROSYN_ROUTE, "run-test-001")

    assert isinstance(ssp, SSP)
    assert ssp.ssp_id.startswith("ssp-")
    assert ssp.run_id == "run-test-001"
    assert ssp.target_smiles == SAMPLE_MOLECULE["smiles"]
    assert ssp.route_id is not None and ssp.route_id.startswith("ROUTE-")
    assert len(ssp.steps) == 2
    assert len(ssp.materials) >= 1
    assert ssp.total_estimated_yield is not None and 0 < ssp.total_estimated_yield <= 1.0
    assert ssp.total_estimated_cost_usd is not None and ssp.total_estimated_cost_usd > 0
    assert ssp.xdl_version == "2.0"
    assert ssp.sila2_endpoint is None


@pytest.mark.asyncio
async def test_compile_ssp_steps_have_srb_fields():
    """Each SSPStep should have SRB extension fields populated."""
    ssp = await compile_ssp(SAMPLE_MOLECULE, SAMPLE_RETROSYN_ROUTE, "run-test-002")

    for step in ssp.steps:
        assert step.reaction_type is not None
        assert len(step.reactants) >= 1
        assert isinstance(step.reactants[0], SSPReactant)
        assert len(step.reagents) > 0
        assert step.temperature_C is not None
        assert step.time_h is not None
        assert step.yield_estimate is not None and 0 < step.yield_estimate <= 1.0
        assert step.yield_uncertainty is not None
        assert step.purification is not None


@pytest.mark.asyncio
async def test_compile_ssp_single_step():
    """Single-step route should produce exactly 1 step."""
    route = {**SAMPLE_RETROSYN_ROUTE, "steps": SAMPLE_RETROSYN_ROUTE["steps"][:1]}
    ssp = await compile_ssp(SAMPLE_MOLECULE, route, "run-test-003")
    assert len(ssp.steps) == 1


# ── Test: _build_steps_from_route ────────────────────────────────────────────

def test_build_steps_correct_count():
    """_build_steps_from_route should return one SSP step per route step."""
    steps = _build_steps_from_route(SAMPLE_RETROSYN_ROUTE["steps"])
    assert len(steps) == len(SAMPLE_RETROSYN_ROUTE["steps"])
    for step in steps:
        assert isinstance(step, SSPStep)
        assert step.reaction_type is not None


def test_build_steps_requires_reactants():
    """_build_steps_from_route should reject route steps without reactants."""
    route_step = {**SAMPLE_RETROSYN_ROUTE["steps"][0], "reactants": []}
    with pytest.raises(RuntimeError, match="reactants"):
        _build_steps_from_route([route_step])


# ── Test: _build_materials ───────────────────────────────────────────────────

def test_build_materials_from_steps():
    """_build_materials should extract unique materials."""
    steps = _build_steps_from_route(SAMPLE_RETROSYN_ROUTE["steps"])
    materials = _build_materials(steps)
    assert len(materials) >= 1
    for mat in materials:
        assert isinstance(mat, SSPMaterial)
        assert mat.unit == "mmol"


# ── Test: yield estimator ────────────────────────────────────────────────────

def test_estimate_step_yield_known():
    yield_est, uncertainty = estimate_step_yield("Suzuki_coupling")
    assert 0 < yield_est <= 1.0
    assert uncertainty > 0


def test_estimate_step_yield_unknown():
    yield_est, uncertainty = estimate_step_yield("unknown_reaction_xyz")
    assert 0 < yield_est <= 1.0  # should fallback to generic


# ── Test: cost estimator ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_estimate_total_cost():
    mats = [SSPMaterial(id="M-001", smiles="CCO", quantity=2.0, unit="mmol")]
    cost = await estimate_total_cost(mats, latency_factor=0.0)
    assert cost > 0


# ── Test: SSP model validation ───────────────────────────────────────────────

def test_ssp_extra_forbid():
    """SSP model should reject unknown fields."""
    with pytest.raises(Exception):
        SSP(
            ssp_id="test", run_id="r1", target_smiles="CCO",
            materials=[], steps=[], unknown_field="bad",
        )


def test_ssp_reactant_model():
    reactant = SSPReactant(smiles="CCO", amount_mmol=1.0, source="Enamine")
    assert reactant.smiles == "CCO"
    assert reactant.amount_mmol == 1.0


def test_ssp_step_srb_defaults():
    step = SSPStep(step_id="1", operation="add")
    assert step.reaction_type is None
    assert step.reactants == []
    assert step.reagents == []
    assert step.temperature_C is None


def test_ssp_top_level_defaults():
    ssp = SSP(
        ssp_id="t1", run_id="r1", target_smiles="CCO",
        materials=[], steps=[],
    )
    assert ssp.route_id is None
    assert ssp.total_estimated_yield is None
    assert ssp.xdl_version == "2.0"
    assert ssp.sila2_endpoint is None
