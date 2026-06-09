"""Unit tests for SRB Agent (srb_agent.agent.SRBAgent)."""

from __future__ import annotations

import sys

import pytest
from srb_agent.auditor import (
    make_compile_completed_event,
    make_compile_start_event,
    make_step_yield_event,
    make_xdl_export_event,
)
from srb_agent.compiler import compile_ssp
from srb_agent.xdl_bridge import _fallback_xml, export_xdl

# ── Fixtures ─────────────────────────────────────────────────────────────────

SAMPLE_MOLECULE = {
    "smiles": "c1ccc(Nc2cccc(C(=O)O)c2)cc1",
    "objective_values": [0.8, 1.2],
}

SAMPLE_ROUTE = {
    "route_id": "route-sample",
    "smiles": "c1ccc(Nc2cccc(C(=O)O)c2)cc1",
    "estimated_yield": 0.75,
    "steps": [
        {
            "step_id": "sample-retro-1",
            "operation": "add",
            "reaction": "c1ccc(N)cc1.O=C(O)c1cccc(N)c1>>c1ccc(Nc2cccc(C(=O)O)c2)cc1",
            "reaction_type": "amide_coupling",
            "reactants": [
                {
                    "smiles": "c1ccc(N)cc1",
                    "amount_mmol": 1.0,
                    "source": "route",
                },
                {
                    "smiles": "O=C(O)c1cccc(N)c1",
                    "amount_mmol": 1.0,
                    "source": "route",
                },
            ],
            "reagents": ["HATU", "DIPEA"],
            "conditions": {"temperature_C": 25.0, "time_h": 4.0},
            "yield": 0.75,
            "yield_uncertainty": 0.08,
            "purification": "column_chromatography",
        }
    ],
}

ROUTE_WITH_STEPS = {
    "route_id": "route-real",
    "steps": [
        {
            "step_id": "retro-1",
            "operation": "add",
            "reaction": "CCO.O=O>>CCOO",
            "reaction_type": "oxidation",
            "reactants": [{"smiles": "CCO", "amount_mmol": 0.5, "source": "route"}],
            "reagents": ["O2"],
            "conditions": {"temperature_C": 35.0, "time_h": 3.0},
            "yield": 0.71,
            "yield_uncertainty": 0.08,
            "purification": "filtration",
        }
    ],
}


# ── Test: Auditor events ─────────────────────────────────────────────────────

def test_make_compile_start_event():
    evt = make_compile_start_event("run-001", "CCO", "ROUTE-ABC")
    assert evt.actor == "SRBAgent"
    assert evt.action == "srb.compile.start"
    assert evt.run_id == "run-001"
    assert len(evt.content_hash) == 64
    assert evt.signature is None  # not auto-signed


def test_make_step_yield_event():
    evt = make_step_yield_event("run-001", "1", "Suzuki_coupling", 0.78, 0.10)
    assert evt.actor == "SRBAgent.yield_estimator"
    assert evt.action == "srb.step.yield_estimated"
    assert evt.payload_summary["reaction_type"] == "Suzuki_coupling"


def test_make_compile_completed_event():
    evt = make_compile_completed_event("run-001", "ssp-001", 0.5, 100.0, 3)
    assert evt.actor == "SRBAgent"
    assert evt.action == "srb.compile.completed"
    assert evt.payload_summary["n_steps"] == 3


def test_make_xdl_export_event():
    evt = make_xdl_export_event("run-001", "ssp-001", 512)
    assert evt.actor == "XDLCompiler"
    assert evt.action == "srb.xdl.exported"


def test_all_events_have_content_hash():
    events = [
        make_compile_start_event("r1", "CCO", "R1"),
        make_step_yield_event("r1", "1", "test", 0.5, 0.1),
        make_compile_completed_event("r1", "ssp-1", 0.5, 10.0, 1),
        make_xdl_export_event("r1", "ssp-1", 100),
    ]
    for evt in events:
        assert len(evt.content_hash) == 64


# ── Test: XDL bridge ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_export_xdl_fallback():
    """When xdl-compiler is importable, export_xdl should produce XML."""
    ssp = await compile_ssp(SAMPLE_MOLECULE, SAMPLE_ROUTE, "run-001")
    xml = export_xdl(ssp)
    assert isinstance(xml, str)
    assert "<Synthesis>" in xml


def test_fallback_xml_structure():
    """_fallback_xml should produce valid XML with required sections."""
    from mf_core.types.ssp import SSP, SSPStep

    ssp = SSP(
        ssp_id="t", run_id="r", target_smiles="CCO",
        materials=[],
        steps=[SSPStep(step_id="1", operation="add")],
    )
    xml = _fallback_xml(ssp)
    assert "<Synthesis>" in xml
    assert "<Hardware>" in xml
    assert "<Reagents>" in xml
    assert "<Procedure>" in xml


# ── Test: compile_ssp integration ────────────────────────────────────────────

@pytest.mark.asyncio
async def test_compile_ssp_full_chain():
    """Full compile_ssp should produce SSP with all required fields."""
    ssp = await compile_ssp(SAMPLE_MOLECULE, SAMPLE_ROUTE, "run-full")

    # Top-level fields
    assert ssp.ssp_id.startswith("ssp-")
    assert ssp.route_id == "route-sample"
    assert ssp.total_estimated_yield is not None
    assert ssp.total_estimated_cost_usd is not None

    # Step-level fields
    for step in ssp.steps:
        assert step.reaction_type is not None
        assert step.yield_estimate is not None
        assert step.yield_uncertainty is not None
        assert step.purification is not None


@pytest.mark.asyncio
async def test_compile_ssp_uses_retrosyn_route_steps() -> None:
    ssp = await compile_ssp({"smiles": "CCOO"}, ROUTE_WITH_STEPS, "run-route")

    assert ssp.route_id == "route-real"
    assert len(ssp.steps) == 1
    step = ssp.steps[0]
    assert step.step_id == "1"
    assert step.reaction_type == "oxidation"
    assert step.parameters["retrosyn_route_step_id"] == "retro-1"
    assert step.parameters["retrosyn_reaction"] == "CCO.O=O>>CCOO"
    assert step.temperature_C == 35.0
    assert step.time_h == 3.0
    assert step.yield_estimate == 0.71
    assert step.purification == "filtration"


@pytest.mark.asyncio
async def test_xdl_steps_trace_ssp_and_retrosyn_route_step_ids() -> None:
    from xdl_compiler.compiler import compile_xdl

    ssp = await compile_ssp({"smiles": "CCOO"}, ROUTE_WITH_STEPS, "run-route")
    procedure = compile_xdl(ssp)

    assert procedure.steps[0].attributes["ssp_step_id"] == "1"
    assert procedure.steps[0].attributes["retrosyn_route_step_id"] == "retro-1"


@pytest.mark.asyncio
async def test_compile_ssp_produces_audit_events():
    """Compiling and then generating audit events should be coherent."""
    ssp = await compile_ssp(SAMPLE_MOLECULE, SAMPLE_ROUTE, "run-audit")

    start_evt = make_compile_start_event("run-audit", ssp.target_smiles, ssp.route_id or "")
    assert start_evt.payload_summary["route_id"] == ssp.route_id

    completed_evt = make_compile_completed_event(
        "run-audit", ssp.ssp_id,
        ssp.total_estimated_yield or 0,
        ssp.total_estimated_cost_usd or 0,
        len(ssp.steps),
    )
    assert completed_evt.payload_summary["n_steps"] == len(ssp.steps)


@pytest.mark.asyncio
async def test_srb_agent_process_compiles_ssp() -> None:
    from srb_agent.agent import SRBAgent

    agent = SRBAgent()

    result = await agent.process(
        {
            "run_id": "run-agent",
            "molecule": {"smiles": "CCOO"},
            "retrosyn_route": ROUTE_WITH_STEPS,
        }
    )

    assert result["agent"] == "srb_agent"
    assert result["status"] == "compiled"
    assert len(result["protocols"]) == 1
    protocol = result["protocols"][0]
    assert protocol["ssp_id"].startswith("ssp-")
    assert protocol["route_id"] == "route-real"
    assert protocol["target_smiles"] == "CCOO"
    assert protocol["steps"][0]["parameters"]["retrosyn_route_step_id"] == "retro-1"
    assert protocol["total_estimated_yield"] is not None
    assert protocol["total_estimated_cost_usd"] is not None
    assert protocol["xdl_version"] == "2.0"
    assert "<Synthesis" in protocol["xdl_xml"]
    assert protocol["sila2_plan"]["steps"][0]["retrosyn_route_step_id"] == "retro-1"


@pytest.mark.asyncio
async def test_srb_agent_submits_sila2_plan_to_configured_command(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    from srb_agent.agent import SRBAgent

    runner = tmp_path / "sila2_runner.py"
    runner.write_text(
        "import json, sys\n"
        "payload = json.load(sys.stdin)\n"
        "assert payload['ssp_id'].startswith('ssp-')\n"
        "assert payload['run_id'] == 'run-agent'\n"
        "assert payload['route_id'] == 'route-real'\n"
        "assert payload['target_smiles'] == 'CCOO'\n"
        "assert payload['sila2_plan']['steps'][0]['retrosyn_route_step_id'] == 'retro-1'\n"
        "assert '<Synthesis' in payload['xdl_xml']\n"
        "print(json.dumps({"
        "'status': 'submitted', "
        "'endpoint': 'sila2://lab-controller', "
        "'job_id': 'job-1'"
        "}))\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("SILA2_PLAN_COMMAND", f"{sys.executable} {runner}")
    agent = SRBAgent()

    result = await agent.process(
        {
            "run_id": "run-agent",
            "molecule": {"smiles": "CCOO"},
            "retrosyn_route": ROUTE_WITH_STEPS,
        }
    )

    protocol = result["protocols"][0]
    assert protocol["sila2_execution"] == {
        "status": "submitted",
        "endpoint": "sila2://lab-controller",
        "job_id": "job-1",
    }
    assert protocol["sila2_endpoint"] == "sila2://lab-controller"
    assert protocol["sila2_plan"]["endpoint"] == "sila2://lab-controller"


@pytest.mark.asyncio
async def test_srb_agent_rejects_missing_sila2_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from srb_agent.agent import SRBAgent

    monkeypatch.setenv("SILA2_PLAN_COMMAND", "missing-sila2-adapter --json")
    agent = SRBAgent()

    with pytest.raises(RuntimeError, match="not found"):
        await agent.process(
            {
                "run_id": "run-agent",
                "molecule": {"smiles": "CCOO"},
                "retrosyn_route": ROUTE_WITH_STEPS,
            }
        )


@pytest.mark.asyncio
async def test_srb_agent_persists_ssp_compiled_belief() -> None:
    from srb_agent.agent import SRBAgent

    class CRGRepository:
        def __init__(self) -> None:
            self.beliefs: list[dict] = []

        async def write_workflow_belief(self, **kwargs) -> None:
            self.beliefs.append(kwargs)

    repository = CRGRepository()
    agent = SRBAgent(crg_repository=repository)

    result = await agent.process(
        {
            "project_id": "project-1",
            "run_id": "run-1",
            "molecule": {"smiles": "CCOO"},
            "retrosyn_route": ROUTE_WITH_STEPS,
        }
    )

    assert len(repository.beliefs) == 1
    belief = repository.beliefs[0]
    assert belief["project_id"] == "project-1"
    assert belief["run_id"] == "run-1"
    assert belief["subject"] == "CCOO"
    assert belief["predicate"] == "ssp_compiled"
    assert belief["object_value"] == "route-real"
    assert belief["source_agent"] == "srb_agent"
    assert belief["evidence_ids"] == [result["protocols"][0]["ssp_id"]]


@pytest.mark.asyncio
async def test_srb_agent_uses_unavailable_supply_belief_from_shared_crg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import srb_agent.agent as module

    class CRGRepository:
        def __init__(self) -> None:
            self.reads: list[str] = []
            self.beliefs: list[dict] = []

        async def get_run_crg(self, run_id: str) -> dict:
            self.reads.append(run_id)
            return {
                "beliefs": [
                    {
                        "subject": "CCOO",
                        "predicate": "supply_feasibility",
                        "object": "unavailable",
                    }
                ],
                "edges": [],
            }

        async def write_workflow_belief(self, **kwargs) -> None:
            self.beliefs.append(kwargs)

    async def fail_compile(*_args, **_kwargs):
        raise AssertionError("shared CRG unavailable supply must skip SSP compile")

    monkeypatch.setattr(module, "compile_ssp", fail_compile)
    repository = CRGRepository()
    agent = module.SRBAgent(crg_repository=repository)

    result = await agent.process(
        {
            "project_id": "project-1",
            "run_id": "run-1",
            "molecule": {"smiles": "CCOO"},
            "retrosyn_route": ROUTE_WITH_STEPS,
        }
    )

    assert repository.reads == ["run-1"]
    assert result["status"] == "skipped"
    assert result["skip_reason"] == "shared CRG contains unavailable supply_feasibility"
    assert result["protocols"] == []
    assert repository.beliefs[0]["predicate"] == "ssp_compiled"
    assert repository.beliefs[0]["object_value"] == "skipped"
    assert repository.beliefs[0]["evidence_ids"] == ["crg_supply_feasibility"]
