"""Unit tests for SRB Agent (srb_agent.agent.SRBAgent)."""

from __future__ import annotations

import asyncio
import builtins
import subprocess
import sys
from pathlib import Path

import pytest
from srb_agent.auditor import (
    make_compile_completed_event,
    make_compile_start_event,
    make_step_yield_event,
    make_xdl_export_event,
)
from srb_agent.compiler import compile_ssp
from srb_agent.xdl_bridge import _fallback_xml, export_xdl

ROOT = Path(__file__).resolve().parents[2]

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


def _full_srb_request(**overrides) -> dict:
    request = {
        "workflow_scope": "full",
        "project_id": "project-1",
        "run_id": "run-agent",
        "request_id": "request-srb-1",
        "schema_version": "srb.request.v1",
        "candidate_id": "candidate-1",
        "candidate_index": 0,
        "canonical_smiles": "CCOO",
        "route_id": "route-real",
        "molecule": {"smiles": "CCOO"},
        "retrosyn_route": ROUTE_WITH_STEPS,
    }
    request.update(overrides)
    return request


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
        ssp_id="t",
        run_id="r",
        target_smiles="CCO",
        materials=[],
        steps=[SSPStep(step_id="1", operation="add")],
    )
    xml = _fallback_xml(ssp)
    assert "<Synthesis>" in xml
    assert "<Hardware>" in xml
    assert "<Reagents>" in xml
    assert "<Procedure>" in xml


def test_fallback_xml_preserves_explicit_zero_degree_condition() -> None:
    from mf_core.types.ssp import SSP, SSPStep

    ssp = SSP(
        ssp_id="t",
        run_id="r",
        target_smiles="CCO",
        materials=[],
        steps=[
            SSPStep(
                step_id="1",
                operation="heat",
                temperature_C=0.0,
                time_h=1.0,
            )
        ],
    )

    assert 'temp="0.0"' in _fallback_xml(ssp)


def test_export_xdl_rejects_fallback_in_production(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mf_core.types.ssp import SSP, SSPStep

    ssp = SSP(
        ssp_id="t",
        run_id="r",
        target_smiles="CCO",
        materials=[],
        steps=[SSPStep(step_id="1", operation="add")],
    )
    real_import = builtins.__import__

    def import_without_xdl(name, *args, **kwargs):
        if name == "xdl_compiler" or name.startswith("xdl_compiler."):
            raise ImportError("xdl-compiler unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setenv("XDL_COMPILER_MODE", "production_real")
    monkeypatch.setattr(builtins, "__import__", import_without_xdl)

    with pytest.raises(
        RuntimeError,
        match="production_real.*xdl-compiler",
    ):
        export_xdl(ssp)


def test_srb_runtime_install_executes_xdl_compiler() -> None:
    script = "\n".join(
        [
            "from mf_core.types.ssp import SSP, SSPMaterial, SSPStep",
            "from xdl_compiler.compiler import compile_xdl",
            "from xdl_compiler.serializer import to_xml",
            "ssp = SSP(",
            "    ssp_id='ssp-1',",
            "    run_id='run-1',",
            "    target_smiles='CCO',",
            "    materials=[SSPMaterial(id='M-001', smiles='CO', quantity=1.0)],",
            "    steps=[SSPStep(step_id='1', operation='add', temperature_C=25.0, time_h=1.0)],",
            ")",
            "print(to_xml(compile_xdl(ssp)))",
        ]
    )

    completed = subprocess.run(  # noqa: S603
        [
            "uv",
            "run",
            "--isolated",
            "--frozen",
            "--package",
            "srb-agent",
            "--no-dev",
            "python",
            "-I",
            "-c",
            script,
        ],
        cwd=ROOT,
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.startswith("<Synthesis>")


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
async def test_compile_ssp_reuses_stable_id_for_the_same_workflow_input() -> None:
    first = await compile_ssp({"smiles": "CCOO"}, ROUTE_WITH_STEPS, "run-route")
    second = await compile_ssp({"smiles": "CCOO"}, ROUTE_WITH_STEPS, "run-route")

    assert first.ssp_id == second.ssp_id


@pytest.mark.asyncio
async def test_compile_ssp_prefers_route_total_yield_and_total_cost() -> None:
    route = {
        **ROUTE_WITH_STEPS,
        "predicted_yield": 0.63,
        "estimated_cost_usd": 42.5,
    }

    ssp = await compile_ssp({"smiles": "CCOO"}, route, "run-route")

    assert ssp.total_estimated_yield == 0.63
    assert ssp.total_estimated_cost_usd == 42.5


@pytest.mark.asyncio
async def test_srb_protocol_preserves_route_per_gram_cost_without_using_it_as_total() -> None:
    from srb_agent.agent import SRBAgent

    route = {
        **ROUTE_WITH_STEPS,
        "estimated_cost_usd_per_g": 12.5,
    }
    result = await SRBAgent(crg_repository=None).process(_full_srb_request(retrosyn_route=route))
    protocol = result["protocols"][0]

    assert protocol["estimated_cost_usd_per_g"] == 12.5
    assert protocol["total_estimated_cost_usd"] != 12.5


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
        "run-audit",
        ssp.ssp_id,
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
async def test_srb_agent_completes_sila2_plan_with_configured_command(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    from srb_agent.agent import SRBAgent

    runner = tmp_path / "sila2_runner.py"
    marker = tmp_path / "sila2_called"
    runner.write_text(
        "import json, sys\n"
        "from pathlib import Path\n"
        "payload = json.load(sys.stdin)\n"
        f"Path({str(marker)!r}).write_text('called', encoding='utf-8')\n"
        "assert payload['project_id'] == 'project-1'\n"
        "assert payload['candidate_id'] == 'candidate-1'\n"
        "assert payload['candidate_index'] == 0\n"
        "assert payload['canonical_smiles'] == 'CCOO'\n"
        "assert payload['request_id'] == 'request-srb-1'\n"
        "assert payload['ssp_id'].startswith('ssp-')\n"
        "assert payload['run_id'] == 'run-agent'\n"
        "assert payload['route_id'] == 'route-real'\n"
        "assert payload['target_smiles'] == 'CCOO'\n"
        "assert payload['sila2_plan']['steps'][0]['retrosyn_route_step_id'] == 'retro-1'\n"
        "assert '<Synthesis' in payload['xdl_xml']\n"
        "print(json.dumps({"
        "'status': 'completed', "
        "'endpoint': 'sila2://lab-controller', "
        "'job_id': 'job-1', "
        "'project_id': payload['project_id'], "
        "'candidate_id': payload['candidate_id'], "
        "'candidate_index': payload['candidate_index'], "
        "'canonical_smiles': payload['canonical_smiles'], "
        "'request_id': payload['request_id'], "
        "'run_id': payload['run_id'], "
        "'ssp_id': payload['ssp_id'], "
        "'route_id': payload['route_id'], "
        "'target_smiles': payload['target_smiles']"
        "}))\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("SILA2_PLAN_COMMAND", f"{sys.executable} {runner}")
    agent = SRBAgent()

    compiled = await agent.process(_full_srb_request())

    assert marker.exists() is False
    result = await agent.process(
        _full_srb_request(
            action="execute",
            retrosyn_route=None,
            protocols=compiled["protocols"],
        )
    )

    protocol = result["protocols"][0]
    assert protocol["sila2_execution"] == {
        "status": "completed",
        "endpoint": "sila2://lab-controller",
        "job_id": "job-1",
        "project_id": "project-1",
        "candidate_id": "candidate-1",
        "candidate_index": 0,
        "canonical_smiles": "CCOO",
        "request_id": "request-srb-1",
        "run_id": "run-agent",
        "ssp_id": protocol["ssp_id"],
        "route_id": "route-real",
        "target_smiles": "CCOO",
    }
    assert marker.read_text(encoding="utf-8") == "called"
    assert result["status"] == "executed"
    assert result["route_id"] == "route-real"
    assert protocol["sila2_endpoint"] == "sila2://lab-controller"
    assert protocol["sila2_plan"]["endpoint"] == "sila2://lab-controller"


@pytest.mark.asyncio
async def test_srb_agent_sila2_command_does_not_block_event_loop(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    from srb_agent.agent import SRBAgent

    runner = tmp_path / "slow_sila2_runner.py"
    runner.write_text(
        "import json, sys, time\n"
        "payload = json.load(sys.stdin)\n"
        "time.sleep(0.2)\n"
        "print(json.dumps({"
        "'status': 'completed', 'job_id': 'job-slow', "
        "**{field: payload[field] for field in ("
        "'project_id', 'candidate_id', 'candidate_index', 'canonical_smiles', "
        "'run_id', 'request_id', 'ssp_id', 'route_id', 'target_smiles'"
        ")}}))\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("SILA2_PLAN_COMMAND", f"{sys.executable} {runner}")
    agent = SRBAgent()
    compiled = await agent.process(_full_srb_request())
    task = asyncio.create_task(
        agent.process(
            _full_srb_request(
                action="execute",
                retrosyn_route=None,
                protocols=compiled["protocols"],
            )
        )
    )

    await asyncio.sleep(0.03)

    assert task.done() is False
    await task


@pytest.mark.asyncio
async def test_srb_agent_rejects_missing_sila2_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from srb_agent.agent import SRBAgent

    monkeypatch.setenv("SILA2_PLAN_COMMAND", "missing-sila2-adapter --json")
    agent = SRBAgent()
    compiled = await agent.process(_full_srb_request())

    with pytest.raises(RuntimeError, match="not found"):
        await agent.process(
            _full_srb_request(
                action="execute",
                retrosyn_route=None,
                protocols=compiled["protocols"],
            )
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        {"status": "failed", "job_id": "job-failed"},
        {"status": "submitted", "job_id": "job-pending"},
        {"status": "completed", "job_id": ""},
        {"status": "completed", "job_id": "job-wrong", "route_id": "route-wrong"},
    ],
)
async def test_srb_agent_rejects_unsuccessful_or_unbound_sila2_response(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    response: dict,
) -> None:
    from srb_agent.agent import SRBAgent

    runner = tmp_path / "invalid_sila2_runner.py"
    runner.write_text(
        "import json, sys\n"
        "payload = json.load(sys.stdin)\n"
        f"response = {response!r}\n"
        "for field in ("
        "'project_id', 'candidate_id', 'candidate_index', 'canonical_smiles', "
        "'run_id', 'request_id', 'ssp_id', 'route_id', 'target_smiles'"
        "):\n"
        "    response.setdefault(field, payload[field])\n"
        "print(json.dumps(response))\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("SILA2_PLAN_COMMAND", f"{sys.executable} {runner}")
    agent = SRBAgent()
    compiled = await agent.process(_full_srb_request())

    with pytest.raises(RuntimeError):
        await agent.process(
            _full_srb_request(
                action="execute",
                retrosyn_route=None,
                protocols=compiled["protocols"],
            )
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
async def test_srb_agent_does_not_use_crg_supply_belief_as_execution_cache() -> None:
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

    repository = CRGRepository()
    agent = module.SRBAgent(crg_repository=repository)

    result = await agent.process(_full_srb_request(run_id="run-1"))

    assert repository.reads == []
    assert result["status"] == "compiled"
    assert len(result["protocols"]) == 1
    assert repository.beliefs[0]["predicate"] == "ssp_compiled"
    assert repository.beliefs[0]["object_value"] == "route-real"
    assert repository.beliefs[0]["evidence_ids"] == [result["protocols"][0]["ssp_id"]]


@pytest.mark.asyncio
async def test_srb_agent_rejects_multiple_routes_for_selected_full_workflow_route() -> None:
    from srb_agent.agent import SRBAgent

    second_route = {
        **ROUTE_WITH_STEPS,
        "route_id": "route-second",
    }

    with pytest.raises(ValueError, match="exactly one"):
        await SRBAgent().process(
            _full_srb_request(
                retrosyn_route=None,
                pathways=[ROUTE_WITH_STEPS, second_route],
            )
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda route: route["steps"][0]["reactants"][0].pop("amount_mmol"),
            "amount_mmol",
        ),
        (
            lambda route: route["steps"][0].pop("conditions"),
            "conditions",
        ),
        (
            lambda route: route["steps"][0].pop("yield"),
            "yield",
        ),
    ],
)
async def test_compile_ssp_rejects_missing_planner_execution_evidence(
    mutate,
    message: str,
) -> None:
    route = {
        **ROUTE_WITH_STEPS,
        "steps": [
            {
                **ROUTE_WITH_STEPS["steps"][0],
                "reactants": [
                    dict(reactant) for reactant in ROUTE_WITH_STEPS["steps"][0]["reactants"]
                ],
                "conditions": dict(ROUTE_WITH_STEPS["steps"][0]["conditions"]),
            }
        ],
    }
    mutate(route)

    with pytest.raises(RuntimeError, match=message):
        await compile_ssp({"smiles": "CCOO"}, route, "run-route")
