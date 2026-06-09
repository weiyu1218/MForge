"""Validation agent oracle cascade behavior."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load_validation_module():
    path = ROOT / "agents/validation_agent/src/validation_agent/agent.py"
    spec = importlib.util.spec_from_file_location("validation_agent_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_validation_agent():
    return _load_validation_module().ValidationAgent


class _Oracle:
    def __init__(self, result: dict) -> None:
        self.result = result
        self.calls: list[tuple[list[str], list[str]]] = []

    async def evaluate(self, molecules: list[str], properties: list[str]):
        self.calls.append((molecules, properties))
        return {molecules[0]: self.result}


class _UncertaintyOracle:
    def __init__(self, scores: dict, uncertainty: dict) -> None:
        self.scores = scores
        self.uncertainty = uncertainty
        self.calls: list[tuple[list[str], list[str]]] = []

    async def predict_with_uncertainty(self, molecules: list[str], properties: list[str]):
        self.calls.append((molecules, properties))
        return {molecules[0]: (self.scores, self.uncertainty)}


@pytest.mark.asyncio
async def test_oracle_cascade_stops_when_l0_filter_fails() -> None:
    ValidationAgent = _load_validation_agent()
    l0 = _Oracle({"admet_score": 0.2, "uncertainty": 0.04})
    l1 = _Oracle({"docking_score": -7.0})
    agent = ValidationAgent(oracles={0: l0, 1: l1})

    result = await agent.process(
        {"smiles": "CCO", "oracle_level": 2, "l0_threshold": 0.5}
    )

    assert result["overall_passed"] is False
    assert result["upgrade_path"] == ["L0"]
    assert "L0_filter" in result["cascade"]
    assert l1.calls == []


@pytest.mark.asyncio
async def test_default_oracle_wiring_runs_l0_and_fails_fast_for_l1() -> None:
    ValidationAgent = _load_validation_agent()
    agent = ValidationAgent()

    assert {0, 1, 2, 3}.issubset(set(agent.oracles))

    l0_result = await agent.process({"smiles": "CCO", "oracle_level": 0})

    assert l0_result["overall_passed"] is True
    assert l0_result["cascade"]["L0_filter"]["result"]["admet_score"] >= 0.0

    with pytest.raises(RuntimeError, match="GNINA_RUNNER is required"):
        await agent.process({"smiles": "CCO", "oracle_level": 1})


def test_default_oracle_wiring_uses_configured_l1_l2_l3_oracle_targets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("L1_DOCKING_ORACLE_TARGET", "localhost:50054")
    monkeypatch.setenv("L2_AFFINITY_ORACLE_TARGET", "localhost:50052")
    monkeypatch.setenv("L3_FEP_ORACLE_TARGET", "localhost:50055")
    module = _load_validation_module()

    agent = module.ValidationAgent()

    assert isinstance(agent.oracles[1], module.OracleGrpcClient)
    assert agent.oracles[1].target == "localhost:50054"
    assert agent.oracles[1].level == 1
    assert agent.oracles[1].oracle_name == "docking"
    assert isinstance(agent.oracles[2], module.OracleGrpcClient)
    assert agent.oracles[2].target == "localhost:50052"
    assert agent.oracles[2].level == 2
    assert agent.oracles[2].oracle_name == "affinity"
    assert isinstance(agent.oracles[3], module.OracleGrpcClient)
    assert agent.oracles[3].target == "localhost:50055"
    assert agent.oracles[3].level == 3
    assert agent.oracles[3].oracle_name == "rbfe"


def test_default_oracle_wiring_uses_configured_l0_admet_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("L0_ADMET_ORACLE_TARGET", "localhost:50056")
    module = _load_validation_module()

    agent = module.ValidationAgent()

    assert isinstance(agent.oracles[0], module.OracleGrpcClient)
    assert agent.oracles[0].target == "localhost:50056"
    assert agent.oracles[0].level == 1
    assert agent.oracles[0].oracle_name == "admet_ai"


@pytest.mark.asyncio
async def test_oracle_cascade_upgrades_through_l1_l2_and_l3_skip() -> None:
    ValidationAgent = _load_validation_agent()
    l0 = _Oracle({"admet_score": 0.9, "uncertainty": 0.03})
    l1 = _Oracle(
        {
            "docking_score": -7.0,
            "input_artifact_hash": "sha256:input",
            "stderr_path": "/tmp/gnina.stderr",
        }
    )
    l2 = _Oracle({"affinity": -8.0, "model_version": "boltz-test", "runtime_ms": 15.0})
    l3 = _Oracle({"skipped": True, "skip_reason": "OPENFE_RUNNER is required"})
    agent = ValidationAgent(oracles={0: l0, 1: l1, 2: l2, 3: l3})

    result = await agent.process(
        {"smiles": "CCO", "oracle_level": 3, "l0_threshold": 0.5}
    )

    assert result["overall_passed"] is True
    assert result["upgrade_path"] == ["L0", "L1", "L2", "L3"]
    assert result["cascade"]["L1_docking"]["result"]["input_artifact_hash"]
    assert result["cascade"]["L2_affinity"]["result"]["model_version"] == "boltz-test"
    assert result["cascade"]["L3_rbfe"]["skipped"] is True


@pytest.mark.asyncio
async def test_oracle_cascade_stops_when_l1_score_fails() -> None:
    ValidationAgent = _load_validation_agent()
    l0 = _UncertaintyOracle({"admet_score": 0.9}, {"admet_score": 0.01})
    l1 = _UncertaintyOracle({"docking_score": -5.5}, {"docking_score": 0.1})
    l2 = _Oracle({"affinity": -8.0})
    agent = ValidationAgent(oracles={0: l0, 1: l1, 2: l2})

    result = await agent.process(
        {
            "smiles": "CCO",
            "oracle_level": 2,
            "l0_threshold": 0.5,
            "l1_max_docking_score": -6.0,
        }
    )

    assert result["overall_passed"] is False
    assert result["upgrade_path"] == ["L0", "L1"]
    assert result["cascade"]["L1_docking"]["uncertainty"] == {"docking_score": 0.1}
    assert l2.calls == []


@pytest.mark.asyncio
async def test_oracle_cascade_stops_when_l2_uncertainty_fails() -> None:
    ValidationAgent = _load_validation_agent()
    l0 = _Oracle({"admet_score": 0.9})
    l1 = _Oracle({"docking_score": -7.0})
    l2 = _UncertaintyOracle({"affinity": -8.0}, {"affinity": 1.5})
    l3 = _Oracle({"rbfe": -0.5})
    agent = ValidationAgent(oracles={0: l0, 1: l1, 2: l2, 3: l3})

    result = await agent.process(
        {"smiles": "CCO", "oracle_level": 3, "l0_threshold": 0.5}
    )

    assert result["overall_passed"] is False
    assert result["upgrade_path"] == ["L0", "L1", "L2"]
    assert result["cascade"]["L2_affinity"]["uncertainty"] == {"affinity": 1.5}
    assert l3.calls == []


@pytest.mark.asyncio
async def test_oracle_cascade_upgrades_through_l4_quantum_correction() -> None:
    ValidationAgent = _load_validation_agent()
    l0 = _Oracle({"admet_score": 0.9})
    l1 = _Oracle({"docking_score": -7.0})
    l2 = _Oracle({"affinity": -8.0})
    l3 = _Oracle({"rbfe": -0.5})
    l4 = _Oracle({"quantum_correction": -0.1})
    agent = ValidationAgent(oracles={0: l0, 1: l1, 2: l2, 3: l3, 4: l4})

    result = await agent.process(
        {"smiles": "CCO", "oracle_level": 4, "l0_threshold": 0.5}
    )

    assert result["overall_passed"] is True
    assert result["upgrade_path"] == ["L0", "L1", "L2", "L3", "L4"]
    assert result["cascade"]["L4_quantum"]["result"]["quantum_correction"] == -0.1
    assert l4.calls == [(["CCO"], ["quantum_correction"])]


@pytest.mark.asyncio
async def test_oracle_cascade_persists_validation_belief_to_crg_repository() -> None:
    ValidationAgent = _load_validation_agent()

    class CRGRepository:
        def __init__(self) -> None:
            self.beliefs: list[dict] = []

        async def write_workflow_belief(self, **kwargs) -> None:
            self.beliefs.append(kwargs)

    repository = CRGRepository()
    l0 = _Oracle({"admet_score": 0.9})
    agent = ValidationAgent(oracles={0: l0}, crg_repository=repository)

    await agent.process(
        {
            "project_id": "project-1",
            "run_id": "run-1",
            "smiles": "CCO",
            "oracle_level": 0,
        }
    )

    assert len(repository.beliefs) == 1
    belief = repository.beliefs[0]
    assert belief["project_id"] == "project-1"
    assert belief["run_id"] == "run-1"
    assert belief["subject"] == "CCO"
    assert belief["predicate"] == "validation_status"
    assert belief["object_value"] == "validated"
    assert belief["source_agent"] == "validation_agent"
    assert belief["evidence_ids"] == ["L0_filter"]


@pytest.mark.asyncio
async def test_oracle_cascade_uses_existing_validation_status_from_shared_crg() -> None:
    ValidationAgent = _load_validation_agent()

    class CRGRepository:
        def __init__(self) -> None:
            self.reads: list[str] = []
            self.beliefs: list[dict] = []

        async def get_run_crg(self, run_id: str) -> dict:
            self.reads.append(run_id)
            return {
                "beliefs": [
                    {
                        "subject": "CCO",
                        "predicate": "validation_status",
                        "object": "failed",
                    }
                ],
                "edges": [],
            }

        async def write_workflow_belief(self, **kwargs) -> None:
            self.beliefs.append(kwargs)

    class Oracle:
        async def evaluate(self, molecules: list[str], properties: list[str]):
            raise AssertionError("cached validation_status must skip oracle cascade")

    repository = CRGRepository()
    agent = ValidationAgent(oracles={0: Oracle()}, crg_repository=repository)

    result = await agent.process(
        {
            "project_id": "project-1",
            "run_id": "run-1",
            "smiles": "CCO",
            "oracle_level": 0,
        }
    )

    assert repository.reads == ["run-1"]
    assert result["status"] == "failed"
    assert result["overall_passed"] is False
    assert result["cascade"]["crg_validation_status"]["skipped"] is True
    assert repository.beliefs[0]["evidence_ids"] == ["crg_validation_status"]


def test_default_oracle_wiring_uses_l4_target(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_validation_module()
    created_targets: list[str] = []

    class OracleClient:
        def __init__(self, target: str, level: int, oracle_name: str) -> None:
            created_targets.append(target)
            self.target = target
            self.level = level
            self.oracle_name = oracle_name

    monkeypatch.setenv("L4_QUANTUM_ORACLE_TARGET", "localhost:50104")
    monkeypatch.setattr(module, "OracleGrpcClient", OracleClient)

    agent = module.ValidationAgent()

    assert created_targets == ["localhost:50104"]
    assert agent.oracles[4].target == "localhost:50104"
    assert agent.oracles[4].level == 4
    assert agent.oracles[4].oracle_name == "quantum"


def test_default_oracle_wiring_uses_l4_quantum_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_validation_module()
    monkeypatch.delenv("L4_QUANTUM_ORACLE_TARGET", raising=False)
    monkeypatch.setenv("L4_QUANTUM_ORACLE_COMMAND", "orca-wrapper --json")

    agent = module.ValidationAgent()

    assert isinstance(agent.oracles[4], module.QuantumCommandOracle)
    assert agent.oracles[4].command == ["orca-wrapper", "--json"]


@pytest.mark.parametrize(
    ("env_var", "command", "engine"),
    [
        ("L4_GPU4PYSCF_COMMAND", "gpu4pyscf-wrapper --json", "gpu4pyscf"),
        ("L4_ORCA_COMMAND", "orca-wrapper --json", "orca"),
    ],
)
def test_default_oracle_wiring_uses_named_l4_quantum_commands(
    monkeypatch: pytest.MonkeyPatch,
    env_var: str,
    command: str,
    engine: str,
) -> None:
    module = _load_validation_module()
    monkeypatch.delenv("L4_QUANTUM_ORACLE_TARGET", raising=False)
    monkeypatch.delenv("L4_QUANTUM_ORACLE_COMMAND", raising=False)
    monkeypatch.setenv(env_var, command)

    agent = module.ValidationAgent()

    assert isinstance(agent.oracles[4], module.QuantumCommandOracle)
    assert agent.oracles[4].command == command.split()
    assert agent.oracles[4].engine == engine


@pytest.mark.asyncio
async def test_quantum_command_oracle_parses_json_scores() -> None:
    module = _load_validation_module()
    calls: list[dict] = []

    def run_command(command, check, capture_output, text, input):
        calls.append(
            {
                "command": command,
                "payload": json.loads(input),
                "check": check,
                "capture_output": capture_output,
                "text": text,
            }
        )
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "scores": {"quantum_correction": -0.12},
                    "uncertainty": {"quantum_correction": 0.03},
                    "engine": "orca",
                }
            ),
            stderr="",
        )

    oracle = module.QuantumCommandOracle(
        "orca-wrapper --json",
        engine="orca",
        run_command=run_command,
    )

    result = await oracle.evaluate(["CCO"], ["quantum_correction"])

    assert calls[0]["command"] == ["orca-wrapper", "--json"]
    assert calls[0]["payload"] == {
        "molecule_smiles": "CCO",
        "requested_properties": ["quantum_correction"],
        "engine": "orca",
    }
    assert result["CCO"]["quantum_correction"] == pytest.approx(-0.12)
    assert result["CCO"]["uncertainty_quantum_correction"] == pytest.approx(0.03)
    assert result["CCO"]["engine"] == "orca"


@pytest.mark.asyncio
async def test_quantum_command_oracle_preflight_rejects_missing_executable() -> None:
    module = _load_validation_module()
    oracle = module.QuantumCommandOracle(
        "missing-l4-quantum --json",
        engine="orca",
    )

    with pytest.raises(RuntimeError, match="not found"):
        await oracle.evaluate(["CCO"], ["quantum_correction"])
