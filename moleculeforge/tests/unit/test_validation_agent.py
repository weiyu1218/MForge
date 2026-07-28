"""Validation agent oracle cascade behavior."""

from __future__ import annotations

import importlib.util
import json
import math
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


def _valid_l1_validation_cache_contract() -> dict:
    return {
        "schema_version": "validation_result.v1",
        "validation_policy": {
            "oracle_level": 1,
            "thresholds": {
                "L0": {"admet_score_min": 0.5},
                "L1": {
                    "docking_score_max": -6.0,
                    "docking_score_uncertainty_max": 1.0,
                },
            },
        },
        "result": {
            "agent": "validation_agent",
            "status": "validated",
            "smiles": "CCO",
            "max_oracle_level": 1,
            "requested_oracle_level": 1,
            "cascade": {
                "L0_filter": {
                    "completed": True,
                    "passed": True,
                    "result": {"admet_score": 0.9},
                    "uncertainty": None,
                    "thresholds": {"admet_score_min": 0.5},
                },
                "L1_docking": {
                    "completed": True,
                    "passed": True,
                    "result": {"docking_score": -7.0},
                    "uncertainty": None,
                    "thresholds": {
                        "docking_score_max": -6.0,
                        "docking_score_uncertainty_max": 1.0,
                    },
                },
            },
            "upgrade_path": ["L0", "L1"],
            "overall_passed": True,
        },
    }


def _damage_validation_cache(contract: dict, damage: str) -> None:
    result = contract["result"]
    if damage == "empty_cascade":
        result["cascade"] = {}
    elif damage == "status_overall_conflict":
        result["status"] = "failed"
    elif damage == "missing_actual_level":
        result.pop("max_oracle_level")
    elif damage == "non_contiguous_upgrade_path":
        result["upgrade_path"] = ["L1"]
    elif damage == "wrong_requested_level":
        result["requested_oracle_level"] = 0
    elif damage == "missing_level_thresholds":
        result["cascade"]["L1_docking"].pop("thresholds")
    elif damage == "last_level_pass_conflict":
        result["cascade"]["L1_docking"]["passed"] = False
    elif damage == "passed_threshold_conflict":
        result["cascade"]["L1_docking"]["result"]["docking_score"] = -5.0
    elif damage == "premature_success":
        result["max_oracle_level"] = 0
        result["cascade"].pop("L1_docking")
        result["upgrade_path"] = ["L0"]
    elif damage == "boolean_score":
        result["cascade"]["L0_filter"]["result"]["admet_score"] = True
    elif damage == "non_finite_score":
        result["cascade"]["L1_docking"]["result"]["docking_score"] = math.inf
    elif damage == "boolean_uncertainty":
        result["cascade"]["L1_docking"]["uncertainty"] = {
            "docking_score": True,
        }
    elif damage == "non_finite_uncertainty":
        result["cascade"]["L1_docking"]["uncertainty"] = {
            "docking_score": math.nan,
        }
    else:
        raise AssertionError(f"unknown cache damage: {damage}")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "damage",
    [
        "empty_cascade",
        "status_overall_conflict",
        "missing_actual_level",
        "non_contiguous_upgrade_path",
        "wrong_requested_level",
        "missing_level_thresholds",
        "last_level_pass_conflict",
        "passed_threshold_conflict",
        "premature_success",
        "boolean_score",
        "non_finite_score",
        "boolean_uncertainty",
        "non_finite_uncertainty",
    ],
)
async def test_damaged_validation_cache_executes_oracles(damage: str) -> None:
    ValidationAgent = _load_validation_agent()
    contract = _valid_l1_validation_cache_contract()
    _damage_validation_cache(contract, damage)

    class CRGRepository:
        async def get_run_crg(self, run_id: str) -> dict:
            return {
                "run_id": run_id,
                "beliefs": [
                    {
                        "subject": "CCO",
                        "predicate": "validation_result",
                        "object_value": json.dumps(contract, sort_keys=True),
                    }
                ],
                "edges": [],
            }

        async def write_workflow_belief(self, **kwargs) -> None:
            return None

    l0 = _Oracle({"admet_score": 0.9})
    l1 = _Oracle({"docking_score": -7.0})
    result = await ValidationAgent(
        oracles={0: l0, 1: l1},
        crg_repository=CRGRepository(),
    ).process(
        {
            "project_id": "project-1",
            "run_id": "run-1",
            "smiles": "CCO",
            "oracle_level": 1,
            "l0_threshold": 0.5,
        }
    )

    assert result.get("cached") is None
    assert l0.calls == [(["CCO"], ["admet_score"])]
    assert l1.calls == [(["CCO"], ["docking_score"])]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "oracle_level",
    [True, False, -1, 5, 1.0, "1", None],
)
async def test_oracle_level_requires_an_integer_between_zero_and_four(
    oracle_level: object,
) -> None:
    ValidationAgent = _load_validation_agent()
    oracle = _Oracle({"admet_score": 0.9})
    agent = ValidationAgent(oracles={0: oracle})

    with pytest.raises(ValueError, match="oracle_level must be an integer between 0 and 4"):
        await agent.process({"smiles": "CCO", "oracle_level": oracle_level})

    assert oracle.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("score", [True, math.nan, math.inf, -math.inf])
async def test_live_oracle_score_requires_a_finite_number(score: object) -> None:
    ValidationAgent = _load_validation_agent()
    oracle = _Oracle({"admet_score": score})

    with pytest.raises(RuntimeError, match="finite number"):
        await ValidationAgent(oracles={0: oracle}).process({"smiles": "CCO", "oracle_level": 0})

    assert oracle.calls == [(["CCO"], ["admet_score"])]


@pytest.mark.asyncio
@pytest.mark.parametrize("uncertainty", [True, math.nan, math.inf, -math.inf])
async def test_live_oracle_uncertainty_requires_a_finite_number(
    uncertainty: object,
) -> None:
    ValidationAgent = _load_validation_agent()
    l0 = _Oracle({"admet_score": 0.9})
    l1 = _UncertaintyOracle(
        {"docking_score": -7.0},
        {"docking_score": uncertainty},
    )

    with pytest.raises(RuntimeError, match="finite number"):
        await ValidationAgent(oracles={0: l0, 1: l1}).process({"smiles": "CCO", "oracle_level": 1})

    assert l1.calls == [(["CCO"], ["docking_score"])]


@pytest.mark.asyncio
@pytest.mark.parametrize("threshold", [True, math.nan, math.inf, -math.inf])
async def test_validation_threshold_requires_a_finite_number(
    threshold: object,
) -> None:
    ValidationAgent = _load_validation_agent()
    oracle = _Oracle({"admet_score": 0.9})

    with pytest.raises(ValueError, match="l0_threshold must be a finite number"):
        await ValidationAgent(oracles={0: oracle}).process(
            {
                "smiles": "CCO",
                "oracle_level": 0,
                "l0_threshold": threshold,
            }
        )

    assert oracle.calls == []


@pytest.mark.asyncio
async def test_oracle_cascade_stops_when_l0_filter_fails() -> None:
    ValidationAgent = _load_validation_agent()
    l0 = _Oracle({"admet_score": 0.2, "uncertainty": 0.04})
    l1 = _Oracle({"docking_score": -7.0})
    agent = ValidationAgent(oracles={0: l0, 1: l1})

    result = await agent.process({"smiles": "CCO", "oracle_level": 2, "l0_threshold": 0.5})

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

    result = await agent.process({"smiles": "CCO", "oracle_level": 3, "l0_threshold": 0.5})

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
    assert result["max_oracle_level"] == 1
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

    result = await agent.process({"smiles": "CCO", "oracle_level": 3, "l0_threshold": 0.5})

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

    result = await agent.process({"smiles": "CCO", "oracle_level": 4, "l0_threshold": 0.5})

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

    assert len(repository.beliefs) == 2
    belief = repository.beliefs[0]
    assert belief["project_id"] == "project-1"
    assert belief["run_id"] == "run-1"
    assert belief["subject"] == "CCO"
    assert belief["predicate"] == "validation_status"
    assert belief["object_value"] == "validated"
    assert belief["source_agent"] == "validation_agent"
    assert belief["evidence_ids"] == ["L0_filter"]
    cached_result = repository.beliefs[1]
    assert cached_result["predicate"] == "validation_result"
    cache_contract = json.loads(cached_result["object_value"])
    assert cache_contract["schema_version"] == "validation_result.v1"
    assert cache_contract["validation_policy"] == {
        "oracle_level": 0,
        "thresholds": {
            "L0": {"admet_score_min": 0.0},
        },
    }
    assert cache_contract["result"]["max_oracle_level"] == 0
    assert cache_contract["result"]["cascade"]["L0_filter"]["passed"] is True


@pytest.mark.asyncio
async def test_l0_scalar_validation_status_does_not_satisfy_l4_policy() -> None:
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
                        "object": "validated",
                        "evidence_ids": ["L0_filter"],
                    }
                ],
                "edges": [],
            }

        async def write_workflow_belief(self, **kwargs) -> None:
            self.beliefs.append(kwargs)

    repository = CRGRepository()
    oracles = {
        0: _Oracle({"admet_score": 0.9}),
        1: _Oracle({"docking_score": -7.0}),
        2: _Oracle({"affinity": -8.0}),
        3: _Oracle({"rbfe": -0.5}),
        4: _Oracle({"quantum_correction": -0.1}),
    }
    agent = ValidationAgent(oracles=oracles, crg_repository=repository)

    result = await agent.process(
        {
            "project_id": "project-1",
            "run_id": "run-1",
            "smiles": "CCO",
            "oracle_level": 4,
            "l0_threshold": 0.5,
        }
    )

    assert repository.reads == ["run-1"]
    assert result["status"] == "validated"
    assert result["overall_passed"] is True
    assert result["max_oracle_level"] == 4
    assert result["upgrade_path"] == ["L0", "L1", "L2", "L3", "L4"]
    assert oracles[4].calls == [(["CCO"], ["quantum_correction"])]


@pytest.mark.asyncio
async def test_matching_validation_result_cache_replays_full_cascade() -> None:
    ValidationAgent = _load_validation_agent()

    class CRGRepository:
        def __init__(self) -> None:
            self.beliefs: list[dict] = []

        async def get_run_crg(self, run_id: str) -> dict:
            return {"run_id": run_id, "beliefs": list(self.beliefs), "edges": []}

        async def write_workflow_belief(self, **kwargs) -> None:
            self.beliefs.append(
                {
                    "subject": kwargs["subject"],
                    "predicate": kwargs["predicate"],
                    "object_value": kwargs["object_value"],
                }
            )

    class MustNotRunOracle:
        async def evaluate(self, molecules: list[str], properties: list[str]):
            raise AssertionError("matching validation_result must skip the oracle")

    repository = CRGRepository()
    first_oracle = _Oracle({"admet_score": 0.9})
    request = {
        "project_id": "project-1",
        "run_id": "run-1",
        "smiles": "CCO",
        "oracle_level": 0,
        "l0_threshold": 0.5,
    }
    first = await ValidationAgent(
        oracles={0: first_oracle},
        crg_repository=repository,
    ).process(request)
    second = await ValidationAgent(
        oracles={0: MustNotRunOracle()},
        crg_repository=repository,
    ).process(request)

    assert second["cached"] is True
    assert second["cache_source"] == "shared_crg"
    assert second["cascade"] == first["cascade"]
    assert second["upgrade_path"] == ["L0"]
    assert second["max_oracle_level"] == 0


@pytest.mark.asyncio
async def test_validation_cache_miss_when_threshold_changes() -> None:
    ValidationAgent = _load_validation_agent()

    class CRGRepository:
        def __init__(self) -> None:
            self.beliefs: list[dict] = []

        async def get_run_crg(self, run_id: str) -> dict:
            return {"run_id": run_id, "beliefs": list(self.beliefs), "edges": []}

        async def write_workflow_belief(self, **kwargs) -> None:
            self.beliefs.append(
                {
                    "subject": kwargs["subject"],
                    "predicate": kwargs["predicate"],
                    "object_value": kwargs["object_value"],
                }
            )

    repository = CRGRepository()
    first_oracle = _Oracle({"admet_score": 0.6})
    second_oracle = _Oracle({"admet_score": 0.6})
    base_request = {
        "project_id": "project-1",
        "run_id": "run-1",
        "smiles": "CCO",
        "oracle_level": 0,
    }

    first = await ValidationAgent(
        oracles={0: first_oracle},
        crg_repository=repository,
    ).process({**base_request, "l0_threshold": 0.5})
    second = await ValidationAgent(
        oracles={0: second_oracle},
        crg_repository=repository,
    ).process({**base_request, "l0_threshold": 0.7})

    assert first["overall_passed"] is True
    assert second["overall_passed"] is False
    assert second_oracle.calls == [(["CCO"], ["admet_score"])]


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
