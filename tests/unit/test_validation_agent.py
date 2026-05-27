"""Validation agent oracle cascade behavior."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load_validation_agent():
    path = ROOT / "agents/validation_agent/src/validation_agent/agent.py"
    spec = importlib.util.spec_from_file_location("validation_agent_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ValidationAgent


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
