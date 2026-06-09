"""Unit tests for the L0 RDKit oracle."""

from __future__ import annotations

import asyncio

import pytest


def _run(coro):
    return asyncio.run(coro)


class TestRDKitOracle:
    def test_import(self) -> None:
        from mf_oracles.rdkit_oracle.oracle import RDKitOracle

        oracle = RDKitOracle()
        assert oracle.name == "rdkit_oracle_l0"
        assert oracle.oracle_level.value == "L0"

    def test_sa_score(self) -> None:
        from mf_oracles.rdkit_oracle.scorer import compute_sa_score

        # Aspirin — should be easy to synthesize
        sa = compute_sa_score("CC(=O)Oc1ccccc1C(=O)O")
        assert sa is not None
        assert 1.0 <= sa <= 10.0

    def test_sa_score_invalid(self) -> None:
        from mf_oracles.rdkit_oracle.scorer import compute_sa_score

        assert compute_sa_score("not_valid!!!") is None

    def test_qed(self) -> None:
        from mf_oracles.rdkit_oracle.scorer import compute_qed

        qed = compute_qed("c1ccccc1")
        assert qed is not None
        assert 0.0 <= qed <= 1.0

    def test_lipinski_violations(self) -> None:
        from mf_oracles.rdkit_oracle.scorer import count_lipinski_violations

        # Simple drug-like molecule
        v = count_lipinski_violations("CC(=O)Oc1ccccc1C(=O)O")
        assert isinstance(v, int)
        assert 0 <= v <= 4

    def test_composite_score(self) -> None:
        from mf_oracles.rdkit_oracle.scorer import compute_composite_score

        score = compute_composite_score("c1ccccc1")
        assert 0.0 <= score <= 1.0

    def test_pains_alerts_penalize_composite_score(self) -> None:
        from mf_oracles.rdkit_oracle.scorer import (
            compute_composite_score,
            has_pains_alert,
            pains_alerts,
        )

        pains_smiles = "O=C1C=CC(=O)C=C1"
        control_smiles = "CCO"

        if has_pains_alert(pains_smiles) is None:
            pytest.skip("RDKit PAINS filter catalog is unavailable")

        assert has_pains_alert(pains_smiles) is True
        assert pains_alerts(pains_smiles)
        assert compute_composite_score(pains_smiles) < compute_composite_score(control_smiles)

    def test_predict_async(self) -> None:
        from mf_core.types.molecule import MoleculeModel
        from mf_oracles.rdkit_oracle.oracle import RDKitOracle

        oracle = RDKitOracle()
        mol = MoleculeModel(id="test", smiles="c1ccccc1")

        score = _run(oracle.predict(mol, "any"))
        assert 0.0 <= score <= 1.0

    def test_predict_with_uncertainty(self) -> None:
        from mf_core.types.molecule import MoleculeModel
        from mf_oracles.rdkit_oracle.oracle import RDKitOracle

        oracle = RDKitOracle()
        mol = MoleculeModel(id="test", smiles="CC(=O)Oc1ccccc1C(=O)O")

        score, unc = _run(oracle.predict_with_uncertainty(mol, "any"))
        assert 0.0 <= score <= 1.0
        assert unc == 0.0

    def test_evaluate_includes_pains_metadata(self) -> None:
        from mf_oracles.rdkit_oracle.oracle import RDKitOracle

        result = _run(RDKitOracle().evaluate(["CCO"], ["admet_score"]))

        assert "pains_alert" in result["CCO"]
        assert "pains_alerts" in result["CCO"]


class _DescriptorADMETRunner:
    def __init__(self) -> None:
        self.evaluate_rows = None
        self.uncertainty_rows = None

    def evaluate(self, descriptor_rows, properties):
        self.evaluate_rows = descriptor_rows
        return {
            row["smiles"]: {prop: row["qed"] for prop in properties}
            for row in descriptor_rows
        }

    def predict_with_uncertainty(self, descriptor_rows, properties):
        self.uncertainty_rows = descriptor_rows
        return {
            row["smiles"]: (
                {prop: row["qed"] for prop in properties},
                {prop: 0.05 for prop in properties},
            )
            for row in descriptor_rows
        }


class TestADMETAIOracle:
    def test_admet_runner_receives_rdkit_descriptors(self) -> None:
        from mf_oracles.admet_ai.oracle import ADMETAIOracle

        runner = _DescriptorADMETRunner()
        oracle = ADMETAIOracle(runner=runner)

        result = _run(oracle.evaluate(["CCO"], ["clearance"]))

        assert runner.evaluate_rows is not None
        row = runner.evaluate_rows[0]
        assert row["smiles"] == "CCO"
        assert row["mol_wt"] > 0.0
        assert row["logp"] is not None
        assert row["qed"] > 0.0
        assert "lipinski_violations" in row
        assert result["CCO"]["clearance"] == row["qed"]

    def test_admet_uncertainty_uses_model_uncertainty(self) -> None:
        from mf_oracles.admet_ai.oracle import ADMETAIOracle

        runner = _DescriptorADMETRunner()
        oracle = ADMETAIOracle(runner=runner)

        result = _run(oracle.predict_with_uncertainty(["CCO"], ["clearance"]))

        assert runner.uncertainty_rows is not None
        scores, uncertainty = result["CCO"]
        assert scores["clearance"] == runner.uncertainty_rows[0]["qed"]
        assert uncertainty["clearance"] == 0.05

    def test_http_runner_calls_chemprop_service_with_smiles_and_targets(self) -> None:
        from mf_oracles.admet_ai.oracle import ADMETHTTPRunner

        calls = []

        def post_json(url, payload, timeout):
            calls.append({"url": url, "payload": payload, "timeout": timeout})
            return {
                "results": [
                    {
                        "smiles": "CCO",
                        "predictions": {
                            "clearance": 1.2,
                            "herg": 0.03,
                        },
                    }
                ]
            }

        runner = ADMETHTTPRunner(
            service_url="http://admet.local",
            targets=["clearance", "herg"],
            batch_size=32,
            post_json=post_json,
        )

        result = runner.evaluate([{"smiles": "CCO", "qed": 0.4}], ["clearance"])

        assert calls == [
            {
                "url": "http://admet.local/predict",
                "payload": {
                    "smiles": ["CCO"],
                    "endpoints": ["clearance"],
                    "batch_size": 32,
                },
                "timeout": 120.0,
            }
        ]
        assert result == {"CCO": {"clearance": 1.2}}

    def test_http_runner_predicts_uncertainty_from_service_response(self) -> None:
        from mf_oracles.admet_ai.oracle import ADMETHTTPRunner

        calls = []

        def post_json(url, payload, timeout):
            calls.append({"url": url, "payload": payload, "timeout": timeout})
            return {
                "results": [
                    {
                        "smiles": "CCO",
                        "predictions": {"clearance": 1.2},
                        "uncertainties": {"clearance": 0.08},
                    }
                ]
            }

        runner = ADMETHTTPRunner(
            service_url="http://admet.local",
            targets=["clearance"],
            batch_size=16,
            post_json=post_json,
        )

        result = runner.predict_with_uncertainty(
            [{"smiles": "CCO", "qed": 0.4}],
            ["clearance"],
        )

        assert calls == [
            {
                "url": "http://admet.local/predict",
                "payload": {
                    "smiles": ["CCO"],
                    "endpoints": ["clearance"],
                    "batch_size": 16,
                    "return_uncertainty": True,
                },
                "timeout": 120.0,
            }
        ]
        assert result == {"CCO": ({"clearance": 1.2}, {"clearance": 0.08})}


class _DockingRunner:
    def __init__(self, include_provenance: bool = True) -> None:
        self.include_provenance = include_provenance

    def evaluate(self, molecules, properties):
        result = {}
        for smiles in molecules:
            values = {prop: -7.1 for prop in properties}
            if self.include_provenance:
                values["input_artifact_hash"] = "sha256:input"
                values["stderr_path"] = "/tmp/gnina.stderr"
            result[smiles] = values
        return result


class _BoltzRunner:
    def __init__(self, include_provenance: bool = True) -> None:
        self.include_provenance = include_provenance

    def evaluate(self, molecules, properties):
        result = {}
        for smiles in molecules:
            values = {prop: -8.2 for prop in properties}
            if self.include_provenance:
                values["model_version"] = "boltz-test"
                values["runtime_ms"] = 12.0
            result[smiles] = values
        return result


class TestOracleRunnerProvenance:
    def test_gnina_requires_input_hash_and_stderr_path(self) -> None:
        from mf_oracles.gnina.oracle import GninaOracle

        oracle = GninaOracle(runner=_DockingRunner())

        result = _run(oracle.evaluate(["CCO"], ["docking_score"]))

        assert result["CCO"]["input_artifact_hash"] == "sha256:input"
        assert result["CCO"]["stderr_path"] == "/tmp/gnina.stderr"

    def test_gnina_rejects_missing_runner_provenance(self) -> None:
        from mf_oracles.gnina.oracle import GninaOracle

        oracle = GninaOracle(runner=_DockingRunner(include_provenance=False))

        with pytest.raises(RuntimeError, match="input_artifact_hash"):
            _run(oracle.evaluate(["CCO"], ["docking_score"]))

    def test_boltz_requires_model_version_and_runtime(self) -> None:
        from mf_oracles.boltz2.oracle import Boltz2Oracle

        oracle = Boltz2Oracle(runner=_BoltzRunner())

        result = _run(oracle.evaluate(["CCO"], ["affinity"]))

        assert result["CCO"]["model_version"] == "boltz-test"
        assert result["CCO"]["runtime_ms"] == 12.0
        assert oracle.oracle_level() == 2

    def test_boltz_rejects_missing_model_version(self) -> None:
        from mf_oracles.boltz2.oracle import Boltz2Oracle

        oracle = Boltz2Oracle(runner=_BoltzRunner(include_provenance=False))

        with pytest.raises(RuntimeError, match="model_version"):
            _run(oracle.evaluate(["CCO"], ["affinity"]))

    def test_openfe_can_skip_slow_path_when_runner_missing(self) -> None:
        from mf_oracles.openfe.oracle import OpenFEOracle

        oracle = OpenFEOracle(skip_when_unavailable=True)

        result = _run(oracle.evaluate(["CCO"], ["rbfe"]))

        assert result["CCO"]["skipped"] is True
        assert result["CCO"]["skip_reason"] == "OPENFE_RUNNER is required"
