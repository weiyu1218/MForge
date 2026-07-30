"""Unit tests for the L0 RDKit oracle."""

from __future__ import annotations

import asyncio
import math
import threading

import httpx
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

    def test_evaluate_computes_complete_l0_metric_contract(self) -> None:
        from mf_oracles.rdkit_oracle.oracle import RDKitOracle

        result = _run(
            RDKitOracle().evaluate(
                ["CCO"],
                [
                    "qed",
                    "sa_score",
                    "logp",
                    "lipinski_violations",
                    "admet_score",
                ],
            )
        )

        assert set(result["CCO"]) >= {
            "qed",
            "sa_score",
            "logp",
            "lipinski_violations",
            "admet_score",
        }
        assert all(
            math.isfinite(float(result["CCO"][metric]))
            for metric in (
                "qed",
                "sa_score",
                "logp",
                "lipinski_violations",
                "admet_score",
            )
        )

    def test_evaluate_rejects_invalid_smiles(self) -> None:
        from mf_oracles.rdkit_oracle.oracle import RDKitOracle

        with pytest.raises(ValueError, match="invalid SMILES"):
            _run(RDKitOracle().evaluate(["not_valid!!!"], ["admet_score"]))


class _DescriptorADMETRunner:
    def __init__(self) -> None:
        self.evaluate_rows = None
        self.uncertainty_rows = None

    def evaluate(self, descriptor_rows, properties):
        self.evaluate_rows = descriptor_rows
        return {row["smiles"]: {prop: row["qed"] for prop in properties} for row in descriptor_rows}

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
    @pytest.mark.asyncio
    async def test_sync_admet_runner_executes_outside_event_loop_thread(self) -> None:
        from mf_oracles.admet_ai.oracle import ADMETAIOracle

        event_loop_thread = threading.get_ident()

        class Runner:
            def evaluate(self, descriptor_rows, properties):
                assert threading.get_ident() != event_loop_thread
                return {
                    row["smiles"]: {property_name: 1.0 for property_name in properties}
                    for row in descriptor_rows
                }

        result = await ADMETAIOracle(runner=Runner()).evaluate(
            ["CCO"],
            ["clearance"],
        )

        assert result == {"CCO": {"clearance": 1.0}}

    def test_admet_runner_receives_rdkit_descriptors(self) -> None:
        from mf_oracles.admet_ai.oracle import ADMETAIOracle

        runner = _DescriptorADMETRunner()
        oracle = ADMETAIOracle(runner=runner)

        result = _run(oracle.evaluate(["CCO"], ["clearance"]))

        assert oracle.oracle_level() == 1
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
                "model_version": "chemprop-2026-07",
                "artifact": {
                    "name": "admet-chemprop",
                    "sha256": "a" * 64,
                },
                "results": [
                    {
                        "smiles": "CCO",
                        "predictions": {
                            "clearance": 1.2,
                            "herg": 0.03,
                        },
                    }
                ],
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
        assert result.model_version == "chemprop-2026-07"
        assert result.artifact_name == "admet-chemprop"
        assert result.artifact_checksum == f"sha256:{'a' * 64}"

    def test_http_runner_predicts_uncertainty_from_service_response(self) -> None:
        from mf_oracles.admet_ai.oracle import ADMETHTTPRunner

        calls = []

        def post_json(url, payload, timeout):
            calls.append({"url": url, "payload": payload, "timeout": timeout})
            return {
                "model_version": "chemprop-2026-07",
                "artifact": {
                    "name": "admet-chemprop",
                    "sha256": "b" * 64,
                },
                "results": [
                    {
                        "smiles": "CCO",
                        "predictions": {"clearance": 1.2},
                        "uncertainties": {"clearance": 0.08},
                    }
                ],
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
        assert result.model_version == "chemprop-2026-07"
        assert result.artifact_checksum == f"sha256:{'b' * 64}"

    @pytest.mark.parametrize(
        "service_url",
        [
            "admet.local",
            "ftp://admet.local",
            "file:///tmp/admet.sock",
            "http:///missing-host",
        ],
    )
    def test_http_runner_requires_http_or_https_url(self, service_url: str) -> None:
        from mf_oracles.admet_ai.oracle import ADMETHTTPRunner

        with pytest.raises(RuntimeError, match=r"http\(s\)"):
            ADMETHTTPRunner(
                service_url=service_url,
                targets=["clearance"],
            )

    def test_http_runner_from_env_reads_finite_positive_timeout(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from mf_oracles.admet_ai.oracle import ADMETHTTPRunner

        monkeypatch.setenv("ADMET_SERVICE_URL", "https://admet.local")
        monkeypatch.setenv("ADMET_TARGETS", "clearance")
        monkeypatch.setenv("ADMET_ORACLE_TIMEOUT_SECONDS", "17.5")

        assert ADMETHTTPRunner.from_env().timeout == pytest.approx(17.5)

        monkeypatch.setenv("ADMET_ORACLE_TIMEOUT_SECONDS", "nan")
        with pytest.raises(RuntimeError, match="finite positive"):
            ADMETHTTPRunner.from_env()

    def test_http_runner_rejects_missing_model_artifact_metadata(self) -> None:
        from mf_core.plugins.oracle import OracleDataError
        from mf_oracles.admet_ai.oracle import ADMETHTTPRunner

        runner = ADMETHTTPRunner(
            service_url="https://admet.local",
            targets=["clearance"],
            post_json=lambda *_: {
                "results": [
                    {
                        "smiles": "CCO",
                        "predictions": {"clearance": 1.2},
                    }
                ]
            },
        )

        with pytest.raises(OracleDataError, match="model_version"):
            runner.evaluate([{"smiles": "CCO"}], ["clearance"])

    def test_http_runner_rejects_result_identity_or_quantity_mismatch(self) -> None:
        from mf_core.plugins.oracle import OracleDataError
        from mf_oracles.admet_ai.oracle import ADMETHTTPRunner

        runner = ADMETHTTPRunner(
            service_url="https://admet.local",
            targets=["clearance"],
            post_json=lambda *_: {
                "model_version": "chemprop-2026-07",
                "artifact": {
                    "name": "admet-chemprop",
                    "sha256": "c" * 64,
                },
                "results": [
                    {
                        "smiles": "CCN",
                        "predictions": {"clearance": 1.2},
                    }
                ],
            },
        )

        with pytest.raises(OracleDataError, match="order"):
            runner.evaluate([{"smiles": "CCO"}], ["clearance"])

    @pytest.mark.parametrize(
        ("status_code", "error_type"),
        [
            (408, "timeout"),
            (422, "data"),
            (429, "unavailable"),
            (503, "unavailable"),
        ],
    )
    def test_http_runner_maps_http_failures(
        self,
        monkeypatch: pytest.MonkeyPatch,
        status_code: int,
        error_type: str,
    ) -> None:
        from mf_core.plugins.oracle import OracleDataError, OracleUnavailableError
        from mf_oracles.admet_ai.oracle import _httpx_post_json

        request = httpx.Request("POST", "https://admet.local/predict")
        response = httpx.Response(status_code, request=request)
        monkeypatch.setattr(httpx, "post", lambda *args, **kwargs: response)
        expected = {
            "data": OracleDataError,
            "timeout": TimeoutError,
            "unavailable": OracleUnavailableError,
        }[error_type]

        with pytest.raises(expected):
            _httpx_post_json(
                "https://admet.local/predict",
                {"smiles": ["CCO"]},
                1.0,
            )

    def test_http_runner_maps_transport_timeout_to_timeout_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from mf_oracles.admet_ai.oracle import _httpx_post_json

        def timeout(*args, **kwargs):
            raise httpx.ReadTimeout("timed out")

        monkeypatch.setattr(httpx, "post", timeout)

        with pytest.raises(TimeoutError):
            _httpx_post_json(
                "https://admet.local/predict",
                {"smiles": ["CCO"]},
                1.0,
            )

    @pytest.mark.asyncio
    async def test_admet_oracle_restores_original_smiles_after_canonical_http_input(self) -> None:
        from mf_oracles.admet_ai.oracle import ADMETAIOracle, ADMETHTTPRunner

        runner = ADMETHTTPRunner(
            service_url="https://admet.local",
            targets=["clearance"],
            post_json=lambda _url, payload, _timeout: {
                "model_version": "chemprop-2026-07",
                "artifact": {
                    "name": "admet-chemprop",
                    "sha256": "d" * 64,
                },
                "results": [
                    {
                        "smiles": payload["smiles"][0],
                        "predictions": {"clearance": 1.2},
                    }
                ],
            },
        )

        result = await ADMETAIOracle(runner=runner).evaluate(
            ["C(C)O"],
            ["clearance"],
        )

        assert result == {"C(C)O": {"clearance": 1.2}}
        assert result.model_version == "chemprop-2026-07"


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
        assert oracle.oracle_level() == 1

    def test_boltz_rejects_missing_model_version(self) -> None:
        from mf_oracles.boltz2.oracle import Boltz2Oracle

        oracle = Boltz2Oracle(runner=_BoltzRunner(include_provenance=False))

        with pytest.raises(RuntimeError, match="model_version"):
            _run(oracle.evaluate(["CCO"], ["affinity"]))

    @pytest.mark.asyncio
    async def test_sync_boltz_plugin_runner_executes_outside_event_loop_thread(self) -> None:
        from mf_oracles.boltz2.oracle import Boltz2Oracle

        event_loop_thread = threading.get_ident()

        class Runner:
            def evaluate(self, molecules, properties):
                assert threading.get_ident() != event_loop_thread
                return {
                    smiles: {
                        properties[0]: -8.0,
                        "model_version": "boltz-test",
                        "runtime_ms": 1.0,
                    }
                    for smiles in molecules
                }

            def predict_with_uncertainty(self, molecules, properties):
                assert threading.get_ident() != event_loop_thread
                return {
                    smiles: (
                        {properties[0]: -8.0},
                        {properties[0]: 0.1},
                    )
                    for smiles in molecules
                }

        oracle = Boltz2Oracle(runner=Runner())

        result = await oracle.evaluate(["CCO"], ["affinity"])
        uncertain = await oracle.predict_with_uncertainty(["CCO"], ["affinity"])

        assert result["CCO"]["affinity"] == pytest.approx(-8.0)
        assert uncertain["CCO"][1]["affinity"] == pytest.approx(0.1)

    def test_openfe_can_skip_slow_path_when_runner_missing(self) -> None:
        from mf_oracles.openfe.oracle import OpenFEOracle

        oracle = OpenFEOracle(skip_when_unavailable=True)

        result = _run(oracle.evaluate(["CCO"], ["rbfe"]))

        assert result["CCO"]["skipped"] is True
        assert result["CCO"]["skip_reason"] == "OPENFE_RUNNER is required"
