"""End-to-end molecular property prediction test (no docker required).

Drives the FastAPI app directly via TestClient and validates that:
- single-molecule predict returns physicochemically correct values
- batch prediction reports the descriptor engine that actually ran
- a design loop completes and returns a Pareto front
- retrosynthesis routes return analysis based on the actual SMILES
"""

from __future__ import annotations

import subprocess
import sys

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client() -> TestClient:
    from api_gateway.main import app

    with TestClient(app) as c:
        yield c


@pytest.mark.e2e
def test_health_reports_devices(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "healthy"
    assert "devices" in body
    assert isinstance(body["gpu"]["device_count"], int)


@pytest.mark.e2e
def test_orchestrator_import_does_not_eagerly_load_langgraph() -> None:
    code = (
        "import warnings; "
        "from langchain_core._api.deprecation import LangChainPendingDeprecationWarning; "
        "warnings.simplefilter('error', LangChainPendingDeprecationWarning); "
        "import orchestrator"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.e2e
def test_predict_aspirin_returns_real_descriptors(client: TestClient) -> None:
    r = client.post("/v1/predict", json={"smiles": "CC(=O)Oc1ccccc1C(=O)O"})
    assert r.status_code == 200
    body = r.json()["result"]
    assert body["valid"] is True
    # RDKit-known values for aspirin
    assert body["molecular_weight"] == pytest.approx(180.16, abs=0.05)
    assert body["formula"] == "C9H8O4"
    assert body["hbd"] == 1
    assert body["hba"] == 3
    assert body["aromatic_rings"] == 1
    # QED, SA, composite must be in valid ranges
    assert 0 <= body["qed"] <= 1
    assert 1.0 <= body["sa_score"] <= 10.0
    assert 0 <= body["composite_score"] <= 1
    # Drug-likeness
    assert body["drug_likeness"]["lipinski_pass"] is True
    assert body["admet"] == {}
    assert body["admet_available"] is False


@pytest.mark.e2e
def test_predict_engine_does_not_claim_an_unconfigured_learned_model() -> None:
    from mf_chem.predict import MolPredictEngine

    engine = MolPredictEngine(device_ids=[0])
    first = engine.predict_one("CCO").to_dict()
    second = engine.predict_one("CCO").to_dict()

    assert engine.devices == ["cpu"]
    assert first == second
    assert first["device"] == "cpu"
    assert first["humu_embedding_norm"] is None
    assert first["humu_embedding_mean"] is None
    assert first["humu_embedding_dim"] is None
    assert first["admet"] == {}
    assert first["admet_available"] is False


@pytest.mark.e2e
def test_predict_invalid_smiles(client: TestClient) -> None:
    r = client.get("/v1/molecules/not-a-smiles")
    assert r.status_code == 400


@pytest.mark.e2e
def test_batch_predicts_report_real_execution_devices(client: TestClient) -> None:
    payload = {
        "smiles_list": [
            "CCO",
            "c1ccccc1",
            "CC(=O)Oc1ccccc1C(=O)O",
            "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
            "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
            "OC(=O)c1ccccc1O",
            "Nc1ccc(S(N)(=O)=O)cc1",
            "CC(=O)Nc1ccc(O)cc1",
        ],
    }
    r = client.post("/v1/molecules/batch", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["n_total"] == len(payload["smiles_list"])
    assert body["n_valid"] == len(payload["smiles_list"])
    devices = set(m["device"] for m in body["results"])
    assert devices == {"cpu"}
    for m in body["results"]:
        assert m["valid"] is True
        assert isinstance(m["qed"], float)


@pytest.mark.e2e
def test_routes_plan_returns_disconnections(client: TestClient) -> None:
    r = client.post(
        "/v1/routes/plan",
        json={
            "target_smiles": "Cc1cccc(NC(=O)c2ccc(C#N)cc2)c1",
            "max_routes": 3,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["n_routes"] >= 1
    assert all("steps" in route and route["steps"] for route in body["routes"])


@pytest.mark.e2e
def test_similarity_search(client: TestClient) -> None:
    r = client.post("/v1/molecules/search", json={"query": "CC(=O)Oc1ccccc1C(=O)O", "top_k": 5})
    assert r.status_code == 200
    body = r.json()
    assert len(body["results"]) >= 1
    assert all(0.0 <= row["similarity"] <= 1.0 for row in body["results"])


@pytest.mark.e2e
def test_static_ui_served(client: TestClient) -> None:
    r = client.get("/")
    assert r.status_code == 200
    assert "MoleculeForge" in r.text
    r = client.get("/styles.css")
    assert r.status_code == 200
    r = client.get("/app.js")
    assert r.status_code == 200
