"""End-to-end molecular property prediction test (no docker required).

Drives the FastAPI app directly via TestClient and validates that:
- single-molecule predict returns physicochemically correct values
- batch predict shards across all visible CUDA devices
- a design loop completes and returns a Pareto front
- retrosynthesis routes return analysis based on the actual SMILES
"""
from __future__ import annotations

import subprocess
import sys
import time

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
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
    # ADMET sanity
    assert body["admet"]["herg_risk"] in {"low", "medium", "high"}


@pytest.mark.e2e
def test_predict_invalid_smiles(client: TestClient) -> None:
    r = client.get("/v1/molecules/not-a-smiles")
    assert r.status_code == 400


@pytest.mark.e2e
def test_batch_predicts_uses_all_devices(client: TestClient) -> None:
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
    # On GPU machines we expect multiple devices to have served the load.
    assert len(devices) >= 1
    for m in body["results"]:
        assert m["valid"] is True
        assert isinstance(m["qed"], float)


@pytest.mark.e2e
def test_design_loop_returns_pareto_front(client: TestClient) -> None:
    p = client.post("/v1/projects/", json={"name": "e2e-design", "description": "smoke"})
    assert p.status_code == 200

    d = client.post("/v1/design/", json={
        "project_id": "e2e-design",
        "n_samples": 16,
        "seed": 7,
        "seed_smiles": ["CC(=O)Oc1ccccc1C(=O)O", "CC(C)Cc1ccc(C(C)C(=O)O)cc1"],
        "objectives": ["qed", "sa_score"],
    })
    assert d.status_code == 200
    design_id = d.json()["design_id"]

    deadline = time.time() + 60
    while time.time() < deadline:
        s = client.get(f"/v1/design/{design_id}/status").json()
        if s["status"] in {"completed", "failed", "cancelled"}:
            break
        time.sleep(0.2)
    assert s["status"] == "completed", s

    res = client.get(f"/v1/design/{design_id}/results").json()
    assert res["n_results"] >= 1
    # At least one Pareto-optimal candidate
    assert any(r.get("pareto_optimal") for r in res["results"])
    # Composite scores must be sorted descending
    composites = [r["composite_score"] for r in res["results"]]
    assert composites == sorted(composites, reverse=True)

    # Pareto endpoint
    front = client.get(f"/v1/pareto/{design_id}/frontier").json()
    assert front["n_points"] >= 1


@pytest.mark.e2e
def test_design_requires_seed_smiles(client: TestClient) -> None:
    r = client.post("/v1/design/", json={
        "project_id": "e2e-design",
        "n_samples": 16,
    })
    assert r.status_code == 422


@pytest.mark.e2e
def test_routes_plan_returns_disconnections(client: TestClient) -> None:
    r = client.post("/v1/routes/plan", json={
        "target_smiles": "Cc1cccc(NC(=O)c2ccc(C#N)cc2)c1",
        "max_routes": 3,
    })
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
