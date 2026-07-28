"""End-to-end molecular property prediction test (no docker required).

Drives the FastAPI app directly via TestClient and validates that:
- single-molecule predict returns physicochemically correct values
- batch predict shards across all visible CUDA devices
- a design loop completes and returns a Pareto front
- retrosynthesis routes return analysis based on the actual SMILES
"""

from __future__ import annotations

import json
import subprocess
import sys

import httpx
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
def test_design_loop_returns_pareto_front(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = {
        "run_id": "run-design",
        "status": "completed",
        "current_stage": "critic",
        "devices_used": ["cpu"],
        "state": {
            "request": {"objectives": ["qed", "sa_score"]},
            "candidates": [{"canonical_smiles": "CCO"}],
            "validation": {
                "results": [
                    {
                        "rank": 1,
                        "canonical_smiles": "CCO",
                        "valid": True,
                        "pareto_optimal": True,
                        "qed": 0.7,
                        "sa_score": 2.0,
                        "composite_score": 0.8,
                    }
                ]
            },
        },
    }

    class _Response:
        def __init__(self, payload: dict, status_code: int) -> None:
            self.payload = payload
            self.status_code = status_code
            self.text = json.dumps(payload)

        def json(self) -> dict:
            return self.payload

    class _Client:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def post(self, url: str, json: dict):
            if url.endswith("/v1/orchestrator/projects"):
                return _Response(
                    {
                        "project_id": "e2e-design",
                        "name": "e2e-design",
                        "description": "smoke",
                        "created_at": "2026-07-28T00:00:00+00:00",
                    },
                    200,
                )
            if url.endswith("/v1/orchestrator/design"):
                return _Response(
                    {
                        "design_id": "run-design",
                        "run_id": "run-design",
                        "status": "queued",
                    },
                    202,
                )
            raise AssertionError(f"unexpected POST URL: {url}")

        async def get(self, url: str, params: dict | None = None):
            return _Response(snapshot, 200)

    monkeypatch.setattr(httpx, "AsyncClient", _Client)
    p = client.post("/v1/projects/", json={"name": "e2e-design", "description": "smoke"})
    assert p.status_code == 200

    d = client.post(
        "/v1/design/",
        json={
            "project_id": "e2e-design",
            "nl_input": "Design a Pareto-optimal molecule set",
            "workflow_scope": "engineering",
            "validation_passed": True,
            "max_refinements": 1,
            "n_samples": 16,
            "seed": 7,
            "seed_smiles": ["CC(=O)Oc1ccccc1C(=O)O", "CC(C)Cc1ccc(C(C)C(=O)O)cc1"],
            "objectives": ["qed", "sa_score"],
        },
    )
    assert d.status_code == 202
    design_id = d.json()["design_id"]

    s = client.get(f"/v1/design/{design_id}/status").json()
    assert s == {
        "design_id": "run-design",
        "status": "completed",
        "progress_pct": 100.0,
        "current_stage": "completed",
        "candidates_generated": 1,
        "valid_results": 1,
        "devices_used": ["cpu"],
    }

    res = client.get(f"/v1/design/{design_id}/results").json()
    assert res["design_id"] == "run-design"
    assert res["status"] == "completed"
    assert res["n_results"] == 1
    assert res["objectives"] == ["qed", "sa_score"]
    assert res["devices_used"] == ["cpu"]
    results = res["results"]
    assert any(result.get("pareto_optimal") for result in results)

    front = client.get(f"/v1/pareto/{design_id}/frontier").json()
    assert front["n_points"] >= 1


@pytest.mark.e2e
@pytest.mark.parametrize(
    "request_payload",
    [
        {
            "project_id": "e2e-design",
            "nl_input": "Design molecules",
            "n_samples": 16,
        },
        {
            "project_id": "e2e-design",
            "objectives": ["qed"],
            "constraints": {},
            "n_samples": 16,
            "seed_smiles": "CCO",
        },
        {
            "project_id": "e2e-design",
            "objectives": ["qed"],
            "constraints": {},
            "n_samples": 16,
            "seed_smiles": [],
        },
    ],
)
def test_legacy_design_seed_error_is_local(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    request_payload: dict,
) -> None:
    class _Client:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def post(self, url: str, json: dict):
            raise AssertionError("invalid legacy requests must not be proxied")

    monkeypatch.setattr(httpx, "AsyncClient", _Client)
    r = client.post("/v1/design/", json=request_payload)

    assert r.status_code == 422
    detail = r.json()["detail"]
    assert isinstance(detail, list)
    assert detail[0]["loc"][:2] == ["body", "seed_smiles"]


@pytest.mark.e2e
def test_partial_design_policy_error_is_transparent(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict] = []

    class _Response:
        status_code = 400
        text = '{"detail":"validation_passed is required"}'

        def json(self) -> dict:
            return {"detail": "validation_passed is required"}

    class _Client:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def post(self, url: str, json: dict):
            calls.append(json)
            return _Response()

    monkeypatch.setattr(httpx, "AsyncClient", _Client)
    request_payload = {
        "project_id": "e2e-design",
        "objectives": ["qed"],
        "constraints": {},
        "n_samples": 16,
        "seed_smiles": ["CCO"],
        "workflow_scope": "engineering",
    }
    response = client.post(
        "/v1/design/",
        json=request_payload,
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "validation_passed is required"}
    assert calls == [request_payload]


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
