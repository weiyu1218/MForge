"""End-to-end tests for the canonical reasoning workbench routes."""
from __future__ import annotations

import json

import httpx
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client() -> TestClient:
    from api_gateway.main import app

    with TestClient(app) as test_client:
        yield test_client


def _install_orchestrator(
    monkeypatch: pytest.MonkeyPatch,
    snapshot: dict,
) -> list[tuple[str, str, dict | None]]:
    calls: list[tuple[str, str, dict | None]] = []

    class _Response:
        status_code = 200

        def __init__(self, payload: dict, status_code: int = 200) -> None:
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
            calls.append(("POST", url, json))
            return _Response(
                {"design_id": "run-reason", "run_id": "run-reason", "status": "queued"},
                202,
            )

        async def get(self, url: str, params: dict | None = None):
            calls.append(("GET", url, params))
            if url.endswith("/runs"):
                return _Response({"runs": [snapshot], "next_page_token": None})
            return _Response(snapshot)

    monkeypatch.setattr(httpx, "AsyncClient", _Client)
    return calls


@pytest.mark.e2e
def test_reason_run_uses_canonical_orchestrator(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = {
        "run_id": "run-reason",
        "intent": "Design KRAS G12C inhibitors",
        "status": "completed",
        "current_stage": "critic",
        "state": {
            "history": ["PLANNING", "GENERATING", "VALIDATING", "CRITIC"],
            "candidates": [{"canonical_smiles": "CCO"}],
        },
    }
    calls = _install_orchestrator(monkeypatch, snapshot)
    request = {
        "intent": "Design KRAS G12C inhibitors",
        "workflow_scope": "engineering",
        "validation_passed": True,
        "max_refinements": 1,
    }

    submitted = client.post("/v1/reason/runs", json=request)
    current = client.get("/v1/reason/runs/run-reason")

    assert submitted.status_code == 202
    assert submitted.json()["status"] == "queued"
    assert current.status_code == 200
    assert current.json() == snapshot
    assert calls[0][2] == {**request, "project_id": None}


@pytest.mark.e2e
def test_reason_snapshot_preserves_known_molecule_result(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = {
        "run_id": "run-reason",
        "status": "completed",
        "state": {
            "results": [
                {
                    "canonical_smiles": "CC(=O)Oc1ccccc1C(=O)O",
                    "is_novel": False,
                    "known_match": {"name": "Aspirin"},
                }
            ]
        },
    }
    _install_orchestrator(monkeypatch, snapshot)

    response = client.get("/v1/reason/runs/run-reason")

    assert response.status_code == 200
    result = response.json()["state"]["results"][0]
    assert result["known_match"]["name"] == "Aspirin"


@pytest.mark.e2e
def test_reason_history_includes_recent(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = {
        "run_id": "run-reason",
        "intent": "Design KRAS G12C inhibitors",
        "status": "completed",
    }
    _install_orchestrator(monkeypatch, snapshot)

    response = client.get("/v1/reason/runs?page_size=10")

    assert response.status_code == 200
    assert response.json()["runs"] == [snapshot]


@pytest.mark.e2e
def test_known_catalog_lookup(client: TestClient) -> None:
    response = client.get("/v1/reason/known?query=aspirin")

    assert response.status_code == 200
    assert any(item["name"] == "Aspirin" for item in response.json()["items"])


@pytest.mark.e2e
def test_static_workbench_is_served(client: TestClient) -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert "Reasoning Workbench" in response.text
    assert client.get("/styles.css").status_code == 200
    assert client.get("/app.js").status_code == 200
