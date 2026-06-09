"""API Gateway orchestrator proxy tests."""
from __future__ import annotations

import json

import httpx
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("ORCHESTRATOR_SVC_URL", "http://orchestrator.test")
    from api_gateway.main import app

    with TestClient(app) as test_client:
        yield test_client


def test_api_gateway_forwards_design_to_orchestrator(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str, dict | None]] = []

    class _Response:
        status_code = 200

        def __init__(self, payload: dict) -> None:
            self._payload = payload
            self.text = json.dumps(payload)

        def json(self) -> dict:
            return self._payload

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
                {
                    "design_id": "design-1",
                    "run_id": "run-1",
                    "status": "completed",
                    "history": ["PLANNING", "GENERATING", "VALIDATING", "RETROSYN", "CRITIC"],
                    "state": {
                        "candidates": [{"canonical_smiles": "CCO"}],
                        "critic": {"total_rules": 1},
                    },
                }
            )

    monkeypatch.setattr(httpx, "AsyncClient", _Client)

    response = client.post(
        "/v1/orchestrator/design",
        json={
            "nl_input": "Design KRAS G12C inhibitors",
            "workflow_scope": "engineering",
            "n_samples": 2,
        },
    )

    assert response.status_code == 200
    assert response.json()["design_id"] == "design-1"
    assert calls == [
        (
            "POST",
            "http://orchestrator.test/v1/orchestrator/design",
            {
                "nl_input": "Design KRAS G12C inhibitors",
                "workflow_scope": "engineering",
                "n_samples": 2,
            },
        )
    ]


def test_api_gateway_forwards_design_status_to_orchestrator(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str]] = []

    class _Response:
        status_code = 200
        text = '{"design_id":"design-1","status":"completed"}'

        def json(self) -> dict:
            return {"design_id": "design-1", "status": "completed"}

    class _Client:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def get(self, url: str):
            calls.append(("GET", url))
            return _Response()

    monkeypatch.setattr(httpx, "AsyncClient", _Client)

    response = client.get("/v1/orchestrator/design-1")

    assert response.status_code == 200
    assert response.json() == {"design_id": "design-1", "status": "completed"}
    assert calls == [("GET", "http://orchestrator.test/v1/orchestrator/design-1")]


def test_static_ui_submits_runs_through_orchestrator_gateway() -> None:
    script = (
        __import__("pathlib")
        .Path("/workspace/MForge/moleculeforge/ui/public/app.js")
        .read_text(encoding="utf-8")
    )

    assert 'api("/orchestrator/design"' in script
    submit_block = script.split('$("#run").addEventListener("click"', 1)[1].split(
        "/* ---------------- run rendering ---------------- */",
        1,
    )[0]
    assert "/reason/runs" not in submit_block
