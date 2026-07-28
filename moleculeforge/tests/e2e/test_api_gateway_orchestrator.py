"""API Gateway orchestrator proxy tests."""
from __future__ import annotations

import ast
import asyncio
import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]


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
        status_code = 202

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
            "validation_passed": True,
            "max_refinements": 1,
            "n_samples": 2,
        },
    )

    assert response.status_code == 202
    assert response.json()["design_id"] == "design-1"
    assert calls == [
        (
            "POST",
            "http://orchestrator.test/v1/orchestrator/design",
            {
                "nl_input": "Design KRAS G12C inhibitors",
                "workflow_scope": "engineering",
                "validation_passed": True,
                "max_refinements": 1,
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

        async def get(self, url: str, params: dict | None = None):
            calls.append(("GET", url))
            return _Response()

    monkeypatch.setattr(httpx, "AsyncClient", _Client)

    response = client.get("/v1/orchestrator/design-1")

    assert response.status_code == 200
    assert response.json() == {"design_id": "design-1", "status": "completed"}
    assert calls == [
        ("GET", "http://orchestrator.test/v1/orchestrator/runs/design-1")
    ]


def test_static_ui_submits_runs_through_orchestrator_gateway() -> None:
    gateway_source = (
        ROOT / "services/api-gateway/src/api_gateway/main.py"
    ).read_text(encoding="utf-8")
    markup = (ROOT / "ui/public/index.html").read_text(encoding="utf-8")
    script = (ROOT / "ui/public/app.js").read_text(encoding="utf-8")

    assert "/workspace" not in gateway_source
    assert 'api("/orchestrator/design"' in script
    for field_id in (
        "workflow-scope",
        "validation-passed",
        "max-refinements",
    ):
        assert f'id="{field_id}"' in markup
    assert markup.count("required") >= 3
    submit_block = script.split('$("#run").addEventListener("click"', 1)[1].split(
        "/* ---------------- run rendering ---------------- */",
        1,
    )[0]
    assert "/reason/runs" not in submit_block
    assert 'workflow_scope: $("#workflow-scope").value' in submit_block
    assert 'validation_passed: $("#validation-passed").value === "true"' in submit_block
    assert 'max_refinements: Number($("#max-refinements").value)' in submit_block
    assert 'workflow_scope: "engineering"' not in submit_block
    assert "openRun(r.run_id" in submit_block
    assert "pollOrchestratorRun(runId, intent)" in script
    assert "live: !isTerminalRun(run.status)" in script


def test_kras_pilot_start_design_calls_supply_explicit_policy() -> None:
    source = (ROOT / "tests/e2e/test_kras_g12c_pilot.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    payloads = [
        call.args[0]
        for call in ast.walk(tree)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "start_design"
        and call.args
        and isinstance(call.args[0], ast.Dict)
    ]

    assert len(payloads) == 3
    for payload in payloads:
        keys = {
            key.value
            for key in payload.keys
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }
        assert {"validation_passed", "max_refinements"} <= keys


def test_gateway_orchestrator_error_detail_is_not_nested(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Response:
        status_code = 400
        text = '{"detail":"workflow_scope is required"}'

        def json(self) -> dict:
            return {"detail": "workflow_scope is required"}

    class _Client:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def post(self, url: str, json: dict):
            return _Response()

    monkeypatch.setattr(httpx, "AsyncClient", _Client)

    response = client.post("/v1/orchestrator/design", json={"nl_input": "intent"})

    assert response.status_code == 400
    assert response.json() == {"detail": "workflow_scope is required"}


def test_reason_history_proxies_orchestrator_run_listing(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict | None]] = []

    class _Response:
        status_code = 200
        text = '{"runs":[{"run_id":"run-1"}],"next_page_token":null}'

        def json(self) -> dict:
            return {"runs": [{"run_id": "run-1"}], "next_page_token": None}

    class _Client:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def get(self, url: str, params: dict | None = None):
            calls.append((url, params))
            return _Response()

    monkeypatch.setattr(httpx, "AsyncClient", _Client)

    response = client.get("/v1/reason/runs?page_size=20")

    assert response.status_code == 200
    assert response.json()["runs"] == [{"run_id": "run-1"}]
    assert calls == [
        (
            "http://orchestrator.test/v1/orchestrator/runs",
            {"page_size": 20, "page_token": None},
        )
    ]


def test_reason_submission_proxies_orchestrator_design(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict]] = []

    class _Response:
        status_code = 202
        text = '{"design_id":"run-1","run_id":"run-1","status":"queued"}'

        def json(self) -> dict:
            return {"design_id": "run-1", "run_id": "run-1", "status": "queued"}

    class _Client:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def post(self, url: str, json: dict):
            calls.append((url, json))
            return _Response()

    monkeypatch.setattr(httpx, "AsyncClient", _Client)

    response = client.post(
        "/v1/reason/runs",
        json={
            "intent": "Design KRAS G12C inhibitors",
            "workflow_scope": "engineering",
            "validation_passed": True,
            "max_refinements": 1,
            "project_id": "project-1",
        },
    )

    assert response.status_code == 202
    assert response.json()["run_id"] == "run-1"
    assert calls == [
        (
            "http://orchestrator.test/v1/orchestrator/design",
            {
                "intent": "Design KRAS G12C inhibitors",
                "workflow_scope": "engineering",
                "validation_passed": True,
                "max_refinements": 1,
                "project_id": "project-1",
            },
        )
    ]


def test_design_router_proxies_instead_of_creating_local_history(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict]] = []

    class _Response:
        status_code = 202
        text = '{"design_id":"run-2","run_id":"run-2","status":"queued"}'

        def json(self) -> dict:
            return {"design_id": "run-2", "run_id": "run-2", "status": "queued"}

    class _Client:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def post(self, url: str, json: dict):
            calls.append((url, json))
            return _Response()

    monkeypatch.setattr(httpx, "AsyncClient", _Client)
    payload = {
        "nl_input": "Design soluble molecules",
        "workflow_scope": "engineering",
        "validation_passed": True,
        "max_refinements": 1,
    }

    response = client.post("/v1/design/", json=payload)

    assert response.status_code == 202
    assert response.json()["design_id"] == "run-2"
    assert calls == [
        ("http://orchestrator.test/v1/orchestrator/design", payload)
    ]


def test_stream_router_emits_persisted_orchestrator_events(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    snapshot_calls = 0

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

        async def get(self, url: str, params: dict | None = None):
            nonlocal snapshot_calls
            calls.append(url)
            if url.endswith("/events"):
                if snapshot_calls > 1:
                    return _Response({"run_id": "run-1", "events": []})
                return _Response(
                    {
                        "run_id": "run-1",
                        "events": [
                            {
                                "step_index": 0,
                                "stage": "planning",
                                "payload": {"source": "persisted"},
                            }
                        ],
                    }
                )
            snapshot_calls += 1
            status = "running" if snapshot_calls == 1 else "completed"
            return _Response({"run_id": "run-1", "status": status})

    monkeypatch.setattr(httpx, "AsyncClient", _Client)

    response = client.get("/v1/stream/run-1")

    assert response.status_code == 200
    assert '"source": "persisted"' in response.text
    assert '"type": "done"' in response.text
    assert '"status": "completed"' in response.text
    assert calls == [
        "http://orchestrator.test/v1/orchestrator/runs/run-1",
        "http://orchestrator.test/v1/orchestrator/runs/run-1/events",
        "http://orchestrator.test/v1/orchestrator/runs/run-1",
        "http://orchestrator.test/v1/orchestrator/runs/run-1/events",
    ]


def test_stream_router_returns_upstream_not_found_before_starting_sse(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Response:
        status_code = 404
        text = '{"detail":"Unknown run_id: missing"}'

        def json(self) -> dict:
            return {"detail": "Unknown run_id: missing"}

    class _Client:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def get(self, url: str, params: dict | None = None):
            return _Response()

    monkeypatch.setattr(httpx, "AsyncClient", _Client)

    response = client.get("/v1/stream/missing")

    assert response.status_code == 404
    assert response.json() == {"detail": "Unknown run_id: missing"}


def test_stream_done_includes_only_terminal_error_fields(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Response:
        status_code = 200
        text = ""

        def __init__(self, payload: dict) -> None:
            self.payload = payload

        def json(self) -> dict:
            return self.payload

    class _Client:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def get(self, url: str, params: dict | None = None):
            if url.endswith("/events"):
                return _Response({"run_id": "run-failed", "events": []})
            return _Response(
                {
                    "run_id": "run-failed",
                    "status": "failed",
                    "error_type": "RuntimeError",
                    "error_message": "validation failed",
                    "state": {"secret": "must not leak"},
                }
            )

    monkeypatch.setattr(httpx, "AsyncClient", _Client)

    response = client.get("/v1/stream/run-failed")

    assert response.status_code == 200
    done = json.loads(response.text.removeprefix("data: ").strip())
    assert done == {
        "type": "done",
        "run_id": "run-failed",
        "status": "failed",
        "error_type": "RuntimeError",
        "error_message": "validation failed",
    }


def test_mvp_runner_is_an_orchestrator_client(
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

        def raise_for_status(self) -> None:
            return None

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
                {"design_id": "run-3", "run_id": "run-3", "status": "queued"}
            )

        async def get(self, url: str):
            calls.append(("GET", url, None))
            return _Response(
                {
                    "run_id": "run-3",
                    "status": "completed",
                    "state": {"candidates": [{"canonical_smiles": "CCN"}]},
                }
            )

    monkeypatch.setattr(httpx, "AsyncClient", _Client)
    monkeypatch.setenv("ORCHESTRATOR_SVC_URL", "http://orchestrator.test")
    from mvp_pipeline.runner import run_pipeline

    result = asyncio.run(
        run_pipeline(
            "Design soluble molecules",
            n_samples=4,
            seed=7,
            workflow_scope="engineering",
            validation_passed=True,
            max_refinements=1,
        )
    )

    assert result["run_id"] == "run-3"
    assert result["state"]["candidates"] == [{"canonical_smiles": "CCN"}]
    assert calls == [
        (
            "POST",
            "http://orchestrator.test/v1/orchestrator/design",
            {
                "nl_input": "Design soluble molecules",
                "n_samples": 4,
                "seed": 7,
                "workflow_scope": "engineering",
                "validation_passed": True,
                "max_refinements": 1,
            },
        ),
        (
            "GET",
            "http://orchestrator.test/v1/orchestrator/runs/run-3",
            None,
        ),
    ]


def test_pareto_routes_read_canonical_orchestrator_snapshot(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Response:
        status_code = 200
        text = ""

        def json(self) -> dict:
            return {
                "run_id": "run-pareto",
                "status": "completed",
                "state": {
                    "objectives": ["qed", "sa_score"],
                    "results": [
                        {
                            "rank": 1,
                            "canonical_smiles": "CCO",
                            "valid": True,
                            "pareto_optimal": True,
                            "qed": 0.7,
                            "sa_score": 2.0,
                            "logp": 1.0,
                            "composite_score": 0.8,
                        }
                    ],
                },
            }

    class _Client:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def get(self, url: str, params: dict | None = None):
            return _Response()

    monkeypatch.setattr(httpx, "AsyncClient", _Client)

    frontier = client.get("/v1/pareto/run-pareto/frontier")
    hypervolume = client.get("/v1/pareto/run-pareto/hypervolume")
    missing_weights = client.post("/v1/pareto/run-pareto/select", json={})
    selected = client.post(
        "/v1/pareto/run-pareto/select",
        json={"weights": {"qed": 1.0}, "top_k": 1},
    )

    assert frontier.status_code == 200
    assert frontier.json()["frontier"][0]["smiles"] == "CCO"
    assert hypervolume.status_code == 200
    assert hypervolume.json()["n_points"] == 1
    assert missing_weights.status_code == 400
    assert missing_weights.json() == {"detail": "weights is required"}
    assert selected.status_code == 200
    assert selected.json()["selected"][0]["smiles"] == "CCO"


def test_pareto_merges_production_candidates_and_validation_results(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Response:
        status_code = 200
        text = ""

        def json(self) -> dict:
            return {
                "run_id": "run-production-shape",
                "status": "completed",
                "state": {
                    "candidates": [
                        {
                            "candidate_id": "candidate-1",
                            "canonical_smiles": "CCO",
                            "properties": {"qed": 0.1},
                        },
                        {
                            "candidate_id": "candidate-2",
                            "canonical_smiles": "CCN",
                        },
                    ],
                    "validation": {
                        "results": [
                            {
                                "candidate_id": "candidate-1",
                                "valid": True,
                                "pareto_optimal": True,
                                "properties": {
                                    "qed": 0.8,
                                    "sa_score": 2.0,
                                    "logp": 1.2,
                                },
                            },
                            {
                                "canonical_smiles": "CCN",
                                "valid": True,
                                "properties": {
                                    "qed": 0.6,
                                    "sa_score": 3.0,
                                    "logp": 1.5,
                                },
                            },
                        ]
                    },
                },
            }

    class _Client:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def get(self, url: str, params: dict | None = None):
            return _Response()

    monkeypatch.setattr(httpx, "AsyncClient", _Client)

    frontier = client.get("/v1/pareto/run-production-shape/frontier")
    hypervolume = client.get("/v1/pareto/run-production-shape/hypervolume")
    selected = client.post(
        "/v1/pareto/run-production-shape/select",
        json={"weights": {"qed": 1.0}, "top_k": 2},
    )

    assert frontier.status_code == 200
    assert frontier.json()["frontier"][0]["objectives"]["qed"] == 0.8
    assert hypervolume.status_code == 200
    assert hypervolume.json()["n_points"] == 2
    assert hypervolume.json()["hypervolume"] > 0
    assert selected.status_code == 200
    assert [row["smiles"] for row in selected.json()["selected"]] == ["CCO", "CCN"]
    assert selected.json()["selected"][0]["qed"] == 0.8


def test_legacy_reasoning_pipeline_only_proxies_canonical_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from orchestrator.pipeline import ReasoningPipeline

    calls: list[tuple[str, dict]] = []

    class _Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"run_id": "run-proxy", "status": "queued"}

    class _Client:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def post(self, url: str, json: dict):
            calls.append((url, json))
            return _Response()

    monkeypatch.setattr(httpx, "AsyncClient", _Client)
    pipeline = ReasoningPipeline("http://orchestrator.test")

    run_id = asyncio.run(
        pipeline.submit(
            "Design soluble molecules",
            workflow_scope="engineering",
            validation_passed=True,
            max_refinements=1,
        )
    )

    assert run_id == "run-proxy"
    assert not hasattr(pipeline, "_runs")
    assert calls == [
        (
            "http://orchestrator.test/v1/orchestrator/design",
            {
                "intent": "Design soluble molecules",
                "workflow_scope": "engineering",
                "validation_passed": True,
                "max_refinements": 1,
                "project_id": None,
            },
        )
    ]
