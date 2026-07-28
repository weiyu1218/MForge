"""API Gateway orchestrator proxy tests."""

from __future__ import annotations

import ast
import asyncio
import json
import sqlite3
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
    assert calls == [("GET", "http://orchestrator.test/v1/orchestrator/runs/design-1")]


def test_static_ui_submits_runs_through_orchestrator_gateway() -> None:
    gateway_source = (ROOT / "services/api-gateway/src/api_gateway/main.py").read_text(
        encoding="utf-8"
    )
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
    source = (ROOT / "tests/e2e/test_kras_g12c_pilot.py").read_text(encoding="utf-8")
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


def test_design_router_canonical_request_cannot_forge_legacy_source_marker(
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
        "_mforge_internal_legacy_design_request": True,
    }

    response = client.post("/v1/design/", json=payload)

    assert response.status_code == 202
    assert response.json()["design_id"] == "run-2"
    assert calls == [
        (
            "http://orchestrator.test/v1/orchestrator/design",
            {
                "nl_input": "Design soluble molecules",
                "workflow_scope": "engineering",
                "validation_passed": True,
                "max_refinements": 1,
            },
        )
    ]


def test_design_router_translates_legacy_seed_request_without_dropping_inputs(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict]] = []

    class _Response:
        status_code = 202

        def json(self) -> dict:
            return {"design_id": "run-legacy", "run_id": "run-legacy", "status": "queued"}

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
    legacy_payload = {
        "project_id": "project-legacy",
        "objectives": ["qed", "sa_score"],
        "constraints": {"molecular_weight": {"max": 500}},
        "n_samples": 12,
        "seed_smiles": ["CCO", "CCN"],
        "seed": 17,
    }

    response = client.post("/v1/design/", json=legacy_payload)

    assert response.status_code == 202
    assert calls == [
        (
            "http://orchestrator.test/v1/orchestrator/design",
            {
                **legacy_payload,
                "intent": (
                    "Legacy molecular design: "
                    '{"constraints":{"molecular_weight":{"max":500}},'
                    '"objectives":["qed","sa_score"]}'
                ),
                "workflow_scope": "engineering",
                "validation_passed": True,
                "max_refinements": 0,
                "_mforge_internal_legacy_design_request": True,
            },
        )
    ]


def test_design_router_restores_legacy_request_defaults(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict] = []

    class _Response:
        status_code = 202

        def json(self) -> dict:
            return {"design_id": "run-defaults", "run_id": "run-defaults", "status": "queued"}

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

    response = client.post(
        "/v1/design/",
        json={"seed_smiles": ["CCO"], "seed": None},
    )

    assert response.status_code == 202
    assert calls == [
        {
            "objectives": ["qed", "sa_score", "logp"],
            "constraints": {},
            "n_samples": 64,
            "seed_smiles": ["CCO"],
            "seed": None,
            "intent": (
                'Legacy molecular design: {"constraints":{},"objectives":["qed","sa_score","logp"]}'
            ),
            "workflow_scope": "engineering",
            "validation_passed": True,
            "max_refinements": 0,
            "_mforge_internal_legacy_design_request": True,
        }
    ]


def test_legacy_request_model_materializes_empty_project_id() -> None:
    from api_gateway.routers.design import DesignRequest

    request = DesignRequest.model_validate({"seed_smiles": ["CCO"]})

    assert request.project_id == ""


@pytest.mark.parametrize("project_id", [None, "project-explicit"])
def test_legacy_project_id_crosses_real_orchestrator_store_boundary(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    project_id: str | None,
) -> None:
    from mf_core.db.store import RunStore
    from orchestrator_svc import main as orchestrator_main
    from orchestrator_svc.main import RunControl

    store = RunStore(tmp_path / "gateway-runs.db")
    asyncio.run(store.initialize())
    if project_id is not None:
        with sqlite3.connect(store.path) as connection:
            connection.execute(
                """
                INSERT INTO projects (project_id, name, description, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (project_id, project_id, "", "2026-07-28T00:00:00+00:00"),
            )
    monkeypatch.setattr(orchestrator_main, "_RUN_STORE", store)
    monkeypatch.setattr(orchestrator_main, "_RUN_CONTROL", RunControl(store))
    monkeypatch.setattr(orchestrator_main, "_RUN_INITIALIZED_STORE", store)
    monkeypatch.setattr(orchestrator_main, "_RUNTIME_INIT_LOCK", None)
    monkeypatch.setattr(orchestrator_main, "_RUN_TASKS", {})
    monkeypatch.setattr(
        orchestrator_main,
        "_register_design_run_task",
        lambda run_id, request, initial_state, **kwargs: None,
    )
    real_async_client = httpx.AsyncClient

    class _Client:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def post(self, url: str, json: dict):
            transport = httpx.ASGITransport(app=orchestrator_main.rest_app)
            async with real_async_client(
                transport=transport,
                base_url="http://orchestrator.test",
            ) as upstream:
                return await upstream.post("/v1/orchestrator/design", json=json)

    monkeypatch.setattr(httpx, "AsyncClient", _Client)
    request_payload = {"seed_smiles": ["CCO"]}
    if project_id is not None:
        request_payload["project_id"] = project_id

    response = client.post("/v1/design/", json=request_payload)

    assert response.status_code == 202
    snapshot = asyncio.run(store.get_run(response.json()["run_id"]))
    assert snapshot is not None
    assert snapshot["project_id"] == project_id


@pytest.mark.parametrize(
    ("project_id", "external_project_id"),
    [
        ("shared-project", "shared-project"),
        ("space name", "space%20name"),
        ("R&D #1", "R%26D%20%231"),
    ],
)
def test_project_routes_proxy_to_independent_orchestrator_store(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    project_id: str,
    external_project_id: str,
) -> None:
    from api_gateway.main import app
    from mf_core.db.store import RunStore
    from orchestrator_svc import main as orchestrator_main
    from orchestrator_svc.main import RunControl

    gateway_database_path = tmp_path / "gateway.db"
    orchestrator_database_path = tmp_path / "orchestrator.db"
    monkeypatch.setenv("MF_DB_PATH", str(gateway_database_path))
    monkeypatch.setenv("ORCHESTRATOR_SVC_URL", "http://orchestrator.test")
    store = RunStore(orchestrator_database_path)
    asyncio.run(store.initialize())
    monkeypatch.setattr(orchestrator_main, "_RUN_STORE", store)
    monkeypatch.setattr(orchestrator_main, "_RUN_CONTROL", RunControl(store))
    monkeypatch.setattr(orchestrator_main, "_RUN_INITIALIZED_STORE", store)
    monkeypatch.setattr(orchestrator_main, "_RUNTIME_INIT_LOCK", None)
    monkeypatch.setattr(orchestrator_main, "_RUN_TASKS", {})
    monkeypatch.setattr(
        orchestrator_main,
        "_register_design_run_task",
        lambda run_id, request, initial_state, **kwargs: None,
    )
    real_async_client = httpx.AsyncClient

    class _Client:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

        async def __aenter__(self) -> _Client:
            return self

        async def __aexit__(
            self,
            exc_type: object,
            exc: object,
            tb: object,
        ) -> None:
            return None

        async def get(
            self,
            url: str,
            params: dict | None = None,
        ) -> httpx.Response:
            return await self._request("GET", url, params=params)

        async def post(self, url: str, json: dict) -> httpx.Response:
            return await self._request("POST", url, json=json)

        async def delete(self, url: str) -> httpx.Response:
            return await self._request("DELETE", url)

        async def _request(
            self,
            method: str,
            url: str,
            *,
            json: dict | None = None,
            params: dict | None = None,
        ) -> httpx.Response:
            transport = httpx.ASGITransport(app=orchestrator_main.rest_app)
            async with real_async_client(
                transport=transport,
                base_url="http://orchestrator.test",
            ) as upstream:
                path = url.removeprefix("http://orchestrator.test")
                return await upstream.request(method, path, json=json, params=params)

    monkeypatch.setattr(httpx, "AsyncClient", _Client)

    with TestClient(app) as gateway:
        created = gateway.post(
            "/v1/projects/",
            json={"name": project_id, "description": "first"},
        )
        assert created.status_code == 200
        created_at = created.json()["created_at"]
        assert created.json() == {
            "project_id": project_id,
            "name": project_id,
            "description": "first",
            "status": "active",
            "created_at": created_at,
            "designs": [],
        }

        orchestrator_project = asyncio.run(store.get_project(project_id))
        assert orchestrator_project == {
            "project_id": project_id,
            "name": project_id,
            "description": "first",
            "created_at": created_at,
        }
        gateway_store = RunStore(gateway_database_path)
        assert asyncio.run(gateway_store.get_project(project_id)) is None

        design = gateway.post(
            "/v1/design/",
            json={
                "project_id": project_id,
                "seed_smiles": ["CCO"],
            },
        )
        assert design.status_code == 202
        run_id = design.json()["run_id"]
        snapshot = asyncio.run(store.get_run(run_id))
        assert snapshot is not None
        assert snapshot["project_id"] == project_id

        updated = gateway.post(
            "/v1/projects/",
            json={"name": project_id, "description": "updated"},
        )
        assert updated.json() == {
            **created.json(),
            "description": "updated",
        }
        fetched = gateway.get(f"/v1/projects/{external_project_id}")
        listed = gateway.get("/v1/projects/")
        deleted = gateway.delete(f"/v1/projects/{external_project_id}")
        missing_get = gateway.get(f"/v1/projects/{external_project_id}")
        missing_delete = gateway.delete(f"/v1/projects/{external_project_id}")

        assert fetched.status_code == 200
        assert fetched.json() == updated.json()
        assert listed.json() == {
            "projects": [updated.json()],
            "n_projects": 1,
        }
        assert deleted.status_code == 200
        assert deleted.json() == {
            "deleted": True,
            "project_id": project_id,
        }
        assert missing_get.status_code == 404
        assert missing_delete.status_code == 404

    snapshot = asyncio.run(store.get_run(run_id))
    assert snapshot is not None
    assert snapshot["project_id"] is None
    assert asyncio.run(store.list_projects()) == []


@pytest.mark.parametrize(
    ("legacy_payload", "expected_n_samples", "expected_seed", "expected_smiles"),
    [
        ({"n_samples": "12"}, 12, None, ["CCO"]),
        ({"n_samples": 12.0}, 12, None, ["CCO"]),
        ({"n_samples": True}, 1, None, ["CCO"]),
        ({"seed": "7"}, 64, 7, ["CCO"]),
        ({"seed": 7.0}, 64, 7, ["CCO"]),
        ({"seed": True}, 64, 1, ["CCO"]),
        ({}, 64, None, [""]),
    ],
)
def test_design_router_preserves_legacy_pydantic_coercion(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    legacy_payload: dict,
    expected_n_samples: int,
    expected_seed: int | None,
    expected_smiles: list[str],
) -> None:
    calls: list[dict] = []

    class _Response:
        status_code = 202

        def json(self) -> dict:
            return {"design_id": "run-coerced", "run_id": "run-coerced", "status": "queued"}

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
        "seed_smiles": expected_smiles,
        "nl_input": "ignored by the legacy request model",
        **legacy_payload,
    }

    response = client.post("/v1/design/", json=request_payload)

    assert response.status_code == 202
    assert calls == [
        {
            "objectives": ["qed", "sa_score", "logp"],
            "constraints": {},
            "n_samples": expected_n_samples,
            "seed_smiles": expected_smiles,
            "seed": expected_seed,
            "intent": (
                'Legacy molecular design: {"constraints":{},"objectives":["qed","sa_score","logp"]}'
            ),
            "workflow_scope": "engineering",
            "validation_passed": True,
            "max_refinements": 0,
            "_mforge_internal_legacy_design_request": True,
        }
    ]


@pytest.mark.parametrize(
    ("legacy_payload", "field"),
    [
        ({"seed_smiles": [7]}, "seed_smiles"),
        ({"seed_smiles": ["CCO"], "objectives": "qed"}, "objectives"),
        ({"seed_smiles": ["CCO"], "objectives": ["qed", 7]}, "objectives"),
        ({"seed_smiles": ["CCO"], "constraints": []}, "constraints"),
        ({"seed_smiles": ["CCO"], "seed": 7.5}, "seed"),
        ({"seed_smiles": ["CCO"], "seed": "not-an-int"}, "seed"),
        ({"seed_smiles": ["CCO"], "n_samples": 0}, "n_samples"),
        ({"seed_smiles": ["CCO"], "n_samples": 2049}, "n_samples"),
        ({"seed_smiles": ["CCO"], "n_samples": 12.5}, "n_samples"),
        ({"seed_smiles": ["CCO"], "n_samples": None}, "n_samples"),
    ],
)
def test_design_router_rejects_values_rejected_by_legacy_pydantic_model(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    legacy_payload: dict,
    field: str,
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

    response = client.post("/v1/design/", json=legacy_payload)

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert isinstance(detail, list)
    assert detail[0]["loc"][:2] == ["body", field]


@pytest.mark.parametrize(
    ("run_id", "snapshot", "expected"),
    [
        (
            "run-queued",
            {
                "run_id": "canonical-run-queued",
                "status": "queued",
                "current_stage": "queued",
                "devices_used": [],
                "progress_pct": 99.0,
                "state": {},
            },
            {
                "design_id": "run-queued",
                "status": "queued",
                "progress_pct": 5.0,
                "current_stage": "queued",
                "candidates_generated": 0,
                "valid_results": 0,
                "devices_used": [],
            },
        ),
        (
            "run-running-empty",
            {
                "run_id": "run-running-empty",
                "status": "running",
                "current_stage": "planning",
                "devices_used": [],
                "progress_pct": 99.0,
                "state": {},
            },
            {
                "design_id": "run-running-empty",
                "status": "running",
                "progress_pct": 20.0,
                "current_stage": "running",
                "candidates_generated": 0,
                "valid_results": 0,
                "devices_used": [],
            },
        ),
        (
            "run-running",
            {
                "run_id": "canonical-run-running",
                "status": "running",
                "current_stage": "validating",
                "devices_used": ["cuda:0"],
                "state": {
                    "progress_pct": 99.0,
                    "candidates": [
                        {"canonical_smiles": "CCO"},
                        {"canonical_smiles": "CCN"},
                    ],
                    "validation": {
                        "results": [
                            {"canonical_smiles": "CCO", "valid": True},
                            {
                                "canonical_smiles": "CCN",
                                "valid": False,
                                "overall_passed": True,
                            },
                        ]
                    },
                },
            },
            {
                "design_id": "run-running",
                "status": "running",
                "progress_pct": 60.0,
                "current_stage": "running",
                "candidates_generated": 2,
                "valid_results": 1,
                "devices_used": ["cuda:0"],
            },
        ),
        (
            "run-failed",
            {
                "run_id": "run-failed",
                "status": "failed",
                "current_stage": "generating",
                "devices_used": [],
                "state": {"progress_pct": 99.0},
                "error_type": "RuntimeError",
                "error_message": "generation failed",
            },
            {
                "design_id": "run-failed",
                "status": "failed",
                "progress_pct": 0.0,
                "current_stage": "failed",
                "candidates_generated": 0,
                "valid_results": 0,
                "devices_used": [],
            },
        ),
        (
            "run-interrupted",
            {
                "run_id": "run-interrupted",
                "status": "interrupted",
                "current_stage": "validating",
                "devices_used": [],
                "state": {},
            },
            {
                "design_id": "run-interrupted",
                "status": "interrupted",
                "progress_pct": 5.0,
                "current_stage": "interrupted",
                "candidates_generated": 0,
                "valid_results": 0,
                "devices_used": [],
            },
        ),
        (
            "run-completed",
            {
                "run_id": "run-completed",
                "status": "completed",
                "current_stage": "critic",
                "devices_used": ["cpu"],
                "state": {
                    "progress_pct": 1.0,
                    "candidates": [{"canonical_smiles": "CCC"}],
                    "validation": {"results": [{"canonical_smiles": "CCC", "valid": True}]},
                },
            },
            {
                "design_id": "run-completed",
                "status": "completed",
                "progress_pct": 100.0,
                "current_stage": "completed",
                "candidates_generated": 1,
                "valid_results": 1,
                "devices_used": ["cpu"],
            },
        ),
    ],
)
def test_design_status_reconstructs_legacy_shape_from_persisted_snapshot(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    run_id: str,
    snapshot: dict,
    expected: dict,
) -> None:
    class _Response:
        status_code = 200

        def json(self) -> dict:
            return snapshot

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

    response = client.get(f"/v1/design/{run_id}/status")

    assert response.status_code == 200
    assert response.json() == expected


@pytest.mark.parametrize(
    ("run_id", "snapshot", "expected"),
    [
        (
            "run-running",
            {
                "run_id": "run-running",
                "status": "running",
                "devices_used": ["cuda:0"],
                "state": {
                    "request": {"objectives": ["qed"]},
                    "candidates": [{"canonical_smiles": "CCO"}],
                },
            },
            {
                "design_id": "run-running",
                "status": "running",
                "results": [],
            },
        ),
        (
            "run-failed",
            {
                "run_id": "run-failed",
                "status": "failed",
                "devices_used": [],
                "state": {
                    "request": {"objectives": ["qed", "sa_score"]},
                },
            },
            {
                "design_id": "run-failed",
                "status": "failed",
                "results": [],
            },
        ),
        (
            "run-completed",
            {
                "run_id": "run-completed",
                "status": "completed",
                "devices_used": ["cpu"],
                "state": {
                    "request": {"objectives": ["qed"]},
                    "validation": {
                        "results": [
                            {
                                "canonical_smiles": "CCO",
                                "valid": True,
                                "qed": 0.7,
                            }
                        ]
                    },
                },
            },
            {
                "design_id": "run-completed",
                "status": "completed",
                "results": [
                    {
                        "canonical_smiles": "CCO",
                        "valid": True,
                        "qed": 0.7,
                    }
                ],
                "n_results": 1,
                "objectives": ["qed"],
                "devices_used": ["cpu"],
            },
        ),
    ],
)
def test_design_results_reconstructs_legacy_shape_from_persisted_snapshot(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    run_id: str,
    snapshot: dict,
    expected: dict,
) -> None:
    class _Response:
        status_code = 200

        def json(self) -> dict:
            return snapshot

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

    response = client.get(f"/v1/design/{run_id}/results")

    assert response.status_code == 200
    assert response.json() == expected


@pytest.mark.parametrize(
    ("upstream_status", "payload"),
    [
        (200, {"run_id": "run-1", "status": "interrupted"}),
        (404, {"detail": "Unknown run_id: missing"}),
        (409, {"detail": "run run-1 cannot cancel from status completed"}),
    ],
)
def test_design_cancel_proxies_orchestrator_status_and_error(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    upstream_status: int,
    payload: dict,
) -> None:
    calls: list[tuple[str, dict]] = []

    class _Response:
        def __init__(self) -> None:
            self.status_code = upstream_status

        def json(self) -> dict:
            return payload

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

    response = client.post("/v1/design/run-1/cancel")

    assert response.status_code == upstream_status
    assert response.json() == payload
    assert calls == [
        (
            "http://orchestrator.test/v1/orchestrator/runs/run-1/cancel",
            {},
        )
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
            return _Response({"design_id": "run-3", "run_id": "run-3", "status": "queued"})

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


def test_pareto_merges_repeated_smiles_by_occurrence(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Response:
        status_code = 200

        def json(self) -> dict:
            return {
                "run_id": "run-repeated-smiles",
                "status": "completed",
                "state": {
                    "candidates": [
                        {"canonical_smiles": "CCO"},
                        {"canonical_smiles": "CCN"},
                        {"canonical_smiles": "CCO"},
                    ],
                    "validation": {
                        "results": [
                            {
                                "canonical_smiles": "CCO",
                                "rank": 1,
                                "valid": True,
                                "pareto_optimal": True,
                                "properties": {"qed": 0.9, "sa_score": 2.0},
                            },
                            {
                                "canonical_smiles": "CCN",
                                "rank": 2,
                                "valid": True,
                                "pareto_optimal": False,
                                "properties": {"qed": 0.8, "sa_score": 2.5},
                            },
                            {
                                "canonical_smiles": "CCO",
                                "rank": 3,
                                "valid": True,
                                "pareto_optimal": True,
                                "properties": {"qed": 0.7, "sa_score": 3.0},
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

    frontier = client.get("/v1/pareto/run-repeated-smiles/frontier")
    hypervolume = client.get("/v1/pareto/run-repeated-smiles/hypervolume")
    selected = client.post(
        "/v1/pareto/run-repeated-smiles/select",
        json={"weights": {"qed": 1.0}, "top_k": 3},
    )

    assert frontier.status_code == 200
    assert frontier.json()["n_points"] == 2
    assert [row["smiles"] for row in frontier.json()["frontier"]] == ["CCO", "CCO"]
    assert [row["rank"] for row in frontier.json()["frontier"]] == [1, 3]
    assert [row["objectives"]["qed"] for row in frontier.json()["frontier"]] == [0.9, 0.7]
    assert hypervolume.status_code == 200
    assert hypervolume.json()["n_points"] == 3
    assert selected.status_code == 200
    assert [row["smiles"] for row in selected.json()["selected"]] == ["CCO", "CCN", "CCO"]
    assert [row["qed"] for row in selected.json()["selected"]] == [0.9, 0.8, 0.7]


@pytest.mark.parametrize(
    "validation_rows",
    [
        [
            {
                "canonical_smiles": "CCO",
                "rank": 2,
                "pareto_optimal": False,
            },
            {
                "candidate_id": "candidate-1",
                "canonical_smiles": "CCO",
                "rank": 1,
                "pareto_optimal": True,
            },
        ],
        [
            {
                "candidate_id": "candidate-1",
                "canonical_smiles": "CCO",
                "rank": 1,
                "pareto_optimal": True,
            },
            {
                "canonical_smiles": "CCO",
                "rank": 2,
                "pareto_optimal": False,
            },
        ],
        [
            {
                "candidate_id": "unknown-candidate",
                "canonical_smiles": "CCO",
                "rank": 2,
                "pareto_optimal": False,
            },
            {
                "candidate_id": "candidate-1",
                "canonical_smiles": "CCO",
                "rank": 1,
                "pareto_optimal": True,
            },
        ],
    ],
)
def test_pareto_reserves_explicit_ids_before_smiles_fallback(
    validation_rows: list[dict],
) -> None:
    from api_gateway.routers.pareto import _merge_candidate_results

    merged = _merge_candidate_results(
        [
            {"candidate_id": "candidate-1", "canonical_smiles": "CCO"},
            {"candidate_id": "candidate-2", "canonical_smiles": "CCO"},
        ],
        validation_rows,
    )

    assert [
        (row["candidate_id"], row.get("rank"), row.get("pareto_optimal")) for row in merged
    ] == [
        ("candidate-1", 1, True),
        ("candidate-2", 2, False),
    ]


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


def test_reasoning_pipeline_unsubscribe_cancels_and_awaits_polling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from orchestrator.pipeline import ReasoningPipeline

    async def scenario() -> None:
        pipeline = ReasoningPipeline("http://orchestrator.test")
        polling_started = asyncio.Event()
        polling_stopped = asyncio.Event()

        async def blocking_get(path: str, *, params: dict | None = None) -> dict:
            polling_started.set()
            try:
                await asyncio.Future()
            finally:
                polling_stopped.set()

        monkeypatch.setattr(pipeline, "_get", blocking_get)
        queue = await pipeline.subscribe("run-blocked")
        await polling_started.wait()
        task = next(
            task
            for task in asyncio.all_tasks()
            if task.get_name() == "orchestrator-subscription-run-blocked"
        )
        try:
            await pipeline.unsubscribe("run-blocked")
            assert task.cancelled()
            assert polling_stopped.is_set()
            assert "run-blocked" not in pipeline._subscription_tasks
            assert await asyncio.wait_for(queue.get(), timeout=0.1) == {
                "type": "done",
                "run_id": "run-blocked",
            }
            await pipeline.unsubscribe("run-blocked")
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())


def test_reasoning_pipeline_replacing_subscription_finishes_previous_consumer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from orchestrator.pipeline import ReasoningPipeline

    async def scenario() -> None:
        pipeline = ReasoningPipeline("http://orchestrator.test")
        first_poll_started = asyncio.Event()
        poll_count = 0

        async def blocking_get(path: str, *, params: dict | None = None) -> dict:
            nonlocal poll_count
            poll_count += 1
            if poll_count == 1:
                first_poll_started.set()
            await asyncio.Future()

        monkeypatch.setattr(pipeline, "_get", blocking_get)
        first_queue = await pipeline.subscribe("run-replaced")
        await first_poll_started.wait()
        second_queue = await pipeline.subscribe("run-replaced")
        try:
            assert await asyncio.wait_for(first_queue.get(), timeout=0.1) == {
                "type": "done",
                "run_id": "run-replaced",
            }
        finally:
            await pipeline.unsubscribe("run-replaced")
        assert await asyncio.wait_for(second_queue.get(), timeout=0.1) == {
            "type": "done",
            "run_id": "run-replaced",
        }

    asyncio.run(scenario())


def test_reasoning_pipeline_aclose_releases_every_subscription(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from orchestrator.pipeline import ReasoningPipeline

    async def scenario() -> None:
        pipeline = ReasoningPipeline("http://orchestrator.test")
        both_started = asyncio.Event()
        started = 0

        async def blocking_get(path: str, *, params: dict | None = None) -> dict:
            nonlocal started
            started += 1
            if started == 2:
                both_started.set()
            await asyncio.Future()

        monkeypatch.setattr(pipeline, "_get", blocking_get)
        first_queue = await pipeline.subscribe("run-close-1")
        second_queue = await pipeline.subscribe("run-close-2")
        await both_started.wait()
        tasks = tuple(pipeline._subscription_tasks.values())

        await pipeline.aclose()
        await pipeline.aclose()

        assert all(task.cancelled() for task in tasks)
        assert pipeline._subscription_tasks == {}
        assert await asyncio.wait_for(first_queue.get(), timeout=0.1) == {
            "type": "done",
            "run_id": "run-close-1",
        }
        assert await asyncio.wait_for(second_queue.get(), timeout=0.1) == {
            "type": "done",
            "run_id": "run-close-2",
        }

    asyncio.run(scenario())


def test_reasoning_pipeline_natural_completion_removes_task_handle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from orchestrator.pipeline import ReasoningPipeline

    async def scenario() -> None:
        pipeline = ReasoningPipeline("http://orchestrator.test")

        async def completed_get(path: str, *, params: dict | None = None) -> dict:
            if path.endswith("/events"):
                return {"run_id": "run-completed", "events": []}
            return {"run_id": "run-completed", "status": "completed"}

        monkeypatch.setattr(pipeline, "_get", completed_get)
        queue = await pipeline.subscribe("run-completed")

        assert await queue.get() == {"type": "done", "run_id": "run-completed"}
        await asyncio.sleep(0)
        assert "run-completed" not in pipeline._subscription_tasks

    asyncio.run(scenario())


def test_reasoning_pipeline_reports_and_retrieves_polling_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from orchestrator.pipeline import ReasoningPipeline

    async def scenario() -> None:
        pipeline = ReasoningPipeline("http://orchestrator.test")

        async def failing_get(path: str, *, params: dict | None = None) -> dict:
            raise RuntimeError("poll failed")

        monkeypatch.setattr(pipeline, "_get", failing_get)
        queue = await pipeline.subscribe("run-error")
        task = pipeline._subscription_tasks["run-error"]
        try:
            message = await asyncio.wait_for(queue.get(), timeout=0.1)
            assert message == {
                "type": "done",
                "run_id": "run-error",
                "status": "failed",
                "error_type": "RuntimeError",
                "error_message": "poll failed",
            }
            await asyncio.sleep(0)
            assert "run-error" not in pipeline._subscription_tasks
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            elif not task.cancelled():
                task.exception()

    asyncio.run(scenario())


def test_reasoning_pipeline_serializes_concurrent_subscriptions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from orchestrator.pipeline import ReasoningPipeline

    async def scenario() -> None:
        pipeline = ReasoningPipeline("http://orchestrator.test")
        first_poll_started = asyncio.Event()
        first_cancel_waiting = asyncio.Event()
        release_first_cancel = asyncio.Event()
        polling_tasks: set[asyncio.Task] = set()
        poll_count = 0

        async def blocking_get(path: str, *, params: dict | None = None) -> dict:
            nonlocal poll_count
            poll_count += 1
            task = asyncio.current_task()
            assert task is not None
            polling_tasks.add(task)
            if poll_count == 1:
                first_poll_started.set()
                try:
                    await asyncio.Future()
                finally:
                    first_cancel_waiting.set()
                    await release_first_cancel.wait()
            await asyncio.Future()

        monkeypatch.setattr(pipeline, "_get", blocking_get)
        await pipeline.subscribe("run-race")
        await first_poll_started.wait()

        first_replacement = asyncio.create_task(pipeline.subscribe("run-race"))
        await first_cancel_waiting.wait()
        second_replacement = asyncio.create_task(pipeline.subscribe("run-race"))
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        release_first_cancel.set()
        await asyncio.gather(first_replacement, second_replacement)
        await asyncio.sleep(0)

        active_tasks = [task for task in polling_tasks if not task.done()]
        try:
            assert len(active_tasks) == 1
            assert pipeline._subscription_tasks["run-race"] is active_tasks[0]
        finally:
            await pipeline.unsubscribe("run-race")
            for task in polling_tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*polling_tasks, return_exceptions=True)

    asyncio.run(scenario())
