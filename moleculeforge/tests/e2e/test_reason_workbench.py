"""End-to-end tests for the canonical reasoning workbench routes."""

from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path

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


def test_reason_first_page_omits_empty_token_across_real_asgi_apps(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from mf_core.db.store import RunStore
    from orchestrator_svc import main as orchestrator_main

    store = RunStore(tmp_path / "reason-pagination.db")
    asyncio.run(store.initialize())
    monkeypatch.setattr(orchestrator_main, "_RUN_STORE", store)
    monkeypatch.setattr(orchestrator_main, "_RUN_CONTROL", orchestrator_main.RunControl(store))
    monkeypatch.setattr(orchestrator_main, "_RUN_INITIALIZED_STORE", store)
    monkeypatch.setattr(orchestrator_main, "_RUNTIME_INIT_LOCK", None)
    real_async_client = httpx.AsyncClient

    class _Client:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def get(
            self,
            url: str,
            params: dict | None = None,
        ) -> httpx.Response:
            transport = httpx.ASGITransport(app=orchestrator_main.rest_app)
            async with real_async_client(
                transport=transport,
                base_url="http://orchestrator.test",
            ) as upstream:
                path = url.removeprefix("http://orchestrator.test")
                return await upstream.get(path, params=params)

    monkeypatch.setattr(httpx, "AsyncClient", _Client)
    monkeypatch.setenv("ORCHESTRATOR_SVC_URL", "http://orchestrator.test")

    response = client.get("/v1/reason/runs?page_size=10")

    assert response.status_code == 200, response.text
    assert response.json() == {"runs": [], "next_page_token": None}


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


def test_workbench_matches_duplicate_candidate_occurrences_like_backend() -> None:
    script_path = Path(__file__).resolve().parents[2] / "ui/public/app.js"
    node_script = r"""
const fs = require("fs");
const vm = require("vm");
const element = {
  value: "",
  dataset: {},
  classList: { add() {}, remove() {}, toggle() {} },
  addEventListener() {},
  dispatchEvent() {},
  focus() {},
  checkValidity() { return true; },
  reportValidity() {},
  querySelector() { return this; },
  appendChild() {},
  getContext() {
    return { fillRect() {}, fillText() {}, set fillStyle(v) {}, set font(v) {} };
  },
};
const context = {
  console,
  document: {
    querySelector() { return element; },
    querySelectorAll() { return []; },
    createElement() { return { ...element, dataset: {}, classList: element.classList }; },
  },
  window: { scrollTo() {} },
  SmilesDrawer: {
    Drawer: class { draw() {} },
    parse() {},
  },
  fetch: async () => ({
    ok: true,
    status: 200,
    statusText: "OK",
    text: async () => JSON.stringify({ runs: [], items: [] }),
    json: async () => ({ status: "healthy", gpu: { device_count: 0 }, devices: [] }),
  }),
  setInterval() { return 0; },
  clearInterval() {},
  setTimeout() { return 0; },
  requestAnimationFrame() {},
  Event: class {},
  EventSource: class {},
  alert() {},
};
vm.createContext(context);
const source = fs.readFileSync(process.argv[1], "utf8");
vm.runInContext(source + "\nglobalThis.__candidateRows = orchestratorCandidateRows;", context);
const rows = context.__candidateRows({
  candidates: [
    { candidate_id: "candidate-duplicate", canonical_smiles: "CCO", composite_score: 1 },
    { candidate_id: "candidate-duplicate", canonical_smiles: "CCO", composite_score: 2 },
    { candidate_id: "candidate-other", canonical_smiles: "CCO", composite_score: 3 },
  ],
  validation: {
    results: [
      { canonical_smiles: "CCO", rank: 3, valid: true, properties: { qed: 0.3 } },
      {
        candidate_id: "candidate-duplicate",
        canonical_smiles: "CCO",
        rank: 1,
        valid: true,
        properties: { qed: 0.9 },
      },
      {
        candidate_id: "candidate-duplicate",
        canonical_smiles: "CCO",
        rank: 2,
        valid: true,
        properties: { qed: 0.8 },
      },
    ],
  },
});
process.stdout.write(JSON.stringify(rows.map((row) => [
  row.candidate_id,
  row.rank,
  row.composite_score,
  row.properties.qed,
])));
"""

    completed = subprocess.run(
        ["node", "-e", node_script, str(script_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == [
        ["candidate-duplicate", 1, 1, 0.9],
        ["candidate-duplicate", 2, 2, 0.8],
        ["candidate-other", 3, 3, 0.3],
    ]


def test_workbench_renders_top_level_run_metadata_only_when_present() -> None:
    script_path = Path(__file__).resolve().parents[2] / "ui/public/app.js"
    node_script = r"""
const fs = require("fs");
const vm = require("vm");
const elements = new Map();
function createElement() {
  return {
    value: "",
    innerHTML: "",
    textContent: "",
    hidden: false,
    dataset: {},
    className: "",
    classList: { add() {}, remove() {}, toggle() {} },
    addEventListener() {},
    dispatchEvent() {},
    focus() {},
    checkValidity() { return true; },
    reportValidity() {},
    querySelector() { return this; },
    appendChild() {},
    getContext() {
      return { fillRect() {}, fillText() {}, set fillStyle(v) {}, set font(v) {} };
    },
  };
}
function elementFor(selector) {
  if (!elements.has(selector)) elements.set(selector, createElement());
  return elements.get(selector);
}
const context = {
  console,
  document: {
    querySelector(selector) { return elementFor(selector); },
    querySelectorAll() { return []; },
    createElement,
  },
  window: { scrollTo() {} },
  SmilesDrawer: {
    Drawer: class { draw() {} },
    parse() {},
  },
  fetch: async () => ({
    ok: true,
    status: 200,
    statusText: "OK",
    text: async () => JSON.stringify({ runs: [], items: [] }),
    json: async () => ({ status: "healthy", gpu: { device_count: 0 }, devices: [] }),
  }),
  setInterval() { return 0; },
  clearInterval() {},
  setTimeout() { return 0; },
  requestAnimationFrame() {},
  Event: class {},
  EventSource: class {},
  alert() {},
};
vm.createContext(context);
const source = fs.readFileSync(process.argv[1], "utf8");
vm.runInContext(source + "\nglobalThis.__renderRun = renderOrchestratorRun;", context);
context.__renderRun({
  run_id: "run-metadata",
  status: "completed",
  objectives: ["qed", "sa_score"],
  summary: "Prioritize drug-like candidates",
  devices_used: ["cpu", "cuda:0"],
  state: { request: {}, candidates: [] },
}, "fallback intent");
const withMetadata = elementFor("#objectives").innerHTML;
context.__renderRun({
  run_id: "run-without-metadata",
  status: "completed",
  state: { request: {}, candidates: [] },
}, "fallback intent");
const withoutMetadata = elementFor("#objectives").innerHTML;
process.stdout.write(JSON.stringify({ withMetadata, withoutMetadata }));
"""

    completed = subprocess.run(
        ["node", "-e", node_script, str(script_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    rendered = json.loads(completed.stdout)
    assert "Run objectives" in rendered["withMetadata"]
    assert "qed, sa_score" in rendered["withMetadata"]
    assert "Run summary" in rendered["withMetadata"]
    assert "Prioritize drug-like candidates" in rendered["withMetadata"]
    assert "Execution devices" in rendered["withMetadata"]
    assert "cpu, cuda:0" in rendered["withMetadata"]
    assert "Run objectives" not in rendered["withoutMetadata"]
    assert "Run summary" not in rendered["withoutMetadata"]
    assert "Execution devices" not in rendered["withoutMetadata"]
