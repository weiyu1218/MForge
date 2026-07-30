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
    assert calls[0][2] == request


@pytest.mark.parametrize("include_explicit_null", [False, True])
def test_reason_run_without_project_crosses_real_orchestrator_store_boundary(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    include_explicit_null: bool,
) -> None:
    from mf_core.db.store import RunStore
    from orchestrator_svc import main as orchestrator_main
    from orchestrator_svc.main import RunControl

    store = RunStore(tmp_path / "reason-runs.db")
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
    forwarded_payloads: list[dict] = []

    class _Client:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def post(self, url: str, json: dict) -> httpx.Response:
            forwarded_payloads.append(dict(json))
            transport = httpx.ASGITransport(app=orchestrator_main.rest_app)
            async with real_async_client(
                transport=transport,
                base_url="http://orchestrator.test",
            ) as upstream:
                return await upstream.post("/v1/orchestrator/design", json=json)

    monkeypatch.setattr(httpx, "AsyncClient", _Client)
    request = {
        "intent": "Design soluble molecules",
        "workflow_scope": "state_only",
        "validation_passed": True,
        "max_refinements": 0,
    }
    if include_explicit_null:
        request["project_id"] = None

    response = client.post("/v1/reason/runs", json=request)

    assert response.status_code == 202
    assert forwarded_payloads == [
        {
            "intent": "Design soluble molecules",
            "workflow_scope": "state_only",
            "validation_passed": True,
            "max_refinements": 0,
        }
    ]
    snapshot = asyncio.run(store.get_run(response.json()["run_id"]))
    assert snapshot is not None
    assert snapshot["project_id"] is None


def test_reason_run_preserves_invalid_empty_project_for_canonical_validation(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from mf_core.db.store import RunStore
    from orchestrator_svc import main as orchestrator_main
    from orchestrator_svc.main import RunControl

    store = RunStore(tmp_path / "reason-runs.db")
    asyncio.run(store.initialize())
    monkeypatch.setattr(orchestrator_main, "_RUN_STORE", store)
    monkeypatch.setattr(orchestrator_main, "_RUN_CONTROL", RunControl(store))
    monkeypatch.setattr(orchestrator_main, "_RUN_INITIALIZED_STORE", store)
    monkeypatch.setattr(orchestrator_main, "_RUNTIME_INIT_LOCK", None)
    monkeypatch.setattr(orchestrator_main, "_RUN_TASKS", {})
    real_async_client = httpx.AsyncClient
    forwarded_payloads: list[dict] = []

    class _Client:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def post(self, url: str, json: dict) -> httpx.Response:
            forwarded_payloads.append(dict(json))
            transport = httpx.ASGITransport(app=orchestrator_main.rest_app)
            async with real_async_client(
                transport=transport,
                base_url="http://orchestrator.test",
            ) as upstream:
                return await upstream.post("/v1/orchestrator/design", json=json)

    monkeypatch.setattr(httpx, "AsyncClient", _Client)

    response = client.post(
        "/v1/reason/runs",
        json={
            "intent": "Design soluble molecules",
            "workflow_scope": "state_only",
            "validation_passed": True,
            "max_refinements": 0,
            "project_id": "",
        },
    )

    assert response.status_code == 400
    assert forwarded_payloads[0]["project_id"] == ""
    assert (asyncio.run(store.list_runs(page_size=10)))["items"] == []


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


def test_workbench_latest_open_owns_poll_and_stream_updates() -> None:
    script_path = Path(__file__).resolve().parents[2] / "ui/public/app.js"
    node_script = r"""
const fs = require("fs");
const vm = require("vm");
const source = fs.readFileSync(process.argv[1], "utf8");

function snapshot(runId, status) {
  return {
    run_id: runId,
    status,
    state: { run_id: runId, request: {}, candidates: [] },
  };
}

function createHarness(snapshotQueues) {
  const elements = new Map();
  const timers = [];
  const streams = [];
  const designRequests = [];

  function createElement() {
    return {
      value: "",
      innerHTML: "",
      textContent: "",
      hidden: false,
      disabled: false,
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

  function response(payload) {
    return {
      ok: true,
      status: 200,
      statusText: "OK",
      text: async () => JSON.stringify(payload),
      json: async () => payload,
    };
  }

  class FakeEventSource {
    constructor(url) {
      this.url = url;
      this.closed = false;
      this.onmessage = null;
      this.onerror = null;
      streams.push(this);
    }
    close() {
      this.closed = true;
    }
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
    fetch: async (url) => {
      if (url === "/health") {
        return response({ status: "healthy", gpu: { device_count: 0 }, devices: [] });
      }
      if (url === "/v1/reason/runs?page_size=30") {
        return response({ runs: [] });
      }
      if (url.startsWith("/v1/design/")) {
        const runId = url.slice("/v1/design/".length);
        designRequests.push(runId);
        const queue = snapshotQueues[runId] || [];
        if (!queue.length) throw new Error(`No snapshot queued for ${runId}`);
        return response(await queue.shift());
      }
      return response({ items: [] });
    },
    setInterval() { return 0; },
    clearInterval() {},
    setTimeout(callback) {
      timers.push(callback);
      return timers.length;
    },
    requestAnimationFrame() {},
    Event: class {},
    EventSource: FakeEventSource,
    alert() {},
  };
  vm.createContext(context);
  vm.runInContext(
    source
      + "\nglobalThis.__openRun = openRun;"
      + "\nglobalThis.__activeRunId = () => activeRunId;",
    context,
  );
  return { context, elements, timers, streams, designRequests };
}

async function flush() {
  await new Promise((resolve) => setImmediate(resolve));
  await new Promise((resolve) => setImmediate(resolve));
}

function deferred() {
  let resolve;
  const promise = new Promise((complete) => { resolve = complete; });
  return { promise, resolve };
}

async function main() {
  const pollHarness = createHarness({
    old: [snapshot("old", "running"), snapshot("old", "completed")],
    current: [snapshot("current", "completed")],
  });
  await pollHarness.context.__openRun("old", { live: true });
  await pollHarness.context.__openRun("current", { live: false });
  pollHarness.timers.shift()();
  await flush();
  if (pollHarness.context.__activeRunId() !== "current") {
    throw new Error("stale poll took active run ownership");
  }
  if (pollHarness.elements.get("#run-id").textContent !== "current") {
    throw new Error("stale poll rendered over the current run");
  }
  if (pollHarness.designRequests.filter((runId) => runId === "old").length !== 1) {
    throw new Error("stale poll fetched after run ownership changed");
  }

  const streamHarness = createHarness({
    old: [snapshot("old", "running"), snapshot("old", "completed")],
    current: [snapshot("current", "completed")],
  });
  await streamHarness.context.__openRun("old", { live: true });
  const staleStream = streamHarness.streams[0];
  await streamHarness.context.__openRun("current", { live: false });
  staleStream.onmessage({ data: JSON.stringify({ type: "done" }) });
  await flush();
  if (streamHarness.context.__activeRunId() !== "current") {
    throw new Error("stale stream took active run ownership");
  }
  if (streamHarness.elements.get("#run-id").textContent !== "current") {
    throw new Error("stale stream rendered over the current run");
  }
  if (streamHarness.designRequests.filter((runId) => runId === "old").length !== 1) {
    throw new Error("closed stream fetched after run ownership changed");
  }

  const reopenHarness = createHarness({
    repeated: [
      snapshot("repeated", "running"),
      snapshot("repeated", "running"),
      snapshot("repeated", "completed"),
      snapshot("repeated", "completed"),
    ],
  });
  await reopenHarness.context.__openRun("repeated", { live: true });
  await reopenHarness.context.__openRun("repeated", { live: true });
  const pendingTimers = reopenHarness.timers.splice(0);
  pendingTimers.forEach((callback) => callback());
  await flush();
  const repeatedRequests = reopenHarness.designRequests.filter(
    (runId) => runId === "repeated",
  ).length;
  if (repeatedRequests !== 3) {
    throw new Error(`expected one owned poll after reopen, observed ${repeatedRequests - 2}`);
  }

  const stalePoll = deferred();
  const raceHarness = createHarness({
    race: [
      snapshot("race", "running"),
      stalePoll.promise,
      snapshot("race", "completed"),
    ],
  });
  await raceHarness.context.__openRun("race", { live: true });
  raceHarness.timers.shift()();
  await flush();
  raceHarness.streams[0].onmessage({
    data: JSON.stringify({ type: "done" }),
  });
  await flush();
  if (raceHarness.elements.get("#run-status").textContent !== "completed") {
    throw new Error("stream terminal refresh was not rendered");
  }
  stalePoll.resolve(snapshot("race", "running"));
  await flush();
  if (raceHarness.elements.get("#run-status").textContent !== "completed") {
    throw new Error("stale poll overwrote the stream terminal state");
  }
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
"""

    completed = subprocess.run(
        ["node", "-e", node_script, str(script_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
