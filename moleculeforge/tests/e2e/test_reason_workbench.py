"""End-to-end tests for the NL reasoning workbench.

Exercises the new /v1/reason/* endpoints with the FastAPI TestClient so
no external services are needed.
"""
from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client() -> TestClient:
    from api_gateway.main import app
    with TestClient(app) as c:
        yield c


@pytest.mark.e2e
def test_reason_run_completes(client: TestClient) -> None:
    r = client.post("/v1/reason/runs", json={
        "intent": "Design 16 KRAS G12C inhibitors with MW < 500, LogP 1-4, "
                  "with Michael acceptor warhead, prioritise drug-likeness."
    })
    assert r.status_code == 200
    run_id = r.json()["run_id"]

    deadline = time.time() + 180
    snap = None
    while time.time() < deadline:
        snap = client.get(f"/v1/reason/runs/{run_id}").json()
        if snap["status"] in {"completed", "failed"}:
            break
        time.sleep(0.5)
    assert snap is not None and snap["status"] == "completed"

    # Reasoning chain — every stage must have fired
    stages = [s["stage"] for s in snap["steps"]]
    for required in [
        "nl_parse", "objectives", "generation", "scoring",
        "constraint_filter", "novelty", "ranking", "summary",
    ]:
        assert required in stages, f"missing stage {required}"

    # Objectives parsed correctly
    obj = snap["objectives"]
    assert "KRAS G12C" in obj["targets"]
    assert "C=CC(=O)" in obj["constraints"]["must_include_smarts"]
    assert obj["n_samples"] == 16

    # Results contain at least one novel candidate
    assert snap["n_candidates"] >= 1
    assert snap["n_novel"] >= 1
    novel = [r for r in snap["results"] if r["is_novel"]]
    assert novel, "expected at least one novel candidate"


@pytest.mark.e2e
def test_reason_run_recognises_known_molecule(client: TestClient) -> None:
    r = client.post("/v1/reason/runs", json={
        "intent": "Lead optimise aspirin and ibuprofen for COX-2, generate 24 candidates, MW < 400.",
    })
    run_id = r.json()["run_id"]

    deadline = time.time() + 180
    while time.time() < deadline:
        snap = client.get(f"/v1/reason/runs/{run_id}").json()
        if snap["status"] in {"completed", "failed"}:
            break
        time.sleep(0.5)
    assert snap["status"] == "completed", snap
    # Aspirin/Ibuprofen are seed scaffolds in the parser, so the unmutated
    # anchors should appear among results and be flagged as known.
    known = [r for r in snap["results"] if not r["is_novel"]]
    assert known, "expected at least one known reference match"
    names = {r["known_match"]["name"] for r in known}
    assert names & {"Aspirin", "Ibuprofen"}, names


@pytest.mark.e2e
def test_reason_history_includes_recent(client: TestClient) -> None:
    r = client.get("/v1/reason/runs?limit=10")
    assert r.status_code == 200
    runs = r.json()["runs"]
    assert isinstance(runs, list)
    assert len(runs) >= 1


@pytest.mark.e2e
def test_known_catalog_lookup(client: TestClient) -> None:
    r = client.get("/v1/reason/known?query=aspirin")
    assert r.status_code == 200
    items = r.json()["items"]
    assert any(it["name"] == "Aspirin" for it in items)


@pytest.mark.e2e
def test_static_workbench_is_served(client: TestClient) -> None:
    r = client.get("/")
    assert r.status_code == 200
    assert "Reasoning Workbench" in r.text
    r = client.get("/styles.css")
    assert r.status_code == 200
    r = client.get("/app.js")
    assert r.status_code == 200
