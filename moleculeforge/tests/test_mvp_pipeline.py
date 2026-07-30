"""End-to-end test for the MVP pipeline."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import sys

import httpx
import pytest


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture
def orchestrator_client(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, dict | None]]:
    calls: list[tuple[str, dict | None]] = []

    class _Response:
        def __init__(self, payload: dict) -> None:
            self.payload = payload

        def raise_for_status(self) -> None:
            return None

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
            calls.append((url, json))
            return _Response({"run_id": "run-mvp", "status": "queued"})

        async def get(self, url: str):
            calls.append((url, None))
            return _Response(
                {
                    "run_id": "run-mvp",
                    "status": "completed",
                    "state": {
                        "candidates": [
                            {"canonical_smiles": "CCO", "pareto_optimal": True}
                        ]
                    },
                }
            )

    monkeypatch.setattr(httpx, "AsyncClient", _Client)
    monkeypatch.setenv("ORCHESTRATOR_SVC_URL", "http://orchestrator.test")
    return calls


class TestMVPPipeline:
    def test_runner_import(self) -> None:
        from mvp_pipeline.runner import run_pipeline

        assert run_pipeline is not None

    def test_runner_single_query(self, orchestrator_client) -> None:
        from mvp_pipeline.runner import run_pipeline

        result = _run(
            run_pipeline(
                "Design a drug-like molecule with high QED",
                n_samples=10,
                seed=42,
                workflow_scope="engineering",
                validation_passed=True,
                max_refinements=1,
            )
        )
        assert result["status"] == "completed"
        assert result["run_id"] == "run-mvp"
        assert result["state"]["candidates"][0]["canonical_smiles"] == "CCO"

    def test_runner_sync_wrapper(self, orchestrator_client) -> None:
        from mvp_pipeline.runner import run_pipeline_sync

        result = run_pipeline_sync(
            "Find novel soluble small molecules",
            n_samples=5,
            seed=42,
            workflow_scope="engineering",
            validation_passed=True,
            max_refinements=1,
        )
        assert result["status"] == "completed"

    def test_orchestrator_graph(self) -> None:
        from orchestrator.workflow.graph_builder import build_graph, create_initial_state

        graph = build_graph()
        assert graph is not None
        compiled = graph.build()
        assert hasattr(compiled, "ainvoke")

        state = create_initial_state("test query")
        assert state["nl_input"] == "test query"
        assert state["status"] == "PLANNING"

    def test_orchestrator_graph_escalates_after_validation_failure(self) -> None:
        from orchestrator.workflow.graph_builder import WorkflowGraph, create_initial_state

        compiled = WorkflowGraph(workflow_scope="engineering").build()
        state = create_initial_state(
            "test query",
            run_id="run-test",
            trace_id="trace-test",
            workflow_scope="engineering",
        )
        state["validation_passed"] = False
        state["max_refinements"] = 0

        result = _run(compiled.ainvoke(state))

        assert result["status"] == "ESCALATING"
        assert result["history"] == [
            "PLANNING",
            "GENERATING",
            "VALIDATING",
            "ESCALATING",
        ]
        assert result["run_id"] == "run-test"
        assert result["trace_id"] == "trace-test"
        assert [event["stage"] for event in result["events"]] == result["history"]
        assert all(event["run_id"] == "run-test" for event in result["events"])
        assert all(event["trace_id"] == "trace-test" for event in result["events"])

    def test_orchestrator_graph_default_matches_initial_state_scope(self) -> None:
        from orchestrator.workflow.graph_builder import build_graph, create_initial_state

        graph = build_graph()
        state = create_initial_state("test query")

        assert graph.workflow_scope == state["workflow_scope"] == "state_only"

    def test_pareto_with_directions(self) -> None:
        from mf_core.types.molecule import MoleculeModel
        from mf_core.types.pareto import ParetoArchive, ParetoSolution

        archive = ParetoArchive(
            archive_id="test",
            run_id="test",
            directions=[1.0, -1.0],  # maximize first, minimize second
        )

        s1 = ParetoSolution(
            molecule=MoleculeModel(id="m1", smiles="c1ccccc1"),
            objective_values=[0.8, 2.0],
        )
        s2 = ParetoSolution(
            molecule=MoleculeModel(id="m2", smiles="CCO"),
            objective_values=[0.6, 1.0],
        )

        assert archive.insert(s1)
        assert archive.insert(s2)  # s2 is better on second obj (lower = better)
        # s1 should still be in archive (better on first obj)
        assert len(archive.solutions) >= 1

    def test_is_valid_real(self) -> None:
        from mf_core.types.molecule import MoleculeModel

        valid = MoleculeModel(id="v", smiles="c1ccccc1")
        assert valid.is_valid is True

        invalid = MoleculeModel(id="i", smiles="not_a_smiles!!!")
        assert invalid.is_valid is False

        empty = MoleculeModel(id="e", smiles="")
        assert empty.is_valid is False

    def test_sigstore_signer(self) -> None:
        from mf_agents.lineage.sigstore_signer import SigstoreSigner

        signer = SigstoreSigner(identity_token="test")
        sig = signer.sign("abc123")
        assert signer.verify("abc123", sig)
        assert not signer.verify("wrong", sig)

    def test_sigstore_signer_local_fallback_uses_hmac_sha256(self) -> None:
        from mf_agents.lineage.sigstore_signer import SigstoreSigner

        signer = SigstoreSigner(identity_token="test")

        assert signer.sign("abc123") == hmac.new(
            b"test",
            b"abc123",
            hashlib.sha256,
        ).digest()

    def test_sigstore_signer_uses_configured_commands(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from mf_agents.lineage.sigstore_signer import SigstoreSigner

        sign_command = (
            f"{sys.executable} -c \"import json,sys;"
            "req=json.load(sys.stdin);"
            "assert req['artifact_type']=='agent_lineage_payload';"
            "assert req['identity']=='generator_coord';"
            "assert req['rekor_url']=='https://rekor.example';"
            "sig='lineage-sig-'+req['payload_hash'][:8];"
            "print(json.dumps({'signature':sig,'signature_type':'sigstore_rekor',"
            "'rekor_entry':{'uuid':'lineage-rekor'}}))\""
        )
        verify_command = (
            f"{sys.executable} -c \"import json,sys;"
            "req=json.load(sys.stdin);"
            "expected='lineage-sig-'+req['payload_hash'][:8];"
            "assert req['artifact_type']=='agent_lineage_payload';"
            "assert req['expected_identity']=='generator_coord';"
            "assert req['rekor_url']=='https://rekor.example';"
            "print(json.dumps({'valid':req['signature']==expected}))\""
        )
        monkeypatch.setenv("SIGSTORE_SIGN_COMMAND", sign_command)
        monkeypatch.setenv("SIGSTORE_VERIFY_COMMAND", verify_command)
        monkeypatch.setenv("SIGSTORE_REKOR_URL", "https://rekor.example")

        signer = SigstoreSigner(identity_token="generator_coord")
        signature = signer.sign("abc123")

        assert signature.startswith(b"lineage-sig-")
        assert signer.verify("abc123", signature) is True
        assert signer.verify("wrong", signature) is False

    def test_sigstore_signer_sign_command_preflight_rejects_missing_executable(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from mf_agents.lineage.sigstore_signer import SigstoreSigner

        monkeypatch.setenv("SIGSTORE_SIGN_COMMAND", "missing-lineage-sigstore-sign --json")

        signer = SigstoreSigner(identity_token="generator_coord")
        with pytest.raises(RuntimeError, match="not found"):
            signer.sign("abc123")

    def test_sigstore_signer_verify_command_preflight_rejects_missing_executable(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from mf_agents.lineage.sigstore_signer import SigstoreSigner

        monkeypatch.setenv("SIGSTORE_VERIFY_COMMAND", "missing-lineage-sigstore-verify --json")

        signer = SigstoreSigner(identity_token="generator_coord")
        with pytest.raises(RuntimeError, match="not found"):
            signer.verify("abc123", b"signature")

    def test_sample_queries_load(self) -> None:
        with open("data/samples/test_queries.json") as f:
            queries = json.load(f)
        assert len(queries) == 5
        assert all("query" in q for q in queries)
