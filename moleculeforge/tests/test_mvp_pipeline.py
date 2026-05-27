"""End-to-end test for the MVP pipeline."""

from __future__ import annotations

import asyncio
import json


def _run(coro):
    return asyncio.run(coro)


class TestMVPPipeline:
    def test_runner_import(self) -> None:
        from mvp_pipeline.runner import run_pipeline

        assert run_pipeline is not None

    def test_runner_single_query(self) -> None:
        from mvp_pipeline.runner import run_pipeline

        result = _run(
            run_pipeline(
                "Design a drug-like molecule with high QED",
                n_samples=10,
                seed=42,
            )
        )
        assert result["status"] == "done"
        assert result["molecules_generated"] > 0
        assert result["molecules_valid"] > 0
        assert len(result["pareto_solutions"]) > 0
        assert result["run_id"].startswith("mvp-")

    def test_runner_sync_wrapper(self) -> None:
        from mvp_pipeline.runner import run_pipeline_sync

        result = run_pipeline_sync(
            "Find novel soluble small molecules",
            n_samples=5,
            seed=42,
        )
        assert result["status"] == "done"

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
        from orchestrator.workflow.graph_builder import build_graph, create_initial_state

        compiled = build_graph().build()
        state = create_initial_state(
            "test query",
            run_id="run-test",
            trace_id="trace-test",
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

    def test_sample_queries_load(self) -> None:
        with open("data/samples/test_queries.json") as f:
            queries = json.load(f)
        assert len(queries) == 5
        assert all("query" in q for q in queries)
