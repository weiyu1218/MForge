"""Unit tests for the CIC compiler."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _run(coro):
    return asyncio.run(coro)


def _load_hciv_train_module():
    script = ROOT / "services/cig-compiler-svc/train_hciv_encoder.py"
    spec = importlib.util.spec_from_file_location("cig_hciv_train", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_hciv_training_jsonl(tmp_path: Path) -> Path:
    from mf_core.types.cig import ChemicalIntentGraph, ObjectiveNode, ObjectiveType

    data_path = tmp_path / "hciv_train.jsonl"
    cig = ChemicalIntentGraph(
        intent_id="cig-train-1",
        objective_nodes=[
            ObjectiveNode(
                id="obj_qed",
                type=ObjectiveType.CONTINUOUS_MAXIMIZE,
                oracle="rdkit",
                weight=1.0,
            )
        ],
        source_user_input="maximize QED",
    )
    data_path.write_text(
        json.dumps(
            {
                "id": "example-1",
                "cig": cig.model_dump(mode="json", by_alias=True),
                "target_hciv": [1.0] + [0.0] * 8,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return data_path


class TestCICCompiler:
    def test_import(self) -> None:
        from cig_compiler_svc.domain.compiler import CIGCompiler

        compiler = CIGCompiler()
        assert compiler is not None

    def test_heuristic_extract(self) -> None:
        from cig_compiler_svc.domain.stages.stage1_semantic import _heuristic_extract

        result = _heuristic_extract("Find drug-like soluble molecules with high potency")
        assert "properties" in result
        assert "constraints" in result
        assert len(result["properties"]) > 0

    def test_build_cig(self) -> None:
        from cig_compiler_svc.domain.stages.stage2_cig_build import build_cig

        extracted = {
            "properties": [
                {"name": "qed", "direction": "maximize", "priority": 1},
                {"name": "sa_score", "direction": "minimize", "priority": 2},
            ],
            "constraints": {"max_mw": 400, "lipinski_strict": True},
        }
        cig = build_cig(extracted, "test input")
        assert cig.intent_id.startswith("cig-")
        assert len(cig.objective_nodes) == 2
        assert cig.source_user_input == "test input"

    def test_chemical_intent_graph_serializes_edges(self) -> None:
        from mf_core.types.cig import (
            ChemicalIntentGraph,
            ObjectiveEdge,
            ObjectiveHyperedge,
            ObjectiveNode,
            ObjectiveType,
        )

        cig = ChemicalIntentGraph(
            intent_id="cig-test",
            objective_nodes=[
                ObjectiveNode(
                    id="obj_affinity",
                    type=ObjectiveType.CONTINUOUS_MAXIMIZE,
                    oracle="boltz2",
                    weight=0.5,
                ),
                ObjectiveNode(
                    id="obj_admet_bundle",
                    type=ObjectiveType.MULTI_CONSTRAINT_SATISFY,
                    oracle="admet_ai",
                    weight=0.5,
                ),
            ],
            edges=[
                ObjectiveEdge(
                    source_id="obj_affinity",
                    target_id="obj_admet_bundle",
                    relation="trade_off",
                    strength=-0.5,
                )
            ],
            hyperedges=[
                ObjectiveHyperedge(
                    source_ids=["obj_affinity"],
                    target_ids=["obj_admet_bundle"],
                    relation="trade_off",
                    strength=-0.5,
                )
            ],
            source_user_input="test input",
        )

        payload = cig.model_dump(mode="json")

        assert payload["edges"] == [
            {
                "source_id": "obj_affinity",
                "target_id": "obj_admet_bundle",
                "relation": "trade_off",
                "strength": -0.5,
            }
        ]
        assert payload["hyperedges"] == [
            {
                "source_ids": ["obj_affinity"],
                "target_ids": ["obj_admet_bundle"],
                "relation": "trade_off",
                "strength": -0.5,
            }
        ]

    def test_chemical_intent_graph_serializes_jsonld_context(self) -> None:
        from mf_core.types.cig import ChemicalIntentGraph, ObjectiveNode, ObjectiveType

        cig = ChemicalIntentGraph(
            intent_id="cig-test",
            objective_nodes=[
                ObjectiveNode(
                    id="obj_qed",
                    type=ObjectiveType.CONTINUOUS_MAXIMIZE,
                    oracle="rdkit",
                    weight=1.0,
                )
            ],
            source_user_input="test input",
        )

        payload = cig.model_dump(mode="json", by_alias=True)

        assert payload["@context"]["mf"] == "https://moleculeforge.io/ontology#"
        assert payload["@context"]["objective_nodes"] == "mf:objectiveNodes"
        assert "jsonld_context" not in payload

    def test_cig_schema_declares_objective_edges(self) -> None:
        schema = json.loads((ROOT / "schemas/cig.schema.json").read_text())

        assert "edges" in schema["properties"]
        edge_schema = schema["properties"]["edges"]["items"]
        assert edge_schema["required"] == ["source_id", "target_id", "relation", "strength"]
        assert set(edge_schema["properties"]) == {
            "source_id",
            "target_id",
            "relation",
            "strength",
        }

        assert "hyperedges" in schema["properties"]
        hyperedge_schema = schema["properties"]["hyperedges"]["items"]
        assert hyperedge_schema["required"] == [
            "source_ids",
            "target_ids",
            "relation",
            "strength",
        ]

    def test_cig_schema_declares_jsonld_context(self) -> None:
        schema = json.loads((ROOT / "schemas/cig.schema.json").read_text())

        assert "@context" in schema["required"]
        context_schema = schema["properties"]["@context"]
        assert context_schema["type"] == "object"
        assert context_schema["required"] == ["mf"]
        assert context_schema["properties"]["mf"]["const"] == "https://moleculeforge.io/ontology#"

    def test_build_cig_adds_affinity_admet_tradeoff_edge(self) -> None:
        from cig_compiler_svc.domain.stages.stage2_cig_build import build_cig

        extracted = {
            "targets": [{"name": "KRAS"}],
            "properties": [],
            "admet_constraints": {
                "oral_bioavailability_min": 0.3,
                "cyp3a4_ic50_min": 10.0,
            },
            "constraints": {},
        }

        cig = build_cig(extracted, "test input")
        objective_ids = {node.id for node in cig.objective_nodes}

        assert objective_ids >= {"obj_affinity", "obj_admet_bundle"}
        assert [
            edge
            for edge in cig.edges
            if edge.source_id == "obj_affinity"
            and edge.target_id == "obj_admet_bundle"
            and edge.relation == "trade_off"
            and edge.strength < 0
        ]
        assert [
            hyperedge
            for hyperedge in cig.hyperedges
            if hyperedge.source_ids == ["obj_affinity"]
            and hyperedge.target_ids == ["obj_admet_bundle"]
            and hyperedge.relation == "trade_off"
            and hyperedge.strength < 0
        ]

    def test_build_cig_includes_jsonld_context(self) -> None:
        from cig_compiler_svc.domain.stages.stage2_cig_build import build_cig

        cig = build_cig(
            {
                "properties": [{"name": "qed", "direction": "maximize", "priority": 1}],
                "constraints": {},
            },
            "test input",
        )

        payload = cig.model_dump(mode="json", by_alias=True)
        assert payload["@context"]["target_context"] == "mf:targetContext"
        assert payload["@context"]["edges"] == "mf:objectiveEdges"

    def test_hciv_generation(self) -> None:
        from cig_compiler_svc.domain.hciv_generator import generate_random_hciv

        hciv = generate_random_hciv(dim=8, seed=42)
        assert len(hciv.coordinates) == 9  # dim + 1 for Lorentz
        assert hciv.coordinates[0] > 0  # time component positive

    def test_cone_generation(self) -> None:
        from cig_compiler_svc.domain.hciv_generator import (
            generate_intent_cone,
            generate_random_hciv,
        )

        hciv = generate_random_hciv(dim=8, seed=42)
        cone = generate_intent_cone(apex=hciv, dim=8, seed=42)
        assert cone.apex == hciv
        assert cone.angle_radians > 0

    def test_full_compile(self) -> None:
        from cig_compiler_svc.domain.compiler import CIGCompiler

        compiler = CIGCompiler(mode="local_demo", encoding_mode="hash")
        cig, hciv, cone = _run(compiler.compile("Design a drug-like molecule", seed=42))
        assert cig.intent_id.startswith("cig-")
        assert len(hciv.coordinates) > 0
        assert cone.apex is not None

    def test_compile_uses_grounding_results(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from cig_compiler_svc.domain.compiler import CIGCompiler

        async def fake_query_uniprot_entry(query: str, limit: int = 5) -> list[dict]:
            assert query == "KRAS"
            return [
                {
                    "accession": "P01116",
                    "protein_name": "GTPase KRas",
                    "gene_name": "KRAS",
                    "organism": "Homo sapiens",
                    "pdb_ids": ["4DSO", "5VBA"],
                    "source_timestamp": "2026-05-15T00:00:00Z",
                }
            ]

        monkeypatch.setattr(
            "cig_compiler_svc.domain.tools.uniprot_tool.query_uniprot_entry",
            fake_query_uniprot_entry,
        )

        compiler = CIGCompiler(mode="local_demo", encoding_mode="hash")
        cig, _, _ = _run(compiler.compile("Design a KRAS G12C inhibitor", seed=42))

        assert cig.target_context["uniprot_ids"] == ["P01116"]
        assert cig.target_context["pdb_ids"] == ["4DSO", "5VBA"]
        assert cig.target_context["grounding_evidence"][0]["source"] == "uniprot"

    def test_grounding_collects_multisource_evidence(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from cig_compiler_svc.domain.stages.stage1b_grounding import (
            ALL_GROUNDING_SOURCES,
            ground_knowledge,
        )

        async def fake_query_uniprot_entry(query: str, limit: int = 5) -> list[dict]:
            assert query == "KRAS"
            return [
                {
                    "accession": "P01116",
                    "protein_name": "GTPase KRas",
                    "gene_name": "KRAS",
                    "organism": "Homo sapiens",
                    "pdb_ids": ["6OIM"],
                    "source_timestamp": "2026-05-15T00:00:00Z",
                }
            ]

        async def fake_query_pdb_entries(
            query: str,
            uniprot_ids: list[str] | None = None,
            limit: int = 5,
        ) -> list[dict]:
            assert query == "KRAS"
            assert uniprot_ids == ["P01116"]
            return [
                {
                    "pdb_id": "6OIM",
                    "title": "KRAS G12C inhibitor complex",
                    "source_timestamp": "2026-05-15T00:00:01Z",
                }
            ]

        async def fake_query_chembl_targets(
            query: str,
            uniprot_ids: list[str] | None = None,
            limit: int = 5,
        ) -> list[dict]:
            assert query == "KRAS"
            assert uniprot_ids == ["P01116"]
            return [
                {
                    "target_chembl_id": "CHEMBL612545",
                    "pref_name": "KRAS",
                    "source_timestamp": "2026-05-15T00:00:02Z",
                }
            ]

        monkeypatch.setattr(
            "cig_compiler_svc.domain.tools.uniprot_tool.query_uniprot_entry",
            fake_query_uniprot_entry,
        )
        monkeypatch.setattr(
            "cig_compiler_svc.domain.tools.pdb_tool.query_pdb_entries",
            fake_query_pdb_entries,
        )
        monkeypatch.setattr(
            "cig_compiler_svc.domain.tools.chembl_tool.query_chembl_targets",
            fake_query_chembl_targets,
        )

        enriched = _run(
            ground_knowledge(
                {
                    "targets": [{"name": "KRAS"}],
                },
                sources=ALL_GROUNDING_SOURCES,
            )
        )

        evidence = enriched["_grounding_evidence"]
        assert {item["source"] for item in evidence} == {
            "uniprot",
            "pdb",
            "chembl",
        }
        assert all(item["source_timestamp"] for item in evidence)
        assert enriched["_grounded_uniprot_ids"] == ["P01116"]
        assert enriched["_grounded_pdb_ids"] == ["6OIM"]
        assert enriched["_grounded_chembl_target_ids"] == ["CHEMBL612545"]

    def test_default_production_mode_requires_real_parser(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from cig_compiler_svc.domain.compiler import CIGCompiler

        monkeypatch.delenv("CIG_SEMANTIC_PARSER_URI", raising=False)
        compiler = CIGCompiler()

        with pytest.raises(RuntimeError, match="CIG_SEMANTIC_PARSER_URI"):
            _run(compiler.compile("Design a drug-like molecule", seed=42))

    def test_production_semantic_parser_adapter_calls_http_endpoint(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from cig_compiler_svc.domain.compiler import ProductionSemanticParserAdapter

        calls = []

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def read(self):
                return json.dumps(
                    {
                        "properties": [
                            {
                                "name": "qed",
                                "direction": "maximize",
                                "priority": 1,
                            }
                        ],
                    }
                ).encode("utf-8")

        def fake_urlopen(request, timeout):
            calls.append((request.full_url, request.data, dict(request.header_items()), timeout))
            return FakeResponse()

        monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

        parsed = ProductionSemanticParserAdapter(
            "https://semantic-parser.example/parse",
        )("Design a molecule")

        assert parsed["properties"][0]["name"] == "qed"
        assert calls[0][0] == "https://semantic-parser.example/parse"
        assert json.loads(calls[0][1].decode("utf-8")) == {"text": "Design a molecule"}
        assert calls[0][2]["Content-type"] == "application/json"

    def test_production_semantic_parser_adapter_calls_json_command(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        from cig_compiler_svc.domain.compiler import ProductionSemanticParserAdapter

        runner = tmp_path / "semantic_parser.py"
        runner.write_text(
            "import json, sys\n"
            "payload = json.load(sys.stdin)\n"
            "assert payload['text'] == 'Design a soluble KRAS inhibitor'\n"
            "print(json.dumps({"
            "'targets': [{'name': 'KRAS'}], "
            "'properties': [{'name': 'solubility', 'direction': 'maximize', 'priority': 1}], "
            "'constraints': {'max_mw': 450}"
            "}))\n",
            encoding="utf-8",
        )
        monkeypatch.delenv("CIG_SEMANTIC_PARSER_URI", raising=False)
        monkeypatch.setenv("CIG_SEMANTIC_PARSER_COMMAND", f"{sys.executable} {runner}")

        parsed = ProductionSemanticParserAdapter()("Design a soluble KRAS inhibitor")

        assert parsed["targets"] == [{"name": "KRAS"}]
        assert parsed["properties"][0]["name"] == "solubility"
        assert parsed["constraints"]["max_mw"] == 450

    def test_hash_mode_requires_local_demo(self) -> None:
        from cig_compiler_svc.domain.compiler import CIGCompiler

        with pytest.raises(ValueError, match="local_demo"):
            CIGCompiler(encoding_mode="hash")

    def test_random_mode_requires_local_demo(self) -> None:
        from cig_compiler_svc.domain.compiler import CIGCompiler

        with pytest.raises(ValueError, match="local_demo"):
            CIGCompiler(encoding_mode="random")

    def test_learned_mode_requires_encoder(self) -> None:
        from cig_compiler_svc.domain.compiler import CIGCompiler

        def parser(_: str) -> dict:
            return {"properties": [{"name": "qed", "direction": "maximize"}]}

        compiler = CIGCompiler(
            mode="local_demo",
            encoding_mode="learned",
            semantic_parser=parser,
        )

        with pytest.raises(RuntimeError, match="learned HCIV encoder"):
            _run(compiler.compile("Design a drug-like molecule", seed=42))

    def test_production_learned_requires_checkpoint(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from cig_compiler_svc.domain.compiler import CIGCompiler

        def parser(_: str) -> dict:
            return {"properties": [{"name": "qed", "direction": "maximize"}]}

        monkeypatch.delenv("HCIV_CHECKPOINT_PATH", raising=False)
        compiler = CIGCompiler(
            semantic_parser=parser,
            hciv_dim=8,
            enable_grounding=False,
        )

        with pytest.raises(RuntimeError, match="HCIV_CHECKPOINT_PATH"):
            _run(compiler.compile("Design a drug-like molecule", seed=42))

    def test_production_learned_loads_checkpoint(
        self,
        tmp_path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import torch
        from cig_compiler_svc.domain.compiler import CIGCompiler
        from cig_compiler_svc.domain.hciv_encoder import HCIVEncoder

        def parser(_: str) -> dict:
            return {"properties": [{"name": "qed", "direction": "maximize"}]}

        checkpoint_path = tmp_path / "hciv.pt"
        torch.save(HCIVEncoder(dim=8).state_dict(), checkpoint_path)
        monkeypatch.setenv("HCIV_CHECKPOINT_PATH", str(checkpoint_path))

        compiler = CIGCompiler(
            semantic_parser=parser,
            hciv_dim=8,
            enable_grounding=False,
        )
        _, hciv, cone = _run(compiler.compile("Design a drug-like molecule", seed=42))

        assert len(hciv.coordinates) == 9
        assert cone.apex == hciv

    def test_hciv_training_examples_load_cig_and_target(self, tmp_path) -> None:
        from cig_compiler_svc.domain.hciv_training import (
            load_hciv_training_examples,
        )

        data_path = _write_hciv_training_jsonl(tmp_path)

        examples = load_hciv_training_examples(data_path, dim=8)

        assert len(examples) == 1
        assert examples[0].example_id == "example-1"
        assert examples[0].cig.intent_id == "cig-train-1"
        assert examples[0].target_coordinates.shape == (9,)

    def test_hciv_training_examples_reject_invalid_lorentz_target(
        self,
        tmp_path,
    ) -> None:
        from cig_compiler_svc.domain.hciv_training import (
            load_hciv_training_examples,
        )

        data_path = _write_hciv_training_jsonl(tmp_path)
        payload = json.loads(data_path.read_text(encoding="utf-8"))
        payload["target_hciv"] = [0.0] * 9
        data_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

        with pytest.raises(ValueError, match="Lorentz"):
            load_hciv_training_examples(data_path, dim=8)

    def test_train_hciv_encoder_checkpoint_writes_loadable_artifact(
        self,
        tmp_path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from cig_compiler_svc.domain.compiler import CIGCompiler
        from cig_compiler_svc.domain.hciv_training import (
            train_hciv_encoder_checkpoint,
        )

        data_path = _write_hciv_training_jsonl(tmp_path)
        checkpoint_path = tmp_path / "hciv.pt"
        manifest_path = tmp_path / "hciv.manifest.json"

        train_hciv_encoder_checkpoint(
            data_path,
            checkpoint_path,
            manifest_path=manifest_path,
            dim=8,
            epochs=1,
            batch_size=1,
            device="cpu",
        )
        monkeypatch.setenv("HCIV_CHECKPOINT_PATH", str(checkpoint_path))
        compiler = CIGCompiler(
            hciv_dim=8,
            semantic_parser=lambda _: {
                "properties": [{"name": "qed", "direction": "maximize"}]
            },
            enable_grounding=False,
        )
        _, hciv, cone = _run(compiler.compile("maximize QED", seed=7))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        assert checkpoint_path.exists()
        assert manifest["schema"] == "moleculeforge.cig_compiler.hciv_encoder.v1"
        assert manifest["example_count"] == 1
        assert len(hciv.coordinates) == 9
        assert cone.apex == hciv

    def test_train_hciv_encoder_cli_writes_checkpoint_and_manifest(
        self,
        tmp_path,
    ) -> None:
        data_path = _write_hciv_training_jsonl(tmp_path)
        checkpoint_path = tmp_path / "hciv.pt"
        manifest_path = tmp_path / "hciv.manifest.json"
        module = _load_hciv_train_module()

        exit_code = module.main(
            [
                "--data",
                str(data_path),
                "--output-checkpoint",
                str(checkpoint_path),
                "--manifest",
                str(manifest_path),
                "--dim",
                "8",
                "--epochs",
                "1",
                "--batch-size",
                "1",
                "--device",
                "cpu",
            ]
        )

        assert exit_code == 0
        assert checkpoint_path.exists()
        assert manifest_path.exists()

    def test_hash_encoding(self) -> None:
        """Test hash-based HCIV encoding for local_demo support."""
        from cig_compiler_svc.domain.hciv_encoder import hash_encode_hciv
        from cig_compiler_svc.domain.stages.stage2_cig_build import build_cig

        extracted = {
            "properties": [{"name": "qed", "direction": "maximize", "priority": 1}],
            "constraints": {},
        }
        cig = build_cig(extracted, "test")
        hciv = hash_encode_hciv(cig, dim=8, seed=42)
        assert len(hciv.coordinates) == 9  # dim + 1 for Lorentz
        assert hciv.coordinates[0] > 0  # time component positive

    def test_hash_encoding_changes_when_edges_change(self) -> None:
        from cig_compiler_svc.domain.hciv_encoder import hash_encode_hciv
        from mf_core.types.cig import (
            ChemicalIntentGraph,
            ObjectiveEdge,
            ObjectiveNode,
            ObjectiveType,
        )

        objectives = [
            ObjectiveNode(
                id="obj_affinity",
                type=ObjectiveType.CONTINUOUS_MAXIMIZE,
                oracle="boltz2",
                weight=0.5,
            ),
            ObjectiveNode(
                id="obj_admet_bundle",
                type=ObjectiveType.MULTI_CONSTRAINT_SATISFY,
                oracle="admet_ai",
                weight=0.5,
            ),
        ]
        cig_without_edges = ChemicalIntentGraph(
            intent_id="cig-test",
            objective_nodes=objectives,
            source_user_input="test",
        )
        cig_with_edges = ChemicalIntentGraph(
            intent_id="cig-test",
            objective_nodes=objectives,
            edges=[
                ObjectiveEdge(
                    source_id="obj_affinity",
                    target_id="obj_admet_bundle",
                    relation="trade_off",
                    strength=-0.5,
                )
            ],
            source_user_input="test",
        )

        assert hash_encode_hciv(
            cig_without_edges,
            dim=8,
            seed=42,
        ).coordinates != hash_encode_hciv(cig_with_edges, dim=8, seed=42).coordinates

    def test_hciv_features_include_edge_topology(self) -> None:
        from cig_compiler_svc.domain.hciv_encoder import cig_to_features
        from mf_core.types.cig import (
            ChemicalIntentGraph,
            ObjectiveEdge,
            ObjectiveHyperedge,
            ObjectiveNode,
            ObjectiveType,
        )

        cig = ChemicalIntentGraph(
            intent_id="cig-test",
            objective_nodes=[
                ObjectiveNode(
                    id="obj_affinity",
                    type=ObjectiveType.CONTINUOUS_MAXIMIZE,
                    oracle="boltz2",
                    weight=0.5,
                ),
                ObjectiveNode(
                    id="obj_admet_bundle",
                    type=ObjectiveType.MULTI_CONSTRAINT_SATISFY,
                    oracle="admet_ai",
                    weight=0.5,
                ),
            ],
            edges=[
                ObjectiveEdge(
                    source_id="obj_affinity",
                    target_id="obj_admet_bundle",
                    relation="trade_off",
                    strength=-0.5,
                )
            ],
            hyperedges=[
                ObjectiveHyperedge(
                    source_ids=["obj_affinity"],
                    target_ids=["obj_admet_bundle"],
                    relation="trade_off",
                    strength=-0.5,
                )
            ],
            source_user_input="test",
        )

        features = cig_to_features(cig)

        assert features[31] == 1.0
        assert features[32] == -0.5
        assert features[33] == 1.0
        assert features[34] == 1.0
        assert features[35] == -0.5
        assert features[36] == 2.0

    def test_hciv_encoder_distinguishes_directed_hypergraph_topology(self) -> None:
        import torch
        from cig_compiler_svc.domain.hciv_encoder import HCIVEncoder, cig_to_features
        from mf_core.types.cig import (
            ChemicalIntentGraph,
            ObjectiveEdge,
            ObjectiveHyperedge,
            ObjectiveNode,
            ObjectiveType,
        )

        objectives = [
            ObjectiveNode(
                id="obj_affinity",
                type=ObjectiveType.CONTINUOUS_MAXIMIZE,
                oracle="boltz2",
                weight=0.5,
            ),
            ObjectiveNode(
                id="obj_admet_bundle",
                type=ObjectiveType.MULTI_CONSTRAINT_SATISFY,
                oracle="admet_ai",
                weight=0.5,
            ),
        ]
        forward = ChemicalIntentGraph(
            intent_id="cig-test",
            objective_nodes=objectives,
            edges=[
                ObjectiveEdge(
                    source_id="obj_affinity",
                    target_id="obj_admet_bundle",
                    relation="trade_off",
                    strength=-0.5,
                )
            ],
            hyperedges=[
                ObjectiveHyperedge(
                    source_ids=["obj_affinity"],
                    target_ids=["obj_admet_bundle"],
                    relation="trade_off",
                    strength=-0.5,
                )
            ],
            source_user_input="test",
        )
        reverse = ChemicalIntentGraph(
            intent_id="cig-test",
            objective_nodes=objectives,
            edges=[
                ObjectiveEdge(
                    source_id="obj_admet_bundle",
                    target_id="obj_affinity",
                    relation="trade_off",
                    strength=-0.5,
                )
            ],
            hyperedges=[
                ObjectiveHyperedge(
                    source_ids=["obj_admet_bundle"],
                    target_ids=["obj_affinity"],
                    relation="trade_off",
                    strength=-0.5,
                )
            ],
            source_user_input="test",
        )

        assert torch.equal(cig_to_features(forward), cig_to_features(reverse))
        torch.manual_seed(7)
        encoder = HCIVEncoder(dim=8)

        forward_hciv, _ = encoder.encode(forward)
        reverse_hciv, _ = encoder.encode(reverse)

        assert forward_hciv.coordinates != reverse_hciv.coordinates

    def test_random_encoding_mode(self) -> None:
        """Test RANDOM encoding mode dispatch in CIGCompiler."""
        from cig_compiler_svc.domain.compiler import CIGCompiler, EncodingMode

        assert EncodingMode.RANDOM.value == "random"
        compiler = CIGCompiler(mode="local_demo", encoding_mode="random")
        cig, hciv, cone = _run(compiler.compile("Design a drug-like molecule", seed=42))
        assert cig.intent_id.startswith("cig-")
        assert len(hciv.coordinates) > 0
        assert cone.apex is not None

    def test_encoding_mode_enum(self) -> None:
        """Test that all three encoding modes are defined."""
        from cig_compiler_svc.domain.compiler import EncodingMode

        assert EncodingMode.LEARNED.value == "learned"
        assert EncodingMode.HASH.value == "hash"
        assert EncodingMode.RANDOM.value == "random"
