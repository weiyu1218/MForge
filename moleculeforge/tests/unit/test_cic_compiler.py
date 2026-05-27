"""Unit tests for the CIC compiler."""

from __future__ import annotations

import asyncio

import pytest


def _run(coro):
    return asyncio.run(coro)


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

        async def fake_query_surechembl_patents(
            query: str,
            patent_ids: list[str] | None = None,
            limit: int = 5,
        ) -> list[dict]:
            assert query == "KRAS"
            assert patent_ids == ["WO2020000001"]
            return [
                {
                    "patent_id": "WO2020000001",
                    "title": "KRAS inhibitors",
                    "source_timestamp": "2026-05-15T00:00:03Z",
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
        monkeypatch.setattr(
            "cig_compiler_svc.domain.tools.surechembl_tool.query_surechembl_patents",
            fake_query_surechembl_patents,
        )

        enriched = _run(
            ground_knowledge(
                {
                    "targets": [{"name": "KRAS"}],
                    "ip_constraints": {
                        "blocked_patent_ids": ["WO2020000001"],
                        "fto_required": True,
                    },
                },
                sources=ALL_GROUNDING_SOURCES,
            )
        )

        evidence = enriched["_grounding_evidence"]
        assert {item["source"] for item in evidence} == {
            "uniprot",
            "pdb",
            "chembl",
            "surechembl",
        }
        assert all(item["source_timestamp"] for item in evidence)
        assert enriched["_grounded_uniprot_ids"] == ["P01116"]
        assert enriched["_grounded_pdb_ids"] == ["6OIM"]
        assert enriched["_grounded_chembl_target_ids"] == ["CHEMBL612545"]
        assert enriched["_grounded_patent_ids"] == ["WO2020000001"]

    def test_default_production_mode_requires_real_parser(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from cig_compiler_svc.domain.compiler import CIGCompiler

        monkeypatch.delenv("CIG_SEMANTIC_PARSER_URI", raising=False)
        compiler = CIGCompiler()

        with pytest.raises(RuntimeError, match="CIG_SEMANTIC_PARSER_URI"):
            _run(compiler.compile("Design a drug-like molecule", seed=42))

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
