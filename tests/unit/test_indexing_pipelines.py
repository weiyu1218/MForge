"""Unit tests for indexing pipeline fail-fast behavior."""

from __future__ import annotations

import asyncio
import json

import pytest


def _run(coro):
    return asyncio.run(coro)


class RecordingIndexClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[dict]]] = []

    def insert(self, collection: str, records: list[dict]) -> int:
        self.calls.append((collection, records))
        return len(records)


class RecordingSearchClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[float], int]] = []

    def search(self, collection: str, query_vector: list[float], limit: int = 1) -> list[dict]:
        self.calls.append((collection, query_vector, limit))
        return [
            {
                "patent_id": "US1111111",
                "distance": 0.12,
                "claim_evidence": "claim 1 covers ethanol analogs",
                "source": "surechembl",
            }
        ]


class DeterministicHUMUEncoder:
    def encode_smiles(self, smiles: str) -> list[float]:
        return [float(len(smiles)), 1.0, 0.5]


class RecordingDeadZoneUpdater:
    async def refresh(self, config: dict) -> dict:
        return {"zones_affected": 1, "config": config}


def test_patent_indexing_requires_surechembl_path() -> None:
    from patent_indexing.pipeline import index_surechembl_to_vector_store

    with pytest.raises(FileNotFoundError, match="surechembl_path"):
        _run(index_surechembl_to_vector_store({"vector_client": RecordingIndexClient()}))


def test_patent_run_indexes_real_records(tmp_path) -> None:
    from patent_indexing.pipeline import run

    surechembl_dir = tmp_path / "surechembl"
    surechembl_dir.mkdir()
    (surechembl_dir / "patents.smi").write_text(
        "CCO US1111111\nCCN US2222222\n",
        encoding="utf-8",
    )

    uspto_dir = tmp_path / "uspto"
    uspto_dir.mkdir()
    (uspto_dir / "grants.smi").write_text(
        "CCCl US3333333\n",
        encoding="utf-8",
    )

    client = RecordingIndexClient()
    result = _run(
        run(
            {
                "surechembl_path": str(surechembl_dir),
                "uspto_path": str(uspto_dir),
                "vector_client": client,
                "humu_encoder": DeterministicHUMUEncoder(),
                "dead_zone_updater": RecordingDeadZoneUpdater(),
                "batch_size": 1,
            }
        )
    )

    assert result["status"] == "completed"
    assert result["surechembl"]["molecules_indexed"] == 2
    assert result["surechembl"]["batches"] == 2
    assert result["uspto"]["structures_extracted"] == 1
    assert result["dead_zone"]["status"] == "completed"
    assert sum(len(records) for _, records in client.calls) == 3


def test_patent_indexing_writes_claim_evidence_and_humu_vectors(tmp_path) -> None:
    from patent_indexing.pipeline import index_surechembl_to_vector_store

    source_dir = tmp_path / "surechembl"
    source_dir.mkdir()
    (source_dir / "patents.tsv").write_text(
        "smiles\tpatent_id\tclaim_evidence\tsource\n"
        "CCO\tUS1111111\tclaim 1 covers ethanol analogs\tsurechembl\n",
        encoding="utf-8",
    )

    client = RecordingIndexClient()
    result = _run(
        index_surechembl_to_vector_store(
            {
                "surechembl_path": str(source_dir),
                "vector_client": client,
                "humu_encoder": DeterministicHUMUEncoder(),
            }
        )
    )

    assert result["molecules_indexed"] == 1
    inserted = client.calls[0][1][0]
    assert inserted["smiles"] == "CCO"
    assert inserted["patent_id"] == "US1111111"
    assert inserted["claim_evidence"] == "claim 1 covers ethanol analogs"
    assert inserted["source"] == "surechembl"
    assert inserted["z_humu"] == [3.0, 1.0, 0.5]


def test_patent_similarity_search_uses_index_client_hits() -> None:
    from patent_indexing.pipeline import search_patent_similarity

    client = RecordingSearchClient()
    result = _run(
        search_patent_similarity(
            "CCO",
            {
                "vector_client": client,
                "humu_encoder": DeterministicHUMUEncoder(),
                "vector_collection": "patents_embedding",
            },
        )
    )

    assert client.calls == [("patents_embedding", [3.0, 1.0, 0.5], 1)]
    assert result["nearest_patent_id"] == "US1111111"
    assert result["nearest_patent_distance"] == 0.12
    assert result["claim_evidence"] == "claim 1 covers ethanol analogs"
    assert result["fto_risk"] == "requires_review"


def test_reaction_indexing_requires_source_path() -> None:
    from reaction_indexing.pipeline import extract_reaction_templates

    with pytest.raises(FileNotFoundError, match="source_paths\\[uspto\\]"):
        _run(extract_reaction_templates({"sources": ["uspto"], "source_paths": {}}))


def test_reaction_indexing_extracts_and_indexes_templates(tmp_path) -> None:
    from reaction_indexing.pipeline import extract_reaction_templates, index_reactions

    source_dir = tmp_path / "uspto"
    source_dir.mkdir()
    (source_dir / "templates.tsv").write_text(
        "template\tfrequency\n[C:1][O:2]>>[C:1]=[O:2]\t7\n",
        encoding="utf-8",
    )
    retropath_dir = tmp_path / "retropath"
    retropath_dir.mkdir()
    (retropath_dir / "templates.tsv").write_text(
        "template\tfrequency\n[N:1][C:2]>>[N:1].[C:2]\t4\n",
        encoding="utf-8",
    )

    templates = _run(
        extract_reaction_templates(
            {
                "sources": ["uspto", "retropath"],
                "source_paths": {
                    "uspto": str(source_dir),
                    "retropath": str(retropath_dir),
                },
            }
        )
    )
    manifest_path = tmp_path / "reaction_template_manifest.json"
    index = _run(
        index_reactions(
            templates,
            {"index_type": "fingerprint", "manifest_path": str(manifest_path)},
        )
    )

    assert templates["status"] == "completed"
    assert templates["total_templates"] == 2
    assert templates["sources"]["uspto"]["templates_extracted"] == 1
    assert templates["sources"]["retropath"]["templates_extracted"] == 1
    assert index["status"] == "completed"
    assert index["templates_indexed"] == 2
    assert index["index_size_bytes"] > 0
    assert index["manifest_path"] == str(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["source_hashes"]["uspto"]
    assert manifest["source_hashes"]["retropath"]
    assert manifest["templates"][0]["template_smarts"] == "[C:1][O:2]>>[C:1]=[O:2]"
    assert manifest["templates"][0]["source"] == "uspto"


class IncompleteRouteRunner:
    def find_routes(self, smiles: str, max_routes: int = 10) -> list[dict]:
        return [
            {
                "route_id": "route-1",
                "steps": [
                    {
                        "reaction": "amide coupling",
                        "reactants": [{"smiles": "CCO"}],
                    }
                ],
            }
        ]


class CompleteRouteRunner:
    def find_routes(self, smiles: str, max_routes: int = 10) -> list[dict]:
        return [
            {
                "route_id": "route-1",
                "smiles": smiles,
                "steps": [
                    {
                        "step_id": "retro-1",
                        "reaction": "CCO.O=O>>CCOO",
                        "reactants": [{"smiles": "CCO"}, {"smiles": "O=O"}],
                        "conditions": {"temperature_C": 25, "time_h": 2},
                        "building_blocks": [{"smiles": "CCO", "source": "enamine_real"}],
                    }
                ],
            }
        ][:max_routes]


@pytest.mark.parametrize(
    ("module_path", "class_name"),
    [
        ("mf_retrosyn.aizynth.retrosyn", "AiZynthRetrosyn"),
        ("mf_retrosyn.rsgpt.retrosyn", "RSGPTRetrosyn"),
        ("mf_retrosyn.ualign.retrosyn", "UAlignRetrosyn"),
    ],
)
def test_retrosyn_wrappers_require_complete_route_steps(module_path, class_name) -> None:
    import importlib

    cls = getattr(importlib.import_module(module_path), class_name)

    with pytest.raises(ValueError, match="conditions"):
        _run(cls(runner=IncompleteRouteRunner()).find_routes("CCOO", max_routes=1))

    routes = _run(cls(runner=CompleteRouteRunner()).find_routes("CCOO", max_routes=1))
    assert routes[0]["steps"][0]["reaction"] == "CCO.O=O>>CCOO"
    assert routes[0]["steps"][0]["building_blocks"][0]["source"] == "enamine_real"


def test_aizynth_from_env_requires_config(monkeypatch: pytest.MonkeyPatch) -> None:
    from mf_retrosyn.aizynth.retrosyn import AiZynthRetrosyn

    monkeypatch.delenv("AIZYNTH_CONFIG_PATH", raising=False)

    with pytest.raises(RuntimeError, match="AIZYNTH_CONFIG_PATH"):
        AiZynthRetrosyn.from_env()


def test_retrosyn_agent_uses_planner_routes() -> None:
    from retrosyn_agent.agent import RetroSynAgent

    class Planner:
        def __init__(self) -> None:
            self.calls: list[tuple[str, int]] = []

        async def find_routes(self, smiles: str, max_routes: int = 10) -> list[dict]:
            self.calls.append((smiles, max_routes))
            return CompleteRouteRunner().find_routes(smiles, max_routes=max_routes)

    planner = Planner()
    agent = RetroSynAgent(planner=planner)
    result = _run(agent.process({"smiles": "CCOO", "max_routes": 1}))

    assert planner.calls == [("CCOO", 1)]
    assert result["status"] == "planned"
    assert result["target_smiles"] == "CCOO"
    assert result["routes"][0]["route_id"] == "route-1"
    assert result["routes"][0]["steps"][0]["reaction"] == "CCO.O=O>>CCOO"
