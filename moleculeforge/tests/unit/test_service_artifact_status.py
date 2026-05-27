"""Service artifact status reporting."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from mf_core.types.molecule import Molecule

ROOT = Path(__file__).resolve().parents[2]


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.asyncio
async def test_feature_store_health_reports_missing_feast_repo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module(
        "feature_store_status_test",
        ROOT / "services/feature-store-svc/src/feature_store_svc/main.py",
    )
    monkeypatch.delenv("FEAST_REPO_PATH", raising=False)

    with pytest.raises(HTTPException) as exc:
        await module.health()

    assert exc.value.status_code == 503
    assert exc.value.detail["artifact_status"][0]["name"] == "feast_repo"
    assert exc.value.detail["artifact_status"][0]["available"] is False


class _RecordingFeatureStore:
    def __init__(self) -> None:
        self.online_calls: list[dict] = []

    def get_online_features(self, *, features: list[str], entity_rows: list[dict]):
        self.online_calls.append({"features": features, "entity_rows": entity_rows})
        return {"rows": [{"entity_id": "mol-1", "qed": 0.8}]}

    def list_feature_views(self):
        return [{"name": "molecule_features", "features": ["qed"]}]


@pytest.mark.asyncio
async def test_feature_store_online_features_delegate_to_feast_store(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo_path = tmp_path / "feast_repo"
    repo_path.mkdir()
    monkeypatch.setenv("FEAST_REPO_PATH", str(repo_path))
    module = _load_module(
        "feature_store_online_test",
        ROOT / "services/feature-store-svc/src/feature_store_svc/main.py",
    )
    store = _RecordingFeatureStore()
    module.app.state.feast_store = store

    response = await module.get_online_features(
        module.FeatureRequest(
            entities=["mol-1"],
            features=["qed"],
            feature_view="molecule_features",
        )
    )
    views = await module.list_feature_views()

    assert store.online_calls == [
        {
            "features": ["molecule_features:qed"],
            "entity_rows": [{"entity_id": "mol-1"}],
        }
    ]
    assert response["rows"][0]["qed"] == 0.8
    assert views["feature_views"][0]["name"] == "molecule_features"


def test_admet_runtime_status_reports_missing_model(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module(
        "admet_status_test",
        ROOT / "services/admet-svc/src/admet_svc/main.py",
    )
    monkeypatch.delenv("ADMET_MODEL_PATH", raising=False)

    statuses = module.runtime_status()

    assert statuses[0]["name"] == "admet_model"
    assert statuses[0]["available"] is False


@pytest.mark.asyncio
async def test_admet_service_predict_uses_http_runner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_module(
        "admet_predict_runner_test",
        ROOT / "services/admet-svc/src/admet_svc/main.py",
    )
    model_dir = tmp_path / "admet-models"
    model_dir.mkdir()
    monkeypatch.setenv("ADMET_MODEL_PATH", str(model_dir))
    monkeypatch.setenv("ADMET_SERVICE_URL", "http://admet.local")
    monkeypatch.setenv("ADMET_TARGETS", "clearance,herg")

    class Runner:
        def __init__(self) -> None:
            self.rows = None
            self.properties = None

        def evaluate(self, rows, properties):
            self.rows = rows
            self.properties = properties
            return {rows[0]["smiles"]: {"clearance": 1.5}}

    runner = Runner()
    response = await module.ADMETServicer(runner=runner).Predict(
        SimpleNamespace(smiles="CCO", properties=["clearance"]),
        None,
    )

    assert runner.rows[0]["smiles"] == "CCO"
    assert runner.properties == ["clearance"]
    assert response.predictions == {"clearance": 1.5}


def test_fragfm_service_builds_generator_with_trained_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import torch
    from mf_generators.fragfm.model.sa_aware_rate_matrix import SAAwareRateMatrix
    from mf_generators.fragfm.model.two_level_dfm import TwoLevelDFM

    module = _load_module(
        "fragfm_service_artifact_test",
        ROOT / "services/fragfm-generator-svc/src/fragfm_generator_svc/main.py",
    )
    vocab_path = tmp_path / "fragfm_vocab.json"
    vocab_path.write_text(
        json.dumps(
            {
                "fragments": ["CC", "O"],
                "assembly_rules": [
                    {
                        "id": "ethanol",
                        "fragments": ["CC", "O"],
                        "product": "CCO",
                        "sa_score_bin": 2,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    checkpoint_path = tmp_path / "best_model.pt"
    torch.save(TwoLevelDFM(vocab_size=2).state_dict(), checkpoint_path)
    rate_matrix_path = tmp_path / "rate_matrix.pt"
    torch.save(SAAwareRateMatrix(vocab_size=2).state_dict(), rate_matrix_path)
    monkeypatch.setenv("FRAGFM_VOCAB_PATH", str(vocab_path))
    monkeypatch.setenv("FRAGFM_CHECKPOINT_PATH", str(checkpoint_path))
    monkeypatch.setenv("FRAGFM_RATE_MATRIX_PATH", str(rate_matrix_path))

    generator = module._build_generator()

    assert generator.checkpoint_path == str(checkpoint_path)
    assert generator.rate_matrix_path == str(rate_matrix_path)
    assert generator._model is not None


def test_mmpt_service_builds_generator_with_trained_index(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_module(
        "mmpt_service_artifact_test",
        ROOT / "services/mmpt-generator-svc/src/mmpt_generator_svc/main.py",
    )
    index_path = tmp_path / "mmpt_index.json"
    index_path.write_text(
        json.dumps(
            {
                "schema_version": "mmpt_mmp_index.v1",
                "transforms": [
                    {
                        "id": "fluoro_to_chloro",
                        "pattern": "F",
                        "replacement": "Cl",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("MMPT_INDEX_URI", f"file://{index_path}")

    generator = module._build_generator()

    assert generator.index_path == str(index_path)
    assert generator.mmp_database == [
        {
            "id": "fluoro_to_chloro",
            "pattern": "F",
            "replacement": "Cl",
        }
    ]


@pytest.mark.asyncio
async def test_humu_encoder_service_loads_checkpoint_and_routes_molecule(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import torch
    from mf_encoders.humu_mol.encoder import HUMUMoleculeEncoder
    from mf_encoders.humu_pocket.encoder import HUMUPocketEncoder
    from mf_encoders.humu_route.encoder import HUMURouteEncoder

    checkpoint_path = tmp_path / "humu.pt"
    torch.save(
        {
            "encoder_mol": HUMUMoleculeEncoder(dim=128).state_dict(),
            "encoder_pocket": HUMUPocketEncoder(dim=128).state_dict(),
            "encoder_route": HUMURouteEncoder(dim=128).state_dict(),
        },
        checkpoint_path,
    )
    monkeypatch.setenv("HUMU_CHECKPOINT_PATH", str(checkpoint_path))
    module = _load_module(
        "humu_encoder_checkpoint_test",
        ROOT / "services/humu-encoder-svc/src/humu_encoder_svc/main.py",
    )
    service = module.HUMUEncoderServicer()

    response = await service.Encode(SimpleNamespace(input_type="molecule", smiles="CCO"), None)

    assert response.input_type == "molecule"
    assert response.checkpoint_path == str(checkpoint_path)
    assert len(response.embedding) == 129


class _RecordingQdrantClient:
    def __init__(self) -> None:
        self.upserts: list[dict] = []
        self.searches: list[dict] = []
        self.deletes: list[list[str]] = []

    async def upsert(self, data: dict[str, list]) -> int:
        self.upserts.append(data)
        return len(data["id"])

    async def search(self, vector: list[float], top_k: int = 10, output_fields=None) -> list[dict]:
        self.searches.append(
            {"vector": vector, "top_k": top_k, "output_fields": output_fields}
        )
        return [{"id": "mol-1", "distance": 0.1, "entity": {"smiles": "CCO"}}]

    async def delete(self, ids: list[str]) -> int:
        self.deletes.append(ids)
        return len(ids)

    async def get_stats(self, collection: str) -> dict:
        return {"collection": collection, "row_count": 1}


@pytest.mark.asyncio
async def test_humu_index_service_delegates_to_qdrant_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module(
        "humu_index_client_test",
        ROOT / "services/humu-index-svc/src/humu_index_svc/main.py",
    )
    monkeypatch.setenv("QDRANT_URL", "http://localhost:16333")
    client = _RecordingQdrantClient()
    module.app.state.qdrant_client = client

    inserted = await module.insert_vectors(
        module.IndexRequest(
            ids=["mol-1"],
            vectors=[[0.1, 0.2, 0.3]],
            collection="humu",
            metadata={"smiles": ["CCO"]},
        )
    )
    searched = await module.search_vectors(
        module.SearchRequest(query_vector=[0.1, 0.2, 0.3], collection="humu", top_k=1)
    )
    deleted = await module.delete_vectors(module.DeleteRequest(ids=["mol-1"], collection="humu"))
    stats = await module.collection_stats("humu")

    assert inserted["inserted"] == 1
    assert client.upserts[0]["id"] == ["mol-1"]
    assert client.upserts[0]["vector"] == [[0.1, 0.2, 0.3]]
    assert client.upserts[0]["smiles"] == ["CCO"]
    assert searched["results"][0]["id"] == "mol-1"
    assert deleted["deleted"] == 1
    assert stats["backend"] == "qdrant"
    assert stats["row_count"] == 1


class _RecordingGenerator:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def generate(self, batch_size: int, intent_cone=None, **kwargs):
        self.calls.append(
            {
                "batch_size": batch_size,
                "intent_cone": intent_cone,
                "kwargs": kwargs,
            }
        )
        return [Molecule(smiles="CCO") for _ in range(batch_size)]


class _RecordingMMPTRAGGenerator:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def generate(self, hciv, cone, cig, n_samples: int = 10, seed: int | None = None):
        self.calls.append(
            {
                "hciv": hciv,
                "cone": cone,
                "cig": cig,
                "n_samples": n_samples,
                "seed": seed,
            }
        )
        for _ in range(n_samples):
            yield Molecule(smiles="CCO")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("module_name", "service_path", "class_name", "env_vars", "generator_name"),
    [
        (
            "hfm_generator_delegate_test",
            ROOT / "services/hfm-generator-svc/src/hfm_generator_svc/main.py",
            "HFMGeneratorServicer",
            ("HFM_CHECKPOINT_PATH", "HFM_DECODER_PATH"),
            "hfm_3d",
        ),
        (
            "fragfm_generator_delegate_test",
            ROOT / "services/fragfm-generator-svc/src/fragfm_generator_svc/main.py",
            "FragFMGeneratorServicer",
            ("FRAGFM_VOCAB_PATH",),
            "fragfm",
        ),
        (
            "crem_generator_delegate_test",
            ROOT / "services/crem-generator-svc/src/crem_generator_svc/main.py",
            "CReMGeneratorServicer",
            ("CREM_MMP_DB_PATH",),
            "crem_3d",
        ),
    ],
)
async def test_generator_service_delegates_to_model_object(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    module_name: str,
    service_path: Path,
    class_name: str,
    env_vars: tuple[str, ...],
    generator_name: str,
) -> None:
    for env_var in env_vars:
        artifact_path = tmp_path / f"{env_var.lower()}.dat"
        artifact_path.write_text("artifact", encoding="utf-8")
        monkeypatch.setenv(env_var, str(artifact_path))
    module = _load_module(module_name, service_path)
    generator = _RecordingGenerator()
    service = getattr(module, class_name)(generator=generator)
    request = SimpleNamespace(
        project_id="project-1",
        batch_size=2,
        generator_params={"temperature": "0.1"},
    )

    response = await service.Generate(request, None)

    assert generator.calls == [
        {
            "batch_size": 2,
            "intent_cone": None,
            "kwargs": {"temperature": "0.1"},
        }
    ]
    assert response.generator_name == generator_name
    assert response.generation_id == "project-1"
    assert len(response.molecules) == 2
    assert json.loads(response.molecules[0].decode("utf-8"))["smiles"] == "CCO"


@pytest.mark.asyncio
async def test_mmpt_generator_service_delegates_to_model_object(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    artifact_path = tmp_path / "mmpt_index.json"
    artifact_path.write_text(
        json.dumps({"transforms": [{"pattern": "F", "replacement": "Cl"}]}),
        encoding="utf-8",
    )
    monkeypatch.setenv("MMPT_INDEX_URI", f"file://{artifact_path}")
    module = _load_module(
        "mmpt_generator_delegate_test",
        ROOT / "services/mmpt-generator-svc/src/mmpt_generator_svc/main.py",
    )
    generator = _RecordingMMPTRAGGenerator()
    service = module.MMPTGeneratorServicer(generator=generator)
    request = SimpleNamespace(
        project_id="project-1",
        batch_size=2,
        generator_params={"seed": "7"},
    )

    response = await service.Generate(request, None)

    assert generator.calls == [
        {
            "hciv": None,
            "cone": None,
            "cig": None,
            "n_samples": 2,
            "seed": 7,
        }
    ]
    assert response.generator_name == "mmpt_rag"
    assert response.generation_id == "project-1"
    assert len(response.molecules) == 2
    assert json.loads(response.molecules[0].decode("utf-8"))["smiles"] == "CCO"


@pytest.mark.asyncio
async def test_supply_service_uses_file_catalog_with_source_timestamp(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    catalog_path = tmp_path / "supply_catalog.json"
    catalog_path.write_text(
        json.dumps(
            [
                {
                    "smiles": "CCO",
                    "catalog_id": "ENA-REAL-1",
                    "source": "enamine_real",
                    "source_timestamp": "2026-05-01T00:00:00Z",
                    "available": True,
                    "price": 12.5,
                    "currency": "USD",
                    "lead_time_days": 3,
                }
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("SUPPLY_CATALOG_URI", catalog_path.as_uri())
    module = _load_module(
        "supply_catalog_file_test",
        ROOT / "services/supply-oracle-svc/src/supply_oracle_svc/main.py",
    )
    service = module.SupplyOracleServicer()

    response = await service.CheckAvailability(SimpleNamespace(smiles="CCO"), None)

    assert response.available is True
    assert response.catalog_id == "ENA-REAL-1"
    assert response.catalog_source == "enamine_real"
    assert response.source_timestamp == "2026-05-01T00:00:00Z"
    assert response.price == 12.5
    assert response.lead_time_days == 3


@pytest.mark.asyncio
async def test_fto_service_uses_file_patent_index_with_claim_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    index_path = tmp_path / "patent_index.json"
    index_path.write_text(
        json.dumps(
            [
                {
                    "smiles": "CCO",
                    "patent_id": "US1111111",
                    "claim_evidence": "claim 1 covers ethanol analogs",
                    "source": "surechembl",
                    "similarity": 0.91,
                }
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("PATENT_INDEX_URI", index_path.as_uri())
    module = _load_module(
        "fto_patent_file_test",
        ROOT / "services/fto-patent-svc/src/fto_patent_svc/main.py",
    )
    service = module.FTOPatentServicer()

    response = await service.SearchPatents(SimpleNamespace(smiles="CCO"), None)

    assert response.verdict == "requires_review"
    assert response.patent_hits == 1
    assert response.hits[0]["patent_id"] == "US1111111"
    assert response.hits[0]["claim_evidence"] == "claim 1 covers ethanol analogs"
    assert response.hits[0]["source"] == "surechembl"


@pytest.mark.asyncio
async def test_orchestrator_service_tracks_real_workflow_state() -> None:
    module = _load_module(
        "orchestrator_state_test",
        ROOT / "services/orchestrator-svc/src/orchestrator_svc/main.py",
    )

    started = await module.start_design(
        {
            "nl_input": "Design KRAS G12C inhibitor",
            "run_id": "run-orch-1",
            "trace_id": "trace-orch-1",
            "artifact_ids": ["artifact-seed-1"],
            "validation_passed": False,
            "max_refinements": 0,
        }
    )
    status = await module.get_design_status(started["design_id"])
    paused = await module.pause_design(started["design_id"])
    resumed = await module.resume_design(started["design_id"])

    assert started["status"] == "completed"
    assert started["run_id"] == "run-orch-1"
    assert started["trace_id"] == "trace-orch-1"
    assert started["artifact_ids"] == ["artifact-seed-1"]
    assert started["history"] == [
        "PLANNING",
        "GENERATING",
        "VALIDATING",
        "ESCALATING",
    ]
    assert status["current_stage"] == "ESCALATING"
    assert status["run_id"] == "run-orch-1"
    assert status["trace_id"] == "trace-orch-1"
    assert status["artifact_ids"] == ["artifact-seed-1"]
    assert status["history"] == [
        "PLANNING",
        "GENERATING",
        "VALIDATING",
        "ESCALATING",
    ]
    assert status["state"]["history"] == [
        "PLANNING",
        "GENERATING",
        "VALIDATING",
        "ESCALATING",
    ]
    assert "molecules_generated" not in status["state"]
    assert paused["status"] == "paused"
    assert resumed["status"] == "completed"

    with pytest.raises(HTTPException) as exc:
        await module.get_design_status("missing-design")
    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_orchestrator_engineering_workflow_calls_injected_clients() -> None:
    module = _load_module(
        "orchestrator_engineering_workflow_test",
        ROOT / "services/orchestrator-svc/src/orchestrator_svc/main.py",
    )

    class Clients:
        async def compile_intent(self, state):
            return {"cig": {"source": state["nl_input"]}, "hciv": {}, "intent_cone": {}}

        async def generate_candidates(self, state):
            return [{"smiles": "CCO", "canonical_smiles": "CCO"}]

        async def validate_candidates(self, state):
            return {"passed": True, "results": [{"smiles": "CCO", "admet_score": 0.8}]}

        async def plan_routes(self, state):
            return {"skipped": True, "reason": "retrosyn resource not configured"}

        async def review_candidates(self, state):
            return {"verdict": "pass", "total_rules": 1}

    started = await module.start_design(
        {
            "nl_input": "Design KRAS G12C inhibitor",
            "workflow_scope": "engineering",
            "clients": Clients(),
            "run_id": "run-orch-engineering-1",
            "trace_id": "trace-orch-engineering-1",
        }
    )

    assert started["status"] == "completed"
    assert started["history"] == [
        "PLANNING",
        "GENERATING",
        "VALIDATING",
        "RETROSYN",
        "CRITIC",
    ]
    assert started["state"]["cig"]["source"] == "Design KRAS G12C inhibitor"
    assert started["state"]["candidates"][0]["canonical_smiles"] == "CCO"
    assert started["state"]["validation"]["passed"] is True
    assert started["state"]["retrosyn"]["skipped"] is True
    assert started["state"]["critic"]["verdict"] == "pass"


@pytest.mark.asyncio
async def test_provenance_service_records_and_returns_actual_chain() -> None:
    module = _load_module(
        "provenance_record_test",
        ROOT / "services/provenance-svc/src/provenance_svc/main.py",
    )

    parent = await module.create_record(
        module.ProvenanceRecord(
            artifact_type="nl_query",
            artifact_id="artifact-parent",
            metadata={"project_id": "project-1", "trace_id": "trace-1"},
        )
    )
    child = await module.create_record(
        module.ProvenanceRecord(
            artifact_type="cig",
            artifact_id="artifact-child",
            parent_ids=["artifact-parent"],
            metadata={"project_id": "project-1", "trace_id": "trace-1"},
        )
    )
    chain = await module.get_provenance("artifact-child")
    audit = await module.audit_project("project-1")
    verified = await module.verify_provenance(
        module.VerifyRequest(
            artifact_id="artifact-child",
            signature=child["signature"],
        )
    )

    assert parent["artifact_id"] == "artifact-parent"
    assert chain["artifact_id"] == "artifact-child"
    assert [node["artifact_id"] for node in chain["chain"]] == [
        "artifact-parent",
        "artifact-child",
    ]
    assert chain["chain"][0]["timestamp"] != "2024-01-01T00:00:00Z"
    assert chain["verified"] is True
    assert audit["total_artifacts"] == 2
    assert audit["verified_count"] == 2
    assert audit["unverified_count"] == 0
    assert verified["signature_valid"] is True

    with pytest.raises(HTTPException) as exc:
        await module.get_provenance("missing-artifact")
    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_provenance_health_rejects_production_without_dki_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PROVENANCE_STORE_MODE", "production_real")
    for env_var in (
        "NEO4J_URI",
        "NEO4J_USER",
        "NEO4J_PASSWORD",
        "PROVENANCE_DATABASE_URL",
        "TEST_DATABASE_URL",
        "MINIO_ENDPOINT_URL",
        "MINIO_ACCESS_KEY",
        "MINIO_SECRET_KEY",
        "MINIO_BUCKET",
    ):
        monkeypatch.delenv(env_var, raising=False)
    module = _load_module(
        "provenance_production_health_test",
        ROOT / "services/provenance-svc/src/provenance_svc/main.py",
    )

    with pytest.raises(HTTPException) as exc:
        await module.health()

    assert exc.value.status_code == 503
    assert exc.value.detail["provenance_store"] == "production_real"
    assert "NEO4J_URI" in exc.value.detail["missing_config"]
    assert "PROVENANCE_DATABASE_URL or TEST_DATABASE_URL" in exc.value.detail["missing_config"]
    assert "MINIO_ENDPOINT_URL" in exc.value.detail["missing_config"]


@pytest.mark.asyncio
async def test_provenance_production_store_writes_run_and_trace_to_backends() -> None:
    module = _load_module(
        "provenance_production_write_test",
        ROOT / "services/provenance-svc/src/provenance_svc/main.py",
    )

    class RecordingGraph:
        def __init__(self) -> None:
            self.artifacts: list[dict] = []
            self.parents: list[tuple[str, str]] = []

        async def write_artifact(
            self,
            *,
            artifact_id: str,
            artifact_type: str,
            project_id: str,
            run_id: str,
            trace_id: str,
            recorded_at: str,
            signature_type: str,
        ) -> None:
            self.artifacts.append(
                {
                    "artifact_id": artifact_id,
                    "artifact_type": artifact_type,
                    "project_id": project_id,
                    "run_id": run_id,
                    "trace_id": trace_id,
                    "recorded_at": recorded_at,
                    "signature_type": signature_type,
                }
            )

        async def write_artifact_parent(self, parent_id: str, child_id: str) -> None:
            self.parents.append((parent_id, child_id))

    class RecordingAuditWriter:
        def __init__(self) -> None:
            self.events: list[dict] = []

        async def write_event(self, stored: dict) -> None:
            self.events.append(stored)

    class RecordingObjectStore:
        def __init__(self) -> None:
            self.objects: list[dict] = []

        async def put_object(
            self,
            object_name: str,
            data: bytes,
            content_type: str,
        ) -> None:
            self.objects.append(
                {
                    "object_name": object_name,
                    "data": data,
                    "content_type": content_type,
                }
            )

    graph = RecordingGraph()
    audit_writer = RecordingAuditWriter()
    object_store = RecordingObjectStore()
    store = module.ProductionProvenanceStore(graph, audit_writer, object_store)

    stored = await store.record(
        module.ProvenanceRecord(
            artifact_type="candidate",
            artifact_id="artifact-1",
            parent_ids=["artifact-parent"],
            metadata={
                "project_id": "project-1",
                "run_id": "run-1",
                "trace_id": "trace-1",
            },
        ),
        {
            "signature": "sig",
            "certificate": None,
            "signature_type": "local_dev_signature",
        },
        "2026-05-19T00:00:00Z",
    )

    assert stored["metadata"]["run_id"] == "run-1"
    assert graph.artifacts[0]["run_id"] == "run-1"
    assert graph.artifacts[0]["trace_id"] == "trace-1"
    assert graph.parents == [("artifact-parent", "artifact-1")]
    assert audit_writer.events[0]["metadata"]["run_id"] == "run-1"
    assert object_store.objects[0]["object_name"] == "provenance/artifact-1.json"
    assert json.loads(object_store.objects[0]["data"])["metadata"]["run_id"] == "run-1"


@pytest.mark.asyncio
async def test_provenance_service_delegates_to_configured_store() -> None:
    module = _load_module(
        "provenance_configured_store_test",
        ROOT / "services/provenance-svc/src/provenance_svc/main.py",
    )

    class RecordingStore:
        store_name = "recording"

        def __init__(self) -> None:
            self.records: list[dict] = []

        async def record(self, record, signed: dict, recorded_at: str) -> dict:
            stored = {
                "artifact_id": record.artifact_id,
                "artifact_type": record.artifact_type,
                "parent_ids": list(record.parent_ids),
                "metadata": dict(record.metadata),
                "signature": signed["signature"],
                "certificate": signed.get("certificate"),
                "recorded_at": recorded_at,
                "signature_type": signed.get("signature_type"),
            }
            self.records.append(stored)
            return stored

        async def get_chain(self, artifact_id: str) -> list[dict]:
            return [record for record in self.records if record["artifact_id"] == artifact_id]

        async def audit(self, project_id: str) -> list[dict]:
            return [
                record
                for record in self.records
                if record["metadata"].get("project_id") == project_id
            ]

        async def child_count(self, artifact_id: str) -> int:
            return sum(artifact_id in record["parent_ids"] for record in self.records)

    store = RecordingStore()
    module.rest_app.state.provenance_store = store

    response = await module.create_record(
        module.ProvenanceRecord(
            artifact_type="candidate",
            artifact_id="artifact-1",
            metadata={"project_id": "project-1", "trace_id": "trace-1"},
        )
    )
    chain = await module.get_provenance("artifact-1")
    audit = await module.audit_project("project-1")

    assert response["artifact_id"] == "artifact-1"
    assert store.records[0]["metadata"]["trace_id"] == "trace-1"
    assert chain["chain"][0]["artifact_id"] == "artifact-1"
    assert audit["total_artifacts"] == 1


@pytest.mark.asyncio
async def test_production_provenance_reads_back_from_dki_adapters() -> None:
    module = _load_module(
        "provenance_production_readback_test",
        ROOT / "services/provenance-svc/src/provenance_svc/main.py",
    )

    parent_record = {
        "artifact_id": "artifact-parent",
        "artifact_type": "nl_query",
        "parent_ids": [],
        "metadata": {"project_id": "project-1", "run_id": "run-1", "trace_id": "trace-1"},
        "signature": "sig-parent",
        "certificate": None,
        "recorded_at": "2026-05-19T00:00:00Z",
        "signature_type": "local_dev_signature",
    }
    child_record = {
        "artifact_id": "artifact-child",
        "artifact_type": "candidate",
        "parent_ids": ["artifact-parent"],
        "metadata": {"project_id": "project-1", "run_id": "run-1", "trace_id": "trace-1"},
        "signature": "sig-child",
        "certificate": None,
        "recorded_at": "2026-05-19T00:01:00Z",
        "signature_type": "local_dev_signature",
    }

    class ReadbackGraph:
        async def get_artifact_chain_ids(self, artifact_id: str) -> list[str]:
            assert artifact_id == "artifact-child"
            return ["artifact-parent", "artifact-child"]

        async def count_artifact_children(self, artifact_id: str) -> int:
            assert artifact_id == "artifact-parent"
            return 1

    class ReadbackAuditWriter:
        async def read_project(self, project_id: str) -> list[dict]:
            assert project_id == "project-1"
            return [parent_record, child_record]

    class ReadbackObjectStore:
        async def get_object(self, object_name: str) -> bytes:
            records = {
                "provenance/artifact-parent.json": parent_record,
                "provenance/artifact-child.json": child_record,
            }
            return json.dumps(records[object_name], sort_keys=True).encode("utf-8")

    store = module.ProductionProvenanceStore(
        ReadbackGraph(),
        ReadbackAuditWriter(),
        ReadbackObjectStore(),
    )

    chain = await store.get_chain("artifact-child")
    audit = await store.audit("project-1")
    children = await store.child_count("artifact-parent")

    assert [record["artifact_id"] for record in chain] == [
        "artifact-parent",
        "artifact-child",
    ]
    assert [record["artifact_id"] for record in audit] == [
        "artifact-parent",
        "artifact-child",
    ]
    assert children == 1


def test_audit_e2e_preflight_lists_missing_requirements(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module(
        "audit_e2e_preflight_test",
        ROOT / "tests/e2e/test_audit_completeness.py",
    )
    for env_var in module.AUDIT_E2E_REQUIRED_ENV:
        monkeypatch.delenv(env_var, raising=False)

    status = module.audit_e2e_preflight_status()

    assert status["ready"] is False
    assert "PROVENANCE_SVC_URL" in status["missing"]
    assert "SIGSTORE_E2E_READY" in status["missing"]
    assert "OTEL_EXPORTER_OTLP_ENDPOINT" in status["missing"]


def test_kras_e2e_preflight_lists_missing_requirements(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module(
        "kras_e2e_preflight_test",
        ROOT / "tests/e2e/test_kras_g12c_pilot.py",
    )
    for env_var in module.KRAS_E2E_REQUIRED_ENV:
        monkeypatch.delenv(env_var, raising=False)
    for env_group in module.KRAS_E2E_ALTERNATIVES:
        for env_var in env_group:
            monkeypatch.delenv(env_var, raising=False)
    monkeypatch.setenv("KRAS_E2E_SCOPE", "full")

    status = module.kras_e2e_preflight_status()

    assert status["ready"] is False
    assert "HFM_CHECKPOINT_PATH" in status["missing"]
    assert "GNINA_BINARY or DIFFDOCK_MODEL_PATH" in status["missing"]


def test_kras_e2e_preflight_rejects_missing_artifact_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module(
        "kras_e2e_preflight_missing_paths_test",
        ROOT / "tests/e2e/test_kras_g12c_pilot.py",
    )
    for env_var in module.KRAS_E2E_REQUIRED_ENV:
        monkeypatch.setenv(env_var, "1" if env_var.endswith("_READY") else "/missing/resource")
    monkeypatch.setenv("GNINA_BINARY", "/missing/gnina")
    monkeypatch.setenv("KRAS_E2E_SCOPE", "full")

    status = module.kras_e2e_preflight_status()

    assert status["ready"] is False
    assert any("hfm_checkpoint" in item for item in status["missing"])
    assert any("hfm_decoder" in item for item in status["missing"])
    assert any("gnina" in item for item in status["missing"])


def test_kras_e2e_reduced_scope_skips_model_fto_and_supply_requirements(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module(
        "kras_e2e_reduced_preflight_test",
        ROOT / "tests/e2e/test_kras_g12c_pilot.py",
    )
    for env_var in module.KRAS_E2E_REQUIRED_ENV:
        monkeypatch.delenv(env_var, raising=False)
    for env_group in module.KRAS_E2E_ALTERNATIVES:
        for env_var in env_group:
            monkeypatch.delenv(env_var, raising=False)
    for env_var in module.KRAS_E2E_DKI_REQUIRED_ENV:
        monkeypatch.setenv(env_var, "configured")
    monkeypatch.setenv("PROVENANCE_STORE_MODE", "production_real")
    monkeypatch.setenv("TEST_DATABASE_URL", "sqlite:///tmp/test.db")
    monkeypatch.setenv("KRAS_E2E_SCOPE", "engineering")
    monkeypatch.setenv("ORCHESTRATOR_E2E_READY", "1")

    status = module.kras_e2e_preflight_status()

    assert status["ready"] is True
    assert status["missing"] == []


def test_audit_e2e_preflight_requires_production_dki_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module(
        "audit_e2e_preflight_dki_config_test",
        ROOT / "tests/e2e/test_audit_completeness.py",
    )
    monkeypatch.setenv("PROVENANCE_SVC_URL", "http://127.0.0.1:8010")
    monkeypatch.setenv("SIGSTORE_E2E_READY", "1")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:4318")
    for env_var in module.AUDIT_E2E_DKI_REQUIRED_ENV:
        monkeypatch.delenv(env_var, raising=False)
    monkeypatch.delenv("TEST_DATABASE_URL", raising=False)
    monkeypatch.delenv("PROVENANCE_DATABASE_URL", raising=False)
    monkeypatch.delenv("PROVENANCE_STORE_MODE", raising=False)

    status = module.audit_e2e_preflight_status()

    assert status["ready"] is False
    assert "PROVENANCE_STORE_MODE=production_real" in status["missing"]
    assert "PROVENANCE_DATABASE_URL or TEST_DATABASE_URL" in status["missing"]
    assert "NEO4J_URI" in status["missing"]


@pytest.mark.asyncio
async def test_retrosyn_service_delegates_to_planner() -> None:
    module = _load_module(
        "retrosyn_service_planner_test",
        ROOT / "services/retrosyn-svc/src/retrosyn_svc/main.py",
    )

    class Planner:
        def __init__(self) -> None:
            self.calls: list[tuple[str, int]] = []

        async def find_routes(self, smiles: str, max_routes: int = 10) -> list[dict]:
            self.calls.append((smiles, max_routes))
            return [
                {
                    "route_id": "route-1",
                    "score": 0.75,
                    "predicted_yield": 0.62,
                    "steps": [
                        {
                            "step_id": "retro-1",
                            "reaction": "CCO.O=O>>CCOO",
                            "reactants": [{"smiles": "CCO"}, {"smiles": "O=O"}],
                            "conditions": {"temperature_C": 25, "time_h": 2},
                            "building_blocks": [
                                {"smiles": "CCO", "source": "enamine_real"},
                                {"smiles": "O=O", "source": "catalog"},
                            ],
                        }
                    ],
                }
            ]

    planner = Planner()
    service = module.RetrosynServicer(planner=planner)
    request = SimpleNamespace(
        project_id="proj-1",
        molecule_smiles="CCOO",
        max_routes=1,
        engine="aizynth",
    )

    response = await service.FindRoutes(request, None)

    assert planner.calls == [("CCOO", 1)]
    assert response.total_routes_found == 1
    assert response.routes[0].route_id == "route-1"
    assert list(response.routes[0].reaction_smiles) == ["CCO.O=O>>CCOO"]
    assert response.routes[0].predicted_score == pytest.approx(0.75)
    assert response.routes[0].predicted_yield == pytest.approx(0.62)
    assert list(response.routes[0].building_blocks) == ["CCO", "O=O"]
