"""Service artifact status reporting."""

from __future__ import annotations

import asyncio
import base64
import builtins
import hashlib
import importlib.util
import json
import re
import struct
import subprocess
import sys
import threading
import uuid
from pathlib import Path
from types import SimpleNamespace

import grpc
import pytest
from fastapi import HTTPException
from mf_core.proto_gen.moleculeforge.v1.core import cig_pb2, humu_pb2
from mf_core.proto_gen.moleculeforge.v1.generator import generator_pb2
from mf_core.types.molecule import Molecule

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def agent_message_hmac_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_MESSAGE_HMAC_SECRET", "service-artifact-test-secret")


@pytest.fixture(autouse=True)
def configure_iclm_checkpoint_directory(request: pytest.FixtureRequest) -> None:
    if not request.node.name.startswith("test_iclm_"):
        return
    monkeypatch = request.getfixturevalue("monkeypatch")
    tmp_path = request.getfixturevalue("tmp_path")
    monkeypatch.setenv("ICLM_CHECKPOINT_DIRECTORY", str(tmp_path))


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_openfe_transformation(path: Path, *, protocol_repeats: int = 1) -> Path:
    path.write_text(
        json.dumps(
            {
                "protocol": {
                    "settings": {
                        "protocol_repeats": protocol_repeats,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    return path


class _AgentRequestClientStub:
    def __init__(self, responder) -> None:
        self.responder = responder
        self.calls: list[dict] = []

    async def request(self, subject, payload, *, payload_type_url, timeout):
        self.calls.append(
            {
                "subject": subject,
                "payload": dict(payload),
                "payload_type_url": payload_type_url,
                "timeout": timeout,
            }
        )
        response = dict(self.responder(subject, dict(payload)))
        response.setdefault("run_id", payload["run_id"])
        response.setdefault("request_id", payload["request_id"])
        response.setdefault("schema_version", payload["schema_version"])
        for field in (
            "project_id",
            "candidate_id",
            "candidate_index",
            "canonical_smiles",
        ):
            if field in payload:
                response.setdefault(field, payload[field])
        return response


def _generator_coord_request_client() -> _AgentRequestClientStub:
    return _AgentRequestClientStub(
        lambda subject, payload: {
            "status": "dispatched",
            "selected_generators": ["hfm_3d"],
            "candidates": [{"smiles": "CCO"}],
        }
    )


def _full_policy_payload(*, oracle_level: int = 0) -> dict:
    thresholds = [
        {
            "level": 0,
            "oracle": "rdkit",
            "metric": "qed",
            "direction": "maximize",
            "value": 0.5,
        },
        {
            "level": 1,
            "oracle": "admet",
            "metric": "admet_score",
            "direction": "maximize",
            "value": 0.5,
        },
        {
            "level": 2,
            "oracle": "dock",
            "metric": "docking_score",
            "direction": "minimize",
            "value": -6.0,
        },
        {
            "level": 3,
            "oracle": "fep",
            "metric": "rbfe",
            "direction": "minimize",
            "value": -7.0,
        },
        {
            "level": 4,
            "oracle": "external",
            "metric": "experimental_activity",
            "direction": "maximize",
            "value": 0.5,
        },
    ]
    oracle_inputs = {}
    if oracle_level >= 2:
        oracle_inputs["dock"] = {
            "receptor_uri": "file:///models/receptor.pdbqt",
            "oracle_parameters": {"engine": "gnina"},
        }
    if oracle_level >= 3:
        oracle_inputs["fep"] = {
            "protein_pdb_id": "1ABC",
            "reference_ligand_smiles": "CCN",
            "oracle_parameters": {"method": "relative", "n_repeats": 3},
        }
    return {
        "validation_policy": {
            "oracle_level": oracle_level,
            "batch_size": 8,
            "max_concurrency": 2,
            "thresholds": [
                threshold for threshold in thresholds if threshold["level"] <= oracle_level
            ],
            "oracle_inputs": oracle_inputs,
        },
        "teacher_policy": {
            "teacher_source": "hypseek",
            "teacher_version": "v1",
            "allow_synthetic": False,
            "kd_weight": 0.25,
        },
        "selection_policy": {
            "criteria": [{"metric": "qed", "direction": "maximize"}],
        },
    }


def _full_candidate(
    *,
    candidate_id: str = "candidate-1",
    smiles: str = "CCO",
    generator_name: str = "hfm_3d",
) -> dict:
    return {
        "candidate_id": candidate_id,
        "canonical_smiles": smiles,
        "generator_name": generator_name,
    }


def _full_validation_record(
    *,
    candidate_id: str = "candidate-1",
    smiles: str = "CCO",
    outcome: str = "PASS",
    qed: float = 0.8,
    oracle_level: int = 0,
) -> dict:
    oracle_by_level = {
        0: "rdkit",
        1: "admet",
        2: "dock",
        3: "fep",
        4: "external",
    }
    metrics_by_level = {
        0: {
            "metric": "qed",
            "value": qed,
            "direction": "maximize",
            "threshold": 0.5,
            "passed": qed >= 0.5,
        },
        1: {
            "metric": "admet_score",
            "value": 0.8,
            "direction": "maximize",
            "threshold": 0.5,
            "passed": True,
        },
        2: {
            "metric": "docking_score",
            "value": -7.0,
            "direction": "minimize",
            "threshold": -6.0,
            "passed": True,
        },
        3: {
            "metric": "rbfe",
            "value": -8.0,
            "direction": "minimize",
            "threshold": -7.0,
            "passed": True,
        },
        4: {
            "metric": "experimental_activity",
            "value": 0.8,
            "direction": "maximize",
            "threshold": 0.5,
            "passed": True,
        },
    }
    metrics = []
    levels = []
    for level in range(oracle_level + 1):
        level_outcome = "PASS" if level < oracle_level else outcome
        metric = {
            "level": level,
            "oracle": oracle_by_level[level],
            **metrics_by_level[level],
        }
        metrics.append(metric)
        levels.append(
            {
                "level": level,
                "outcome": level_outcome,
                "oracles": [
                    {
                        "oracle": oracle_by_level[level],
                        "outcome": level_outcome,
                        "metrics": [dict(metric)],
                        "evidence_ids": [],
                    }
                ],
            }
        )
    return {
        "schema_version": "validation.record.v1",
        "candidate_id": candidate_id,
        "canonical_smiles": smiles,
        "outcome": outcome,
        "metrics": metrics,
        "evidence": [
            {
                "evidence_id": f"evidence-{candidate_id}",
                "level": oracle_level,
                "oracle": "validation_agent",
            }
        ],
        "levels": levels,
    }


def _validation_batch_response(
    payload: dict,
    records: list[dict],
    *,
    outcome: str,
    **extra: object,
) -> dict:
    return {
        "validation_schema_version": "validation.batch.v1",
        "agent": "validation_agent",
        "project_id": payload["project_id"],
        "run_id": payload["run_id"],
        "request_id": payload["request_id"],
        "validation_policy": payload["validation_policy"],
        "outcome": outcome,
        "records": records,
        **extra,
    }


def _full_selected_state(
    *,
    candidate: dict | None = None,
    routes: list[dict] | None = None,
) -> dict:
    selected = dict(candidate or _full_candidate())
    record = _full_validation_record(
        candidate_id=selected["candidate_id"],
        smiles=selected["canonical_smiles"],
    )
    return {
        "run_id": "run-1",
        "trace_id": "trace-1",
        "request": {
            "project_id": "project-1",
            "retrosyn_engine": "rsgpt",
            **_full_policy_payload(),
        },
        "candidates": [selected],
        "validation": {
            "outcome": "PASS",
            "records": [record],
            "results": [record],
        },
        "retrosyn": {"routes": list(routes or [])},
    }


def _feedback_ack(payload: dict) -> dict:
    groups = payload.get("groups")
    submitted = len(groups) if isinstance(groups, list) else 0
    return {
        "action": "generator_coord/feedback/v1",
        "status": "feedback_submitted",
        "submitted": submitted,
        "duplicates": 0,
    }


async def _configure_project_run_store(
    module,
    tmp_path: Path,
    project_id: str,
) -> None:
    module._RUN_STORE = module.RunStore(tmp_path / "runs.db")
    await module._RUN_STORE.initialize()
    await module._RUN_STORE.create_project(
        project_id,
        name=project_id,
        description="",
        created_at="2026-07-29T00:00:00+00:00",
    )


def _k8s_configmap_data(manifest: str, namespace: str, name: str) -> dict:
    import yaml

    configmaps = {
        (doc["metadata"]["namespace"], doc["metadata"]["name"]): doc.get("data", {})
        for doc in yaml.safe_load_all(manifest)
        if isinstance(doc, dict) and doc.get("kind") == "ConfigMap"
    }
    return configmaps[(namespace, name)]


def _helm_configmap_data(values: str, namespace: str, name: str) -> dict:
    import yaml

    config = yaml.safe_load(values)
    configmaps = {
        (item["namespace"], item["name"]): item.get("data", {})
        for item in config.get("configMaps", {}).values()
    }
    return configmaps[(namespace, name)]


def _k8s_secret_string_data(manifest: str, namespace: str, name: str) -> dict:
    import yaml

    secrets = {
        (doc["metadata"]["namespace"], doc["metadata"]["name"]): doc.get("stringData", {})
        for doc in yaml.safe_load_all(manifest)
        if isinstance(doc, dict) and doc.get("kind") == "Secret"
    }
    return secrets[(namespace, name)]


def _helm_secret_string_data(values: str, namespace: str, name: str) -> dict:
    import yaml

    config = yaml.safe_load(values)
    secrets = {
        (item["namespace"], item["name"]): item.get("stringData", {})
        for item in config.get("secrets", {}).values()
    }
    return secrets[(namespace, name)]


def test_cig_compiler_grpc_server_registers_compiler_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mf_core.proto_gen.moleculeforge.v1.core import cig_pb2_grpc

    module = _load_module(
        "cig_compiler_grpc_registration_test",
        ROOT / "services/cig-compiler-svc/src/cig_compiler_svc/main.py",
    )
    registrations: list[tuple[object, object]] = []

    def record_registration(servicer, server):
        registrations.append((servicer, server))

    monkeypatch.setattr(
        cig_pb2_grpc,
        "add_CIGCompilerServiceServicer_to_server",
        record_registration,
    )
    server = object()

    module.register_grpc_services(server)

    assert len(registrations) == 1
    assert isinstance(registrations[0][0], module.CIGCompilerGrpcServicer)
    assert registrations[0][1] is server


def test_cig_proto_conversion_includes_objective_edges() -> None:
    module = _load_module(
        "cig_compiler_proto_edge_test",
        ROOT / "services/cig-compiler-svc/src/cig_compiler_svc/main.py",
    )

    proto_cig = module._cig_to_proto(
        {
            "intent_id": "cig-test",
            "objective_nodes": [
                {
                    "id": "obj_affinity",
                    "name": "binding_affinity",
                    "type": "continuous_maximize",
                    "target_value": 0.0,
                    "target_min": None,
                    "target_max": None,
                    "property": "binding_affinity",
                    "weight": 0.5,
                    "pareto_tier": 1,
                },
                {
                    "id": "obj_admet_bundle",
                    "name": "admet_bundle",
                    "type": "multi_constraint_satisfy",
                    "target_value": 0.0,
                    "target_min": None,
                    "target_max": None,
                    "property": "admet_bundle",
                    "weight": 0.5,
                    "pareto_tier": 1,
                },
            ],
            "edges": [
                {
                    "source_id": "obj_affinity",
                    "target_id": "obj_admet_bundle",
                    "relation": "trade_off",
                    "strength": -0.5,
                }
            ],
            "hyperedges": [
                {
                    "source_ids": ["obj_affinity"],
                    "target_ids": ["obj_admet_bundle"],
                    "relation": "trade_off",
                    "strength": -0.5,
                }
            ],
            "generative_priors": {},
            "created_by": "cig_compiler_svc",
        }
    )

    assert len(proto_cig.edges) == 1
    assert proto_cig.edges[0].source_id == "obj_affinity"
    assert proto_cig.edges[0].target_id == "obj_admet_bundle"
    assert proto_cig.edges[0].relation == "trade_off"
    assert proto_cig.edges[0].strength == -0.5
    assert len(proto_cig.hyperedges) == 1
    assert list(proto_cig.hyperedges[0].source_ids) == ["obj_affinity"]
    assert list(proto_cig.hyperedges[0].target_ids) == ["obj_admet_bundle"]


@pytest.mark.asyncio
async def test_cig_refine_runs_configured_json_command(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_module(
        "cig_compiler_refine_command_test",
        ROOT / "services/cig-compiler-svc/src/cig_compiler_svc/main.py",
    )
    runner = tmp_path / "cig_refiner.py"
    runner.write_text(
        "import json, sys\n"
        "payload = json.load(sys.stdin)\n"
        "assert payload['feedback'] == 'add solubility constraint'\n"
        "assert payload['context']['run_id'] == 'run-1'\n"
        "assert payload['cig']['project_id'] == 'cig-original'\n"
        "print(json.dumps({"
        "'cig': {"
        "'intent_id': 'cig-refined', "
        "'objective_nodes': [{"
        "'id': 'obj_solubility', "
        "'name': 'solubility', "
        "'type': 'continuous_maximize', "
        "'target_value': 0.0, "
        "'target_min': None, "
        "'target_max': None, "
        "'property': 'solubility', "
        "'weight': 1.0, "
        "'pareto_tier': 1"
        "}], "
        "'edges': [], "
        "'hyperedges': [], "
        "'generative_priors': {}, "
        "'created_by': 'cig_refiner'"
        "}, "
        "'hciv': {'coordinates': [1.0, 0.0], 'curvature': 1.0}, "
        "'intent_cone': {'axis': [1.0, 0.0], 'half_angle': 0.5, 'curvature': 1.0}, "
        "'parse_confidence': 0.91, "
        "'ambiguities': ['confirm assay']"
        "}))\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("CIG_REFINEMENT_COMMAND", f"{sys.executable} {runner}")

    response = await module.CIGCompilerServicer().Refine(
        SimpleNamespace(
            cig={"project_id": "cig-original"},
            feedback="add solubility constraint",
            context={"run_id": "run-1"},
        ),
        None,
    )

    assert response.cig["intent_id"] == "cig-refined"
    assert response.hciv["coordinates"] == [1.0, 0.0]
    assert response.intent_cone["axis"] == [1.0, 0.0]
    assert response.parse_confidence == pytest.approx(0.91)
    assert response.ambiguities == ["confirm assay"]


def test_cig_compiler_deployment_wires_parser_refinement_and_hciv_env() -> None:
    compose = (ROOT / "infra/docker/docker-compose.dev.yml").read_text(encoding="utf-8")
    k8s = (ROOT / "infra/kubernetes/deployments/moleculeforge-services.yaml").read_text(
        encoding="utf-8"
    )
    helm_values = (ROOT / "infra/helm/moleculeforge/values.yaml").read_text(encoding="utf-8")

    for env_name in (
        "CIG_SEMANTIC_PARSER_URI",
        "CIG_SEMANTIC_PARSER_COMMAND",
        "CIG_SEMANTIC_PARSER_TIMEOUT_SECONDS",
        "CIG_REFINEMENT_COMMAND",
        "CIG_REFINEMENT_TIMEOUT_SECONDS",
        "HCIV_CHECKPOINT_PATH",
        "CHEMBL_TARGET_URL",
        "CHEMBL_TARGET_SEARCH_URL",
        "UNIPROT_SEARCH_URL",
        "RCSB_SEARCH_URL",
    ):
        assert env_name in compose
        assert env_name in k8s
        assert env_name in helm_values

    assert (
        "CIG_SEMANTIC_PARSER_TIMEOUT_SECONDS: ${CIG_SEMANTIC_PARSER_TIMEOUT_SECONDS:-30}"
    ) in compose
    assert "CIG_REFINEMENT_TIMEOUT_SECONDS: ${CIG_REFINEMENT_TIMEOUT_SECONDS:-60}" in compose
    assert "name: cig-compiler-config" in k8s
    assert "configMapKeyRef:" in k8s
    assert "envValueFrom:" in helm_values


def test_cig_compiler_runtime_rejects_missing_external_commands(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CIG_SEMANTIC_PARSER_COMMAND", "missing-cig-parser --json")
    monkeypatch.setenv("CIG_REFINEMENT_COMMAND", "missing-cig-refiner --json")
    module = _load_module(
        "cig_compiler_missing_external_commands_test",
        ROOT / "services/cig-compiler-svc/src/cig_compiler_svc/main.py",
    )

    status = module.runtime_status()

    parser_status = next(item for item in status if item["name"] == "cig_semantic_parser_command")
    refiner_status = next(item for item in status if item["name"] == "cig_refinement_command")
    assert parser_status["configured"] is True
    assert parser_status["available"] is False
    assert parser_status["source"] == "CIG_SEMANTIC_PARSER_COMMAND"
    assert "not found" in parser_status["message"]
    assert refiner_status["configured"] is True
    assert refiner_status["available"] is False
    assert refiner_status["source"] == "CIG_REFINEMENT_COMMAND"
    assert "not found" in refiner_status["message"]


def test_cig_compiler_external_command_preflight_rejects_missing_executable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module(
        "cig_compiler_external_command_preflight_test",
        ROOT / "services/cig-compiler-svc/src/cig_compiler_svc/main.py",
    )
    compiler_module = _load_module(
        "cig_compiler_domain_external_command_preflight_test",
        ROOT / "services/cig-compiler-svc/src/cig_compiler_svc/domain/compiler.py",
    )
    monkeypatch.setenv("CIG_REFINEMENT_COMMAND", "missing-cig-refiner --json")

    with pytest.raises(RuntimeError, match="not found"):
        module._run_refinement_command(
            SimpleNamespace(cig={"project_id": "cig-1"}, feedback="", context={})
        )
    with pytest.raises(RuntimeError, match="not found"):
        compiler_module.ProductionSemanticParserAdapter()._command_parser(
            "missing-cig-parser --json"
        )("optimize EGFR potency")


def test_nl2obj_grpc_server_registers_nl2obj_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mf_core.proto_gen.moleculeforge.v1.agent import orchestrator_pb2_grpc

    module = _load_module(
        "nl2obj_grpc_registration_test",
        ROOT / "services/nl2obj-svc/src/nl2obj_svc/main.py",
    )
    registrations: list[tuple[object, object]] = []

    def record_registration(servicer, server):
        registrations.append((servicer, server))

    monkeypatch.setattr(
        orchestrator_pb2_grpc,
        "add_NL2ObjServiceServicer_to_server",
        record_registration,
    )
    server = object()

    module.register_grpc_services(server)

    assert len(registrations) == 1
    assert isinstance(registrations[0][0], module.NL2ObjGrpcServicer)
    assert registrations[0][1] is server


@pytest.mark.asyncio
async def test_nl2obj_agent_uses_parser_output() -> None:
    module = _load_module(
        "nl2obj_agent_parser_output_test",
        ROOT / "agents/nl2obj/src/nl2obj/agent.py",
    )
    agent = module.NL2ObjAgent()

    result = await agent.process(
        {
            "intent": "Design KRAS G12C inhibitors with IC50 < 10 nM and good solubility",
        }
    )

    objectives = result["objectives"]
    assert result["parsed_intent"].startswith("targets: KRAS G12C")
    assert objectives["targets"][0]["label"] == "KRAS G12C"
    assert objectives["activity"]["type"] == "IC50"
    assert objectives["activity"]["target_value"] == 10.0
    assert objectives["objectives_priority"] == ["solubility", "potency"]
    assert result["confidence"] > 0.6


@pytest.mark.asyncio
async def test_nl2obj_agent_persists_parsed_intent_belief() -> None:
    module = _load_module(
        "nl2obj_agent_crg_repository_test",
        ROOT / "agents/nl2obj/src/nl2obj/agent.py",
    )

    class CRGRepository:
        def __init__(self) -> None:
            self.beliefs: list[dict] = []

        async def write_workflow_belief(self, **kwargs) -> None:
            self.beliefs.append(kwargs)

    repository = CRGRepository()
    agent = module.NL2ObjAgent(crg_repository=repository)

    await agent.process(
        {
            "project_id": "project-1",
            "run_id": "run-1",
            "intent": "Design KRAS G12C inhibitors with IC50 < 10 nM",
        }
    )

    assert len(repository.beliefs) == 1
    belief = repository.beliefs[0]
    assert belief["project_id"] == "project-1"
    assert belief["run_id"] == "run-1"
    assert belief["subject"] == "Design KRAS G12C inhibitors with IC50 < 10 nM"
    assert belief["predicate"] == "parsed_intent"
    assert belief["object_value"].startswith("targets: KRAS G12C")
    assert belief["source_agent"] == "nl2obj"


@pytest.mark.asyncio
async def test_nl2obj_agent_calls_cig_compiler_client() -> None:
    module = _load_module(
        "nl2obj_agent_cig_compiler_test",
        ROOT / "agents/nl2obj/src/nl2obj/agent.py",
    )

    class CIGCompilerClient:
        def __init__(self) -> None:
            self.requests: list[dict] = []

        async def compile_intent(self, request: dict) -> dict:
            self.requests.append(request)
            return {
                "cig": {"project_id": "cig-kras", "objectives": []},
                "hciv": {"coordinates": [1.0, 0.0], "curvature": 1.0},
                "intent_cone": {"axis": [1.0, 0.0], "half_angle": 0.5},
            }

    client = CIGCompilerClient()
    agent = module.NL2ObjAgent(cig_compiler_client=client)

    result = await agent.process(
        {
            "project_id": "project-kras",
            "intent": "Design KRAS G12C inhibitors with IC50 < 10 nM",
            "seed": 42,
        }
    )

    assert client.requests == [
        {
            "project_id": "project-kras",
            "nl_query": "Design KRAS G12C inhibitors with IC50 < 10 nM",
            "seed": 42,
        }
    ]
    assert result["cig"] == {"project_id": "cig-kras", "objectives": []}
    assert result["hciv"] == {"coordinates": [1.0, 0.0], "curvature": 1.0}
    assert result["intent_cone"] == {"axis": [1.0, 0.0], "half_angle": 0.5}


@pytest.mark.asyncio
async def test_nl2obj_agent_uses_compiled_cig_from_shared_crg() -> None:
    module = _load_module(
        "nl2obj_agent_compiled_cig_readback_test",
        ROOT / "agents/nl2obj/src/nl2obj/agent.py",
    )
    intent = "Design KRAS G12C inhibitors with IC50 < 10 nM"
    compiled = {
        "cig": {"project_id": "cig-cached", "objectives": []},
        "hciv": {"coordinates": [1.0, 0.0], "curvature": 1.0},
        "intent_cone": {"axis": [1.0, 0.0], "half_angle": 0.5},
    }

    class CIGCompilerClient:
        async def compile_intent(self, request: dict) -> dict:
            raise AssertionError("CIG compiler should not run when CRG has compiled_cig")

    class CRGRepository:
        def __init__(self) -> None:
            self.reads: list[str] = []
            self.beliefs: list[dict] = []

        async def get_run_crg(self, run_id: str) -> dict:
            self.reads.append(run_id)
            return {
                "beliefs": [
                    {
                        "subject": intent,
                        "predicate": "compiled_cig",
                        "object_value": json.dumps(compiled, sort_keys=True),
                        "confidence": 0.88,
                        "evidence_ids": ["cig-cache"],
                    }
                ]
            }

        async def write_workflow_belief(self, **kwargs) -> None:
            self.beliefs.append(kwargs)

    repository = CRGRepository()
    agent = module.NL2ObjAgent(
        cig_compiler_client=CIGCompilerClient(),
        crg_repository=repository,
    )

    result = await agent.process(
        {
            "project_id": "project-kras",
            "run_id": "run-kras",
            "intent": intent,
        }
    )

    assert repository.reads == ["run-kras"]
    assert result["cig"] == compiled["cig"]
    assert result["hciv"] == compiled["hciv"]
    assert result["intent_cone"] == compiled["intent_cone"]
    assert result["cached"] is True
    assert result["cache_source"] == "shared_crg"
    assert [belief["predicate"] for belief in repository.beliefs] == ["parsed_intent"]


@pytest.mark.asyncio
async def test_nl2obj_agent_persists_compiled_cig_belief() -> None:
    module = _load_module(
        "nl2obj_agent_compiled_cig_crg_test",
        ROOT / "agents/nl2obj/src/nl2obj/agent.py",
    )

    class CRGRepository:
        def __init__(self) -> None:
            self.beliefs: list[dict] = []

        async def write_workflow_belief(self, **kwargs) -> None:
            self.beliefs.append(kwargs)

    class CIGCompilerClient:
        async def compile_intent(self, request: dict) -> dict:
            return {
                "cig": {"project_id": "cig-kras", "objectives": []},
                "hciv": {"coordinates": [1.0, 0.0], "curvature": 1.0},
                "intent_cone": {"axis": [1.0, 0.0], "half_angle": 0.5},
            }

    repository = CRGRepository()
    agent = module.NL2ObjAgent(
        cig_compiler_client=CIGCompilerClient(),
        crg_repository=repository,
    )

    await agent.process(
        {
            "project_id": "project-kras",
            "run_id": "run-kras",
            "intent": "Design KRAS G12C inhibitors with IC50 < 10 nM",
        }
    )

    predicates = [belief["predicate"] for belief in repository.beliefs]
    assert predicates == ["parsed_intent", "compiled_cig"]
    assert json.loads(repository.beliefs[1]["object_value"]) == {
        "cig": {"project_id": "cig-kras", "objectives": []},
        "hciv": {"coordinates": [1.0, 0.0], "curvature": 1.0},
        "intent_cone": {"axis": [1.0, 0.0], "half_angle": 0.5},
    }
    assert repository.beliefs[1]["run_id"] == "run-kras"


@pytest.mark.asyncio
async def test_nl2obj_agent_requires_intent_text() -> None:
    module = _load_module(
        "nl2obj_agent_requires_intent_test",
        ROOT / "agents/nl2obj/src/nl2obj/agent.py",
    )
    agent = module.NL2ObjAgent()

    with pytest.raises(ValueError, match="intent"):
        await agent.process({})


@pytest.mark.parametrize(
    ("module_name", "service_path", "servicer_class_name"),
    [
        (
            "admet_grpc_registration_test",
            ROOT / "services/admet-svc/src/admet_svc/main.py",
            "ADMETOracleServicer",
        ),
        (
            "dock_grpc_registration_test",
            ROOT / "services/dock-svc/src/dock_svc/main.py",
            "DockOracleServicer",
        ),
        (
            "boltz2_oracle_grpc_registration_test",
            ROOT / "services/boltz2-svc/src/boltz2_svc/main.py",
            "Boltz2OracleServicer",
        ),
        (
            "fep_oracle_grpc_registration_test",
            ROOT / "services/fep-svc/src/fep_svc/main.py",
            "FEPOracleServicer",
        ),
    ],
)
def test_oracle_grpc_servers_register_oracle_service(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
    service_path: Path,
    servicer_class_name: str,
) -> None:
    from mf_core.proto_gen.moleculeforge.v1.oracle import oracle_pb2_grpc

    module = _load_module(module_name, service_path)
    registrations: list[tuple[object, object]] = []

    def record_registration(servicer, server):
        registrations.append((servicer, server))

    monkeypatch.setattr(
        oracle_pb2_grpc,
        "add_OracleServiceServicer_to_server",
        record_registration,
    )
    server = object()

    module.register_grpc_services(server)

    assert len(registrations) == 1
    assert isinstance(registrations[0][0], getattr(module, servicer_class_name))
    assert registrations[0][1] is server


@pytest.mark.asyncio
async def test_dock_service_runs_configured_json_command(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_module(
        "dock_json_command_test",
        ROOT / "services/dock-svc/src/dock_svc/main.py",
    )
    receptor = tmp_path / "protein.pdb"
    receptor.write_text("HEADER TEST\nEND\n", encoding="utf-8")
    runner = tmp_path / "dock_runner.py"
    runner.write_text(
        "import json, sys\n"
        "request = json.load(sys.stdin)\n"
        "assert request['smiles'] == 'CCO'\n"
        "assert request['engine'] == 'diffdock'\n"
        "print(json.dumps({"
        "'smiles': request['smiles'], "
        "'receptor_uri': request['protein_pdb'], "
        "'engine': 'diffdock_l', "
        "'scores': {'docking_score': -8.5}, "
        "'uncertainties': {'docking_score': 0.2}, "
        "'elapsed_ms': 17"
        "}))\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("DOCK_ORACLE_COMMAND", f"{sys.executable} {runner}")

    response = await module.DockServicer().Dock(
        SimpleNamespace(smiles="CCO", engine="diffdock", protein_pdb=str(receptor)),
        None,
    )

    assert response.engine == "diffdock_l"
    assert response.scores == {"docking_score": -8.5}
    assert response.uncertainties == {"docking_score": 0.2}
    assert response.elapsed_ms == 17


@pytest.mark.asyncio
async def test_dock_oracle_uses_request_receptor_for_oracle_requests(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_module(
        "dock_oracle_default_receptor_test",
        ROOT / "services/dock-svc/src/dock_svc/main.py",
    )
    receptor = tmp_path / "protein.pdb"
    receptor.write_text(
        "HEADER    TEST RECEPTOR\n"
        "ATOM      1  N   GLY A   1       0.000   0.000   0.000  1.00  0.00           N\n"
        "END\n",
        encoding="utf-8",
    )
    runner = tmp_path / "dock_runner.py"
    runner.write_text(
        "import json, sys\n"
        "request = json.load(sys.stdin)\n"
        f"assert request['protein_pdb'] == {str(receptor)!r}\n"
        "assert request['smiles'] == 'CCO'\n"
        "print(json.dumps({"
        "'smiles': request['smiles'], "
        "'receptor_uri': request['protein_pdb'], "
        "'engine': 'gnina', "
        "'scores': {'docking_score': -6.5}, "
        "'elapsed_ms': 19"
        "}))\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("DOCK_ORACLE_COMMAND", f"{sys.executable} {runner}")

    from mf_core.proto_gen.moleculeforge.v1.oracle import oracle_pb2

    response = await module.DockOracleServicer().Evaluate(
        oracle_pb2.OracleBatchRequest(
            project_id="project-1",
            request_id="request-1",
            molecule_smiles=["CCO"],
            level=oracle_pb2.L2_DOCKING,
            requested_properties=["docking_score"],
            receptor_uri=str(receptor),
            oracle_parameters={"engine": "gnina"},
        ),
        None,
    )

    assert len(response.evaluations) == 1
    assert response.evaluations[0].scores["docking_score"] == pytest.approx(-6.5)


def test_dock_command_preflight_rejects_missing_executable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module(
        "dock_missing_command_preflight_test",
        ROOT / "services/dock-svc/src/dock_svc/main.py",
    )
    monkeypatch.setenv("DOCK_ORACLE_COMMAND", "missing-dock-oracle-runner --json")

    with pytest.raises(RuntimeError, match="not available|not found"):
        module._run_dock_command(SimpleNamespace(smiles="CCO"), "gnina")


def test_dock_runtime_rejects_configured_missing_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module(
        "dock_runtime_missing_command_test",
        ROOT / "services/dock-svc/src/dock_svc/main.py",
    )
    monkeypatch.setenv("GNINA_BINARY", sys.executable)
    monkeypatch.delenv("DIFFDOCK_MODEL_PATH", raising=False)
    monkeypatch.setenv("DOCK_ORACLE_COMMAND", "missing-dock-oracle-runner --json")

    with pytest.raises(RuntimeError, match="dock_oracle_command"):
        module._require_runtime("gnina")


@pytest.mark.asyncio
async def test_dock_service_reports_configured_missing_command(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_module(
        "dock_service_missing_command_abort_test",
        ROOT / "services/dock-svc/src/dock_svc/main.py",
    )
    diffdock_model = tmp_path / "diffdock-model.pt"
    diffdock_model.write_text("model", encoding="utf-8")
    monkeypatch.setenv("GNINA_BINARY", sys.executable)
    monkeypatch.setenv("DIFFDOCK_MODEL_PATH", str(diffdock_model))
    monkeypatch.setenv("DOCK_ORACLE_COMMAND", "missing-dock-oracle-runner --json")

    with pytest.raises(RuntimeError, match="dock_oracle_command"):
        await module.DockServicer().Dock(
            SimpleNamespace(smiles="CCO", engine="gnina"),
            None,
        )


def test_supply_oracle_grpc_server_registers_supply_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mf_core.proto_gen.moleculeforge.v1.oracle import supply_pb2_grpc

    module = _load_module(
        "supply_grpc_registration_test",
        ROOT / "services/supply-oracle-svc/src/supply_oracle_svc/main.py",
    )
    registrations: list[tuple[object, object]] = []

    def record_registration(servicer, server):
        registrations.append((servicer, server))

    monkeypatch.setattr(
        supply_pb2_grpc,
        "add_SupplyOracleServiceServicer_to_server",
        record_registration,
    )
    server = object()

    module.register_grpc_services(server)

    assert len(registrations) == 1
    assert isinstance(registrations[0][0], module.SupplyOracleGrpcServicer)
    assert registrations[0][1] is server


def test_orchestrator_grpc_server_registers_orchestrator_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mf_core.proto_gen.moleculeforge.v1.agent import orchestrator_pb2_grpc

    module = _load_module(
        "orchestrator_grpc_registration_test",
        ROOT / "services/orchestrator-svc/src/orchestrator_svc/main.py",
    )
    registrations: list[tuple[object, object]] = []

    def record_registration(servicer, server):
        registrations.append((servicer, server))

    monkeypatch.setattr(
        orchestrator_pb2_grpc,
        "add_OrchestratorServiceServicer_to_server",
        record_registration,
    )
    server = object()

    module.register_grpc_services(server)

    assert len(registrations) == 1
    assert isinstance(registrations[0][0], module.OrchestratorGrpcServicer)
    assert registrations[0][1] is server


@pytest.mark.asyncio
async def test_orchestrator_process_starts_grpc_after_readiness_and_drains_before_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module(
        "orchestrator_process_lifecycle_order_test",
        ROOT / "services/orchestrator-svc/src/orchestrator_svc/main.py",
    )
    events: list[str] = []

    async def startup():
        events.append("run-store-agent-ready")

    async def shutdown():
        assert events[-2:] == ["grpc-stop", "grpc-wait"]
        events.append("agent-bus-close")

    class GrpcServer:
        def add_insecure_port(self, address):
            assert events == ["run-store-agent-ready", "grpc-register"]
            events.append("grpc-bind")

        async def start(self):
            assert events[-1] == "grpc-bind"
            events.append("grpc-start")

        async def stop(self, grace):
            assert events[-1] == "rest-stop"
            assert grace is not None and grace > 0
            events.append("grpc-stop")

        async def wait_for_termination(self):
            assert events[-1] == "grpc-stop"
            events.append("grpc-wait")

    class RestServer:
        async def serve(self):
            assert events[-1] == "grpc-start"
            events.extend(["rest-start", "rest-stop"])

    grpc_server = GrpcServer()
    monkeypatch.setattr(module, "_orchestrator_startup", startup)
    monkeypatch.setattr(module, "_orchestrator_shutdown", shutdown)
    monkeypatch.setattr(module.grpc.aio, "server", lambda executor: grpc_server)
    monkeypatch.setattr(
        module,
        "register_grpc_services",
        lambda server: events.append("grpc-register"),
    )

    await module._serve_process(RestServer())

    assert events == [
        "run-store-agent-ready",
        "grpc-register",
        "grpc-bind",
        "grpc-start",
        "rest-start",
        "rest-stop",
        "grpc-stop",
        "grpc-wait",
        "agent-bus-close",
    ]


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


def test_feature_store_deployment_wires_feast_repo_env() -> None:
    compose = (ROOT / "infra/docker/docker-compose.dev.yml").read_text(encoding="utf-8")
    k8s = (ROOT / "infra/kubernetes/deployments/moleculeforge-services.yaml").read_text(
        encoding="utf-8"
    )
    helm_values = (ROOT / "infra/helm/moleculeforge/values.yaml").read_text(encoding="utf-8")

    assert "FEAST_REPO_PATH" in compose
    assert "FEAST_REPO_PATH" in k8s
    assert "FEAST_REPO_PATH" in helm_values
    assert "name: feature-store-config" in k8s
    assert "configMapKeyRef:" in k8s
    assert "envValueFrom:" in helm_values


def test_admet_runtime_status_requires_http_runner_configuration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_module(
        "admet_status_test",
        ROOT / "services/admet-svc/src/admet_svc/main.py",
    )
    model_dir = tmp_path / "unused-model"
    model_dir.mkdir()
    monkeypatch.setenv("ADMET_MODEL_PATH", str(model_dir))
    monkeypatch.delenv("ADMET_SERVICE_URL", raising=False)
    monkeypatch.delenv("ADMET_TARGETS", raising=False)
    monkeypatch.delenv("ADMET_ORACLE_COMMAND", raising=False)

    statuses = module.runtime_status()

    status_by_name = {item["name"]: item for item in statuses}
    assert "admet_model" not in status_by_name
    assert status_by_name["admet_service_url"]["available"] is False
    assert status_by_name["admet_targets"]["available"] is False
    with pytest.raises(RuntimeError, match="ADMET_SERVICE_URL|admet_service_url"):
        module._require_runtime()


def test_admet_runtime_accepts_complete_http_runner_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module(
        "admet_http_status_test",
        ROOT / "services/admet-svc/src/admet_svc/main.py",
    )
    monkeypatch.delenv("ADMET_ORACLE_COMMAND", raising=False)
    monkeypatch.setenv("ADMET_SERVICE_URL", "http://admet.local")
    monkeypatch.setenv("ADMET_TARGETS", "clearance,herg")

    statuses = module._require_runtime()
    status_by_name = {item.name: item for item in statuses}

    assert status_by_name["admet_service_url"].available is True
    assert status_by_name["admet_targets"].available is True


def test_admet_runtime_rejects_invalid_http_batch_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module(
        "admet_invalid_batch_status_test",
        ROOT / "services/admet-svc/src/admet_svc/main.py",
    )
    monkeypatch.delenv("ADMET_ORACLE_COMMAND", raising=False)
    monkeypatch.setenv("ADMET_SERVICE_URL", "http://admet.local")
    monkeypatch.setenv("ADMET_TARGETS", "clearance")
    monkeypatch.setenv("ADMET_BATCH_SIZE", "invalid")

    with pytest.raises(RuntimeError, match="ADMET_BATCH_SIZE|admet_batch_size"):
        module._require_runtime()


@pytest.mark.parametrize(
    ("service_url", "timeout"),
    [
        ("file:///tmp/admet.sock", "120"),
        ("admet.local", "120"),
        ("https://admet.local", "nan"),
        ("https://admet.local", "0"),
    ],
)
def test_admet_runtime_rejects_invalid_http_url_or_timeout(
    monkeypatch: pytest.MonkeyPatch,
    service_url: str,
    timeout: str,
) -> None:
    module = _load_module(
        f"admet_invalid_http_runtime_{abs(hash((service_url, timeout)))}",
        ROOT / "services/admet-svc/src/admet_svc/main.py",
    )
    monkeypatch.delenv("ADMET_ORACLE_COMMAND", raising=False)
    monkeypatch.setenv("ADMET_SERVICE_URL", service_url)
    monkeypatch.setenv("ADMET_TARGETS", "clearance")
    monkeypatch.setenv("ADMET_ORACLE_TIMEOUT_SECONDS", timeout)

    with pytest.raises(RuntimeError, match="admet_service_url|admet_timeout"):
        module._require_runtime()


def test_dock_runtime_rejects_unwired_native_tools(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_module(
        "dock_unwired_native_runtime_test",
        ROOT / "services/dock-svc/src/dock_svc/main.py",
    )
    model = tmp_path / "diffdock.pt"
    model.write_bytes(b"model")
    monkeypatch.setenv("GNINA_BINARY", sys.executable)
    monkeypatch.setenv("DIFFDOCK_MODEL_PATH", str(model))
    monkeypatch.delenv("DOCK_ORACLE_COMMAND", raising=False)

    with pytest.raises(RuntimeError, match="DOCK_ORACLE_COMMAND"):
        module._require_runtime("gnina")

    status = {item["name"]: item for item in module.runtime_status()}
    assert status["dock_oracle_command"]["required"] is True
    assert status["dock_oracle_command"]["available"] is False


def test_fep_runtime_rejects_unwired_openfe_tool(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_module(
        "fep_unwired_native_runtime_test",
        ROOT / "services/fep-svc/src/fep_svc/main.py",
    )
    runner = tmp_path / "openfe"
    runner.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    runner.chmod(0o755)
    monkeypatch.setenv("OPENFE_RUNNER_PATH", str(runner))
    monkeypatch.delenv("FEP_ORACLE_COMMAND", raising=False)

    with pytest.raises(RuntimeError, match="FEP_ORACLE_COMMAND|fep_oracle_command"):
        module._require_runtime()

    status = module.runtime_status()
    assert status[0]["name"] == "fep_oracle_command"
    assert status[0]["required"] is True
    assert status[0]["available"] is False


def test_admet_runtime_rejects_configured_missing_command(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_module(
        "admet_missing_command_runtime_test",
        ROOT / "services/admet-svc/src/admet_svc/main.py",
    )
    model_dir = tmp_path / "admet-models"
    model_dir.mkdir()
    monkeypatch.setenv("ADMET_MODEL_PATH", str(model_dir))
    monkeypatch.setenv("ADMET_ORACLE_COMMAND", "missing-admet-runner --json")

    with pytest.raises(RuntimeError, match="admet_oracle_command"):
        module._require_runtime()


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


@pytest.mark.asyncio
async def test_admet_service_runs_configured_json_command(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_module(
        "admet_json_command_test",
        ROOT / "services/admet-svc/src/admet_svc/main.py",
    )
    runner = tmp_path / "admet_runner.py"
    runner.write_text(
        "import json, sys\n"
        "payload = json.load(sys.stdin)\n"
        "assert payload['smiles'] == ['CCO']\n"
        "assert payload['properties'] == ['clearance', 'herg']\n"
        "assert payload['return_uncertainty'] is False\n"
        "print(json.dumps({"
        "'results': [{"
        "'smiles': 'CCO', "
        "'predictions': {'clearance': 1.5, 'herg': 0.2}, "
        "'uncertainties': {'clearance': 0.1, 'herg': 0.03}, "
        "'elapsed_ms': 19"
        "}]"
        "}))\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("ADMET_MODEL_PATH", raising=False)
    monkeypatch.setenv("ADMET_ORACLE_COMMAND", f"{sys.executable} {runner}")

    service = module.ADMETServicer()
    response = await service.Predict(
        SimpleNamespace(smiles="CCO", properties=["clearance", "herg"]),
        None,
    )
    cached_response = await service.Predict(
        SimpleNamespace(smiles="CCO", properties=["clearance", "herg"]),
        None,
    )

    assert response.predictions == {"clearance": 1.5, "herg": 0.2}
    assert response.elapsed_ms == 19
    assert cached_response.predictions == {"clearance": 1.5, "herg": 0.2}


@pytest.mark.asyncio
async def test_fragfm_validation_artifacts_require_explicit_opt_in_without_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import torch
    from mf_generators.fragfm.generator import (
        bootstrap_validation_artifacts,
        load_validation_artifact_metadata,
    )
    from rdkit import Chem

    paths = await bootstrap_validation_artifacts(tmp_path / "fragfm-validation")
    copied_directory = tmp_path / "copied-fragfm-validation"
    copied_directory.mkdir()
    copied_vocabulary = copied_directory / paths["vocabulary"].name
    copied_rate_matrix = copied_directory / paths["rate_matrix"].name
    copied_vocabulary.write_bytes(paths["vocabulary"].read_bytes())
    copied_rate_matrix.write_bytes(paths["rate_matrix"].read_bytes())
    assert not (copied_directory / "moleculeforge_validation_artifact.json").exists()
    metadata = load_validation_artifact_metadata(copied_vocabulary)
    assert metadata is not None
    assert metadata["schema_version"] == "moleculeforge.validation_artifact.v1"

    monkeypatch.setenv("FRAGFM_VOCAB_PATH", str(copied_vocabulary))
    monkeypatch.setenv("FRAGFM_RATE_MATRIX_PATH", str(copied_rate_matrix))
    monkeypatch.delenv("FRAGFM_CHECKPOINT_PATH", raising=False)
    monkeypatch.delenv("FRAGFM_ALLOW_VALIDATION_ARTIFACT", raising=False)
    module = _load_module(
        "fragfm_validation_artifact_opt_in_test",
        ROOT / "services/fragfm-generator-svc/src/fragfm_generator_svc/main.py",
    )

    with pytest.raises(RuntimeError, match="FRAGFM_ALLOW_VALIDATION_ARTIFACT=true"):
        module._require_runtime()

    monkeypatch.setenv("FRAGFM_ALLOW_VALIDATION_ARTIFACT", "true")
    statuses = module._require_runtime()

    assert all(status.available for status in statuses if status.required)
    generator = module._build_generator()
    molecules = await generator.generate(batch_size=2)
    assert generator._model is None
    assert len(molecules) == 2
    assert all(Chem.MolFromSmiles(molecule.smiles) is not None for molecule in molecules)
    assert all(
        molecule.metadata["model_checkpoint_applied"] == "false"
        for molecule in molecules
    )

    rate_state = torch.load(copied_rate_matrix, map_location="cpu", weights_only=True)
    rate_state["moleculeforge_validation_artifact"]["seed"] = 8
    torch.save(rate_state, copied_rate_matrix)
    with pytest.raises(RuntimeError, match="validation artifact metadata is invalid"):
        module._require_runtime()


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
    monkeypatch.setenv("FRAGFM_DECODER_COMMAND", f"{sys.executable} -c pass")
    monkeypatch.setenv("FRAGFM_DECODER_TIMEOUT_SECONDS", "17")

    generator = module._build_generator()

    assert generator.checkpoint_path == str(checkpoint_path)
    assert generator.rate_matrix_path == str(rate_matrix_path)
    assert generator._model is not None
    assert generator.decoder.command == f"{sys.executable} -c pass"
    assert generator.decoder.timeout_seconds == 17.0
    assert generator.humu_latent_sampler is not None


def test_fragfm_runtime_reports_checkpoint_decoder_pair_and_command_preflight(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_module(
        "fragfm_runtime_decoder_pair_test",
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
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    checkpoint_path = tmp_path / "best_model.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    monkeypatch.setenv("FRAGFM_VOCAB_PATH", str(vocab_path))
    monkeypatch.setenv("FRAGFM_CHECKPOINT_PATH", str(checkpoint_path))
    monkeypatch.delenv("FRAGFM_DECODER_COMMAND", raising=False)

    pair_status = next(
        status
        for status in module.runtime_status()
        if status["name"] == "fragfm_checkpoint_decoder_pair"
    )
    assert pair_status["configured"] is True
    assert pair_status["available"] is False
    with pytest.raises(RuntimeError, match="checkpoint_decoder_pair"):
        module._require_runtime()

    monkeypatch.setenv(
        "FRAGFM_DECODER_COMMAND",
        "missing-fragfm-decoder --json",
    )
    command_status = next(
        status
        for status in module.runtime_status()
        if status["name"] == "fragfm_decoder_command"
    )
    assert command_status["configured"] is True
    assert command_status["available"] is False
    with pytest.raises(RuntimeError, match="fragfm_decoder_command"):
        module._require_runtime()

    monkeypatch.setenv("FRAGFM_DECODER_COMMAND", f"{sys.executable} -c pass")
    statuses = module._require_runtime()
    assert all(status.available for status in statuses if status.configured)


def test_fragfm_runtime_rejects_configured_missing_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_module(
        "fragfm_runtime_missing_checkpoint_test",
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
    monkeypatch.setenv("FRAGFM_VOCAB_PATH", str(vocab_path))
    monkeypatch.setenv("FRAGFM_CHECKPOINT_PATH", str(tmp_path / "missing_model.pt"))
    monkeypatch.delenv("FRAGFM_RATE_MATRIX_PATH", raising=False)

    with pytest.raises(RuntimeError, match="fragfm_checkpoint"):
        module._require_runtime()


def test_fragfm_runtime_rejects_configured_missing_rate_matrix(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_module(
        "fragfm_runtime_missing_rate_matrix_test",
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
    monkeypatch.setenv("FRAGFM_VOCAB_PATH", str(vocab_path))
    monkeypatch.delenv("FRAGFM_CHECKPOINT_PATH", raising=False)
    monkeypatch.setenv("FRAGFM_RATE_MATRIX_PATH", str(tmp_path / "missing_rate_matrix.pt"))

    with pytest.raises(RuntimeError, match="fragfm_rate_matrix"):
        module._require_runtime()


def test_fragfm_service_shared_humu_sampler_samples_intent_cone() -> None:
    module = _load_module(
        "fragfm_service_humu_sampler_test",
        ROOT / "services/fragfm-generator-svc/src/fragfm_generator_svc/main.py",
    )
    from mf_core.types.humu import IntentCone

    sampler = module.SharedHUMULatentSampler(curvature=1.0)
    latents = sampler.sample(
        batch_size=2,
        intent_cone=IntentCone(axis=[1.0, *([0.0] * 128)], half_angle=0.1),
    )

    assert len(latents) == 2
    assert len(latents[0]) == 129


def test_fragfm_deployment_wires_artifact_and_sampler_env() -> None:
    compose = (ROOT / "infra/docker/docker-compose.dev.yml").read_text(encoding="utf-8")
    k8s = (ROOT / "infra/kubernetes/deployments/moleculeforge-services.yaml").read_text(
        encoding="utf-8"
    )
    helm_values = (ROOT / "infra/helm/moleculeforge/values.yaml").read_text(encoding="utf-8")

    for env_name in (
        "FRAGFM_VOCAB_PATH",
        "FRAGFM_CHECKPOINT_PATH",
        "FRAGFM_RATE_MATRIX_PATH",
        "FRAGFM_DECODER_COMMAND",
        "FRAGFM_DECODER_TIMEOUT_SECONDS",
        "FRAGFM_HUMU_CURVATURE",
    ):
        assert env_name in compose
        assert env_name in k8s
        assert env_name in helm_values
    assert "FRAGFM_ALLOW_VALIDATION_ARTIFACT" in compose
    assert "FRAGFM_ALLOW_VALIDATION_ARTIFACT" not in k8s
    assert "FRAGFM_ALLOW_VALIDATION_ARTIFACT" not in helm_values

    assert "FRAGFM_HUMU_CURVATURE: ${FRAGFM_HUMU_CURVATURE:-1.0}" in compose
    assert (
        "FRAGFM_VOCAB_PATH: /var/lib/moleculeforge/validation-artifacts/fragfm/vocab.json"
        in compose
    )
    assert 'FRAGFM_CHECKPOINT_PATH: ""' in compose
    assert 'FRAGFM_DECODER_COMMAND: ""' in compose
    assert (
        "FRAGFM_DECODER_TIMEOUT_SECONDS: "
        "${FRAGFM_DECODER_TIMEOUT_SECONDS:-300}" in compose
    )
    assert (
        "FRAGFM_RATE_MATRIX_PATH: "
        "/var/lib/moleculeforge/validation-artifacts/fragfm/rate_matrix.pt" in compose
    )
    assert "name: fragfm-generator-config" in k8s
    assert "configMapKeyRef:" in k8s
    assert "envValueFrom:" in helm_values
    for config in (
        _k8s_configmap_data(k8s, "mf-generators", "fragfm-generator-config"),
        _helm_configmap_data(helm_values, "mf-generators", "fragfm-generator-config"),
    ):
        assert config["vocab-path"] == ""
        assert config["checkpoint-path"] == ""
        assert config["decoder-command"] == ""
        assert config["decoder-timeout-seconds"] == "300"
        assert config["rate-matrix-path"] == ""
        assert config["humu-curvature"] == "1.0"

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
async def test_mmpt_validation_index_requires_explicit_opt_in_and_serves_exact_batch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from mf_generators.mmpt_rag.generator import (
        bootstrap_validation_artifacts,
        load_validation_artifact_metadata,
    )

    paths = await bootstrap_validation_artifacts(tmp_path / "mmpt-validation")
    copied_directory = tmp_path / "copied-mmpt-validation"
    copied_directory.mkdir()
    copied_index = copied_directory / paths["index"].name
    copied_index.write_bytes(paths["index"].read_bytes())
    assert not (copied_directory / "moleculeforge_validation_artifact.json").exists()
    metadata = load_validation_artifact_metadata(copied_index)
    assert metadata is not None
    assert metadata["schema_version"] == "moleculeforge.validation_artifact.v1"

    monkeypatch.setenv("MMPT_INDEX_URI", copied_index.as_uri())
    monkeypatch.delenv("MMPT_PATENT_RAG_COMMAND", raising=False)
    monkeypatch.delenv("MMPT_SEQ2SEQ_DECODER_COMMAND", raising=False)
    monkeypatch.delenv("MMPT_ALLOW_VALIDATION_ARTIFACT", raising=False)
    module = _load_module(
        "mmpt_validation_artifact_opt_in_test",
        ROOT / "services/mmpt-generator-svc/src/mmpt_generator_svc/main.py",
    )

    with pytest.raises(RuntimeError, match="MMPT_ALLOW_VALIDATION_ARTIFACT=true"):
        module._require_runtime()

    monkeypatch.setenv("MMPT_ALLOW_VALIDATION_ARTIFACT", "true")
    statuses = module._require_runtime()
    response = await module.MMPTGeneratorServicer().Generate(
        _valid_generator_request(
            batch_size=256,
            generator_params={"seed": "7"},
        ),
        None,
    )

    assert all(status.available for status in statuses)
    assert len(response.molecules) == 256
    payloads = [json.loads(molecule.decode("utf-8")) for molecule in response.molecules]
    assert len({payload["canonical_smiles"] for payload in payloads}) == 256

    malformed_index = json.loads(copied_index.read_text(encoding="utf-8"))
    malformed_index["moleculeforge_validation_artifact"]["purpose"] = "production"
    copied_index.write_text(json.dumps(malformed_index), encoding="utf-8")
    with pytest.raises(RuntimeError, match="validation artifact metadata is invalid"):
        module._require_runtime()


@pytest.mark.parametrize(
    ("module_name", "service_path", "expected_generator"),
    [
        (
            "hfm_validation_bootstrap_cli_test",
            ROOT / "services/hfm-generator-svc/src/hfm_generator_svc/main.py",
            "hfm_3d",
        ),
        (
            "crem_validation_bootstrap_cli_test",
            ROOT / "services/crem-generator-svc/src/crem_generator_svc/main.py",
            "crem_3d",
        ),
        (
            "fragfm_validation_bootstrap_cli_test",
            ROOT / "services/fragfm-generator-svc/src/fragfm_generator_svc/main.py",
            "fragfm",
        ),
        (
            "mmpt_validation_bootstrap_cli_test",
            ROOT / "services/mmpt-generator-svc/src/mmpt_generator_svc/main.py",
            "mmpt_rag",
        ),
    ],
)
def test_generator_services_expose_uniform_validation_bootstrap_cli(
    module_name: str,
    service_path: Path,
    expected_generator: str,
    tmp_path: Path,
) -> None:
    module = _load_module(module_name, service_path)
    target = tmp_path / expected_generator

    module._main(["--bootstrap-validation-artifacts", str(target)])

    metadata = json.loads(
        (target / "moleculeforge_validation_artifact.json").read_text(
            encoding="utf-8"
        )
    )
    assert metadata["generator"] == expected_generator
    assert metadata["purpose"] == "synthetic_pipeline_validation_only"


def test_mmpt_deployment_wires_index_rag_and_decoder_env() -> None:
    compose = (ROOT / "infra/docker/docker-compose.dev.yml").read_text(encoding="utf-8")
    k8s = (ROOT / "infra/kubernetes/deployments/moleculeforge-services.yaml").read_text(
        encoding="utf-8"
    )
    helm_values = (ROOT / "infra/helm/moleculeforge/values.yaml").read_text(encoding="utf-8")

    for env_name in (
        "MMPT_INDEX_URI",
        "MMPT_PATENT_RAG_COMMAND",
        "MMPT_PATENT_RAG_TIMEOUT_SECONDS",
        "MMPT_SEQ2SEQ_DECODER_COMMAND",
        "MMPT_SEQ2SEQ_DECODER_TIMEOUT_SECONDS",
    ):
        assert env_name in compose
        assert env_name in k8s
        assert env_name in helm_values
    assert "MMPT_ALLOW_VALIDATION_ARTIFACT" in compose
    assert "MMPT_ALLOW_VALIDATION_ARTIFACT" not in k8s
    assert "MMPT_ALLOW_VALIDATION_ARTIFACT" not in helm_values

    assert "MMPT_PATENT_RAG_TIMEOUT_SECONDS: ${MMPT_PATENT_RAG_TIMEOUT_SECONDS:-300}" in compose
    assert (
        "MMPT_INDEX_URI: "
        "file:///var/lib/moleculeforge/validation-artifacts/mmpt/mmpt_index.json"
        in compose
    )
    assert (
        "MMPT_SEQ2SEQ_DECODER_TIMEOUT_SECONDS: ${MMPT_SEQ2SEQ_DECODER_TIMEOUT_SECONDS:-300}"
    ) in compose
    assert "name: mmpt-generator-config" in k8s
    assert "configMapKeyRef:" in k8s
    assert "envValueFrom:" in helm_values
    for config in (
        _k8s_configmap_data(k8s, "mf-generators", "mmpt-generator-config"),
        _helm_configmap_data(helm_values, "mf-generators", "mmpt-generator-config"),
    ):
        assert config["index-uri"] == ""
        assert config["patent-rag-command"] == ""
        assert config["patent-rag-timeout-seconds"] == "300"
        assert config["seq2seq-decoder-command"] == ""
        assert config["seq2seq-decoder-timeout-seconds"] == "300"


def test_mmpt_runtime_rejects_missing_external_rag_command(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_module(
        "mmpt_missing_rag_runtime_test",
        ROOT / "services/mmpt-generator-svc/src/mmpt_generator_svc/main.py",
    )
    index_path = tmp_path / "mmpt_index.json"
    index_path.write_text('{"mmp_database": []}', encoding="utf-8")
    monkeypatch.setenv("MMPT_INDEX_URI", index_path.as_uri())
    monkeypatch.setenv("MMPT_PATENT_RAG_COMMAND", "missing-mmpt-rag --json")
    monkeypatch.delenv("MMPT_SEQ2SEQ_DECODER_COMMAND", raising=False)

    status = module.runtime_status()

    rag_status = next(item for item in status if item["name"] == "mmpt_patent_rag_command")
    assert rag_status["configured"] is True
    assert rag_status["available"] is False
    assert "not found" in rag_status["message"]


def test_mmpt_runtime_rejects_unsupported_index_uri(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module(
        "mmpt_unsupported_index_uri_runtime_status_test",
        ROOT / "services/mmpt-generator-svc/src/mmpt_generator_svc/main.py",
    )
    monkeypatch.setenv("MMPT_INDEX_URI", "https://indexes.example/mmpt.json")
    monkeypatch.delenv("MMPT_PATENT_RAG_COMMAND", raising=False)
    monkeypatch.delenv("MMPT_SEQ2SEQ_DECODER_COMMAND", raising=False)

    with pytest.raises(RuntimeError, match="MMPT_INDEX_URI must use file://"):
        module._require_runtime()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("module_name", "service_path", "class_name", "generator_name"),
    [
        (
            "hfm_generator_info_test",
            ROOT / "services/hfm-generator-svc/src/hfm_generator_svc/main.py",
            "HFMGeneratorServicer",
            "hfm_3d",
        ),
        (
            "fragfm_generator_info_test",
            ROOT / "services/fragfm-generator-svc/src/fragfm_generator_svc/main.py",
            "FragFMGeneratorServicer",
            "fragfm",
        ),
        (
            "crem_generator_info_test",
            ROOT / "services/crem-generator-svc/src/crem_generator_svc/main.py",
            "CReMGeneratorServicer",
            "crem_3d",
        ),
        (
            "iclm_generator_info_test",
            ROOT / "services/iclm-svc/src/iclm_svc/main.py",
            "ICLMServicer",
            "iclm",
        ),
        (
            "mmpt_generator_info_test",
            ROOT / "services/mmpt-generator-svc/src/mmpt_generator_svc/main.py",
            "MMPTGeneratorServicer",
            "mmpt_rag",
        ),
    ],
)
async def test_generator_service_info_returns_health_metadata(
    module_name: str,
    service_path: Path,
    class_name: str,
    generator_name: str,
) -> None:
    module = _load_module(module_name, service_path)
    service = getattr(module, class_name)(generator=object())

    response = await service.Info(SimpleNamespace(), None)

    assert response.name == generator_name
    assert response.version


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


@pytest.mark.asyncio
async def test_humu_encoder_service_accepts_proto_route_input_data(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import torch
    from mf_core.proto_gen.moleculeforge.v1.humu import encoder_pb2
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
        "humu_encoder_proto_route_test",
        ROOT / "services/humu-encoder-svc/src/humu_encoder_svc/main.py",
    )
    service = module.HUMUEncoderServicer()
    request = encoder_pb2.EncodeRequest(
        entity_type="route",
        input_data=json.dumps({"reactions": ["CCO>>CC=O"], "steps": 1}).encode(),
    )

    response = await service.Encode(request, None)

    assert len(response.humu_embedding) == 129 * 4
    assert response.curvature == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_humu_encoder_service_preserves_molecule_3d_input_data(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import torch
    from mf_core.proto_gen.moleculeforge.v1.humu import encoder_pb2
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
        "humu_encoder_proto_molecule_3d_test",
        ROOT / "services/humu-encoder-svc/src/humu_encoder_svc/main.py",
    )
    service = module.HUMUEncoderServicer()

    base = encoder_pb2.EncodeRequest(
        entity_type="molecule",
        input_data=json.dumps(
            {
                "smiles": "CCO",
                "coords": [[0.0, 0.0, 0.0], [1.4, 0.0, 0.0], [2.1, 0.8, 0.0]],
            }
        ).encode(),
    )
    stretched = encoder_pb2.EncodeRequest(
        entity_type="molecule",
        input_data=json.dumps(
            {
                "smiles": "CCO",
                "coords": [[0.0, 0.0, 0.0], [2.2, 0.0, 0.0], [3.4, 1.3, 0.0]],
            }
        ).encode(),
    )

    base_response = await service.Encode(base, None)
    stretched_response = await service.Encode(stretched, None)

    assert base_response.humu_embedding != stretched_response.humu_embedding


def test_humu_encoder_deployment_wires_checkpoint_and_device_env() -> None:
    compose = (ROOT / "infra/docker/docker-compose.dev.yml").read_text(encoding="utf-8")
    k8s = (ROOT / "infra/kubernetes/deployments/moleculeforge-services.yaml").read_text(
        encoding="utf-8"
    )
    helm_values = (ROOT / "infra/helm/moleculeforge/values.yaml").read_text(encoding="utf-8")

    for env_name in (
        "HUMU_CHECKPOINT_PATH",
        "HUMU_DEVICE",
        "HUMU_ESM2_CHECKPOINT_PATH",
        "HUMU_ESM2_CHECKPOINT_SHA256",
        "HUMU_LEGACY_MODEL_CONFIG_PATH",
    ):
        assert env_name in compose
        assert env_name in k8s
        assert env_name in helm_values
    assert "HUMU_ALLOW_VALIDATION_ARTIFACT" in compose
    assert "HUMU_ALLOW_VALIDATION_ARTIFACT" not in k8s
    assert "HUMU_ALLOW_VALIDATION_ARTIFACT" not in helm_values

    assert "HUMU_DEVICE: ${HUMU_DEVICE:-cpu}" in compose
    assert (
        "HUMU_CHECKPOINT_PATH: /var/lib/moleculeforge/validation-artifacts/humu/humu.pt"
        in compose
    )
    assert "name: humu-encoder-config" in k8s
    assert "configMapKeyRef:" in k8s
    assert "envValueFrom:" in helm_values
    for config in (
        _k8s_configmap_data(k8s, "mf-generators", "humu-encoder-config"),
        _helm_configmap_data(helm_values, "mf-generators", "humu-encoder-config"),
    ):
        assert config["checkpoint-path"] == ""
        assert config["device"] == "cpu"
        assert config["esm2-checkpoint-path"] == ""
        assert config["esm2-checkpoint-sha256"] == ""
        assert config["legacy-model-config-path"] == ""


def test_deployment_config_references_have_declared_sources() -> None:
    import yaml

    k8s_docs = list(
        yaml.safe_load_all(
            (ROOT / "infra/kubernetes/deployments/moleculeforge-services.yaml").read_text(
                encoding="utf-8"
            )
        )
    )
    k8s_configmaps = {
        (doc["metadata"]["namespace"], doc["metadata"]["name"])
        for doc in k8s_docs
        if isinstance(doc, dict) and doc.get("kind") == "ConfigMap"
    }
    k8s_secrets = {
        (doc["metadata"]["namespace"], doc["metadata"]["name"])
        for doc in k8s_docs
        if isinstance(doc, dict) and doc.get("kind") == "Secret"
    }
    missing_k8s_refs: list[tuple[str, str, str, str, str]] = []
    for doc in k8s_docs:
        if not isinstance(doc, dict) or doc.get("kind") != "Deployment":
            continue
        namespace = doc["metadata"]["namespace"]
        deployment_name = doc["metadata"]["name"]
        for container in doc["spec"]["template"]["spec"]["containers"]:
            for env in container.get("env", []) or []:
                value_from = env.get("valueFrom") or {}
                if "configMapKeyRef" in value_from:
                    ref_name = value_from["configMapKeyRef"]["name"]
                    if (namespace, ref_name) not in k8s_configmaps:
                        missing_k8s_refs.append(
                            (deployment_name, namespace, env["name"], "ConfigMap", ref_name)
                        )
                if "secretKeyRef" in value_from:
                    ref_name = value_from["secretKeyRef"]["name"]
                    if (namespace, ref_name) not in k8s_secrets:
                        missing_k8s_refs.append(
                            (deployment_name, namespace, env["name"], "Secret", ref_name)
                        )
    assert missing_k8s_refs == []

    helm_values = yaml.safe_load(
        (ROOT / "infra/helm/moleculeforge/values.yaml").read_text(encoding="utf-8")
    )
    helm_configmaps = {
        (item["namespace"], item["name"]) for item in helm_values.get("configMaps", {}).values()
    }
    helm_secrets = {
        (item["namespace"], item["name"]) for item in helm_values.get("secrets", {}).values()
    }
    missing_helm_refs: list[tuple[str, str, str, str, str]] = []
    for service_name, service_config in helm_values["services"].items():
        namespace = service_config.get("namespace", helm_values["global"]["namespace"])
        for env_name, value_from in (service_config.get("envValueFrom") or {}).items():
            if "configMapKeyRef" in value_from:
                ref_name = value_from["configMapKeyRef"]["name"]
                if (namespace, ref_name) not in helm_configmaps:
                    missing_helm_refs.append(
                        (service_name, namespace, env_name, "ConfigMap", ref_name)
                    )
            if "secretKeyRef" in value_from:
                ref_name = value_from["secretKeyRef"]["name"]
                if (namespace, ref_name) not in helm_secrets:
                    missing_helm_refs.append(
                        (service_name, namespace, env_name, "Secret", ref_name)
                    )
    assert missing_helm_refs == []


def test_deployment_service_dns_targets_resolve_to_declared_services() -> None:
    import yaml

    target_pattern = re.compile(r"([a-z0-9-]+)\.([a-z0-9-]+)\.svc\.cluster\.local")
    k8s_docs = list(
        yaml.safe_load_all(
            (ROOT / "infra/kubernetes/deployments/moleculeforge-services.yaml").read_text(
                encoding="utf-8"
            )
        )
    )
    k8s_services = {
        (doc["metadata"]["namespace"], doc["metadata"]["name"])
        for doc in k8s_docs
        if isinstance(doc, dict) and doc.get("kind") == "Service"
    }
    missing_k8s_targets: list[tuple[str, str, str, str]] = []
    for doc in k8s_docs:
        if not isinstance(doc, dict) or doc.get("kind") != "Deployment":
            continue
        deployment_name = doc["metadata"]["name"]
        for container in doc["spec"]["template"]["spec"]["containers"]:
            for env in container.get("env", []) or []:
                for service_name, namespace in target_pattern.findall(str(env.get("value", ""))):
                    if (namespace, service_name) not in k8s_services:
                        missing_k8s_targets.append(
                            (deployment_name, env["name"], namespace, service_name)
                        )
    assert missing_k8s_targets == []

    helm_values = yaml.safe_load(
        (ROOT / "infra/helm/moleculeforge/values.yaml").read_text(encoding="utf-8")
    )
    helm_services = {
        (service_config.get("namespace", helm_values["global"]["namespace"]), service_name)
        for service_name, service_config in helm_values["services"].items()
    }
    missing_helm_targets: list[tuple[str, str, str, str]] = []
    for service_name, service_config in helm_values["services"].items():
        for env_name, value in (service_config.get("env") or {}).items():
            for target_service, namespace in target_pattern.findall(str(value)):
                if (namespace, target_service) not in helm_services:
                    missing_helm_targets.append((service_name, env_name, namespace, target_service))
    assert missing_helm_targets == []


class _RecordingQdrantClient:
    def __init__(self) -> None:
        self.upserts: list[dict] = []
        self.searches: list[dict] = []
        self.deletes: list[list[str]] = []

    async def upsert(self, data: dict[str, list]) -> int:
        self.upserts.append(data)
        return len(data["id"])

    async def search(self, vector: list[float], top_k: int = 10, output_fields=None) -> list[dict]:
        self.searches.append({"vector": vector, "top_k": top_k, "output_fields": output_fields})
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


class _RecordingAbortContext:
    def __init__(self) -> None:
        self.code = None
        self.message = ""

    async def abort(self, code, message: str) -> None:
        self.code = code
        self.message = message


def _valid_generator_request(
    *,
    batch_size: int,
    generator_params: dict[str, str] | None = None,
    half_angle: float = 0.2,
) -> generator_pb2.GenerateRequest:
    cone = humu_pb2.IntentCone(
        axis=[1.0, *([0.0] * 128)],
        half_angle=half_angle,
        curvature=1.0,
        property_weights={"qed": 1.0},
    )
    return generator_pb2.GenerateRequest(
        project_id="project-1",
        request_id="request-1",
        batch_size=batch_size,
        total_molecules=batch_size,
        intent_cone=cone.SerializeToString(),
        cig=cig_pb2.CIG(
            project_id="project-1",
            objectives=[
                cig_pb2.ObjectiveNode(
                    id="qed",
                    name="QED",
                    type=cig_pb2.MAXIMIZE,
                    property="qed",
                    weight=1.0,
                )
            ],
            created_by="test",
        ),
        hciv=humu_pb2.HCIV(
            coordinates=[1.0, *([0.0] * 128)],
            curvature=1.0,
        ),
        context_schema_version="generator_context.v1",
        generator_params=generator_params or {},
    )


def _valid_model_update_request(
    *,
    samples: list[dict[str, object]],
    teacher_embeddings: list[list[float]],
    kd_weight: float,
    target_version: str = "iclm-v2",
) -> generator_pb2.ModelUpdateRequest:
    rows = len(samples)
    dim = len(teacher_embeddings[0])
    flat_embeddings = [value for embedding in teacher_embeddings for value in embedding]
    normalized_samples = [
        {
            "candidate_id": f"candidate-{index + 1}",
            "reward": 1.0,
            **sample,
        }
        for index, sample in enumerate(samples)
    ]
    return generator_pb2.ModelUpdateRequest(
        run_id="run-iclm",
        request_id="update-iclm",
        training_batch_json=json.dumps(
            {
                "schema_version": "training-batch.v1",
                "samples": normalized_samples,
                "kd_weight": kd_weight,
            },
            sort_keys=True,
        ),
        teacher_embeddings=struct.pack(f"<{rows * dim}f", *flat_embeddings),
        rows=rows,
        dim=dim,
        teacher_source="hypseek",
        teacher_version="teacher-v1",
        target_checkpoint_version=target_version,
    )


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
    request = _valid_generator_request(
        batch_size=2,
        generator_params={"temperature": "0.1"},
    )

    response = await service.Generate(request, None)

    assert generator.calls[0]["batch_size"] == 2
    assert generator.calls[0]["intent_cone"].half_angle == pytest.approx(0.2)
    assert generator.calls[0]["kwargs"] == {"temperature": "0.1"}
    assert response.generator_name == generator_name
    assert response.generation_id == "project-1"
    assert len(response.molecules) == 2
    assert json.loads(response.molecules[0].decode("utf-8"))["smiles"] == "CCO"


def test_crem_service_builds_configured_external_scorers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_module(
        "crem_service_external_scorers_test",
        ROOT / "services/crem-generator-svc/src/crem_generator_svc/main.py",
    )
    mmp_db_path = tmp_path / "crem_mmp.json"
    mmp_db_path.write_text(
        json.dumps(
            {
                "mutations": [
                    {
                        "id": "ethanol",
                        "seed_smiles": "CC",
                        "product": "CCO",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("CREM_MMP_DB_PATH", str(mmp_db_path))
    monkeypatch.setenv("CREM_DOCK_ORACLE_TARGET", "dock-svc:50054")
    monkeypatch.setenv("CREM_PHARMACOPHORE_SCORER_COMMAND", "python pharmacophore.py")
    monkeypatch.setenv("CREM_HUMU_SCORER_COMMAND", "python humu.py")

    generator = module._build_generator()

    assert generator.mmp_db_path == str(mmp_db_path)
    assert generator.docking_scorer.target == "dock-svc:50054"
    assert generator.pharmacophore_scorer.command == "python pharmacophore.py"
    assert generator.pharmacophore_scorer.source == "pharmacophore"
    assert generator.humu_embedding_scorer.command == "python humu.py"
    assert generator.humu_embedding_scorer.source == "humu_embedding"


@pytest.mark.asyncio
async def test_crem_external_json_score_provider_returns_smiles_records(
    tmp_path: Path,
) -> None:
    module = _load_module(
        "crem_service_json_score_provider_test",
        ROOT / "services/crem-generator-svc/src/crem_generator_svc/main.py",
    )
    runner = tmp_path / "crem_scorer.py"
    runner.write_text(
        "import json, sys\n"
        "payload = json.load(sys.stdin)\n"
        "assert payload['smiles'] == ['CCO', 'CCN']\n"
        "assert payload['source'] == 'pharmacophore'\n"
        "assert payload['intent_cone']['half_angle'] == 0.2\n"
        "print(json.dumps({'records': {"
        "'CCO': {'pharmacophore_score': 0.8}, "
        "'CCN': {'pharmacophore_score': 0.4}"
        "}}))\n",
        encoding="utf-8",
    )
    from mf_core.types.humu import IntentCone

    provider = module.ExternalJSONScoreProvider(
        command=f"{sys.executable} {runner}",
        source="pharmacophore",
    )

    records = await provider.score_batch(
        ["CCO", "CCN"],
        intent_cone=IntentCone(axis=[1.0] + [0.0] * 128, half_angle=0.2),
    )

    assert records == {
        "CCO": {"pharmacophore_score": 0.8},
        "CCN": {"pharmacophore_score": 0.4},
    }


@pytest.mark.asyncio
async def test_crem_external_json_score_provider_preflight_rejects_missing_executable() -> None:
    module = _load_module(
        "crem_service_json_score_provider_missing_command_test",
        ROOT / "services/crem-generator-svc/src/crem_generator_svc/main.py",
    )
    provider = module.ExternalJSONScoreProvider(
        command="missing-crem-scorer-provider --json",
        source="pharmacophore",
    )

    with pytest.raises(RuntimeError, match="not found"):
        await provider.score_batch(["CCO"])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("module_name", "service_path", "class_name", "env_vars"),
    [
        (
            "fragfm_generator_intent_cone_test",
            ROOT / "services/fragfm-generator-svc/src/fragfm_generator_svc/main.py",
            "FragFMGeneratorServicer",
            ("FRAGFM_VOCAB_PATH",),
        ),
        (
            "crem_generator_intent_cone_test",
            ROOT / "services/crem-generator-svc/src/crem_generator_svc/main.py",
            "CReMGeneratorServicer",
            ("CREM_MMP_DB_PATH",),
        ),
        (
            "iclm_generator_intent_cone_test",
            ROOT / "services/iclm-svc/src/iclm_svc/main.py",
            "ICLMServicer",
            ("ICLM_MODEL_PATH",),
        ),
    ],
)
async def test_generator_services_pass_request_intent_cone_to_model_object(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    module_name: str,
    service_path: Path,
    class_name: str,
    env_vars: tuple[str, ...],
) -> None:
    for env_var in env_vars:
        artifact_path = tmp_path / f"{env_var.lower()}.dat"
        artifact_path.write_text("artifact", encoding="utf-8")
        monkeypatch.setenv(env_var, str(artifact_path))
    module = _load_module(module_name, service_path)
    generator = _RecordingGenerator()
    service = getattr(module, class_name)(generator=generator)

    await service.Generate(
        _valid_generator_request(
            batch_size=1,
            generator_params={"sampling_seed": "7"},
        ),
        None,
    )

    cone = generator.calls[0]["intent_cone"]
    assert cone is not None
    assert cone.axis[0] == 1.0
    assert cone.half_angle == pytest.approx(0.2)


def test_fragfm_service_rejects_invalid_intent_cone_as_invalid_argument(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    artifact_path = tmp_path / "fragfm_vocab.json"
    artifact_path.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("FRAGFM_VOCAB_PATH", str(artifact_path))
    module = _load_module(
        "fragfm_generator_invalid_intent_cone_test",
        ROOT / "services/fragfm-generator-svc/src/fragfm_generator_svc/main.py",
    )
    generator = _RecordingGenerator()
    service = module.FragFMGeneratorServicer(generator=generator)
    context = _RecordingAbortContext()
    request = _valid_generator_request(batch_size=1)
    request.intent_cone = b"not-a-serialized-intent-cone"

    with pytest.raises(ValueError, match="intent_cone"):
        asyncio.run(
            service.Generate(
                request,
                context,
            ),
        )

    assert context.code == module.grpc.StatusCode.INVALID_ARGUMENT
    assert "intent_cone" in context.message
    assert generator.calls == []


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
    request = _valid_generator_request(
        batch_size=2,
        generator_params={"seed": "7"},
    )

    response = await service.Generate(request, None)

    assert len(generator.calls) == 1
    call = generator.calls[0]
    assert call["hciv"] is not None
    assert call["cone"] is not None
    assert call["cig"] is not None
    assert call["n_samples"] == 2
    assert call["seed"] == 7
    assert response.generator_name == "mmpt_rag"
    assert response.generation_id == "project-1"
    assert len(response.molecules) == 2
    assert json.loads(response.molecules[0].decode("utf-8"))["smiles"] == "CCO"


@pytest.mark.asyncio
async def test_mmpt_generator_service_passes_request_intent_cone_to_model_object(
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
        "mmpt_generator_intent_cone_test",
        ROOT / "services/mmpt-generator-svc/src/mmpt_generator_svc/main.py",
    )
    generator = _RecordingMMPTRAGGenerator()
    service = module.MMPTGeneratorServicer(generator=generator)

    await service.Generate(
        _valid_generator_request(
            batch_size=1,
            generator_params={"seed": "7"},
        ),
        None,
    )

    cone = generator.calls[0]["cone"]
    assert cone is not None
    assert cone.axis[0] == 1.0
    assert cone.half_angle == pytest.approx(0.2)


@pytest.mark.asyncio
async def test_supply_service_uses_file_catalog_with_source_timestamp(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    catalog_path = tmp_path / "supply_catalog.json"
    catalog_path.write_text(
        json.dumps(
            {
                "catalog_version": "catalog-2026-05",
                "records": [
                    {
                        "smiles": "CCO",
                        "catalog_id": "CAT-1",
                        "source": "local_catalog",
                        "source_timestamp": "2026-05-01T00:00:00Z",
                        "available": True,
                        "price": 12.5,
                        "currency": "USD",
                        "lead_time_days": 3,
                    },
                    {
                        "smiles": "CCN",
                        "available": False,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("SUPPLY_CATALOG_URI", catalog_path.as_uri())
    module = _load_module(
        "supply_catalog_file_test",
        ROOT / "services/supply-oracle-svc/src/supply_oracle_svc/main.py",
    )
    service = module.SupplyOracleServicer()

    request = SimpleNamespace(
        smiles="CCO",
        request_id="supply-1",
        project_id="project-1",
        candidate_id="candidate-1",
        candidate_index=2,
        canonical_smiles="CCO",
    )
    response = await service.CheckAvailability(request, None)
    missing = await service.CheckAvailability(
        SimpleNamespace(**{**vars(request), "smiles": "CCN", "request_id": "supply-2"}),
        None,
    )
    expected_checksum = f"sha256:{hashlib.sha256(catalog_path.read_bytes()).hexdigest()}"

    assert response.available is True
    assert response.catalog_id == "CAT-1"
    assert response.catalog_source == "local_catalog"
    assert response.source_timestamp == "2026-05-01T00:00:00Z"
    assert response.price == 12.5
    assert response.lead_time_days == 3
    assert response.request_id == "supply-1"
    assert response.project_id == "project-1"
    assert response.candidate_id == "candidate-1"
    assert response.candidate_index == 2
    assert response.canonical_smiles == "CCO"
    assert response.evidence_id.startswith("sha256:")
    assert response.catalog_version == "catalog-2026-05"
    assert response.catalog_checksum == expected_checksum
    assert missing.available is False
    assert missing.request_id == "supply-2"
    assert missing.evidence_id.startswith("sha256:")
    assert missing.evidence_id != response.evidence_id
    assert missing.catalog_version == "catalog-2026-05"
    assert missing.catalog_checksum == expected_checksum


@pytest.mark.asyncio
async def test_supply_service_uses_aizynth_hdf5_stock(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import pandas as pd
    from rdkit import Chem

    mol = Chem.MolFromSmiles("CCO")
    assert mol is not None
    inchi_key = Chem.MolToInchiKey(mol)
    stock_path = tmp_path / "zinc_stock.hdf5"
    pd.DataFrame({"inchi_key": [inchi_key]}).to_hdf(stock_path, key="table")
    with pd.HDFStore(stock_path, mode="a") as store:
        store.get_storer("table").attrs.catalog_version = "zinc-2026-05"
    monkeypatch.setenv("SUPPLY_CATALOG_URI", stock_path.as_uri())
    module = _load_module(
        "supply_catalog_hdf5_test",
        ROOT / "services/supply-oracle-svc/src/supply_oracle_svc/main.py",
    )
    service = module.SupplyOracleServicer()

    request = SimpleNamespace(
        smiles="CCO",
        request_id="supply-hdf5-1",
        project_id="project-1",
        candidate_id="candidate-1",
        candidate_index=0,
        canonical_smiles="CCO",
    )
    response = await service.CheckAvailability(request, None)
    missing = await service.CheckAvailability(
        SimpleNamespace(**{**vars(request), "smiles": "CCN", "request_id": "supply-hdf5-2"}),
        None,
    )
    status = module.runtime_status()[0]

    assert response.available is True
    assert response.catalog_id == inchi_key
    assert response.catalog_source == "aizynth_stock"
    assert response.source_timestamp
    assert response.price is None
    assert response.request_id == "supply-hdf5-1"
    assert response.evidence_id.startswith("sha256:")
    assert response.catalog_version == "zinc-2026-05"
    assert response.catalog_checksum == (
        f"sha256:{hashlib.sha256(stock_path.read_bytes()).hexdigest()}"
    )
    assert missing.available is False
    assert missing.request_id == "supply-hdf5-2"
    assert missing.evidence_id
    assert missing.catalog_version == "zinc-2026-05"
    assert missing.catalog_checksum == response.catalog_checksum
    assert status["available"] is True


def test_supply_runtime_opens_hdf5_and_rejects_invalid_schema(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import pandas as pd

    stock_path = tmp_path / "invalid_stock.hdf5"
    pd.DataFrame({"smiles": ["CCO"]}).to_hdf(stock_path, key="table")
    monkeypatch.setenv("SUPPLY_CATALOG_URI", stock_path.as_uri())
    module = _load_module(
        "supply_catalog_invalid_hdf5_schema_test",
        ROOT / "services/supply-oracle-svc/src/supply_oracle_svc/main.py",
    )

    status = module.runtime_status()[0]

    assert status["configured"] is True
    assert status["available"] is False
    assert "inchi_key" in status["message"]


def test_supply_hdf5_readiness_validates_schema_without_loading_catalog(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import pandas as pd

    stock_path = tmp_path / "stock.hdf5"
    pd.DataFrame({"inchi_key": ["LFQSCWFLJHTTHZ-UHFFFAOYSA-N"]}).to_hdf(
        stock_path,
        key="table",
    )
    monkeypatch.setenv("SUPPLY_CATALOG_URI", stock_path.as_uri())
    module = _load_module(
        "supply_catalog_hdf5_readiness_scope_test",
        ROOT / "services/supply-oracle-svc/src/supply_oracle_svc/main.py",
    )

    def reject_checksum(path: Path) -> str:
        raise AssertionError(f"readiness hashed the full catalog: {path}")

    monkeypatch.setattr(module, "_catalog_checksum", reject_checksum)

    status = module.runtime_status()[0]

    assert status["available"] is True


def test_supply_catalog_checksum_streams_file_bytes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    catalog_path = tmp_path / "catalog.bin"
    catalog_path.write_bytes(b"catalog-bytes")
    module = _load_module(
        "supply_catalog_streaming_checksum_test",
        ROOT / "services/supply-oracle-svc/src/supply_oracle_svc/main.py",
    )

    def reject_read_bytes(path: Path) -> bytes:
        raise AssertionError(f"read_bytes loaded the entire catalog: {path}")

    monkeypatch.setattr(Path, "read_bytes", reject_read_bytes)

    checksum = module._catalog_checksum(catalog_path)

    assert checksum == f"sha256:{hashlib.sha256(b'catalog-bytes').hexdigest()}"


def test_supply_runtime_rejects_unsupported_catalog_uri(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SUPPLY_CATALOG_URI", "https://catalog.example/supply.json")
    module = _load_module(
        "supply_unsupported_catalog_uri_runtime_status_test",
        ROOT / "services/supply-oracle-svc/src/supply_oracle_svc/main.py",
    )

    with pytest.raises(RuntimeError, match="SUPPLY_CATALOG_URI must use file://"):
        module._require_runtime()


@pytest.mark.asyncio
async def test_supply_grpc_batch_echoes_request_and_candidate_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from mf_core.proto_gen.moleculeforge.v1.oracle import supply_pb2

    catalog_path = tmp_path / "supply_catalog.json"
    catalog_path.write_text(
        json.dumps(
            {
                "catalog_version": "catalog-v1",
                "records": [
                    {
                        "smiles": "CCO",
                        "catalog_id": "CAT-1",
                        "source": "local_catalog",
                        "source_timestamp": "2026-05-01T00:00:00Z",
                        "available": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("SUPPLY_CATALOG_URI", catalog_path.as_uri())
    module = _load_module(
        "supply_batch_correlation_test",
        ROOT / "services/supply-oracle-svc/src/supply_oracle_svc/main.py",
    )
    identity = {
        "project_id": "project-1",
        "candidate_id": "candidate-1",
        "candidate_index": 0,
        "canonical_smiles": "CCO",
    }

    response = await _supply_grpc_call(
        module,
        module.SupplyOracleGrpcServicer(),
        supply_pb2.BatchAvailabilityRequest(
            request_id="batch-1",
            requests=[
                supply_pb2.AvailabilityRequest(
                    smiles="CCO",
                    request_id="item-1",
                    **identity,
                ),
                supply_pb2.AvailabilityRequest(
                    smiles="CCN",
                    request_id="item-2",
                    **identity,
                ),
            ],
            **identity,
        ),
        "BatchCheck",
    )

    assert response.request_id == "batch-1"
    assert response.project_id == "project-1"
    assert response.candidate_id == "candidate-1"
    assert response.HasField("candidate_index")
    assert response.candidate_index == 0
    assert response.canonical_smiles == "CCO"
    assert [item.request_id for item in response.results] == ["item-1", "item-2"]
    assert [item.available for item in response.results] == [True, False]
    assert all(item.evidence_id for item in response.results)
    assert all(item.catalog_version == "catalog-v1" for item in response.results)
    assert all(
        item.catalog_checksum == response.results[0].catalog_checksum for item in response.results
    )


@pytest.mark.asyncio
async def test_supply_agent_grpc_client_preserves_correlation_and_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    catalog_path = tmp_path / "supply_catalog.json"
    catalog_path.write_text(
        json.dumps(
            {
                "catalog_version": "catalog-v1",
                "records": [
                    {
                        "smiles": "CCO",
                        "catalog_id": "CAT-1",
                        "source": "local_catalog",
                        "source_timestamp": "2026-05-01T00:00:00Z",
                        "available": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("SUPPLY_CATALOG_URI", catalog_path.as_uri())
    service_module = _load_module(
        "supply_agent_grpc_service_test",
        ROOT / "services/supply-oracle-svc/src/supply_oracle_svc/main.py",
    )
    agent_module = _load_module(
        "supply_agent_grpc_client_test",
        ROOT / "agents/supply_agent/src/supply_agent/agent.py",
    )
    server = grpc.aio.server()
    service_module.register_grpc_services(server)
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    client = agent_module.SupplyOracleGrpcClient(f"127.0.0.1:{port}")
    identity = {
        "request_id": "request-1:supply:0",
        "project_id": "project-1",
        "candidate_id": "candidate-1",
        "candidate_index": 0,
        "canonical_smiles": "CCO",
    }
    try:
        response = await client.check_availability("CCO", **identity)
    finally:
        await client.close()
        await server.stop(None)

    assert response["smiles"] == "CCO"
    assert response["available"] is True
    assert response["evidence_id"].startswith("sha256:")
    assert response["catalog_version"] == "catalog-v1"
    assert response["catalog_checksum"] == (
        f"sha256:{hashlib.sha256(catalog_path.read_bytes()).hexdigest()}"
    )
    assert {field: response[field] for field in identity} == identity


@pytest.mark.asyncio
async def test_supply_agent_grpc_client_uses_one_ordered_batch_rpc() -> None:
    from mf_core.proto_gen.moleculeforge.v1.oracle import supply_pb2

    module = _load_module(
        "supply_agent_batch_client_test",
        ROOT / "agents/supply_agent/src/supply_agent/agent.py",
    )

    class Stub:
        def __init__(self) -> None:
            self.requests = []

        async def BatchCheck(self, request):
            self.requests.append(request)
            return supply_pb2.BatchAvailabilityResponse(
                request_id=request.request_id,
                project_id=request.project_id,
                candidate_id=request.candidate_id,
                candidate_index=request.candidate_index,
                canonical_smiles=request.canonical_smiles,
                results=[
                    supply_pb2.AvailabilityResponse(
                        smiles=item.smiles,
                        available=True,
                        catalog_id=f"catalog-{index}",
                        catalog_source="test",
                        source_timestamp="2026-07-29T00:00:00Z",
                        evidence_id=f"evidence-{index}",
                        catalog_version="catalog-v1",
                        catalog_checksum="sha256:" + "a" * 64,
                        request_id=item.request_id,
                        project_id=item.project_id,
                        candidate_id=item.candidate_id,
                        candidate_index=item.candidate_index,
                        canonical_smiles=item.canonical_smiles,
                    )
                    for index, item in enumerate(request.requests)
                ],
            )

    client = module.SupplyOracleGrpcClient.__new__(module.SupplyOracleGrpcClient)
    client.stub = Stub()
    result = await client.batch_check(
        ["CC", "CN"],
        request_id="request-batch",
        project_id="project-1",
        candidate_id="candidate-1",
        candidate_index=0,
        canonical_smiles="CCO",
    )

    assert len(client.stub.requests) == 1
    assert [item.smiles for item in client.stub.requests[0].requests] == ["CC", "CN"]
    assert [item.request_id for item in client.stub.requests[0].requests] == [
        "request-batch:supply:0",
        "request-batch:supply:1",
    ]
    assert [item["smiles"] for item in result["results"]] == ["CC", "CN"]
    assert [item["request_id"] for item in result["results"]] == [
        "request-batch:supply:0",
        "request-batch:supply:1",
    ]


@pytest.mark.asyncio
async def test_supply_agent_batches_selected_route_and_echoes_route_id() -> None:
    module = _load_module(
        "supply_agent_selected_route_batch_test",
        ROOT / "agents/supply_agent/src/supply_agent/agent.py",
    )

    class Client:
        def __init__(self) -> None:
            self.calls = []

        async def batch_check(self, smiles_list, **identity):
            self.calls.append((list(smiles_list), dict(identity)))
            return {
                **identity,
                "results": [
                    {
                        "smiles": smiles,
                        "available": True,
                        "catalog_id": f"catalog-{index}",
                        "source": "test",
                        "source_timestamp": "2026-07-29T00:00:00Z",
                        "evidence_id": f"evidence-{index}",
                        "catalog_version": "catalog-v1",
                        "catalog_checksum": "sha256:" + "a" * 64,
                        **{
                            **identity,
                            "request_id": f"{identity['request_id']}:supply:{index}",
                        },
                    }
                    for index, smiles in enumerate(smiles_list)
                ],
            }

        async def check_availability(self, *_args, **_kwargs):
            raise AssertionError("selected route must use BatchCheck")

    client = Client()
    result = await module.SupplyAgent(
        supply_client=client,
        crg_repository=None,
    ).process(
        {
            "workflow_scope": "full",
            "route_id": "route-a",
            "request_id": "request-supply",
            "project_id": "project-1",
            "candidate_id": "candidate-1",
            "candidate_index": 0,
            "canonical_smiles": "CCO",
            "smiles": "CCO",
            "building_blocks": [{"smiles": "CC"}, {"smiles": "CN"}],
        }
    )

    assert client.calls == [
        (
            ["CC", "CN"],
            {
                "request_id": "request-supply",
                "project_id": "project-1",
                "candidate_id": "candidate-1",
                "candidate_index": 0,
                "canonical_smiles": "CCO",
            },
        )
    ]
    assert result["route_id"] == "route-a"
    assert [item["smiles"] for item in result["block_assessments"]] == ["CC", "CN"]


@pytest.mark.asyncio
async def test_supply_grpc_maps_request_catalog_runtime_timeout_and_internal_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mf_core.proto_gen.moleculeforge.v1.oracle import supply_pb2

    module = _load_module(
        "supply_grpc_error_mapping_test",
        ROOT / "services/supply-oracle-svc/src/supply_oracle_svc/main.py",
    )
    identity = {
        "project_id": "project-1",
        "candidate_id": "candidate-1",
        "candidate_index": 0,
        "canonical_smiles": "CCO",
    }

    with pytest.raises(grpc.aio.AioRpcError) as invalid:
        await _supply_grpc_call(
            module,
            module.SupplyOracleGrpcServicer(),
            supply_pb2.AvailabilityRequest(smiles="CCO", **identity),
            "CheckAvailability",
        )
    assert invalid.value.code() == grpc.StatusCode.INVALID_ARGUMENT

    with pytest.raises(grpc.aio.AioRpcError) as invalid_smiles:
        await _supply_grpc_call(
            module,
            module.SupplyOracleGrpcServicer(),
            supply_pb2.AvailabilityRequest(
                smiles="not-smiles",
                request_id="request-invalid-smiles",
                **identity,
            ),
            "CheckAvailability",
        )
    assert invalid_smiles.value.code() == grpc.StatusCode.INVALID_ARGUMENT

    monkeypatch.delenv("SUPPLY_CATALOG_URI", raising=False)
    with pytest.raises(grpc.aio.AioRpcError) as unavailable:
        await _supply_grpc_call(
            module,
            module.SupplyOracleGrpcServicer(),
            supply_pb2.AvailabilityRequest(
                smiles="CCO",
                request_id="request-unavailable",
                **identity,
            ),
            "CheckAvailability",
        )
    assert unavailable.value.code() == grpc.StatusCode.FAILED_PRECONDITION

    class MissingCatalog:
        async def get_price(self, smiles=None, catalog_id=None) -> dict:
            raise KeyError("catalog entry was not found")

    with pytest.raises(grpc.aio.AioRpcError) as not_found:
        await _supply_grpc_call(
            module,
            module.SupplyOracleGrpcServicer(
                service=module.SupplyOracleServicer(catalog_client=MissingCatalog())
            ),
            supply_pb2.CatalogPriceRequest(
                smiles="CCO",
                catalog_id="missing",
                request_id="request-not-found",
                **identity,
            ),
            "GetCatalogPrice",
        )
    assert not_found.value.code() == grpc.StatusCode.NOT_FOUND

    class TimedOutService:
        async def CheckAvailability(self, request, context):
            raise TimeoutError("provider timed out")

    with pytest.raises(grpc.aio.AioRpcError) as timed_out:
        await _supply_grpc_call(
            module,
            module.SupplyOracleGrpcServicer(service=TimedOutService()),
            supply_pb2.AvailabilityRequest(
                smiles="CCO",
                request_id="request-timeout",
                **identity,
            ),
            "CheckAvailability",
        )
    assert timed_out.value.code() == grpc.StatusCode.DEADLINE_EXCEEDED

    class MalformedCatalog:
        async def check_availability(self, smiles: str) -> dict:
            return {"smiles": smiles, "available": False}

    with pytest.raises(grpc.aio.AioRpcError) as data_loss:
        await _supply_grpc_call(
            module,
            module.SupplyOracleGrpcServicer(
                service=module.SupplyOracleServicer(catalog_client=MalformedCatalog())
            ),
            supply_pb2.AvailabilityRequest(
                smiles="CCO",
                request_id="request-data-loss",
                **identity,
            ),
            "CheckAvailability",
        )
    assert data_loss.value.code() == grpc.StatusCode.DATA_LOSS

    class MalformedPriceCatalog:
        async def get_price(self, smiles=None, catalog_id=None) -> str:
            return "not-a-catalog-record"

    with pytest.raises(grpc.aio.AioRpcError) as price_data_loss:
        await _supply_grpc_call(
            module,
            module.SupplyOracleGrpcServicer(
                service=module.SupplyOracleServicer(catalog_client=MalformedPriceCatalog())
            ),
            supply_pb2.CatalogPriceRequest(
                catalog_id="malformed",
                request_id="request-price-data-loss",
                **identity,
            ),
            "GetCatalogPrice",
        )
    assert price_data_loss.value.code() == grpc.StatusCode.DATA_LOSS

    class BrokenService:
        async def CheckAvailability(self, request, context):
            raise RuntimeError("unexpected provider failure")

    with pytest.raises(grpc.aio.AioRpcError) as internal:
        await _supply_grpc_call(
            module,
            module.SupplyOracleGrpcServicer(service=BrokenService()),
            supply_pb2.AvailabilityRequest(
                smiles="CCO",
                request_id="request-internal",
                **identity,
            ),
            "CheckAvailability",
        )
    assert internal.value.code() == grpc.StatusCode.INTERNAL


def test_deployment_declares_remaining_runtime_config_data() -> None:
    k8s = (ROOT / "infra/kubernetes/deployments/moleculeforge-services.yaml").read_text(
        encoding="utf-8"
    )
    helm_values = (ROOT / "infra/helm/moleculeforge/values.yaml").read_text(encoding="utf-8")

    for config in (
        _k8s_configmap_data(k8s, "mf-agents", "cig-compiler-config"),
        _helm_configmap_data(helm_values, "mf-agents", "cig-compiler-config"),
    ):
        assert config["semantic-parser-uri"] == ""
        assert config["semantic-parser-command"] == ""
        assert config["semantic-parser-timeout-seconds"] == "30"
        assert config["refinement-command"] == ""
        assert config["refinement-timeout-seconds"] == "60"
        assert config["hciv-checkpoint-path"] == ""
        assert config["chembl-target-url"] == ""
        assert config["chembl-target-search-url"] == ""
        assert config["uniprot-search-url"] == ""
        assert config["rcsb-search-url"] == ""

    for config in (
        _k8s_configmap_data(k8s, "mf-mlops", "feature-store-config"),
        _helm_configmap_data(helm_values, "mf-mlops", "feature-store-config"),
    ):
        assert config["feast-repo-path"] == "feature_repo"

    for config in (
        _k8s_configmap_data(k8s, "mf-agents", "hypseek-teacher-config"),
        _helm_configmap_data(helm_values, "mf-agents", "hypseek-teacher-config"),
    ):
        assert config["teacher-source"] == "hypseek"
        assert config["teacher-version"] == ""
        assert config["teacher-command"] == ""
        assert config["teacher-timeout-seconds"] == "60"

    for config in (
        _k8s_configmap_data(k8s, "mf-agents", "tar-proxyless-search-config"),
        _helm_configmap_data(helm_values, "mf-agents", "tar-proxyless-search-config"),
    ):
        assert config["proxyless-search-command"] == ""
        assert config["proxyless-search-timeout-seconds"] == "300"

    for config in (
        _k8s_configmap_data(k8s, "mf-agents", "sila2-adapter-config"),
        _helm_configmap_data(helm_values, "mf-agents", "sila2-adapter-config"),
    ):
        assert config["plan-command"] == ""
        assert config["plan-timeout-seconds"] == "120"

    for config in (
        _k8s_configmap_data(k8s, "mf-mlops", "pareto-bo-config"),
        _helm_configmap_data(helm_values, "mf-mlops", "pareto-bo-config"),
    ):
        assert config["candidate-provider"] == ""
        assert config["candidate-provider-command"] == ""
        assert config["oracle-evaluate"] == ""
        assert config["oracle-evaluate-command"] == ""
        assert config["command-timeout-seconds"] == "300"

    for config in (
        _k8s_configmap_data(k8s, "mf-agents", "sigstore-provenance-config"),
        _helm_configmap_data(helm_values, "mf-agents", "sigstore-provenance-config"),
    ):
        assert config["sign-command"] == (
            "python /workspace/tools/sigstore/cosign_audit_wrapper.py sign"
        )
        assert config["verify-command"] == (
            "python /workspace/tools/sigstore/cosign_audit_wrapper.py verify"
        )
        assert config["expected-identity"] == ""
        assert config["rekor-url"] == "https://rekor.sigstore.dev"
        assert config["command-timeout-seconds"] == "30"

    for secret in (
        _k8s_secret_string_data(k8s, "mf-agents", "sigstore-provenance"),
        _helm_secret_string_data(helm_values, "mf-agents", "sigstore-provenance"),
    ):
        assert secret["identity-token"] == ""


def test_supply_oracle_deployment_readiness_validates_catalog_schema() -> None:
    import yaml

    compose = yaml.safe_load(
        (ROOT / "infra/docker/docker-compose.dev.yml").read_text(encoding="utf-8")
    )
    k8s = (ROOT / "infra/kubernetes/deployments/moleculeforge-services.yaml").read_text(
        encoding="utf-8"
    )
    deployments = {
        item["metadata"]["name"]: item
        for item in yaml.safe_load_all(k8s)
        if item and item.get("kind") == "Deployment"
    }
    helm_values_text = (ROOT / "infra/helm/moleculeforge/values.yaml").read_text(encoding="utf-8")
    helm_values = yaml.safe_load(helm_values_text)
    probe_command = [
        "python",
        "-c",
        "from supply_oracle_svc.main import _require_runtime; _require_runtime()",
    ]

    compose_service = compose["services"]["supply-oracle-svc"]
    assert "SUPPLY_CATALOG_URI" in compose_service["environment"]
    assert compose_service["healthcheck"]["test"] == ["CMD", *probe_command]

    expected_catalog_uri = ""
    assert _k8s_configmap_data(k8s, "mf-oracles", "supply-oracle-config") == {
        "catalog-uri": expected_catalog_uri
    }
    k8s_container = deployments["supply-oracle-svc"]["spec"]["template"]["spec"]["containers"][0]
    k8s_env = {item["name"]: item for item in k8s_container["env"]}
    assert k8s_env["SUPPLY_CATALOG_URI"]["valueFrom"]["configMapKeyRef"] == {
        "name": "supply-oracle-config",
        "key": "catalog-uri",
    }
    assert k8s_container["readinessProbe"]["exec"]["command"] == probe_command

    assert _helm_configmap_data(
        helm_values_text,
        "mf-oracles",
        "supply-oracle-config",
    ) == {"catalog-uri": expected_catalog_uri}
    helm_service = helm_values["services"]["supply-oracle-svc"]
    assert helm_service["envValueFrom"]["SUPPLY_CATALOG_URI"]["configMapKeyRef"] == {
        "name": "supply-oracle-config",
        "key": "catalog-uri",
    }
    assert helm_service["readinessProbe"]["exec"]["command"] == probe_command


@pytest.mark.asyncio
async def test_supply_agent_aggregates_catalog_availability() -> None:
    module = _load_module(
        "supply_agent_catalog_aggregation_test",
        ROOT / "agents/supply_agent/src/supply_agent/agent.py",
    )

    class CatalogClient:
        def __init__(self) -> None:
            self.calls: list[str] = []

        async def check_availability(self, smiles: str, **identity) -> dict:
            self.calls.append(smiles)
            records = {
                "CCO": {
                    "smiles": "CCO",
                    "available": True,
                    "catalog_id": "CAT-A-1",
                    "source": "local_catalog_a",
                    "source_timestamp": "2026-05-01T00:00:00Z",
                    "price": 10.0,
                    "currency": "USD",
                    "lead_time_days": 2,
                    "evidence_id": "evidence-cco",
                    "catalog_version": "catalog-v1",
                    "catalog_checksum": f"sha256:{'a' * 64}",
                    **identity,
                },
                "O=O": {
                    "smiles": "O=O",
                    "available": True,
                    "catalog_id": "CAT-B-1",
                    "source": "local_catalog_b",
                    "source_timestamp": "2026-05-02T00:00:00Z",
                    "price": 20.0,
                    "currency": "USD",
                    "lead_time_days": 6,
                    "evidence_id": "evidence-oo",
                    "catalog_version": "catalog-v1",
                    "catalog_checksum": f"sha256:{'a' * 64}",
                    **identity,
                },
                "N#N": {
                    "smiles": "N#N",
                    "available": False,
                    "catalog_id": None,
                    "source": None,
                    "source_timestamp": None,
                    "price": None,
                    "currency": None,
                    "lead_time_days": None,
                    "evidence_id": "evidence-nn",
                    "catalog_version": "catalog-v1",
                    "catalog_checksum": f"sha256:{'a' * 64}",
                    **identity,
                },
            }
            return records[smiles]

    catalog_client = CatalogClient()
    agent = module.SupplyAgent(supply_client=catalog_client)

    result = await agent.process(
        {
            "project_id": "project-1",
            "run_id": "run-1",
            "request_id": "request-1",
            "smiles": "CCOON",
            "canonical_smiles": "CCOON",
            "candidate_id": "candidate-1",
            "candidate_index": 2,
            "building_blocks": ["CCO", {"smiles": "O=O"}, {"building_block_smiles": "N#N"}],
        }
    )

    assessment = result["supply_assessment"]
    assert catalog_client.calls == ["CCO", "O=O", "N#N"]
    assert assessment["total_blocks"] == 3
    assert assessment["commercially_available"] == 2
    assert assessment["avg_price_per_gram"] == pytest.approx(15.0)
    assert assessment["avg_lead_time_days"] == pytest.approx(4.0)
    assert assessment["supplier_diversity"] == 2
    assert assessment["overall_feasibility"] == "partial"
    assert result["block_assessments"][0]["catalog_source"] == "local_catalog_a"
    assert result["block_assessments"][0]["request_id"] == "request-1:supply:0"
    assert result["block_assessments"][0]["evidence_id"] == "evidence-cco"
    assert result["block_assessments"][2]["evidence_id"] == "evidence-nn"
    assert result["project_id"] == "project-1"
    assert result["candidate_id"] == "candidate-1"
    assert result["candidate_index"] == 2
    assert result["canonical_smiles"] == "CCOON"


@pytest.mark.asyncio
async def test_supply_agent_preserves_legacy_smiles_and_building_blocks_request() -> None:
    module = _load_module(
        "supply_agent_legacy_request_test",
        ROOT / "agents/supply_agent/src/supply_agent/agent.py",
    )

    class CatalogClient:
        async def check_availability(self, smiles: str) -> dict:
            return {
                "smiles": smiles,
                "available": True,
                "catalog_id": "CAT-1",
                "source": "legacy_catalog",
                "source_timestamp": "2026-07-30T00:00:00Z",
                "price": 5.0,
                "currency": "USD",
                "lead_time_days": 1,
            }

    result = await module.SupplyAgent(
        supply_client=CatalogClient(),
        crg_repository=None,
    ).process(
        {
            "smiles": "CCO",
            "building_blocks": ["CC"],
        }
    )

    assert result["smiles"] == "CCO"
    assert result["supply_assessment"]["overall_feasibility"] == "available"
    assert result["block_assessments"][0]["catalog_id"] == "CAT-1"
    assert "candidate_id" not in result


@pytest.mark.asyncio
async def test_supply_agent_rejects_provider_correlation_mismatch() -> None:
    module = _load_module(
        "supply_agent_provider_correlation_test",
        ROOT / "agents/supply_agent/src/supply_agent/agent.py",
    )

    class CatalogClient:
        async def check_availability(self, smiles: str, **identity) -> dict:
            return {
                "smiles": smiles,
                "available": False,
                "evidence_id": "evidence-1",
                "catalog_version": "catalog-v1",
                "catalog_checksum": f"sha256:{'a' * 64}",
                **identity,
                "candidate_id": "different-candidate",
            }

    agent = module.SupplyAgent(supply_client=CatalogClient())

    with pytest.raises(RuntimeError, match="candidate_id does not match request"):
        await agent.process(
            {
                "project_id": "project-1",
                "run_id": "run-1",
                "request_id": "request-1",
                "smiles": "CCO",
                "canonical_smiles": "CCO",
                "candidate_id": "candidate-1",
                "candidate_index": 0,
                "building_blocks": ["CCO"],
            }
        )


@pytest.mark.asyncio
async def test_supply_agent_persists_supply_feasibility_belief() -> None:
    module = _load_module(
        "supply_agent_crg_repository_test",
        ROOT / "agents/supply_agent/src/supply_agent/agent.py",
    )

    class CatalogClient:
        async def check_availability(self, smiles: str, **identity) -> dict:
            return {
                "smiles": smiles,
                "available": True,
                "catalog_id": "CAT-1",
                "source": "local_catalog",
                "source_timestamp": "2026-05-01T00:00:00Z",
                "price": 10.0,
                "currency": "USD",
                "lead_time_days": 2,
                "evidence_id": "evidence-1",
                "catalog_version": "catalog-v1",
                "catalog_checksum": f"sha256:{'a' * 64}",
                **identity,
            }

    class CRGRepository:
        def __init__(self) -> None:
            self.beliefs: list[dict] = []

        async def write_workflow_belief(self, **kwargs) -> None:
            self.beliefs.append(kwargs)

    repository = CRGRepository()
    agent = module.SupplyAgent(
        supply_client=CatalogClient(),
        crg_repository=repository,
    )

    await agent.process(
        {
            "project_id": "project-1",
            "run_id": "run-1",
            "request_id": "request-1",
            "smiles": "CCO",
            "canonical_smiles": "CCO",
            "candidate_id": "candidate-1",
            "candidate_index": 0,
            "building_blocks": ["CCO"],
        }
    )

    assert len(repository.beliefs) == 1
    belief = repository.beliefs[0]
    assert belief["project_id"] == "project-1"
    assert belief["run_id"] == "run-1"
    assert belief["subject"] == "CCO"
    assert belief["predicate"] == "supply_feasibility"
    assert belief["object_value"] == "available"
    assert belief["source_agent"] == "supply_agent"
    assert belief["evidence_ids"] == ["evidence-1"]


@pytest.mark.asyncio
async def test_supply_agent_does_not_skip_provider_for_zero_retrosyn_routes_crg_belief() -> None:
    module = _load_module(
        "supply_agent_crg_readback_test",
        ROOT / "agents/supply_agent/src/supply_agent/agent.py",
    )

    class CRGRepository:
        def __init__(self) -> None:
            self.beliefs: list[dict] = []

        async def get_run_crg(self, run_id: str) -> dict:
            assert run_id == "run-1"
            return {
                "beliefs": [
                    {
                        "subject": "CCO",
                        "predicate": "retrosyn_routes",
                        "object_value": "0",
                    }
                ]
            }

        async def write_workflow_belief(self, **kwargs) -> None:
            self.beliefs.append(kwargs)

    class CatalogClient:
        def __init__(self) -> None:
            self.calls: list[str] = []

        async def check_availability(self, smiles: str, **identity) -> dict:
            self.calls.append(smiles)
            return {
                "smiles": smiles,
                "available": True,
                "catalog_id": "CAT-1",
                "source": "real_catalog",
                "source_timestamp": "2026-05-01T00:00:00Z",
                "price": 4.0,
                "currency": "USD",
                "lead_time_days": 1,
                "evidence_id": "evidence-real",
                "catalog_version": "catalog-v1",
                "catalog_checksum": f"sha256:{'a' * 64}",
                **identity,
            }

    repository = CRGRepository()
    catalog_client = CatalogClient()
    agent = module.SupplyAgent(supply_client=catalog_client, crg_repository=repository)

    result = await agent.process(
        {
            "project_id": "project-1",
            "run_id": "run-1",
            "request_id": "request-1",
            "smiles": "CCO",
            "canonical_smiles": "CCO",
            "candidate_id": "candidate-1",
            "candidate_index": 0,
            "building_blocks": ["CCO"],
        }
    )

    assert result["status"] == "assessed"
    assert catalog_client.calls == ["CCO"]
    assert result["supply_assessment"]["overall_feasibility"] == "available"
    assert result["block_assessments"][0]["catalog_id"] == "CAT-1"
    assert repository.beliefs[0]["predicate"] == "supply_feasibility"
    assert repository.beliefs[0]["object_value"] == "available"
    assert repository.beliefs[0]["evidence_ids"] == ["evidence-real"]


@pytest.mark.asyncio
async def test_supply_agent_does_not_reconstruct_blocks_from_supply_feasibility_crg_belief() -> (
    None
):
    module = _load_module(
        "supply_agent_supply_cache_readback_test",
        ROOT / "agents/supply_agent/src/supply_agent/agent.py",
    )

    class CRGRepository:
        def __init__(self) -> None:
            self.beliefs: list[dict] = []

        async def get_run_crg(self, run_id: str) -> dict:
            assert run_id == "run-1"
            return {
                "beliefs": [
                    {
                        "subject": "CCO",
                        "predicate": "supply_feasibility",
                        "object_value": "available",
                        "confidence": 0.9,
                    }
                ]
            }

        async def write_workflow_belief(self, **kwargs) -> None:
            self.beliefs.append(kwargs)

    class CatalogClient:
        def __init__(self) -> None:
            self.calls: list[str] = []

        async def check_availability(self, smiles: str, **identity) -> dict:
            self.calls.append(smiles)
            return {
                "smiles": smiles,
                "available": smiles == "CCO",
                "catalog_id": "CAT-1" if smiles == "CCO" else None,
                "source": "real_catalog" if smiles == "CCO" else None,
                "source_timestamp": ("2026-05-01T00:00:00Z" if smiles == "CCO" else None),
                "price": 4.0 if smiles == "CCO" else None,
                "currency": "USD" if smiles == "CCO" else None,
                "lead_time_days": 1 if smiles == "CCO" else None,
                "evidence_id": f"evidence-{smiles}",
                "catalog_version": "catalog-v1",
                "catalog_checksum": f"sha256:{'a' * 64}",
                **identity,
            }

    repository = CRGRepository()
    catalog_client = CatalogClient()
    agent = module.SupplyAgent(supply_client=catalog_client, crg_repository=repository)

    result = await agent.process(
        {
            "project_id": "project-1",
            "run_id": "run-1",
            "request_id": "request-1",
            "smiles": "CCO",
            "canonical_smiles": "CCO",
            "candidate_id": "candidate-1",
            "candidate_index": 0,
            "building_blocks": ["CCO", "CCN"],
        }
    )

    assert result["status"] == "assessed"
    assert catalog_client.calls == ["CCO", "CCN"]
    assert result["supply_assessment"]["overall_feasibility"] == "partial"
    assert result["supply_assessment"]["commercially_available"] == 1
    assert result["block_assessments"][0]["catalog_id"] == "CAT-1"
    assert result["block_assessments"][1]["evidence_id"] == "evidence-CCN"
    assert repository.beliefs[0]["object_value"] == "partial"


@pytest.mark.asyncio
async def test_supply_agent_requires_catalog_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SUPPLY_ORACLE_TARGET", raising=False)
    monkeypatch.delenv("SUPPLY_CATALOG_URI", raising=False)
    module = _load_module(
        "supply_agent_requires_catalog_client_test",
        ROOT / "agents/supply_agent/src/supply_agent/agent.py",
    )
    agent = module.SupplyAgent()

    with pytest.raises(RuntimeError, match="SUPPLY_ORACLE_TARGET"):
        await agent.process(
            {
                "project_id": "project-1",
                "run_id": "run-1",
                "request_id": "request-1",
                "smiles": "CCO",
                "canonical_smiles": "CCO",
                "candidate_id": "candidate-1",
                "candidate_index": 0,
                "building_blocks": ["CCO"],
            }
        )


def test_fep_runtime_does_not_claim_unwired_openfe_executable_ready(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = tmp_path / "openfe"
    runner.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    runner.chmod(0o755)
    monkeypatch.setenv("OPENFE_RUNNER_PATH", str(runner))
    module = _load_module(
        "fep_runner_only_test",
        ROOT / "services/fep-svc/src/fep_svc/main.py",
    )

    status = module.runtime_status()

    assert status[0]["name"] == "fep_oracle_command"
    assert status[0]["available"] is False
    assert status[0]["required"] is True


def test_fep_runtime_rejects_missing_oracle_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FEP_ORACLE_COMMAND", "missing-fep-runner --json")
    module = _load_module(
        "fep_missing_oracle_command_runtime_test",
        ROOT / "services/fep-svc/src/fep_svc/main.py",
    )

    status = module.runtime_status()

    command_status = next(item for item in status if item["name"] == "fep_oracle_command")
    assert command_status["configured"] is True
    assert command_status["available"] is False
    assert command_status["source"] == "FEP_ORACLE_COMMAND"
    assert "not found" in command_status["message"]


def test_fep_runtime_rejects_builtin_chain_without_input_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module(
        "fep_missing_input_registry_runtime_test",
        ROOT / "services/fep-svc/src/fep_svc/main.py",
    )
    monkeypatch.setenv(
        "FEP_ORACLE_COMMAND",
        f"{sys.executable} {ROOT / 'tools/oracles/fep_oracle_wrapper.py'}",
    )
    monkeypatch.setenv(
        "OPENFE_RUNNER_PATH",
        f"{sys.executable} {ROOT / 'tools/oracles/openfe_json_runner.py'}",
    )
    for name in (
        "OPENFE_RESULT_REPLAY_PATH",
        "OPENFE_RESULT_REGISTRY",
        "OPENFE_TRANSFORMATION_REGISTRY",
    ):
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(RuntimeError, match="openfe_input_source"):
        module._require_runtime()

    status = {item["name"]: item for item in module.runtime_status()}
    assert status["openfe_runner_command"]["available"] is True
    assert status["openfe_input_source"]["available"] is False
    assert status["openfe_input_source"]["required"] is True


def test_fep_runtime_accepts_relative_transformation_registry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_module(
        "fep_relative_registry_runtime_test",
        ROOT / "services/fep-svc/src/fep_svc/main.py",
    )
    complex_transformation = tmp_path / "transformations" / "edge-complex.json"
    solvent_transformation = tmp_path / "transformations" / "edge-solvent.json"
    complex_transformation.parent.mkdir()
    _write_openfe_transformation(complex_transformation)
    _write_openfe_transformation(solvent_transformation)
    registry = tmp_path / "transformation-registry.json"
    registry.write_text(
        json.dumps(
            {
                "7abc": {
                    "CCO>>CCN": {
                        "complex": "transformations/edge-complex.json",
                        "solvent": "transformations/edge-solvent.json",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    cli = tmp_path / "openfe"
    cli.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    cli.chmod(0o755)
    monkeypatch.setenv(
        "FEP_ORACLE_COMMAND",
        f"{sys.executable} {ROOT / 'tools/oracles/fep_oracle_wrapper.py'}",
    )
    monkeypatch.setenv(
        "OPENFE_RUNNER_PATH",
        f"{sys.executable} {ROOT / 'tools/oracles/openfe_json_runner.py'}",
    )
    monkeypatch.setenv("OPENFE_CLI_PATH", str(cli))
    monkeypatch.setenv("OPENFE_TRANSFORMATION_REGISTRY", str(registry))

    module._require_runtime()

    status = {item["name"]: item for item in module.runtime_status()}
    assert status["openfe_transformation_registry"]["available"] is True
    assert status["openfe_cli_command"]["available"] is True


@pytest.mark.parametrize(
    "registry_format",
    ("missing_leg", "string", "one_item_list", "two_item_list"),
)
def test_fep_runtime_rejects_incomplete_rbfe_transformation_registry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    registry_format: str,
) -> None:
    module = _load_module(
        "fep_incomplete_registry_runtime_test",
        ROOT / "services/fep-svc/src/fep_svc/main.py",
    )
    transformation = tmp_path / "edge-complex.json"
    transformation.write_text("{}", encoding="utf-8")
    registry_value: object
    if registry_format == "missing_leg":
        registry_value = {"complex": str(transformation)}
    elif registry_format == "string":
        registry_value = str(transformation)
    elif registry_format == "one_item_list":
        registry_value = [str(transformation)]
    else:
        registry_value = [str(transformation), str(transformation)]
    registry = tmp_path / "transformation-registry.json"
    registry.write_text(
        json.dumps(
            {
                "7abc": {
                    "CCO>>CCN": registry_value,
                }
            }
        ),
        encoding="utf-8",
    )
    cli = tmp_path / "openfe"
    cli.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    cli.chmod(0o755)
    monkeypatch.setenv(
        "FEP_ORACLE_COMMAND",
        f"{sys.executable} {ROOT / 'tools/oracles/fep_oracle_wrapper.py'}",
    )
    monkeypatch.setenv(
        "OPENFE_RUNNER_PATH",
        f"{sys.executable} {ROOT / 'tools/oracles/openfe_json_runner.py'}",
    )
    monkeypatch.setenv("OPENFE_CLI_PATH", str(cli))
    monkeypatch.setenv("OPENFE_TRANSFORMATION_REGISTRY", str(registry))

    with pytest.raises(RuntimeError, match="openfe_transformation_registry"):
        module._require_runtime()

    status = {item["name"]: item for item in module.runtime_status()}
    assert status["openfe_transformation_registry"]["available"] is False
    assert "complex and solvent" in status["openfe_transformation_registry"]["message"]


@pytest.mark.parametrize(
    ("protein_id", "pair_key"),
    (
        ("", "CCO>>CCN"),
        (" 7abc", "CCO>>CCN"),
        ("7abc", ""),
        ("7abc", "CCO"),
        ("7abc", "CCO>>>>CCN"),
        ("7abc", ">>CCN"),
        ("7abc", "CCO>>"),
    ),
)
def test_fep_runtime_rejects_unaddressable_transformation_registry_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    protein_id: str,
    pair_key: str,
) -> None:
    module = _load_module(
        "fep_registry_identity_runtime_test",
        ROOT / "services/fep-svc/src/fep_svc/main.py",
    )
    complex_transformation = tmp_path / "edge-complex.json"
    solvent_transformation = tmp_path / "edge-solvent.json"
    _write_openfe_transformation(complex_transformation)
    _write_openfe_transformation(solvent_transformation)
    registry = tmp_path / "transformation-registry.json"
    registry.write_text(
        json.dumps(
            {
                protein_id: {
                    pair_key: {
                        "complex": str(complex_transformation),
                        "solvent": str(solvent_transformation),
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    cli = tmp_path / "openfe"
    cli.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    cli.chmod(0o755)
    monkeypatch.setenv(
        "FEP_ORACLE_COMMAND",
        f"{sys.executable} {ROOT / 'tools/oracles/fep_oracle_wrapper.py'}",
    )
    monkeypatch.setenv(
        "OPENFE_RUNNER_PATH",
        f"{sys.executable} {ROOT / 'tools/oracles/openfe_json_runner.py'}",
    )
    monkeypatch.setenv("OPENFE_CLI_PATH", str(cli))
    monkeypatch.setenv("OPENFE_TRANSFORMATION_REGISTRY", str(registry))

    with pytest.raises(RuntimeError, match="openfe_transformation_registry"):
        module._require_runtime()


def test_fep_runtime_rejects_nested_openfe_protocol_repeats(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_module(
        "fep_nested_protocol_repeats_runtime_test",
        ROOT / "services/fep-svc/src/fep_svc/main.py",
    )
    complex_transformation = _write_openfe_transformation(
        tmp_path / "edge-complex.json",
        protocol_repeats=3,
    )
    solvent_transformation = _write_openfe_transformation(
        tmp_path / "edge-solvent.json"
    )
    registry = tmp_path / "transformation-registry.json"
    registry.write_text(
        json.dumps(
            {
                "7abc": {
                    "CCO>>CCN": {
                        "complex": str(complex_transformation),
                        "solvent": str(solvent_transformation),
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    cli = tmp_path / "openfe"
    cli.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    cli.chmod(0o755)
    monkeypatch.setenv(
        "FEP_ORACLE_COMMAND",
        f"{sys.executable} {ROOT / 'tools/oracles/fep_oracle_wrapper.py'}",
    )
    monkeypatch.setenv(
        "OPENFE_RUNNER_PATH",
        f"{sys.executable} {ROOT / 'tools/oracles/openfe_json_runner.py'}",
    )
    monkeypatch.setenv("OPENFE_CLI_PATH", str(cli))
    monkeypatch.setenv("OPENFE_TRANSFORMATION_REGISTRY", str(registry))

    with pytest.raises(RuntimeError, match="openfe_transformation_registry"):
        module._require_runtime()

    status = {item["name"]: item for item in module.runtime_status()}
    assert "protocol_repeats=1" in status["openfe_transformation_registry"]["message"]


def test_fep_runtime_rejects_timeout_shorter_than_openfe_budget(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_module(
        "fep_short_timeout_runtime_test",
        ROOT / "services/fep-svc/src/fep_svc/main.py",
    )
    registry = tmp_path / "result-registry.json"
    registry.write_text('{"7abc": {"pair": {}}}', encoding="utf-8")
    monkeypatch.setenv(
        "FEP_ORACLE_COMMAND",
        f"{sys.executable} {ROOT / 'tools/oracles/fep_oracle_wrapper.py'}",
    )
    monkeypatch.setenv(
        "OPENFE_RUNNER_PATH",
        f"{sys.executable} {ROOT / 'tools/oracles/openfe_json_runner.py'}",
    )
    monkeypatch.setenv("OPENFE_RESULT_REGISTRY", str(registry))
    monkeypatch.setenv("FEP_ORACLE_TIMEOUT_SECONDS", "120")

    with pytest.raises(RuntimeError, match="openfe_timeout_configuration"):
        module._require_runtime()

    status = {item["name"]: item for item in module.runtime_status()}
    assert status["openfe_timeout_configuration"]["available"] is False
    assert "shorter" in status["openfe_timeout_configuration"]["message"]


def test_fep_runtime_rejects_malformed_result_registry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_module(
        "fep_malformed_result_registry_runtime_test",
        ROOT / "services/fep-svc/src/fep_svc/main.py",
    )
    registry = tmp_path / "result-registry.json"
    registry.write_text('{"7abc": {"CCO>>CCN": {"ddg_kcal_mol": -1.0}}}', encoding="utf-8")
    monkeypatch.setenv(
        "FEP_ORACLE_COMMAND",
        f"{sys.executable} {ROOT / 'tools/oracles/fep_oracle_wrapper.py'}",
    )
    monkeypatch.setenv(
        "OPENFE_RUNNER_PATH",
        f"{sys.executable} {ROOT / 'tools/oracles/openfe_json_runner.py'}",
    )
    monkeypatch.setenv("OPENFE_RESULT_REGISTRY", str(registry))

    with pytest.raises(RuntimeError, match="openfe_result_registry"):
        module._require_runtime()

    status = {item["name"]: item for item in module.runtime_status()}
    assert status["openfe_result_registry"]["available"] is False
    assert "missing fields" in status["openfe_result_registry"]["message"]


@pytest.mark.asyncio
async def test_fep_service_runs_configured_json_command(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from mf_core.proto_gen.moleculeforge.v1.oracle import fep_pb2

    module = _load_module(
        "fep_json_command_test",
        ROOT / "services/fep-svc/src/fep_svc/main.py",
    )
    runner = tmp_path / "fep_runner.py"
    runner.write_text(
        "import json, sys\n"
        "request = json.load(sys.stdin)\n"
        "assert request['reference_ligand_smiles'] == 'CCO'\n"
        "assert request['test_ligand_smiles'] == ['CCN']\n"
        "print(json.dumps({"
        "'batch_id': request['batch_id'], "
        "'request_id': request['request_id'], "
        "'project_id': request['project_id'], "
        "'protein_pdb_id': request['protein_pdb_id'], "
        "'reference_ligand_smiles': request['reference_ligand_smiles'], "
        "'test_ligand_smiles': request['test_ligand_smiles'], "
        "'method': request['method'], "
        "'n_repeats': request['n_repeats'], "
        "'total_elapsed_ms': 33, "
        "'results': [{"
        "'ligand_a_smiles': 'CCO', "
        "'ligand_b_smiles': 'CCN', "
        "'ddg_kcal_mol': -1.2, "
        "'ddg_uncertainty': 0.3, "
        "'n_repeats': 2, "
        "'method': 'openfe', "
        "'per_repeat_ddg': {'repeat_1': -1.1, 'repeat_2': -1.3}, "
        "'converged': True"
        "}]"
        "}))\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("FEP_ORACLE_COMMAND", f"{sys.executable} {runner}")
    request = fep_pb2.FEPBatchRequest(
        project_id="project-1",
        request_id="request-1",
        batch_id="batch-1",
        protein_pdb_id="7abc",
        reference_ligand_smiles="CCO",
        test_ligand_smiles=["CCN"],
        method="openfe",
        n_repeats=2,
    )

    response = await module.FEPServicer().RunFEP(request, None)

    assert response.batch_id == "batch-1"
    assert response.request_id == "request-1"
    assert response.total_elapsed_ms == 33
    assert response.results[0].ligand_a_smiles == "CCO"
    assert response.results[0].ligand_b_smiles == "CCN"
    assert response.results[0].ddg_kcal_mol == pytest.approx(-1.2)
    assert response.results[0].ddg_uncertainty == pytest.approx(0.3)
    assert response.results[0].per_repeat_ddg["repeat_1"] == pytest.approx(-1.1)
    assert response.results[0].converged is True


@pytest.mark.asyncio
async def test_fep_service_runs_wrapper_openfe_registry_chain_with_exact_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from mf_core.proto_gen.moleculeforge.v1.oracle import fep_pb2

    registry = tmp_path / "openfe-result-registry.json"
    registry.write_text(
        json.dumps(
            {
                "7abc": {
                    "CCO>>CCN": {
                        "ligand_a_smiles": "CCO",
                        "ligand_b_smiles": "CCN",
                        "ddg_kcal_mol": -1.2,
                        "ddg_uncertainty": 0.3,
                        "n_repeats": 1,
                        "method": "openfe",
                        "per_repeat_ddg": {"repeat_1": -1.2},
                        "converged": True,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    module = _load_module(
        "fep_wrapper_openfe_registry_chain_test",
        ROOT / "services/fep-svc/src/fep_svc/main.py",
    )
    monkeypatch.setenv(
        "FEP_ORACLE_COMMAND",
        f"{sys.executable} {ROOT / 'tools/oracles/fep_oracle_wrapper.py'}",
    )
    monkeypatch.setenv(
        "OPENFE_RUNNER_PATH",
        f"{sys.executable} {ROOT / 'tools/oracles/openfe_json_runner.py'}",
    )
    monkeypatch.setenv("OPENFE_RESULT_REGISTRY", str(registry))

    response = await module.FEPServicer().RunFEP(
        fep_pb2.FEPBatchRequest(
            project_id="project-1",
            request_id="request-1",
            batch_id="batch-1",
            protein_pdb_id="7abc",
            reference_ligand_smiles="OCC",
            test_ligand_smiles=["NCC"],
            method="openfe",
            n_repeats=1,
        ),
        None,
    )

    assert response.request_id == "request-1"
    assert response.batch_id == "batch-1"
    assert response.results[0].ligand_a_smiles == "OCC"
    assert response.results[0].ligand_b_smiles == "NCC"
    assert response.results[0].ddg_kcal_mol == pytest.approx(-1.2)
    assert response.results[0].per_repeat_ddg == {"repeat_1": pytest.approx(-1.2)}


@pytest.mark.asyncio
async def test_fep_service_submits_background_json_command_job(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from mf_core.proto_gen.moleculeforge.v1.oracle import fep_pb2

    module = _load_module(
        "fep_background_json_command_test",
        ROOT / "services/fep-svc/src/fep_svc/main.py",
    )
    runner = tmp_path / "fep_runner.py"
    runner.write_text(
        "import json, sys, time\n"
        "request = json.load(sys.stdin)\n"
        "assert request['project_id'] == 'project-async'\n"
        "time.sleep(0.05)\n"
        "print(json.dumps({"
        "'request_id': request['request_id'], "
        "'batch_id': request['batch_id'], "
        "'project_id': request['project_id'], "
        "'protein_pdb_id': request['protein_pdb_id'], "
        "'reference_ligand_smiles': request['reference_ligand_smiles'], "
        "'test_ligand_smiles': request['test_ligand_smiles'], "
        "'method': request['method'], "
        "'n_repeats': request['n_repeats'], "
        "'total_elapsed_ms': 44, "
        "'results': [{"
        "'ligand_a_smiles': 'CCO', "
        "'ligand_b_smiles': 'CCN', "
        "'ddg_kcal_mol': -1.4, "
        "'ddg_uncertainty': 0.2, "
        "'n_repeats': 1, "
        "'method': 'openfe', "
        "'per_repeat_ddg': {'repeat_1': -1.4}, "
        "'converged': True"
        "}]"
        "}))\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("FEP_ORACLE_COMMAND", f"{sys.executable} {runner}")
    monkeypatch.setenv("FEP_JOB_DIR", str(tmp_path / "jobs"))
    request = fep_pb2.FEPBatchRequest(
        project_id="project-async",
        request_id="request-async",
        batch_id="batch-async",
        protein_pdb_id="7abc",
        reference_ligand_smiles="CCO",
        test_ligand_smiles=["CCN"],
        method="openfe",
        n_repeats=1,
    )

    service = module.FEPServicer()
    submitted = await service.SubmitFEP(request, None)

    assert submitted.job_id
    assert submitted.request_id == "request-async"
    assert submitted.batch_id == "batch-async"
    for _ in range(50):
        status = await service.GetStatus(
            fep_pb2.FEPJobStatusRequest(job_id=submitted.job_id),
            None,
        )
        if status.state == "completed":
            break
        await asyncio.sleep(0.05)

    assert status.state == "completed"
    assert status.error == ""
    assert status.request_id == "request-async"
    assert status.batch_id == "batch-async"
    assert status.response.request_id == "request-async"
    assert status.response.batch_id == "batch-async"
    assert status.response.total_elapsed_ms == 44
    assert status.response.results[0].ddg_kcal_mol == pytest.approx(-1.4)
    assert status.response.results[0].converged is True
    monkeypatch.delenv("FEP_ORACLE_COMMAND")
    recovered = await service.GetStatus(
        fep_pb2.FEPJobStatusRequest(job_id=submitted.job_id),
        None,
    )
    assert recovered == status


@pytest.mark.asyncio
async def test_fep_service_limits_direct_and_background_execution_together(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from mf_core.proto_gen.moleculeforge.v1.oracle import fep_pb2

    module = _load_module(
        "fep_shared_concurrency_limit_test",
        ROOT / "services/fep-svc/src/fep_svc/main.py",
    )
    monkeypatch.setenv("FEP_ORACLE_COMMAND", "configured")
    monkeypatch.setattr(module, "_require_runtime", lambda: [])
    started = [asyncio.Event(), asyncio.Event()]
    release = [asyncio.Event(), asyncio.Event()]
    calls: list[str] = []

    async def run_command(request):
        index = len(calls)
        calls.append(request.request_id)
        started[index].set()
        await release[index].wait()
        return fep_pb2.FEPBatchResponse(
            request_id=request.request_id,
            batch_id=request.batch_id,
        )

    monkeypatch.setattr(module, "_run_fep_command_async", run_command)
    service = module.FEPServicer(job_dir=tmp_path, max_concurrent_jobs=1)
    direct_request = fep_pb2.FEPBatchRequest(
        project_id="project-direct",
        request_id="request-direct",
        batch_id="batch-direct",
        protein_pdb_id="7abc",
        reference_ligand_smiles="CCO",
        test_ligand_smiles=["CCN"],
        method="openfe",
        n_repeats=1,
    )
    queued_request = fep_pb2.FEPBatchRequest(
        project_id="project-queued",
        request_id="request-queued",
        batch_id="batch-queued",
        protein_pdb_id="7abc",
        reference_ligand_smiles="CCO",
        test_ligand_smiles=["CCC"],
        method="openfe",
        n_repeats=1,
    )

    direct_task = asyncio.create_task(service.RunFEP(direct_request, None))
    await asyncio.wait_for(started[0].wait(), timeout=1)
    submitted = await service.SubmitFEP(queued_request, None)
    background_task = service._tasks[submitted.job_id]
    await asyncio.sleep(0.05)

    assert calls == ["request-direct"]
    assert service._read_job_status(submitted.job_id).state == "queued"

    release[0].set()
    await direct_task
    await asyncio.wait_for(started[1].wait(), timeout=1)
    assert calls == ["request-direct", "request-queued"]
    assert service._read_job_status(submitted.job_id).state == "running"

    release[1].set()
    await background_task
    assert service._read_job_status(submitted.job_id).state == "completed"


@pytest.mark.asyncio
async def test_fep_service_rejects_missing_transport_identity_as_invalid_argument(
) -> None:
    from mf_core.proto_gen.moleculeforge.v1.oracle import fep_pb2

    module = _load_module(
        "fep_missing_transport_identity_test",
        ROOT / "services/fep-svc/src/fep_svc/main.py",
    )
    server = grpc.aio.server()
    module.fep_pb2_grpc.add_FEPServiceServicer_to_server(module.FEPServicer(), server)
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    channel = grpc.aio.insecure_channel(f"127.0.0.1:{port}")
    try:
        with pytest.raises(grpc.aio.AioRpcError) as error:
            await module.fep_pb2_grpc.FEPServiceStub(channel).RunFEP(
                fep_pb2.FEPBatchRequest(
                    project_id="project-1",
                    protein_pdb_id="7ABC",
                    reference_ligand_smiles="CCO",
                    test_ligand_smiles=["CCN"],
                    method="openfe",
                    n_repeats=1,
                )
            )
    finally:
        await channel.close()
        await server.stop(None)

    assert error.value.code() == grpc.StatusCode.INVALID_ARGUMENT


def test_fep_job_status_rejects_mismatched_persisted_job_identity(tmp_path: Path) -> None:
    from google.protobuf.json_format import MessageToJson
    from mf_core.proto_gen.moleculeforge.v1.oracle import fep_pb2

    module = _load_module(
        "fep_job_status_identity_test",
        ROOT / "services/fep-svc/src/fep_svc/main.py",
    )
    service = module.FEPServicer(job_dir=tmp_path)
    (tmp_path / "job-requested.json").write_text(
        MessageToJson(
            fep_pb2.FEPJobStatus(
                job_id="job-other",
                state="queued",
                submitted_at_ms=1,
                request_id="request-1",
                batch_id="batch-1",
            ),
            preserving_proto_field_name=True,
        ),
        encoding="utf-8",
    )

    with pytest.raises(module.OracleDataError, match="job_id"):
        service._read_job_status("job-requested")


def test_fep_service_recovers_interrupted_jobs_as_failed(tmp_path: Path) -> None:
    module = _load_module(
        "fep_job_recovery_test",
        ROOT / "services/fep-svc/src/fep_svc/main.py",
    )
    service = module.FEPServicer(job_dir=tmp_path)
    service._write_job_status(
        module._job_status(
            "queued-job",
            "queued",
            submitted_at_ms=1,
            request_id="request-queued",
            batch_id="batch-queued",
        )
    )
    service._write_job_status(
        module._job_status(
            "running-job",
            "running",
            submitted_at_ms=1,
            started_at_ms=2,
            request_id="request-running",
            batch_id="batch-running",
        )
    )

    recovered = service.recover_interrupted_jobs()

    assert recovered == 2
    for job_id in ("queued-job", "running-job"):
        status = service._read_job_status(job_id)
        assert status.state == "failed"
        assert status.error == "FEP service restarted before job completion"
        assert status.started_at_ms >= status.submitted_at_ms
        assert status.completed_at_ms >= status.started_at_ms


def test_fep_service_quarantines_invalid_jobs_without_blocking_recovery(
    tmp_path: Path,
) -> None:
    from google.protobuf.json_format import MessageToJson
    from mf_core.proto_gen.moleculeforge.v1.oracle import fep_pb2

    module = _load_module(
        "fep_job_quarantine_test",
        ROOT / "services/fep-svc/src/fep_svc/main.py",
    )
    service = module.FEPServicer(job_dir=tmp_path)
    service._write_job_status(
        module._job_status(
            "valid-job",
            "queued",
            submitted_at_ms=1,
            request_id="request-valid",
            batch_id="batch-valid",
        )
    )
    (tmp_path / "legacy-job.json").write_text(
        MessageToJson(
            fep_pb2.FEPJobStatus(
                job_id="legacy-job",
                state="running",
                submitted_at_ms=1,
                started_at_ms=2,
            ),
            preserving_proto_field_name=True,
        ),
        encoding="utf-8",
    )
    (tmp_path / "corrupt.job.json").write_text("{", encoding="utf-8")

    recovered = service.recover_interrupted_jobs()

    assert recovered == 1
    assert service._read_job_status("valid-job").state == "failed"
    assert not (tmp_path / "legacy-job.json").exists()
    assert not (tmp_path / "corrupt.job.json").exists()
    assert len(list(tmp_path.glob("legacy-job.json.invalid*"))) == 1
    assert len(list(tmp_path.glob("corrupt.job.json.invalid*"))) == 1


def test_fep_service_migrates_recoverable_job_identity_from_response(
    tmp_path: Path,
) -> None:
    from google.protobuf.json_format import MessageToJson
    from mf_core.proto_gen.moleculeforge.v1.oracle import fep_pb2

    module = _load_module(
        "fep_job_identity_migration_test",
        ROOT / "services/fep-svc/src/fep_svc/main.py",
    )
    status = fep_pb2.FEPJobStatus(
        job_id="completed-job",
        state="completed",
        submitted_at_ms=1,
        started_at_ms=2,
        completed_at_ms=3,
        response=fep_pb2.FEPBatchResponse(
            request_id="request-completed",
            batch_id="batch-completed",
        ),
    )
    (tmp_path / "completed-job.json").write_text(
        MessageToJson(status, preserving_proto_field_name=True),
        encoding="utf-8",
    )

    service = module.FEPServicer(job_dir=tmp_path)
    recovered = service.recover_interrupted_jobs()

    assert recovered == 0
    migrated = service._read_job_status("completed-job")
    assert migrated.request_id == "request-completed"
    assert migrated.batch_id == "batch-completed"
    assert not list(tmp_path.glob("completed-job.json.invalid*"))


@pytest.mark.asyncio
async def test_fep_async_cancellation_terminates_descendant_processes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from mf_core.proto_gen.moleculeforge.v1.oracle import fep_pb2

    module = _load_module(
        "fep_async_process_group_cancellation_test",
        ROOT / "services/fep-svc/src/fep_svc/main.py",
    )
    marker = tmp_path / "async-descendant-finished"
    runner = tmp_path / "fep_parent.py"
    runner.write_text(
        "import subprocess, sys, time\n"
        "subprocess.Popen([sys.executable, '-c', "
        f"\"import pathlib, signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        f"time.sleep(0.5); pathlib.Path({str(marker)!r}).write_text('done')\""
        "])\n"
        "time.sleep(10)\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("FEP_ORACLE_COMMAND", f"{sys.executable} {runner}")
    request = fep_pb2.FEPBatchRequest(
        project_id="project-1",
        request_id="request-1",
        batch_id="batch-1",
        protein_pdb_id="7abc",
        reference_ligand_smiles="CCO",
        test_ligand_smiles=["CCN"],
        method="openfe",
        n_repeats=1,
    )
    task = asyncio.create_task(module._run_fep_command_async(request))
    await asyncio.sleep(0.1)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    await asyncio.sleep(0.7)
    assert not marker.exists()


@pytest.mark.asyncio
async def test_fep_builtin_chain_cancellation_terminates_openfe_cli(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from mf_core.proto_gen.moleculeforge.v1.oracle import fep_pb2

    module = _load_module(
        "fep_builtin_chain_process_group_cancellation_test",
        ROOT / "services/fep-svc/src/fep_svc/main.py",
    )
    started = tmp_path / "openfe-started"
    finished = tmp_path / "openfe-finished"
    openfe = tmp_path / "openfe"
    openfe.write_text(
        "#!/usr/bin/env python3\n"
        "import signal, time\n"
        "from pathlib import Path\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        f"Path({str(started)!r}).write_text('started')\n"
        "time.sleep(0.5)\n"
        f"Path({str(finished)!r}).write_text('finished')\n",
        encoding="utf-8",
    )
    openfe.chmod(0o755)
    complex_transformation = tmp_path / "complex.json"
    solvent_transformation = tmp_path / "solvent.json"
    _write_openfe_transformation(complex_transformation)
    _write_openfe_transformation(solvent_transformation)
    registry = tmp_path / "transformation-registry.json"
    registry.write_text(
        json.dumps(
            {
                "7abc": {
                    "CCO>>CCN": {
                        "complex": str(complex_transformation),
                        "solvent": str(solvent_transformation),
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv(
        "FEP_ORACLE_COMMAND",
        f"{sys.executable} {ROOT / 'tools/oracles/fep_oracle_wrapper.py'}",
    )
    monkeypatch.setenv(
        "OPENFE_RUNNER_PATH",
        f"{sys.executable} {ROOT / 'tools/oracles/openfe_json_runner.py'}",
    )
    monkeypatch.setenv("OPENFE_CLI_PATH", str(openfe))
    monkeypatch.setenv("OPENFE_TRANSFORMATION_REGISTRY", str(registry))
    monkeypatch.setenv("OPENFE_WORK_DIR", str(tmp_path / "work"))
    request = fep_pb2.FEPBatchRequest(
        project_id="project-1",
        request_id="request-1",
        batch_id="batch-1",
        protein_pdb_id="7abc",
        reference_ligand_smiles="CCO",
        test_ligand_smiles=["CCN"],
        method="openfe",
        n_repeats=1,
    )
    task = asyncio.create_task(module._run_fep_command_async(request))
    for _ in range(300):
        if started.exists():
            break
        if task.done():
            await task
        await asyncio.sleep(0.01)
    assert started.exists()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    await asyncio.sleep(0.7)
    assert not finished.exists()
    assert not list((tmp_path / "work").glob("mforge-openfe-*"))


def test_fep_startup_removes_only_stale_openfe_request_work_directories(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_module(
        "fep_stale_work_cleanup_test",
        ROOT / "services/fep-svc/src/fep_svc/main.py",
    )
    work_base = tmp_path / "work"
    stale = work_base / "mforge-openfe-stale"
    unrelated = work_base / "operator-data"
    stale.mkdir(parents=True)
    unrelated.mkdir()
    (stale / "result.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("OPENFE_WORK_DIR", str(work_base))

    removed = module._cleanup_stale_openfe_work_directories()

    assert removed == 1
    assert not stale.exists()
    assert unrelated.is_dir()


@pytest.mark.asyncio
async def test_fep_async_timeout_terminates_descendant_processes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from mf_core.proto_gen.moleculeforge.v1.oracle import fep_pb2

    module = _load_module(
        "fep_async_process_group_timeout_test",
        ROOT / "services/fep-svc/src/fep_svc/main.py",
    )
    marker = tmp_path / "async-timeout-descendant-finished"
    runner = tmp_path / "fep_timeout_parent.py"
    runner.write_text(
        "import subprocess, sys, time\n"
        "subprocess.Popen([sys.executable, '-c', "
        f"\"import pathlib, signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        f"time.sleep(0.5); pathlib.Path({str(marker)!r}).write_text('done')\""
        "])\n"
        "time.sleep(10)\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("FEP_ORACLE_COMMAND", f"{sys.executable} {runner}")
    monkeypatch.setenv("FEP_ORACLE_TIMEOUT_SECONDS", "0.1")
    request = fep_pb2.FEPBatchRequest(
        project_id="project-1",
        request_id="request-1",
        batch_id="batch-1",
        protein_pdb_id="7abc",
        reference_ligand_smiles="CCO",
        test_ligand_smiles=["CCN"],
        method="openfe",
        n_repeats=1,
    )

    with pytest.raises(subprocess.TimeoutExpired):
        await module._run_fep_command_async(request)

    await asyncio.sleep(0.7)
    assert not marker.exists()


@pytest.mark.asyncio
async def test_fep_oracle_service_maps_evaluations_to_rbfe_scores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mf_core.proto_gen.moleculeforge.v1.oracle import fep_pb2, oracle_pb2

    module = _load_module(
        "fep_oracle_adapter_test",
        ROOT / "services/fep-svc/src/fep_svc/main.py",
    )

    class FEPService:
        def __init__(self) -> None:
            self.requests = []

        async def RunFEP(self, request, context):
            self.requests.append(request)
            return fep_pb2.FEPBatchResponse(
                results=[
                    fep_pb2.FEPResult(
                        ligand_a_smiles=request.reference_ligand_smiles,
                        ligand_b_smiles=request.test_ligand_smiles[0],
                        ddg_kcal_mol=-1.2,
                        ddg_uncertainty=0.3,
                        n_repeats=request.n_repeats,
                        method=request.method,
                        per_repeat_ddg={
                            f"repeat_{index}": -1.2 for index in range(1, request.n_repeats + 1)
                        },
                        converged=True,
                    )
                ],
                request_id=request.request_id,
                batch_id=request.batch_id,
                total_elapsed_ms=33,
                project_id=request.project_id,
                protein_pdb_id=request.protein_pdb_id,
                reference_ligand_smiles=request.reference_ligand_smiles,
                test_ligand_smiles=request.test_ligand_smiles,
                method=request.method,
                n_repeats=request.n_repeats,
            )

    service = FEPService()
    oracle = module.FEPOracleServicer(service=service)

    response = await oracle.PredictWithUncertainty(
        oracle_pb2.OracleBatchRequest(
            project_id="project-1",
            request_id="request-1",
            molecule_smiles=["CCN"],
            requested_properties=["rbfe"],
            level=oracle_pb2.L3_FEP,
            return_uncertainty=True,
            protein_pdb_id="7ABC",
            reference_ligand_smiles="CCO",
            oracle_parameters={"method": "openfe", "n_repeats": "1"},
        ),
        None,
    )

    assert service.requests[0].reference_ligand_smiles == "CCO"
    assert list(service.requests[0].test_ligand_smiles) == ["CCN"]
    assert response.batch_id == "request-1"
    assert response.total_elapsed_ms == 33
    assert response.evaluations[0].oracle_name == "openfe"
    assert response.evaluations[0].molecule_smiles == "CCN"
    assert response.evaluations[0].level == oracle_pb2.L3_FEP
    assert response.evaluations[0].scores == {"rbfe": -1.2}
    assert response.evaluations[0].uncertainties == {"rbfe": 0.3}
    assert response.evaluations[0].success is True


def test_iclm_service_builds_local_transformers_runner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("ICLM_MODEL_PATH", str(tmp_path))
    module = _load_module(
        "iclm_runner_build_test",
        ROOT / "services/iclm-svc/src/iclm_svc/main.py",
    )

    generator = module._build_generator()

    assert generator.checkpoint_path == str(tmp_path)
    assert generator.runner.model_path == str(tmp_path)


@pytest.mark.asyncio
async def test_iclm_serve_rejects_missing_internal_service_token(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_module(
        "iclm_missing_service_token_test",
        ROOT / "services/iclm-svc/src/iclm_svc/main.py",
    )
    monkeypatch.setenv("ICLM_STATE_PATH", str(tmp_path / "state.json"))
    monkeypatch.delenv("INTERNAL_SERVICE_TOKEN", raising=False)

    with pytest.raises(RuntimeError, match="INTERNAL_SERVICE_TOKEN is required"):
        await module.serve()


def test_iclm_deployment_wires_model_and_update_runner_env() -> None:
    import yaml

    compose = (ROOT / "infra/docker/docker-compose.dev.yml").read_text(encoding="utf-8")
    compose_config = yaml.safe_load(compose)
    generator_dockerfile = (ROOT / "infra/docker/base/Dockerfile.generator").read_text(
        encoding="utf-8"
    )
    image_build_script = (ROOT / "infra/scripts/build_all_images.sh").read_text(
        encoding="utf-8"
    )
    k8s = (ROOT / "infra/kubernetes/deployments/moleculeforge-services.yaml").read_text(
        encoding="utf-8"
    )
    k8s_docs = list(yaml.safe_load_all(k8s))
    helm_values = (ROOT / "infra/helm/moleculeforge/values.yaml").read_text(encoding="utf-8")
    helm_config = yaml.safe_load(helm_values)

    for env_name in (
        "ICLM_MODEL_PATH",
        "ICLM_ALLOW_VALIDATION_MODEL",
        "ICLM_DEVICE",
        "ICLM_UPDATE_COMMAND",
        "ICLM_UPDATE_TIMEOUT_SECONDS",
        "ICLM_STATE_PATH",
        "ICLM_CHECKPOINT_DIRECTORY",
    ):
        assert env_name in compose
        assert env_name in k8s
        assert env_name in helm_values

    assert "ICLM_DEVICE: ${ICLM_DEVICE:-cpu}" in compose
    assert (
        "ICLM_MODEL_PATH: ${ICLM_MODEL_PATH:-/var/lib/moleculeforge/iclm/model}"
    ) in compose
    assert "ICLM_ALLOW_VALIDATION_MODEL: ${ICLM_ALLOW_VALIDATION_MODEL:-true}" in compose
    assert "ICLM_UPDATE_TIMEOUT_SECONDS: ${ICLM_UPDATE_TIMEOUT_SECONDS:-300}" in compose
    assert "name: iclm-generator-config" in k8s
    assert "configMapKeyRef:" in k8s
    assert "envValueFrom:" in helm_values
    for config in (
        _k8s_configmap_data(k8s, "mf-generators", "iclm-generator-config"),
        _helm_configmap_data(helm_values, "mf-generators", "iclm-generator-config"),
    ):
        assert config["model-path"] == "/var/lib/moleculeforge/iclm/model"
        assert config["allow-validation-model"] == "false"
        assert config["device"] == "cpu"
        assert config["update-command"] == ""
        assert config["update-timeout-seconds"] == "300"
        assert config["state-path"] == "/var/lib/moleculeforge/iclm/state.json"
        assert config["checkpoint-directory"] == (
            "/var/lib/moleculeforge/iclm/checkpoints"
        )

    compose_iclm = compose_config["services"]["iclm-svc"]
    assert compose_iclm["image"] == "moleculeforge/generator:dev"
    assert "--bootstrap-validation-model" in compose_iclm["command"][2]
    assert compose_iclm["environment"]["INTERNAL_SERVICE_TOKEN"] == (
        "${INTERNAL_SERVICE_TOKEN:-mf_dev_internal_service_token}"
    )
    assert any(
        str(volume).endswith(":/var/lib/moleculeforge/iclm")
        for volume in compose_iclm["volumes"]
    )
    assert "iclm_state_data" in compose_config["volumes"]

    claims = {
        (document["metadata"]["namespace"], document["metadata"]["name"])
        for document in k8s_docs
        if document and document.get("kind") == "PersistentVolumeClaim"
    }
    assert ("mf-generators", "iclm-state") in claims
    iclm_deployment = next(
        document
        for document in k8s_docs
        if document
        and document.get("kind") == "Deployment"
        and document["metadata"]["name"] == "iclm-svc"
    )
    iclm_pod_spec = iclm_deployment["spec"]["template"]["spec"]
    iclm_container = iclm_pod_spec["containers"][0]
    assert iclm_container["image"] == "moleculeforge/generator:latest"
    assert iclm_container["command"] == ["python", "-m", "iclm_svc.main"]
    iclm_env = {item["name"]: item for item in iclm_container["env"]}
    assert iclm_env["INTERNAL_SERVICE_TOKEN"]["valueFrom"]["secretKeyRef"] == {
        "name": "generator-runtime-secrets",
        "key": "INTERNAL_SERVICE_TOKEN",
    }
    generator_secret = next(
        document
        for document in k8s_docs
        if document
        and document.get("kind") == "Secret"
        and document["metadata"]["namespace"] == "mf-generators"
        and document["metadata"]["name"] == "generator-runtime-secrets"
    )
    assert set(generator_secret["stringData"]) == {"INTERNAL_SERVICE_TOKEN"}
    assert generator_secret["stringData"]["INTERNAL_SERVICE_TOKEN"] == ""
    assert {mount["mountPath"] for mount in iclm_container["volumeMounts"]} == {
        "/var/lib/moleculeforge/iclm"
    }
    assert iclm_pod_spec["volumes"][0]["persistentVolumeClaim"]["claimName"] == (
        "iclm-state"
    )

    assert "iclm-state" in helm_config["persistentVolumeClaims"]
    helm_iclm = helm_config["services"]["iclm-svc"]
    assert helm_iclm["image"] == {"repository": "generator"}
    assert helm_iclm["command"] == ["python", "-m", "iclm_svc.main"]
    assert helm_iclm["envValueFrom"]["INTERNAL_SERVICE_TOKEN"][
        "secretKeyRef"
    ] == {
        "name": "generator-runtime-secrets",
        "key": "internal-service-token",
    }
    assert helm_config["secrets"]["generator-runtime-secrets"] == {
        "name": "generator-runtime-secrets",
        "namespace": "mf-generators",
        "stringData": {"internal-service-token": ""},
    }
    assert helm_iclm["volumeMounts"] == [
        {"name": "iclm-state", "mountPath": "/var/lib/moleculeforge/iclm"}
    ]
    assert helm_iclm["volumes"][0]["persistentVolumeClaim"]["claimName"] == (
        "iclm-state"
    )
    assert "--extra generator-runtime" in generator_dockerfile.replace("\\\n", " ")
    assert "COPY services ./services" in generator_dockerfile
    for module_name in (
        "mf_generators.crem_3d.generator",
        "mf_generators.fragfm.generator",
        "mf_generators.hfm_3d.generator",
        "mf_generators.mmpt_rag.generator",
        "mf_generators.uas.generator",
    ):
        assert module_name in generator_dockerfile
    assert '"generator:infra/docker/base/Dockerfile.generator"' in image_build_script

    assert compose_config["services"]["generator-coord-agent"]["environment"][
        "ICLM_MODEL_UPDATE_TIMEOUT_SECONDS"
    ] == "${ICLM_MODEL_UPDATE_TIMEOUT_SECONDS:-330}"
    compose_generator_coord = compose_config["services"]["generator-coord-agent"]
    assert compose_generator_coord["environment"]["GENERATOR_COORD_STATE_PATH"] == (
        "/var/lib/moleculeforge/generator-coord/state.json"
    )
    assert compose_generator_coord["volumes"] == [
        "generator_coord_state_data:/var/lib/moleculeforge/generator-coord"
    ]
    assert "generator_coord_state_data" in compose_config["volumes"]
    assert ("mf-agents", "generator-coord-state") in claims
    generator_coord = next(
        document
        for document in k8s_docs
        if document
        and document.get("kind") == "Deployment"
        and document["metadata"]["name"] == "generator-coord-agent"
    )
    generator_coord_env = {
        item["name"]: item.get("value")
        for item in generator_coord["spec"]["template"]["spec"]["containers"][0]["env"]
    }
    assert generator_coord_env["ICLM_MODEL_UPDATE_TIMEOUT_SECONDS"] == "330"
    assert generator_coord_env["GENERATOR_COORD_STATE_PATH"] == (
        "/var/lib/moleculeforge/generator-coord/state.json"
    )
    assert generator_coord["spec"]["strategy"] == {"type": "Recreate"}
    generator_coord_pod = generator_coord["spec"]["template"]["spec"]
    assert generator_coord_pod["containers"][0]["volumeMounts"] == [
        {
            "name": "generator-coord-state",
            "mountPath": "/var/lib/moleculeforge/generator-coord",
        }
    ]
    assert generator_coord_pod["volumes"][0]["persistentVolumeClaim"] == {
        "claimName": "generator-coord-state"
    }
    assert "generator-coord-state" in helm_config["persistentVolumeClaims"]
    helm_generator_coord = helm_config["services"]["generator-coord-agent"]
    assert helm_generator_coord["env"][
        "ICLM_MODEL_UPDATE_TIMEOUT_SECONDS"
    ] == "330"
    assert helm_generator_coord["env"]["GENERATOR_COORD_STATE_PATH"] == (
        "/var/lib/moleculeforge/generator-coord/state.json"
    )
    assert helm_generator_coord["strategy"] == {"type": "Recreate"}
    assert helm_generator_coord["volumeMounts"] == [
        {
            "name": "generator-coord-state",
            "mountPath": "/var/lib/moleculeforge/generator-coord",
        }
    ]
    assert helm_generator_coord["volumes"][0]["persistentVolumeClaim"] == {
        "claimName": "generator-coord-state"
    }


def test_iclm_runtime_rejects_missing_update_command(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "iclm_model"
    model_path.mkdir()
    monkeypatch.setenv("ICLM_MODEL_PATH", str(model_path))
    monkeypatch.setenv("ICLM_UPDATE_COMMAND", "missing-iclm-update --json")
    module = _load_module(
        "iclm_runtime_missing_update_command_test",
        ROOT / "services/iclm-svc/src/iclm_svc/main.py",
    )

    status = module.runtime_status()

    command_status = next(item for item in status if item["name"] == "iclm_update_command")
    assert command_status["configured"] is True
    assert command_status["available"] is False
    assert command_status["source"] == "ICLM_UPDATE_COMMAND"
    assert "not found" in command_status["message"]


def test_iclm_runtime_requires_valid_ewc_baseline_for_builtin_update(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "iclm_model"
    model_path.mkdir()
    monkeypatch.setenv("ICLM_MODEL_PATH", str(model_path))
    monkeypatch.delenv("ICLM_UPDATE_COMMAND", raising=False)
    module = _load_module(
        "iclm_runtime_ewc_baseline_test",
        ROOT / "services/iclm-svc/src/iclm_svc/main.py",
    )

    missing_status = {item["name"]: item for item in module.runtime_status()}

    assert missing_status["iclm_ewc_baseline"]["available"] is False
    assert missing_status["iclm_ewc_baseline"]["required"] is True

    replay_path = model_path / "moleculeforge_ewc_replay.json"
    replay_path.write_text(
        json.dumps(
            {
                "schema_version": "iclm-ewc-replay.v1",
                "dataset_id": "runtime-calibration-v1",
                "samples": [{"smiles": "CCO", "weight": 1.0}],
            }
        ),
        encoding="utf-8",
    )

    ready_status = {item["name"]: item for item in module.runtime_status()}

    assert ready_status["iclm_ewc_baseline"]["available"] is True
    assert ready_status["iclm_ewc_baseline"]["path"] == str(replay_path)

    replay_path.write_text("{}", encoding="utf-8")
    invalid_status = {item["name"]: item for item in module.runtime_status()}

    assert invalid_status["iclm_ewc_baseline"]["available"] is False
    assert "schema is invalid" in invalid_status["iclm_ewc_baseline"]["message"]


def test_iclm_validation_model_requires_explicit_opt_in_and_valid_continual_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from mf_generators.incremental_clm.hf_runner import (
        bootstrap_validation_checkpoint,
    )

    model_path = bootstrap_validation_checkpoint(tmp_path / "validation-model")
    monkeypatch.setenv("ICLM_MODEL_PATH", str(model_path))
    monkeypatch.delenv("ICLM_UPDATE_COMMAND", raising=False)
    monkeypatch.delenv("ICLM_ALLOW_VALIDATION_MODEL", raising=False)
    module = _load_module(
        "iclm_validation_model_readiness_test",
        ROOT / "services/iclm-svc/src/iclm_svc/main.py",
    )

    blocked = {item["name"]: item for item in module.runtime_status()}

    assert blocked["iclm_validation_model_opt_in"]["available"] is False
    assert "ICLM_ALLOW_VALIDATION_MODEL=true" in blocked[
        "iclm_validation_model_opt_in"
    ]["message"]

    monkeypatch.setenv("ICLM_ALLOW_VALIDATION_MODEL", "true")
    ready = {item["name"]: item for item in module.runtime_status()}

    assert ready["iclm_validation_model_opt_in"]["available"] is True
    assert ready["iclm_ewc_baseline"]["available"] is True

    (model_path / "moleculeforge_continual_state.pt").write_bytes(b"corrupt")
    corrupt = {item["name"]: item for item in module.runtime_status()}

    assert corrupt["iclm_ewc_baseline"]["available"] is False
    assert "loadable" in corrupt["iclm_ewc_baseline"]["message"]


@pytest.mark.asyncio
async def test_iclm_validation_model_runs_generate_update_generate_service_flow(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from mf_generators.incremental_clm.hf_runner import (
        bootstrap_validation_checkpoint,
    )

    model_path = bootstrap_validation_checkpoint(tmp_path / "validation-model")
    checkpoint_directory = tmp_path / "checkpoints"
    state_path = tmp_path / "state.json"
    monkeypatch.setenv("ICLM_MODEL_PATH", str(model_path))
    monkeypatch.setenv("ICLM_ALLOW_VALIDATION_MODEL", "true")
    monkeypatch.setenv("ICLM_CHECKPOINT_DIRECTORY", str(checkpoint_directory))
    monkeypatch.setenv("ICLM_STATE_PATH", str(state_path))
    monkeypatch.delenv("ICLM_UPDATE_COMMAND", raising=False)
    module = _load_module(
        "iclm_validation_model_service_flow_test",
        ROOT / "services/iclm-svc/src/iclm_svc/main.py",
    )
    service = module.ICLMServicer(state_path=state_path)
    generate_request = _valid_generator_request(
        batch_size=8,
        generator_params={"sampling_seed": "0"},
    )

    initial_response = await service.Generate(generate_request, None)
    update_response = await service.UpdateModel(
        _valid_model_update_request(
            samples=[
                {"smiles": "CCO", "reward": 1.0, "outcome": "PASS"},
                {"smiles": "CCN", "reward": 0.0, "outcome": "FAIL"},
            ],
            teacher_embeddings=[[0.0] * 8, [0.0] * 8],
            kd_weight=0.5,
            target_version="validation-update",
        ),
        None,
    )
    updated_response = await service.Generate(generate_request, None)

    assert len(initial_response.molecules) == 8
    assert update_response.acknowledged is True
    assert update_response.status == generator_pb2.MODEL_UPDATE_STATUS_APPLIED
    assert update_response.active_version == "validation-update"
    assert len(updated_response.molecules) == 8
    assert (
        checkpoint_directory
        / "validation-update"
        / "moleculeforge_validation_model.json"
    ).is_file()
    assert state_path.is_file()


@pytest.mark.asyncio
async def test_iclm_update_command_preflight_rejects_missing_executable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "iclm_model"
    model_path.mkdir()
    monkeypatch.setenv("ICLM_MODEL_PATH", str(model_path))
    monkeypatch.setenv("ICLM_UPDATE_COMMAND", "missing-iclm-update-runner --json")
    module = _load_module(
        "iclm_update_missing_command_preflight_test",
        ROOT / "services/iclm-svc/src/iclm_svc/main.py",
    )

    with pytest.raises(RuntimeError, match="not found"):
        await module._run_update_command(
            module._validate_model_update_request(
                _valid_model_update_request(
                    samples=[{"smiles": "CCO"}],
                    teacher_embeddings=[[0.1]],
                    kd_weight=0.5,
                )
            )
        )


def test_generator_coord_deployment_wires_hypseek_teacher_env() -> None:
    import yaml

    compose = (ROOT / "infra/docker/docker-compose.dev.yml").read_text(encoding="utf-8")
    compose_config = yaml.safe_load(compose)
    k8s = (ROOT / "infra/kubernetes/deployments/moleculeforge-services.yaml").read_text(
        encoding="utf-8"
    )
    k8s_docs = list(yaml.safe_load_all(k8s))
    helm_values = (ROOT / "infra/helm/moleculeforge/values.yaml").read_text(encoding="utf-8")
    helm_config = yaml.safe_load(helm_values)
    helm_template = (ROOT / "infra/helm/moleculeforge/templates/services.yaml").read_text(
        encoding="utf-8"
    )

    expected_url = "http://hypseek-teacher-svc:8012/teacher"
    compose_router_env = compose_config["services"]["generator-router-svc"]["environment"]
    compose_teacher_env = compose_config["services"]["hypseek-teacher-svc"]["environment"]
    compose_generator_coord_env = compose_config["services"]["generator-coord-agent"]["environment"]
    compose_orchestrator_env = compose_config["services"]["orchestrator-svc"]["environment"]
    assert compose_generator_coord_env["HYPSEEK_TEACHER_URL"] == expected_url
    assert "HYPSEEK_TEACHER_TIMEOUT_SECONDS" in compose_generator_coord_env
    assert not {
        "HYPSEEK_TEACHER_URL",
        "HYPSEEK_TEACHER_COMMAND",
        "HYPSEEK_TEACHER_TIMEOUT_SECONDS",
    } & set(compose_router_env)
    assert not {
        "HYPSEEK_TEACHER_URL",
        "HYPSEEK_TEACHER_COMMAND",
        "HYPSEEK_TEACHER_TIMEOUT_SECONDS",
    } & set(compose_orchestrator_env)
    assert {
        "HYPSEEK_TEACHER_SOURCE",
        "HYPSEEK_TEACHER_VERSION",
        "HYPSEEK_TEACHER_COMMAND",
        "HYPSEEK_TEACHER_TIMEOUT_SECONDS",
    } <= set(compose_teacher_env)

    deployments = {
        item["metadata"]["name"]: item
        for item in k8s_docs
        if item and item.get("kind") == "Deployment"
    }

    def deployment_env(name: str) -> dict[str, dict]:
        container = deployments[name]["spec"]["template"]["spec"]["containers"][0]
        return {item["name"]: item for item in container.get("env", [])}

    router_env = deployment_env("generator-router-svc")
    teacher_env = deployment_env("hypseek-teacher-svc")
    generator_coord_env = deployment_env("generator-coord-agent")
    orchestrator_env = deployment_env("orchestrator-svc")
    assert generator_coord_env["HYPSEEK_TEACHER_URL"]["value"] == expected_url
    assert "HYPSEEK_TEACHER_TIMEOUT_SECONDS" in generator_coord_env
    assert not {
        "HYPSEEK_TEACHER_URL",
        "HYPSEEK_TEACHER_COMMAND",
        "HYPSEEK_TEACHER_TIMEOUT_SECONDS",
    } & set(router_env)
    assert not {
        "HYPSEEK_TEACHER_URL",
        "HYPSEEK_TEACHER_COMMAND",
        "HYPSEEK_TEACHER_TIMEOUT_SECONDS",
    } & set(orchestrator_env)
    assert {
        "HYPSEEK_TEACHER_SOURCE",
        "HYPSEEK_TEACHER_VERSION",
        "HYPSEEK_TEACHER_COMMAND",
        "HYPSEEK_TEACHER_TIMEOUT_SECONDS",
    } <= set(teacher_env)

    helm_router = helm_config["services"]["generator-router-svc"]
    helm_teacher = helm_config["services"]["hypseek-teacher-svc"]
    helm_generator_coord = helm_config["services"]["generator-coord-agent"]
    helm_orchestrator = helm_config["services"]["orchestrator-svc"]
    assert helm_generator_coord["env"]["HYPSEEK_TEACHER_URL"] == expected_url
    assert "HYPSEEK_TEACHER_TIMEOUT_SECONDS" in helm_generator_coord["env"]
    assert "HYPSEEK_TEACHER_URL" not in helm_router["env"]
    assert "HYPSEEK_TEACHER_COMMAND" not in helm_router.get("envValueFrom", {})
    assert "HYPSEEK_TEACHER_URL" not in helm_orchestrator["env"]
    assert "HYPSEEK_TEACHER_COMMAND" not in helm_orchestrator.get("envValueFrom", {})
    assert {
        "HYPSEEK_TEACHER_SOURCE",
        "HYPSEEK_TEACHER_VERSION",
        "HYPSEEK_TEACHER_COMMAND",
        "HYPSEEK_TEACHER_TIMEOUT_SECONDS",
    } <= set(helm_teacher["envValueFrom"])
    assert ".Values.persistentVolumeClaims" in helm_template
    assert "$service.strategy" in helm_template
    assert "$service.volumeMounts" in helm_template
    assert "$service.volumes" in helm_template

    compose_healthcheck = compose_config["services"]["hypseek-teacher-svc"]["healthcheck"]
    assert "http://localhost:8012/healthz" in " ".join(compose_healthcheck["test"])
    hypseek_deployment = deployments["hypseek-teacher-svc"]
    container = hypseek_deployment["spec"]["template"]["spec"]["containers"][0]
    assert container["readinessProbe"]["httpGet"] == {"path": "/healthz", "port": "http"}
    assert container["livenessProbe"]["httpGet"] == {"path": "/healthz", "port": "http"}
    hypseek_values = helm_config["services"]["hypseek-teacher-svc"]
    assert hypseek_values["readinessProbe"]["httpGet"] == {
        "path": "/healthz",
        "port": "http",
    }
    assert hypseek_values["livenessProbe"]["httpGet"] == {
        "path": "/healthz",
        "port": "http",
    }
    assert "readinessProbe" in helm_template
    assert "livenessProbe" in helm_template


@pytest.mark.asyncio
async def test_iclm_service_update_model_runs_configured_json_command(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_module(
        "iclm_update_command_test",
        ROOT / "services/iclm-svc/src/iclm_svc/main.py",
    )
    model_path = tmp_path / "iclm_model"
    model_path.mkdir()
    runner = tmp_path / "iclm_update_runner.py"
    runner.write_text(
        "import json, sys\n"
        "from pathlib import Path\n"
        "payload = json.load(sys.stdin)\n"
        "assert payload['model_path'].endswith('iclm_model')\n"
        "assert payload['device'] == 'cpu'\n"
        "assert payload['run_id'] == 'run-iclm'\n"
        "assert [item['smiles'] for item in payload['samples']] == ['CCO', 'CCN']\n"
        "assert len(payload['teacher_embeddings']) == 2\n"
        "assert len(payload['teacher_embeddings'][0]) == 4\n"
        "assert abs(payload['teacher_embeddings'][0][0] - 0.1) < 1e-6\n"
        "assert payload['teacher_weight'] == 0.25\n"
        "checkpoint = Path(payload['model_path']) / 'updated'\n"
        "checkpoint.write_text('updated checkpoint')\n"
        "print(json.dumps({"
        "'checkpoint_path': str(checkpoint), "
        "'updated_samples': 2"
        "}))\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("ICLM_MODEL_PATH", str(model_path))
    monkeypatch.setenv("ICLM_UPDATE_COMMAND", f"{sys.executable} {runner}")

    activated_generator = _ICLMRecordingGenerator(model_path / "updated")
    response = await module.ICLMServicer(
        generator=SimpleNamespace(checkpoint_path=str(model_path)),
        generator_factory=lambda checkpoint_path: activated_generator,
    ).UpdateModel(
        _valid_model_update_request(
            samples=[{"smiles": "CCO"}, {"smiles": "CCN"}],
            teacher_embeddings=[[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]],
            kd_weight=0.25,
        ),
        None,
    )

    assert response.acknowledged is True
    assert response.active_version == "iclm-v2"
    assert response.updated_samples == 2
    assert response.artifacts[0].checksum.startswith("sha256:")
    assert response.artifacts[1].version == "teacher-v1"


@pytest.mark.asyncio
async def test_iclm_service_update_model_uses_injected_online_learner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_module(
        "iclm_update_online_learner_test",
        ROOT / "services/iclm-svc/src/iclm_svc/main.py",
    )
    model_path = tmp_path / "iclm_model"
    model_path.mkdir()
    monkeypatch.setenv("ICLM_MODEL_PATH", str(model_path))
    monkeypatch.delenv("ICLM_UPDATE_COMMAND", raising=False)
    updated_checkpoint = model_path / "online-updated"
    updated_checkpoint.write_text("updated", encoding="utf-8")
    calls: list[dict] = []

    class Learner:
        def update(self, payload):
            calls.append(payload)
            return {
                "checkpoint_path": str(updated_checkpoint),
                "updated_samples": 2,
            }

    training_generator = SimpleNamespace(
        checkpoint_path=str(model_path),
        online_learner=Learner(),
    )
    activated_generator = _ICLMRecordingGenerator(updated_checkpoint)
    factory_results = iter((training_generator, activated_generator))
    response = await module.ICLMServicer(
        generator=SimpleNamespace(checkpoint_path=str(model_path)),
        generator_factory=lambda checkpoint_path: next(factory_results),
    ).UpdateModel(
        _valid_model_update_request(
            samples=[{"smiles": "CCO"}, {"smiles": "CCN"}],
            teacher_embeddings=[[0.1, 0.2], [0.3, 0.4]],
            kd_weight=0.5,
        ),
        None,
    )

    assert calls == [
        {
            "schema_version": "training-batch.v1",
            "samples": [
                {"candidate_id": "candidate-1", "reward": 1.0, "smiles": "CCO"},
                {"candidate_id": "candidate-2", "reward": 1.0, "smiles": "CCN"},
            ],
            "teacher_weight": 0.5,
            "run_id": "run-iclm",
            "request_id": "update-iclm",
            "teacher_embeddings": [
                [pytest.approx(0.1), pytest.approx(0.2)],
                [pytest.approx(0.3), pytest.approx(0.4)],
            ],
            "teacher_source": "hypseek",
            "teacher_version": "teacher-v1",
            "target_checkpoint_version": "iclm-v2",
        }
    ]
    assert response.acknowledged is True
    assert response.active_version == "iclm-v2"
    assert response.updated_samples == 2


@pytest.mark.asyncio
async def test_iclm_service_rejects_legacy_online_learner_path(
    tmp_path: Path,
) -> None:
    import torch
    from mf_generators.incremental_clm.learning.online_learner import OnlineLearner

    module = _load_module(
        "iclm_legacy_online_learner_rejection_test",
        ROOT / "services/iclm-svc/src/iclm_svc/main.py",
    )
    learner = OnlineLearner(
        torch.nn.Linear(1, 1, bias=False),
        checkpoint_directory=tmp_path,
    )

    with pytest.raises(module.UpdateRunnerUnavailable, match="HuggingFaceCausalLMRunner"):
        await module._run_update(
            {"samples": [{"smiles": "CCO"}]},
            SimpleNamespace(online_learner=learner),
        )


@pytest.mark.asyncio
async def test_iclm_service_update_model_accepts_online_learner_metrics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_module(
        "iclm_update_online_learner_metric_test",
        ROOT / "services/iclm-svc/src/iclm_svc/main.py",
    )
    model_path = tmp_path / "iclm_model"
    model_path.mkdir()
    monkeypatch.setenv("ICLM_MODEL_PATH", str(model_path))
    monkeypatch.delenv("ICLM_UPDATE_COMMAND", raising=False)
    updated_checkpoint = tmp_path / "iclm_model_updated"
    updated_checkpoint.write_text("updated", encoding="utf-8")

    class Learner:
        last_task_loss = 0.25
        last_teacher_loss = 4.0

        def update(self, payload):
            assert payload["teacher_weight"] == 0.5
            return {
                "checkpoint_path": str(updated_checkpoint),
                "updated_samples": 1,
                "ewc_loss": 0.25,
                "teacher_loss": 4.0,
            }

    training_generator = SimpleNamespace(
        checkpoint_path=str(model_path),
        online_learner=Learner(),
    )
    activated_generator = _ICLMRecordingGenerator(updated_checkpoint)
    factory_results = iter((training_generator, activated_generator))
    response = await module.ICLMServicer(
        generator=SimpleNamespace(checkpoint_path=str(model_path)),
        generator_factory=lambda checkpoint_path: next(factory_results),
    ).UpdateModel(
        _valid_model_update_request(
            samples=[{"smiles": "CCO"}],
            teacher_embeddings=[[0.0]],
            kd_weight=0.5,
        ),
        None,
    )

    assert response.acknowledged is True
    assert response.active_version == "iclm-v2"
    assert response.updated_samples == 1


def test_iclm_update_rejects_unsafe_target_checkpoint_version() -> None:
    module = _load_module(
        "iclm_unsafe_target_version_test",
        ROOT / "services/iclm-svc/src/iclm_svc/main.py",
    )
    request = _valid_model_update_request(
        samples=[{"smiles": "CCO"}],
        teacher_embeddings=[[0.0]],
        kd_weight=0.5,
        target_version="../escape",
    )

    with pytest.raises(
        module.ModelUpdateRequestError,
        match="target_checkpoint_version must be a file-safe name",
    ):
        module._validate_model_update_request(request)


@pytest.mark.parametrize(
    "field",
    ["run_id", "request_id", "teacher_source", "teacher_version", "target_checkpoint_version"],
)
def test_iclm_update_rejects_whitespace_padded_identity(field: str) -> None:
    module = _load_module(
        f"iclm_whitespace_identity_{field}_test",
        ROOT / "services/iclm-svc/src/iclm_svc/main.py",
    )
    request = _valid_model_update_request(
        samples=[{"smiles": "CCO"}],
        teacher_embeddings=[[0.0]],
        kd_weight=0.5,
    )
    setattr(request, field, f" {getattr(request, field)} ")

    with pytest.raises(module.ModelUpdateRequestError, match=field):
        module._validate_model_update_request(request)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("candidate_id", "", "candidate_id"),
        ("reward", float("nan"), "reward"),
        ("reward", -0.1, "reward"),
        ("reward", 1.1, "reward"),
        ("outcome", "UNKNOWN", "outcome"),
    ],
)
def test_iclm_update_rejects_invalid_sample_contract(
    field: str,
    value: object,
    message: str,
) -> None:
    module = _load_module(
        f"iclm_invalid_sample_{field}_{value!s}_test",
        ROOT / "services/iclm-svc/src/iclm_svc/main.py",
    )
    request = _valid_model_update_request(
        samples=[{"smiles": "CCO"}],
        teacher_embeddings=[[0.0]],
        kd_weight=0.5,
    )
    payload = json.loads(request.training_batch_json)
    if value is None:
        payload["samples"][0].pop(field)
    else:
        payload["samples"][0][field] = value
    request.training_batch_json = json.dumps(payload)

    with pytest.raises(module.ModelUpdateRequestError, match=message):
        module._validate_model_update_request(request)


def test_iclm_update_accepts_all_zero_sample_rewards() -> None:
    module = _load_module(
        "iclm_zero_sample_rewards_test",
        ROOT / "services/iclm-svc/src/iclm_svc/main.py",
    )
    request = _valid_model_update_request(
        samples=[{"smiles": "CCO", "reward": 0.0}],
        teacher_embeddings=[[0.0]],
        kd_weight=0.5,
    )

    payload = module._validate_model_update_request(request)

    assert payload["samples"][0]["reward"] == 0.0


def test_iclm_update_accepts_zero_teacher_weight() -> None:
    module = _load_module(
        "iclm_zero_teacher_weight_test",
        ROOT / "services/iclm-svc/src/iclm_svc/main.py",
    )
    request = _valid_model_update_request(
        samples=[{"smiles": "CCO"}],
        teacher_embeddings=[[0.0]],
        kd_weight=0.0,
    )

    payload = module._validate_model_update_request(request)

    assert payload["teacher_weight"] == 0.0


def test_iclm_update_rejects_positive_teacher_weight_without_embeddings() -> None:
    module = _load_module(
        "iclm_positive_teacher_weight_without_embeddings_test",
        ROOT / "services/iclm-svc/src/iclm_svc/main.py",
    )
    request = _valid_model_update_request(
        samples=[{"smiles": "CCO"}],
        teacher_embeddings=[[0.0]],
        kd_weight=0.5,
    )
    request.teacher_embeddings = b""
    request.dim = 0

    with pytest.raises(
        module.ModelUpdateRequestError,
        match="positive teacher_weight requires teacher_embeddings",
    ):
        module._validate_model_update_request(request)


def test_iclm_update_defaults_legacy_sample_reward_to_one() -> None:
    module = _load_module(
        "iclm_legacy_sample_reward_test",
        ROOT / "services/iclm-svc/src/iclm_svc/main.py",
    )
    request = _valid_model_update_request(
        samples=[{"smiles": "CCO"}],
        teacher_embeddings=[[0.0]],
        kd_weight=0.5,
    )
    training_batch = json.loads(request.training_batch_json)
    training_batch["samples"][0].pop("candidate_id")
    training_batch["samples"][0].pop("reward")
    request.training_batch_json = json.dumps(training_batch)

    payload = module._validate_model_update_request(request)

    assert payload["samples"][0]["reward"] == 1.0
    assert payload["samples"][0]["candidate_id"] == "update-iclm:1"


def test_iclm_update_canonicalizes_training_smiles() -> None:
    module = _load_module(
        "iclm_canonical_training_smiles_test",
        ROOT / "services/iclm-svc/src/iclm_svc/main.py",
    )
    request = _valid_model_update_request(
        samples=[{"smiles": "OCC", "outcome": "PASS"}],
        teacher_embeddings=[[0.0]],
        kd_weight=0.5,
    )

    payload = module._validate_model_update_request(request)

    assert payload["samples"][0]["smiles"] == "CCO"
    assert payload["samples"][0]["outcome"] == "PASS"


def test_iclm_update_rejects_invalid_training_smiles() -> None:
    module = _load_module(
        "iclm_invalid_training_smiles_test",
        ROOT / "services/iclm-svc/src/iclm_svc/main.py",
    )
    request = _valid_model_update_request(
        samples=[{"smiles": "not-a-smiles"}],
        teacher_embeddings=[[0.0]],
        kd_weight=0.5,
    )

    with pytest.raises(module.ModelUpdateRequestError, match="sample smiles"):
        module._validate_model_update_request(request)


def test_iclm_teacher_artifact_checksum_binds_normalized_supervision() -> None:
    module = _load_module(
        "iclm_teacher_supervision_checksum_test",
        ROOT / "services/iclm-svc/src/iclm_svc/main.py",
    )
    first_request = _valid_model_update_request(
        samples=[{"smiles": "OCC", "reward": 0.2, "outcome": "FAIL"}],
        teacher_embeddings=[[0.0]],
        kd_weight=0.0,
    )
    first_request.teacher_embeddings = b""
    first_request.dim = 0
    second_request = generator_pb2.ModelUpdateRequest.FromString(
        first_request.SerializeToString()
    )
    second_batch = json.loads(second_request.training_batch_json)
    second_batch["samples"][0]["reward"] = 0.8
    second_batch["samples"][0]["outcome"] = "PASS"
    second_request.training_batch_json = json.dumps(second_batch)

    first_payload = module._validate_model_update_request(first_request)
    second_payload = module._validate_model_update_request(second_request)

    first_checksum = module._teacher_supervision_checksum(
        first_request,
        first_payload,
    )
    second_checksum = module._teacher_supervision_checksum(
        second_request,
        second_payload,
    )
    assert first_checksum.startswith("sha256:")
    assert second_checksum.startswith("sha256:")
    assert first_checksum != second_checksum


class _ICLMRecordingGenerator:
    def __init__(
        self,
        checkpoint_path: Path,
        *,
        smiles: str = "CCO",
        online_learner=None,
        validation_error: Exception | None = None,
        embedding_dimension: int = 1,
    ) -> None:
        self.checkpoint_path = str(checkpoint_path)
        self.smiles = smiles
        self.online_learner = online_learner
        self.validation_error = validation_error
        self._embedding_dimension = embedding_dimension
        self.generate_calls = 0
        self.validation_calls = 0

    async def generate(self, batch_size: int, **kwargs):
        self.generate_calls += 1
        return [Molecule(smiles=self.smiles) for _ in range(batch_size)]

    def validate_checkpoint(self) -> None:
        self.validation_calls += 1
        if self.validation_error is not None:
            raise self.validation_error

    def embedding_dimension(self) -> int:
        return self._embedding_dimension


@pytest.mark.asyncio
async def test_iclm_info_exposes_student_embedding_dimension(tmp_path: Path) -> None:
    module = _load_module(
        "iclm_student_embedding_info_test",
        ROOT / "services/iclm-svc/src/iclm_svc/main.py",
    )
    checkpoint = tmp_path / "active"
    checkpoint.write_bytes(b"active")

    response = await module.ICLMServicer(
        generator=_ICLMRecordingGenerator(checkpoint, embedding_dimension=2),
    ).Info(None, None)

    assert response.default_params["student_embedding_dim"] == "2"


@pytest.mark.asyncio
async def test_iclm_update_rejects_teacher_embedding_dimension_before_training(
    tmp_path: Path,
) -> None:
    module = _load_module(
        "iclm_update_embedding_dimension_test",
        ROOT / "services/iclm-svc/src/iclm_svc/main.py",
    )
    checkpoint = tmp_path / "active"
    checkpoint.write_bytes(b"active")
    factory_calls = 0

    def generator_factory(checkpoint_path: str):
        nonlocal factory_calls
        factory_calls += 1
        return _ICLMRecordingGenerator(Path(checkpoint_path))

    active_generator = _ICLMRecordingGenerator(
        checkpoint,
        embedding_dimension=2,
    )
    servicer = module.ICLMServicer(
        generator=active_generator,
        generator_factory=generator_factory,
    )

    with pytest.raises(
        ValueError,
        match="teacher embedding dimension 1 does not match student embedding dimension 2",
    ):
        await servicer.UpdateModel(
            _valid_model_update_request(
                samples=[{"smiles": "CCO"}],
                teacher_embeddings=[[0.1]],
                kd_weight=0.5,
            ),
            None,
        )

    assert factory_calls == 0
    assert servicer.generator is active_generator
    assert servicer._update_records == {}


@pytest.mark.asyncio
async def test_iclm_update_activates_fresh_runner_without_mutating_inflight_generate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_module(
        "iclm_atomic_activation_test",
        ROOT / "services/iclm-svc/src/iclm_svc/main.py",
    )
    old_checkpoint = tmp_path / "old"
    old_checkpoint.write_bytes(b"old")
    new_checkpoint = tmp_path / "new"
    new_checkpoint.write_bytes(b"new")
    monkeypatch.setenv("ICLM_MODEL_PATH", str(old_checkpoint))
    monkeypatch.delenv("ICLM_UPDATE_COMMAND", raising=False)
    monkeypatch.setattr(module, "_require_runtime", lambda **kwargs: [])

    generate_started = asyncio.Event()
    release_generate = asyncio.Event()

    class OldGenerator(_ICLMRecordingGenerator):
        async def generate(self, batch_size: int, **kwargs):
            self.generate_calls += 1
            generate_started.set()
            await release_generate.wait()
            return [Molecule(smiles="CCO") for _ in range(batch_size)]

    class Learner:
        def update(self, payload):
            return {
                "checkpoint_path": str(new_checkpoint),
                "updated_samples": 1,
            }

    old_generator = OldGenerator(old_checkpoint)
    staged_generator = _ICLMRecordingGenerator(
        old_checkpoint,
        online_learner=Learner(),
    )
    new_generator = _ICLMRecordingGenerator(new_checkpoint, smiles="CCN")
    factory_calls: list[str] = []

    def generator_factory(checkpoint_path: str):
        factory_calls.append(checkpoint_path)
        return staged_generator if len(factory_calls) == 1 else new_generator

    servicer = module.ICLMServicer(
        generator=old_generator,
        generator_factory=generator_factory,
    )
    generate_task = asyncio.create_task(
        servicer.Generate(_valid_generator_request(batch_size=1), None)
    )
    await generate_started.wait()

    response = await servicer.UpdateModel(
        _valid_model_update_request(
            samples=[{"smiles": "CCO"}],
            teacher_embeddings=[[0.1]],
            kd_weight=0.5,
        ),
        None,
    )
    release_generate.set()
    await generate_task
    await servicer.Generate(_valid_generator_request(batch_size=1), None)

    assert response.acknowledged is True
    assert response.active_version == "iclm-v2"
    assert factory_calls == [str(old_checkpoint), str(new_checkpoint)]
    assert old_generator.generate_calls == 1
    assert new_generator.generate_calls == 1
    assert servicer.generator is new_generator


@pytest.mark.asyncio
async def test_iclm_failed_runner_construction_keeps_previous_model_active(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_module(
        "iclm_activation_rollback_test",
        ROOT / "services/iclm-svc/src/iclm_svc/main.py",
    )
    old_checkpoint = tmp_path / "old"
    old_checkpoint.write_bytes(b"old")
    new_checkpoint = tmp_path / "new"
    new_checkpoint.write_bytes(b"new")
    monkeypatch.setenv("ICLM_MODEL_PATH", str(old_checkpoint))
    monkeypatch.delenv("ICLM_UPDATE_COMMAND", raising=False)
    monkeypatch.setattr(module, "_require_runtime", lambda **kwargs: [])

    class Learner:
        def update(self, payload):
            return {
                "checkpoint_path": str(new_checkpoint),
                "updated_samples": 1,
            }

    old_generator = _ICLMRecordingGenerator(old_checkpoint)
    staged_generator = _ICLMRecordingGenerator(
        old_checkpoint,
        online_learner=Learner(),
    )
    factory_calls = 0

    def generator_factory(checkpoint_path: str):
        nonlocal factory_calls
        factory_calls += 1
        if factory_calls == 1:
            return staged_generator
        raise RuntimeError("checkpoint cannot be loaded")

    servicer = module.ICLMServicer(
        generator=old_generator,
        generator_factory=generator_factory,
    )

    with pytest.raises(RuntimeError, match="checkpoint cannot be loaded"):
        await servicer.UpdateModel(
            _valid_model_update_request(
                samples=[{"smiles": "CCO"}],
                teacher_embeddings=[[0.1]],
                kd_weight=0.5,
            ),
            None,
        )

    assert servicer.generator is old_generator


@pytest.mark.asyncio
async def test_iclm_update_request_id_is_idempotent_and_bound_to_request(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_module(
        "iclm_update_idempotency_test",
        ROOT / "services/iclm-svc/src/iclm_svc/main.py",
    )
    old_checkpoint = tmp_path / "old"
    old_checkpoint.write_bytes(b"old")
    new_checkpoint = tmp_path / "new"
    new_checkpoint.write_bytes(b"new")
    monkeypatch.setenv("ICLM_MODEL_PATH", str(old_checkpoint))
    monkeypatch.delenv("ICLM_UPDATE_COMMAND", raising=False)
    monkeypatch.setattr(module, "_require_runtime", lambda **kwargs: [])
    learner_calls = 0

    class Learner:
        def update(self, payload):
            nonlocal learner_calls
            learner_calls += 1
            return {
                "checkpoint_path": str(new_checkpoint),
                "updated_samples": 1,
            }

    staged_generator = _ICLMRecordingGenerator(
        old_checkpoint,
        online_learner=Learner(),
    )
    new_generator = _ICLMRecordingGenerator(new_checkpoint)
    factory_results = iter((staged_generator, new_generator))
    servicer = module.ICLMServicer(
        generator=_ICLMRecordingGenerator(old_checkpoint),
        generator_factory=lambda checkpoint_path: next(factory_results),
    )
    request = _valid_model_update_request(
        samples=[{"smiles": "CCO"}],
        teacher_embeddings=[[0.1]],
        kd_weight=0.5,
    )

    first = await servicer.UpdateModel(request, None)
    second = await servicer.UpdateModel(
        generator_pb2.ModelUpdateRequest.FromString(request.SerializeToString()),
        None,
    )
    conflicting = generator_pb2.ModelUpdateRequest.FromString(request.SerializeToString())
    conflicting.target_checkpoint_version = "iclm-v3"
    with pytest.raises(ValueError, match="request_id"):
        await servicer.UpdateModel(conflicting, None)

    assert first == second
    assert learner_calls == 1


@pytest.mark.asyncio
async def test_iclm_success_state_recovers_checkpoint_and_idempotency_after_restart(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_module(
        "iclm_persistent_update_state_test",
        ROOT / "services/iclm-svc/src/iclm_svc/main.py",
    )
    old_checkpoint = tmp_path / "old"
    old_checkpoint.write_bytes(b"old")
    new_checkpoint = tmp_path / "new"
    new_checkpoint.write_bytes(b"new")
    state_path = tmp_path / "state" / "iclm.json"
    monkeypatch.setenv("ICLM_MODEL_PATH", str(old_checkpoint))
    monkeypatch.setenv("ICLM_STATE_PATH", str(state_path))
    monkeypatch.delenv("ICLM_UPDATE_COMMAND", raising=False)
    monkeypatch.setattr(module, "_require_runtime", lambda **kwargs: [])
    learner_calls = 0

    class Learner:
        def update(self, payload):
            nonlocal learner_calls
            learner_calls += 1
            return {
                "checkpoint_path": str(new_checkpoint),
                "updated_samples": 1,
            }

    first_factory_results = iter(
        (
            _ICLMRecordingGenerator(old_checkpoint, online_learner=Learner()),
            _ICLMRecordingGenerator(new_checkpoint),
        )
    )
    initial_generator = _ICLMRecordingGenerator(old_checkpoint)
    first_servicer = module.ICLMServicer(
        generator=initial_generator,
        generator_factory=lambda checkpoint_path: next(first_factory_results),
    )
    request = _valid_model_update_request(
        samples=[{"smiles": "CCO"}],
        teacher_embeddings=[[0.1]],
        kd_weight=0.5,
    )

    await first_servicer.initialize()

    assert state_path.is_file()
    assert initial_generator.validation_calls == 1

    first_response = await first_servicer.UpdateModel(request, None)

    class RecoveredGenerator(_ICLMRecordingGenerator):
        def __init__(self, checkpoint_path: Path) -> None:
            super().__init__(checkpoint_path)
            self.loaded_checkpoint = b""

        def validate_checkpoint(self) -> None:
            super().validate_checkpoint()
            self.loaded_checkpoint = Path(self.checkpoint_path).read_bytes()

    old_checkpoint.unlink()
    recovered_generator = RecoveredGenerator(new_checkpoint)
    recovery_calls: list[str] = []

    def recovery_factory(checkpoint_path: str):
        recovery_calls.append(checkpoint_path)
        return recovered_generator

    restarted_servicer = module.ICLMServicer(
        generator=_ICLMRecordingGenerator(old_checkpoint),
        generator_factory=recovery_factory,
    )
    await restarted_servicer.initialize()
    replay_response = await restarted_servicer.UpdateModel(
        generator_pb2.ModelUpdateRequest.FromString(request.SerializeToString()),
        None,
    )
    conflicting = generator_pb2.ModelUpdateRequest.FromString(request.SerializeToString())
    conflicting.target_checkpoint_version = "iclm-v3"
    with pytest.raises(ValueError, match="request_id"):
        await restarted_servicer.UpdateModel(conflicting, None)
    info = await restarted_servicer.Info(None, None)

    assert replay_response == first_response
    assert learner_calls == 1
    assert recovery_calls == [str(new_checkpoint)]
    assert recovered_generator.validation_calls == 1
    assert recovered_generator.loaded_checkpoint == b"new"
    assert restarted_servicer.generator is recovered_generator
    assert info.version == "iclm-v2"


@pytest.mark.asyncio
async def test_iclm_recovery_rejects_tampered_active_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_module(
        "iclm_tampered_checkpoint_state_test",
        ROOT / "services/iclm-svc/src/iclm_svc/main.py",
    )
    active_checkpoint = tmp_path / "active"
    active_checkpoint.write_bytes(b"trusted-checkpoint")
    state_path = tmp_path / "iclm-state.json"
    monkeypatch.setenv("ICLM_MODEL_PATH", str(active_checkpoint))
    monkeypatch.setenv("ICLM_STATE_PATH", str(state_path))
    monkeypatch.setattr(module, "_require_runtime", lambda **kwargs: [])

    first_servicer = module.ICLMServicer(
        generator=_ICLMRecordingGenerator(active_checkpoint),
        generator_factory=lambda checkpoint_path: _ICLMRecordingGenerator(
            Path(checkpoint_path)
        ),
    )
    await first_servicer.initialize()
    active_checkpoint.write_bytes(b"tampered-checkpoint")
    recovery_calls: list[str] = []

    def recovery_factory(checkpoint_path: str):
        recovery_calls.append(checkpoint_path)
        return _ICLMRecordingGenerator(Path(checkpoint_path))

    restarted_servicer = module.ICLMServicer(
        generator=_ICLMRecordingGenerator(active_checkpoint),
        generator_factory=recovery_factory,
    )

    with pytest.raises(RuntimeError, match="checkpoint checksum"):
        await restarted_servicer.initialize()

    assert recovery_calls == []


def test_iclm_state_rejects_missing_active_checkpoint_checksum(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_module(
        "iclm_missing_checkpoint_checksum_test",
        ROOT / "services/iclm-svc/src/iclm_svc/main.py",
    )
    active_checkpoint = tmp_path / "active"
    active_checkpoint.write_bytes(b"trusted-checkpoint")
    state_path = tmp_path / "iclm-state.json"
    state_path.write_text(
        json.dumps(
            {
                "schema_version": "iclm-state.v2",
                "active_checkpoint_path": str(active_checkpoint),
                "active_version": "iclm-v1",
                "retryable_updates": {},
                "successful_updates": {},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("ICLM_MODEL_PATH", str(active_checkpoint))
    monkeypatch.setenv("ICLM_STATE_PATH", str(state_path))

    with pytest.raises(RuntimeError, match="active_checkpoint_checksum"):
        module.ICLMServicer(
            generator=_ICLMRecordingGenerator(active_checkpoint),
            generator_factory=lambda checkpoint_path: _ICLMRecordingGenerator(
                Path(checkpoint_path)
            ),
        )


def test_iclm_legacy_success_record_survives_v3_state_rewrite(
    tmp_path: Path,
) -> None:
    module = _load_module(
        "iclm_legacy_success_migration_test",
        ROOT / "services/iclm-svc/src/iclm_svc/main.py",
    )
    active_checkpoint = tmp_path / "active"
    active_checkpoint.write_bytes(b"active")
    state_path = tmp_path / "iclm-state.json"
    response = generator_pb2.ModelUpdateResponse(
        acknowledged=True,
        active_version="iclm-v2",
        updated_samples=1,
        status=generator_pb2.MODEL_UPDATE_STATUS_APPLIED,
    )
    state_path.write_text(
        json.dumps(
            {
                "schema_version": "iclm-state.v2",
                "active_checkpoint_path": str(active_checkpoint),
                "active_checkpoint_checksum": f"sha256:{'a' * 64}",
                "active_version": "iclm-v2",
                "retryable_updates": {},
                "successful_updates": {
                    "legacy-request": {
                        "fingerprint": "b" * 64,
                        "response": base64.b64encode(
                            response.SerializeToString(deterministic=True)
                        ).decode("ascii"),
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    checkpoint_path, active_version, checksum, records = module._load_service_state(
        state_path
    )
    module._write_service_state(
        state_path,
        checkpoint_path=checkpoint_path,
        checkpoint_checksum=checksum,
        active_version=active_version,
        records=records,
    )
    _, _, _, reloaded_records = module._load_service_state(state_path)

    persisted = json.loads(state_path.read_text(encoding="utf-8"))
    assert persisted["schema_version"] == "iclm-state.v3"
    assert persisted["successful_updates"]["legacy-request"][
        "fingerprint_schema"
    ] == "iclm-update.v1"
    assert reloaded_records["legacy-request"].fingerprint_schema == "iclm-update.v1"
    assert reloaded_records["legacy-request"].response == response.SerializeToString(
        deterministic=True
    )


@pytest.mark.asyncio
async def test_iclm_legacy_success_record_replays_original_v1_request(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_module(
        "iclm_legacy_success_replay_test",
        ROOT / "services/iclm-svc/src/iclm_svc/main.py",
    )
    active_checkpoint = tmp_path / "active"
    active_checkpoint.write_bytes(b"active")
    request = _valid_model_update_request(
        samples=[{"smiles": "CCO"}],
        teacher_embeddings=[[0.25]],
        kd_weight=0.5,
    )
    request.request_id = "legacy-request"
    training_batch = json.loads(request.training_batch_json)
    training_batch["samples"][0].pop("candidate_id")
    training_batch["samples"][0].pop("reward")
    request.training_batch_json = json.dumps(training_batch, sort_keys=True)
    legacy_payload = {
        **training_batch,
        "run_id": request.run_id,
        "request_id": request.request_id,
        "kd_teacher_embeddings": [[0.25]],
        "teacher_source": request.teacher_source,
        "teacher_version": request.teacher_version,
        "target_checkpoint_version": request.target_checkpoint_version,
    }
    legacy_fingerprint = hashlib.sha256(
        json.dumps(
            legacy_payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    response = generator_pb2.ModelUpdateResponse(
        acknowledged=True,
        active_version="iclm-v2",
        updated_samples=1,
        status=generator_pb2.MODEL_UPDATE_STATUS_APPLIED,
    )
    checkpoint_checksum = module._checkpoint_artifact_sha256(str(active_checkpoint))
    state_path = tmp_path / "iclm-state.json"
    state_path.write_text(
        json.dumps(
            {
                "schema_version": "iclm-state.v2",
                "active_checkpoint_path": str(active_checkpoint),
                "active_checkpoint_checksum": checkpoint_checksum,
                "active_version": "iclm-v2",
                "retryable_updates": {},
                "successful_updates": {
                    request.request_id: {
                        "fingerprint": legacy_fingerprint,
                        "response": base64.b64encode(
                            response.SerializeToString(deterministic=True)
                        ).decode("ascii"),
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("ICLM_MODEL_PATH", str(active_checkpoint))
    monkeypatch.setenv("ICLM_STATE_PATH", str(state_path))
    monkeypatch.setattr(module, "_require_runtime", lambda **kwargs: [])
    recovery_calls: list[str] = []

    def recovery_factory(checkpoint_path: str):
        recovery_calls.append(checkpoint_path)
        return _ICLMRecordingGenerator(Path(checkpoint_path))

    servicer = module.ICLMServicer(
        generator=_ICLMRecordingGenerator(active_checkpoint),
        generator_factory=recovery_factory,
    )

    replay = await servicer.UpdateModel(request, None)

    assert replay == response
    assert recovery_calls == [str(active_checkpoint)]


@pytest.mark.asyncio
async def test_iclm_persists_absolute_active_checkpoint_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_module(
        "iclm_absolute_checkpoint_state_test",
        ROOT / "services/iclm-svc/src/iclm_svc/main.py",
    )
    checkpoint = tmp_path / "models" / "active"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"active")
    state_path = tmp_path / "iclm-state.json"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ICLM_MODEL_PATH", "models/active")
    monkeypatch.setenv("ICLM_STATE_PATH", str(state_path))
    monkeypatch.setattr(module, "_require_runtime", lambda **kwargs: [])
    generator = _ICLMRecordingGenerator(Path("models/active"))
    servicer = module.ICLMServicer(generator=generator)

    await servicer.initialize()

    persisted = json.loads(state_path.read_text(encoding="utf-8"))
    assert servicer._active_checkpoint_path == str(checkpoint.resolve())
    assert persisted["active_checkpoint_path"] == str(checkpoint.resolve())


@pytest.mark.asyncio
async def test_iclm_retryable_state_rejects_conflicting_request_after_restart(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_module(
        "iclm_retryable_update_state_test",
        ROOT / "services/iclm-svc/src/iclm_svc/main.py",
    )
    active_checkpoint = tmp_path / "active"
    active_checkpoint.write_bytes(b"active")
    state_path = tmp_path / "iclm-state.json"
    monkeypatch.setenv("ICLM_MODEL_PATH", str(active_checkpoint))
    monkeypatch.setenv("ICLM_STATE_PATH", str(state_path))
    monkeypatch.delenv("ICLM_UPDATE_COMMAND", raising=False)
    monkeypatch.setattr(module, "_require_runtime", lambda **kwargs: [])
    learner_calls = 0

    class FailingLearner:
        def update(self, payload):
            nonlocal learner_calls
            learner_calls += 1
            raise RuntimeError("training failed")

    training_generator = _ICLMRecordingGenerator(
        active_checkpoint,
        online_learner=FailingLearner(),
    )
    initial_servicer = module.ICLMServicer(
        generator=_ICLMRecordingGenerator(active_checkpoint),
        generator_factory=lambda checkpoint_path: training_generator,
    )
    request = _valid_model_update_request(
        samples=[{"smiles": "CCO"}],
        teacher_embeddings=[[0.1]],
        kd_weight=0.5,
    )
    await initial_servicer.initialize()

    with pytest.raises(RuntimeError, match="training failed"):
        await initial_servicer.UpdateModel(request, None)

    restarted_servicer = module.ICLMServicer(
        generator=_ICLMRecordingGenerator(active_checkpoint),
        generator_factory=lambda checkpoint_path: _ICLMRecordingGenerator(
            Path(checkpoint_path)
        ),
    )
    conflicting = generator_pb2.ModelUpdateRequest.FromString(
        request.SerializeToString()
    )
    conflicting.target_checkpoint_version = "iclm-v3"

    with pytest.raises(ValueError, match="request_id"):
        await restarted_servicer.UpdateModel(conflicting, None)

    assert learner_calls == 1


@pytest.mark.asyncio
async def test_iclm_persists_request_binding_before_update_runner_starts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_module(
        "iclm_pre_runner_request_binding_test",
        ROOT / "services/iclm-svc/src/iclm_svc/main.py",
    )
    active_checkpoint = tmp_path / "active"
    active_checkpoint.write_bytes(b"active")
    state_path = tmp_path / "iclm-state.json"
    monkeypatch.setenv("ICLM_MODEL_PATH", str(active_checkpoint))
    monkeypatch.setenv("ICLM_STATE_PATH", str(state_path))
    monkeypatch.delenv("ICLM_UPDATE_COMMAND", raising=False)
    monkeypatch.setattr(module, "_require_runtime", lambda **kwargs: [])
    learner_calls = 0

    class SimulatedProcessExit(BaseException):
        pass

    class CrashingLearner:
        def update(self, payload):
            nonlocal learner_calls
            learner_calls += 1
            raise SimulatedProcessExit

    initial_servicer = module.ICLMServicer(
        generator=_ICLMRecordingGenerator(active_checkpoint),
        generator_factory=lambda checkpoint_path: _ICLMRecordingGenerator(
            active_checkpoint,
            online_learner=CrashingLearner(),
        ),
    )
    request = _valid_model_update_request(
        samples=[{"smiles": "CCO"}],
        teacher_embeddings=[[0.1]],
        kd_weight=0.5,
    )
    await initial_servicer.initialize()

    with pytest.raises(SimulatedProcessExit):
        await initial_servicer.UpdateModel(request, None)

    restarted_servicer = module.ICLMServicer(
        generator=_ICLMRecordingGenerator(active_checkpoint),
        generator_factory=lambda checkpoint_path: _ICLMRecordingGenerator(
            Path(checkpoint_path)
        ),
    )
    conflicting = generator_pb2.ModelUpdateRequest.FromString(
        request.SerializeToString()
    )
    conflicting.target_checkpoint_version = "iclm-v3"

    with pytest.raises(ValueError, match="request_id"):
        await restarted_servicer.UpdateModel(conflicting, None)

    assert learner_calls == 1


@pytest.mark.asyncio
async def test_iclm_revalidates_active_checkpoint_checksum_before_new_update(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_module(
        "iclm_cached_checkpoint_checksum_test",
        ROOT / "services/iclm-svc/src/iclm_svc/main.py",
    )
    active_checkpoint = tmp_path / "active"
    active_checkpoint.write_bytes(b"active")
    state_path = tmp_path / "iclm-state.json"
    monkeypatch.setenv("ICLM_MODEL_PATH", str(active_checkpoint))
    monkeypatch.setenv("ICLM_STATE_PATH", str(state_path))
    monkeypatch.delenv("ICLM_UPDATE_COMMAND", raising=False)
    monkeypatch.setattr(module, "_require_runtime", lambda **kwargs: [])
    checksum_calls: list[str] = []
    checksum = module._checkpoint_artifact_sha256

    def recording_checksum(checkpoint_path: str) -> str:
        checksum_calls.append(checkpoint_path)
        return checksum(checkpoint_path)

    class FailingLearner:
        def update(self, payload):
            raise RuntimeError("training failed")

    monkeypatch.setattr(
        module,
        "_checkpoint_artifact_sha256",
        recording_checksum,
    )
    servicer = module.ICLMServicer(
        generator=_ICLMRecordingGenerator(active_checkpoint),
        generator_factory=lambda checkpoint_path: _ICLMRecordingGenerator(
            active_checkpoint,
            online_learner=FailingLearner(),
        ),
    )
    await servicer.initialize()

    with pytest.raises(RuntimeError, match="training failed"):
        await servicer.UpdateModel(
            _valid_model_update_request(
                samples=[{"smiles": "CCO"}],
                teacher_embeddings=[[0.1]],
                kd_weight=0.5,
            ),
            None,
        )

    assert checksum_calls == [str(active_checkpoint), str(active_checkpoint)]


@pytest.mark.asyncio
async def test_iclm_rejects_new_update_after_active_checkpoint_tampering(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_module(
        "iclm_tampered_active_update_test",
        ROOT / "services/iclm-svc/src/iclm_svc/main.py",
    )
    active_checkpoint = tmp_path / "active"
    active_checkpoint.write_bytes(b"trusted")
    state_path = tmp_path / "iclm-state.json"
    monkeypatch.setenv("ICLM_MODEL_PATH", str(active_checkpoint))
    monkeypatch.setenv("ICLM_STATE_PATH", str(state_path))
    monkeypatch.delenv("ICLM_UPDATE_COMMAND", raising=False)
    monkeypatch.setattr(module, "_require_runtime", lambda **kwargs: [])
    learner_calls = 0

    class Learner:
        def update(self, payload):
            nonlocal learner_calls
            learner_calls += 1
            raise AssertionError("tampered base must not reach the learner")

    servicer = module.ICLMServicer(
        generator=_ICLMRecordingGenerator(active_checkpoint),
        generator_factory=lambda checkpoint_path: _ICLMRecordingGenerator(
            active_checkpoint,
            online_learner=Learner(),
        ),
    )
    await servicer.initialize()
    active_checkpoint.write_bytes(b"tampered")

    with pytest.raises(RuntimeError, match="checkpoint checksum"):
        await servicer.UpdateModel(
            _valid_model_update_request(
                samples=[{"smiles": "CCO"}],
                teacher_embeddings=[[0.1]],
                kd_weight=0.5,
            ),
            None,
        )

    assert learner_calls == 0


@pytest.mark.asyncio
async def test_iclm_state_replace_failure_keeps_active_checkpoint_and_allows_retry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_module(
        "iclm_atomic_state_write_test",
        ROOT / "services/iclm-svc/src/iclm_svc/main.py",
    )
    old_checkpoint, v2_checkpoint, v3_checkpoint = [
        tmp_path / name for name in ("old", "v2", "v3")
    ]
    for checkpoint in (old_checkpoint, v2_checkpoint, v3_checkpoint):
        checkpoint.write_bytes(checkpoint.name.encode())
    state_path = tmp_path / "iclm-state.json"
    monkeypatch.setenv("ICLM_MODEL_PATH", str(old_checkpoint))
    monkeypatch.setenv("ICLM_STATE_PATH", str(state_path))
    monkeypatch.delenv("ICLM_UPDATE_COMMAND", raising=False)
    monkeypatch.setattr(module, "_require_runtime", lambda **kwargs: [])

    class Learner:
        def __init__(self, checkpoint_path: Path) -> None:
            self.checkpoint_path = checkpoint_path

        def update(self, payload):
            return {
                "checkpoint_path": str(self.checkpoint_path),
                "updated_samples": 1,
            }

    active_v2_generator = _ICLMRecordingGenerator(v2_checkpoint)
    active_v3_generator = _ICLMRecordingGenerator(v3_checkpoint)
    factory_results = iter(
        (
            _ICLMRecordingGenerator(
                old_checkpoint,
                online_learner=Learner(v2_checkpoint),
            ),
            active_v2_generator,
            _ICLMRecordingGenerator(
                v2_checkpoint,
                online_learner=Learner(v3_checkpoint),
            ),
            active_v3_generator,
        )
    )
    servicer = module.ICLMServicer(
        generator=_ICLMRecordingGenerator(old_checkpoint),
        generator_factory=lambda checkpoint_path: next(factory_results),
    )
    await servicer.UpdateModel(
        _valid_model_update_request(
            samples=[{"smiles": "CCO"}],
            teacher_embeddings=[[0.1]],
            kd_weight=0.5,
            target_version="iclm-v2",
        ),
        None,
    )
    persisted_v2_state = state_path.read_bytes()
    v3_request = _valid_model_update_request(
        samples=[{"smiles": "CCN"}],
        teacher_embeddings=[[0.2]],
        kd_weight=0.5,
        target_version="iclm-v3",
    )
    v3_request.request_id = "update-iclm-v3"
    real_replace = module.os.replace

    with monkeypatch.context() as state_failure:

        def reject_state_replace(source, destination) -> None:
            if Path(destination) == state_path:
                raise OSError("state disk unavailable")
            real_replace(source, destination)

        state_failure.setattr(module.os, "replace", reject_state_replace)
        with pytest.raises(RuntimeError, match="state disk unavailable"):
            await servicer.UpdateModel(v3_request, None)

    assert servicer.generator is active_v2_generator
    assert state_path.read_bytes() == persisted_v2_state
    assert list(tmp_path.glob(".iclm-state.json.*.tmp")) == []

    response = await servicer.UpdateModel(
        generator_pb2.ModelUpdateRequest.FromString(v3_request.SerializeToString()),
        None,
    )

    assert response.active_version == "iclm-v3"
    assert servicer.generator is active_v3_generator
    assert state_path.read_bytes() != persisted_v2_state


@pytest.mark.asyncio
async def test_iclm_concurrent_updates_are_serialized(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_module(
        "iclm_serial_update_test",
        ROOT / "services/iclm-svc/src/iclm_svc/main.py",
    )
    checkpoints = [tmp_path / name for name in ("old", "v2", "v3")]
    for index, checkpoint in enumerate(checkpoints):
        checkpoint.write_bytes(f"checkpoint-{index}".encode())
    monkeypatch.setenv("ICLM_MODEL_PATH", str(checkpoints[0]))
    monkeypatch.delenv("ICLM_UPDATE_COMMAND", raising=False)
    monkeypatch.setattr(module, "_require_runtime", lambda **kwargs: [])
    running_updates = 0
    max_running_updates = 0

    class Learner:
        def __init__(self, checkpoint: Path) -> None:
            self.checkpoint = checkpoint

        async def update(self, payload):
            nonlocal running_updates, max_running_updates
            running_updates += 1
            max_running_updates = max(max_running_updates, running_updates)
            await asyncio.sleep(0)
            running_updates -= 1
            return {
                "checkpoint_path": str(self.checkpoint),
                "updated_samples": 1,
            }

    factory_results = iter(
        (
            _ICLMRecordingGenerator(checkpoints[0], online_learner=Learner(checkpoints[1])),
            _ICLMRecordingGenerator(checkpoints[1]),
            _ICLMRecordingGenerator(checkpoints[1], online_learner=Learner(checkpoints[2])),
            _ICLMRecordingGenerator(checkpoints[2]),
        )
    )
    servicer = module.ICLMServicer(
        generator=_ICLMRecordingGenerator(checkpoints[0]),
        generator_factory=lambda checkpoint_path: next(factory_results),
    )
    first_request = _valid_model_update_request(
        samples=[{"smiles": "CCO"}],
        teacher_embeddings=[[0.1]],
        kd_weight=0.5,
        target_version="iclm-v2",
    )
    second_request = _valid_model_update_request(
        samples=[{"smiles": "CCN"}],
        teacher_embeddings=[[0.2]],
        kd_weight=0.5,
        target_version="iclm-v3",
    )
    second_request.request_id = "update-iclm-2"

    first, second = await asyncio.gather(
        servicer.UpdateModel(first_request, None),
        servicer.UpdateModel(second_request, None),
    )

    assert first.active_version == "iclm-v2"
    assert second.active_version == "iclm-v3"
    assert max_running_updates == 1
    assert servicer.generator.checkpoint_path == str(checkpoints[2])


@pytest.mark.asyncio
async def test_iclm_update_command_does_not_block_event_loop(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_module(
        "iclm_async_update_command_test",
        ROOT / "services/iclm-svc/src/iclm_svc/main.py",
    )
    old_checkpoint = tmp_path / "old"
    old_checkpoint.write_bytes(b"old")
    new_checkpoint = tmp_path / "new"
    marker = tmp_path / "event-loop-marker"
    runner = tmp_path / "wait_for_marker.py"
    runner.write_text(
        "import json, sys, time\n"
        "from pathlib import Path\n"
        "payload = json.load(sys.stdin)\n"
        f"marker = Path({str(marker)!r})\n"
        "deadline = time.monotonic() + 0.5\n"
        "while not marker.exists() and time.monotonic() < deadline:\n"
        "    time.sleep(0.01)\n"
        "if not marker.exists():\n"
        "    raise SystemExit(2)\n"
        f"checkpoint = Path({str(new_checkpoint)!r})\n"
        "checkpoint.write_bytes(b'new')\n"
        "print(json.dumps({'checkpoint_path': str(checkpoint), 'updated_samples': 1}))\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("ICLM_MODEL_PATH", str(old_checkpoint))
    monkeypatch.setenv("ICLM_UPDATE_COMMAND", f"{sys.executable} {runner}")
    monkeypatch.setenv("ICLM_UPDATE_TIMEOUT_SECONDS", "1")
    monkeypatch.setattr(module, "_require_runtime", lambda **kwargs: [])
    new_generator = _ICLMRecordingGenerator(new_checkpoint)
    servicer = module.ICLMServicer(
        generator=_ICLMRecordingGenerator(old_checkpoint),
        generator_factory=lambda checkpoint_path: new_generator,
    )
    request = _valid_model_update_request(
        samples=[{"smiles": "CCO"}],
        teacher_embeddings=[[0.1]],
        kd_weight=0.5,
    )

    update_task = asyncio.create_task(servicer.UpdateModel(request, None))
    await asyncio.sleep(0)
    marker.write_text("released", encoding="utf-8")
    response = await update_task

    assert response.acknowledged is True
    assert servicer.generator is new_generator


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "learner_result",
    (
        0.25,
        "active-checkpoint",
    ),
)
async def test_iclm_online_update_rejects_scalar_and_active_checkpoint_results(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    learner_result: float | str,
) -> None:
    module = _load_module(
        f"iclm_update_result_{type(learner_result).__name__}_test",
        ROOT / "services/iclm-svc/src/iclm_svc/main.py",
    )
    active_checkpoint = tmp_path / "active"
    active_checkpoint.write_bytes(b"active")
    monkeypatch.setenv("ICLM_MODEL_PATH", str(active_checkpoint))
    monkeypatch.delenv("ICLM_UPDATE_COMMAND", raising=False)
    monkeypatch.setattr(module, "_require_runtime", lambda **kwargs: [])

    class Learner:
        def update(self, payload):
            if isinstance(learner_result, str):
                return {
                    "checkpoint_path": str(active_checkpoint),
                    "updated_samples": 1,
                }
            return learner_result

    active_generator = _ICLMRecordingGenerator(active_checkpoint)
    staged_generator = _ICLMRecordingGenerator(
        active_checkpoint,
        online_learner=Learner(),
    )
    servicer = module.ICLMServicer(
        generator=active_generator,
        generator_factory=lambda checkpoint_path: staged_generator,
    )

    with pytest.raises(RuntimeError, match="new checkpoint"):
        await servicer.UpdateModel(
            _valid_model_update_request(
                samples=[{"smiles": "CCO"}],
                teacher_embeddings=[[0.1]],
                kd_weight=0.5,
            ),
            None,
        )

    assert servicer.generator is active_generator


@pytest.mark.asyncio
async def test_iclm_command_update_rejects_active_checkpoint_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_module(
        "iclm_command_active_checkpoint_test",
        ROOT / "services/iclm-svc/src/iclm_svc/main.py",
    )
    active_checkpoint = tmp_path / "active"
    active_checkpoint.write_bytes(b"active")
    runner = tmp_path / "same_checkpoint.py"
    runner.write_text(
        "import json, sys\n"
        "payload = json.load(sys.stdin)\n"
        "print(json.dumps({"
        "'checkpoint_path': payload['model_path'], "
        "'updated_samples': 1"
        "}))\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("ICLM_MODEL_PATH", str(active_checkpoint))
    monkeypatch.setenv("ICLM_UPDATE_COMMAND", f"{sys.executable} {runner}")
    monkeypatch.setattr(module, "_require_runtime", lambda **kwargs: [])
    active_generator = _ICLMRecordingGenerator(active_checkpoint)
    servicer = module.ICLMServicer(
        generator=active_generator,
        generator_factory=lambda checkpoint_path: _ICLMRecordingGenerator(Path(checkpoint_path)),
    )

    with pytest.raises(RuntimeError, match="new checkpoint"):
        await servicer.UpdateModel(
            _valid_model_update_request(
                samples=[{"smiles": "CCO"}],
                teacher_embeddings=[[0.1]],
                kd_weight=0.5,
            ),
            None,
        )

    assert servicer.generator is active_generator


@pytest.mark.asyncio
async def test_iclm_generate_and_info_use_one_active_state_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_module(
        "iclm_active_snapshot_test",
        ROOT / "services/iclm-svc/src/iclm_svc/main.py",
    )
    active_checkpoint = tmp_path / "active"
    active_checkpoint.write_bytes(b"active")
    new_checkpoint = tmp_path / "new"
    new_checkpoint.write_bytes(b"new")
    monkeypatch.setenv("ICLM_MODEL_PATH", str(active_checkpoint))
    monkeypatch.delenv("ICLM_UPDATE_COMMAND", raising=False)

    class Learner:
        def update(self, payload):
            return {
                "checkpoint_path": str(new_checkpoint),
                "updated_samples": 1,
            }

    staged_generator = _ICLMRecordingGenerator(
        active_checkpoint,
        online_learner=Learner(),
    )
    activated_generator = _ICLMRecordingGenerator(new_checkpoint)
    factory_results = iter((staged_generator, activated_generator))
    servicer = module.ICLMServicer(
        generator=_ICLMRecordingGenerator(active_checkpoint),
        generator_factory=lambda checkpoint_path: next(factory_results),
    )

    await servicer.UpdateModel(
        _valid_model_update_request(
            samples=[{"smiles": "CCO"}],
            teacher_embeddings=[[0.1]],
            kd_weight=0.5,
        ),
        None,
    )
    active_checkpoint.unlink()
    generated = await servicer.Generate(_valid_generator_request(batch_size=1), None)
    info = await servicer.Info(None, None)

    assert info.version == "iclm-v2"
    assert info.runtime_status == 1
    assert generated.artifacts[0].checksum == info.artifacts[0].checksum
    assert generated.artifacts[0].checksum.startswith("sha256:")
    assert activated_generator.validation_calls == 1


@pytest.mark.asyncio
async def test_iclm_eager_validation_failure_does_not_activate_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_module(
        "iclm_eager_validation_failure_test",
        ROOT / "services/iclm-svc/src/iclm_svc/main.py",
    )
    active_checkpoint = tmp_path / "active"
    active_checkpoint.write_bytes(b"active")
    new_checkpoint = tmp_path / "new"
    new_checkpoint.write_bytes(b"new")
    monkeypatch.setenv("ICLM_MODEL_PATH", str(active_checkpoint))
    monkeypatch.delenv("ICLM_UPDATE_COMMAND", raising=False)
    monkeypatch.setattr(module, "_require_runtime", lambda **kwargs: [])

    class Learner:
        def update(self, payload):
            return {
                "checkpoint_path": str(new_checkpoint),
                "updated_samples": 1,
            }

    active_generator = _ICLMRecordingGenerator(active_checkpoint)
    invalid_generator = _ICLMRecordingGenerator(
        new_checkpoint,
        validation_error=RuntimeError("weights cannot be loaded"),
    )
    factory_results = iter(
        (
            _ICLMRecordingGenerator(active_checkpoint, online_learner=Learner()),
            invalid_generator,
        )
    )
    servicer = module.ICLMServicer(
        generator=active_generator,
        generator_factory=lambda checkpoint_path: next(factory_results),
    )

    with pytest.raises(RuntimeError, match="weights cannot be loaded"):
        await servicer.UpdateModel(
            _valid_model_update_request(
                samples=[{"smiles": "CCO"}],
                teacher_embeddings=[[0.1]],
                kd_weight=0.5,
            ),
            None,
        )

    assert invalid_generator.validation_calls == 1
    assert servicer.generator is active_generator


@pytest.mark.asyncio
async def test_iclm_eager_validation_loads_default_huggingface_runner(
    tmp_path: Path,
) -> None:
    module = _load_module(
        "iclm_default_runner_eager_load_test",
        ROOT / "services/iclm-svc/src/iclm_svc/main.py",
    )
    checkpoint = tmp_path / "checkpoint"
    checkpoint.write_bytes(b"checkpoint")
    load_calls = 0

    class HuggingFaceRunner:
        def _load(self):
            nonlocal load_calls
            load_calls += 1
            return object(), object()

    generator = SimpleNamespace(
        checkpoint_path=str(checkpoint),
        runner=HuggingFaceRunner(),
    )

    await module._eager_validate_generator(generator)

    assert load_calls == 1


@pytest.mark.asyncio
async def test_iclm_failed_request_keeps_binding_and_same_request_can_retry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_module(
        "iclm_failed_request_binding_test",
        ROOT / "services/iclm-svc/src/iclm_svc/main.py",
    )
    active_checkpoint = tmp_path / "active"
    active_checkpoint.write_bytes(b"active")
    new_checkpoint = tmp_path / "new"
    new_checkpoint.write_bytes(b"new")
    monkeypatch.setenv("ICLM_MODEL_PATH", str(active_checkpoint))
    monkeypatch.delenv("ICLM_UPDATE_COMMAND", raising=False)
    monkeypatch.setattr(module, "_require_runtime", lambda **kwargs: [])

    class Learner:
        def update(self, payload):
            return {
                "checkpoint_path": str(new_checkpoint),
                "updated_samples": 1,
            }

    factory_calls = 0

    def generator_factory(checkpoint_path: str):
        nonlocal factory_calls
        factory_calls += 1
        if factory_calls in (1, 3):
            return _ICLMRecordingGenerator(
                active_checkpoint,
                online_learner=Learner(),
            )
        if factory_calls == 2:
            raise RuntimeError("first activation failed")
        return _ICLMRecordingGenerator(new_checkpoint)

    servicer = module.ICLMServicer(
        generator=_ICLMRecordingGenerator(active_checkpoint),
        generator_factory=generator_factory,
    )
    request = _valid_model_update_request(
        samples=[{"smiles": "CCO"}],
        teacher_embeddings=[[0.1]],
        kd_weight=0.5,
    )

    with pytest.raises(RuntimeError, match="first activation failed"):
        await servicer.UpdateModel(request, None)
    conflicting = generator_pb2.ModelUpdateRequest.FromString(request.SerializeToString())
    conflicting.target_checkpoint_version = "iclm-v3"
    with pytest.raises(ValueError, match="request_id"):
        await servicer.UpdateModel(conflicting, None)
    response = await servicer.UpdateModel(
        generator_pb2.ModelUpdateRequest.FromString(request.SerializeToString()),
        None,
    )

    assert response.acknowledged is True
    assert response.active_version == "iclm-v2"
    assert factory_calls == 4


@pytest.mark.asyncio
async def test_iclm_retry_is_bound_to_original_active_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_module(
        "iclm_retry_base_checkpoint_binding_test",
        ROOT / "services/iclm-svc/src/iclm_svc/main.py",
    )
    active_checkpoint = tmp_path / "active"
    recovered_checkpoint = tmp_path / "recovered"
    next_checkpoint = tmp_path / "next"
    active_checkpoint.write_bytes(b"active")
    recovered_checkpoint.write_bytes(b"recovered")
    next_checkpoint.write_bytes(b"next")
    state_path = tmp_path / "iclm-state.json"
    monkeypatch.setenv("ICLM_MODEL_PATH", str(active_checkpoint))
    monkeypatch.setenv("ICLM_STATE_PATH", str(state_path))
    monkeypatch.delenv("ICLM_UPDATE_COMMAND", raising=False)
    monkeypatch.setattr(module, "_require_runtime", lambda **kwargs: [])

    class FailingLearner:
        def update(self, payload):
            raise RuntimeError("training failed")

    class SuccessfulLearner:
        def __init__(self, checkpoint_path: Path) -> None:
            self.checkpoint_path = checkpoint_path

        def update(self, payload):
            return {
                "checkpoint_path": str(self.checkpoint_path),
                "updated_samples": 1,
            }

    factory_results = iter(
        (
            _ICLMRecordingGenerator(
                active_checkpoint,
                online_learner=FailingLearner(),
            ),
            _ICLMRecordingGenerator(
                active_checkpoint,
                online_learner=SuccessfulLearner(recovered_checkpoint),
            ),
            _ICLMRecordingGenerator(recovered_checkpoint),
            _ICLMRecordingGenerator(
                recovered_checkpoint,
                online_learner=SuccessfulLearner(next_checkpoint),
            ),
            _ICLMRecordingGenerator(next_checkpoint),
        )
    )

    def factory(checkpoint_path: str):
        try:
            return next(factory_results)
        except StopIteration as exc:
            raise AssertionError("stale retry must not start a new learner") from exc

    servicer = module.ICLMServicer(
        generator=_ICLMRecordingGenerator(active_checkpoint),
        generator_factory=factory,
    )
    await servicer.initialize()
    failed_request = _valid_model_update_request(
        samples=[{"smiles": "CCO"}],
        teacher_embeddings=[[0.1]],
        kd_weight=0.5,
        target_version="failed-update",
    )

    with pytest.raises(RuntimeError, match="training failed"):
        await servicer.UpdateModel(failed_request, None)

    persisted = json.loads(state_path.read_text(encoding="utf-8"))
    retryable = persisted["retryable_updates"][failed_request.request_id]
    assert retryable["base_checkpoint_path"] == str(active_checkpoint)
    assert retryable["base_version"] == "0.1.0"
    assert retryable["base_checkpoint_checksum"].startswith("sha256:")

    next_request = _valid_model_update_request(
        samples=[{"smiles": "CCN"}],
        teacher_embeddings=[[0.2]],
        kd_weight=0.5,
        target_version="next-update",
    )
    next_request.request_id = "update-iclm-next"

    with pytest.raises(RuntimeError, match="retryable update must complete"):
        await servicer.UpdateModel(next_request, None)

    recovered = await servicer.UpdateModel(
        generator_pb2.ModelUpdateRequest.FromString(
            failed_request.SerializeToString()
        ),
        None,
    )
    advanced = await servicer.UpdateModel(next_request, None)

    assert recovered.active_version == "failed-update"
    assert advanced.active_version == "next-update"


@pytest.mark.asyncio
async def test_iclm_zero_reward_update_is_durable_skipped_acknowledgement(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_module(
        "iclm_zero_reward_skip_test",
        ROOT / "services/iclm-svc/src/iclm_svc/main.py",
    )
    active_checkpoint = tmp_path / "active"
    active_checkpoint.write_bytes(b"active")
    state_path = tmp_path / "iclm-state.json"
    monkeypatch.setenv("ICLM_MODEL_PATH", str(active_checkpoint))
    monkeypatch.setenv("ICLM_STATE_PATH", str(state_path))
    monkeypatch.setattr(module, "_require_runtime", lambda **kwargs: [])
    active_generator = _ICLMRecordingGenerator(active_checkpoint)
    servicer = module.ICLMServicer(
        generator=active_generator,
        generator_factory=lambda _checkpoint_path: (_ for _ in ()).throw(
            AssertionError("zero reward must not start a learner")
        ),
    )
    await servicer.initialize()
    request = _valid_model_update_request(
        samples=[{"smiles": "CCO", "reward": 0.0}],
        teacher_embeddings=[[0.1]],
        kd_weight=0.5,
    )

    first = await servicer.UpdateModel(request, None)
    replay = await servicer.UpdateModel(
        generator_pb2.ModelUpdateRequest.FromString(request.SerializeToString()),
        None,
    )

    assert first == replay
    assert first.acknowledged is True
    assert first.status == generator_pb2.MODEL_UPDATE_STATUS_SKIPPED
    assert first.active_version == "0.1.0"
    assert first.updated_samples == 0
    assert servicer.generator is active_generator
    persisted = json.loads(state_path.read_text(encoding="utf-8"))
    assert request.request_id in persisted["successful_updates"]


@pytest.mark.asyncio
async def test_iclm_zero_reward_failure_is_trained_as_maximum_unlikelihood(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_module(
        "iclm_zero_reward_failure_update_test",
        ROOT / "services/iclm-svc/src/iclm_svc/main.py",
    )
    active_checkpoint = tmp_path / "active"
    updated_checkpoint = tmp_path / "updated"
    active_checkpoint.write_bytes(b"active")
    updated_checkpoint.write_bytes(b"updated")
    monkeypatch.setenv("ICLM_MODEL_PATH", str(active_checkpoint))
    monkeypatch.setattr(module, "_require_runtime", lambda **kwargs: [])
    calls: list[dict[str, object]] = []

    class Learner:
        def update(self, payload):
            calls.append(payload)
            return {
                "checkpoint_path": str(updated_checkpoint),
                "updated_samples": 1,
            }

    factory_results = iter(
        (
            _ICLMRecordingGenerator(active_checkpoint, online_learner=Learner()),
            _ICLMRecordingGenerator(updated_checkpoint),
        )
    )
    servicer = module.ICLMServicer(
        generator=_ICLMRecordingGenerator(active_checkpoint),
        generator_factory=lambda _checkpoint_path: next(factory_results),
    )
    request = _valid_model_update_request(
        samples=[{"smiles": "CCO", "reward": 0.0, "outcome": "FAIL"}],
        teacher_embeddings=[[0.1]],
        kd_weight=0.5,
    )

    response = await servicer.UpdateModel(request, None)

    assert response.status == generator_pb2.MODEL_UPDATE_STATUS_APPLIED
    assert calls[0]["samples"] == [
        {
            "candidate_id": "candidate-1",
            "outcome": "FAIL",
            "reward": 0.0,
            "smiles": "CCO",
        }
    ]


def test_iclm_state_directory_fsync_failure_is_not_acknowledged(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_module(
        "iclm_state_directory_fsync_test",
        ROOT / "services/iclm-svc/src/iclm_svc/main.py",
    )
    state_path = tmp_path / "iclm-state.json"
    real_fsync = module.os.fsync
    fsync_calls = 0

    def fail_directory_fsync(descriptor: int) -> None:
        nonlocal fsync_calls
        fsync_calls += 1
        if fsync_calls == 2:
            raise OSError("state directory fsync failed")
        real_fsync(descriptor)

    arguments = {
        "checkpoint_path": str(tmp_path / "active"),
        "checkpoint_checksum": "sha256:" + "0" * 64,
        "active_version": "iclm-v1",
        "records": {},
    }
    with monkeypatch.context() as failure:
        failure.setattr(module.os, "fsync", fail_directory_fsync)
        with pytest.raises(OSError, match="state directory fsync failed"):
            module._write_service_state(state_path, **arguments)

    module._write_service_state(state_path, **arguments)
    assert json.loads(state_path.read_text(encoding="utf-8"))["schema_version"] == (
        "iclm-state.v3"
    )


@pytest.mark.asyncio
async def test_iclm_cancellation_waits_for_sync_learner_before_next_update(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_module(
        "iclm_cancelled_sync_update_test",
        ROOT / "services/iclm-svc/src/iclm_svc/main.py",
    )
    active_checkpoint, v2_checkpoint, v3_checkpoint = [
        tmp_path / name for name in ("active", "v2", "v3")
    ]
    for checkpoint in (active_checkpoint, v2_checkpoint, v3_checkpoint):
        checkpoint.write_bytes(checkpoint.name.encode())
    monkeypatch.setenv("ICLM_MODEL_PATH", str(active_checkpoint))
    monkeypatch.delenv("ICLM_UPDATE_COMMAND", raising=False)
    monkeypatch.setattr(module, "_require_runtime", lambda **kwargs: [])
    first_started = threading.Event()
    first_release = threading.Event()
    second_started = threading.Event()
    running_updates = 0
    max_running_updates = 0
    state_lock = threading.Lock()

    class BlockingLearner:
        def update(self, payload):
            nonlocal running_updates, max_running_updates
            with state_lock:
                running_updates += 1
                max_running_updates = max(max_running_updates, running_updates)
            first_started.set()
            first_release.wait(timeout=2)
            with state_lock:
                running_updates -= 1
            return {
                "checkpoint_path": str(v2_checkpoint),
                "updated_samples": 1,
            }

    class SecondLearner:
        def update(self, payload):
            nonlocal running_updates, max_running_updates
            with state_lock:
                running_updates += 1
                max_running_updates = max(max_running_updates, running_updates)
            second_started.set()
            with state_lock:
                running_updates -= 1
            return {
                "checkpoint_path": str(v3_checkpoint),
                "updated_samples": 1,
            }

    path_calls: dict[str, int] = {}

    def generator_factory(checkpoint_path: str):
        path_calls[checkpoint_path] = path_calls.get(checkpoint_path, 0) + 1
        if checkpoint_path == str(active_checkpoint):
            learner = BlockingLearner() if path_calls[checkpoint_path] == 1 else SecondLearner()
            return _ICLMRecordingGenerator(
                active_checkpoint,
                online_learner=learner,
            )
        if checkpoint_path == str(v2_checkpoint):
            if path_calls[checkpoint_path] == 1:
                return _ICLMRecordingGenerator(v2_checkpoint)
            return _ICLMRecordingGenerator(
                v2_checkpoint,
                online_learner=SecondLearner(),
            )
        return _ICLMRecordingGenerator(v3_checkpoint)

    servicer = module.ICLMServicer(
        generator=_ICLMRecordingGenerator(active_checkpoint),
        generator_factory=generator_factory,
    )
    first_request = _valid_model_update_request(
        samples=[{"smiles": "CCO"}],
        teacher_embeddings=[[0.1]],
        kd_weight=0.5,
        target_version="iclm-v2",
    )
    second_request = _valid_model_update_request(
        samples=[{"smiles": "CCN"}],
        teacher_embeddings=[[0.2]],
        kd_weight=0.5,
        target_version="iclm-v3",
    )
    second_request.request_id = "update-iclm-2"
    first_task = asyncio.create_task(servicer.UpdateModel(first_request, None))
    await asyncio.to_thread(first_started.wait, 1)
    first_task.cancel()
    second_task = asyncio.create_task(servicer.UpdateModel(second_request, None))

    try:
        await asyncio.sleep(0.05)
        assert second_started.is_set() is False
    finally:
        first_release.set()
    with pytest.raises(asyncio.CancelledError):
        await first_task
    with pytest.raises(RuntimeError, match="retryable update must complete"):
        await second_task

    assert max_running_updates == 1


@pytest.mark.asyncio
async def test_iclm_update_command_timeout_kills_descendant_processes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_module(
        "iclm_command_process_group_timeout_test",
        ROOT / "services/iclm-svc/src/iclm_svc/main.py",
    )
    active_checkpoint = tmp_path / "active"
    active_checkpoint.write_bytes(b"active")
    descendant_marker = tmp_path / "descendant-finished"
    runner = tmp_path / "spawn_descendant.py"
    child_code = (
        "import signal, time\n"
        "from pathlib import Path\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "time.sleep(0.25)\n"
        f"Path({str(descendant_marker)!r}).write_text('finished')\n"
    )
    runner.write_text(
        "import subprocess, sys, time\n"
        f"subprocess.Popen([sys.executable, '-c', {child_code!r}])\n"
        "time.sleep(5)\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("ICLM_MODEL_PATH", str(active_checkpoint))
    monkeypatch.setenv("ICLM_UPDATE_COMMAND", f"{sys.executable} {runner}")
    monkeypatch.setenv("ICLM_UPDATE_TIMEOUT_SECONDS", "0.05")
    monkeypatch.setattr(module, "_require_runtime", lambda **kwargs: [])
    servicer = module.ICLMServicer(
        generator=_ICLMRecordingGenerator(active_checkpoint),
        generator_factory=lambda checkpoint_path: _ICLMRecordingGenerator(Path(checkpoint_path)),
    )

    with pytest.raises(RuntimeError, match="execution failed"):
        await servicer.UpdateModel(
            _valid_model_update_request(
                samples=[{"smiles": "CCO"}],
                teacher_embeddings=[[0.1]],
                kd_weight=0.5,
            ),
            None,
        )
    await asyncio.sleep(0.35)

    assert descendant_marker.exists() is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "command_outcome",
    ("success", "nonzero", "invalid_json"),
)
async def test_iclm_update_command_all_exits_kill_descendant_processes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    command_outcome: str,
) -> None:
    module = _load_module(
        f"iclm_command_process_group_{command_outcome}_test",
        ROOT / "services/iclm-svc/src/iclm_svc/main.py",
    )
    active_checkpoint = tmp_path / "active"
    active_checkpoint.write_bytes(b"active")
    new_checkpoint = tmp_path / "new"
    new_checkpoint.write_bytes(b"new")
    descendant_marker = tmp_path / "descendant-finished"
    runner = tmp_path / "spawn_descendant.py"
    child_code = (
        "import signal, time\n"
        "from pathlib import Path\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "time.sleep(0.25)\n"
        f"Path({str(descendant_marker)!r}).write_text('finished')\n"
    )
    outcome_code = {
        "success": (
            "print(json.dumps({"
            f"'checkpoint_path': {str(new_checkpoint)!r}, "
            "'updated_samples': 1"
            "}))\n"
        ),
        "nonzero": "raise SystemExit(3)\n",
        "invalid_json": "print('not-json')\n",
    }[command_outcome]
    runner.write_text(
        "import json, subprocess, sys\n"
        "subprocess.Popen("
        f"[sys.executable, '-c', {child_code!r}], "
        "stdin=subprocess.DEVNULL, "
        "stdout=subprocess.DEVNULL, "
        "stderr=subprocess.DEVNULL"
        ")\n"
        f"{outcome_code}",
        encoding="utf-8",
    )
    monkeypatch.setenv("ICLM_MODEL_PATH", str(active_checkpoint))
    monkeypatch.setenv("ICLM_UPDATE_COMMAND", f"{sys.executable} {runner}")
    monkeypatch.setenv("ICLM_UPDATE_TIMEOUT_SECONDS", "1")
    monkeypatch.setattr(module, "_require_runtime", lambda **kwargs: [])
    servicer = module.ICLMServicer(
        generator=_ICLMRecordingGenerator(active_checkpoint),
        generator_factory=lambda checkpoint_path: _ICLMRecordingGenerator(Path(checkpoint_path)),
    )
    update = servicer.UpdateModel(
        _valid_model_update_request(
            samples=[{"smiles": "CCO"}],
            teacher_embeddings=[[0.1]],
            kd_weight=0.5,
        ),
        None,
    )

    if command_outcome == "success":
        response = await update
        assert response.acknowledged is True
    else:
        with pytest.raises(RuntimeError):
            await update
    await asyncio.sleep(0.35)

    assert descendant_marker.exists() is False


@pytest.mark.asyncio
async def test_iclm_update_command_cancellation_kills_descendant_processes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_module(
        "iclm_command_process_group_cancel_test",
        ROOT / "services/iclm-svc/src/iclm_svc/main.py",
    )
    active_checkpoint = tmp_path / "active"
    active_checkpoint.write_bytes(b"active")
    started_marker = tmp_path / "started"
    descendant_marker = tmp_path / "descendant-finished"
    runner = tmp_path / "spawn_descendant.py"
    child_code = (
        "import signal, time\n"
        "from pathlib import Path\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "time.sleep(0.25)\n"
        f"Path({str(descendant_marker)!r}).write_text('finished')\n"
    )
    runner.write_text(
        "import signal, subprocess, sys, time\n"
        "from pathlib import Path\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        f"Path({str(started_marker)!r}).write_text('started')\n"
        f"subprocess.Popen([sys.executable, '-c', {child_code!r}])\n"
        "time.sleep(5)\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("ICLM_MODEL_PATH", str(active_checkpoint))
    monkeypatch.setenv("ICLM_UPDATE_COMMAND", f"{sys.executable} {runner}")
    monkeypatch.setenv("ICLM_UPDATE_TIMEOUT_SECONDS", "10")
    monkeypatch.setattr(module, "_require_runtime", lambda **kwargs: [])
    servicer = module.ICLMServicer(
        generator=_ICLMRecordingGenerator(active_checkpoint),
        generator_factory=lambda checkpoint_path: _ICLMRecordingGenerator(Path(checkpoint_path)),
    )
    update_task = asyncio.create_task(
        servicer.UpdateModel(
            _valid_model_update_request(
                samples=[{"smiles": "CCO"}],
                teacher_embeddings=[[0.1]],
                kd_weight=0.5,
            ),
            None,
        )
    )
    for _ in range(100):
        if started_marker.exists():
            break
        await asyncio.sleep(0.01)
    assert started_marker.exists()
    update_task.cancel()
    await asyncio.sleep(0.02)
    update_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await update_task
    await asyncio.sleep(0.35)

    assert descendant_marker.exists() is False


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
            "workflow_scope": "state_only",
            "validation_passed": False,
            "max_refinements": 0,
        }
    )
    status = await module.get_design_status(started["design_id"])

    assert started["status"] == "completed"
    assert started["run_id"] == "run-orch-1"
    assert started["trace_id"] == "trace-orch-1"
    assert started["artifact_ids"] == ["artifact-seed-1"]
    assert started["history"] == ["PLANNING"]
    assert status["current_stage"] == "planning"
    assert status["run_id"] == "run-orch-1"
    assert status["trace_id"] == "trace-orch-1"
    assert status["artifact_ids"] == ["artifact-seed-1"]
    assert status["history"] == ["PLANNING"]
    assert status["state"]["history"] == ["PLANNING"]
    assert "molecules_generated" not in status["state"]
    with pytest.raises(HTTPException) as pause_error:
        await module.pause_design(started["design_id"])
    assert pause_error.value.status_code == 409
    with pytest.raises(HTTPException) as resume_error:
        await module.resume_design(started["design_id"])
    assert resume_error.value.status_code == 409

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
            "validation_passed": True,
            "max_refinements": 1,
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
        "CRITIC",
    ]
    assert started["state"]["cig"]["source"] == "Design KRAS G12C inhibitor"
    assert started["state"]["candidates"][0]["canonical_smiles"] == "CCO"
    assert started["state"]["validation"]["passed"] is True
    assert "retrosyn" not in started["state"]
    assert started["state"]["critic"]["verdict"] == "pass"


@pytest.mark.asyncio
async def test_orchestrator_default_clients_receive_shared_agent_request_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module(
        "orchestrator_default_agent_request_client_test",
        ROOT / "services/orchestrator-svc/src/orchestrator_svc/main.py",
    )
    shared_client = object()
    module._AGENT_REQUEST_CLIENT = shared_client
    module._AGENT_RUNTIME_LOOP = asyncio.get_running_loop()
    built_clients: list[object] = []

    class Compiled:
        async def ainvoke(self, state):
            return state

    class Graph:
        def __init__(self, clients, workflow_scope):
            built_clients.append(clients)

        def build(self):
            return Compiled()

    monkeypatch.setattr(module, "WorkflowGraph", Graph)
    state = {
        "run_id": "run-default-client",
        "trace_id": "trace-default-client",
    }
    request = {
        "workflow_scope": "engineering",
        "validation_passed": True,
        "max_refinements": 0,
    }

    await module._invoke_workflow(request, state)

    assert len(built_clients) == 1
    assert isinstance(built_clients[0], module.EngineeringWorkflowClients)
    assert built_clients[0].request_client is shared_client


@pytest.mark.asyncio
async def test_legacy_gateway_seeds_drive_engineering_generation_and_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from api_gateway.routers import design as gateway_module

    orchestrator_module = _load_module(
        "orchestrator_legacy_seed_engineering_adapter_test",
        ROOT / "services/orchestrator-svc/src/orchestrator_svc/main.py",
    )
    from mf_chem.predict import engine as predict_engine_module
    from mf_oracles.rdkit_oracle import oracle as rdkit_oracle_module

    oracle_inputs: list[list[str]] = []

    class RecordingOracle:
        async def evaluate(
            self,
            smiles: list[str],
            properties: list[str],
        ) -> dict[str, dict[str, float]]:
            oracle_inputs.append(list(smiles))
            return {item: {"admet_score": 0.8} for item in dict.fromkeys(smiles)}

    class Prediction:
        def __init__(self, smiles: str) -> None:
            self.smiles = smiles

        def to_dict(self) -> dict:
            properties = (
                {
                    "qed": 0.8,
                    "sa_score": 2.0,
                    "logp": 2.5,
                    "composite_score": 0.7,
                }
                if self.smiles == "CCO"
                else {
                    "qed": 0.7,
                    "sa_score": 3.0,
                    "logp": 3.0,
                    "composite_score": 0.9,
                }
            )
            return {
                "smiles": self.smiles,
                "canonical_smiles": self.smiles,
                "valid": True,
                "admet": {},
                **properties,
            }

    class Predictor:
        def __init__(self, device_ids: list[int]) -> None:
            self.device_ids = device_ids

        def predict_one(self, smiles: str) -> Prediction:
            return Prediction(smiles)

    original_import = builtins.__import__

    def reject_random_generator_import(
        name: str,
        globals: dict | None = None,
        locals: dict | None = None,
        fromlist: tuple = (),
        level: int = 0,
    ):
        if name == "mf_generators.rdkit_random":
            raise AssertionError("caller seed_smiles must bypass random generation")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", reject_random_generator_import)
    monkeypatch.setattr(rdkit_oracle_module, "RDKitOracle", RecordingOracle)
    monkeypatch.setattr(predict_engine_module, "MolPredictEngine", Predictor)
    legacy_request = gateway_module._canonical_design_request(
        {
            "seed_smiles": ["CCO", "CCN"],
            "n_samples": 5,
            "seed": 17,
        }
    )
    clients = orchestrator_module.EngineeringWorkflowClients()

    candidates = await clients.generate_candidates({"request": legacy_request})
    validation = await clients.validate_candidates(
        {
            "candidates": candidates,
            "request": legacy_request,
        }
    )

    expected_smiles = ["CCN", "CCO", "CCN", "CCO", "CCN"]
    assert [candidate["canonical_smiles"] for candidate in candidates] == expected_smiles
    assert oracle_inputs == [["CCN", "CCO"]]
    assert validation["passed"] is True
    assert [row["smiles"] for row in validation["results"]] == [
        "CCN",
        "CCN",
        "CCN",
        "CCO",
        "CCO",
    ]
    assert [row["candidate_id"] for row in validation["results"]] == [
        "candidate-1",
        "candidate-3",
        "candidate-5",
        "candidate-2",
        "candidate-4",
    ]
    assert [row["rank"] for row in validation["results"]] == [1, 2, 3, 4, 5]
    assert [row["pareto_optimal"] for row in validation["results"]] == [
        False,
        False,
        False,
        True,
        True,
    ]


@pytest.mark.asyncio
async def test_legacy_gateway_repeated_seeds_keep_real_oracle_result_per_occurrence() -> None:
    from api_gateway.routers import design as gateway_module

    orchestrator_module = _load_module(
        "orchestrator_legacy_repeated_seed_real_oracle_test",
        ROOT / "services/orchestrator-svc/src/orchestrator_svc/main.py",
    )
    legacy_request = gateway_module._canonical_design_request(
        {
            "seed_smiles": ["CCO", "CCN"],
            "n_samples": 5,
            "seed": 0,
        }
    )
    clients = orchestrator_module.EngineeringWorkflowClients()

    candidates = await clients.generate_candidates({"request": legacy_request})
    validation = await clients.validate_candidates(
        {
            "candidates": candidates,
            "request": legacy_request,
        }
    )

    candidate_smiles = [candidate["canonical_smiles"] for candidate in candidates]
    result_smiles = [row["smiles"] for row in validation["results"]]
    assert candidate_smiles == ["CCO", "CCN", "CCO", "CCN", "CCO"]
    assert [candidate["candidate_id"] for candidate in candidates] == [
        "candidate-1",
        "candidate-2",
        "candidate-3",
        "candidate-4",
        "candidate-5",
    ]
    assert len(validation["results"]) == len(candidates) == 5
    assert sorted(result_smiles) == sorted(candidate_smiles)
    assert [row["candidate_id"] for row in validation["results"] if row["smiles"] == "CCO"] == [
        "candidate-1",
        "candidate-3",
        "candidate-5",
    ]
    assert [row["candidate_id"] for row in validation["results"] if row["smiles"] == "CCN"] == [
        "candidate-2",
        "candidate-4",
    ]
    assert [row["rank"] for row in validation["results"]] == [1, 2, 3, 4, 5]
    assert all(type(row["pareto_optimal"]) is bool for row in validation["results"])


@pytest.mark.asyncio
async def test_legacy_gateway_empty_seed_completes_without_engineering_result() -> None:
    from api_gateway.routers import design as gateway_module

    module = _load_module(
        "orchestrator_legacy_empty_seed_test",
        ROOT / "services/orchestrator-svc/src/orchestrator_svc/main.py",
    )
    request = gateway_module._canonical_design_request(
        {
            "seed_smiles": [""],
            "n_samples": 1,
            "seed": 0,
        }
    )
    request.update(
        {
            "run_id": "run-legacy-empty-seed",
            "trace_id": "trace-legacy-empty-seed",
            "clients": module.EngineeringWorkflowClients(),
        }
    )
    assert request["_mforge_internal_legacy_design_request"] is True

    response = await module.start_design(request)
    persisted = await module.get_design_status(response["run_id"])

    assert response["status"] == "completed"
    assert response["state"]["validation_passed"] is False
    assert response["state"]["validation"] == {
        "passed": False,
        "threshold": 0.0,
        "results": [],
        "reason": "no valid candidates",
    }
    assert persisted["status"] == "completed"
    assert persisted["state"]["validation"]["results"] == []
    assert "_mforge_internal_legacy_design_request" not in response["state"]["request"]
    assert "_mforge_internal_legacy_design_request" not in persisted["state"]["request"]


@pytest.mark.asyncio
async def test_legacy_gateway_empty_seed_direct_injection_completes_without_marker() -> None:
    from api_gateway.routers import design as gateway_module

    module = _load_module(
        "orchestrator_legacy_empty_seed_background_test",
        ROOT / "services/orchestrator-svc/src/orchestrator_svc/main.py",
    )
    request = gateway_module._canonical_design_request(
        {
            "seed_smiles": [""],
            "n_samples": 1,
            "seed": 0,
        }
    )
    request.update(
        {
            "run_id": "run-legacy-empty-seed-background",
            "trace_id": "trace-legacy-empty-seed-background",
            "clients": module.EngineeringWorkflowClients(),
        }
    )

    completed = await module.start_design(request)
    persisted = await module.get_design_status(completed["run_id"])

    assert completed["status"] == "completed"
    assert persisted["status"] == "completed"
    assert persisted["state"]["validation_passed"] is False
    assert persisted["state"]["validation"] == {
        "passed": False,
        "threshold": 0.0,
        "results": [],
        "reason": "no valid candidates",
    }
    assert "_mforge_internal_legacy_design_request" not in persisted["state"]["request"]


@pytest.mark.asyncio
async def test_canonical_engineering_empty_seed_remains_rejected() -> None:
    module = _load_module(
        "orchestrator_canonical_empty_seed_test",
        ROOT / "services/orchestrator-svc/src/orchestrator_svc/main.py",
    )

    response = await module.start_design(
        {
            "nl_input": "Design a soluble molecule",
            "workflow_scope": "engineering",
            "validation_passed": True,
            "max_refinements": 0,
            "seed_smiles": [""],
            "n_samples": 1,
            "seed": 0,
            "run_id": "run-canonical-empty-seed",
            "trace_id": "trace-canonical-empty-seed",
            "clients": module.EngineeringWorkflowClients(),
        }
    )

    assert response["status"] == "rejected"
    assert response["state"]["validation_passed"] is False
    assert response["state"]["validation"] == {
        "passed": False,
        "threshold": 0.0,
        "results": [],
        "reason": "no valid candidates",
    }


@pytest.mark.asyncio
async def test_engineering_validation_failure_without_no_valid_reason_stays_rejected() -> None:
    module = _load_module(
        "orchestrator_engineering_other_validation_failure_test",
        ROOT / "services/orchestrator-svc/src/orchestrator_svc/main.py",
    )

    class Clients:
        async def compile_intent(self, state: dict) -> dict:
            return {"cig": {}, "hciv": {}, "intent_cone": {}}

        async def generate_candidates(self, state: dict) -> list[dict]:
            return [{"candidate_id": "candidate-1", "canonical_smiles": "CCO"}]

        async def validate_candidates(self, state: dict) -> dict:
            return {
                "passed": False,
                "results": [],
                "reason": "quality gate failed",
            }

    response = await module.start_design(
        {
            "nl_input": "Design KRAS G12C inhibitor",
            "workflow_scope": "engineering",
            "validation_passed": True,
            "max_refinements": 0,
            "run_id": "run-engineering-other-validation-failure",
            "trace_id": "trace-engineering-other-validation-failure",
            "clients": Clients(),
        }
    )

    assert response["status"] == "rejected"
    assert response["state"]["validation_passed"] is False
    assert response["state"]["validation"]["reason"] == "quality gate failed"


@pytest.mark.asyncio
async def test_legacy_gateway_mixed_empty_seed_keeps_only_real_engineering_outcomes() -> None:
    from api_gateway.routers import design as gateway_module

    module = _load_module(
        "orchestrator_legacy_mixed_empty_seed_test",
        ROOT / "services/orchestrator-svc/src/orchestrator_svc/main.py",
    )
    request = gateway_module._canonical_design_request(
        {
            "seed_smiles": ["", "CCO"],
            "n_samples": 4,
            "seed": 0,
        }
    )
    clients = module.EngineeringWorkflowClients()

    candidates = await clients.generate_candidates({"request": request})
    validation = await clients.validate_candidates(
        {
            "candidates": candidates,
            "request": request,
        }
    )

    assert [candidate.get("smiles") for candidate in candidates] == ["", "CCO", "", "CCO"]
    assert [candidate["candidate_id"] for candidate in candidates] == [
        "candidate-1",
        "candidate-2",
        "candidate-3",
        "candidate-4",
    ]
    assert validation["passed"] is True
    assert [row["smiles"] for row in validation["results"]] == ["CCO", "CCO"]
    assert [row["candidate_id"] for row in validation["results"]] == [
        "candidate-2",
        "candidate-4",
    ]
    assert all(row["valid"] is True for row in validation["results"])


@pytest.mark.asyncio
async def test_engineering_generation_without_seeds_keeps_random_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module(
        "orchestrator_engineering_random_fallback_test",
        ROOT / "services/orchestrator-svc/src/orchestrator_svc/main.py",
    )
    from mf_generators import rdkit_random as rdkit_random_module

    calls: list[dict[str, object]] = []

    class GeneratedMolecule:
        def __init__(self, index: int) -> None:
            self.index = index

        def model_dump(self, mode: str) -> dict:
            return {
                "smiles": f"CC{'C' * self.index}",
                "canonical_smiles": f"CC{'C' * self.index}",
            }

    class RecordingGenerator:
        def __init__(self, seed: int) -> None:
            calls.append({"init_seed": seed})

        async def generate(
            self,
            hciv: object,
            cone: object,
            cig: object,
            *,
            n_samples: int,
            seed: int,
        ):
            calls.append(
                {
                    "hciv": hciv,
                    "cone": cone,
                    "cig": cig,
                    "n_samples": n_samples,
                    "seed": seed,
                }
            )
            for index in range(n_samples):
                yield GeneratedMolecule(index)

    monkeypatch.setattr(rdkit_random_module, "RDKitRandomGenerator", RecordingGenerator)

    candidates = await module.EngineeringWorkflowClients().generate_candidates(
        {
            "hciv": {"source": "hciv"},
            "intent_cone": {"source": "cone"},
            "cig": {"source": "cig"},
            "request": {"n_samples": 2},
        }
    )

    assert calls == [
        {"init_seed": 42},
        {
            "hciv": {"source": "hciv"},
            "cone": {"source": "cone"},
            "cig": {"source": "cig"},
            "n_samples": 2,
            "seed": 42,
        },
    ]
    assert [
        (candidate["candidate_id"], candidate["canonical_smiles"]) for candidate in candidates
    ] == [
        ("candidate-1", "CC"),
        ("candidate-2", "CCC"),
    ]


@pytest.mark.asyncio
async def test_orchestrator_injected_clients_do_not_initialize_agent_control(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module(
        "orchestrator_injected_clients_no_redis_test",
        ROOT / "services/orchestrator-svc/src/orchestrator_svc/main.py",
    )
    injected_clients = object()
    built_clients: list[object] = []

    class Compiled:
        async def ainvoke(self, state):
            return state

    class Graph:
        def __init__(self, clients, workflow_scope):
            built_clients.append(clients)

        def build(self):
            return Compiled()

    def reject_redis(*args, **kwargs):
        raise AssertionError("explicit workflow clients must not initialize Redis")

    monkeypatch.setattr(module, "WorkflowGraph", Graph)
    monkeypatch.setattr(module, "RedisBus", reject_redis, raising=False)

    await module._invoke_workflow(
        {
            "workflow_scope": "engineering",
            "validation_passed": True,
            "max_refinements": 0,
            "clients": injected_clients,
        },
        {
            "run_id": "run-injected-client",
            "trace_id": "trace-injected-client",
        },
    )

    assert built_clients == [injected_clients]


@pytest.mark.asyncio
async def test_orchestrator_agent_boundaries_emit_canonical_correlated_requests() -> None:
    module = _load_module(
        "orchestrator_agent_boundary_contract_test",
        ROOT / "services/orchestrator-svc/src/orchestrator_svc/main.py",
    )

    class RecordingRequestClient:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        async def request(self, subject, payload, *, payload_type_url, timeout):
            call = {
                "subject": subject,
                "payload": dict(payload),
                "payload_type_url": payload_type_url,
                "timeout": timeout,
            }
            self.calls.append(call)
            if payload.get("action") == "generator_coord/feedback/v1":
                response = _feedback_ack(payload)
                return {
                    **response,
                    "run_id": payload["run_id"],
                    "request_id": payload["request_id"],
                    "schema_version": payload["schema_version"],
                }
            if subject == "agent.validation.request":
                records = [
                    _full_validation_record(
                        candidate_id=candidate["candidate_id"],
                        smiles=candidate["canonical_smiles"],
                    )
                    for candidate in payload["candidates"]
                ]
                response = _validation_batch_response(
                    payload,
                    records,
                    outcome="PASS",
                )
                return {
                    **response,
                    "run_id": payload["run_id"],
                    "request_id": payload["request_id"],
                    "schema_version": payload["schema_version"],
                }
            responses = {
                "agent.generator_coord.request": {
                    "status": "dispatched",
                    "candidates": [
                        {
                            "candidate_id": "candidate-generated",
                            "smiles": "CCN",
                            "generator_name": "hfm_3d",
                        }
                    ],
                },
                "agent.retrosyn.request": {
                    "status": "planned",
                    "routes": [
                        {
                            "route_id": "route-1",
                            "building_blocks": [{"smiles": "CC"}],
                        }
                    ],
                },
                "agent.supply.request": {
                    "status": "assessed",
                    "route_id": payload.get("route_id"),
                    "supply_assessment": {"overall_feasibility": "available"},
                },
                "agent.srb.request": {
                    "status": "compiled",
                    "route_id": payload.get("route_id"),
                    "protocols": [
                        {
                            "ssp_id": "ssp-1",
                            "route_id": payload.get("route_id"),
                        }
                    ],
                },
                "agent.critic.request": {
                    "verdict": "pass",
                    "total_rules": 1,
                },
            }
            return {
                **responses[subject],
                **{
                    field: payload[field]
                    for field in (
                        "project_id",
                        "candidate_id",
                        "candidate_index",
                        "canonical_smiles",
                    )
                    if field in payload
                },
                "run_id": payload["run_id"],
                "request_id": payload["request_id"],
                "schema_version": payload["schema_version"],
            }

    client = RecordingRequestClient()
    full_clients = module.FullWorkflowClients(request_client=client)
    engineering_clients = module.EngineeringWorkflowClients(request_client=client)
    spoofed_correlation = {
        "trace_id": "client-trace",
        "parent_id": "client-parent",
        "run_id": "client-run",
        "request_id": "client-request",
        "schema_version": "client-schema",
    }

    generated = await full_clients.generate_candidates(
        {
            "run_id": "run-boundary",
            "trace_id": "trace-boundary",
            "refinement_count": 2,
            "request": {
                **spoofed_correlation,
                "project_id": "project-1",
                "generation_strategy": "auto",
                "n_samples": 2,
                "seed": 11,
            },
        }
    )
    validated = await full_clients.validate_candidates(
        {
            "run_id": "run-boundary",
            "trace_id": "trace-boundary",
            "refinement_count": 2,
            "candidates": [
                _full_candidate(candidate_id="candidate-cco", smiles="CCO"),
                _full_candidate(candidate_id="candidate-ccn", smiles="CCN"),
            ],
            "request": {
                **spoofed_correlation,
                "project_id": "project-1",
                **_full_policy_payload(),
            },
        }
    )
    selected_state = _full_selected_state()
    selected_state.update(
        {
            "run_id": "run-boundary",
            "trace_id": "trace-boundary",
            "refinement_count": 2,
        }
    )
    selected_state["request"].update(
        {
            **spoofed_correlation,
            "project_id": "project-1",
            "retrosyn_max_routes": 2,
        }
    )
    retrosyn = await full_clients.plan_routes(selected_state)
    route = retrosyn["routes"][0]
    selected_state["retrosyn"] = {"routes": [route]}
    supplied = await full_clients.assess_supply(selected_state)
    selected_state["supply"] = supplied
    synthesised = await full_clients.compile_synthesis(selected_state)
    reviewed = await engineering_clients.review_candidates(
        {
            "run_id": "run-boundary",
            "trace_id": "trace-boundary",
            "refinement_count": 2,
            "candidates": [{"canonical_smiles": "CCO"}],
            "validation": {
                "results": [
                    {
                        "smiles": "CCO",
                        "admet_score": 0.8,
                        "_critic_blocking_rule_ids": [],
                    }
                ],
            },
            "request": spoofed_correlation,
        }
    )

    assert generated == [
        {
            "candidate_id": "candidate-generated",
            "smiles": "CCN",
            "canonical_smiles": "CCN",
            "generator_name": "hfm_3d",
        }
    ]
    assert validated["passed"] is True
    assert [row["canonical_smiles"] for row in validated["results"]] == ["CCO", "CCN"]
    assert retrosyn == {
        "status": "planned",
        "routes": [
            {
                "route_id": "route-1",
                "building_blocks": [{"smiles": "CC"}],
            }
        ],
        "project_id": "project-1",
        "candidate_id": "candidate-1",
        "candidate_index": 0,
        "canonical_smiles": "CCO",
    }
    assert supplied == {
        "status": "assessed",
        "route_id": "route-1",
        "supply_assessment": {"overall_feasibility": "available"},
        "route_assessments": [
            {
                "route_id": "route-1",
                "status": "assessed",
                "supply_assessment": {"overall_feasibility": "available"},
            }
        ],
        "project_id": "project-1",
        "candidate_id": "candidate-1",
        "candidate_index": 0,
        "canonical_smiles": "CCO",
    }
    assert synthesised == {
        "status": "compiled",
        "route_id": "route-1",
        "protocols": [{"ssp_id": "ssp-1", "route_id": "route-1"}],
        "project_id": "project-1",
        "candidate_id": "candidate-1",
        "candidate_index": 0,
        "canonical_smiles": "CCO",
    }
    assert reviewed == {"verdict": "pass", "total_rules": 1}

    assert [call["subject"] for call in client.calls] == [
        "agent.generator_coord.request",
        "agent.validation.request",
        "agent.generator_coord.request",
        "agent.retrosyn.request",
        "agent.supply.request",
        "agent.srb.request",
        "agent.critic.request",
    ]
    assert [call["payload_type_url"] for call in client.calls] == [
        "type.moleculeforge.ai/agent/generator_coord/request.v1",
        "type.moleculeforge.ai/agent/validation/request.v1",
        "type.moleculeforge.ai/agent/generator_coord/request.v1",
        "type.moleculeforge.ai/agent/retrosyn/request.v1",
        "type.moleculeforge.ai/agent/supply/request.v1",
        "type.moleculeforge.ai/agent/srb/request.v1",
        "type.moleculeforge.ai/agent/critic/request.v1",
    ]
    assert [call["payload"]["schema_version"] for call in client.calls] == [
        "generator_coord.request.v1",
        "validation.request.v1",
        "generator_coord.request.v1",
        "retrosyn.request.v1",
        "supply.request.v1",
        "srb.request.v1",
        "critic.request.v1",
    ]
    assert [call["payload"]["request_id"] for call in client.calls] == [
        "run-boundary:generator_coord:2",
        "run-boundary:validation:2",
        "run-boundary:generator_coord_feedback:2",
        "run-boundary:retrosyn:2:candidate-0",
        "run-boundary:supply:2:candidate-0:route-route-1",
        "run-boundary:srb:2:candidate-0",
        "run-boundary:critic:2",
    ]
    assert [call["payload"]["parent_id"] for call in client.calls] == [
        "run-boundary:generating:2",
        "run-boundary:validating:2",
        "run-boundary:validating:2",
        "run-boundary:retrosyn:2",
        "run-boundary:supply:2",
        "run-boundary:srb:2",
        "run-boundary:critic:2",
    ]
    assert all(call["payload"]["trace_id"] == "trace-boundary" for call in client.calls)
    assert all(call["payload"]["run_id"] == "run-boundary" for call in client.calls)
    assert [call["timeout"] for call in client.calls] == [
        60.0,
        360.0,
        900.0,
        60.0,
        60.0,
        60.0,
        60.0,
    ]
    correlation_fields = {
        "trace_id",
        "parent_id",
        "run_id",
        "request_id",
        "schema_version",
    }
    assert [
        {key: value for key, value in call["payload"].items() if key not in correlation_fields}
        for call in client.calls
    ] == [
        {
            "project_id": "project-1",
            "generation_strategy": "auto",
            "objectives": {},
            "cig": None,
            "hciv": None,
            "intent_cone": None,
            "n_samples": 2,
            "batch_size": 2,
            "generator_params": {"sampling_seed": 11},
        },
        {
            "project_id": "project-1",
            "validation_policy": _full_policy_payload()["validation_policy"],
            "teacher_policy": _full_policy_payload()["teacher_policy"],
            "selection_policy": _full_policy_payload()["selection_policy"],
            "candidates": [
                _full_candidate(candidate_id="candidate-cco", smiles="CCO"),
                _full_candidate(candidate_id="candidate-ccn", smiles="CCN"),
            ],
            "external_evidence": None,
        },
        {
            "action": "generator_coord/feedback/v1",
            "route_request_id": "run-boundary:generator_coord:2",
            "iteration": 2,
            "groups": [
                {
                    "phase": "validation",
                    "generator_name": "hfm_3d",
                    "canonical_smiles": "CCO",
                    "candidate_ids": ["candidate-cco"],
                    "evidence_ids": ["evidence-candidate-cco"],
                    "records": [
                        {
                            **_full_validation_record(
                                candidate_id="candidate-cco",
                                smiles="CCO",
                            ),
                            "passed": True,
                        }
                    ],
                    "teacher_policy": _full_policy_payload()["teacher_policy"],
                },
                {
                    "phase": "validation",
                    "generator_name": "hfm_3d",
                    "canonical_smiles": "CCN",
                    "candidate_ids": ["candidate-ccn"],
                    "evidence_ids": ["evidence-candidate-ccn"],
                    "records": [
                        {
                            **_full_validation_record(
                                candidate_id="candidate-ccn",
                                smiles="CCN",
                            ),
                            "passed": True,
                        }
                    ],
                    "teacher_policy": _full_policy_payload()["teacher_policy"],
                },
            ],
        },
        {
            "project_id": "project-1",
            "smiles": "CCO",
            "candidate_id": "candidate-1",
            "candidate_index": 0,
            "canonical_smiles": "CCO",
            "engine": "rsgpt",
            "max_routes": 2,
        },
        {
            "project_id": "project-1",
            "smiles": "CCO",
            "candidate_id": "candidate-1",
            "candidate_index": 0,
            "canonical_smiles": "CCO",
            "workflow_scope": "full",
            "route_id": "route-1",
            "building_blocks": [{"smiles": "CC"}],
        },
        {
            "project_id": "project-1",
            "candidate_id": "candidate-1",
            "candidate_index": 0,
            "canonical_smiles": "CCO",
            "workflow_scope": "full",
            "route_id": "route-1",
            "molecule": {"smiles": "CCO"},
            "pathways": [route],
        },
        {
            "smiles": "CCO",
            "properties": {"smiles": "CCO", "admet_score": 0.8},
        },
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("missing_field", ("run_id", "trace_id"))
async def test_orchestrator_agent_request_rejects_missing_workflow_correlation(
    missing_field: str,
) -> None:
    module = _load_module(
        f"orchestrator_missing_{missing_field}_test",
        ROOT / "services/orchestrator-svc/src/orchestrator_svc/main.py",
    )

    class RejectingClient:
        async def request(self, *args, **kwargs):
            raise AssertionError("invalid workflow correlation must fail before request")

    state = {
        "run_id": "run-correlation",
        "trace_id": "trace-correlation",
        "candidates": [{"canonical_smiles": "CCO"}],
        "validation": {"results": [{"smiles": "CCO"}]},
        "request": {},
    }
    state.pop(missing_field)

    with pytest.raises(ValueError, match=f"{missing_field} is required"):
        await module.EngineeringWorkflowClients(request_client=RejectingClient()).review_candidates(
            state
        )


@pytest.mark.asyncio
async def test_orchestrator_agent_request_timeout_must_be_positive() -> None:
    module = _load_module(
        "orchestrator_agent_request_timeout_test",
        ROOT / "services/orchestrator-svc/src/orchestrator_svc/main.py",
    )

    class RejectingClient:
        async def request(self, *args, **kwargs):
            raise AssertionError("invalid timeout must fail before request")

    with pytest.raises(
        ValueError,
        match="agent_request_timeout_seconds must be positive",
    ):
        await module.EngineeringWorkflowClients(request_client=RejectingClient()).review_candidates(
            {
                "run_id": "run-timeout",
                "trace_id": "trace-timeout",
                "candidates": [{"canonical_smiles": "CCO"}],
                "validation": {"results": [{"smiles": "CCO"}]},
                "request": {"agent_request_timeout_seconds": 0},
            }
        )


@pytest.mark.asyncio
async def test_orchestrator_validation_request_timeout_covers_oracle_rounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VALIDATION_ORACLE_TIMEOUT_SECONDS", "7")
    module = _load_module(
        "orchestrator_validation_request_timeout_budget_test",
        ROOT / "services/orchestrator-svc/src/orchestrator_svc/main.py",
    )
    policies = _full_policy_payload()

    def respond(_subject: str, payload: dict) -> dict:
        if payload.get("action") == "generator_coord/feedback/v1":
            return _feedback_ack(payload)
        return _validation_batch_response(
            payload,
            [_full_validation_record()],
            outcome="PASS",
        )

    request_client = _AgentRequestClientStub(respond)

    await module.FullWorkflowClients(request_client=request_client).validate_candidates(
        {
            "run_id": "run-timeout-budget",
            "trace_id": "trace-timeout-budget",
            "candidates": [_full_candidate()],
            "request": {"project_id": "project-1", **policies},
        }
    )

    assert request_client.calls[0]["timeout"] == 67.0


def test_orchestrator_validation_timeout_covers_fep_chunk_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VALIDATION_ORACLE_TIMEOUT_SECONDS", "7")
    monkeypatch.setenv("OPENFE_QUICKRUN_TIMEOUT_SECONDS", "10")
    monkeypatch.setenv("OPENFE_GATHER_TIMEOUT_SECONDS", "2")
    monkeypatch.setenv("OPENFE_MAX_TRANSFORMATIONS_PER_PAIR", "2")
    module = _load_module(
        "orchestrator_fep_timeout_budget_test",
        ROOT / "services/orchestrator-svc/src/orchestrator_svc/main.py",
    )
    payload = {
        "candidates": [
            {"candidate_id": "candidate-1", "canonical_smiles": "CCO"},
            {"candidate_id": "candidate-2", "canonical_smiles": "CCN"},
            {"candidate_id": "candidate-3", "canonical_smiles": "CCC"},
        ],
        "validation_policy": {
            "oracle_level": 3,
            "batch_size": 2,
            "max_concurrency": 1,
            "thresholds": [],
            "oracle_inputs": {
                "fep": {
                    "oracle_parameters": {
                        "method": "openfe",
                        "n_repeats": 3,
                    }
                }
            },
        },
    }

    timeout = module.agent_request_timeout_seconds({}, "validation", payload)

    assert timeout == 546.0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("is_redis", "roundtrip_result", "expected_error"),
    [
        (False, True, "production Orchestrator Agent control requires Redis"),
        (True, False, "Redis roundtrip failed"),
    ],
)
async def test_orchestrator_agent_control_rejects_unready_bus(
    monkeypatch: pytest.MonkeyPatch,
    agent_message_hmac_secret: None,
    is_redis: bool,
    roundtrip_result: bool,
    expected_error: str,
) -> None:
    module = _load_module(
        f"orchestrator_agent_control_unready_{is_redis}_{roundtrip_result}",
        ROOT / "services/orchestrator-svc/src/orchestrator_svc/main.py",
    )
    buses: list[object] = []

    class Bus:
        def __init__(self, *, allow_fallback):
            assert allow_fallback is False
            self.is_redis = is_redis
            self.closed = False
            buses.append(self)

        async def connect(self):
            return None

        async def roundtrip(self, timeout):
            return roundtrip_result

        async def close(self):
            self.closed = True

    monkeypatch.setattr(module, "RedisBus", Bus, raising=False)

    with pytest.raises(RuntimeError, match=expected_error):
        await module._agent_control_startup()

    assert len(buses) == 1
    assert buses[0].closed is True


@pytest.mark.asyncio
async def test_orchestrator_agent_control_rejects_redis_connect_failure(
    monkeypatch: pytest.MonkeyPatch,
    agent_message_hmac_secret: None,
) -> None:
    module = _load_module(
        "orchestrator_agent_control_connect_failure_test",
        ROOT / "services/orchestrator-svc/src/orchestrator_svc/main.py",
    )
    buses: list[object] = []

    class Bus:
        is_redis = False

        def __init__(self, *, allow_fallback):
            assert allow_fallback is False
            self.closed = False
            buses.append(self)

        async def connect(self):
            raise ConnectionError("redis unavailable")

        async def close(self):
            self.closed = True

    monkeypatch.setattr(module, "RedisBus", Bus, raising=False)

    with pytest.raises(RuntimeError, match="Redis connection failed: redis unavailable"):
        await module._agent_control_startup()

    assert buses[0].closed is True


@pytest.mark.asyncio
async def test_orchestrator_agent_control_rejects_missing_signing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AGENT_MESSAGE_HMAC_SECRET", raising=False)
    monkeypatch.delenv("SIGSTORE_SIGN_COMMAND", raising=False)
    monkeypatch.delenv("SIGSTORE_VERIFY_COMMAND", raising=False)
    module = _load_module(
        "orchestrator_agent_control_missing_signing_test",
        ROOT / "services/orchestrator-svc/src/orchestrator_svc/main.py",
    )

    class Bus:
        is_redis = True

        def __init__(self, *, allow_fallback):
            self.closed = False

        async def connect(self):
            return None

        async def roundtrip(self, timeout):
            return True

        async def close(self):
            self.closed = True

    monkeypatch.setattr(module, "RedisBus", Bus, raising=False)

    with pytest.raises(RuntimeError, match="production Agent signing requires"):
        await module._agent_control_startup()


@pytest.mark.asyncio
async def test_orchestrator_agent_control_concurrent_startup_creates_one_runtime(
    monkeypatch: pytest.MonkeyPatch,
    agent_message_hmac_secret: None,
) -> None:
    module = _load_module(
        "orchestrator_agent_control_concurrent_startup_test",
        ROOT / "services/orchestrator-svc/src/orchestrator_svc/main.py",
    )
    buses: list[object] = []
    first_connect_started = asyncio.Event()
    release_connect = asyncio.Event()

    class Bus:
        is_redis = True

        def __init__(self, *, allow_fallback):
            self.closed = False
            buses.append(self)

        async def connect(self):
            if len(buses) == 1:
                first_connect_started.set()
            await release_connect.wait()

        async def roundtrip(self, timeout):
            return True

        async def close(self):
            self.closed = True

    monkeypatch.setattr(module, "RedisBus", Bus)

    first_startup = asyncio.create_task(module._agent_control_startup())
    await first_connect_started.wait()
    second_startup = asyncio.create_task(module._agent_control_startup())
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    release_connect.set()
    clients = await asyncio.gather(first_startup, second_startup)

    assert clients[0] is clients[1]
    assert len(buses) == 1

    await module._agent_control_shutdown()

    assert buses[0].closed is True


@pytest.mark.asyncio
async def test_orchestrator_agent_control_failed_startup_releases_init_lock(
    monkeypatch: pytest.MonkeyPatch,
    agent_message_hmac_secret: None,
) -> None:
    module = _load_module(
        "orchestrator_agent_control_failed_startup_lock_test",
        ROOT / "services/orchestrator-svc/src/orchestrator_svc/main.py",
    )
    buses: list[object] = []

    class Bus:
        is_redis = True

        def __init__(self, *, allow_fallback):
            self.closed = False
            buses.append(self)

        async def connect(self):
            if len(buses) == 1:
                raise ConnectionError("first startup failed")

        async def roundtrip(self, timeout):
            return True

        async def close(self):
            self.closed = True

    monkeypatch.setattr(module, "RedisBus", Bus)

    with pytest.raises(RuntimeError, match="first startup failed"):
        await module._agent_control_startup()

    client = await asyncio.wait_for(module._agent_control_startup(), timeout=0.5)

    assert client is module._AGENT_REQUEST_CLIENT
    assert len(buses) == 2
    assert buses[0].closed is True

    await module._agent_control_shutdown()

    assert buses[1].closed is True


@pytest.mark.asyncio
async def test_orchestrator_shutdown_waits_for_blocked_agent_control_startup(
    monkeypatch: pytest.MonkeyPatch,
    agent_message_hmac_secret: None,
) -> None:
    module = _load_module(
        "orchestrator_shutdown_blocked_agent_startup_test",
        ROOT / "services/orchestrator-svc/src/orchestrator_svc/main.py",
    )
    connect_started = asyncio.Event()
    release_connect = asyncio.Event()
    buses: list[object] = []

    class Bus:
        is_redis = True

        def __init__(self, *, allow_fallback: bool) -> None:
            assert allow_fallback is False
            self.closed = False
            buses.append(self)

        async def connect(self) -> None:
            connect_started.set()
            await release_connect.wait()

        async def roundtrip(self, timeout: float) -> bool:
            return True

        async def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(module, "RedisBus", Bus)

    startup_task = asyncio.create_task(module._agent_control_startup())
    await connect_started.wait()
    shutdown_task = asyncio.create_task(module._orchestrator_shutdown())
    await asyncio.sleep(0)
    shutdown_completed_before_startup = shutdown_task.done()
    release_connect.set()
    await startup_task
    await shutdown_task

    assert shutdown_completed_before_startup is False
    assert len(buses) == 1
    assert buses[0].closed is True
    assert module._AGENT_BUS is None
    assert module._AGENT_REQUEST_CLIENT is None
    assert module._AGENT_RUNTIME_LOOP is None


@pytest.mark.asyncio
async def test_orchestrator_agent_control_rejects_startup_begun_during_shutdown(
    monkeypatch: pytest.MonkeyPatch,
    agent_message_hmac_secret: None,
) -> None:
    module = _load_module(
        "orchestrator_agent_startup_during_shutdown_test",
        ROOT / "services/orchestrator-svc/src/orchestrator_svc/main.py",
    )
    close_started = asyncio.Event()
    release_close = asyncio.Event()
    buses: list[object] = []

    class Bus:
        is_redis = True

        def __init__(self, *, allow_fallback: bool) -> None:
            assert allow_fallback is False
            self.closed = False
            buses.append(self)

        async def connect(self) -> None:
            return None

        async def roundtrip(self, timeout: float) -> bool:
            return True

        async def close(self) -> None:
            if self is buses[0]:
                close_started.set()
                await release_close.wait()
            self.closed = True

    monkeypatch.setattr(module, "RedisBus", Bus)

    await module._agent_control_startup()
    shutdown_task = asyncio.create_task(module._orchestrator_shutdown())
    await close_started.wait()
    late_startup = asyncio.create_task(module._agent_control_startup())
    await asyncio.sleep(0)
    completed_during_shutdown = late_startup.done()
    release_close.set()
    await shutdown_task
    outcome = (await asyncio.gather(late_startup, return_exceptions=True))[0]
    runtime_republished = module._AGENT_BUS is not None
    if runtime_republished:
        await module._orchestrator_shutdown()

    assert completed_during_shutdown is True
    assert isinstance(outcome, RuntimeError)
    assert str(outcome) == "Orchestrator Agent control is shutting down"
    assert len(buses) == 1
    assert buses[0].closed is True
    assert runtime_republished is False
    assert module._AGENT_SHUTDOWN_COUNT == 0


@pytest.mark.asyncio
async def test_orchestrator_agent_control_sequential_startup_after_shutdown(
    monkeypatch: pytest.MonkeyPatch,
    agent_message_hmac_secret: None,
) -> None:
    module = _load_module(
        "orchestrator_agent_control_sequential_reuse_test",
        ROOT / "services/orchestrator-svc/src/orchestrator_svc/main.py",
    )
    buses: list[object] = []

    class Bus:
        is_redis = True

        def __init__(self, *, allow_fallback: bool) -> None:
            assert allow_fallback is False
            self.closed = False
            buses.append(self)

        async def connect(self) -> None:
            return None

        async def roundtrip(self, timeout: float) -> bool:
            return True

        async def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(module, "RedisBus", Bus)

    first_client = await module._agent_control_startup()
    await module._orchestrator_shutdown()
    second_client = await module._agent_control_startup()

    assert len(buses) == 2
    assert buses[0].closed is True
    assert buses[1].closed is False
    assert second_client is not first_client
    assert module._AGENT_REQUEST_CLIENT is second_client

    await module._orchestrator_shutdown()

    assert buses[1].closed is True


def test_orchestrator_agent_control_reuses_owner_loop_and_rejects_other_loop(
    monkeypatch: pytest.MonkeyPatch,
    agent_message_hmac_secret: None,
) -> None:
    module = _load_module(
        "orchestrator_agent_control_loop_ownership_test",
        ROOT / "services/orchestrator-svc/src/orchestrator_svc/main.py",
    )
    buses: list[object] = []

    class Bus:
        is_redis = True

        def __init__(self, *, allow_fallback):
            self.closed = False
            buses.append(self)

        async def connect(self):
            return None

        async def roundtrip(self, timeout):
            return True

        async def close(self):
            self.closed = True

    monkeypatch.setattr(module, "RedisBus", Bus, raising=False)

    async def start_twice():
        first = await module._agent_control_startup()
        second = await module._agent_control_startup()
        assert first is second

    owner_loop = asyncio.new_event_loop()
    owner_loop.run_until_complete(start_twice())
    first_bus = buses[0]
    owner_task = owner_loop.create_task(asyncio.sleep(3600))
    module._RUN_TASKS["owner-run"] = owner_task

    with pytest.raises(RuntimeError, match="owned by another event loop"):
        asyncio.run(module._agent_control_startup())
    with pytest.raises(RuntimeError, match="owned by another event loop"):
        asyncio.run(module._orchestrator_shutdown())

    assert len(buses) == 1
    assert first_bus.closed is False
    assert module._AGENT_BUS is first_bus
    assert owner_task.cancelled() is False
    assert module._RUN_TASKS == {"owner-run": owner_task}

    owner_loop.run_until_complete(module._orchestrator_shutdown())
    owner_loop.close()
    assert first_bus.closed is True
    assert module._AGENT_BUS is None


@pytest.mark.asyncio
async def test_orchestrator_shutdown_waits_for_runs_before_closing_agent_bus() -> None:
    module = _load_module(
        "orchestrator_shutdown_order_test",
        ROOT / "services/orchestrator-svc/src/orchestrator_svc/main.py",
    )
    task_cancelled = asyncio.Event()

    async def active_run():
        try:
            await asyncio.Event().wait()
        finally:
            task_cancelled.set()

    task = asyncio.create_task(active_run())
    await asyncio.sleep(0)
    module._RUN_TASKS["run-active"] = task

    class Bus:
        async def close(self):
            assert task.done()
            assert task_cancelled.is_set()

    module._AGENT_BUS = Bus()
    module._AGENT_REQUEST_CLIENT = object()
    module._AGENT_RUNTIME_LOOP = asyncio.get_running_loop()

    await module._orchestrator_shutdown()

    assert task.cancelled()
    assert module._RUN_TASKS == {}
    assert module._AGENT_BUS is None
    assert module._AGENT_REQUEST_CLIENT is None


@pytest.mark.asyncio
async def test_orchestrator_shutdown_excludes_current_task_from_all_registries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module(
        "orchestrator_shutdown_current_run_task_test",
        ROOT / "services/orchestrator-svc/src/orchestrator_svc/main.py",
    )
    shutdown_marker = asyncio.get_running_loop().create_future()
    module._RUN_TASKS["shutdown"] = shutdown_marker
    module._DIRECT_AGENT_TASKS.add(shutdown_marker)

    with monkeypatch.context() as context:
        context.setattr(module.asyncio, "current_task", lambda: shutdown_marker)
        await module._orchestrator_shutdown()

    marker_was_cancelled = shutdown_marker.cancelled()
    if not shutdown_marker.done():
        shutdown_marker.cancel()

    assert marker_was_cancelled is False
    assert module._RUN_TASKS == {}
    assert module._DIRECT_AGENT_TASKS == set()


@pytest.mark.asyncio
async def test_orchestrator_shutdown_does_not_close_injected_request_client_bus() -> None:
    module = _load_module(
        "orchestrator_injected_request_client_shutdown_test",
        ROOT / "services/orchestrator-svc/src/orchestrator_svc/main.py",
    )

    class Bus:
        def __init__(self) -> None:
            self.closed = False

        async def close(self):
            self.closed = True

    bus = Bus()
    request_client = SimpleNamespace(message_bus=bus)
    module.EngineeringWorkflowClients(request_client=request_client)

    await module._orchestrator_shutdown()

    assert bus.closed is False


@pytest.mark.asyncio
async def test_direct_start_design_uses_independent_local_agent_clients(
    monkeypatch: pytest.MonkeyPatch,
    agent_message_hmac_secret: None,
    tmp_path: Path,
) -> None:
    module = _load_module(
        "orchestrator_direct_local_agent_clients_test",
        ROOT / "services/orchestrator-svc/src/orchestrator_svc/main.py",
    )
    module._RUN_STORE = module.RunStore(tmp_path / "runs.db")
    built_clients: list[object] = []
    buses: list[object] = []

    class Compiled:
        async def ainvoke(self, state):
            assert asyncio.current_task() not in getattr(
                module,
                "_DIRECT_AGENT_TASKS",
                set(),
            )
            return {
                **state,
                "status": "CRITIC",
                "history": ["PLANNING", "CRITIC"],
                "events": [],
            }

    class Graph:
        def __init__(self, clients, workflow_scope):
            built_clients.append(clients)

        def build(self):
            return Compiled()

    class Bus:
        is_redis = True

        def __init__(self, *, allow_fallback):
            assert allow_fallback is False
            self.owner_loop = asyncio.get_running_loop()
            self.closed = False
            self.roundtrips = 0
            buses.append(self)

        async def connect(self):
            await asyncio.sleep(0)

        async def roundtrip(self, timeout):
            self.roundtrips += 1
            return True

        async def close(self):
            assert asyncio.get_running_loop() is self.owner_loop
            self.closed = True

    monkeypatch.setattr(module, "WorkflowGraph", Graph)
    monkeypatch.setattr(module, "RedisBus", Bus)

    results = await asyncio.gather(
        module.start_design(
            {
                "nl_input": "Design molecule A",
                "workflow_scope": "engineering",
                "validation_passed": True,
                "max_refinements": 0,
                "run_id": "run-direct-a",
                "trace_id": "trace-direct-a",
            }
        ),
        module.start_design(
            {
                "nl_input": "Design molecule B",
                "workflow_scope": "engineering",
                "validation_passed": True,
                "max_refinements": 0,
                "run_id": "run-direct-b",
                "trace_id": "trace-direct-b",
            }
        ),
    )

    assert [result["status"] for result in results] == ["completed", "completed"]
    assert len(buses) == 2
    assert all(bus.roundtrips == 1 for bus in buses)
    assert all(bus.closed for bus in buses)
    assert len({id(clients.request_client) for clients in built_clients}) == 2
    assert module._AGENT_BUS is None
    assert module._AGENT_REQUEST_CLIENT is None
    assert module._AGENT_RUNTIME_LOOP is None


@pytest.mark.asyncio
async def test_direct_start_design_registers_same_loop_shared_client_user(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_module(
        "orchestrator_direct_shared_user_registration_test",
        ROOT / "services/orchestrator-svc/src/orchestrator_svc/main.py",
    )
    module._RUN_STORE = module.RunStore(tmp_path / "runs.db")
    module._AGENT_BUS = SimpleNamespace(close=lambda: None)
    module._AGENT_REQUEST_CLIENT = object()
    module._AGENT_RUNTIME_LOOP = asyncio.get_running_loop()
    entered = asyncio.Event()
    release = asyncio.Event()

    class Compiled:
        async def ainvoke(self, state):
            entered.set()
            await release.wait()
            return {
                **state,
                "status": "CRITIC",
                "history": ["PLANNING", "CRITIC"],
                "events": [],
            }

    class Graph:
        def __init__(self, clients, workflow_scope):
            return None

        def build(self):
            return Compiled()

    monkeypatch.setattr(module, "WorkflowGraph", Graph)
    direct_task = asyncio.create_task(
        module.start_design(
            {
                "nl_input": "Design shared-client molecule",
                "workflow_scope": "engineering",
                "validation_passed": True,
                "max_refinements": 0,
                "run_id": "run-direct-shared-registration",
                "trace_id": "trace-direct-shared-registration",
            }
        )
    )

    try:
        await entered.wait()
        assert direct_task in getattr(module, "_DIRECT_AGENT_TASKS", set())
        release.set()
        result = await direct_task
        assert result["status"] == "completed"
        assert getattr(module, "_DIRECT_AGENT_TASKS", set()) == set()
    finally:
        release.set()
        if not direct_task.done():
            direct_task.cancel()
            await asyncio.gather(direct_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_orchestrator_shutdown_cancels_shared_direct_users_before_bus_close(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_module(
        "orchestrator_shared_direct_user_shutdown_test",
        ROOT / "services/orchestrator-svc/src/orchestrator_svc/main.py",
    )
    module._RUN_STORE = module.RunStore(tmp_path / "runs.db")
    entered = asyncio.Event()
    direct_finished = asyncio.Event()

    class Compiled:
        async def ainvoke(self, state):
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                direct_finished.set()

    class Graph:
        def __init__(self, clients, workflow_scope):
            return None

        def build(self):
            return Compiled()

    class Bus:
        def __init__(self) -> None:
            self.closed = False
            self.direct_finished_when_closed = False

        async def close(self):
            self.direct_finished_when_closed = direct_finished.is_set()
            self.closed = True

    bus = Bus()
    module._AGENT_BUS = bus
    module._AGENT_REQUEST_CLIENT = object()
    module._AGENT_RUNTIME_LOOP = asyncio.get_running_loop()
    monkeypatch.setattr(module, "WorkflowGraph", Graph)
    direct_task = asyncio.create_task(
        module.start_design(
            {
                "nl_input": "Design shutdown molecule",
                "workflow_scope": "engineering",
                "validation_passed": True,
                "max_refinements": 0,
                "run_id": "run-direct-shutdown",
                "trace_id": "trace-direct-shutdown",
            }
        )
    )

    try:
        await entered.wait()
        shutdown_task = asyncio.current_task()
        assert shutdown_task is not None
        direct_users = getattr(module, "_DIRECT_AGENT_TASKS", set())
        direct_users.add(shutdown_task)
        module._DIRECT_AGENT_TASKS = direct_users

        await module._orchestrator_shutdown()

        assert direct_task.cancelled()
        assert shutdown_task.cancelled() is False
        assert direct_finished.is_set()
        assert bus.direct_finished_when_closed is True
        assert module._DIRECT_AGENT_TASKS == set()
    finally:
        if not direct_task.done():
            direct_task.cancel()
            await asyncio.gather(direct_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_direct_agent_registration_after_shutdown_uses_local_client(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_module(
        "orchestrator_late_direct_agent_registration_test",
        ROOT / "services/orchestrator-svc/src/orchestrator_svc/main.py",
    )
    module._RUN_STORE = module.RunStore(tmp_path / "runs.db")
    existing_cancelled = asyncio.Event()
    release_existing = asyncio.Event()
    late_entered = asyncio.Event()
    release_late = asyncio.Event()
    built_clients: list[object] = []

    async def existing_shared_user() -> None:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            existing_cancelled.set()
            await release_existing.wait()
            raise

    class SharedBus:
        def __init__(self) -> None:
            self.closed = False

        async def close(self) -> None:
            self.closed = True

    class LocalBus:
        def __init__(self) -> None:
            self.closed = False

        async def close(self) -> None:
            self.closed = True

    class Compiled:
        async def ainvoke(self, state: dict) -> dict:
            late_entered.set()
            await release_late.wait()
            return {
                **state,
                "status": "CRITIC",
                "history": ["PLANNING", "CRITIC"],
                "events": [],
            }

    class Graph:
        def __init__(self, clients: object, workflow_scope: str) -> None:
            built_clients.append(clients)

        def build(self) -> Compiled:
            return Compiled()

    local_bus = LocalBus()
    local_client = object()

    async def create_local_client() -> tuple[LocalBus, object]:
        return local_bus, local_client

    shared_bus = SharedBus()
    shared_client = object()
    module._AGENT_BUS = shared_bus
    module._AGENT_REQUEST_CLIENT = shared_client
    module._AGENT_RUNTIME_LOOP = asyncio.get_running_loop()
    monkeypatch.setattr(module, "_create_agent_request_client", create_local_client)
    monkeypatch.setattr(module, "WorkflowGraph", Graph)

    existing_task = asyncio.create_task(existing_shared_user())
    module._DIRECT_AGENT_TASKS.add(existing_task)
    shutdown_task = asyncio.create_task(module._orchestrator_shutdown())
    late_task = None
    try:
        await existing_cancelled.wait()
        late_task = asyncio.create_task(
            module.start_design(
                {
                    "nl_input": "Design late shutdown molecule",
                    "workflow_scope": "engineering",
                    "validation_passed": True,
                    "max_refinements": 0,
                    "run_id": "run-direct-late-shutdown",
                    "trace_id": "trace-direct-late-shutdown",
                }
            )
        )
        await late_entered.wait()
        release_existing.set()
        await shutdown_task

        assert shared_bus.closed is True
        assert late_task.done() is False
        assert len(built_clients) == 1
        assert built_clients[0].request_client is local_client
        assert late_task not in module._DIRECT_AGENT_TASKS
    finally:
        release_existing.set()
        release_late.set()
        if late_task is not None:
            await asyncio.gather(late_task, return_exceptions=True)
        if not existing_task.done():
            existing_task.cancel()
            await asyncio.gather(existing_task, return_exceptions=True)
        if not shutdown_task.done():
            shutdown_task.cancel()
            await asyncio.gather(shutdown_task, return_exceptions=True)

    assert local_bus.closed is True


def test_direct_start_design_uses_local_client_beside_other_loop_shared_runtime(
    monkeypatch: pytest.MonkeyPatch,
    agent_message_hmac_secret: None,
    tmp_path: Path,
) -> None:
    module = _load_module(
        "orchestrator_direct_cross_loop_local_client_test",
        ROOT / "services/orchestrator-svc/src/orchestrator_svc/main.py",
    )
    module._RUN_STORE = module.RunStore(tmp_path / "runs.db")
    buses: list[object] = []
    built_clients: list[object] = []

    class Bus:
        is_redis = True

        def __init__(self, *, allow_fallback):
            self.owner_loop = asyncio.get_running_loop()
            self.closed = False
            buses.append(self)

        async def connect(self):
            return None

        async def roundtrip(self, timeout):
            return True

        async def close(self):
            assert asyncio.get_running_loop() is self.owner_loop
            self.closed = True

    class Compiled:
        async def ainvoke(self, state):
            assert asyncio.current_task() not in getattr(
                module,
                "_DIRECT_AGENT_TASKS",
                set(),
            )
            return {
                **state,
                "status": "CRITIC",
                "history": ["PLANNING", "CRITIC"],
                "events": [],
            }

    class Graph:
        def __init__(self, clients, workflow_scope):
            built_clients.append(clients)

        def build(self):
            return Compiled()

    monkeypatch.setattr(module, "RedisBus", Bus)
    monkeypatch.setattr(module, "WorkflowGraph", Graph)
    owner_loop = asyncio.new_event_loop()
    owner_loop.run_until_complete(module._agent_control_startup())
    shared_bus = buses[0]
    shared_client = module._AGENT_REQUEST_CLIENT

    result = asyncio.run(
        module.start_design(
            {
                "nl_input": "Design cross-loop molecule",
                "workflow_scope": "engineering",
                "validation_passed": True,
                "max_refinements": 0,
                "run_id": "run-direct-cross-loop",
                "trace_id": "trace-direct-cross-loop",
            }
        )
    )

    assert result["status"] == "completed"
    assert len(buses) == 2
    assert built_clients[0].request_client is not shared_client
    assert built_clients[0].request_client.message_bus is buses[1]
    assert buses[1].closed is True
    assert shared_bus.closed is False
    assert module._AGENT_BUS is shared_bus
    assert module._AGENT_REQUEST_CLIENT is shared_client

    owner_loop.run_until_complete(module._orchestrator_shutdown())
    owner_loop.close()
    assert shared_bus.closed is True


@pytest.mark.asyncio
async def test_direct_start_design_with_injected_clients_never_creates_agent_bus(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_module(
        "orchestrator_direct_injected_clients_test",
        ROOT / "services/orchestrator-svc/src/orchestrator_svc/main.py",
    )
    module._RUN_STORE = module.RunStore(tmp_path / "runs.db")

    class Clients:
        pass

    injected_clients = Clients()

    class Compiled:
        async def ainvoke(self, state):
            assert asyncio.current_task() not in getattr(
                module,
                "_DIRECT_AGENT_TASKS",
                set(),
            )
            return {
                **state,
                "status": "CRITIC",
                "history": ["PLANNING", "CRITIC"],
                "events": [],
            }

    class Graph:
        def __init__(self, clients, workflow_scope):
            assert clients is injected_clients

        def build(self):
            return Compiled()

    def reject_bus(*args, **kwargs):
        raise AssertionError("explicit clients must not initialize Agent Redis")

    monkeypatch.setattr(module, "WorkflowGraph", Graph)
    monkeypatch.setattr(module, "RedisBus", reject_bus)

    result = await module.start_design(
        {
            "nl_input": "Design injected molecule",
            "workflow_scope": "engineering",
            "validation_passed": True,
            "max_refinements": 0,
            "run_id": "run-direct-injected",
            "trace_id": "trace-direct-injected",
            "clients": injected_clients,
        }
    )

    assert result["status"] == "completed"


@pytest.mark.asyncio
async def test_orchestrator_workflow_runs_supply_and_srb_hooks_after_retrosyn(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_module(
        "orchestrator_supply_srb_hooks_test",
        ROOT / "services/orchestrator-svc/src/orchestrator_svc/main.py",
    )
    await _configure_project_run_store(
        module,
        tmp_path,
        "project-orch-supply-srb-1",
    )

    async def record_provenance(state: dict) -> None:
        state["provenance"] = {"recorded": True, "artifact_id": "workflow-artifact"}

    monkeypatch.setattr(module, "_record_workflow_provenance", record_provenance)

    class Clients:
        async def compile_intent(self, state):
            return {"cig": {"source": state["nl_input"]}, "hciv": {}, "intent_cone": {}}

        async def generate_candidates(self, state):
            return [_full_candidate()]

        async def validate_candidates(self, state):
            record = _full_validation_record()
            return {
                "outcome": "PASS",
                "passed": True,
                "records": [record],
                "results": [record],
            }

        async def plan_routes(self, state):
            return {
                "skipped": False,
                "routes": [
                    {
                        "route_id": "route-1",
                        "building_blocks": [{"smiles": "CC"}],
                    }
                ],
            }

        async def assess_supply(self, state):
            assert state["retrosyn"]["routes"][0]["route_id"] == "route-1"
            return {
                "route_id": "route-1",
                "supply_assessment": {"overall_feasibility": "available"},
            }

        async def compile_synthesis(self, state):
            assert state["supply"]["supply_assessment"]["overall_feasibility"] == "available"
            return {
                "status": "compiled",
                "route_id": "route-1",
                "protocols": [{"route_id": "route-1", "ssp_id": "ssp-1"}],
            }

        async def review_candidates(self, state):
            assert state["srb"]["protocols"][0]["ssp_id"] == "ssp-1"
            return {"verdict": "pass", "total_rules": 1}

        async def execute_synthesis(self, state):
            assert state["srb"]["route_id"] == "route-1"
            return {
                "status": "executed",
                "route_id": "route-1",
                "protocols": list(state["srb"]["protocols"]),
            }

    started = await module.start_design(
        {
            "nl_input": "Design KRAS G12C inhibitor",
            "workflow_scope": "full",
            "project_id": "project-orch-supply-srb-1",
            "validation_passed": True,
            "max_refinements": 1,
            "retrosyn_engine": "rsgpt",
            **_full_policy_payload(),
            "clients": Clients(),
            "run_id": "run-orch-supply-srb-1",
            "trace_id": "trace-orch-supply-srb-1",
        }
    )

    assert started["status"] == "completed"
    assert started["history"] == [
        "PLANNING",
        "GENERATING",
        "VALIDATING",
        "RETROSYN",
        "CRITIC",
        "EXECUTING",
    ]
    assert started["state"]["supply"]["supply_assessment"]["overall_feasibility"] == "available"
    assert started["state"]["srb"]["protocols"][0]["ssp_id"] == "ssp-1"
    assert started["state"]["srb_execution"]["status"] == "executed"


@pytest.mark.asyncio
async def test_orchestrator_refinement_feeds_validation_back_to_generation() -> None:
    module = _load_module(
        "orchestrator_refinement_feedback_test",
        ROOT / "services/orchestrator-svc/src/orchestrator_svc/main.py",
    )

    class Clients:
        def __init__(self) -> None:
            self.generation_states: list[dict] = []
            self.validation_calls = 0

        async def compile_intent(self, state):
            return {"cig": {"source": state["nl_input"]}, "hciv": {}, "intent_cone": {}}

        async def generate_candidates(self, state):
            self.generation_states.append(dict(state))
            return [{"smiles": "CCO", "canonical_smiles": "CCO"}]

        async def validate_candidates(self, state):
            self.validation_calls += 1
            if self.validation_calls == 1:
                return {
                    "passed": False,
                    "results": [{"smiles": "CCO", "ki_nm": 1000.0}],
                    "reason": "affinity gate failed",
                }
            return {"passed": True, "results": [{"smiles": "CCO", "ki_nm": 5.0}]}

        async def plan_routes(self, state):
            return {"skipped": True, "reason": "retrosyn resource not configured"}

        async def review_candidates(self, state):
            return {"verdict": "pass", "total_rules": 1}

    clients = Clients()
    started = await module.start_design(
        {
            "nl_input": "Design KRAS G12C inhibitor",
            "workflow_scope": "engineering",
            "clients": clients,
            "run_id": "run-orch-feedback-1",
            "trace_id": "trace-orch-feedback-1",
            "validation_passed": True,
            "max_refinements": 1,
        }
    )

    assert started["status"] == "completed"
    assert len(clients.generation_states) == 2
    assert "generation_feedback" not in clients.generation_states[0]
    feedback = clients.generation_states[1]["generation_feedback"]
    assert feedback == [
        {
            "source": "validation",
            "refinement_count": 1,
            "passed": False,
            "reason": "affinity gate failed",
            "results": [{"smiles": "CCO", "ki_nm": 1000.0}],
        }
    ]


@pytest.mark.asyncio
async def test_orchestrator_refinement_feeds_critic_back_to_generation() -> None:
    module = _load_module(
        "orchestrator_critic_feedback_test",
        ROOT / "services/orchestrator-svc/src/orchestrator_svc/main.py",
    )

    class Clients:
        def __init__(self) -> None:
            self.generation_states: list[dict] = []
            self.critic_calls = 0

        async def compile_intent(self, state):
            return {"cig": {"source": state["nl_input"]}, "hciv": {}, "intent_cone": {}}

        async def generate_candidates(self, state):
            self.generation_states.append(dict(state))
            return [{"smiles": "CCO", "canonical_smiles": "CCO"}]

        async def validate_candidates(self, state):
            return {"passed": True, "results": [{"smiles": "CCO", "ki_nm": 5.0}]}

        async def plan_routes(self, state):
            return {"skipped": False, "routes": [{"route_id": "route-1"}]}

        async def review_candidates(self, state):
            self.critic_calls += 1
            if self.critic_calls == 1:
                return {
                    "verdict": "fail",
                    "reason": "route feasibility failed",
                    "rule_results": [
                        {
                            "rule_id": "route_feasibility",
                            "verdict": "fail",
                            "score": 0.0,
                        }
                    ],
                }
            return {"verdict": "pass", "total_rules": 1}

    clients = Clients()
    started = await module.start_design(
        {
            "nl_input": "Design KRAS G12C inhibitor",
            "workflow_scope": "engineering",
            "clients": clients,
            "run_id": "run-orch-critic-feedback-1",
            "trace_id": "trace-orch-critic-feedback-1",
            "validation_passed": True,
            "max_refinements": 1,
        }
    )

    assert started["status"] == "completed"
    assert len(clients.generation_states) == 2
    assert "generation_feedback" not in clients.generation_states[0]
    feedback = clients.generation_states[1]["generation_feedback"]
    assert feedback == [
        {
            "source": "critic",
            "refinement_count": 1,
            "verdict": "fail",
            "reason": "route feasibility failed",
            "rule_results": [
                {
                    "rule_id": "route_feasibility",
                    "verdict": "fail",
                    "score": 0.0,
                }
            ],
        }
    ]


@pytest.mark.asyncio
async def test_orchestrator_engineering_workflow_records_crg_state() -> None:
    from mf_core.db import store as run_store

    run_store.init_db()
    run_store.insert_project(
        "project-crg-1",
        "project-crg-1",
        "",
        "2026-07-27T10:00:00+00:00",
    )
    module = _load_module(
        "orchestrator_engineering_crg_test",
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
            "validation_passed": True,
            "max_refinements": 1,
            "clients": Clients(),
            "run_id": "run-orch-crg-1",
            "trace_id": "trace-orch-crg-1",
            "project_id": "project-crg-1",
        }
    )

    crg = started["state"]["crg"]
    beliefs = crg["beliefs"]
    edges = crg["edges"]

    assert crg["project_id"] == "project-crg-1"
    assert [belief["object"] for belief in beliefs] == [
        "PLANNING",
        "GENERATING",
        "VALIDATING",
        "CRITIC",
    ]
    assert all(belief["subject"] == "run-orch-crg-1" for belief in beliefs)
    assert all(belief["predicate"] == "workflow_stage" for belief in beliefs)
    assert all(belief["source_agent"] == "orchestrator" for belief in beliefs)
    assert [edge["relation"] for edge in edges] == [
        "derives_from",
        "derives_from",
        "derives_from",
    ]
    assert edges[0]["source_belief_id"] == beliefs[0]["id"]
    assert edges[0]["target_belief_id"] == beliefs[1]["id"]


@pytest.mark.asyncio
async def test_orchestrator_full_workflow_uses_runtime_clients(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_module(
        "orchestrator_full_workflow_test",
        ROOT / "services/orchestrator-svc/src/orchestrator_svc/main.py",
    )
    await _configure_project_run_store(module, tmp_path, "project-orch-full-1")
    shared_client = object()
    module._AGENT_REQUEST_CLIENT = shared_client
    module._AGENT_RUNTIME_LOOP = asyncio.get_running_loop()

    async def record_provenance(state: dict) -> None:
        state["provenance"] = {"recorded": True, "artifact_id": "workflow-artifact"}

    monkeypatch.setattr(module, "_record_workflow_provenance", record_provenance)

    class Clients:
        def __init__(self, request_client):
            assert request_client is shared_client

        async def compile_intent(self, state):
            return {"cig": {"source": state["nl_input"]}, "hciv": {}, "intent_cone": {}}

        async def generate_candidates(self, state):
            return [_full_candidate()]

        async def validate_candidates(self, state):
            record = _full_validation_record()
            record["delta_g_kcal_mol"] = -8.0
            return {
                "outcome": "PASS",
                "passed": True,
                "records": [record],
                "results": [record],
            }

        async def plan_routes(self, state):
            return {"skipped": False, "routes": [{"route_id": "route-1"}]}

        async def assess_supply(self, state):
            return {
                "route_id": "route-1",
                "supply_assessment": {"overall_feasibility": "available"},
            }

        async def compile_synthesis(self, state):
            return {
                "status": "compiled",
                "route_id": "route-1",
                "protocols": [{"route_id": "route-1", "ssp_id": "ssp-1"}],
            }

        async def review_candidates(self, state):
            return {"verdict": "pass", "total_rules": 1}

        async def execute_synthesis(self, state):
            return {
                "status": "executed",
                "route_id": "route-1",
                "protocols": list(state["srb"]["protocols"]),
            }

    monkeypatch.setattr(module, "FullWorkflowClients", Clients, raising=False)
    monkeypatch.setattr(
        module,
        "RedisBus",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("same-loop shared Agent client must be reused")
        ),
    )

    started = await module.start_design(
        {
            "nl_input": "Design KRAS G12C inhibitor",
            "workflow_scope": "full",
            "project_id": "project-orch-full-1",
            "validation_passed": True,
            "max_refinements": 1,
            "retrosyn_engine": "rsgpt",
            **_full_policy_payload(),
            "run_id": "run-orch-full-1",
            "trace_id": "trace-orch-full-1",
        }
    )

    assert started["status"] == "completed"
    assert started["history"] == [
        "PLANNING",
        "GENERATING",
        "VALIDATING",
        "RETROSYN",
        "CRITIC",
        "EXECUTING",
    ]
    assert started["state"]["candidates"][0]["canonical_smiles"] == "CCO"
    assert started["state"]["validation"]["results"][0]["delta_g_kcal_mol"] == -8.0
    assert started["state"]["retrosyn"]["routes"][0]["route_id"] == "route-1"
    assert started["state"]["critic"]["verdict"] == "pass"
    assert started["state"]["srb_execution"]["status"] == "executed"


@pytest.mark.parametrize("entrypoint", ["direct", "grpc"])
@pytest.mark.asyncio
async def test_default_full_workflow_keeps_runtime_clients_out_of_business_state(
    entrypoint: str,
    agent_message_hmac_secret: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from mf_agents.base.agent import BaseAgent
    from mf_agents.messaging.redis_bus import InMemoryBus
    from mf_agents.messaging.request_client import AgentRequestClient
    from mf_core.proto_gen.moleculeforge.v1.agent import orchestrator_pb2

    module = _load_module(
        f"orchestrator_full_workflow_runtime_boundary_{entrypoint}_test",
        ROOT / "services/orchestrator-svc/src/orchestrator_svc/main.py",
    )
    await _configure_project_run_store(module, tmp_path, "project-runtime-boundary")
    bus = InMemoryBus()
    await bus.connect()
    request_client = AgentRequestClient(bus)
    agent_payloads: list[tuple[str, dict]] = []

    class GeneratorAgent(BaseAgent):
        def __init__(self) -> None:
            super().__init__("generator_coord", message_bus=bus)
            self._subscription_subjects = ["agent.generator_coord.request"]

        async def process(self, payload):
            agent_payloads.append(("generator_coord", dict(payload)))
            if payload.get("action") == "generator_coord/feedback/v1":
                return _feedback_ack(payload)
            return {
                "status": "dispatched",
                "selected_generators": ["rdkit_random"],
                "candidates": [
                    {
                        "candidate_id": "candidate-runtime-boundary",
                        "smiles": "CCO",
                        "canonical_smiles": "CCO",
                        "generator_name": "rdkit_random",
                    }
                ],
            }

    class ValidationAgent(BaseAgent):
        def __init__(self) -> None:
            super().__init__("validation_agent", message_bus=bus)
            self._subscription_subjects = ["agent.validation.request"]

        async def process(self, payload):
            agent_payloads.append(("validation_agent", dict(payload)))
            candidate = payload["candidates"][0]
            return _validation_batch_response(
                payload,
                [
                    _full_validation_record(
                        candidate_id=candidate["candidate_id"],
                        smiles=candidate["canonical_smiles"],
                    )
                ],
                outcome="PASS",
            )

    class Compiled:
        def __init__(self, clients) -> None:
            self.clients = clients

        async def ainvoke(self, state):
            state["hciv"] = {}
            state["intent_cone"] = {}
            state["cig"] = {}
            state["candidates"] = await self.clients.generate_candidates(state)
            state["validation"] = await self.clients.validate_candidates(state)
            state["validation_passed"] = bool(state["validation"]["passed"])
            state["status"] = "EXECUTING"
            return state

    class Graph:
        def __init__(self, clients, workflow_scope) -> None:
            assert isinstance(clients, module.FullWorkflowClients)
            assert clients.request_client is request_client
            assert workflow_scope == "full"
            self.clients = clients

        def build(self):
            return Compiled(self.clients)

    async def ignore_provenance(state):
        return None

    module._AGENT_REQUEST_CLIENT = request_client
    module._AGENT_RUNTIME_LOOP = asyncio.get_running_loop()
    monkeypatch.setattr(module, "WorkflowGraph", Graph)
    monkeypatch.setattr(module, "_record_workflow_provenance", ignore_provenance)
    generator_agent = GeneratorAgent()
    validation_agent = ValidationAgent()
    await generator_agent.start()
    await validation_agent.start()
    run_id = f"run-runtime-boundary-{entrypoint}"
    trace_id = f"trace-runtime-boundary-{entrypoint}"

    try:
        if entrypoint == "direct":
            response = await module.start_design(
                {
                    "nl_input": "Design KRAS G12C inhibitor",
                    "workflow_scope": "full",
                    "project_id": "project-runtime-boundary",
                    "validation_passed": True,
                    "max_refinements": 1,
                    **_full_policy_payload(),
                    "run_id": run_id,
                    "trace_id": trace_id,
                    "_mforge_internal_legacy_design_request": True,
                }
            )
            final_state = response["state"]
        else:
            request = orchestrator_pb2.StartPipelineRequest(
                nl_input="Design KRAS G12C inhibitor",
                workflow_scope="full",
                project_id="project-runtime-boundary",
                validation_passed=True,
                max_refinements=1,
                run_id=run_id,
                trace_id=trace_id,
                validation_policy_json=json.dumps(
                    _full_policy_payload()["validation_policy"],
                    sort_keys=True,
                ),
                teacher_policy_json=json.dumps(
                    _full_policy_payload()["teacher_policy"],
                    sort_keys=True,
                ),
                selection_policy_json=json.dumps(
                    _full_policy_payload()["selection_policy"],
                    sort_keys=True,
                ),
            )
            response = await module.OrchestratorServicer().StartPipeline(request, None)
            assert response.status == "completed"
            final_state = (await module.get_design_status(run_id))["state"]
    finally:
        await bus.close()

    assert [agent_name for agent_name, _ in agent_payloads] == [
        "generator_coord",
        "validation_agent",
        "generator_coord",
    ]
    assert "clients" not in final_state["request"]
    assert "_mforge_internal_legacy_design_request" not in final_state["request"]
    assert all(
        "clients" not in payload and "_mforge_internal_legacy_design_request" not in payload
        for _, payload in agent_payloads
    )


@pytest.mark.asyncio
async def test_full_workflow_compile_intent_uses_signed_nl2obj_boundary() -> None:
    module = _load_module(
        "orchestrator_full_workflow_nl2obj_boundary_test",
        ROOT / "services/orchestrator-svc/src/orchestrator_svc/main.py",
    )
    request_client = _AgentRequestClientStub(
        lambda subject, payload: {
            "status": "resolved",
            "cig": {"project_id": payload["project_id"]},
            "hciv": {"coordinates": [1.0]},
            "intent_cone": {"axis": [1.0]},
            "objectives": {"qed": "maximize"},
        }
    )

    result = await module.FullWorkflowClients(request_client=request_client).compile_intent(
        {
            "run_id": "run-nl2obj",
            "trace_id": "trace-nl2obj",
            "nl_input": "Design a selective inhibitor",
            "request": {
                "project_id": "project-1",
                "target_family": "KRAS",
                "workflow_scope": "full",
                "clients": object(),
                "_mforge_internal_legacy_design_request": True,
                "run_id": "spoofed-run",
                "trace_id": "spoofed-trace",
                "request_id": "spoofed-request",
                "schema_version": "spoofed-schema",
            },
        }
    )

    assert result == {
        "cig": {"project_id": "project-1"},
        "hciv": {"coordinates": [1.0]},
        "intent_cone": {"axis": [1.0]},
        "objectives": {"qed": "maximize"},
    }
    call = request_client.calls[0]
    assert call["subject"] == "agent.nl2obj.request"
    assert call["payload_type_url"] == "type.moleculeforge.ai/agent/nl2obj/request.v1"
    assert call["payload"] == {
        "project_id": "project-1",
        "target_family": "KRAS",
        "intent": "Design a selective inhibitor",
        "trace_id": "trace-nl2obj",
        "parent_id": "run-nl2obj:planning:0",
        "run_id": "run-nl2obj",
        "request_id": "run-nl2obj:nl2obj:0",
        "schema_version": "nl2obj.request.v1",
    }


@pytest.mark.asyncio
async def test_full_workflow_generator_coord_receives_typed_generation_context() -> None:
    module = _load_module(
        "orchestrator_full_workflow_intent_cone_test",
        ROOT / "services/orchestrator-svc/src/orchestrator_svc/main.py",
    )
    request_client = _generator_coord_request_client()
    cig = {
        "project_id": "project-1",
        "objectives": [{"id": "qed", "type": "MAXIMIZE"}],
    }
    cone = {"axis": [1.0] + [0.0] * 128, "half_angle": 0.25}
    candidates = await module.FullWorkflowClients(
        request_client=request_client
    ).generate_candidates(
        {
            "run_id": "run-full-intent",
            "trace_id": "trace-full-intent",
            "cig": cig,
            "intent_cone": cone,
            "request": {"n_samples": 1, "seed": 7},
        }
    )

    assert candidates[0]["canonical_smiles"] == "CCO"
    payload = request_client.calls[0]["payload"]
    assert payload["cig"] == cig
    assert payload["intent_cone"] == cone
    assert payload["generator_params"]["sampling_seed"] == 7


@pytest.mark.asyncio
async def test_full_workflow_generator_coord_preserves_generation_feedback_envelope() -> None:
    module = _load_module(
        "orchestrator_full_workflow_generation_feedback_test",
        ROOT / "services/orchestrator-svc/src/orchestrator_svc/main.py",
    )
    request_client = _generator_coord_request_client()
    feedback = [
        {
            "source": "validation",
            "reason": "affinity gate failed",
            "passed": False,
            "evidence_ids": "validation-belief-1",
        }
    ]
    candidates = await module.FullWorkflowClients(
        request_client=request_client
    ).generate_candidates(
        {
            "run_id": "run-full-feedback",
            "trace_id": "trace-full-feedback",
            "intent_cone": {"axis": [1.0] + [0.0] * 128, "half_angle": 0.25},
            "generation_feedback": feedback,
            "request": {"n_samples": 1, "seed": 7},
        }
    )

    assert candidates[0]["canonical_smiles"] == "CCO"
    generator_params = request_client.calls[0]["payload"]["generator_params"]
    assert generator_params["sampling_seed"] == 7
    assert json.loads(generator_params["generation_feedback"]) == feedback
    jmcg_feedback = json.loads(generator_params["jmcg_feedback"])
    assert jmcg_feedback["schema"] == "moleculeforge.jmcg.feedback.v1"
    assert jmcg_feedback["run_id"] == "run-full-feedback"
    assert [record["kind"] for record in jmcg_feedback["records"]] == [
        "intent",
        "property",
    ]
    assert len(jmcg_feedback["records"][0]["humu_embedding"]) == 129
    assert jmcg_feedback["records"][0]["metadata"]["embedding_source"] == "intent_cone.axis"
    assert jmcg_feedback["records"][1] == (
        {
            "kind": "property",
            "source": "validation",
            "run_id": "run-full-feedback",
            "subject": {"type": "workflow_feedback", "id": "validation-0"},
            "weight": 1.0,
            "polarity": "repel",
            "confidence": 1.0,
            "evidence_ids": ["validation-belief-1"],
            "metadata": {
                "passed": False,
                "reason": "affinity gate failed",
            },
        }
    )


@pytest.mark.asyncio
async def test_full_workflow_generator_coord_receives_non_steering_feedback() -> None:
    module = _load_module(
        "orchestrator_full_workflow_intent_pocket_feedback_test",
        ROOT / "services/orchestrator-svc/src/orchestrator_svc/main.py",
    )
    request_client = _generator_coord_request_client()
    candidates = await module.FullWorkflowClients(
        request_client=request_client
    ).generate_candidates(
        {
            "run_id": "run-intent-pocket-feedback",
            "trace_id": "trace-intent-pocket-feedback",
            "hciv": {"coordinates": [1.0, 0.0], "curvature": 1.0},
            "intent_cone": {"axis": [1.0] + [0.0] * 128, "half_angle": 0.25},
            "cig": {
                "intent_id": "cig-1",
                "target_context": {
                    "pdb_id": "6OIM",
                    "pocket_id": "switch-ii",
                    "binding_mode_prior": "covalent_reversible",
                },
            },
            "request": {"n_samples": 1, "seed": 7},
        }
    )

    assert candidates[0]["canonical_smiles"] == "CCO"
    generator_params = request_client.calls[0]["payload"]["generator_params"]
    jmcg_feedback = json.loads(generator_params["jmcg_feedback"])
    assert [record["kind"] for record in jmcg_feedback["records"]] == [
        "intent",
        "pocket",
    ]
    assert jmcg_feedback["records"][0]["subject"] == {
        "type": "intent",
        "id": "run-intent-pocket-feedback",
    }
    assert len(jmcg_feedback["records"][0]["humu_embedding"]) == 129
    assert jmcg_feedback["records"][0]["metadata"]["embedding_source"] == "intent_cone.axis"
    assert jmcg_feedback["records"][1]["subject"] == {
        "type": "pocket",
        "id": "switch-ii",
    }
    assert jmcg_feedback["records"][1]["metadata"] == {
        "binding_mode_prior": "covalent_reversible",
        "pdb_id": "6OIM",
        "pocket_id": "switch-ii",
    }
    assert "humu_embedding" not in jmcg_feedback["records"][1]


@pytest.mark.asyncio
async def test_full_workflow_generator_receives_pocket_embedding_when_encoder_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module(
        "orchestrator_full_workflow_pocket_embedding_test",
        ROOT / "services/orchestrator-svc/src/orchestrator_svc/main.py",
    )
    request_client = _generator_coord_request_client()

    async def encode_pocket(payload):
        assert payload == {
            "coords": [[0.0, 0.0, 0.0]],
            "elements": ["C"],
            "residue_types": ["GLY"],
        }
        return {
            "humu_embedding": [1.0] + [0.0] * 128,
            "curvature": 1.0,
            "source": "humu_encoder_svc",
            "evidence_ids": ["pocket-geometry"],
        }

    monkeypatch.setattr(module, "_encode_pocket_humu_feedback", encode_pocket)

    candidates = await module.FullWorkflowClients(
        request_client=request_client
    ).generate_candidates(
        {
            "run_id": "run-pocket-embedding",
            "trace_id": "trace-pocket-embedding",
            "cig": {
                "target_context": {
                    "pocket_id": "switch-ii",
                    "coords": [[0.0, 0.0, 0.0]],
                    "elements": ["C"],
                    "residue_types": ["GLY"],
                },
            },
            "request": {"n_samples": 1, "seed": 7},
        }
    )

    assert candidates[0]["canonical_smiles"] == "CCO"
    generator_params = request_client.calls[0]["payload"]["generator_params"]
    jmcg_feedback = json.loads(generator_params["jmcg_feedback"])
    pocket_record = next(
        record for record in jmcg_feedback["records"] if record["kind"] == "pocket"
    )
    assert len(pocket_record["humu_embedding"]) == 129
    assert pocket_record["curvature"] == 1.0
    assert pocket_record["source"] == "humu_encoder_svc"
    assert pocket_record["evidence_ids"] == ["pocket-geometry"]
    assert pocket_record["metadata"]["pocket_id"] == "switch-ii"


@pytest.mark.asyncio
async def test_full_workflow_metadata_only_pocket_feedback_stays_non_steering() -> None:
    module = _load_module(
        "orchestrator_full_workflow_metadata_only_pocket_feedback_test",
        ROOT / "services/orchestrator-svc/src/orchestrator_svc/main.py",
    )
    request_client = _generator_coord_request_client()
    await module.FullWorkflowClients(request_client=request_client).generate_candidates(
        {
            "run_id": "run-pocket-metadata-only",
            "trace_id": "trace-pocket-metadata-only",
            "cig": {
                "target_context": {
                    "pdb_id": "6OIM",
                    "pocket_id": "switch-ii",
                },
            },
            "request": {"n_samples": 1, "seed": 7},
        }
    )

    generator_params = request_client.calls[0]["payload"]["generator_params"]
    jmcg_feedback = json.loads(generator_params["jmcg_feedback"])
    pocket_record = next(
        record for record in jmcg_feedback["records"] if record["kind"] == "pocket"
    )
    assert "humu_embedding" not in pocket_record
    assert pocket_record["metadata"] == {
        "pdb_id": "6OIM",
        "pocket_id": "switch-ii",
    }


@pytest.mark.asyncio
async def test_full_workflow_intent_axis_embedding_becomes_steering_capable() -> None:
    module = _load_module(
        "orchestrator_full_workflow_intent_axis_embedding_test",
        ROOT / "services/orchestrator-svc/src/orchestrator_svc/main.py",
    )
    request_client = _generator_coord_request_client()
    axis = [1.0] + [0.0] * 128
    await module.FullWorkflowClients(request_client=request_client).generate_candidates(
        {
            "run_id": "run-intent-axis",
            "trace_id": "trace-intent-axis",
            "intent_cone": {"axis": axis, "half_angle": 0.25},
            "request": {"n_samples": 1, "seed": 7},
        }
    )

    generator_params = request_client.calls[0]["payload"]["generator_params"]
    jmcg_feedback = json.loads(generator_params["jmcg_feedback"])
    intent_record = next(
        record for record in jmcg_feedback["records"] if record["kind"] == "intent"
    )
    assert intent_record["humu_embedding"] == axis
    assert intent_record["metadata"]["embedding_source"] == "intent_cone.axis"


@pytest.mark.asyncio
async def test_full_workflow_invalid_lorentz_intent_axis_stays_non_steering() -> None:
    module = _load_module(
        "orchestrator_full_workflow_invalid_lorentz_intent_axis_test",
        ROOT / "services/orchestrator-svc/src/orchestrator_svc/main.py",
    )
    request_client = _generator_coord_request_client()
    await module.FullWorkflowClients(request_client=request_client).generate_candidates(
        {
            "run_id": "run-invalid-intent-axis",
            "trace_id": "trace-invalid-intent-axis",
            "intent_cone": {"axis": [0.0] * 129, "half_angle": 0.25},
            "request": {"n_samples": 1, "seed": 7},
        }
    )

    generator_params = request_client.calls[0]["payload"]["generator_params"]
    jmcg_feedback = json.loads(generator_params["jmcg_feedback"])
    intent_record = next(
        record for record in jmcg_feedback["records"] if record["kind"] == "intent"
    )
    assert "humu_embedding" not in intent_record


@pytest.mark.asyncio
async def test_full_workflow_hciv_vector_does_not_become_humu_embedding() -> None:
    module = _load_module(
        "orchestrator_full_workflow_hciv_non_embedding_test",
        ROOT / "services/orchestrator-svc/src/orchestrator_svc/main.py",
    )
    request_client = _generator_coord_request_client()
    await module.FullWorkflowClients(request_client=request_client).generate_candidates(
        {
            "run_id": "run-hciv-non-embedding",
            "trace_id": "trace-hciv-non-embedding",
            "hciv": {"coordinates": [0.0] * 128, "curvature": 1.0},
            "intent_cone": {"axis": [0.0] * 128, "half_angle": 0.25},
            "request": {"n_samples": 1, "seed": 7},
        }
    )

    generator_params = request_client.calls[0]["payload"]["generator_params"]
    jmcg_feedback = json.loads(generator_params["jmcg_feedback"])
    intent_record = next(
        record for record in jmcg_feedback["records"] if record["kind"] == "intent"
    )
    assert "humu_embedding" not in intent_record
    assert intent_record["metadata"]["has_hciv"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("generation_strategy", [None, "hfm_3d"])
async def test_full_workflow_default_generation_uses_generator_coord_without_hfm_import(
    monkeypatch: pytest.MonkeyPatch,
    generation_strategy: str | None,
) -> None:
    module = _load_module(
        f"orchestrator_full_workflow_default_generator_coord_{generation_strategy}_test",
        ROOT / "services/orchestrator-svc/src/orchestrator_svc/main.py",
    )
    original_import = builtins.__import__

    def reject_hfm_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name in {"hfm_generator_svc", "hfm_generator_svc.main"}:
            raise AssertionError("full workflow generation must not import HFM directly")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", reject_hfm_import)
    request_client = _AgentRequestClientStub(
        lambda subject, payload: {
            "status": "dispatched",
            "selected_generators": ["hfm_3d"],
            "candidates": [{"smiles": "CCO"}],
        }
    )
    request = {
        "project_id": "project-1",
        "n_samples": 1,
        "seed": 7,
    }
    if generation_strategy is not None:
        request["generation_strategy"] = generation_strategy

    candidates = await module.FullWorkflowClients(
        request_client=request_client
    ).generate_candidates(
        {
            "run_id": "run-default-generator-coord",
            "trace_id": "trace-default-generator-coord",
            "request": request,
        }
    )

    assert candidates == [{"smiles": "CCO", "canonical_smiles": "CCO"}]
    call = request_client.calls[0]
    assert call["subject"] == "agent.generator_coord.request"
    assert call["payload"]["run_id"] == "run-default-generator-coord"
    assert call["payload"]["trace_id"] == "trace-default-generator-coord"
    if generation_strategy is None:
        assert "generation_strategy" not in call["payload"]
    else:
        assert call["payload"]["generation_strategy"] == generation_strategy


@pytest.mark.asyncio
async def test_full_workflow_generation_delegates_to_generator_coord_strategy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module(
        "orchestrator_full_workflow_generator_coord_test",
        ROOT / "services/orchestrator-svc/src/orchestrator_svc/main.py",
    )
    request_client = _AgentRequestClientStub(
        lambda subject, payload: {
            "status": "dispatched",
            "selected_generators": ["hfm_3d", "fragfm"],
            "candidates": [{"smiles": "CCN"}],
        }
    )

    cone = {"axis": [1.0] + [0.0] * 128, "half_angle": 0.25}
    candidates = await module.FullWorkflowClients(
        request_client=request_client
    ).generate_candidates(
        {
            "run_id": "run-generator-coord",
            "trace_id": "trace-generator-coord",
            "hciv": {"coordinates": [1.0, 0.0], "curvature": 1.0},
            "intent_cone": cone,
            "request": {
                "project_id": "project-1",
                "generation_strategy": "auto",
                "objectives": {"complexity": "high"},
                "n_samples": 2,
                "seed": 11,
            },
        }
    )

    assert candidates == [{"smiles": "CCN", "canonical_smiles": "CCN"}]
    assert request_client.calls[0]["subject"] == "agent.generator_coord.request"
    assert request_client.calls[0]["payload_type_url"] == (
        "type.moleculeforge.ai/agent/generator_coord/request.v1"
    )
    assert request_client.calls[0]["payload"] == (
        {
            "project_id": "project-1",
            "run_id": "run-generator-coord",
            "trace_id": "trace-generator-coord",
            "parent_id": "run-generator-coord:generating:0",
            "request_id": "run-generator-coord:generator_coord:0",
            "schema_version": "generator_coord.request.v1",
            "generation_strategy": "auto",
            "objectives": {"complexity": "high"},
            "cig": None,
            "hciv": {"coordinates": [1.0, 0.0], "curvature": 1.0},
            "intent_cone": cone,
            "n_samples": 2,
            "batch_size": 2,
            "generator_params": {
                "sampling_seed": 11,
                "jmcg_feedback": json.dumps(
                    {
                        "schema": "moleculeforge.jmcg.feedback.v1",
                        "run_id": "run-generator-coord",
                        "project_id": "project-1",
                        "records": [
                            {
                                "kind": "intent",
                                "source": "orchestrator_svc",
                                "run_id": "run-generator-coord",
                                "subject": {
                                    "type": "intent",
                                    "id": "run-generator-coord",
                                },
                                "weight": 1.0,
                                "polarity": "attract",
                                "confidence": 1.0,
                                "evidence_ids": [],
                                "humu_embedding": cone["axis"],
                                "metadata": {
                                    "has_hciv": True,
                                    "hciv_keys": ["coordinates", "curvature"],
                                    "has_intent_cone": True,
                                    "intent_cone_keys": ["axis", "half_angle"],
                                    "half_angle": 0.25,
                                    "embedding_source": "intent_cone.axis",
                                },
                            }
                        ],
                    },
                    sort_keys=True,
                ),
            },
        }
    )


@pytest.mark.asyncio
async def test_full_workflow_generator_coord_receives_generation_feedback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module(
        "orchestrator_full_workflow_generator_coord_feedback_test",
        ROOT / "services/orchestrator-svc/src/orchestrator_svc/main.py",
    )
    request_client = _AgentRequestClientStub(
        lambda subject, payload: {
            "status": "dispatched",
            "selected_generators": ["hfm_3d"],
            "candidates": [{"smiles": "CCN"}],
        }
    )

    feedback = [
        {
            "source": "critic",
            "verdict": "fail",
            "reason": "synthetic accessibility failed",
        }
    ]
    candidates = await module.FullWorkflowClients(
        request_client=request_client
    ).generate_candidates(
        {
            "run_id": "run-generator-coord-feedback",
            "trace_id": "trace-generator-coord-feedback",
            "hciv": {"coordinates": [1.0, 0.0], "curvature": 1.0},
            "intent_cone": {"axis": [1.0] + [0.0] * 128, "half_angle": 0.25},
            "generation_feedback": feedback,
            "request": {
                "project_id": "project-1",
                "generation_strategy": "auto",
                "n_samples": 1,
                "seed": 11,
            },
        }
    )

    assert candidates == [{"smiles": "CCN", "canonical_smiles": "CCN"}]
    generator_params = request_client.calls[0]["payload"]["generator_params"]
    assert generator_params["sampling_seed"] == 11
    assert json.loads(generator_params["generation_feedback"]) == feedback
    jmcg_feedback = json.loads(generator_params["jmcg_feedback"])
    assert [record["kind"] for record in jmcg_feedback["records"]] == [
        "intent",
        "property",
    ]
    assert jmcg_feedback["records"][1] == (
        {
            "kind": "property",
            "source": "critic",
            "run_id": "run-generator-coord-feedback",
            "subject": {"type": "workflow_feedback", "id": "critic-0"},
            "weight": 1.0,
            "polarity": "repel",
            "confidence": 1.0,
            "evidence_ids": [],
            "metadata": {
                "reason": "synthetic accessibility failed",
                "verdict": "fail",
            },
        }
    )


@pytest.mark.asyncio
async def test_full_workflow_validation_preserves_quality_gate_inputs_for_agent() -> None:
    module = _load_module(
        "orchestrator_full_workflow_quality_gate_test",
        ROOT / "services/orchestrator-svc/src/orchestrator_svc/main.py",
    )
    candidate = _full_candidate()
    record = _full_validation_record(outcome="FAIL", qed=0.2)

    def respond(_subject, payload):
        if payload.get("action") == "generator_coord/feedback/v1":
            return _feedback_ack(payload)
        return _validation_batch_response(
            payload,
            [record],
            outcome="FAIL",
        )

    request_client = _AgentRequestClientStub(respond)
    policies = _full_policy_payload()
    state = {
        "run_id": "run-quality-gate",
        "trace_id": "trace-quality-gate",
        "candidates": [candidate],
        "request": {
            "project_id": "project-1",
            **policies,
            "protein_pdb_id": "6OIM",
            "boltz_ensemble_size": 1,
            "boltz_max_ki_nm": 10.0,
        },
    }

    result = await module.FullWorkflowClients(request_client=request_client).validate_candidates(
        state
    )

    assert result["passed"] is False
    call = request_client.calls[0]
    assert call["subject"] == "agent.validation.request"
    assert call["payload"]["validation_policy"] == policies["validation_policy"]
    assert call["payload"]["teacher_policy"] == policies["teacher_policy"]
    assert call["payload"]["selection_policy"] == policies["selection_policy"]
    assert call["payload"]["candidates"] == [candidate]
    assert all(
        key not in call["payload"]
        for key in ("protein_pdb_id", "boltz_ensemble_size", "boltz_max_ki_nm")
    )


@pytest.mark.asyncio
async def test_full_workflow_l0_policy_uses_validation_agent_without_boltz_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module(
        "orchestrator_full_workflow_missing_oracle_validation_agent_test",
        ROOT / "services/orchestrator-svc/src/orchestrator_svc/main.py",
    )
    original_import = builtins.__import__

    def reject_boltz_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name in {"boltz2_svc", "boltz2_svc.main"}:
            raise AssertionError("full workflow validation must not import Boltz directly")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", reject_boltz_import)

    def respond(_subject, payload):
        if payload.get("action") == "generator_coord/feedback/v1":
            return _feedback_ack(payload)
        return _validation_batch_response(
            payload,
            [_full_validation_record()],
            outcome="PASS",
        )

    request_client = _AgentRequestClientStub(respond)
    policies = _full_policy_payload()

    result = await module.FullWorkflowClients(request_client=request_client).validate_candidates(
        {
            "run_id": "run-validation-l0",
            "trace_id": "trace-validation-l0",
            "candidates": [_full_candidate()],
            "request": {"project_id": "project-1", **policies},
        }
    )

    assert result["passed"] is True
    assert result["outcome"] == "PASS"
    call = request_client.calls[0]
    assert call["subject"] == "agent.validation.request"
    assert call["payload"]["run_id"] == "run-validation-l0"
    assert call["payload"]["trace_id"] == "trace-validation-l0"
    assert call["payload"]["validation_policy"]["oracle_level"] == 0
    assert call["payload"]["candidates"] == [_full_candidate()]


@pytest.mark.asyncio
async def test_full_workflow_l0_policy_uses_real_validation_agent(
    agent_message_hmac_secret: None,
) -> None:
    from mf_agents.base.agent import BaseAgent
    from mf_agents.messaging.redis_bus import InMemoryBus
    from mf_agents.messaging.request_client import AgentRequestClient
    from validation_agent.agent import ValidationAgent

    module = _load_module(
        "orchestrator_full_workflow_null_oracle_validation_agent_test",
        ROOT / "services/orchestrator-svc/src/orchestrator_svc/main.py",
    )
    bus = InMemoryBus()
    await bus.connect()
    received_payloads: list[dict] = []

    class Oracle:
        def __init__(self) -> None:
            self.calls: list[tuple[list[str], list[str]]] = []

        async def evaluate(
            self,
            molecules: list[str],
            properties: list[str],
            *,
            request_context=None,
        ) -> dict:
            self.calls.append((list(molecules), list(properties)))
            return {molecule: {"qed": 0.9} for molecule in molecules}

    class CRGRepository:
        async def get_run_crg(self, run_id: str) -> dict:
            return {"beliefs": [], "edges": []}

        async def write_workflow_belief(self, **kwargs) -> None:
            return None

    class RecordingValidationAgent(ValidationAgent):
        async def process(self, payload):
            received_payloads.append(dict(payload))
            return await super().process(payload)

    class FeedbackAgent(BaseAgent):
        def __init__(self) -> None:
            super().__init__("generator_coord", message_bus=bus)
            self._subscription_subjects = ["agent.generator_coord.request"]

        async def process(self, payload):
            return _feedback_ack(payload)

    oracle = Oracle()
    agent = RecordingValidationAgent(
        message_bus=bus,
        oracles={"rdkit": oracle},
        crg_repository=CRGRepository(),
    )
    feedback_agent = FeedbackAgent()
    await agent.start()
    await feedback_agent.start()

    try:
        result = await module.FullWorkflowClients(
            request_client=AgentRequestClient(bus)
        ).validate_candidates(
            {
                "run_id": "run-validation-null-l0",
                "trace_id": "trace-validation-null-l0",
                "candidates": [_full_candidate()],
                "request": {
                    "project_id": "project-1",
                    **_full_policy_payload(),
                },
            }
        )
    finally:
        await bus.close()

    assert result["passed"] is True
    assert result["outcome"] == "PASS"
    assert oracle.calls == [(["CCO"], ["qed"])]
    assert len(received_payloads) == 1
    assert received_payloads[0]["validation_policy"]["oracle_level"] == 0


@pytest.mark.parametrize(
    "oracle_level",
    [0, 2, 4],
)
@pytest.mark.asyncio
async def test_full_workflow_forwards_exact_validation_policy_level(oracle_level: int) -> None:
    module = _load_module(
        f"orchestrator_full_workflow_oracle_level_{oracle_level}_test",
        ROOT / "services/orchestrator-svc/src/orchestrator_svc/main.py",
    )

    def respond(_subject, payload):
        if payload.get("action") == "generator_coord/feedback/v1":
            return _feedback_ack(payload)
        return _validation_batch_response(
            payload,
            [_full_validation_record(oracle_level=oracle_level)],
            outcome="PASS",
        )

    request_client = _AgentRequestClientStub(respond)
    policies = _full_policy_payload(oracle_level=oracle_level)

    await module.FullWorkflowClients(request_client=request_client).validate_candidates(
        {
            "run_id": f"run-validation-{oracle_level}",
            "trace_id": f"trace-validation-{oracle_level}",
            "candidates": [_full_candidate()],
            "request": {
                "project_id": "project-1",
                **policies,
            },
        }
    )

    payload = request_client.calls[0]["payload"]
    assert payload["validation_policy"] == policies["validation_policy"]
    assert payload["validation_policy"]["oracle_level"] == oracle_level


@pytest.mark.asyncio
async def test_full_workflow_validation_forwards_verified_l4_evidence_resume() -> None:
    module = _load_module(
        "orchestrator_full_workflow_validation_agent_test",
        ROOT / "services/orchestrator-svc/src/orchestrator_svc/main.py",
    )

    def respond(_subject, payload):
        if payload.get("action") == "generator_coord/feedback/v1":
            return _feedback_ack(payload)
        return _validation_batch_response(
            payload,
            [_full_validation_record(oracle_level=4)],
            outcome="PASS",
        )

    request_client = _AgentRequestClientStub(respond)
    policies = _full_policy_payload(oracle_level=4)
    awaiting_record = _full_validation_record(
        oracle_level=4,
        outcome="AWAITING_EVIDENCE",
    )
    awaiting_record["metrics"] = awaiting_record["metrics"][:-1]
    awaiting_record["levels"][-1] = {
        "level": 4,
        "outcome": "AWAITING_EVIDENCE",
        "oracles": [
            {
                "oracle": "external",
                "outcome": "AWAITING_EVIDENCE",
                "metrics": [],
                "evidence_ids": [],
                "reason": "external evidence is required",
            }
        ],
    }
    external_evidence = [
        {
            "candidate_id": "candidate-1",
            "canonical_smiles": "CCO",
            "metrics": {"experimental_activity": 0.8},
            "evidence_ids": ["external-evidence-1"],
        }
    ]

    result = await module.FullWorkflowClients(request_client=request_client).validate_candidates(
        {
            "run_id": "run-validation-cascade",
            "trace_id": "trace-validation-cascade",
            "candidates": [_full_candidate()],
            "validation": {
                "validation_schema_version": "validation.batch.v1",
                "agent": "validation_agent",
                "project_id": "project-1",
                "outcome": "AWAITING_EVIDENCE",
                "validation_policy": policies["validation_policy"],
                "records": [awaiting_record],
            },
            "external_evidence_resume_verified": True,
            "request": {
                "project_id": "project-1",
                "request_id": "request-validation-cascade",
                **policies,
                "external_evidence": external_evidence,
            },
        }
    )

    assert result["passed"] is True
    assert result["outcome"] == "PASS"
    call = request_client.calls[0]
    assert call["subject"] == "agent.validation.request"
    assert call["payload"]["project_id"] == "project-1"
    assert call["payload"]["run_id"] == "run-validation-cascade"
    assert call["payload"]["trace_id"] == "trace-validation-cascade"
    assert call["payload"]["validation_policy"] == policies["validation_policy"]
    assert call["payload"]["external_evidence"] == external_evidence
    assert call["payload"]["resume_external_evidence"] is True
    assert call["payload"]["prior_validation_records"] == [awaiting_record]


@pytest.mark.asyncio
async def test_full_workflow_validation_requires_explicit_pass_and_preserves_occurrence() -> None:
    module = _load_module(
        "orchestrator_full_validation_occurrence_contract_test",
        ROOT / "services/orchestrator-svc/src/orchestrator_svc/main.py",
    )
    record = _full_validation_record(outcome="FAIL", qed=0.2)

    def respond(_subject, payload):
        if payload.get("action") == "generator_coord/feedback/v1":
            return _feedback_ack(payload)
        return _validation_batch_response(
            payload,
            [record],
            outcome="FAIL",
            status="validated",
        )

    request_client = _AgentRequestClientStub(respond)

    result = await module.FullWorkflowClients(request_client=request_client).validate_candidates(
        {
            "run_id": "run-validation-occurrence",
            "trace_id": "trace-validation-occurrence",
            "candidates": [_full_candidate()],
            "request": {"project_id": "project-1", **_full_policy_payload()},
        }
    )

    assert result["passed"] is False
    assert result["outcome"] == "FAIL"
    assert result["results"][0]["outcome"] == "FAIL"
    assert result["results"][0]["candidate_id"] == "candidate-1"
    assert request_client.calls[0]["payload"]["candidates"] == [_full_candidate()]


@pytest.mark.asyncio
async def test_full_workflow_downstream_uses_same_passing_candidate_occurrence() -> None:
    module = _load_module(
        "orchestrator_full_candidate_occurrence_pipeline_test",
        ROOT / "services/orchestrator-svc/src/orchestrator_svc/main.py",
    )

    def respond(subject: str, payload: dict) -> dict:
        if payload.get("action") == "generator_coord/feedback/v1":
            return _feedback_ack(payload)
        if subject == "agent.validation.request":
            return _validation_batch_response(
                payload,
                [
                    _full_validation_record(
                        candidate_id="candidate-failing",
                        outcome="FAIL",
                        qed=0.1,
                    ),
                    _full_validation_record(
                        candidate_id="candidate-passing",
                        outcome="PASS",
                        qed=0.9,
                    ),
                ],
                outcome="PASS",
            )
        if subject == "agent.retrosyn.request":
            return {
                "status": "planned",
                "routes": [
                    {
                        "route_id": "route-1",
                        "building_blocks": [{"smiles": "CC"}],
                    }
                ],
            }
        if subject == "agent.supply.request":
            return {
                "status": "assessed",
                "route_id": payload["route_id"],
                "supply_assessment": {"overall_feasibility": "available"},
            }
        if subject == "agent.srb.request":
            return {
                "status": "compiled",
                "route_id": payload["route_id"],
                "protocols": [
                    {"route_id": payload["route_id"], "ssp_id": "ssp-1"}
                ],
            }
        if subject == "agent.critic.request":
            return {"verdict": "pass", "total_rules": 1}
        raise AssertionError(f"unexpected subject: {subject}")

    request_client = _AgentRequestClientStub(respond)
    state = {
        "run_id": "run-candidate-occurrence",
        "trace_id": "trace-candidate-occurrence",
        "request": {
            "project_id": "project-1",
            "retrosyn_engine": "rsgpt",
            **_full_policy_payload(),
        },
        "candidates": [
            {
                "candidate_id": "candidate-failing",
                "canonical_smiles": "CCO",
                "generator_name": "hfm_3d",
                "rank": 2,
                "marker": "failing",
                "mw": 46.0,
                "logp": 0.0,
                "tpsa": 20.0,
                "qed": 0.1,
                "sa_score": 5.0,
            },
            {
                "candidate_id": "candidate-passing",
                "canonical_smiles": "CCO",
                "generator_name": "hfm_3d",
                "rank": 1,
                "marker": "passing",
                "mw": 46.0,
                "logp": 0.0,
                "tpsa": 20.0,
                "qed": 0.9,
                "sa_score": 1.0,
            },
        ],
    }
    clients = module.FullWorkflowClients(request_client=request_client)

    state["validation"] = await clients.validate_candidates(state)
    state["retrosyn"] = await clients.plan_routes(state)
    state["supply"] = await clients.assess_supply(state)
    state["srb"] = await clients.compile_synthesis(state)
    await clients.review_candidates(state)

    assert [row["candidate_id"] for row in state["validation"]["results"]] == [
        "candidate-failing",
        "candidate-passing",
    ]
    downstream = [
        call
        for call in request_client.calls
        if call["subject"] != "agent.validation.request"
        and call["payload"].get("action") != "generator_coord/feedback/v1"
    ]
    assert [call["subject"] for call in downstream] == [
        "agent.retrosyn.request",
        "agent.supply.request",
        "agent.srb.request",
        "agent.critic.request",
    ]
    assert all(call["payload"]["candidate_id"] == "candidate-passing" for call in downstream)
    assert all(call["payload"]["candidate_index"] == 1 for call in downstream)
    assert downstream[-1]["payload"]["properties"]["marker"] == "passing"


@pytest.mark.asyncio
async def test_full_workflow_clients_plan_routes_delegates_to_retrosyn_agent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module(
        "orchestrator_full_retrosyn_client_test",
        ROOT / "services/orchestrator-svc/src/orchestrator_svc/main.py",
    )
    request_client = _AgentRequestClientStub(
        lambda subject, payload: {
            "status": "planned",
            "routes": [{"route_id": "route-1"}],
        }
    )
    monkeypatch.delenv("AIZYNTH_CONFIG_PATH", raising=False)
    state = _full_selected_state()
    state["request"]["retrosyn_max_routes"] = 2

    result = await module.FullWorkflowClients(request_client=request_client).plan_routes(state)

    assert result["routes"][0]["route_id"] == "route-1"
    assert request_client.calls[0]["subject"] == "agent.retrosyn.request"
    assert request_client.calls[0]["payload"] == {
        "project_id": "project-1",
        "run_id": "run-1",
        "trace_id": "trace-1",
        "parent_id": "run-1:retrosyn:0",
        "request_id": "run-1:retrosyn:0:candidate-0",
        "schema_version": "retrosyn.request.v1",
        "smiles": "CCO",
        "canonical_smiles": "CCO",
        "candidate_id": "candidate-1",
        "candidate_index": 0,
        "engine": "rsgpt",
        "max_routes": 2,
    }


@pytest.mark.asyncio
async def test_full_workflow_clients_assess_supply_delegates_to_supply_agent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module(
        "orchestrator_full_supply_client_test",
        ROOT / "services/orchestrator-svc/src/orchestrator_svc/main.py",
    )
    request_client = _AgentRequestClientStub(
        lambda subject, payload: {
            "status": "assessed",
            "route_id": payload["route_id"],
            "supply_assessment": {"overall_feasibility": "available"},
        }
    )
    state = _full_selected_state(
        routes=[
            {
                "route_id": "route-1",
                "building_blocks": [{"smiles": "CC"}, {"smiles": "CO"}],
            }
        ]
    )

    result = await module.FullWorkflowClients(request_client=request_client).assess_supply(state)

    assert result["supply_assessment"]["overall_feasibility"] == "available"
    assert request_client.calls[0]["subject"] == "agent.supply.request"
    assert request_client.calls[0]["payload"]["building_blocks"] == [
        {"smiles": "CC"},
        {"smiles": "CO"},
    ]
    assert request_client.calls[0]["payload"]["workflow_scope"] == "full"
    assert request_client.calls[0]["payload"]["route_id"] == "route-1"


@pytest.mark.asyncio
async def test_full_workflow_clients_preserve_supply_catalog_result() -> None:
    module = _load_module(
        "orchestrator_full_supply_local_catalog_test",
        ROOT / "services/orchestrator-svc/src/orchestrator_svc/main.py",
    )
    request_client = _AgentRequestClientStub(
        lambda subject, payload: {
            "status": "assessed",
            "route_id": payload["route_id"],
            "supply_assessment": {"overall_feasibility": "available"},
            "block_assessments": [
                {
                    "smiles": "CCO",
                    "catalog_id": "CAT-1",
                    "catalog_source": "local_catalog",
                }
            ],
        }
    )
    state = _full_selected_state(
        routes=[
            {
                "route_id": "route-1",
                "building_blocks": [{"smiles": "CCO"}],
            }
        ]
    )

    result = await module.FullWorkflowClients(request_client=request_client).assess_supply(state)

    assert result["supply_assessment"]["overall_feasibility"] == "available"
    assert result["block_assessments"][0]["catalog_id"] == "CAT-1"
    assert result["block_assessments"][0]["catalog_source"] == "local_catalog"


@pytest.mark.asyncio
async def test_full_workflow_clients_assess_supply_marks_unavailable_without_routes() -> None:
    module = _load_module(
        "orchestrator_full_supply_no_routes_test",
        ROOT / "services/orchestrator-svc/src/orchestrator_svc/main.py",
    )
    state = _full_selected_state()

    result = await module.FullWorkflowClients().assess_supply(state)

    assert result["status"] == "assessed"
    assert result["smiles"] == "CCO"
    assert result["skip_reason"] == "retrosyn.routes is empty"
    assert result["supply_assessment"]["overall_feasibility"] == "unavailable"
    assert result["block_assessments"] == []


@pytest.mark.asyncio
async def test_full_workflow_clients_review_candidates_merges_runtime_properties(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module(
        "orchestrator_full_critic_properties_test",
        ROOT / "services/orchestrator-svc/src/orchestrator_svc/main.py",
    )
    request_client = _AgentRequestClientStub(
        lambda subject, payload: {"verdict": "pass", "total_rules": 1}
    )
    candidate = _full_candidate()
    candidate.update(
        {
            "mw": 46.07,
            "logp": -0.1,
            "tpsa": 20.23,
            "qed": 0.4,
        }
    )
    state = _full_selected_state(candidate=candidate)
    state["request"].update(
        {
            "target_family": "KRAS",
            "isoform_data_count": 2,
            "kinase_selectivity_ratio": 100.0,
        }
    )
    state["validation"]["records"][0].update(
        {
            "delta_g_kcal_mol": -8.0,
            "ki_nm": 12.0,
        }
    )
    state["validation"]["results"] = list(state["validation"]["records"])
    state["supply"] = {
        "supply_assessment": {
            "total_blocks": 2,
            "commercially_available": 1,
            "supplier_diversity": 3,
            "avg_price_per_gram": 120.0,
        }
    }
    state["srb"] = {
        "protocols": [
            {
                "steps": [{"step_id": "1"}, {"step_id": "2"}],
                "total_estimated_cost_usd": 240.0,
            }
        ]
    }

    result = await module.FullWorkflowClients(request_client=request_client).review_candidates(
        state
    )

    assert result["verdict"] == "pass"
    payload = request_client.calls[0]["payload"]
    properties = payload["properties"]
    assert request_client.calls[0]["subject"] == "agent.critic.request"
    assert payload["project_id"] == "project-1"
    assert payload["run_id"] == "run-1"
    assert payload["smiles"] == "CCO"
    assert properties["mw"] == 46.07
    assert properties["delta_g_kcal_mol"] == -8.0
    assert properties["ki_nm"] == 12.0
    assert properties["building_block_availability"] == 0.5
    assert properties["critical_material_suppliers"] == 3
    assert properties["estimated_cost_per_gram"] == 120.0
    assert properties["synthesis_steps"] == 2
    assert properties["isoform_data_count"] == 2
    assert properties["kinase_selectivity_ratio"] == 100.0


@pytest.mark.asyncio
async def test_critic_agent_non_blocking_full_workflow_concerns_do_not_fail() -> None:
    module = _load_module(
        "critic_agent_blocking_rule_scope_test",
        ROOT / "agents/critic_agent/src/critic_agent/agent.py",
    )

    class Rule:
        rule_id = "rule_non_blocking"
        name = "Non-blocking concern"

        def evaluate(self, smiles, properties):
            return {
                "rule_id": self.rule_id,
                "rule_name": self.name,
                "verdict": "fail",
                "score": 0.1,
                "reasoning": "concern recorded",
            }

    agent = module.ScientificCriticAgent(crg_repository=None)
    agent.rules = [Rule()]

    result = await agent.evaluate_molecule(
        {
            "smiles": "CCO",
            "properties": {"_critic_blocking_rule_ids": []},
        }
    )

    assert result["verdict"] == "pass"
    assert result["failed"] == 1
    assert result["blocking_failed"] == 0
    assert result["non_blocking_failed"] == 1
    assert result["rule_results"][0]["blocking"] is False


@pytest.mark.asyncio
async def test_critic_agent_blocking_full_workflow_concerns_fail() -> None:
    module = _load_module(
        "critic_agent_blocking_rule_fail_test",
        ROOT / "agents/critic_agent/src/critic_agent/agent.py",
    )

    class Rule:
        rule_id = "rule_blocking"
        name = "Blocking concern"

        def evaluate(self, smiles, properties):
            return {
                "rule_id": self.rule_id,
                "rule_name": self.name,
                "verdict": "fail",
                "score": 0.1,
                "reasoning": "blocking concern",
            }

    agent = module.ScientificCriticAgent(crg_repository=None)
    agent.rules = [Rule()]

    result = await agent.evaluate_molecule(
        {
            "smiles": "CCO",
            "properties": {"_critic_blocking_rule_ids": ["rule_blocking"]},
        }
    )

    assert result["verdict"] == "fail"
    assert result["failed"] == 1
    assert result["blocking_failed"] == 1
    assert result["non_blocking_failed"] == 0
    assert result["rule_results"][0]["blocking"] is True


@pytest.mark.asyncio
async def test_critic_agent_non_blocking_crg_supply_concern_does_not_fail() -> None:
    module = _load_module(
        "critic_agent_crg_supply_non_blocking_test",
        ROOT / "agents/critic_agent/src/critic_agent/agent.py",
    )

    class CRGRepository:
        def __init__(self) -> None:
            self.beliefs: list[dict] = []

        async def get_run_crg(self, run_id: str) -> dict:
            assert run_id == "run-1"
            return {
                "beliefs": [
                    {
                        "id": "belief-supply-unavailable",
                        "subject": "CCO",
                        "predicate": "supply_feasibility",
                        "object_value": "unavailable",
                        "confidence": 1.0,
                    }
                ]
            }

        async def write_workflow_belief(self, **kwargs) -> None:
            self.beliefs.append(kwargs)

    repository = CRGRepository()
    agent = module.ScientificCriticAgent(crg_repository=repository)
    agent.rules = []

    result = await agent.evaluate_molecule(
        {
            "project_id": "project-1",
            "run_id": "run-1",
            "smiles": "CCO",
            "properties": {
                "_critic_blocking_rule_ids": [
                    "crg_validation_status",
                    "crg_retrosyn_routes",
                ]
            },
        }
    )

    assert result["verdict"] == "pass"
    assert result["failed"] == 1
    assert result["blocking_failed"] == 0
    assert result["non_blocking_failed"] == 1
    assert result["rule_results"][0]["rule_id"] == "crg_supply_feasibility"
    assert result["rule_results"][0]["blocking"] is False
    assert repository.beliefs[0]["object_value"] == "pass"
    assert repository.beliefs[0]["evidence_ids"] == ["crg_supply_feasibility"]


@pytest.mark.asyncio
async def test_critic_agent_blocking_crg_validation_status_fails_when_scoped() -> None:
    module = _load_module(
        "critic_agent_crg_validation_blocking_test",
        ROOT / "agents/critic_agent/src/critic_agent/agent.py",
    )

    class CRGRepository:
        def __init__(self) -> None:
            self.beliefs: list[dict] = []

        async def get_run_crg(self, run_id: str) -> dict:
            assert run_id == "run-1"
            return {
                "beliefs": [
                    {
                        "id": "belief-validation-failed",
                        "subject": "CCO",
                        "predicate": "validation_status",
                        "object_value": "failed",
                        "confidence": 1.0,
                    }
                ]
            }

        async def write_workflow_belief(self, **kwargs) -> None:
            self.beliefs.append(kwargs)

    repository = CRGRepository()
    agent = module.ScientificCriticAgent(crg_repository=repository)
    agent.rules = []

    result = await agent.evaluate_molecule(
        {
            "project_id": "project-1",
            "run_id": "run-1",
            "smiles": "CCO",
            "properties": {"_critic_blocking_rule_ids": ["crg_validation_status"]},
        }
    )

    assert result["verdict"] == "fail"
    assert result["failed"] == 1
    assert result["blocking_failed"] == 1
    assert result["rule_results"][0]["rule_id"] == "crg_validation_status"
    assert result["rule_results"][0]["blocking"] is True


@pytest.mark.asyncio
async def test_full_workflow_clients_compile_synthesis_delegates_to_srb_agent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module(
        "orchestrator_full_srb_client_test",
        ROOT / "services/orchestrator-svc/src/orchestrator_svc/main.py",
    )
    request_client = _AgentRequestClientStub(
        lambda subject, payload: {
            "status": "compiled",
            "route_id": payload["route_id"],
            "protocols": [
                {"route_id": payload["route_id"], "ssp_id": "ssp-1"}
            ],
        }
    )
    route = {
        "route_id": "route-1",
        "steps": [
            {
                "step_id": "retro-1",
                "reaction": "CC.O>>CCO",
                "reaction_type": "oxidation",
                "reactants": [{"smiles": "CC"}],
            }
        ],
    }
    state = _full_selected_state(routes=[route])
    state["supply"] = {
        "route_id": "route-1",
        "supply_assessment": {"overall_feasibility": "available"},
    }

    result = await module.FullWorkflowClients(request_client=request_client).compile_synthesis(
        state
    )

    assert result["protocols"][0]["ssp_id"] == "ssp-1"
    assert request_client.calls[0]["subject"] == "agent.srb.request"
    assert request_client.calls[0]["payload"]["project_id"] == "project-1"
    assert request_client.calls[0]["payload"]["run_id"] == "run-1"
    assert request_client.calls[0]["payload"]["molecule"] == {"smiles": "CCO"}
    assert request_client.calls[0]["payload"]["candidate_id"] == "candidate-1"
    assert request_client.calls[0]["payload"]["canonical_smiles"] == "CCO"
    assert request_client.calls[0]["payload"]["pathways"] == [route]
    assert request_client.calls[0]["payload"]["route_id"] == "route-1"


def test_orchestrator_deployment_wires_sila2_adapter_env() -> None:
    compose = (ROOT / "infra/docker/docker-compose.dev.yml").read_text(encoding="utf-8")
    k8s = (ROOT / "infra/kubernetes/deployments/moleculeforge-services.yaml").read_text(
        encoding="utf-8"
    )
    helm_values = (ROOT / "infra/helm/moleculeforge/values.yaml").read_text(encoding="utf-8")

    for env_name in ("SILA2_PLAN_COMMAND", "SILA2_PLAN_TIMEOUT_SECONDS"):
        assert env_name in compose
        assert env_name in k8s
        assert env_name in helm_values

    assert "SILA2_PLAN_TIMEOUT_SECONDS: ${SILA2_PLAN_TIMEOUT_SECONDS:-120}" in compose
    assert "name: sila2-adapter-config" in k8s
    assert "configMapKeyRef:" in k8s
    assert "envValueFrom:" in helm_values


def test_full_workflow_dependency_env_is_owned_by_runtime_consumer() -> None:
    import yaml

    compose_config = yaml.safe_load(
        (ROOT / "infra/docker/docker-compose.dev.yml").read_text(encoding="utf-8")
    )
    k8s_docs = list(
        yaml.safe_load_all(
            (ROOT / "infra/kubernetes/deployments/moleculeforge-services.yaml").read_text(
                encoding="utf-8"
            )
        )
    )
    helm_values = yaml.safe_load(
        (ROOT / "infra/helm/moleculeforge/values.yaml").read_text(encoding="utf-8")
    )

    deployments = {
        item["metadata"]["name"]: item
        for item in k8s_docs
        if item and item.get("kind") == "Deployment"
    }

    def deployment_env(service_name: str) -> tuple[set[str], set[str], set[str]]:
        compose_env = set(compose_config["services"][service_name].get("environment", {}))
        container = deployments[service_name]["spec"]["template"]["spec"]["containers"][0]
        k8s_env = {item["name"] for item in container.get("env", [])}
        helm_service = helm_values["services"][service_name]
        helm_env = set(helm_service.get("env", {})) | set(helm_service.get("envValueFrom", {}))
        return compose_env, k8s_env, helm_env

    ownership = {
        "generator-coord-agent": {
            "GENERATOR_DISCOVERY_URI",
            "GENERATOR_DISCOVERY_TIMEOUT_SECONDS",
            "GENERATOR_CLIENT_TARGETS",
            "HFM_3D_GENERATOR_TARGET",
            "FRAGFM_GENERATOR_TARGET",
            "CREM_3D_GENERATOR_TARGET",
            "MMPT_RAG_GENERATOR_TARGET",
            "ICLM_GENERATOR_TARGET",
            "UAS_GENERATOR_TARGET",
            "UAS_RUNNER_COMMAND",
            "UAS_RUNNER_TIMEOUT_SECONDS",
        },
        "retrosyn-agent": {
            "RETROSYN_PLANNER_COMMAND",
            "RETROSYN_PLANNER_COMMANDS_JSON",
            "RASCORE_PLANNER_COMMAND",
            "RSGPT_PLANNER_COMMAND",
            "UALIGN_PLANNER_COMMAND",
            "AIZYNTH_PLANNER_COMMAND",
            "AIZYNTH_CONFIG_PATH",
            "RETROSYN_PLANNER_COMMAND_TIMEOUT_SECONDS",
            "HUMU_ENCODER_TARGET",
        },
        "supply-agent": {"SUPPLY_ORACLE_TARGET"},
        "srb-agent": {"SILA2_PLAN_COMMAND", "SILA2_PLAN_TIMEOUT_SECONDS"},
    }
    moved_env: set[str] = set()
    for service_name, required_env in ownership.items():
        moved_env.update(required_env)
        for configured_env in deployment_env(service_name):
            assert required_env <= configured_env

    orchestrator_env = deployment_env("orchestrator-svc")
    for configured_env in orchestrator_env:
        assert {"AIZYNTH_CONFIG_PATH", "HUMU_ENCODER_TARGET"} <= configured_env
        assert not (moved_env - {"AIZYNTH_CONFIG_PATH", "HUMU_ENCODER_TARGET"}) & configured_env
    assert "SUPPLY_ORACLE_TARGET: ${SUPPLY_ORACLE_TARGET:-supply-oracle-svc:50059}" in (
        ROOT / "infra/docker/docker-compose.dev.yml"
    ).read_text(encoding="utf-8")
    assert "HUMU_ENCODER_TARGET: ${HUMU_ENCODER_TARGET:-humu-encoder-svc:50051}" in (
        ROOT / "infra/docker/docker-compose.dev.yml"
    ).read_text(encoding="utf-8")


def test_oracle_deployments_wire_external_runner_env() -> None:
    compose = (ROOT / "infra/docker/docker-compose.dev.yml").read_text(encoding="utf-8")
    k8s = (ROOT / "infra/kubernetes/deployments/moleculeforge-services.yaml").read_text(
        encoding="utf-8"
    )
    helm_values = (ROOT / "infra/helm/moleculeforge/values.yaml").read_text(encoding="utf-8")

    for env_name in (
        "ADMET_ORACLE_COMMAND",
        "ADMET_ORACLE_TIMEOUT_SECONDS",
        "ADMET_MODEL_PATH",
        "ADMET_SERVICE_URL",
        "ADMET_TARGETS",
        "ADMET_BATCH_SIZE",
        "DOCK_ORACLE_COMMAND",
        "DOCK_ORACLE_TIMEOUT_SECONDS",
        "GNINA_BINARY",
        "DIFFDOCK_MODEL_PATH",
        "BOLTZ2_ORACLE_COMMAND",
        "BOLTZ2_ORACLE_TIMEOUT_SECONDS",
        "BOLTZ2_PROTEIN_PDB_ID",
        "BOLTZ2_ENSEMBLE_SIZE",
        "BOLTZ_MODEL_PATH",
        "BOLTZ_INPUT_TEMPLATE_DIR",
        "BOLTZ_WORK_DIR",
        "BOLTZ_BINARY",
        "FEP_ORACLE_COMMAND",
        "FEP_ORACLE_TIMEOUT_SECONDS",
        "FEP_REFERENCE_LIGAND_SMILES",
        "FEP_METHOD",
        "FEP_N_REPEATS",
        "OPENFE_RUNNER_PATH",
        "OPENFE_RUNNER_TIMEOUT_SECONDS",
        "OPENFE_CLI_PATH",
        "OPENFE_QUICKRUN_TIMEOUT_SECONDS",
        "OPENFE_GATHER_TIMEOUT_SECONDS",
        "OPENFE_MAX_TRANSFORMATIONS_PER_PAIR",
        "OPENFE_RESULT_REPLAY_PATH",
        "OPENFE_RESULT_REGISTRY",
        "OPENFE_TRANSFORMATION_REGISTRY",
        "OPENFE_WORK_DIR",
    ):
        assert env_name in compose
        assert env_name in k8s
        assert env_name in helm_values

    assert "ADMET_ORACLE_TIMEOUT_SECONDS: ${ADMET_ORACLE_TIMEOUT_SECONDS:-120}" in compose
    assert "ADMET_BATCH_SIZE: ${ADMET_BATCH_SIZE:-64}" in compose
    assert "DOCK_ORACLE_TIMEOUT_SECONDS: ${DOCK_ORACLE_TIMEOUT_SECONDS:-120}" in compose
    assert "GNINA_BINARY: ${GNINA_BINARY:-models/artifacts/gnina/gnina.1.3.2.cuda12.8}" in compose
    assert "DIFFDOCK_MODEL_PATH: ${DIFFDOCK_MODEL_PATH:-models/artifacts/diffdock}" in compose
    assert "BOLTZ_MODEL_PATH: ${BOLTZ_MODEL_PATH:-models/artifacts/boltz-2}" in compose
    assert (
        "BOLTZ_INPUT_TEMPLATE_DIR: "
        "${BOLTZ_INPUT_TEMPLATE_DIR:-models/artifacts/boltz-input-templates}"
    ) in compose
    assert "BOLTZ2_ORACLE_TIMEOUT_SECONDS: ${BOLTZ2_ORACLE_TIMEOUT_SECONDS:-300}" in compose
    assert "BOLTZ2_ENSEMBLE_SIZE: ${BOLTZ2_ENSEMBLE_SIZE:-5}" in compose
    assert "BOLTZ_WORK_DIR: ${BOLTZ_WORK_DIR:-runs/boltz2}" in compose
    assert "BOLTZ_BINARY: ${BOLTZ_BINARY:-boltz}" in compose
    assert "FEP_ORACLE_TIMEOUT_SECONDS: ${FEP_ORACLE_TIMEOUT_SECONDS:-}" in compose
    assert "FEP_METHOD: ${FEP_METHOD:-openfe}" in compose
    assert "FEP_N_REPEATS: ${FEP_N_REPEATS:-1}" in compose
    assert "FEP_ORACLE_COMMAND: python -m fep_svc.main --validation-runner" in compose
    assert (
        "OPENFE_RUNNER_PATH: "
        "${OPENFE_RUNNER_PATH:-python tools/oracles/openfe_json_runner.py}"
    ) in compose
    assert "OPENFE_CLI_PATH: ${OPENFE_CLI_PATH:-/opt/openfe/bin/openfe}" in compose
    assert (
        "OPENFE_TRANSFORMATION_REGISTRY: "
        "${OPENFE_TRANSFORMATION_REGISTRY-/var/lib/moleculeforge/fep/input/"
        "transformation-registry.json}"
    ) in compose
    assert "FEP_JOB_DIR: /var/lib/moleculeforge/fep/jobs" in compose
    assert "image: moleculeforge/oracle:latest" in k8s
    assert "claimName: fep-data" in k8s
    assert "repository: oracle" in helm_values
    assert "name: oracle-runner-config" in k8s
    assert "configMapKeyRef:" in k8s
    assert "envValueFrom:" in helm_values
    for config in (
        _k8s_configmap_data(k8s, "mf-oracles", "oracle-runner-config"),
        _k8s_configmap_data(k8s, "mf-agents", "oracle-runner-config"),
        _helm_configmap_data(helm_values, "mf-oracles", "oracle-runner-config"),
        _helm_configmap_data(helm_values, "mf-agents", "oracle-runner-config"),
    ):
        assert config["admet-oracle-command"] == ""
        assert config["admet-oracle-timeout-seconds"] == "120"
        assert config["admet-batch-size"] == "64"
        assert config["dock-oracle-command"] == (
            "python tools/oracles/dock_oracle_wrapper.py"
        )
        assert config["dock-oracle-timeout-seconds"] == "120"
        assert config["gnina-binary"] == "gnina"
        assert config["diffdock-model-path"] == "models/artifacts/diffdock"
        assert config["boltz2-oracle-command"] == ""
        assert config["boltz2-oracle-timeout-seconds"] == "300"
        assert config["boltz2-ensemble-size"] == "5"
        assert config["boltz-model-path"] == "models/artifacts/boltz-2"
        assert config["boltz-input-template-dir"] == "models/artifacts/boltz-input-templates"
        assert config["boltz-work-dir"] == "runs/boltz2"
        assert config["boltz-binary"] == "boltz"
        assert config["fep-oracle-command"] == (
            "python tools/oracles/fep_oracle_wrapper.py"
        )
        assert config["fep-oracle-timeout-seconds"] == ""
        assert config["fep-method"] == "openfe"
        assert config["fep-n-repeats"] == "1"
        assert config["openfe-runner-path"] == (
            "python tools/oracles/openfe_json_runner.py"
        )
        assert config["openfe-runner-timeout-seconds"] == ""
        assert config["openfe-cli-path"] == "/opt/openfe/bin/openfe"
        assert config["openfe-quickrun-timeout-seconds"] == "3600"
        assert config["openfe-gather-timeout-seconds"] == "600"
        assert config["openfe-max-transformations-per-pair"] == "2"
        assert config["openfe-result-replay-path"] == ""
        assert config["openfe-result-registry"] == ""
        assert config["openfe-transformation-registry"] == (
            "/var/lib/moleculeforge/fep/input/transformation-registry.json"
        )
        assert config["openfe-work-dir"] == "/var/lib/moleculeforge/fep/work"
        assert config["l4-quantum-oracle-command"] == ""
        assert config["l4-quantum-engine"] == "quantum"
        assert config["l4-gpu4pyscf-command"] == ""
        assert config["l4-orca-command"] == ""


def test_oracle_image_uses_official_openfe_distribution_and_installs_services() -> None:
    import tomllib

    base = (ROOT / "infra/docker/base/Dockerfile.base").read_text(encoding="utf-8")
    oracle = (ROOT / "infra/docker/base/Dockerfile.oracle").read_text(encoding="utf-8")
    admet_project = tomllib.loads(
        (ROOT / "services/admet-svc/pyproject.toml").read_text(encoding="utf-8")
    )

    assert "uv venv --python 3.12.13 --seed /opt/venv" in base
    assert "FROM nvidia/cuda:13.0.2-cudnn-runtime-ubuntu22.04" in base
    assert '"openfe=1.12.0"' in oracle
    assert "micromamba create" in oracle
    assert "openfe>=1.0" not in oracle
    assert "ARG TARGETARCH=amd64" in oracle
    assert 'test "${TARGETARCH}" = "amd64"' in oracle
    assert "--extra oracle-runtime" in oracle
    for module_name in (
        "admet_svc.main",
        "boltz2_svc.main",
        "dock_svc.main",
        "fep_svc.main",
        "retrosyn_svc.main",
        "supply_oracle_svc.main",
        "mf_oracles.admet_ai.oracle",
    ):
        assert module_name in oracle
    assert "mf-oracles-admet-ai" in admet_project["project"]["dependencies"]


def test_oracle_runtime_closes_real_adapter_dependencies() -> None:
    import tomllib

    from packaging.requirements import Requirement

    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    rsgpt_project = tomllib.loads(
        (
            ROOT / "models/mf-retrosyn/rsgpt/pyproject.toml"
        ).read_text(encoding="utf-8")
    )
    ualign_project = tomllib.loads(
        (
            ROOT / "models/mf-retrosyn/ualign/pyproject.toml"
        ).read_text(encoding="utf-8")
    )

    oracle_runtime = set(project["project"]["optional-dependencies"]["oracle-runtime"])
    assert {"mf-retrosyn-rsgpt", "mf-retrosyn-ualign"} <= oracle_runtime

    rsgpt_dependencies = {
        Requirement(value).name for value in rsgpt_project["project"]["dependencies"]
    }
    assert {
        "einops",
        "numpy",
        "omegaconf",
        "pytorch-lightning",
        "rdkit",
        "torch",
        "transformers",
        "wandb",
    } <= rsgpt_dependencies

    ualign_dependencies = {
        Requirement(value).name for value in ualign_project["project"]["dependencies"]
    }
    assert {
        "numpy",
        "ogb",
        "pandas",
        "rdkit",
        "torch",
        "torch-geometric",
    } <= ualign_dependencies


def test_oracle_image_probes_real_adapter_entrypoints() -> None:
    oracle = (ROOT / "infra/docker/base/Dockerfile.oracle").read_text(encoding="utf-8")

    assert "boltz[cuda]==2.2.1" in oracle
    assert '"torch==2.11.0"' in oracle
    assert "aizynthfinder[all]==4.4.1" in oracle
    assert 'AIZYNTH_PYTHON="/opt/aizynth-runtime/bin/python"' in oracle
    assert "boltz --help" in oracle
    assert "from aizynthfinder.aizynthfinder import AiZynthFinder" in oracle
    assert "from transformers import LlamaForCausalLM, PreTrainedTokenizerFast" in oracle
    assert "from omegaconf import OmegaConf" in oracle
    assert "import ogb, torch_geometric" in oracle


def test_oracle_service_images_and_fep_gpu_resources_match_built_runtimes() -> None:
    import yaml

    compose = yaml.safe_load(
        (ROOT / "infra/docker/docker-compose.dev.yml").read_text(encoding="utf-8")
    )
    helm = yaml.safe_load(
        (ROOT / "infra/helm/moleculeforge/values.yaml").read_text(encoding="utf-8")
    )
    k8s_documents = list(
        yaml.safe_load_all(
            (
                ROOT / "infra/kubernetes/deployments/moleculeforge-services.yaml"
            ).read_text(encoding="utf-8")
        )
    )
    deployments = {
        document["metadata"]["name"]: document
        for document in k8s_documents
        if isinstance(document, dict) and document.get("kind") == "Deployment"
    }
    oracle_services = (
        "admet-svc",
        "boltz2-svc",
        "dock-svc",
        "fep-svc",
        "retrosyn-svc",
        "supply-oracle-svc",
    )

    for service_name in oracle_services:
        assert compose["services"][service_name]["image"] == "moleculeforge/oracle:dev"
        assert (
            helm["services"][service_name]["image"]["repository"]
            == "oracle"
        )
        container = deployments[service_name]["spec"]["template"]["spec"]["containers"][0]
        assert container["image"] == "moleculeforge/oracle:latest"

    agent_runtime_services = (
        "api-gateway",
        "cig-compiler-svc",
        "critic-svc",
        "feature-store-svc",
        "generator-router-svc",
        "hypseek-teacher-svc",
        "humu-index-svc",
        "nl2obj-svc",
        "orchestrator-svc",
        "pareto-bo-svc",
        "provenance-svc",
    )
    for service_name in agent_runtime_services:
        assert (
            compose["services"][service_name]["image"]
            == "moleculeforge/agent-runtime:dev"
        )
        assert (
            helm["services"][service_name]["image"]["repository"]
            == "agent-runtime"
        )
        container = deployments[service_name]["spec"]["template"]["spec"]["containers"][0]
        assert container["image"] == "moleculeforge/agent-runtime:latest"

    expected_gpu = {"nvidia.com/gpu": 1}
    assert helm["services"]["fep-svc"]["resources"]["requests"] == expected_gpu
    assert helm["services"]["fep-svc"]["resources"]["limits"] == expected_gpu
    fep_resources = deployments["fep-svc"]["spec"]["template"]["spec"]["containers"][0][
        "resources"
    ]
    assert fep_resources["requests"] == expected_gpu
    assert fep_resources["limits"] == expected_gpu


def test_agent_services_are_installed_in_shared_runtime_with_cosign() -> None:
    import tomllib

    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    optional = project["project"]["optional-dependencies"]
    agent = (ROOT / "infra/docker/base/Dockerfile.agent").read_text(encoding="utf-8")
    oracle = (ROOT / "infra/docker/base/Dockerfile.oracle").read_text(encoding="utf-8")

    service_packages = {
        "api-gateway": "api_gateway.main",
        "cig-compiler-svc": "cig_compiler_svc.main",
        "critic-svc": "critic_svc.main",
        "feature-store-svc": "feature_store_svc.main",
        "generator-router-svc": "generator_router_svc.main",
        "humu-index-svc": "humu_index_svc.main",
        "nl2obj-svc": "nl2obj_svc.main",
        "orchestrator-svc": "orchestrator_svc.main",
        "pareto-bo": "pareto_bo.service",
        "provenance-svc": "provenance_svc.main",
    }
    assert service_packages.keys() <= set(optional["agent-runtime"])
    assert "provenance-svc" not in optional["oracle-runtime"]
    for import_name in service_packages.values():
        assert import_name in agent
    assert "provenance_svc" not in oracle


def test_minimal_compose_uses_executable_service_modules() -> None:
    import yaml

    compose = yaml.safe_load(
        (ROOT / "infra/docker/docker-compose.minimal.yml").read_text(encoding="utf-8")
    )

    assert compose["services"]["api-gateway"]["command"] == [
        "python",
        "-m",
        "api_gateway.main",
    ]
    humu = compose["services"]["humu-encoder-svc"]
    assert humu["command"][:2] == ["sh", "-c"]
    assert "--bootstrap-validation-checkpoint" in humu["command"][2]
    assert "exec python -m humu_encoder_svc.main" in humu["command"][2]
    assert humu["environment"]["HUMU_CHECKPOINT_PATH"] == (
        "/var/lib/moleculeforge/validation-artifacts/humu/humu.pt"
    )
    assert humu["environment"]["HUMU_ALLOW_VALIDATION_ARTIFACT"] == "true"
    assert humu["environment"]["HUMU_DEVICE"] == "cpu"
    assert compose["services"]["humu-encoder-svc"]["ports"] == ["50051:50051"]


def test_generator_validation_artifacts_are_bootstrapped_only_in_development() -> None:
    import yaml

    compose = yaml.safe_load(
        (ROOT / "infra/docker/docker-compose.dev.yml").read_text(encoding="utf-8")
    )
    helm = yaml.safe_load(
        (ROOT / "infra/helm/moleculeforge/values.yaml").read_text(encoding="utf-8")
    )
    kubernetes_documents = list(
        yaml.safe_load_all(
            (
                ROOT / "infra/kubernetes/deployments/moleculeforge-services.yaml"
            ).read_text(encoding="utf-8")
        )
    )
    kubernetes_deployments = {
        document["metadata"]["name"]: document
        for document in kubernetes_documents
        if isinstance(document, dict) and document.get("kind") == "Deployment"
    }
    kubernetes_config_maps = {
        (document["metadata"]["namespace"], document["metadata"]["name"]): document[
            "data"
        ]
        for document in kubernetes_documents
        if isinstance(document, dict) and document.get("kind") == "ConfigMap"
    }
    helm_config_maps = {
        (config["namespace"], config["name"]): config["data"]
        for config in helm["configMaps"].values()
    }

    def helm_env(service: dict[str, object], variable: str) -> object:
        direct = service.get("env", {})
        if variable in direct:
            return direct[variable]
        reference = service["envValueFrom"][variable]["configMapKeyRef"]
        return helm_config_maps[(service["namespace"], reference["name"])][
            reference["key"]
        ]

    def kubernetes_env(
        deployment: dict[str, object],
        variable: str,
    ) -> object:
        namespace = deployment["metadata"]["namespace"]
        container = deployment["spec"]["template"]["spec"]["containers"][0]
        item = next(item for item in container["env"] if item["name"] == variable)
        if "value" in item:
            return item["value"]
        reference = item["valueFrom"]["configMapKeyRef"]
        return kubernetes_config_maps[(namespace, reference["name"])][reference["key"]]

    expected = {
        "hfm-generator-svc": {
            "module": "hfm_generator_svc.main",
            "allow": "HFM_ALLOW_VALIDATION_ARTIFACT",
            "directory": "/var/lib/moleculeforge/validation-artifacts/hfm",
            "paths": {
                "HFM_CHECKPOINT_PATH": "hfm_checkpoint.pt",
                "HFM_DECODER_PATH": "decoder.json",
            },
            "bootstrap": "--bootstrap-validation-artifacts",
        },
        "crem-generator-svc": {
            "module": "crem_generator_svc.main",
            "allow": "CREM_ALLOW_VALIDATION_ARTIFACT",
            "directory": "/var/lib/moleculeforge/validation-artifacts/crem",
            "paths": {"CREM_MMP_DB_PATH": "crem_mmp_database.json"},
            "bootstrap": "--bootstrap-validation-artifacts",
        },
        "fragfm-generator-svc": {
            "module": "fragfm_generator_svc.main",
            "allow": "FRAGFM_ALLOW_VALIDATION_ARTIFACT",
            "directory": "/var/lib/moleculeforge/validation-artifacts/fragfm",
            "paths": {
                "FRAGFM_VOCAB_PATH": "vocab.json",
                "FRAGFM_RATE_MATRIX_PATH": "rate_matrix.pt",
            },
            "bootstrap": "--bootstrap-validation-artifacts",
        },
        "mmpt-generator-svc": {
            "module": "mmpt_generator_svc.main",
            "allow": "MMPT_ALLOW_VALIDATION_ARTIFACT",
            "directory": "/var/lib/moleculeforge/validation-artifacts/mmpt",
            "paths": {"MMPT_INDEX_URI": "mmpt_index.json"},
            "bootstrap": "--bootstrap-validation-artifacts",
        },
        "humu-encoder-svc": {
            "module": "humu_encoder_svc.main",
            "allow": "HUMU_ALLOW_VALIDATION_ARTIFACT",
            "directory": "/var/lib/moleculeforge/validation-artifacts/humu",
            "paths": {"HUMU_CHECKPOINT_PATH": "humu.pt"},
            "bootstrap": "--bootstrap-validation-checkpoint",
        },
        "uas-generator-svc": {
            "module": "uas_generator_svc.main",
            "allow": "UAS_ALLOW_VALIDATION_ARTIFACT",
            "directory": "/var/lib/moleculeforge/validation-artifacts/uas",
            "paths": {
                "UAS_AUTOENCODER_PATH": "autoencoder.pt",
                "UAS_ARTIFACT_MANIFEST_PATH": "training_manifest.json",
            },
            "bootstrap": "bootstrap-validation-artifacts",
        },
    }

    for service_name, contract in expected.items():
        compose_service = compose["services"][service_name]
        helm_service = helm["services"][service_name]
        kubernetes_container = kubernetes_deployments[service_name]["spec"][
            "template"
        ]["spec"]["containers"][0]
        directory = contract["directory"]

        assert compose_service["environment"][contract["allow"]] == "true"
        assert contract["allow"] not in helm_service.get("env", {})
        assert contract["allow"] not in helm_service.get("envValueFrom", {})
        assert contract["allow"] not in {
            item["name"] for item in kubernetes_container.get("env", [])
        }

        for variable, filename in contract["paths"].items():
            expected_path = f"{directory}/{filename}"
            if variable == "MMPT_INDEX_URI":
                expected_path = f"file://{expected_path}"
            assert compose_service["environment"][variable] == expected_path
            assert helm_env(helm_service, variable) == ""
            assert kubernetes_env(
                kubernetes_deployments[service_name], variable
            ) == ""

        compose_command = compose_service["command"]
        assert compose_command[:2] == ["sh", "-c"]
        assert contract["bootstrap"] in compose_command[2]
        if service_name != "humu-encoder-svc":
            assert directory in compose_command[2]
        assert "exec python -m" in compose_command[2]
        production_command = ["python", "-m", contract["module"]]
        assert helm_service["command"] == production_command
        assert kubernetes_container["command"] == production_command

    assert compose["services"]["fragfm-generator-svc"]["environment"][
        "FRAGFM_CHECKPOINT_PATH"
    ] == ""
    assert (
        helm_env(
            helm["services"]["fragfm-generator-svc"], "FRAGFM_CHECKPOINT_PATH"
        )
        == ""
    )
    assert (
        kubernetes_env(
            kubernetes_deployments["fragfm-generator-svc"],
            "FRAGFM_CHECKPOINT_PATH",
        )
        == ""
    )

    uas_commands = {
        "UAS_CANDIDATE_SOURCE_COMMAND": (
            "python -m uas_generator_svc.main validation-candidate"
        ),
        "UAS_DECODER_COMMAND": "python -m uas_generator_svc.main validation-decoder",
    }
    for variable, expected_command in uas_commands.items():
        assert compose["services"]["uas-generator-svc"]["environment"][
            variable
        ] == expected_command
        assert helm_env(helm["services"]["uas-generator-svc"], variable) == ""
        assert kubernetes_env(kubernetes_deployments["uas-generator-svc"], variable) == ""

    assert compose["services"]["uas-generator-svc"]["ports"] == ["50068:50068"]
    assert helm["services"]["uas-generator-svc"]["ports"] == [
        {"name": "grpc", "port": 50068}
    ]
    assert compose["services"]["generator-coord-agent"]["environment"][
        "UAS_GENERATOR_TARGET"
    ] == "uas-generator-svc:50068"
    assert helm["services"]["generator-coord-agent"]["env"][
        "UAS_GENERATOR_TARGET"
    ] == "uas-generator-svc.mf-generators.svc.cluster.local:50068"
    generator_coordinator = kubernetes_deployments["generator-coord-agent"]
    coordinator_env = {
        item["name"]: item.get("value")
        for item in generator_coordinator["spec"]["template"]["spec"]["containers"][
            0
        ]["env"]
    }
    assert coordinator_env["UAS_GENERATOR_TARGET"] == (
        "uas-generator-svc.mf-generators.svc.cluster.local:50068"
    )


def test_generator_namespace_packages_are_importable_without_pythonpath() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import mf_generators.crem_3d, mf_generators.fragfm, "
                "mf_generators.hfm_3d, mf_generators.mmpt_rag, "
                "mf_generators.rdkit_random, mf_generators.uas"
            ),
        ],
        cwd=ROOT,
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_synthetic_oracles_are_confined_to_dev_compose() -> None:
    import yaml

    compose = yaml.safe_load(
        (ROOT / "infra/docker/docker-compose.dev.yml").read_text(encoding="utf-8")
    )
    helm = yaml.safe_load(
        (ROOT / "infra/helm/moleculeforge/values.yaml").read_text(encoding="utf-8")
    )
    kubernetes_documents = list(
        yaml.safe_load_all(
            (
                ROOT / "infra/kubernetes/deployments/moleculeforge-services.yaml"
            ).read_text(encoding="utf-8")
        )
    )
    deployments = {
        document["metadata"]["name"]: document
        for document in kubernetes_documents
        if isinstance(document, dict) and document.get("kind") == "Deployment"
    }
    kubernetes_config_maps = {
        (document["metadata"]["namespace"], document["metadata"]["name"]): document[
            "data"
        ]
        for document in kubernetes_documents
        if isinstance(document, dict) and document.get("kind") == "ConfigMap"
    }
    helm_config_maps = {
        (config["namespace"], config["name"]): config["data"]
        for config in helm["configMaps"].values()
    }
    expected = {
        "admet-svc": (
            "ADMET_ORACLE_COMMAND",
            "python -m admet_svc.main --validation-runner",
            "",
            "oracle-runner-config",
            "admet-oracle-command",
        ),
        "boltz2-svc": (
            "BOLTZ2_ORACLE_COMMAND",
            "python -m boltz2_svc.main --validation-runner",
            "",
            "oracle-runner-config",
            "boltz2-oracle-command",
        ),
        "dock-svc": (
            "DOCK_ORACLE_COMMAND",
            "python -m dock_svc.main --validation-runner",
            "python tools/oracles/dock_oracle_wrapper.py",
            "oracle-runner-config",
            "dock-oracle-command",
        ),
        "fep-svc": (
            "FEP_ORACLE_COMMAND",
            "python -m fep_svc.main --validation-runner",
            "python tools/oracles/fep_oracle_wrapper.py",
            "oracle-runner-config",
            "fep-oracle-command",
        ),
        "retrosyn-svc": (
            "RETROSYN_PLANNER_COMMAND",
            "python -m retrosyn_svc.main --validation-runner",
            "",
            "retrosyn-planner-config",
            "planner-command",
        ),
    }

    for service_name, (
        variable,
        validation_command,
        production_command,
        config_map_name,
        config_key,
    ) in expected.items():
        compose_service = compose["services"][service_name]
        helm_service = helm["services"][service_name]
        deployment = deployments[service_name]
        namespace = deployment["metadata"]["namespace"]
        container = deployment["spec"]["template"]["spec"]["containers"][0]
        container_env = {item["name"]: item for item in container["env"]}

        assert compose_service["environment"][variable] == validation_command
        assert compose_service["environment"]["MF_ALLOW_SYNTHETIC_VALIDATION"] == "true"
        assert "MF_ALLOW_SYNTHETIC_VALIDATION" not in helm_service["env"]
        assert "MF_ALLOW_SYNTHETIC_VALIDATION" not in container_env

        helm_reference = helm_service["envValueFrom"][variable]["configMapKeyRef"]
        assert helm_reference == {"name": config_map_name, "key": config_key}
        assert helm_config_maps[(helm_service["namespace"], config_map_name)][
            config_key
        ] == production_command
        kubernetes_reference = container_env[variable]["valueFrom"]["configMapKeyRef"]
        assert kubernetes_reference == {"name": config_map_name, "key": config_key}
        assert kubernetes_config_maps[(namespace, config_map_name)][
            config_key
        ] == production_command

    supply_path = "/var/lib/moleculeforge/validation-artifacts/supply/catalog.json"
    supply_uri = f"file://{supply_path}"
    supply_compose = compose["services"]["supply-oracle-svc"]
    supply_helm = helm["services"]["supply-oracle-svc"]
    supply_deployment = deployments["supply-oracle-svc"]
    supply_container = supply_deployment["spec"]["template"]["spec"]["containers"][0]
    supply_env = {item["name"]: item for item in supply_container["env"]}

    assert supply_compose["environment"]["SUPPLY_CATALOG_URI"] == supply_uri
    assert supply_compose["environment"]["MF_ALLOW_SYNTHETIC_VALIDATION"] == "true"
    assert "MF_ALLOW_SYNTHETIC_VALIDATION" not in supply_helm["env"]
    assert "MF_ALLOW_SYNTHETIC_VALIDATION" not in supply_env
    validation_command = supply_compose["command"]
    assert validation_command[:2] == ["sh", "-c"]
    assert "--bootstrap-validation-catalog" in validation_command[2]
    assert supply_path in validation_command[2]
    assert "exec python -m supply_oracle_svc.main" in validation_command[2]
    assert supply_helm["command"] == ["python", "-m", "supply_oracle_svc.main"]
    assert supply_container["command"] == ["python", "-m", "supply_oracle_svc.main"]
    assert helm_config_maps[("mf-oracles", "supply-oracle-config")][
        "catalog-uri"
    ] == ""
    assert kubernetes_config_maps[("mf-oracles", "supply-oracle-config")][
        "catalog-uri"
    ] == ""

    assert compose["services"]["retrosyn-agent"]["environment"][
        "MF_ALLOW_SYNTHETIC_VALIDATION"
    ] == "true"
    assert "MF_ALLOW_SYNTHETIC_VALIDATION" not in helm["services"]["retrosyn-agent"][
        "env"
    ]
    retrosyn_agent_env = {
        item["name"]: item
        for item in deployments["retrosyn-agent"]["spec"]["template"]["spec"][
            "containers"
        ][0]["env"]
    }
    assert "MF_ALLOW_SYNTHETIC_VALIDATION" not in retrosyn_agent_env

    assert "--validation-runner" not in (
        ROOT / "infra/helm/moleculeforge/values.yaml"
    ).read_text(encoding="utf-8")
    assert "--validation-runner" not in (
        ROOT / "infra/kubernetes/deployments/moleculeforge-services.yaml"
    ).read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_full_workflow_clients_block_routes_when_supply_is_unavailable() -> None:
    module = _load_module(
        "orchestrator_full_srb_supply_compile_test",
        ROOT / "services/orchestrator-svc/src/orchestrator_svc/main.py",
    )
    route = {"route_id": "route-1", "steps": []}
    request_client = _AgentRequestClientStub(
        lambda _subject, payload: {
            "project_id": payload["project_id"],
            "candidate_id": payload["candidate_id"],
            "candidate_index": payload["candidate_index"],
            "canonical_smiles": payload["canonical_smiles"],
            "status": "compiled",
            "protocols": [{"ssp_id": "ssp-1"}],
        }
    )
    state = _full_selected_state(routes=[route])
    state["supply"] = {
        "route_id": "route-1",
        "supply_assessment": {"overall_feasibility": "unavailable"},
    }

    result = await module.FullWorkflowClients(request_client).compile_synthesis(state)

    assert result["status"] == "not_compiled"
    assert result["route_id"] == "route-1"
    assert result["protocols"] == []
    assert result["blocking_evidence"] == [
        {
            "rule_id": "workflow_supply_feasibility",
            "reason": "selected route supply feasibility is unavailable",
        }
    ]
    assert request_client.calls == []


@pytest.mark.asyncio
async def test_full_workflow_clients_report_blocking_evidence_without_routes() -> None:
    module = _load_module(
        "orchestrator_full_srb_no_routes_test",
        ROOT / "services/orchestrator-svc/src/orchestrator_svc/main.py",
    )
    state = _full_selected_state()

    result = await module.FullWorkflowClients().compile_synthesis(state)

    assert result == {
        "status": "not_compiled",
        "protocols": [],
        "blocking_evidence": [
            {
                "rule_id": "workflow_retrosyn_routes",
                "reason": "retrosyn.routes is empty",
            }
        ],
        "project_id": "project-1",
        "candidate_id": "candidate-1",
        "candidate_index": 0,
        "canonical_smiles": "CCO",
    }


@pytest.mark.asyncio
async def test_full_workflow_records_provenance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_module(
        "orchestrator_full_workflow_provenance_test",
        ROOT / "services/orchestrator-svc/src/orchestrator_svc/main.py",
    )
    await _configure_project_run_store(module, tmp_path, "project-provenance-1")
    records: list[dict] = []

    class Response:
        def __init__(self, payload: dict) -> None:
            self.payload = payload

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "artifact_id": self.payload["artifact_id"],
                "signature": "sig-test",
                "recorded_at": "2026-05-30T00:00:00+00:00",
            }

    class ProvenanceClient:
        def __init__(self, *, timeout: float) -> None:
            assert timeout == 30.0

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback) -> None:
            return None

        async def post(self, url: str, *, json: dict) -> Response:
            assert url == "http://provenance-svc:8010/v1/provenance/record"
            records.append(json)
            return Response(json)

    monkeypatch.setenv("PROVENANCE_SVC_URL", "http://provenance-svc:8010")
    monkeypatch.setattr(module.httpx, "AsyncClient", ProvenanceClient)

    class Clients:
        async def compile_intent(self, state):
            return {"cig": {"source": state["nl_input"]}, "hciv": {}, "intent_cone": {}}

        async def generate_candidates(self, state):
            return [_full_candidate()]

        async def validate_candidates(self, state):
            record = _full_validation_record()
            record["ki_nm"] = 5.0
            return {
                "outcome": "PASS",
                "passed": True,
                "records": [record],
                "results": [record],
            }

        async def plan_routes(self, state):
            return {"skipped": False, "routes": [{"route_id": "route-1"}]}

        async def assess_supply(self, state):
            return {
                "route_id": "route-1",
                "supply_assessment": {"overall_feasibility": "available"},
            }

        async def compile_synthesis(self, state):
            return {
                "status": "compiled",
                "route_id": "route-1",
                "protocols": [
                    {
                        "ssp_id": "ssp-1",
                        "route_id": "route-1",
                        "status": "compiled",
                    }
                ],
            }

        async def execute_synthesis(self, state):
            return {
                "status": "executed",
                "route_id": "route-1",
                "protocols": [
                    {
                        "ssp_id": "ssp-1",
                        "route_id": "route-1",
                        "status": "executed",
                    }
                ],
            }

        async def review_candidates(self, state):
            return {"verdict": "pass", "total_rules": 1}

    started = await module.start_design(
        {
            "nl_input": "Design KRAS G12C inhibitor",
            "workflow_scope": "full",
            "project_id": "project-provenance-1",
            "validation_passed": True,
            "max_refinements": 1,
            **_full_policy_payload(),
            "clients": Clients(),
            "run_id": "run-provenance-1",
            "trace_id": "trace-provenance-1",
            "artifact_ids": ["artifact-input"],
        }
    )

    assert records
    assert records[0]["artifact_type"] == "workflow_state"
    assert records[0]["parent_ids"] == ["artifact-input"]
    assert records[0]["metadata"]["crg"]["project_id"] == "project-provenance-1"
    assert records[0]["metadata"]["supply_feasibility"] == "available"
    assert records[0]["metadata"]["srb_protocol_count"] == 1
    assert len(records[0]["metadata"]["crg"]["beliefs"]) == 6
    assert len(records[0]["metadata"]["crg"]["edges"]) == 5
    assert records[0]["metadata"]["crg_belief_count"] == 6
    assert records[0]["metadata"]["crg_edge_count"] == 5
    signed_state = json.loads(
        base64.b64decode(records[0]["payload_base64"], validate=True).decode("utf-8")
    )
    assert signed_state["run_id"] == "run-provenance-1"
    assert signed_state["artifact_ids"] == ["artifact-input"]
    assert "provenance" not in signed_state
    assert started["state"]["provenance"]["recorded"] is True
    assert "artifact-run-provenance-1-workflow-state" in started["artifact_ids"]


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
            payload_base64=base64.b64encode(b"parent payload").decode("ascii"),
            metadata={"project_id": "project-1", "trace_id": "trace-1"},
        )
    )
    child = await module.create_record(
        module.ProvenanceRecord(
            artifact_type="cig",
            artifact_id="artifact-child",
            parent_ids=["artifact-parent"],
            payload_base64=base64.b64encode(b"child payload").decode("ascii"),
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
        "SIGSTORE_SIGN_COMMAND",
        "SIGSTORE_VERIFY_COMMAND",
        "SIGSTORE_IDENTITY_TOKEN",
        "SIGSTORE_EXPECTED_IDENTITY",
        "SIGSTORE_REKOR_URL",
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
    assert "SIGSTORE_SIGN_COMMAND" in exc.value.detail["missing_config"]
    assert "SIGSTORE_VERIFY_COMMAND" in exc.value.detail["missing_config"]
    assert "SIGSTORE_IDENTITY_TOKEN" in exc.value.detail["missing_config"]
    assert "SIGSTORE_EXPECTED_IDENTITY" in exc.value.detail["missing_config"]
    assert "SIGSTORE_REKOR_URL" in exc.value.detail["missing_config"]


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
            self.beliefs: list[dict] = []
            self.crg_edges: list[dict] = []

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

        async def write_workflow_belief(self, **kwargs) -> None:
            self.beliefs.append(kwargs)

        async def write_crg_edge(self, **kwargs) -> None:
            self.crg_edges.append(kwargs)

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

        async def put_object_if_absent(
            self,
            object_name: str,
            data: bytes,
            content_type: str,
        ) -> bool:
            await self.put_object(object_name, data, content_type)
            return True

    graph = RecordingGraph()
    audit_writer = RecordingAuditWriter()
    object_store = RecordingObjectStore()
    store = module.ProductionProvenanceStore(graph, audit_writer, object_store)
    recorded_at = "2026-05-19T00:00:00Z"

    raw_payload = b"candidate payload"
    record = module.ProvenanceRecord(
        artifact_type="candidate",
        artifact_id="artifact-1",
        parent_ids=["artifact-parent"],
        payload_base64=base64.b64encode(raw_payload).decode("ascii"),
        metadata={
            "project_id": "project-1",
            "run_id": "run-1",
            "trace_id": "trace-1",
            "crg": {
                "project_id": "project-1",
                "beliefs": [
                    {
                        "id": "belief-1",
                        "subject": "run-1",
                        "predicate": "workflow_stage",
                        "object": "PLANNING",
                        "confidence": 1.0,
                        "source_agent": "orchestrator",
                        "timestamp_ns": 123,
                        "evidence_ids": ["artifact-parent"],
                    },
                    {
                        "id": "belief-2",
                        "subject": "run-1",
                        "predicate": "workflow_stage",
                        "object": "GENERATING",
                        "confidence": 1.0,
                        "source_agent": "orchestrator",
                        "timestamp_ns": 124,
                        "evidence_ids": ["artifact-parent"],
                    },
                ],
                "edges": [
                    {
                        "source_belief_id": "belief-1",
                        "target_belief_id": "belief-2",
                        "relation": "derives_from",
                        "weight": 1.0,
                    }
                ],
            },
        },
    )
    signed = module.sigstore.sign_artifact(
        record.artifact_id,
        record.artifact_type,
        record.metadata,
        checksum=f"sha256:{hashlib.sha256(raw_payload).hexdigest()}",
        parent_ids=record.parent_ids,
        recorded_at=recorded_at,
    )
    signed["signature_type"] = "sigstore_rekor"

    class ProductionVerifier:
        def verify_record(self, *args, **kwargs) -> bool:
            return True

    module.sigstore = ProductionVerifier()
    stored = await store.record(
        record,
        signed,
        recorded_at,
    )

    assert stored["metadata"]["run_id"] == "run-1"
    assert graph.artifacts[0]["run_id"] == "run-1"
    assert graph.artifacts[0]["trace_id"] == "trace-1"
    assert graph.parents == [("artifact-parent", "artifact-1")]
    assert [belief["belief_id"] for belief in graph.beliefs] == [
        "belief-1",
        "belief-2",
    ]
    assert graph.beliefs[0]["project_id"] == "project-1"
    assert graph.beliefs[0]["run_id"] == "run-1"
    assert graph.crg_edges == [
        {
            "source_belief_id": "belief-1",
            "target_belief_id": "belief-2",
            "relation": "derives_from",
            "weight": 1.0,
        }
    ]
    assert audit_writer.events[0]["metadata"]["run_id"] == "run-1"
    assert object_store.objects[0]["object_name"] == "provenance/artifact-1.json"
    persisted_record = json.loads(object_store.objects[0]["data"])
    assert persisted_record["metadata"]["run_id"] == "run-1"
    assert persisted_record["payload_base64"] == record.payload_base64
    assert persisted_record["checksum"] == f"sha256:{hashlib.sha256(raw_payload).hexdigest()}"
    assert persisted_record["signature_bundle"]["payload_hash"]
    assert module.sigstore.verify_record(persisted_record) is True


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
            stored = module._stored_record(record, signed, recorded_at)
            self.records.append(stored)
            return stored

        async def get_record(self, artifact_id: str) -> dict:
            return next(record for record in self.records if record["artifact_id"] == artifact_id)

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
            payload_base64=base64.b64encode(b"candidate payload").decode("ascii"),
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
    monkeypatch.setenv("KRAS_E2E_SCOPE", "full")

    status = module.kras_e2e_preflight_status()

    assert status["ready"] is False
    assert "HFM_CHECKPOINT_PATH" in status["missing"]


def test_kras_e2e_preflight_rejects_missing_artifact_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module(
        "kras_e2e_preflight_missing_paths_test",
        ROOT / "tests/e2e/test_kras_g12c_pilot.py",
    )
    for env_var in module.KRAS_E2E_REQUIRED_ENV:
        monkeypatch.setenv(env_var, "1" if env_var.endswith("_READY") else "/missing/resource")
    monkeypatch.setenv("KRAS_E2E_SCOPE", "full")

    status = module.kras_e2e_preflight_status()

    assert status["ready"] is False
    assert any("hfm_checkpoint" in item for item in status["missing"])
    assert any("hfm_decoder" in item for item in status["missing"])


def test_kras_e2e_preflight_uses_actual_service_inputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_module(
        "kras_e2e_preflight_service_inputs_test",
        ROOT / "tests/e2e/test_kras_g12c_pilot.py",
    )
    hfm_checkpoint = tmp_path / "hfm.pt"
    hfm_decoder = tmp_path / "decoder.json"
    hfm_checkpoint.write_bytes(b"checkpoint")
    hfm_decoder.write_text("{}", encoding="utf-8")
    boltz_model = tmp_path / "boltz-2"
    boltz_model.mkdir()
    aizynth_config = tmp_path / "aizynth.yml"
    aizynth_config.write_text("expansion: {}\nstock: {}\n", encoding="utf-8")
    monkeypatch.setenv("HFM_CHECKPOINT_PATH", str(hfm_checkpoint))
    monkeypatch.setenv("HFM_DECODER_PATH", str(hfm_decoder))
    monkeypatch.setenv("BOLTZ_MODEL_PATH", str(boltz_model))
    monkeypatch.setenv("AIZYNTH_CONFIG_PATH", str(aizynth_config))
    monkeypatch.delenv("RETROSYN_RUNNER_URI", raising=False)
    monkeypatch.delenv("BOLTZ_INPUT_TEMPLATE_DIR", raising=False)
    monkeypatch.setenv("CRITIC_AGENT_READY", "1")
    monkeypatch.setenv("ORCHESTRATOR_E2E_READY", "1")
    for env_var in module.KRAS_E2E_DKI_REQUIRED_ENV:
        monkeypatch.setenv(env_var, "configured")
    monkeypatch.setenv("PROVENANCE_STORE_MODE", "production_real")
    monkeypatch.setenv("TEST_DATABASE_URL", "sqlite:///tmp/test.db")
    monkeypatch.setenv("KRAS_E2E_SCOPE", "full")

    status = module.kras_e2e_preflight_status()

    assert status["ready"] is False
    assert "RETROSYN_RUNNER_URI" not in status["missing"]
    assert "BOLTZ_INPUT_TEMPLATE_DIR" in status["missing"]


def test_kras_e2e_reduced_scope_skips_model_and_supply_requirements(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module(
        "kras_e2e_reduced_preflight_test",
        ROOT / "tests/e2e/test_kras_g12c_pilot.py",
    )
    for env_var in module.KRAS_E2E_REQUIRED_ENV:
        monkeypatch.delenv(env_var, raising=False)
    for env_var in module.KRAS_E2E_DKI_REQUIRED_ENV:
        monkeypatch.setenv(env_var, "configured")
    monkeypatch.setenv("PROVENANCE_STORE_MODE", "production_real")
    monkeypatch.setenv("TEST_DATABASE_URL", "sqlite:///tmp/test.db")
    monkeypatch.setenv("SIGSTORE_IDENTITY_TOKEN", "oidc-token")
    monkeypatch.setenv("SIGSTORE_EXPECTED_IDENTITY", "fulcio@example.com")
    monkeypatch.setenv("SIGSTORE_SIGN_COMMAND", "cosign sign-blob")
    monkeypatch.setenv("SIGSTORE_VERIFY_COMMAND", "cosign verify-blob")
    monkeypatch.setenv("SIGSTORE_REKOR_URL", "https://rekor.example")
    monkeypatch.setenv("KRAS_E2E_SCOPE", "engineering")
    monkeypatch.setenv("ORCHESTRATOR_E2E_READY", "1")

    status = module.kras_e2e_preflight_status()

    assert status["ready"] is True
    assert status["missing"] == []


def test_kras_e2e_preflight_requires_sigstore_identity_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module(
        "kras_e2e_preflight_sigstore_identity_test",
        ROOT / "tests/e2e/test_kras_g12c_pilot.py",
    )
    for env_var in module.KRAS_E2E_DKI_REQUIRED_ENV:
        monkeypatch.setenv(env_var, "configured")
    monkeypatch.setenv("PROVENANCE_STORE_MODE", "production_real")
    monkeypatch.setenv("TEST_DATABASE_URL", "sqlite:///tmp/test.db")
    monkeypatch.setenv("KRAS_E2E_SCOPE", "full")
    monkeypatch.setenv("ORCHESTRATOR_E2E_READY", "1")
    monkeypatch.delenv("SIGSTORE_IDENTITY_TOKEN", raising=False)
    monkeypatch.delenv("SIGSTORE_EXPECTED_IDENTITY", raising=False)

    status = module.kras_e2e_preflight_status()

    assert status["ready"] is False
    assert "SIGSTORE_IDENTITY_TOKEN" in status["missing"]
    assert "SIGSTORE_EXPECTED_IDENTITY" in status["missing"]


def test_kras_e2e_preflight_requires_sigstore_command_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module(
        "kras_e2e_preflight_sigstore_command_test",
        ROOT / "tests/e2e/test_kras_g12c_pilot.py",
    )
    for env_var in module.KRAS_E2E_DKI_REQUIRED_ENV:
        monkeypatch.setenv(env_var, "configured")
    monkeypatch.setenv("PROVENANCE_STORE_MODE", "production_real")
    monkeypatch.setenv("TEST_DATABASE_URL", "sqlite:///tmp/test.db")
    monkeypatch.setenv("KRAS_E2E_SCOPE", "full")
    monkeypatch.setenv("ORCHESTRATOR_E2E_READY", "1")
    monkeypatch.setenv("SIGSTORE_IDENTITY_TOKEN", "oidc-token")
    monkeypatch.setenv("SIGSTORE_EXPECTED_IDENTITY", "fulcio@example.com")
    monkeypatch.delenv("SIGSTORE_SIGN_COMMAND", raising=False)
    monkeypatch.delenv("SIGSTORE_VERIFY_COMMAND", raising=False)

    status = module.kras_e2e_preflight_status()

    assert status["ready"] is False
    assert "SIGSTORE_SIGN_COMMAND" in status["missing"]
    assert "SIGSTORE_VERIFY_COMMAND" in status["missing"]


def test_kras_e2e_preflight_requires_sigstore_rekor_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module(
        "kras_e2e_preflight_sigstore_rekor_url_test",
        ROOT / "tests/e2e/test_kras_g12c_pilot.py",
    )
    for env_var in module.KRAS_E2E_DKI_REQUIRED_ENV:
        monkeypatch.setenv(env_var, "configured")
    monkeypatch.setenv("PROVENANCE_STORE_MODE", "production_real")
    monkeypatch.setenv("TEST_DATABASE_URL", "sqlite:///tmp/test.db")
    monkeypatch.setenv("KRAS_E2E_SCOPE", "full")
    monkeypatch.setenv("ORCHESTRATOR_E2E_READY", "1")
    monkeypatch.setenv("SIGSTORE_IDENTITY_TOKEN", "oidc-token")
    monkeypatch.setenv("SIGSTORE_EXPECTED_IDENTITY", "fulcio@example.com")
    monkeypatch.setenv("SIGSTORE_SIGN_COMMAND", "cosign sign-blob")
    monkeypatch.setenv("SIGSTORE_VERIFY_COMMAND", "cosign verify-blob")
    monkeypatch.delenv("SIGSTORE_REKOR_URL", raising=False)

    status = module.kras_e2e_preflight_status()

    assert status["ready"] is False
    assert "SIGSTORE_REKOR_URL" in status["missing"]


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


def test_audit_e2e_preflight_requires_sigstore_identity_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module(
        "audit_e2e_preflight_sigstore_identity_test",
        ROOT / "tests/e2e/test_audit_completeness.py",
    )
    monkeypatch.setenv("PROVENANCE_SVC_URL", "http://127.0.0.1:8010")
    monkeypatch.setenv("SIGSTORE_E2E_READY", "1")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:4318")
    monkeypatch.setenv("PROVENANCE_STORE_MODE", "production_real")
    monkeypatch.setenv("TEST_DATABASE_URL", "sqlite:///tmp/test.db")
    for env_var in module.AUDIT_E2E_DKI_REQUIRED_ENV:
        monkeypatch.setenv(env_var, "configured")
    monkeypatch.delenv("SIGSTORE_IDENTITY_TOKEN", raising=False)
    monkeypatch.delenv("SIGSTORE_EXPECTED_IDENTITY", raising=False)

    status = module.audit_e2e_preflight_status()

    assert status["ready"] is False
    assert "SIGSTORE_IDENTITY_TOKEN" in status["missing"]
    assert "SIGSTORE_EXPECTED_IDENTITY" in status["missing"]


def test_audit_e2e_preflight_requires_sigstore_command_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module(
        "audit_e2e_preflight_sigstore_command_test",
        ROOT / "tests/e2e/test_audit_completeness.py",
    )
    monkeypatch.setenv("PROVENANCE_SVC_URL", "http://127.0.0.1:8010")
    monkeypatch.setenv("SIGSTORE_E2E_READY", "1")
    monkeypatch.setenv("SIGSTORE_IDENTITY_TOKEN", "oidc-token")
    monkeypatch.setenv("SIGSTORE_EXPECTED_IDENTITY", "fulcio@example.com")
    monkeypatch.setenv("SIGSTORE_REKOR_URL", "https://rekor.example")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:4318")
    monkeypatch.setenv("PROVENANCE_STORE_MODE", "production_real")
    monkeypatch.setenv("TEST_DATABASE_URL", "sqlite:///tmp/test.db")
    for env_var in module.AUDIT_E2E_DKI_REQUIRED_ENV:
        monkeypatch.setenv(env_var, "configured")
    monkeypatch.delenv("SIGSTORE_SIGN_COMMAND", raising=False)
    monkeypatch.delenv("SIGSTORE_VERIFY_COMMAND", raising=False)

    status = module.audit_e2e_preflight_status()

    assert status["ready"] is False
    assert "SIGSTORE_SIGN_COMMAND" in status["missing"]
    assert "SIGSTORE_VERIFY_COMMAND" in status["missing"]


def test_audit_e2e_preflight_requires_sigstore_rekor_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module(
        "audit_e2e_preflight_sigstore_rekor_url_test",
        ROOT / "tests/e2e/test_audit_completeness.py",
    )
    monkeypatch.setenv("PROVENANCE_SVC_URL", "http://127.0.0.1:8010")
    monkeypatch.setenv("SIGSTORE_E2E_READY", "1")
    monkeypatch.setenv("SIGSTORE_IDENTITY_TOKEN", "oidc-token")
    monkeypatch.setenv("SIGSTORE_EXPECTED_IDENTITY", "fulcio@example.com")
    monkeypatch.setenv("SIGSTORE_SIGN_COMMAND", "cosign sign-blob")
    monkeypatch.setenv("SIGSTORE_VERIFY_COMMAND", "cosign verify-blob")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:4318")
    monkeypatch.setenv("PROVENANCE_STORE_MODE", "production_real")
    monkeypatch.setenv("TEST_DATABASE_URL", "sqlite:///tmp/test.db")
    for env_var in module.AUDIT_E2E_DKI_REQUIRED_ENV:
        monkeypatch.setenv(env_var, "configured")
    monkeypatch.delenv("SIGSTORE_REKOR_URL", raising=False)

    status = module.audit_e2e_preflight_status()

    assert status["ready"] is False
    assert "SIGSTORE_REKOR_URL" in status["missing"]


def _valid_retrosyn_step(reaction: str, step_id: str = "step-1") -> dict:
    return {
        "step_id": step_id,
        "reaction": reaction,
        "reaction_type": "generic",
        "reactants": [{"smiles": "C", "amount_mmol": 1.0}],
        "conditions": {
            "temperature_C": 25.0,
            "time_h": 2.0,
            "source": "test",
        },
        "yield": 0.8,
        "building_blocks": [{"smiles": "C"}],
    }


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
                            "reaction_type": "oxidation",
                            "reactants": [
                                {"smiles": "CCO", "amount_mmol": 1.0},
                                {"smiles": "O=O", "amount_mmol": 1.2},
                            ],
                            "conditions": {"temperature_C": 25, "time_h": 2},
                            "yield": 0.62,
                            "reagents": ["catalyst"],
                            "purification": "filtration",
                            "building_blocks": [
                                {"smiles": "CCO", "source": "local_catalog"},
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
        request_id="request-1",
        run_id="run-1",
        candidate_id="candidate-1",
        candidate_index=2,
        canonical_smiles="CCOO",
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
    assert response.request_id == "request-1"
    assert response.project_id == "proj-1"
    assert response.run_id == "run-1"
    assert response.candidate_id == "candidate-1"
    assert response.candidate_index == 2
    assert response.canonical_smiles == "CCOO"
    step = response.routes[0].steps[0]
    assert step.step_id == "retro-1"
    assert step.reaction == "CCO.O=O>>CCOO"
    assert step.reaction_type == "oxidation"
    assert step.reactants[0]["smiles"] == "CCO"
    assert step.reactants[0]["amount_mmol"] == 1.0
    assert step.conditions["temperature_C"] == 25
    assert step.conditions["time_h"] == 2
    assert step.reagents[0] == "catalyst"
    assert step.purification == "filtration"
    assert step.HasField("yield_fraction")
    assert step.yield_fraction == pytest.approx(0.62)


def test_retrosyn_grpc_client_rejects_step_without_yield_fraction() -> None:
    module = _load_module(
        "retrosyn_agent_step_yield_contract_test",
        ROOT / "agents/retrosyn_agent/src/retrosyn_agent/agent.py",
    )
    from mf_core.proto_gen.moleculeforge.v1.retrosyn import retrosyn_pb2

    step = retrosyn_pb2.SyntheticRouteStep(
        step_id="retro-1",
        reaction="CCO.O=O>>CCOO",
        reaction_type="oxidation",
        reactants=[{"smiles": "CCO", "amount_mmol": 1.0}],
        conditions={"temperature_C": 25.0, "time_h": 2.0},
        building_blocks=[{"smiles": "CCO"}],
    )

    with pytest.raises(module.RetrosynRouteValueError, match="yield_fraction"):
        module._route_step_from_proto(step)


async def _retrosyn_grpc_call(module, servicer, request):
    server = grpc.aio.server()
    module.retrosyn_pb2_grpc.add_RetrosynServiceServicer_to_server(servicer, server)
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    channel = grpc.aio.insecure_channel(f"127.0.0.1:{port}")
    try:
        stub = module.retrosyn_pb2_grpc.RetrosynServiceStub(channel)
        return await stub.FindRoutes(request)
    finally:
        await channel.close()
        await server.stop(None)


@pytest.mark.asyncio
async def test_retrosyn_service_maps_invalid_timeout_and_malformed_routes() -> None:
    module = _load_module(
        "retrosyn_service_error_mapping_test",
        ROOT / "services/retrosyn-svc/src/retrosyn_svc/main.py",
    )
    from mf_core.proto_gen.moleculeforge.v1.retrosyn import retrosyn_pb2

    with pytest.raises(grpc.aio.AioRpcError) as invalid:
        await _retrosyn_grpc_call(
            module,
            module.RetrosynServicer(planner=object()),
            retrosyn_pb2.RetrosynthesisRequest(max_routes=-1),
        )
    assert invalid.value.code() == grpc.StatusCode.INVALID_ARGUMENT

    class TimedOutPlanner:
        async def find_routes(self, smiles: str, max_routes: int) -> list[dict]:
            raise TimeoutError("planner deadline exceeded")

    with pytest.raises(grpc.aio.AioRpcError) as timed_out:
        await _retrosyn_grpc_call(
            module,
            module.RetrosynServicer(planner=TimedOutPlanner()),
            retrosyn_pb2.RetrosynthesisRequest(
                molecule_smiles="CCO",
                max_routes=1,
                engine="aizynth",
            ),
        )
    assert timed_out.value.code() == grpc.StatusCode.DEADLINE_EXCEEDED

    class MalformedPlanner:
        async def find_routes(self, smiles: str, max_routes: int) -> list[dict]:
            step = _valid_retrosyn_step("CO.C>>CCO")
            step["reactants"][0].pop("amount_mmol")
            return [
                {
                    "route_id": "route-invalid",
                    "steps": [step],
                }
            ]

    with pytest.raises(grpc.aio.AioRpcError) as malformed:
        await _retrosyn_grpc_call(
            module,
            module.RetrosynServicer(planner=MalformedPlanner()),
            retrosyn_pb2.RetrosynthesisRequest(
                molecule_smiles="CCO",
                max_routes=1,
                engine="aizynth",
            ),
        )
    assert malformed.value.code() == grpc.StatusCode.DATA_LOSS
    assert "amount_mmol" in malformed.value.details()


@pytest.mark.asyncio
async def test_retrosyn_service_merges_injected_planner_ensemble() -> None:
    module = _load_module(
        "retrosyn_service_ensemble_test",
        ROOT / "services/retrosyn-svc/src/retrosyn_svc/main.py",
    )

    class Planner:
        def __init__(self, route_id: str, score: float) -> None:
            self.route_id = route_id
            self.score = score
            self.calls: list[tuple[str, int]] = []

        async def find_routes(self, smiles: str, max_routes: int = 10) -> list[dict]:
            self.calls.append((smiles, max_routes))
            return [
                {
                    "route_id": self.route_id,
                    "score": self.score,
                    "steps": [
                        _valid_retrosyn_step(
                            f"{smiles}>>{self.route_id}",
                            f"{self.route_id}-step-1",
                        )
                    ],
                }
            ]

    aizynth = Planner("route-aizynth", 0.4)
    rsgpt = Planner("route-rsgpt", 0.9)
    service = module.RetrosynServicer(
        route_planners={"aizynth": aizynth, "rsgpt": rsgpt},
    )
    request = SimpleNamespace(
        molecule_smiles="CCO",
        max_routes=2,
        engine="ensemble",
    )

    response = await service.FindRoutes(request, None)

    assert aizynth.calls == [("CCO", 2)]
    assert rsgpt.calls == [("CCO", 2)]
    assert response.total_routes_found == 2
    assert [route.route_id for route in response.routes] == [
        "route-rsgpt",
        "route-aizynth",
    ]
    assert response.routes[0].predicted_score == pytest.approx(0.9)


@pytest.mark.asyncio
async def test_retrosyn_service_keeps_route_planners_ahead_of_accessibility_scores() -> None:
    module = _load_module(
        "retrosyn_service_accessibility_score_rank_test",
        ROOT / "services/retrosyn-svc/src/retrosyn_svc/main.py",
    )

    class Planner:
        def __init__(self, route: dict) -> None:
            self.route = route

        async def find_routes(self, smiles: str, max_routes: int = 10) -> list[dict]:
            return [dict(self.route)]

    service = module.RetrosynServicer(
        route_planners={
            "rascore": Planner(
                {
                    "route_id": "rascore-1",
                    "route_type": "retrosynthetic_accessibility_score",
                    "score": 0.99,
                    "steps": [],
                }
            ),
            "rsgpt": Planner(
                {
                    "route_id": "route-rsgpt",
                    "score": 0.5,
                    "steps": [
                        _valid_retrosyn_step(
                            "CCO>>route-rsgpt",
                            "route-rsgpt-step-1",
                        )
                    ],
                }
            ),
        },
    )
    request = SimpleNamespace(molecule_smiles="CCO", max_routes=1, engine="ensemble")

    response = await service.FindRoutes(request, None)

    assert response.total_routes_found == 1
    assert [route.route_id for route in response.routes] == ["route-rsgpt"]
    assert [item.assessment_id for item in response.assessments] == ["rascore-1"]


@pytest.mark.asyncio
async def test_retrosyn_service_builds_planner_ensemble_from_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_module(
        "retrosyn_service_env_ensemble_test",
        ROOT / "services/retrosyn-svc/src/retrosyn_svc/main.py",
    )
    runner = tmp_path / "retrosyn_runner.py"
    runner.write_text(
        "\n".join(
            [
                "import json",
                "import sys",
                "payload = json.load(sys.stdin)",
                "route_id = sys.argv[1]",
                "score = float(sys.argv[2])",
                "assert payload['smiles'] == 'CCO'",
                "assert payload['max_routes'] == 2",
                "json.dump(",
                "    {'routes': [{'route_id': route_id, 'score': score, 'steps': [{"
                "'step_id': route_id + '-step-1', 'reaction': 'CCO>>' + route_id,"
                " 'reaction_type': 'generic',"
                " 'reactants': [{'smiles': 'C', 'amount_mmol': 1.0}],"
                " 'conditions': {'temperature_C': 25.0, 'time_h': 2.0,"
                " 'source': 'test'},"
                " 'yield': 0.8,"
                " 'building_blocks': [{'smiles': 'C'}]}]}]},",
                "    sys.stdout,",
                ")",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv(
        "RETROSYN_PLANNER_COMMANDS_JSON",
        json.dumps(
            {
                "aizynth": f"{sys.executable} {runner} route-aizynth 0.4",
                "rsgpt": f"{sys.executable} {runner} route-rsgpt 0.9",
            }
        ),
    )
    service = module.RetrosynServicer()
    request = SimpleNamespace(molecule_smiles="CCO", max_routes=2, engine="ensemble")

    response = await service.FindRoutes(request, None)

    assert response.total_routes_found == 2
    assert [route.route_id for route in response.routes] == [
        "route-rsgpt",
        "route-aizynth",
    ]
    assert list(response.routes[0].reaction_smiles) == ["CCO>>route-rsgpt"]


@pytest.mark.asyncio
async def test_retrosyn_service_builds_named_planner_ensemble_from_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_module(
        "retrosyn_service_named_env_ensemble_test",
        ROOT / "services/retrosyn-svc/src/retrosyn_svc/main.py",
    )
    runner = tmp_path / "retrosyn_runner.py"
    runner.write_text(
        "\n".join(
            [
                "import json",
                "import sys",
                "payload = json.load(sys.stdin)",
                "route_id = sys.argv[1]",
                "score = float(sys.argv[2])",
                "assert payload['smiles'] == 'CCO'",
                "assert payload['max_routes'] == 3",
                "assert payload['engine'] in {'rascore', 'rsgpt', 'ualign', 'aizynth'}",
                "json.dump(",
                "    {'routes': [{'route_id': route_id, 'score': score, 'steps': [{"
                "'step_id': route_id + '-step-1', 'reaction': 'CCO>>' + route_id,"
                " 'reaction_type': 'generic',"
                " 'reactants': [{'smiles': 'C', 'amount_mmol': 1.0}],"
                " 'conditions': {'temperature_C': 25.0, 'time_h': 2.0,"
                " 'source': 'test'},"
                " 'yield': 0.8,"
                " 'building_blocks': [{'smiles': 'C'}]}]}]},",
                "    sys.stdout,",
                ")",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv("RETROSYN_PLANNER_COMMANDS_JSON", raising=False)
    monkeypatch.setenv("RASCORE_PLANNER_COMMAND", f"{sys.executable} {runner} route-rascore 0.7")
    monkeypatch.setenv("RSGPT_PLANNER_COMMAND", f"{sys.executable} {runner} route-rsgpt 0.9")
    monkeypatch.setenv("UALIGN_PLANNER_COMMAND", f"{sys.executable} {runner} route-ualign 0.8")
    monkeypatch.setenv("AIZYNTH_PLANNER_COMMAND", f"{sys.executable} {runner} route-aizynth 0.4")
    service = module.RetrosynServicer()
    request = SimpleNamespace(molecule_smiles="CCO", max_routes=3, engine="ensemble")

    response = await service.FindRoutes(request, None)

    assert response.total_routes_found == 3
    assert [route.route_id for route in response.routes] == [
        "route-rsgpt",
        "route-ualign",
        "route-rascore",
    ]
    assert list(response.routes[0].reaction_smiles) == ["CCO>>route-rsgpt"]


@pytest.mark.asyncio
async def test_retrosyn_service_runs_configured_json_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module(
        "retrosyn_service_json_command_test",
        ROOT / "services/retrosyn-svc/src/retrosyn_svc/main.py",
    )
    command = (
        f"{sys.executable} -c "
        '"import json,sys; '
        "payload=json.load(sys.stdin); "
        "assert payload['smiles'] == 'CCO'; "
        "assert payload['max_routes'] == 1; "
        "print(json.dumps({'routes':[{'route_id':'route-command',"
        "'score':0.8,'predicted_yield':0.6,"
        "'steps':[{'step_id':'route-command-step-1','reaction':'CCO>>CC=O',"
        "'reaction_type':'generic',"
        "'reactants':[{'smiles':'C','amount_mmol':1.0}],"
        "'conditions':{'temperature_C':25.0,'time_h':2.0,'source':'test'},"
        "'yield':0.6,"
        "'building_blocks':[{'smiles':'C'}]}]}]}))\""
    )
    monkeypatch.setenv("RETROSYN_PLANNER_COMMAND", command)
    service = module.RetrosynServicer()
    request = SimpleNamespace(
        molecule_smiles="CCO",
        max_routes=1,
        engine="rsgpt",
    )

    response = await service.FindRoutes(request, None)

    assert response.total_routes_found == 1
    assert response.routes[0].route_id == "route-command"
    assert response.routes[0].predicted_score == pytest.approx(0.8)
    assert response.routes[0].predicted_yield == pytest.approx(0.6)
    assert list(response.routes[0].reaction_smiles) == ["CCO>>CC=O"]


def test_retrosyn_service_runtime_tracks_aizynth_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_module(
        "retrosyn_service_runtime_status_test",
        ROOT / "services/retrosyn-svc/src/retrosyn_svc/main.py",
    )
    config = tmp_path / "config.yml"
    config.write_text("expansion: {}\nstock: {}\n", encoding="utf-8")
    monkeypatch.setenv("AIZYNTH_CONFIG_PATH", str(config))
    monkeypatch.delenv("RETROSYN_RUNNER_URI", raising=False)
    monkeypatch.delenv("RETROSYN_PLANNER_COMMAND", raising=False)

    status = module.runtime_status()

    assert [item["source"] for item in status] == [
        "AIZYNTH_CONFIG_PATH",
        "RETROSYN_SCORER_URI",
        "RETROSYN_PLANNER_COMMAND",
    ]
    assert status[0]["available"] is True
    assert status[2]["required"] is False


def test_retrosyn_runtime_rejects_missing_planner_command(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_module(
        "retrosyn_service_missing_planner_command_runtime_test",
        ROOT / "services/retrosyn-svc/src/retrosyn_svc/main.py",
    )
    config = tmp_path / "config.yml"
    config.write_text("expansion: {}\nstock: {}\n", encoding="utf-8")
    monkeypatch.setenv("AIZYNTH_CONFIG_PATH", str(config))
    monkeypatch.setenv("RETROSYN_PLANNER_COMMAND", "missing-retrosyn-planner --json")

    status = module.runtime_status()

    planner_status = next(item for item in status if item["name"] == "retrosyn_planner_command")
    assert planner_status["configured"] is True
    assert planner_status["available"] is False
    assert planner_status["source"] == "RETROSYN_PLANNER_COMMAND"
    assert "not found" in planner_status["message"]


def test_retrosyn_service_startup_accepts_external_planner_without_aizynth_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module(
        "retrosyn_service_startup_external_planner_test",
        ROOT / "services/retrosyn-svc/src/retrosyn_svc/main.py",
    )
    monkeypatch.delenv("AIZYNTH_CONFIG_PATH", raising=False)
    monkeypatch.setenv(
        "RETROSYN_PLANNER_COMMAND",
        f"{sys.executable} -c \"import json,sys;json.dump({{'routes': []}}, sys.stdout)\"",
    )

    statuses = module._require_planner_runtime()
    planner_status = next(
        status for status in statuses if status.name == "retrosyn_planner_command"
    )

    assert planner_status.available is True
    assert planner_status.source == "RETROSYN_PLANNER_COMMAND"


def test_retrosyn_service_startup_accepts_json_planner_ensemble_without_aizynth_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module(
        "retrosyn_service_startup_json_ensemble_test",
        ROOT / "services/retrosyn-svc/src/retrosyn_svc/main.py",
    )
    monkeypatch.delenv("AIZYNTH_CONFIG_PATH", raising=False)
    monkeypatch.delenv("RETROSYN_PLANNER_COMMAND", raising=False)
    monkeypatch.setenv(
        "RETROSYN_PLANNER_COMMANDS_JSON",
        json.dumps({"rsgpt": f"{sys.executable} -c \"print('ok')\""}),
    )

    statuses = module._require_planner_runtime()
    planner_status = next(
        status for status in statuses if status.name == "retrosyn_rsgpt_planner_command"
    )

    assert planner_status.available is True
    assert planner_status.source == "RETROSYN_PLANNER_COMMANDS_JSON"


def test_retrosyn_service_startup_rejects_partially_missing_json_planner_ensemble(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module(
        "retrosyn_service_startup_partial_json_ensemble_test",
        ROOT / "services/retrosyn-svc/src/retrosyn_svc/main.py",
    )
    monkeypatch.delenv("AIZYNTH_CONFIG_PATH", raising=False)
    monkeypatch.delenv("RETROSYN_PLANNER_COMMAND", raising=False)
    monkeypatch.setenv(
        "RETROSYN_PLANNER_COMMANDS_JSON",
        json.dumps(
            {
                "rsgpt": f"{sys.executable} -c \"print('ok')\"",
                "ualign": "missing-retrosyn-planner --json",
            }
        ),
    )

    with pytest.raises(RuntimeError, match="retrosyn_ualign_planner_command"):
        module._require_planner_runtime()


def test_retrosyn_service_startup_rejects_missing_external_planner_without_aizynth_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module(
        "retrosyn_service_startup_missing_external_planner_test",
        ROOT / "services/retrosyn-svc/src/retrosyn_svc/main.py",
    )
    monkeypatch.delenv("AIZYNTH_CONFIG_PATH", raising=False)
    monkeypatch.setenv("RETROSYN_PLANNER_COMMAND", "missing-retrosyn-planner --json")

    with pytest.raises(RuntimeError, match="retrosyn_planner_command"):
        module._require_planner_runtime()


def test_retrosyn_planner_command_preflight_rejects_missing_executable() -> None:
    module = _load_module(
        "retrosyn_service_planner_command_preflight_test",
        ROOT / "services/retrosyn-svc/src/retrosyn_svc/main.py",
    )

    with pytest.raises(RuntimeError, match="not found"):
        module._run_planner_command_sync(
            "missing-retrosyn-planner --json",
            {"smiles": "CCO", "max_routes": 1, "engine": "rsgpt"},
        )


def test_retrosyn_deployment_wires_external_planner_env() -> None:
    import yaml

    compose = (ROOT / "infra/docker/docker-compose.dev.yml").read_text(encoding="utf-8")
    k8s = (ROOT / "infra/kubernetes/deployments/moleculeforge-services.yaml").read_text(
        encoding="utf-8"
    )
    helm_values = (ROOT / "infra/helm/moleculeforge/values.yaml").read_text(encoding="utf-8")

    for env_name in (
        "RETROSYN_PLANNER_COMMAND",
        "RETROSYN_PLANNER_COMMANDS_JSON",
        "RASCORE_PLANNER_COMMAND",
        "RSGPT_PLANNER_COMMAND",
        "UALIGN_PLANNER_COMMAND",
        "AIZYNTH_PLANNER_COMMAND",
        "AIZYNTH_CONFIG_PATH",
        "RETROSYN_PLANNER_COMMAND_TIMEOUT_SECONDS",
        "HUMU_ENCODER_TARGET",
    ):
        assert env_name in compose
        assert env_name in k8s
        assert env_name in helm_values

    assert (
        "RETROSYN_PLANNER_COMMAND_TIMEOUT_SECONDS: ${RETROSYN_PLANNER_COMMAND_TIMEOUT_SECONDS:-300}"
    ) in compose
    assert (
        "AIZYNTH_CONFIG_PATH: ${AIZYNTH_CONFIG_PATH:-models/artifacts/aizynthfinder/config.yml}"
    ) in compose
    assert "name: retrosyn-planner-config" in k8s
    assert "configMapKeyRef:" in k8s
    assert "envValueFrom:" in helm_values
    k8s_configmaps = {
        (doc["metadata"]["namespace"], doc["metadata"]["name"]): doc
        for doc in yaml.safe_load_all(k8s)
        if isinstance(doc, dict) and doc.get("kind") == "ConfigMap"
    }
    for namespace in ("mf-agents", "mf-oracles"):
        retrosyn_config = k8s_configmaps[(namespace, "retrosyn-planner-config")]["data"]
        assert retrosyn_config["planner-command"] == ""
        assert retrosyn_config["aizynth-planner-command"] == (
            "python tools/retrosyn/aizynth_planner_wrapper.py"
        )
        assert retrosyn_config["rsgpt-planner-command"] == (
            "python tools/retrosyn/rsgpt_planner_wrapper.py"
        )
        assert retrosyn_config["ualign-planner-command"] == (
            "python tools/retrosyn/ualign_planner_wrapper.py"
        )
        assert retrosyn_config["aizynth-config-path"] == (
            "models/artifacts/aizynthfinder/config.yml"
        )
        assert retrosyn_config["planner-command-timeout-seconds"] == "300"

    helm_config = yaml.safe_load(helm_values)
    helm_configmaps = {
        (config["namespace"], config["name"]): config
        for config in helm_config["configMaps"].values()
    }
    for namespace in ("mf-agents", "mf-oracles"):
        helm_retrosyn_config = helm_configmaps[(namespace, "retrosyn-planner-config")]["data"]
        assert helm_retrosyn_config["planner-command"] == ""
        assert helm_retrosyn_config["aizynth-planner-command"] == (
            "python tools/retrosyn/aizynth_planner_wrapper.py"
        )
        assert helm_retrosyn_config["rsgpt-planner-command"] == (
            "python tools/retrosyn/rsgpt_planner_wrapper.py"
        )
        assert helm_retrosyn_config["ualign-planner-command"] == (
            "python tools/retrosyn/ualign_planner_wrapper.py"
        )
        assert helm_retrosyn_config["aizynth-config-path"] == (
            "models/artifacts/aizynthfinder/config.yml"
        )

    helm_template = (ROOT / "infra/helm/moleculeforge/templates/services.yaml").read_text(
        encoding="utf-8"
    )
    assert "kind: ConfigMap" in helm_template
    assert ".Values.configMaps" in helm_template


def test_retrosyn_agent_deployment_uses_retrosyn_service() -> None:
    import yaml

    compose = yaml.safe_load(
        (ROOT / "infra/docker/docker-compose.dev.yml").read_text(encoding="utf-8")
    )
    compose_agent = compose["services"]["retrosyn-agent"]
    python_path = compose["x-service-common"]["environment"]["PYTHONPATH"].split(":")
    assert "/workspace/models/mf-retrosyn/aizynth_wrapper/src" in python_path
    assert compose_agent["environment"]["RETROSYN_SERVICE_TARGET"] == (
        "${RETROSYN_SERVICE_TARGET:-retrosyn-svc:50057}"
    )
    assert "retrosyn-svc" in compose_agent["depends_on"]

    k8s_documents = list(
        yaml.safe_load_all(
            (ROOT / "infra/kubernetes/deployments/moleculeforge-services.yaml").read_text(
                encoding="utf-8"
            )
        )
    )
    k8s_deployments = {
        doc["metadata"]["name"]: doc
        for doc in k8s_documents
        if isinstance(doc, dict) and doc.get("kind") == "Deployment"
    }
    k8s_agent_env = {
        item["name"]: item
        for item in k8s_deployments["retrosyn-agent"]["spec"]["template"]["spec"]["containers"][0][
            "env"
        ]
    }
    assert k8s_agent_env["RETROSYN_SERVICE_TARGET"]["value"] == (
        "retrosyn-svc.mf-oracles.svc.cluster.local:50057"
    )

    helm_values = yaml.safe_load(
        (ROOT / "infra/helm/moleculeforge/values.yaml").read_text(encoding="utf-8")
    )
    assert helm_values["services"]["retrosyn-agent"]["env"]["RETROSYN_SERVICE_TARGET"] == (
        "retrosyn-svc.mf-oracles.svc.cluster.local:50057"
    )


@pytest.mark.asyncio
async def test_hfm_service_passes_request_intent_cone_to_generator(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_module(
        "hfm_service_intent_cone_test",
        ROOT / "services/hfm-generator-svc/src/hfm_generator_svc/main.py",
    )
    checkpoint = tmp_path / "hfm.ckpt"
    decoder = tmp_path / "decoder.json"
    checkpoint.write_bytes(b"checkpoint")
    decoder.write_text('{"entries": [{"smiles": "CCO", "latent": []}]}', encoding="utf-8")
    monkeypatch.setenv("HFM_CHECKPOINT_PATH", str(checkpoint))
    monkeypatch.setenv("HFM_DECODER_PATH", str(decoder))

    class Generator:
        def __init__(self) -> None:
            self.intent_cone = None

        async def generate(self, batch_size: int, intent_cone=None, **kwargs):
            self.intent_cone = intent_cone
            return [Molecule(smiles="CCO")]

    generator = Generator()
    service = module.HFMGeneratorServicer(generator=generator)

    await service.Generate(
        _valid_generator_request(
            batch_size=1,
            generator_params={"sampling_seed": "7"},
        ),
        None,
    )

    assert generator.intent_cone is not None
    assert generator.intent_cone.axis[0] == 1.0
    assert generator.intent_cone.half_angle == pytest.approx(0.2)


def test_hfm_service_runtime_accepts_molecular_decoder_command(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_module(
        "hfm_service_decoder_command_runtime_test",
        ROOT / "services/hfm-generator-svc/src/hfm_generator_svc/main.py",
    )
    checkpoint = tmp_path / "hfm.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setenv("HFM_CHECKPOINT_PATH", str(checkpoint))
    monkeypatch.delenv("HFM_DECODER_PATH", raising=False)
    monkeypatch.setenv("HFM_MOLECULAR_DECODER_COMMAND", "python decoder.py")

    status = module.runtime_status()

    assert all(item["available"] for item in status if item["required"])
    assert [item["name"] for item in status] == [
        "hfm_checkpoint",
        "hfm_molecular_decoder_command",
    ]


def test_hfm_service_runtime_rejects_missing_molecular_decoder_command(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_module(
        "hfm_service_missing_decoder_command_runtime_test",
        ROOT / "services/hfm-generator-svc/src/hfm_generator_svc/main.py",
    )
    checkpoint = tmp_path / "hfm.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setenv("HFM_CHECKPOINT_PATH", str(checkpoint))
    monkeypatch.delenv("HFM_DECODER_PATH", raising=False)
    monkeypatch.setenv("HFM_MOLECULAR_DECODER_COMMAND", "missing-hfm-decoder --json")

    status = module.runtime_status()

    decoder_status = next(
        item for item in status if item["name"] == "hfm_molecular_decoder_command"
    )
    assert decoder_status["configured"] is True
    assert decoder_status["available"] is False
    assert "not found" in decoder_status["message"]


@pytest.mark.asyncio
async def test_hfm_validation_artifacts_require_explicit_opt_in(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from mf_generators.hfm_3d.generator import (
        bootstrap_validation_artifacts,
        load_validation_artifact_metadata,
    )
    from rdkit import Chem

    paths = await bootstrap_validation_artifacts(tmp_path / "hfm-validation")
    copied_directory = tmp_path / "copied-hfm-validation"
    copied_directory.mkdir()
    copied_checkpoint = copied_directory / paths["checkpoint"].name
    copied_decoder = copied_directory / paths["decoder"].name
    copied_checkpoint.write_bytes(paths["checkpoint"].read_bytes())
    copied_decoder.write_bytes(paths["decoder"].read_bytes())
    assert not (copied_directory / "moleculeforge_validation_artifact.json").exists()
    metadata = load_validation_artifact_metadata(copied_checkpoint)
    assert metadata is not None
    assert metadata["schema_version"] == "moleculeforge.validation_artifact.v1"

    monkeypatch.setenv("HFM_CHECKPOINT_PATH", str(copied_checkpoint))
    monkeypatch.setenv("HFM_DECODER_PATH", str(copied_decoder))
    monkeypatch.delenv("HFM_MOLECULAR_DECODER_COMMAND", raising=False)
    monkeypatch.delenv("HFM_ALLOW_VALIDATION_ARTIFACT", raising=False)
    module = _load_module(
        "hfm_validation_artifact_opt_in_test",
        ROOT / "services/hfm-generator-svc/src/hfm_generator_svc/main.py",
    )

    with pytest.raises(RuntimeError, match="HFM_ALLOW_VALIDATION_ARTIFACT=true"):
        module._require_runtime()

    monkeypatch.setenv("HFM_ALLOW_VALIDATION_ARTIFACT", "true")
    statuses = module._require_runtime()

    assert all(status.available for status in statuses)
    generator = module._build_generator()
    molecules = await generator.generate(
        batch_size=1,
        sampling_seed=7,
        flow_steps=1,
    )
    assert Chem.MolFromSmiles(molecules[0].smiles) is not None

    decoder_payload = json.loads(copied_decoder.read_text(encoding="utf-8"))
    decoder_payload["moleculeforge_validation_artifact"]["generator"] = "other"
    copied_decoder.write_text(json.dumps(decoder_payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="validation artifact metadata is invalid"):
        module._require_runtime()


def test_hfm_deployment_wires_checkpoint_and_decoder_env() -> None:
    compose = (ROOT / "infra/docker/docker-compose.dev.yml").read_text(encoding="utf-8")
    k8s = (ROOT / "infra/kubernetes/deployments/moleculeforge-services.yaml").read_text(
        encoding="utf-8"
    )
    helm_values = (ROOT / "infra/helm/moleculeforge/values.yaml").read_text(encoding="utf-8")

    for env_name in (
        "HFM_CHECKPOINT_PATH",
        "HFM_DECODER_PATH",
        "HFM_MOLECULAR_DECODER_COMMAND",
    ):
        assert env_name in compose
        assert env_name in k8s
        assert env_name in helm_values
    assert "HFM_ALLOW_VALIDATION_ARTIFACT" in compose
    assert "HFM_ALLOW_VALIDATION_ARTIFACT" not in k8s
    assert "HFM_ALLOW_VALIDATION_ARTIFACT" not in helm_values

    assert "name: hfm-generator-config" in k8s
    assert "configMapKeyRef:" in k8s
    assert "envValueFrom:" in helm_values
    assert (
        "HFM_CHECKPOINT_PATH: "
        "/var/lib/moleculeforge/validation-artifacts/hfm/hfm_checkpoint.pt"
        in compose
    )
    assert (
        "HFM_DECODER_PATH: /var/lib/moleculeforge/validation-artifacts/hfm/decoder.json"
        in compose
    )
    for config in (
        _k8s_configmap_data(k8s, "mf-generators", "hfm-generator-config"),
        _helm_configmap_data(helm_values, "mf-generators", "hfm-generator-config"),
    ):
        assert config["checkpoint-path"] == ""
        assert config["decoder-path"] == ""
        assert config["molecular-decoder-command"] == ""


@pytest.mark.asyncio
async def test_crem_validation_artifact_requires_explicit_opt_in(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from mf_generators.crem_3d.generator import (
        bootstrap_validation_artifacts,
        load_validation_artifact_metadata,
    )
    from rdkit import Chem

    paths = await bootstrap_validation_artifacts(tmp_path / "crem-validation")
    copied_directory = tmp_path / "copied-crem-validation"
    copied_directory.mkdir()
    copied_database = copied_directory / paths["mmp_database"].name
    copied_database.write_bytes(paths["mmp_database"].read_bytes())
    assert not (copied_directory / "moleculeforge_validation_artifact.json").exists()
    metadata = load_validation_artifact_metadata(copied_database)
    assert metadata is not None
    assert metadata["schema_version"] == "moleculeforge.validation_artifact.v1"

    monkeypatch.setenv("CREM_MMP_DB_PATH", str(copied_database))
    monkeypatch.delenv("CREM_PHARMACOPHORE_SCORER_COMMAND", raising=False)
    monkeypatch.delenv("CREM_HUMU_SCORER_COMMAND", raising=False)
    monkeypatch.delenv("CREM_ALLOW_VALIDATION_ARTIFACT", raising=False)
    module = _load_module(
        "crem_validation_artifact_opt_in_test",
        ROOT / "services/crem-generator-svc/src/crem_generator_svc/main.py",
    )

    with pytest.raises(RuntimeError, match="CREM_ALLOW_VALIDATION_ARTIFACT=true"):
        module._require_runtime()

    monkeypatch.setenv("CREM_ALLOW_VALIDATION_ARTIFACT", "true")
    statuses = module._require_runtime()

    assert all(status.available for status in statuses)
    molecules = await module._build_generator().generate(
        batch_size=2,
        seed_smiles="c1ccccc1",
    )
    assert len(molecules) == 2
    assert all(Chem.MolFromSmiles(molecule.smiles) is not None for molecule in molecules)

    malformed_database = json.loads(copied_database.read_text(encoding="utf-8"))
    malformed_database["moleculeforge_validation_artifact"]["seed"] = -1
    copied_database.write_text(json.dumps(malformed_database), encoding="utf-8")
    with pytest.raises(RuntimeError, match="validation artifact metadata is invalid"):
        module._require_runtime()


def test_crem_deployment_wires_mmp_and_external_scorer_env() -> None:
    compose = (ROOT / "infra/docker/docker-compose.dev.yml").read_text(encoding="utf-8")
    k8s = (ROOT / "infra/kubernetes/deployments/moleculeforge-services.yaml").read_text(
        encoding="utf-8"
    )
    helm_values = (ROOT / "infra/helm/moleculeforge/values.yaml").read_text(encoding="utf-8")

    for env_name in (
        "CREM_MMP_DB_PATH",
        "CREM_DOCK_ORACLE_TARGET",
        "CREM_PHARMACOPHORE_SCORER_COMMAND",
        "CREM_HUMU_SCORER_COMMAND",
        "CREM_SCORER_COMMAND_TIMEOUT_SECONDS",
    ):
        assert env_name in compose
        assert env_name in k8s
        assert env_name in helm_values
    assert "CREM_ALLOW_VALIDATION_ARTIFACT" in compose
    assert "CREM_ALLOW_VALIDATION_ARTIFACT" not in k8s
    assert "CREM_ALLOW_VALIDATION_ARTIFACT" not in helm_values

    assert (
        "CREM_SCORER_COMMAND_TIMEOUT_SECONDS: ${CREM_SCORER_COMMAND_TIMEOUT_SECONDS:-120}"
    ) in compose
    assert (
        "CREM_MMP_DB_PATH: "
        "/var/lib/moleculeforge/validation-artifacts/crem/crem_mmp_database.json"
        in compose
    )
    assert "name: crem-generator-config" in k8s
    assert "configMapKeyRef:" in k8s
    assert "envValueFrom:" in helm_values
    for config in (
        _k8s_configmap_data(k8s, "mf-generators", "crem-generator-config"),
        _helm_configmap_data(helm_values, "mf-generators", "crem-generator-config"),
    ):
        assert config["mmp-db-path"] == ""
        assert config["dock-oracle-target"] == ""
        assert config["pharmacophore-scorer-command"] == ""
        assert config["humu-scorer-command"] == ""
        assert config["scorer-command-timeout-seconds"] == "120"


def test_crem_runtime_rejects_missing_external_scorer_command(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_module(
        "crem_missing_scorer_runtime_test",
        ROOT / "services/crem-generator-svc/src/crem_generator_svc/main.py",
    )
    mmp_db = tmp_path / "crem_mmp_database.json"
    mmp_db.write_text("[]", encoding="utf-8")
    monkeypatch.setenv("CREM_MMP_DB_PATH", str(mmp_db))
    monkeypatch.setenv("CREM_PHARMACOPHORE_SCORER_COMMAND", "missing-crem-scorer --json")
    monkeypatch.delenv("CREM_HUMU_SCORER_COMMAND", raising=False)

    status = module.runtime_status()

    scorer_status = next(
        item for item in status if item["name"] == "crem_pharmacophore_scorer_command"
    )
    assert scorer_status["configured"] is True
    assert scorer_status["available"] is False
    assert "not found" in scorer_status["message"]


@pytest.mark.asyncio
async def test_aizynth_retrosyn_rejects_empty_step_routes() -> None:
    from mf_retrosyn.aizynth.retrosyn import AiZynthRetrosyn

    class Runner:
        def find_routes(self, smiles: str, max_routes: int = 10) -> list[dict]:
            return [{"route_id": "aizynth-1", "smiles": smiles, "steps": []}]

    with pytest.raises(ValueError, match="must contain non-empty steps"):
        await AiZynthRetrosyn(runner=Runner()).find_routes("CCO", max_routes=1)


@pytest.mark.asyncio
async def test_aizynth_retrosyn_rejects_incomplete_steps() -> None:
    from mf_retrosyn.aizynth.retrosyn import AiZynthRetrosyn

    class Runner:
        def find_routes(self, smiles: str, max_routes: int = 10) -> list[dict]:
            return [
                {
                    "route_id": "aizynth-1",
                    "smiles": smiles,
                    "steps": [
                        {
                            "step_id": "retro-1",
                            "reaction": "",
                            "reactants": [{"smiles": "CCO"}, {"smiles": "O=O"}],
                        }
                    ],
                }
            ]

    with pytest.raises(ValueError, match="missing reaction"):
        await AiZynthRetrosyn(runner=Runner()).find_routes("CCOO", max_routes=1)


@pytest.mark.asyncio
async def test_boltz2_service_delegates_to_runner() -> None:
    module = _load_module(
        "boltz2_service_runner_test",
        ROOT / "services/boltz2-svc/src/boltz2_svc/main.py",
    )

    class Runner:
        def __init__(self) -> None:
            self.calls: list[tuple[str, list[str], int]] = []

        async def predict_affinity(
            self,
            protein_pdb_id: str,
            ligand_smiles: list[str],
            ensemble_size: int,
        ) -> list[dict]:
            self.calls.append((protein_pdb_id, ligand_smiles, ensemble_size))
            return [
                {
                    "protein_pdb_id": protein_pdb_id,
                    "ligand_smiles": ligand_smiles[0],
                    "delta_g_kcal_mol": -8.2,
                    "uncertainty": 0.2,
                    "ki_nm": 12.0,
                    "ensemble_size": ensemble_size,
                    "per_member_dg": [-8.0, -8.4],
                }
            ]

    runner = Runner()
    service = module.Boltz2Servicer(runner=runner)
    request = SimpleNamespace(
        project_id="proj-1",
        protein_pdb_id="6OIM",
        ligand_smiles=["CCO"],
        ensemble_size=2,
    )

    response = await service.PredictAffinity(request, None)

    assert runner.calls == [("6OIM", ["CCO"], 2)]
    assert response.protein_pdb_id == "6OIM"
    assert response.affinities[0].ligand_smiles == "CCO"
    assert response.affinities[0].delta_g_kcal_mol == pytest.approx(-8.2)
    assert list(response.affinities[0].per_member_dg) == [-8.0, -8.4]


@pytest.mark.asyncio
async def test_boltz2_service_runs_configured_json_command(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_module(
        "boltz2_service_command_test",
        ROOT / "services/boltz2-svc/src/boltz2_svc/main.py",
    )
    runner = tmp_path / "boltz2_runner.py"
    runner.write_text(
        "import json, sys\n"
        "payload = json.load(sys.stdin)\n"
        "assert payload['protein_pdb_id'] == '6OIM'\n"
        "assert payload['ligand_smiles'] == ['CCO', 'CCN']\n"
        "assert payload['ensemble_size'] == 2\n"
        "print(json.dumps({"
        "'affinities': ["
        "{'protein_pdb_id': '6OIM', 'ligand_smiles': 'CCO', "
        "'delta_g_kcal_mol': -8.2, 'uncertainty': 0.2, "
        "'ki_nm': 12.0, 'ensemble_size': 2, 'per_member_dg': [-8.0, -8.4]},"
        "{'protein_pdb_id': '6OIM', 'ligand_smiles': 'CCN', "
        "'delta_g_kcal_mol': -7.1, 'uncertainty': 0.3, "
        "'ki_nm': 20.0, 'ensemble_size': 2, 'per_member_dg': [-6.8, -7.4]}"
        "]"
        "}))\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("BOLTZ_MODEL_PATH", raising=False)
    monkeypatch.delenv("BOLTZ_INPUT_TEMPLATE_DIR", raising=False)
    monkeypatch.setenv("BOLTZ2_ORACLE_COMMAND", f"{sys.executable} {runner}")
    service = module.Boltz2Servicer()

    response = await service.PredictAffinity(
        SimpleNamespace(
            protein_pdb_id="6OIM",
            ligand_smiles=["CCO", "CCN"],
            ensemble_size=2,
        ),
        None,
    )

    assert response.protein_pdb_id == "6OIM"
    assert [affinity.ligand_smiles for affinity in response.affinities] == ["CCO", "CCN"]
    assert response.affinities[0].delta_g_kcal_mol == pytest.approx(-8.2)
    assert response.affinities[1].ki_nm == pytest.approx(20.0)


def test_boltz2_runtime_rejects_missing_oracle_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BOLTZ2_ORACLE_COMMAND", "missing-boltz2-runner --json")
    module = _load_module(
        "boltz2_missing_oracle_command_runtime_test",
        ROOT / "services/boltz2-svc/src/boltz2_svc/main.py",
    )

    status = module.runtime_status()

    command_status = next(item for item in status if item["name"] == "boltz2_oracle_command")
    assert command_status["configured"] is True
    assert command_status["available"] is False
    assert command_status["source"] == "BOLTZ2_ORACLE_COMMAND"
    assert "not found" in command_status["message"]


def test_boltz2_command_runner_preflight_rejects_missing_executable() -> None:
    module = _load_module(
        "boltz2_command_preflight_test",
        ROOT / "services/boltz2-svc/src/boltz2_svc/main.py",
    )

    with pytest.raises(RuntimeError, match="not found"):
        module.BoltzCommandRunner("missing-boltz2-runner --json").predict_affinity(
            "6OIM",
            ["CCO"],
            1,
        )


@pytest.mark.asyncio
async def test_boltz2_oracle_service_maps_affinity_scores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mf_core.proto_gen.moleculeforge.v1.oracle import boltz2_pb2, oracle_pb2

    module = _load_module(
        "boltz2_oracle_adapter_test",
        ROOT / "services/boltz2-svc/src/boltz2_svc/main.py",
    )

    class Boltz2Service:
        def __init__(self) -> None:
            self.requests = []

        async def PredictAffinity(self, request, context):
            self.requests.append(request)
            return boltz2_pb2.Boltz2BatchResponse(
                protein_pdb_id=request.protein_pdb_id,
                affinities=[
                    boltz2_pb2.Boltz2BindingAffinity(
                        protein_pdb_id=request.protein_pdb_id,
                        ligand_smiles=request.ligand_smiles[0],
                        delta_g_kcal_mol=-8.2,
                        uncertainty=0.2,
                        ki_nm=12.0,
                        ensemble_size=request.ensemble_size,
                        per_member_dg=[-8.0, -8.4],
                    )
                ],
                elapsed_ms=21,
            )

    service = Boltz2Service()
    oracle = module.Boltz2OracleServicer(service=service)

    response = await oracle.PredictWithUncertainty(
        oracle_pb2.OracleBatchRequest(
            project_id="project-1",
            request_id="request-1",
            molecule_smiles=["CCO"],
            requested_properties=["affinity"],
            level=oracle_pb2.L1_ML_SURROGATE,
            return_uncertainty=True,
            protein_pdb_id="6OIM",
            oracle_parameters={"ensemble_size": "2"},
        ),
        None,
    )

    assert service.requests[0].project_id == "project-1"
    assert service.requests[0].protein_pdb_id == "6OIM"
    assert list(service.requests[0].ligand_smiles) == ["CCO"]
    assert service.requests[0].ensemble_size == 2
    assert response.batch_id == "request-1"
    assert response.total_elapsed_ms == 21
    assert response.evaluations[0].oracle_name == "boltz2"
    assert response.evaluations[0].molecule_smiles == "CCO"
    assert response.evaluations[0].level == oracle_pb2.L1_ML_SURROGATE
    assert response.evaluations[0].scores == {"affinity": -8.2}
    assert response.evaluations[0].uncertainties == {"affinity": 0.2}
    assert response.evaluations[0].success is True


def test_boltz_cli_runner_parses_affinity_json(tmp_path: Path) -> None:
    module = _load_module(
        "boltz2_cli_runner_test",
        ROOT / "services/boltz2-svc/src/boltz2_svc/main.py",
    )
    model_path = tmp_path / "boltz-2"
    model_path.mkdir()
    (model_path / "boltz2_conf.ckpt").write_bytes(b"conf")
    (model_path / "boltz2_aff.ckpt").write_bytes(b"aff")
    template_dir = tmp_path / "templates"
    template_dir.mkdir()
    (template_dir / "6OIM.yaml").write_text(
        """
version: 1
sequences:
  - protein:
      id: A
      sequence: MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQV
      msa: empty
  - ligand:
      id: L
      smiles: "__LIGAND_SMILES__"
properties:
  - affinity:
      binder: L
""".strip(),
        encoding="utf-8",
    )
    calls: list[list[str]] = []

    def run_command(command, **kwargs):
        calls.append(command)
        input_path = Path(command[2])
        out_dir = Path(command[command.index("--out_dir") + 1])
        prediction_dir = out_dir / "predictions" / input_path.stem
        prediction_dir.mkdir(parents=True)
        (prediction_dir / f"affinity_{input_path.stem}.json").write_text(
            json.dumps(
                {
                    "affinity_pred_value": -3.0,
                    "affinity_probability_binary": 0.91,
                    "affinity_pred_value1": -3.1,
                    "affinity_pred_value2": -2.9,
                }
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    runner = module.BoltzCliRunner(
        model_path=model_path,
        template_dir=template_dir,
        work_dir=tmp_path / "work",
        run_command=run_command,
    )

    result = runner.predict_affinity("6OIM", ["CCO"], ensemble_size=2)

    assert calls
    assert calls[0][:3] == ["boltz", "predict", str(tmp_path / "work/inputs/6OIM_0.yaml")]
    assert "--checkpoint" in calls[0]
    assert "--affinity_checkpoint" in calls[0]
    assert result[0]["ligand_smiles"] == "CCO"
    assert result[0]["delta_g_kcal_mol"] == pytest.approx(-12.276)
    assert result[0]["ki_nm"] == pytest.approx(1.0)
    assert result[0]["ensemble_size"] == 2
    assert result[0]["per_member_dg"] == pytest.approx([-12.4124, -12.1396])


def test_retrosyn_route_without_explicit_availability_is_not_marked_commercial() -> None:
    module = _load_module(
        "retrosyn_route_availability_test",
        ROOT / "services/retrosyn-svc/src/retrosyn_svc/main.py",
    )

    route = module._synthetic_route(
        {
            "route_id": "route-1",
            "steps": [_valid_retrosyn_step("CCO>>CC=O")],
        }
    )

    assert route.all_commercially_available is False


@pytest.mark.asyncio
async def test_critic_service_evaluate_runs_scientific_critic() -> None:
    module = _load_module(
        "critic_service_evaluate_test",
        ROOT / "services/critic-svc/src/critic_svc/main.py",
    )
    from mf_core.proto_gen.moleculeforge.v1.agent import critic_pb2

    class Agent:
        async def evaluate_molecule(self, payload):
            assert payload == {
                "workflow_scope": "full",
                "project_id": "project-critic",
                "run_id": "run-critic",
                "request_id": "request-critic",
                "schema_version": "critic.batch.v1",
                "candidate_id": "candidate-critic",
                "candidate_index": 2,
                "canonical_smiles": "CCO",
                "smiles": "CCO",
                "properties": {},
            }
            return {
                "smiles": "CCO",
                "verdict": "pass",
                "passed": 1,
                "failed": 0,
                "total_rules": 1,
                "rule_results": [
                    {
                        "rule_id": "rule-test",
                        "rule_name": "Test rule",
                        "verdict": "pass",
                        "score": 1.0,
                        "reasoning": "passed",
                    }
                ],
            }

    response = await module.CriticServicer(agent=Agent()).Evaluate(
        critic_pb2.CriticBatchResult(
            molecule_smiles="CCO",
            project_id="project-critic",
            run_id="run-critic",
            request_id="request-critic",
            schema_version="critic.batch.v1",
            candidate_id="candidate-critic",
            candidate_index=2,
            canonical_smiles="CCO",
        ),
        None,
    )

    assert response.molecule_smiles == "CCO"
    assert response.project_id == "project-critic"
    assert response.run_id == "run-critic"
    assert response.request_id == "request-critic"
    assert response.schema_version == "critic.batch.v1"
    assert response.candidate_id == "candidate-critic"
    assert response.candidate_index == 2
    assert response.canonical_smiles == "CCO"
    assert response.rules_evaluated == 1
    assert response.rule_results


@pytest.mark.asyncio
async def test_critic_service_rejects_candidate_smiles_identity_mismatch() -> None:
    module = _load_module(
        "critic_service_identity_mismatch_test",
        ROOT / "services/critic-svc/src/critic_svc/main.py",
    )
    from mf_core.proto_gen.moleculeforge.v1.agent import critic_pb2

    context = _RecordingAbortContext()
    with pytest.raises(ValueError, match="canonical_smiles"):
        await module.CriticServicer(agent=object()).Evaluate(
            critic_pb2.CriticBatchResult(
                molecule_smiles="CCO",
                project_id="project-critic",
                run_id="run-critic",
                request_id="request-critic",
                schema_version="critic.batch.v1",
                candidate_id="candidate-critic",
                candidate_index=2,
                canonical_smiles="CCC",
            ),
            context,
        )
    assert context.code == grpc.StatusCode.INVALID_ARGUMENT
    assert "canonical_smiles" in context.message


@pytest.mark.asyncio
async def test_critic_agent_persists_critic_verdict_belief() -> None:
    module = _load_module(
        "critic_agent_crg_repository_test",
        ROOT / "agents/critic_agent/src/critic_agent/agent.py",
    )

    class CRGRepository:
        def __init__(self) -> None:
            self.beliefs: list[dict] = []

        async def write_workflow_belief(self, **kwargs) -> None:
            self.beliefs.append(kwargs)

    repository = CRGRepository()
    agent = module.ScientificCriticAgent(crg_repository=repository)
    agent.rules = []

    await agent.evaluate_molecule(
        {
            "project_id": "project-1",
            "run_id": "run-1",
            "smiles": "CCO",
            "properties": {},
        }
    )

    assert len(repository.beliefs) == 2
    belief = repository.beliefs[0]
    assert belief["project_id"] == "project-1"
    assert belief["run_id"] == "run-1"
    assert belief["subject"] == "CCO"
    assert belief["predicate"] == "critic_verdict"
    assert belief["object_value"] == "pass"
    assert belief["source_agent"] == "critic_agent"
    result_belief = repository.beliefs[1]
    assert result_belief["predicate"] == "critic_result"
    contract = json.loads(result_belief["object_value"])
    assert contract["schema_version"] == "critic_result.v1"
    assert len(contract["input_fingerprint"]) == 64
    assert contract["result"]["verdict"] == "pass"
    assert contract["result"]["rule_results"] == []


@pytest.mark.asyncio
async def test_critic_agent_uses_failed_validation_belief_from_shared_crg() -> None:
    module = _load_module(
        "critic_agent_crg_readback_test",
        ROOT / "agents/critic_agent/src/critic_agent/agent.py",
    )

    class CRGRepository:
        def __init__(self) -> None:
            self.beliefs: list[dict] = []

        async def get_run_crg(self, run_id: str) -> dict:
            assert run_id == "run-1"
            return {
                "beliefs": [
                    {
                        "id": "belief-validation-failed",
                        "subject": "CCO",
                        "predicate": "validation_status",
                        "object_value": "failed",
                        "confidence": 0.9,
                    }
                ]
            }

        async def write_workflow_belief(self, **kwargs) -> None:
            self.beliefs.append(kwargs)

    repository = CRGRepository()
    agent = module.ScientificCriticAgent(crg_repository=repository)
    agent.rules = []

    result = await agent.evaluate_molecule(
        {
            "project_id": "project-1",
            "run_id": "run-1",
            "smiles": "CCO",
            "properties": {},
        }
    )

    assert result["verdict"] == "fail"
    assert result["failed"] == 1
    assert result["rule_results"][0]["rule_id"] == "crg_validation_status"
    assert repository.beliefs[0]["object_value"] == "fail"
    assert repository.beliefs[0]["evidence_ids"] == ["crg_validation_status"]


@pytest.mark.asyncio
async def test_critic_agent_uses_zero_retrosyn_routes_belief_from_shared_crg() -> None:
    module = _load_module(
        "critic_agent_retrosyn_crg_readback_test",
        ROOT / "agents/critic_agent/src/critic_agent/agent.py",
    )

    class CRGRepository:
        def __init__(self) -> None:
            self.beliefs: list[dict] = []

        async def get_run_crg(self, run_id: str) -> dict:
            assert run_id == "run-1"
            return {
                "beliefs": [
                    {
                        "id": "belief-zero-routes",
                        "subject": "CCO",
                        "predicate": "retrosyn_routes",
                        "object_value": "0",
                        "confidence": 1.0,
                    }
                ]
            }

        async def write_workflow_belief(self, **kwargs) -> None:
            self.beliefs.append(kwargs)

    repository = CRGRepository()
    agent = module.ScientificCriticAgent(crg_repository=repository)
    agent.rules = []

    result = await agent.evaluate_molecule(
        {
            "project_id": "project-1",
            "run_id": "run-1",
            "smiles": "CCO",
            "properties": {},
        }
    )

    assert result["verdict"] == "fail"
    assert result["failed"] == 1
    assert result["rule_results"][0]["rule_id"] == "crg_retrosyn_routes"
    assert repository.beliefs[0]["object_value"] == "fail"
    assert repository.beliefs[0]["evidence_ids"] == ["crg_retrosyn_routes"]


@pytest.mark.asyncio
async def test_critic_agent_does_not_use_scalar_verdict_as_cached_result() -> None:
    module = _load_module(
        "critic_agent_existing_verdict_crg_readback_test",
        ROOT / "agents/critic_agent/src/critic_agent/agent.py",
    )

    class CRGRepository:
        def __init__(self) -> None:
            self.beliefs: list[dict] = []

        async def get_run_crg(self, run_id: str) -> dict:
            assert run_id == "run-1"
            return {
                "beliefs": [
                    {
                        "id": "belief-existing-verdict",
                        "subject": "CCO",
                        "predicate": "critic_verdict",
                        "object_value": "pass",
                        "confidence": 0.8,
                    }
                ]
            }

        async def write_workflow_belief(self, **kwargs) -> None:
            self.beliefs.append(kwargs)

    class Rule:
        rule_id = "rule-current-input"
        name = "Current input"

        def __init__(self) -> None:
            self.calls = 0

        def evaluate(self, smiles, properties):
            self.calls += 1
            return {
                "rule_id": self.rule_id,
                "rule_name": self.name,
                "verdict": "pass",
                "score": 1.0,
                "reasoning": "current input evaluated",
            }

    repository = CRGRepository()
    agent = module.ScientificCriticAgent(crg_repository=repository)
    rule = Rule()
    agent.rules = [rule]

    result = await agent.evaluate_molecule(
        {
            "project_id": "project-1",
            "run_id": "run-1",
            "smiles": "CCO",
            "properties": {},
        }
    )

    assert result.get("cache_source") is None
    assert result["verdict"] == "pass"
    assert result["passed"] == 1
    assert result["failed"] == 0
    assert result["rule_results"][0]["rule_id"] == "rule-current-input"
    assert rule.calls == 1
    assert [belief["predicate"] for belief in repository.beliefs] == [
        "critic_verdict",
        "critic_result",
    ]


@pytest.mark.asyncio
async def test_critic_agent_cache_rejects_result_from_untrusted_agent() -> None:
    module = _load_module(
        "critic_agent_untrusted_cache_source_test",
        ROOT / "agents/critic_agent/src/critic_agent/agent.py",
    )

    class CRGRepository:
        def __init__(self) -> None:
            self.beliefs: list[dict] = []

        async def get_run_crg(self, run_id: str) -> dict:
            return {"beliefs": list(self.beliefs)}

        async def write_workflow_belief(self, **kwargs) -> None:
            self.beliefs.append(kwargs)

    class Rule:
        rule_id = "rule_current"
        name = "Current rule"

        def __init__(self) -> None:
            self.calls = 0

        def evaluate(self, smiles, properties):
            self.calls += 1
            return {
                "rule_id": self.rule_id,
                "rule_name": self.name,
                "verdict": "pass",
                "score": 1.0,
                "reasoning": "evaluated",
            }

    repository = CRGRepository()
    rule = Rule()
    agent = module.ScientificCriticAgent(crg_repository=repository)
    agent.rules = [rule]
    request = {
        "project_id": "project-1",
        "run_id": "run-1",
        "smiles": "CCO",
        "properties": {},
    }

    await agent.evaluate_molecule(request)
    cached_belief = next(
        belief for belief in repository.beliefs if belief["predicate"] == "critic_result"
    )
    cached_belief["source_agent"] = "validation_agent"
    second = await agent.evaluate_molecule(request)

    assert second.get("cache_source") is None
    assert rule.calls == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "damage",
    ["total_rules", "row_verdict_counts", "row_blocking_count"],
)
async def test_critic_agent_cache_rejects_internally_inconsistent_result(
    damage: str,
) -> None:
    module = _load_module(
        f"critic_agent_inconsistent_cache_{damage}_test",
        ROOT / "agents/critic_agent/src/critic_agent/agent.py",
    )

    class CRGRepository:
        def __init__(self) -> None:
            self.beliefs: list[dict] = []

        async def get_run_crg(self, run_id: str) -> dict:
            return {"beliefs": list(self.beliefs)}

        async def write_workflow_belief(self, **kwargs) -> None:
            self.beliefs.append(kwargs)

    class Rule:
        rule_id = "rule_current"
        name = "Current rule"

        def __init__(self) -> None:
            self.calls = 0

        def evaluate(self, smiles, properties):
            self.calls += 1
            return {
                "rule_id": self.rule_id,
                "rule_name": self.name,
                "verdict": "fail",
                "score": 0.0,
                "reasoning": "evaluated",
            }

    repository = CRGRepository()
    rule = Rule()
    agent = module.ScientificCriticAgent(crg_repository=repository)
    agent.rules = [rule]
    request = {
        "project_id": "project-1",
        "run_id": "run-1",
        "smiles": "CCO",
        "properties": {"_critic_blocking_rule_ids": ["rule_current"]},
    }

    await agent.evaluate_molecule(request)
    cached_belief = next(
        belief for belief in repository.beliefs if belief["predicate"] == "critic_result"
    )
    contract = json.loads(cached_belief["object_value"])
    result = contract["result"]
    if damage == "total_rules":
        result["total_rules"] = 2
    elif damage == "row_verdict_counts":
        result.update(
            {
                "verdict": "pass",
                "passed": 1,
                "failed": 0,
                "blocking_failed": 0,
                "non_blocking_failed": 0,
            }
        )
    else:
        result["rule_results"][0]["blocking"] = False
    cached_belief["object_value"] = json.dumps(contract, sort_keys=True)

    second = await agent.evaluate_molecule(request)

    assert second.get("cache_source") is None
    assert rule.calls == 2


@pytest.mark.asyncio
async def test_critic_agent_cache_reuses_identical_semantic_input() -> None:
    module = _load_module(
        "critic_agent_semantic_cache_hit_test",
        ROOT / "agents/critic_agent/src/critic_agent/agent.py",
    )

    class CRGRepository:
        def __init__(self) -> None:
            self.beliefs: list[dict] = []

        async def get_run_crg(self, run_id: str) -> dict:
            return {"beliefs": list(self.beliefs)}

        async def write_workflow_belief(self, **kwargs) -> None:
            self.beliefs.append(kwargs)

    class Rule:
        rule_id = "rule_dynamic"
        name = "Dynamic rule"

        def __init__(self) -> None:
            self.calls = 0

        def evaluate(self, smiles, properties):
            self.calls += 1
            return {
                "rule_id": self.rule_id,
                "rule_name": self.name,
                "verdict": "fail",
                "score": 0.1,
                "reasoning": "dynamic concern",
            }

    repository = CRGRepository()
    rule = Rule()
    agent = module.ScientificCriticAgent(crg_repository=repository)
    agent.rules = [rule]
    base = {
        "project_id": "project-1",
        "run_id": "run-1",
        "smiles": "CCO",
    }

    first = await agent.evaluate_molecule(
        {
            **base,
            "properties": {
                "risk": 0.9,
                "_critic_blocking_rule_ids": ["rule_dynamic", "other"],
            },
        }
    )
    second = await agent.evaluate_molecule(
        {
            **base,
            "properties": {
                "_critic_blocking_rule_ids": ["other", "rule_dynamic"],
                "risk": 0.9,
            },
        }
    )

    assert rule.calls == 1
    assert second["cache_source"] == "shared_crg"
    assert {key: value for key, value in second.items() if key != "cache_source"} == first


@pytest.mark.asyncio
async def test_critic_agent_cache_tracks_rule_configuration_not_runtime_counters() -> None:
    module = _load_module(
        "critic_agent_rule_configuration_cache_identity_test",
        ROOT / "agents/critic_agent/src/critic_agent/agent.py",
    )

    class CRGRepository:
        def __init__(self) -> None:
            self.beliefs: list[dict] = []

        async def get_run_crg(self, run_id: str) -> dict:
            return {"beliefs": list(self.beliefs)}

        async def write_workflow_belief(self, **kwargs) -> None:
            self.beliefs.append(kwargs)

    class ConfigurableRule:
        rule_id = "rule_configurable"
        name = "Configurable rule"

        def __init__(self, threshold: float) -> None:
            self.threshold = threshold
            self.calls = 0

        def evaluate(self, smiles, properties):
            self.calls += 1
            failed = float(properties["risk"]) >= self.threshold
            return {
                "rule_id": self.rule_id,
                "rule_name": self.name,
                "verdict": "fail" if failed else "pass",
                "score": float(properties["risk"]),
                "reasoning": "configured threshold",
            }

    repository = CRGRepository()
    rule = ConfigurableRule(threshold=0.5)
    agent = module.ScientificCriticAgent(crg_repository=repository)
    agent.rules = [rule]
    request = {
        "project_id": "project-1",
        "run_id": "run-1",
        "smiles": "CCO",
        "properties": {"risk": 0.7},
    }

    first = await agent.evaluate_molecule(request)
    identical = await agent.evaluate_molecule(request)
    rule.threshold = 0.9
    changed = await agent.evaluate_molecule(request)

    assert first["verdict"] == "fail"
    assert identical["cache_source"] == "shared_crg"
    assert changed["verdict"] == "pass"
    assert changed.get("cache_source") is None
    assert rule.calls == 2


@pytest.mark.asyncio
async def test_critic_agent_cache_tracks_effective_rule_implementation() -> None:
    module = _load_module(
        "critic_agent_rule_implementation_cache_identity_test",
        ROOT / "agents/critic_agent/src/critic_agent/agent.py",
    )

    class CRGRepository:
        def __init__(self) -> None:
            self.beliefs: list[dict] = []

        async def get_run_crg(self, run_id: str) -> dict:
            return {"beliefs": list(self.beliefs)}

        async def write_workflow_belief(self, **kwargs) -> None:
            self.beliefs.append(kwargs)

    class Rule:
        rule_id = "rule_dynamic_implementation"
        name = "Dynamic implementation"

        def __init__(self) -> None:
            self.calls = 0

        def evaluate(self, smiles, properties):
            self.calls += 1
            return {
                "rule_id": self.rule_id,
                "rule_name": self.name,
                "verdict": "fail",
                "score": 0.0,
                "reasoning": "original implementation",
            }

    def changed_evaluate(self, smiles, properties):
        self.calls += 1
        return {
            "rule_id": self.rule_id,
            "rule_name": self.name,
            "verdict": "pass",
            "score": 1.0,
            "reasoning": "changed implementation",
        }

    repository = CRGRepository()
    rule = Rule()
    agent = module.ScientificCriticAgent(crg_repository=repository)
    agent.rules = [rule]
    request = {
        "project_id": "project-1",
        "run_id": "run-1",
        "smiles": "CCO",
        "properties": {},
    }

    first = await agent.evaluate_molecule(request)
    rule.evaluate = changed_evaluate.__get__(rule, Rule)
    second = await agent.evaluate_molecule(request)

    assert first["verdict"] == "fail"
    assert second["verdict"] == "pass"
    assert second.get("cache_source") is None
    assert rule.calls == 2


@pytest.mark.asyncio
async def test_critic_agent_cache_accepts_explicit_rule_identity() -> None:
    module = _load_module(
        "critic_agent_explicit_rule_cache_identity_test",
        ROOT / "agents/critic_agent/src/critic_agent/agent.py",
    )

    class CRGRepository:
        def __init__(self) -> None:
            self.beliefs: list[dict] = []

        async def get_run_crg(self, run_id: str) -> dict:
            return {"beliefs": list(self.beliefs)}

        async def write_workflow_belief(self, **kwargs) -> None:
            self.beliefs.append(kwargs)

    class Rule:
        rule_id = "rule_explicit_identity"
        name = "Explicit identity"

        def __init__(self) -> None:
            self._threshold = 0.5
            self.identity_version = 1
            self.calls = 0

        def cache_identity(self):
            return {
                "version": self.identity_version,
                "threshold": self._threshold,
            }

        def evaluate(self, smiles, properties):
            self.calls += 1
            failed = float(properties["risk"]) >= self._threshold
            return {
                "rule_id": self.rule_id,
                "rule_name": self.name,
                "verdict": "fail" if failed else "pass",
                "score": float(properties["risk"]),
                "reasoning": "explicit identity threshold",
            }

    repository = CRGRepository()
    rule = Rule()
    agent = module.ScientificCriticAgent(crg_repository=repository)
    agent.rules = [rule]
    request = {
        "project_id": "project-1",
        "run_id": "run-1",
        "smiles": "CCO",
        "properties": {"risk": 0.7},
    }

    first = await agent.evaluate_molecule(request)
    rule._threshold = 0.9
    rule.identity_version = 2
    second = await agent.evaluate_molecule(request)

    assert first["verdict"] == "fail"
    assert second["verdict"] == "pass"
    assert second.get("cache_source") is None
    assert rule.calls == 2


@pytest.mark.asyncio
async def test_critic_agent_explicit_identity_still_tracks_rule_implementation() -> None:
    module = _load_module(
        "critic_agent_explicit_identity_implementation_test",
        ROOT / "agents/critic_agent/src/critic_agent/agent.py",
    )

    class CRGRepository:
        def __init__(self) -> None:
            self.beliefs: list[dict] = []

        async def get_run_crg(self, run_id: str) -> dict:
            return {"beliefs": list(self.beliefs)}

        async def write_workflow_belief(self, **kwargs) -> None:
            self.beliefs.append(kwargs)

    class Rule:
        rule_id = "rule_explicit_identity_implementation"
        name = "Explicit identity implementation"

        def __init__(self) -> None:
            self.calls = 0

        def cache_identity(self):
            return {"version": 1}

        def evaluate(self, smiles, properties):
            self.calls += 1
            return {
                "rule_id": self.rule_id,
                "rule_name": self.name,
                "verdict": "fail",
                "score": 0.0,
                "reasoning": "original implementation",
            }

    def changed_evaluate(self, smiles, properties):
        self.calls += 1
        return {
            "rule_id": self.rule_id,
            "rule_name": self.name,
            "verdict": "pass",
            "score": 1.0,
            "reasoning": "changed implementation",
        }

    repository = CRGRepository()
    rule = Rule()
    agent = module.ScientificCriticAgent(crg_repository=repository)
    agent.rules = [rule]
    request = {
        "project_id": "project-1",
        "run_id": "run-1",
        "smiles": "CCO",
        "properties": {},
    }

    first = await agent.evaluate_molecule(request)
    identical = await agent.evaluate_molecule(request)
    rule.evaluate = changed_evaluate.__get__(rule, Rule)
    changed = await agent.evaluate_molecule(request)

    assert first["verdict"] == "fail"
    assert identical["cache_source"] == "shared_crg"
    assert changed["verdict"] == "pass"
    assert changed.get("cache_source") is None
    assert rule.calls == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("first_properties", "second_properties"),
    [
        ({"risk": 0.1}, {"risk": 0.9}),
        (
            {"risk": 0.9, "_critic_blocking_rule_ids": []},
            {
                "risk": 0.9,
                "_critic_blocking_rule_ids": ["rule_dynamic"],
            },
        ),
    ],
)
async def test_critic_agent_cache_misses_for_changed_properties_or_policy(
    first_properties: dict,
    second_properties: dict,
) -> None:
    module = _load_module(
        "critic_agent_semantic_cache_miss_test",
        ROOT / "agents/critic_agent/src/critic_agent/agent.py",
    )

    class CRGRepository:
        def __init__(self) -> None:
            self.beliefs: list[dict] = []

        async def get_run_crg(self, run_id: str) -> dict:
            return {"beliefs": list(self.beliefs)}

        async def write_workflow_belief(self, **kwargs) -> None:
            self.beliefs.append(kwargs)

    class Rule:
        rule_id = "rule_dynamic"
        name = "Dynamic rule"

        def __init__(self) -> None:
            self.calls = 0

        def evaluate(self, smiles, properties):
            self.calls += 1
            failed = float(properties["risk"]) >= 0.5
            return {
                "rule_id": self.rule_id,
                "rule_name": self.name,
                "verdict": "fail" if failed else "pass",
                "score": float(properties["risk"]),
                "reasoning": "risk threshold",
            }

    repository = CRGRepository()
    rule = Rule()
    agent = module.ScientificCriticAgent(crg_repository=repository)
    agent.rules = [rule]
    base = {
        "project_id": "project-1",
        "run_id": "run-1",
        "smiles": "CCO",
    }

    first = await agent.evaluate_molecule({**base, "properties": first_properties})
    second = await agent.evaluate_molecule({**base, "properties": second_properties})

    assert first["verdict"] == "pass"
    assert second["verdict"] == "fail"
    assert second.get("cache_source") is None
    assert rule.calls == 2


@pytest.mark.asyncio
async def test_critic_agent_cache_misses_when_validation_evidence_changes() -> None:
    module = _load_module(
        "critic_agent_validation_evidence_cache_miss_test",
        ROOT / "agents/critic_agent/src/critic_agent/agent.py",
    )

    class CRGRepository:
        def __init__(self) -> None:
            self.external_beliefs: list[dict] = []
            self.persisted_beliefs: list[dict] = []

        async def get_run_crg(self, run_id: str) -> dict:
            return {
                "beliefs": [
                    *self.external_beliefs,
                    *self.persisted_beliefs,
                ]
            }

        async def write_workflow_belief(self, **kwargs) -> None:
            self.persisted_beliefs.append(kwargs)

    repository = CRGRepository()
    agent = module.ScientificCriticAgent(crg_repository=repository)
    agent.rules = []
    request = {
        "project_id": "project-1",
        "run_id": "run-1",
        "smiles": "CCO",
        "properties": {
            "_critic_blocking_rule_ids": ["crg_validation_status"],
        },
    }

    first = await agent.evaluate_molecule(request)
    repository.external_beliefs.append(
        {
            "id": "belief-validation-failed",
            "subject": "CCO",
            "predicate": "validation_status",
            "object_value": "failed",
            "confidence": 1.0,
        }
    )
    second = await agent.evaluate_molecule(request)

    assert first["verdict"] == "pass"
    assert second["verdict"] == "fail"
    assert second.get("cache_source") is None
    assert second["rule_results"][0]["rule_id"] == "crg_validation_status"


@pytest.mark.asyncio
async def test_critic_agent_uses_latest_validation_belief_by_timestamp() -> None:
    module = _load_module(
        "critic_agent_latest_validation_belief_test",
        ROOT / "agents/critic_agent/src/critic_agent/agent.py",
    )

    failed = {
        "id": "belief-validation-failed",
        "subject": "CCO",
        "predicate": "validation_status",
        "object_value": "failed",
        "confidence": 1.0,
        "timestamp_ns": 10,
    }
    validated = {
        "id": "belief-validation-validated",
        "subject": "CCO",
        "predicate": "validation_status",
        "object_value": "validated",
        "confidence": 1.0,
        "timestamp_ns": 20,
    }

    class CRGRepository:
        def __init__(self) -> None:
            self.external_beliefs = [failed]
            self.persisted_beliefs: list[dict] = []

        async def get_run_crg(self, run_id: str) -> dict:
            return {
                "beliefs": [
                    *self.external_beliefs,
                    *self.persisted_beliefs,
                ]
            }

        async def write_workflow_belief(self, **kwargs) -> None:
            self.persisted_beliefs.append(kwargs)

    repository = CRGRepository()
    agent = module.ScientificCriticAgent(crg_repository=repository)
    agent.rules = []
    request = {
        "project_id": "project-1",
        "run_id": "run-1",
        "smiles": "CCO",
        "properties": {
            "_critic_blocking_rule_ids": ["crg_validation_status"],
        },
    }

    first = await agent.evaluate_molecule(request)
    repository.external_beliefs = [validated, failed]
    second = await agent.evaluate_molecule(request)

    assert first["verdict"] == "fail"
    assert second["verdict"] == "pass"
    assert second.get("cache_source") is None
    assert second["rule_results"] == []


@pytest.mark.asyncio
async def test_critic_agent_cache_preserves_latest_zero_confidence() -> None:
    module = _load_module(
        "critic_agent_zero_confidence_cache_identity_test",
        ROOT / "agents/critic_agent/src/critic_agent/agent.py",
    )

    zero_confidence = {
        "id": "belief-validation-failed-zero",
        "subject": "CCO",
        "predicate": "validation_status",
        "object_value": "failed",
        "confidence": 0.0,
        "timestamp_ns": 10,
    }
    full_confidence = {
        "id": "belief-validation-failed-full",
        "subject": "CCO",
        "predicate": "validation_status",
        "object_value": "failed",
        "confidence": 1.0,
        "timestamp_ns": 20,
    }

    class CRGRepository:
        def __init__(self) -> None:
            self.external_beliefs = [zero_confidence]
            self.persisted_beliefs: list[dict] = []

        async def get_run_crg(self, run_id: str) -> dict:
            return {
                "beliefs": [
                    *self.external_beliefs,
                    *self.persisted_beliefs,
                ]
            }

        async def write_workflow_belief(self, **kwargs) -> None:
            self.persisted_beliefs.append(kwargs)

    repository = CRGRepository()
    agent = module.ScientificCriticAgent(crg_repository=repository)
    agent.rules = []
    request = {
        "project_id": "project-1",
        "run_id": "run-1",
        "smiles": "CCO",
        "properties": {
            "_critic_blocking_rule_ids": ["crg_validation_status"],
        },
    }

    first = await agent.evaluate_molecule(request)
    repository.external_beliefs = [full_confidence, zero_confidence]
    second = await agent.evaluate_molecule(request)

    assert first["rule_results"][0]["score"] == 0.0
    assert second["rule_results"][0]["score"] == 1.0
    assert second.get("cache_source") is None
    assert len(second["rule_results"]) == 1


@pytest.mark.asyncio
async def test_retrosyn_agent_persists_route_belief() -> None:
    module = _load_module(
        "retrosyn_agent_crg_repository_test",
        ROOT / "agents/retrosyn_agent/src/retrosyn_agent/agent.py",
    )

    class Planner:
        async def find_routes(self, smiles: str, max_routes: int) -> list[dict]:
            return [
                {
                    "route_id": "route-1",
                    "target_smiles": smiles,
                    "steps": [_valid_retrosyn_step("CCO>>CC=O")],
                    "max_routes": max_routes,
                }
            ]

    class CRGRepository:
        def __init__(self) -> None:
            self.beliefs: list[dict] = []

        async def write_workflow_belief(self, **kwargs) -> None:
            self.beliefs.append(kwargs)

    repository = CRGRepository()
    agent = module.RetroSynAgent(planner=Planner(), crg_repository=repository)

    await agent.process(
        {
            "project_id": "project-1",
            "run_id": "run-1",
            "smiles": "CCO",
            "max_routes": 1,
        }
    )

    assert len(repository.beliefs) == 1
    belief = repository.beliefs[0]
    assert belief["project_id"] == "project-1"
    assert belief["run_id"] == "run-1"
    assert belief["subject"] == "CCO"
    assert belief["predicate"] == "retrosyn_routes"
    assert belief["object_value"] == "1"
    assert belief["source_agent"] == "retrosyn_agent"
    assert belief["evidence_ids"] == ["route-1"]


@pytest.mark.asyncio
async def test_retrosyn_agent_writes_route_humu_embeddings_to_crg() -> None:
    module = _load_module(
        "retrosyn_agent_route_humu_embedding_test",
        ROOT / "agents/retrosyn_agent/src/retrosyn_agent/agent.py",
    )

    class Planner:
        async def find_routes(self, smiles: str, max_routes: int) -> list[dict]:
            return [
                {
                    "route_id": "route-1",
                    "target_smiles": smiles,
                    "steps": [_valid_retrosyn_step("CCO>>CC=O")],
                    "max_routes": max_routes,
                }
            ]

    class RouteEncoder:
        async def encode_route(self, route: dict) -> dict:
            assert route["route_id"] == "route-1"
            return {
                "humu_embedding": [1.0, 0.0],
                "curvature": 1.0,
            }

    class CRGRepository:
        def __init__(self) -> None:
            self.beliefs: list[dict] = []

        async def write_workflow_belief(self, **kwargs) -> None:
            self.beliefs.append(kwargs)

    repository = CRGRepository()
    agent = module.RetroSynAgent(
        planner=Planner(),
        route_encoder_client=RouteEncoder(),
        crg_repository=repository,
    )

    result = await agent.process(
        {
            "project_id": "project-1",
            "run_id": "run-1",
            "smiles": "CCO",
            "max_routes": 1,
        }
    )

    assert result["routes"][0]["humu_embedding"] == [1.0, 0.0]
    assert result["routes"][0]["humu_curvature"] == 1.0
    predicates = [belief["predicate"] for belief in repository.beliefs]
    assert predicates == ["retrosyn_routes", "route_humu_embedding"]
    route_embedding = json.loads(repository.beliefs[1]["object_value"])
    assert route_embedding == {
        "curvature": 1.0,
        "humu_embedding": [1.0, 0.0],
        "route_id": "route-1",
    }
    assert repository.beliefs[1]["evidence_ids"] == ["route-1"]


@pytest.mark.asyncio
async def test_retrosyn_agent_merges_injected_route_planner_ensemble() -> None:
    module = _load_module(
        "retrosyn_agent_route_ensemble_test",
        ROOT / "agents/retrosyn_agent/src/retrosyn_agent/agent.py",
    )

    class Planner:
        def __init__(self, route_id: str, score: float) -> None:
            self.route_id = route_id
            self.score = score
            self.calls: list[tuple[str, int]] = []

        async def find_routes(self, smiles: str, max_routes: int) -> list[dict]:
            self.calls.append((smiles, max_routes))
            return [
                {
                    "route_id": self.route_id,
                    "score": self.score,
                    "steps": [
                        _valid_retrosyn_step(
                            f"{smiles}>>{self.route_id}",
                            f"{self.route_id}-step-1",
                        )
                    ],
                }
            ]

    aizynth = Planner("route-aizynth", 0.4)
    rsgpt = Planner("route-rsgpt", 0.9)
    agent = module.RetroSynAgent(
        route_planners={"aizynth": aizynth, "rsgpt": rsgpt},
        crg_repository=None,
    )

    result = await agent.process({"smiles": "CCO", "max_routes": 2, "engine": "ensemble"})

    assert aizynth.calls == [("CCO", 2)]
    assert rsgpt.calls == [("CCO", 2)]
    assert [route["route_id"] for route in result["routes"]] == [
        "route-rsgpt",
        "route-aizynth",
    ]
    assert result["routes"][0]["source_engine"] == "rsgpt"
    assert result["layers"]["strategy"]["engine"] == "ensemble"
    assert result["layers"]["strategy"]["engines"] == ["aizynth", "rsgpt"]


@pytest.mark.asyncio
async def test_retrosyn_agent_runs_only_explicit_named_engine() -> None:
    module = _load_module(
        "retrosyn_agent_explicit_engine_test",
        ROOT / "agents/retrosyn_agent/src/retrosyn_agent/agent.py",
    )

    class Planner:
        def __init__(self, route_id: str) -> None:
            self.route_id = route_id
            self.calls = []

        async def find_routes(self, smiles: str, max_routes: int) -> list[dict]:
            self.calls.append((smiles, max_routes))
            return [
                {
                    "route_id": self.route_id,
                    "steps": [
                        _valid_retrosyn_step(
                            f"{smiles}>>{self.route_id}",
                            f"{self.route_id}-step-1",
                        )
                    ],
                }
            ]

    aizynth = Planner("route-aizynth")
    rsgpt = Planner("route-rsgpt")
    agent = module.RetroSynAgent(
        route_planners={"aizynth": aizynth, "rsgpt": rsgpt},
        crg_repository=None,
    )

    result = await agent.process(
        {
            "smiles": "CCO",
            "max_routes": 1,
            "engine": "rsgpt",
        }
    )

    assert aizynth.calls == []
    assert rsgpt.calls == [("CCO", 1)]
    assert [route["route_id"] for route in result["routes"]] == ["route-rsgpt"]
    assert result["layers"]["strategy"]["engine"] == "rsgpt"


@pytest.mark.asyncio
async def test_retrosyn_agent_rejects_missing_engine() -> None:
    module = _load_module(
        "retrosyn_agent_missing_engine_test",
        ROOT / "agents/retrosyn_agent/src/retrosyn_agent/agent.py",
    )

    class Planner:
        async def find_routes(self, _smiles: str, max_routes: int) -> list[dict]:
            raise AssertionError("planner must not run without an explicit engine")

    with pytest.raises(ValueError, match="engine"):
        await module.RetroSynAgent(
            route_planners={"rsgpt": Planner()},
            crg_repository=None,
        ).process({"smiles": "CCO", "max_routes": 1})


@pytest.mark.asyncio
async def test_retrosyn_grpc_health_uses_channel_readiness_without_planning() -> None:
    module = _load_module(
        "retrosyn_agent_channel_health_test",
        ROOT / "agents/retrosyn_agent/src/retrosyn_agent/agent.py",
    )

    class Channel:
        def __init__(self) -> None:
            self.calls = 0

        async def channel_ready(self) -> None:
            self.calls += 1

    class Stub:
        async def FindRoutes(self, *_args, **_kwargs):
            raise AssertionError("health check must not submit a retrosynthesis job")

    client = module.RetrosynGrpcClient.__new__(module.RetrosynGrpcClient)
    client.channel = Channel()
    client.stub = Stub()

    assert await client.health_check() == {"healthy": True}
    assert client.channel.calls == 1


@pytest.mark.asyncio
async def test_retrosyn_agent_keeps_route_planners_ahead_of_accessibility_scores() -> None:
    module = _load_module(
        "retrosyn_agent_accessibility_score_rank_test",
        ROOT / "agents/retrosyn_agent/src/retrosyn_agent/agent.py",
    )

    class Planner:
        def __init__(self, route: dict) -> None:
            self.route = route

        async def find_routes(self, smiles: str, max_routes: int) -> list[dict]:
            return [dict(self.route)]

    agent = module.RetroSynAgent(
        route_planners={
            "rascore": Planner(
                {
                    "route_id": "rascore-1",
                    "route_type": "retrosynthetic_accessibility_score",
                    "score": 0.99,
                    "steps": [],
                }
            ),
            "rsgpt": Planner(
                {
                    "route_id": "route-rsgpt",
                    "score": 0.5,
                    "steps": [
                        _valid_retrosyn_step(
                            "CCO>>route-rsgpt",
                            "route-rsgpt-step-1",
                        )
                    ],
                }
            ),
        },
        crg_repository=None,
    )

    result = await agent.process({"smiles": "CCO", "max_routes": 1, "engine": "ensemble"})

    assert [route["route_id"] for route in result["routes"]] == ["route-rsgpt"]
    assert [item["route_id"] for item in result["assessments"]] == ["rascore-1"]


@pytest.mark.asyncio
async def test_retrosyn_agent_rejects_route_without_executable_steps() -> None:
    module = _load_module(
        "retrosyn_agent_supply_ready_rank_test",
        ROOT / "agents/retrosyn_agent/src/retrosyn_agent/agent.py",
    )

    class Planner:
        def __init__(self, route: dict) -> None:
            self.route = route

        async def find_routes(self, smiles: str, max_routes: int) -> list[dict]:
            return [dict(self.route)]

    agent = module.RetroSynAgent(
        route_planners={
            "aizynth": Planner(
                {
                    "route_id": "aizynth-1",
                    "score": 1.0,
                    "steps": [],
                }
            ),
            "rsgpt": Planner(
                {
                    "route_id": "route-rsgpt",
                    "score": 0.1,
                    "steps": [_valid_retrosyn_step("CO.C>>CCO")],
                }
            ),
        },
        crg_repository=None,
    )

    with pytest.raises(ValueError, match="must contain non-empty steps"):
        await agent.process({"smiles": "CCO", "max_routes": 1, "engine": "ensemble"})


@pytest.mark.asyncio
async def test_retrosyn_agent_builds_planner_ensemble_from_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_module(
        "retrosyn_agent_env_ensemble_test",
        ROOT / "agents/retrosyn_agent/src/retrosyn_agent/agent.py",
    )
    runner = tmp_path / "retrosyn_runner.py"
    runner.write_text(
        "\n".join(
            [
                "import json",
                "import sys",
                "payload = json.load(sys.stdin)",
                "route_id = sys.argv[1]",
                "score = float(sys.argv[2])",
                "assert payload['smiles'] == 'CCO'",
                "assert payload['max_routes'] == 2",
                "json.dump(",
                "    {'routes': [{'route_id': route_id, 'score': score, 'steps': [{"
                "'step_id': route_id + '-step-1', 'reaction': 'CCO>>' + route_id,"
                " 'reaction_type': 'generic',"
                " 'reactants': [{'smiles': 'C', 'amount_mmol': 1.0}],"
                " 'conditions': {'temperature_C': 25.0, 'time_h': 2.0,"
                " 'source': 'test'},"
                " 'yield': 0.8,"
                " 'building_blocks': [{'smiles': 'C'}]}]}]},",
                "    sys.stdout,",
                ")",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv(
        "RETROSYN_PLANNER_COMMANDS_JSON",
        json.dumps(
            {
                "aizynth": f"{sys.executable} {runner} route-aizynth 0.4",
                "rsgpt": f"{sys.executable} {runner} route-rsgpt 0.9",
            }
        ),
    )
    agent = module.RetroSynAgent(crg_repository=None)

    result = await agent.process({"smiles": "CCO", "max_routes": 2, "engine": "ensemble"})

    assert [route["route_id"] for route in result["routes"]] == [
        "route-rsgpt",
        "route-aizynth",
    ]
    assert result["routes"][0]["source_engine"] == "rsgpt"
    assert result["layers"]["strategy"]["engine"] == "ensemble"
    assert result["layers"]["strategy"]["engines"] == ["aizynth", "rsgpt"]


@pytest.mark.asyncio
async def test_retrosyn_agent_builds_named_planner_ensemble_from_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_module(
        "retrosyn_agent_named_env_ensemble_test",
        ROOT / "agents/retrosyn_agent/src/retrosyn_agent/agent.py",
    )
    runner = tmp_path / "retrosyn_runner.py"
    runner.write_text(
        "\n".join(
            [
                "import json",
                "import sys",
                "payload = json.load(sys.stdin)",
                "route_id = sys.argv[1]",
                "score = float(sys.argv[2])",
                "assert payload['smiles'] == 'CCO'",
                "assert payload['max_routes'] == 3",
                "assert payload['engine'] in {'rascore', 'rsgpt', 'ualign', 'aizynth'}",
                "json.dump(",
                "    {'routes': [{'route_id': route_id, 'score': score, 'steps': [{"
                "'step_id': route_id + '-step-1', 'reaction': 'CCO>>' + route_id,"
                " 'reaction_type': 'generic',"
                " 'reactants': [{'smiles': 'C', 'amount_mmol': 1.0}],"
                " 'conditions': {'temperature_C': 25.0, 'time_h': 2.0,"
                " 'source': 'test'},"
                " 'yield': 0.8,"
                " 'building_blocks': [{'smiles': 'C'}]}]}]},",
                "    sys.stdout,",
                ")",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv("RETROSYN_PLANNER_COMMANDS_JSON", raising=False)
    monkeypatch.setenv("RASCORE_PLANNER_COMMAND", f"{sys.executable} {runner} route-rascore 0.7")
    monkeypatch.setenv("RSGPT_PLANNER_COMMAND", f"{sys.executable} {runner} route-rsgpt 0.9")
    monkeypatch.setenv("UALIGN_PLANNER_COMMAND", f"{sys.executable} {runner} route-ualign 0.8")
    monkeypatch.setenv("AIZYNTH_PLANNER_COMMAND", f"{sys.executable} {runner} route-aizynth 0.4")
    agent = module.RetroSynAgent(crg_repository=None)

    result = await agent.process({"smiles": "CCO", "max_routes": 3, "engine": "ensemble"})

    assert [route["route_id"] for route in result["routes"]] == [
        "route-rsgpt",
        "route-ualign",
        "route-rascore",
    ]
    assert result["routes"][0]["source_engine"] == "rsgpt"
    assert result["layers"]["strategy"]["engine"] == "ensemble"
    assert result["layers"]["strategy"]["engines"] == [
        "rascore",
        "rsgpt",
        "ualign",
        "aizynth",
    ]


@pytest.mark.asyncio
async def test_retrosyn_agent_runs_configured_json_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module(
        "retrosyn_agent_json_command_test",
        ROOT / "agents/retrosyn_agent/src/retrosyn_agent/agent.py",
    )
    command = (
        f"{sys.executable} -c "
        '"import json,sys; '
        "payload=json.load(sys.stdin); "
        "assert payload['smiles'] == 'CCO'; "
        "assert payload['max_routes'] == 1; "
        "print(json.dumps({'routes':[{'route_id':'route-command',"
        "'score':0.8,'steps':[{'step_id':'route-command-step-1',"
        "'reaction':'CCO>>CC=O','reaction_type':'generic',"
        "'reactants':[{'smiles':'C','amount_mmol':1.0}],"
        "'conditions':{'temperature_C':25.0,'time_h':2.0,'source':'test'},"
        "'yield':0.8,"
        "'building_blocks':[{'smiles':'C'}]}]}]}))\""
    )
    monkeypatch.setenv("RETROSYN_PLANNER_COMMAND", command)
    agent = module.RetroSynAgent(crg_repository=None)

    result = await agent.process({"smiles": "CCO", "max_routes": 1})

    assert result["routes"][0]["route_id"] == "route-command"
    assert result["routes"][0]["source_engine"] == "external_command"
    assert result["layers"]["strategy"]["engine"] == "ExternalCommandRetrosynPlanner"


@pytest.mark.asyncio
async def test_retrosyn_agent_planner_command_preflight_rejects_missing_executable() -> None:
    module = _load_module(
        "retrosyn_agent_missing_command_preflight_test",
        ROOT / "agents/retrosyn_agent/src/retrosyn_agent/agent.py",
    )
    planner = module.ExternalCommandRetrosynPlanner("missing-retrosyn-agent-planner --json")

    with pytest.raises(RuntimeError, match="not found"):
        await planner.find_routes("CCO", max_routes=1)


@pytest.mark.asyncio
async def test_retrosyn_agent_does_not_use_failed_validation_belief_as_control_flow() -> None:
    module = _load_module(
        "retrosyn_agent_crg_readback_test",
        ROOT / "agents/retrosyn_agent/src/retrosyn_agent/agent.py",
    )

    class Planner:
        def __init__(self) -> None:
            self.calls = []

        async def find_routes(self, smiles: str, max_routes: int) -> list[dict]:
            self.calls.append((smiles, max_routes))
            return [
                {
                    "route_id": "route-1",
                    "steps": [
                        {
                            "step_id": "step-1",
                            "reaction": "CO.C>>CCO",
                            "reaction_type": "coupling",
                            "reactants": [
                                {"smiles": "CO", "amount_mmol": 1.0},
                                {"smiles": "C", "amount_mmol": 1.2},
                            ],
                            "conditions": {
                                "temperature_C": 25.0,
                                "time_h": 2.0,
                                "source": "planner",
                            },
                            "yield": 0.8,
                            "building_blocks": [{"smiles": "CO"}, {"smiles": "C"}],
                        }
                    ],
                }
            ]

    class CRGRepository:
        def __init__(self) -> None:
            self.beliefs: list[dict] = []

        async def get_run_crg(self, run_id: str) -> dict:
            assert run_id == "run-1"
            return {
                "beliefs": [
                    {
                        "subject": "CCO",
                        "predicate": "validation_status",
                        "object_value": "failed",
                    }
                ]
            }

        async def write_workflow_belief(self, **kwargs) -> None:
            self.beliefs.append(kwargs)

    repository = CRGRepository()
    planner = Planner()
    agent = module.RetroSynAgent(planner=planner, crg_repository=repository)

    result = await agent.process(
        {
            "project_id": "project-1",
            "run_id": "run-1",
            "smiles": "CCO",
            "max_routes": 1,
        }
    )

    assert planner.calls == [("CCO", 1)]
    assert result["status"] == "planned"
    assert result["routes"][0]["route_id"] == "route-1"
    assert repository.beliefs[0]["predicate"] == "retrosyn_routes"
    assert repository.beliefs[0]["object_value"] == "1"
    assert repository.beliefs[0]["evidence_ids"] == ["route-1"]


@pytest.mark.asyncio
async def test_retrosyn_agent_does_not_use_route_count_belief_as_cache() -> None:
    module = _load_module(
        "retrosyn_agent_zero_routes_crg_readback_test",
        ROOT / "agents/retrosyn_agent/src/retrosyn_agent/agent.py",
    )

    class Planner:
        def __init__(self) -> None:
            self.calls = []

        async def find_routes(self, smiles: str, max_routes: int) -> list[dict]:
            self.calls.append((smiles, max_routes))
            return [
                {
                    "route_id": "route-1",
                    "steps": [
                        {
                            "step_id": "step-1",
                            "reaction": "CO.C>>CCO",
                            "reaction_type": "coupling",
                            "reactants": [
                                {"smiles": "CO", "amount_mmol": 1.0},
                                {"smiles": "C", "amount_mmol": 1.2},
                            ],
                            "conditions": {
                                "temperature_C": 25.0,
                                "time_h": 2.0,
                                "source": "planner",
                            },
                            "yield": 0.8,
                            "building_blocks": [{"smiles": "CO"}, {"smiles": "C"}],
                        }
                    ],
                }
            ]

    class CRGRepository:
        def __init__(self) -> None:
            self.beliefs: list[dict] = []

        async def get_run_crg(self, run_id: str) -> dict:
            assert run_id == "run-1"
            return {
                "beliefs": [
                    {
                        "subject": "CCO",
                        "predicate": "retrosyn_routes",
                        "object_value": "0",
                    }
                ]
            }

        async def write_workflow_belief(self, **kwargs) -> None:
            self.beliefs.append(kwargs)

    repository = CRGRepository()
    planner = Planner()
    agent = module.RetroSynAgent(planner=planner, crg_repository=repository)

    result = await agent.process(
        {
            "project_id": "project-1",
            "run_id": "run-1",
            "smiles": "CCO",
            "max_routes": 1,
        }
    )

    assert planner.calls == [("CCO", 1)]
    assert result["status"] == "planned"
    assert result["routes"][0]["route_id"] == "route-1"
    assert repository.beliefs[0]["predicate"] == "retrosyn_routes"
    assert repository.beliefs[0]["object_value"] == "1"


@pytest.mark.asyncio
async def test_orchestrator_agent_persists_workflow_status_belief() -> None:
    module = _load_module(
        "orchestrator_agent_crg_repository_test",
        ROOT / "agents/orchestrator/src/orchestrator/agent.py",
    )

    class CRGRepository:
        def __init__(self) -> None:
            self.beliefs: list[dict] = []

        async def write_workflow_belief(self, **kwargs) -> None:
            self.beliefs.append(kwargs)

    repository = CRGRepository()
    agent = module.OrchestratorAgent(crg_repository=repository)

    await agent.run_design_workflow(
        {
            "project_id": "project-1",
            "run_id": "run-1",
            "trace_id": "trace-1",
            "intent": "design a molecule",
            "workflow_scope": "state_only",
            "validation_passed": True,
            "max_refinements": 0,
        }
    )

    assert len(repository.beliefs) == 1
    belief = repository.beliefs[0]
    assert belief["project_id"] == "project-1"
    assert belief["run_id"] == "run-1"
    assert belief["subject"] == "project-1"
    assert belief["predicate"] == "workflow_status"
    assert belief["object_value"] == "completed"
    assert belief["source_agent"] == "orchestrator"
    assert belief["evidence_ids"] == ["PLANNING"]


@pytest.mark.asyncio
async def test_orchestrator_agent_does_not_treat_scalar_workflow_status_as_cached_state() -> None:
    module = _load_module(
        "orchestrator_agent_crg_readback_test",
        ROOT / "agents/orchestrator/src/orchestrator/agent.py",
    )

    class CRGRepository:
        def __init__(self) -> None:
            self.reads: list[str] = []
            self.beliefs: list[dict] = []

        async def get_run_crg(self, run_id: str) -> dict:
            self.reads.append(run_id)
            return {
                "beliefs": [
                    {
                        "subject": "project-1",
                        "predicate": "workflow_status",
                        "object": "completed",
                        "evidence_ids": ["nl2obj", "generate", "critic"],
                    }
                ],
                "edges": [],
            }

        async def write_workflow_belief(self, **kwargs) -> None:
            self.beliefs.append(kwargs)

    repository = CRGRepository()
    agent = module.OrchestratorAgent(crg_repository=repository)

    result = await agent.run_design_workflow(
        {
            "project_id": "project-1",
            "run_id": "run-1",
            "trace_id": "trace-1",
            "intent": "design a molecule",
            "workflow_scope": "state_only",
            "validation_passed": True,
            "max_refinements": 0,
        }
    )

    assert repository.reads == []
    assert result["status"] == "completed"
    assert result["current_stage"] == "PLANNING"
    assert result["history"] == ["PLANNING"]
    assert result.get("cached") is None
    assert repository.beliefs[0]["evidence_ids"] == result["history"]


def test_shared_crg_repository_factory_uses_neo4j_env() -> None:
    from mf_core.db.repositories import GraphRepository, build_shared_crg_repository_from_env

    created: list[dict] = []

    def driver_factory(uri: str, auth: tuple[str, str]):
        created.append({"uri": uri, "auth": auth})
        return object()

    repository = build_shared_crg_repository_from_env(
        env={
            "NEO4J_URI": "bolt://neo4j:7687",
            "NEO4J_USER": "neo4j",
            "NEO4J_PASSWORD": "secret",
        },
        driver_factory=driver_factory,
    )

    assert isinstance(repository, GraphRepository)
    assert created == [
        {
            "uri": "bolt://neo4j:7687",
            "auth": ("neo4j", "secret"),
        }
    ]


def test_shared_crg_repository_factory_returns_none_without_env() -> None:
    from mf_core.db.repositories import build_shared_crg_repository_from_env

    assert build_shared_crg_repository_from_env(env={}) is None


@pytest.mark.parametrize(
    ("module_name", "agent_path", "class_name", "kwargs"),
    [
        (
            "orchestrator_agent_shared_crg_factory_test",
            ROOT / "agents/orchestrator/src/orchestrator/agent.py",
            "OrchestratorAgent",
            {},
        ),
        (
            "nl2obj_agent_shared_crg_factory_test",
            ROOT / "agents/nl2obj/src/nl2obj/agent.py",
            "NL2ObjAgent",
            {},
        ),
        (
            "generator_coord_agent_shared_crg_factory_test",
            ROOT / "agents/generator_coord/src/generator_coord/agent.py",
            "GeneratorCoordAgent",
            {},
        ),
        (
            "validation_agent_shared_crg_factory_test",
            ROOT / "agents/validation_agent/src/validation_agent/agent.py",
            "ValidationAgent",
            {"oracles": {}},
        ),
        (
            "retrosyn_agent_shared_crg_factory_test",
            ROOT / "agents/retrosyn_agent/src/retrosyn_agent/agent.py",
            "RetroSynAgent",
            {"planner": object()},
        ),
        (
            "supply_agent_shared_crg_factory_test",
            ROOT / "agents/supply_agent/src/supply_agent/agent.py",
            "SupplyAgent",
            {},
        ),
        (
            "srb_agent_shared_crg_factory_test",
            ROOT / "agents/srb_agent/src/srb_agent/agent.py",
            "SRBAgent",
            {},
        ),
        (
            "critic_agent_shared_crg_factory_test",
            ROOT / "agents/critic_agent/src/critic_agent/agent.py",
            "ScientificCriticAgent",
            {},
        ),
    ],
)
def test_agents_default_to_shared_crg_repository_factory(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
    agent_path: Path,
    class_name: str,
    kwargs: dict,
) -> None:
    module = _load_module(module_name, agent_path)
    sentinel = object()

    monkeypatch.setattr(module, "build_shared_crg_repository_from_env", lambda: sentinel)

    agent = getattr(module, class_name)(**kwargs)

    assert agent.crg_repository is sentinel


@pytest.mark.asyncio
async def test_base_agent_reads_shared_crg_repository() -> None:
    from mf_agents.base.agent import BaseAgent

    class CRGRepository:
        def __init__(self) -> None:
            self.calls: list[str] = []

        async def get_run_crg(self, run_id: str) -> dict:
            self.calls.append(run_id)
            return {"run_id": run_id, "beliefs": [{"id": "belief-1"}], "edges": []}

    class Agent(BaseAgent):
        async def handle_message(self, subject: str, payload: bytes, reply_to: str = "") -> None:
            return None

    repository = CRGRepository()
    agent = Agent("test_agent")
    agent.crg_repository = repository

    crg = await agent.read_shared_crg("run-1")

    assert crg["run_id"] == "run-1"
    assert crg["beliefs"] == [{"id": "belief-1"}]
    assert repository.calls == ["run-1"]


@pytest.mark.asyncio
async def test_base_agent_publishes_signed_agent_message_envelope(
    agent_message_hmac_secret,
) -> None:
    from mf_agents.base.agent import BaseAgent
    from mf_core.proto_gen.moleculeforge.v1.agent.message_pb2 import AgentMessage

    class Bus:
        def __init__(self) -> None:
            self.published: list[tuple[str, bytes]] = []

        async def publish(self, subject: str, payload: bytes) -> None:
            self.published.append((subject, payload))

    class Agent(BaseAgent):
        async def handle_message(self, subject: str, payload: bytes, reply_to: str = "") -> None:
            return None

    bus = Bus()
    agent = Agent("generator_coord", message_bus=bus)

    await agent.publish_agent_message(
        "mf.agent.critic",
        recipient="critic_agent",
        message_type="request",
        payload=b'{"smiles":"CCO"}',
        payload_type_url="type.googleapis.com/moleculeforge.v1.agent.ValidationResult",
        trace_id="trace-1",
        message_id="message-1",
        lineage={"parent_trace": "trace-0"},
        ttl=4,
    )

    assert len(bus.published) == 1
    assert bus.published[0][0] == "mf.agent.critic"

    envelope = AgentMessage()
    envelope.ParseFromString(bus.published[0][1])

    assert envelope.trace_id == "trace-1"
    assert envelope.message_id == "message-1"
    assert envelope.sender == "generator_coord"
    assert envelope.recipient == "critic_agent"
    assert envelope.message_type == "request"
    assert envelope.payload == b'{"smiles":"CCO"}'
    assert (
        envelope.payload_type_url == "type.googleapis.com/moleculeforge.v1.agent.ValidationResult"
    )
    assert envelope.lineage["parent_trace"] == "trace-0"
    assert envelope.ttl == 4
    assert envelope.signature
    assert agent.verify_agent_message(envelope) is True

    envelope.payload = b'{"smiles":"CCN"}'
    assert agent.verify_agent_message(envelope) is False


@pytest.mark.asyncio
async def test_base_agent_generates_uuidv7_message_id_by_default(
    agent_message_hmac_secret,
) -> None:
    from mf_agents.base.agent import BaseAgent

    class Agent(BaseAgent):
        async def handle_message(self, subject: str, payload: bytes, reply_to: str = "") -> None:
            return None

    agent = Agent("generator_coord")

    envelope = await agent.publish_agent_message(
        "mf.agent.validation",
        recipient="validation_agent",
        message_type="event",
        payload=b'{"smiles":"CCO"}',
        payload_type_url="type.googleapis.com/moleculeforge.v1.agent.MoleculeCandidate",
    )

    assert uuid.UUID(envelope.message_id).version == 7


@pytest.mark.asyncio
async def test_base_agent_uses_sigstore_commands_for_agent_message_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mf_agents.base.agent import BaseAgent
    from mf_core.proto_gen.moleculeforge.v1.agent.message_pb2 import AgentMessage

    sign_command = (
        f'{sys.executable} -c "import json,sys;'
        "req=json.load(sys.stdin);"
        "sig='agent-sig-'+req['payload_hash'][:8];"
        "print(json.dumps({'signature':sig,'signature_type':'sigstore_rekor',"
        "'rekor_entry':{'uuid':'agent-rekor'}}))\""
    )
    verify_command = (
        f'{sys.executable} -c "import json,sys;'
        "req=json.load(sys.stdin);"
        "expected='agent-sig-'+req['payload_hash'][:8];"
        "print(json.dumps({'signature_valid':req['signature']==expected}))\""
    )
    monkeypatch.setenv("SIGSTORE_SIGN_COMMAND", sign_command)
    monkeypatch.setenv("SIGSTORE_VERIFY_COMMAND", verify_command)

    class Bus:
        def __init__(self) -> None:
            self.published: list[tuple[str, bytes]] = []

        async def publish(self, subject: str, payload: bytes) -> None:
            self.published.append((subject, payload))

    class Agent(BaseAgent):
        async def handle_message(self, subject: str, payload: bytes, reply_to: str = "") -> None:
            return None

    agent = Agent("validation_agent", message_bus=Bus())

    envelope = await agent.publish_agent_message(
        "mf.agent.critic",
        recipient="critic_agent",
        message_type="event",
        payload=b'{"validation":"passed"}',
        payload_type_url="type.googleapis.com/moleculeforge.v1.agent.ValidationResult",
        trace_id="trace-sigstore",
        message_id="message-sigstore",
    )

    assert envelope.signature.startswith(b"agent-sig-")
    assert agent.verify_agent_message(envelope) is True

    delivered = AgentMessage()
    delivered.ParseFromString(agent.message_bus.published[0][1])
    assert delivered.signature == envelope.signature
    assert agent.verify_agent_message(delivered.SerializeToString()) is True

    delivered.recipient = "supply_agent"
    assert agent.verify_agent_message(delivered) is False


@pytest.mark.asyncio
async def test_base_agent_passes_identity_token_to_sigstore_sign_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mf_agents.base.agent import BaseAgent

    sign_command = (
        f'{sys.executable} -c "import json,sys;'
        "req=json.load(sys.stdin);"
        "assert req['identity_token']=='oidc-token';"
        "print(json.dumps({'signature':'identity-token-sig'}))\""
    )
    monkeypatch.setenv("SIGSTORE_SIGN_COMMAND", sign_command)
    monkeypatch.setenv("SIGSTORE_IDENTITY_TOKEN", "oidc-token")

    class Agent(BaseAgent):
        async def handle_message(self, subject: str, payload: bytes, reply_to: str = "") -> None:
            return None

    envelope = await Agent("validation_agent").publish_agent_message(
        "mf.agent.critic",
        recipient="critic_agent",
        message_type="event",
        payload=b'{"validation":"passed"}',
        payload_type_url="type.googleapis.com/moleculeforge.v1.agent.ValidationResult",
        trace_id="trace-identity-token",
        message_id="message-identity-token",
    )

    assert envelope.signature == b"identity-token-sig"


@pytest.mark.asyncio
async def test_base_agent_passes_message_identity_to_sigstore_verify_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mf_agents.base.agent import BaseAgent

    sign_command = (
        f'{sys.executable} -c "import json,sys;'
        "req=json.load(sys.stdin);"
        "sig='agent-sig-'+req['payload_hash'][:8];"
        "print(json.dumps({'signature':sig}))\""
    )
    verify_command = (
        f'{sys.executable} -c "import json,sys;'
        "req=json.load(sys.stdin);"
        "assert req['sender']=='generator_coord';"
        "assert req['recipient']=='validation_agent';"
        "assert req['message_type']=='event';"
        "assert req['expected_identity']=='generator_coord';"
        "print(json.dumps({'valid':True}))\""
    )
    monkeypatch.setenv("SIGSTORE_SIGN_COMMAND", sign_command)
    monkeypatch.setenv("SIGSTORE_VERIFY_COMMAND", verify_command)

    class Agent(BaseAgent):
        async def handle_message(self, subject: str, payload: bytes, reply_to: str = "") -> None:
            return None

    envelope = await Agent("generator_coord").publish_agent_message(
        "mf.agent.validation",
        recipient="validation_agent",
        message_type="event",
        payload=b'{"smiles":"CCO"}',
        payload_type_url="type.googleapis.com/moleculeforge.v1.agent.MoleculeCandidate",
        trace_id="trace-verify-identity",
        message_id="message-verify-identity",
    )

    assert Agent("validation_agent").verify_agent_message(envelope) is True


@pytest.mark.asyncio
async def test_base_agent_sigstore_sign_command_preflight_rejects_missing_executable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mf_agents.base.agent import BaseAgent

    monkeypatch.setenv("SIGSTORE_SIGN_COMMAND", "missing-agent-sigstore-sign --json")

    class Agent(BaseAgent):
        async def handle_message(self, subject: str, payload: bytes, reply_to: str = "") -> None:
            return None

    with pytest.raises(RuntimeError, match="not found"):
        await Agent("validation_agent").publish_agent_message(
            "mf.agent.critic",
            recipient="critic_agent",
            message_type="event",
            payload=b'{"validation":"passed"}',
            payload_type_url="type.googleapis.com/moleculeforge.v1.agent.ValidationResult",
            trace_id="trace-missing-agent-sign",
            message_id="message-missing-agent-sign",
        )


def test_base_agent_sigstore_verify_command_preflight_rejects_missing_executable(
    monkeypatch: pytest.MonkeyPatch,
    agent_message_hmac_secret,
) -> None:
    from mf_agents.base.agent import BaseAgent
    from mf_core.proto_gen.moleculeforge.v1.agent.message_pb2 import AgentMessage

    monkeypatch.setenv("SIGSTORE_VERIFY_COMMAND", "missing-agent-sigstore-verify --json")

    class Agent(BaseAgent):
        async def handle_message(self, subject: str, payload: bytes, reply_to: str = "") -> None:
            return None

    sender = Agent("generator_coord")
    envelope = AgentMessage(
        trace_id="trace-missing-agent-verify",
        message_id="message-missing-agent-verify",
        sender="generator_coord",
        recipient="validation_agent",
        message_type="event",
        payload=b'{"smiles":"CCO"}',
        payload_type_url="type.googleapis.com/moleculeforge.v1.agent.MoleculeCandidate",
        ttl=4,
    )
    envelope.signature = sender._sign_agent_message(envelope)

    with pytest.raises(RuntimeError, match="not found"):
        Agent("validation_agent").verify_agent_message(envelope)


def test_agent_grpc_clients_create_default_event_loop_when_missing() -> None:
    import asyncio

    modules = {
        "nl2obj": _load_module(
            "nl2obj_grpc_client_event_loop_test",
            ROOT / "agents/nl2obj/src/nl2obj/agent.py",
        ),
        "supply": _load_module(
            "supply_grpc_client_event_loop_test",
            ROOT / "agents/supply_agent/src/supply_agent/agent.py",
        ),
        "generator_coord": _load_module(
            "generator_coord_grpc_client_event_loop_test",
            ROOT / "agents/generator_coord/src/generator_coord/agent.py",
        ),
        "retrosyn": _load_module(
            "retrosyn_grpc_client_event_loop_test",
            ROOT / "agents/retrosyn_agent/src/retrosyn_agent/agent.py",
        ),
        "validation": _load_module(
            "validation_grpc_client_event_loop_test",
            ROOT / "agents/validation_agent/src/validation_agent/agent.py",
        ),
    }
    asyncio.set_event_loop(None)
    channels = []

    try:
        nl2obj_client = modules["nl2obj"].CIGCompilerGrpcClient("localhost:1")
        channels.append(nl2obj_client.channel)
        supply_client = modules["supply"].SupplyOracleGrpcClient("localhost:1")
        channels.append(supply_client.channel)
        generator_client = modules["generator_coord"].GeneratorGrpcClient("localhost:1")
        channels.append(generator_client.channel)
        route_encoder_client = modules["retrosyn"].HUMURouteEncoderGrpcClient("localhost:1")
        channels.append(route_encoder_client.channel)
        oracle_client = modules["validation"].OracleGrpcClient("localhost:1", 1, "test")
        oracle_client._stub()
        channels.append(oracle_client.channel)
        loop = asyncio.get_event_loop()
        for channel in channels:
            loop.run_until_complete(channel.close())
    finally:
        try:
            cleanup_loop = asyncio.get_event_loop()
        except RuntimeError:
            cleanup_loop = None
        if cleanup_loop is not None and not cleanup_loop.is_closed():
            cleanup_loop.close()
        asyncio.set_event_loop(asyncio.new_event_loop())


@pytest.mark.asyncio
async def test_base_agent_encodes_jsonld_payload_before_signing(
    agent_message_hmac_secret,
) -> None:
    from mf_agents.base.agent import BaseAgent
    from mf_core.proto_gen.moleculeforge.v1.agent.message_pb2 import AgentMessage

    class Bus:
        def __init__(self) -> None:
            self.published: list[tuple[str, bytes]] = []

        async def publish(self, subject: str, payload: bytes) -> None:
            self.published.append((subject, payload))

    class Agent(BaseAgent):
        async def handle_message(self, subject: str, payload: bytes, reply_to: str = "") -> None:
            return None

    agent = Agent("generator_coord", message_bus=Bus())

    await agent.publish_agent_message(
        "mf.agent.validation",
        recipient="validation_agent",
        message_type="event",
        payload={"smiles": "CCO", "score": 0.8},
        payload_type_url="type.googleapis.com/moleculeforge.v1.agent.MoleculeCandidate",
        trace_id="trace-jsonld",
        message_id="message-jsonld",
        jsonld_context={"mf": "https://moleculeforge.ai/context#"},
        jsonld_type="mf:MoleculeCandidate",
        jsonld_id="urn:mf:molecule:CCO",
    )

    envelope = AgentMessage()
    envelope.ParseFromString(agent.message_bus.published[0][1])
    payload = json.loads(envelope.payload.decode("utf-8"))

    assert payload == {
        "@context": {"mf": "https://moleculeforge.ai/context#"},
        "@id": "urn:mf:molecule:CCO",
        "@type": "mf:MoleculeCandidate",
        "score": 0.8,
        "smiles": "CCO",
    }
    assert agent.verify_agent_message(envelope) is True

    payload["score"] = 0.1
    envelope.payload = json.dumps(payload, sort_keys=True).encode("utf-8")
    assert agent.verify_agent_message(envelope) is False


@pytest.mark.asyncio
async def test_base_agent_verifies_messages_with_shared_hmac_secret(
    agent_message_hmac_secret,
) -> None:
    from mf_agents.base.agent import BaseAgent
    from mf_core.proto_gen.moleculeforge.v1.agent.message_pb2 import AgentMessage

    class Bus:
        def __init__(self) -> None:
            self.published: list[tuple[str, bytes]] = []

        async def publish(self, subject: str, payload: bytes) -> None:
            self.published.append((subject, payload))

    class Agent(BaseAgent):
        async def handle_message(self, subject: str, payload: bytes, reply_to: str = "") -> None:
            return None

    bus = Bus()
    sender = Agent("generator_coord", message_bus=bus)
    receiver = Agent("validation_agent")

    await sender.publish_agent_message(
        "mf.agent.validation",
        recipient="validation_agent",
        message_type="event",
        payload=b'{"smiles":"CCO"}',
        payload_type_url="type.googleapis.com/moleculeforge.v1.agent.MoleculeCandidate",
        trace_id="trace-cross-agent",
        message_id="message-cross-agent",
    )

    envelope = AgentMessage()
    envelope.ParseFromString(bus.published[0][1])

    assert sender.verify_agent_message(envelope) is True
    assert receiver.verify_agent_message(envelope) is True

    envelope.sender = "critic_agent"
    assert receiver.verify_agent_message(envelope) is False


@pytest.mark.asyncio
async def test_base_agent_start_dispatches_verified_agent_message_payload(
    agent_message_hmac_secret,
) -> None:
    from mf_agents.base.agent import BaseAgent

    class Bus:
        def __init__(self) -> None:
            self.callbacks: dict[str, object] = {}
            self.published: list[tuple[str, bytes]] = []

        async def subscribe(self, subject: str, cb) -> None:
            self.callbacks[subject] = cb

        async def publish(self, subject: str, payload: bytes) -> None:
            self.published.append((subject, payload))

    class Receiver(BaseAgent):
        def __init__(self, message_bus) -> None:
            super().__init__("validation_agent", message_bus=message_bus)
            self._subscription_subjects = ["mf.agent.validation"]
            self.received: list[tuple[str, bytes, str]] = []

        async def handle_message(self, subject: str, payload: bytes, reply_to: str = "") -> None:
            self.received.append((subject, payload, reply_to))

    class Sender(BaseAgent):
        async def handle_message(self, subject: str, payload: bytes, reply_to: str = "") -> None:
            return None

    bus = Bus()
    receiver = Receiver(bus)
    sender = Sender("generator_coord", message_bus=bus)

    await receiver.start()
    envelope = await sender.publish_agent_message(
        "mf.agent.validation",
        recipient="validation_agent",
        message_type="event",
        payload={"smiles": "CCO"},
        payload_type_url="type.googleapis.com/moleculeforge.v1.agent.MoleculeCandidate",
        trace_id="trace-dispatch",
        message_id="message-dispatch",
        reply_to="mf.reply.validation",
        jsonld_context={"mf": "https://moleculeforge.ai/context#"},
        jsonld_type="mf:MoleculeCandidate",
    )

    await bus.callbacks["mf.agent.validation"](
        "mf.agent.validation",
        envelope.SerializeToString(),
        "",
    )

    assert receiver.received == [
        (
            "mf.agent.validation",
            b'{"@context":{"mf":"https://moleculeforge.ai/context#"},'
            b'"@type":"mf:MoleculeCandidate","smiles":"CCO"}',
            "mf.reply.validation",
        )
    ]

    envelope.payload = b'{"smiles":"CCN"}'
    with pytest.raises(RuntimeError, match="signature verification failed"):
        await bus.callbacks["mf.agent.validation"](
            "mf.agent.validation",
            envelope.SerializeToString(),
            "",
        )


@pytest.mark.asyncio
async def test_base_agent_start_preserves_raw_payload_dispatch() -> None:
    from mf_agents.base.agent import BaseAgent

    class Bus:
        def __init__(self) -> None:
            self.callback = None

        async def subscribe(self, subject: str, cb) -> None:
            self.callback = cb

    class Agent(BaseAgent):
        def __init__(self, message_bus) -> None:
            super().__init__("legacy_agent", message_bus=message_bus)
            self._subscription_subjects = ["legacy.subject"]
            self.received: list[tuple[str, bytes, str]] = []

        async def handle_message(self, subject: str, payload: bytes, reply_to: str = "") -> None:
            self.received.append((subject, payload, reply_to))

    bus = Bus()
    agent = Agent(bus)

    await agent.start()
    await bus.callback("legacy.subject", b'{"raw":true}', "legacy.reply")

    assert agent.received == [("legacy.subject", b'{"raw":true}', "legacy.reply")]


@pytest.mark.asyncio
async def test_base_agent_rejects_expired_agent_message_ttl(
    agent_message_hmac_secret,
) -> None:
    from mf_agents.base.agent import BaseAgent

    class Bus:
        def __init__(self) -> None:
            self.callbacks: dict[str, object] = {}
            self.published: list[tuple[str, bytes]] = []

        async def subscribe(self, subject: str, cb) -> None:
            self.callbacks[subject] = cb

        async def publish(self, subject: str, payload: bytes) -> None:
            self.published.append((subject, payload))

    class Receiver(BaseAgent):
        def __init__(self, message_bus) -> None:
            super().__init__("validation_agent", message_bus=message_bus)
            self._subscription_subjects = ["mf.agent.validation"]
            self.received: list[bytes] = []

        async def handle_message(self, subject: str, payload: bytes, reply_to: str = "") -> None:
            self.received.append(payload)

    class Sender(BaseAgent):
        async def handle_message(self, subject: str, payload: bytes, reply_to: str = "") -> None:
            return None

    bus = Bus()
    receiver = Receiver(bus)
    sender = Sender("generator_coord", message_bus=bus)

    await receiver.start()
    envelope = await sender.publish_agent_message(
        "mf.agent.validation",
        recipient="validation_agent",
        message_type="event",
        payload=b'{"smiles":"CCO"}',
        payload_type_url="type.googleapis.com/moleculeforge.v1.agent.MoleculeCandidate",
        trace_id="trace-ttl",
        message_id="message-ttl",
        ttl=0,
    )

    with pytest.raises(RuntimeError, match="ttl expired"):
        await bus.callbacks["mf.agent.validation"](
            "mf.agent.validation",
            envelope.SerializeToString(),
            "",
        )

    assert receiver.received == []


@pytest.mark.asyncio
async def test_base_agent_rejects_invalid_agent_message_type_before_publish() -> None:
    from mf_agents.base.agent import BaseAgent

    class Agent(BaseAgent):
        async def handle_message(self, subject: str, payload: bytes, reply_to: str = "") -> None:
            return None

    agent = Agent("generator_coord")

    with pytest.raises(ValueError, match="agent message_type must be one of"):
        await agent.publish_agent_message(
            "mf.agent.validation",
            recipient="validation_agent",
            message_type="notify",
            payload=b'{"smiles":"CCO"}',
            payload_type_url="type.googleapis.com/moleculeforge.v1.agent.MoleculeCandidate",
        )


@pytest.mark.asyncio
async def test_base_agent_rejects_missing_payload_type_url_before_publish() -> None:
    from mf_agents.base.agent import BaseAgent

    class Agent(BaseAgent):
        async def handle_message(self, subject: str, payload: bytes, reply_to: str = "") -> None:
            return None

    agent = Agent("generator_coord")

    with pytest.raises(ValueError, match="agent payload_type_url is required"):
        await agent.publish_agent_message(
            "mf.agent.validation",
            recipient="validation_agent",
            message_type="event",
            payload=b'{"smiles":"CCO"}',
        )


@pytest.mark.asyncio
async def test_base_agent_rejects_missing_recipient_before_publish() -> None:
    from mf_agents.base.agent import BaseAgent

    class Agent(BaseAgent):
        async def handle_message(self, subject: str, payload: bytes, reply_to: str = "") -> None:
            return None

    agent = Agent("generator_coord")

    with pytest.raises(ValueError, match="agent recipient is required"):
        await agent.publish_agent_message(
            "mf.agent.validation",
            recipient="",
            message_type="event",
            payload=b'{"smiles":"CCO"}',
            payload_type_url="type.googleapis.com/moleculeforge.v1.agent.MoleculeCandidate",
        )


@pytest.mark.asyncio
async def test_base_agent_rejects_invalid_received_agent_message_type(
    agent_message_hmac_secret,
) -> None:
    from mf_agents.base.agent import BaseAgent
    from mf_core.proto_gen.moleculeforge.v1.agent.message_pb2 import AgentMessage

    class Receiver(BaseAgent):
        def __init__(self) -> None:
            super().__init__("validation_agent")
            self.received: list[bytes] = []

        async def handle_message(self, subject: str, payload: bytes, reply_to: str = "") -> None:
            self.received.append(payload)

    class Sender(BaseAgent):
        async def handle_message(self, subject: str, payload: bytes, reply_to: str = "") -> None:
            return None

    sender = Sender("generator_coord")
    receiver = Receiver()
    envelope = AgentMessage(
        trace_id="trace-invalid-type",
        message_id="message-invalid-type",
        sender="generator_coord",
        recipient="validation_agent",
        message_type="notify",
        payload=b'{"smiles":"CCO"}',
        payload_type_url="type.googleapis.com/moleculeforge.v1.agent.MoleculeCandidate",
        ttl=4,
    )
    envelope.signature = sender._sign_agent_message(envelope)

    with pytest.raises(ValueError, match="agent message_type must be one of"):
        await receiver.handle_bus_message(
            "mf.agent.validation",
            envelope.SerializeToString(),
            "",
        )

    assert receiver.received == []


@pytest.mark.asyncio
async def test_base_agent_rejects_missing_received_recipient(
    agent_message_hmac_secret,
) -> None:
    from mf_agents.base.agent import BaseAgent
    from mf_core.proto_gen.moleculeforge.v1.agent.message_pb2 import AgentMessage

    class Receiver(BaseAgent):
        def __init__(self) -> None:
            super().__init__("validation_agent")
            self.received: list[bytes] = []

        async def handle_message(self, subject: str, payload: bytes, reply_to: str = "") -> None:
            self.received.append(payload)

    class Sender(BaseAgent):
        async def handle_message(self, subject: str, payload: bytes, reply_to: str = "") -> None:
            return None

    sender = Sender("generator_coord")
    receiver = Receiver()
    envelope = AgentMessage(
        trace_id="trace-missing-recipient",
        message_id="message-missing-recipient",
        sender="generator_coord",
        message_type="event",
        payload=b'{"smiles":"CCO"}',
        payload_type_url="type.googleapis.com/moleculeforge.v1.agent.MoleculeCandidate",
        ttl=4,
    )
    envelope.signature = sender._sign_agent_message(envelope)

    with pytest.raises(ValueError, match="agent recipient is required"):
        await receiver.handle_bus_message(
            "mf.agent.validation",
            envelope.SerializeToString(),
            "",
        )

    assert receiver.received == []


@pytest.mark.asyncio
async def test_base_agent_rejects_missing_received_payload_type_url(
    agent_message_hmac_secret,
) -> None:
    from mf_agents.base.agent import BaseAgent
    from mf_core.proto_gen.moleculeforge.v1.agent.message_pb2 import AgentMessage

    class Receiver(BaseAgent):
        def __init__(self) -> None:
            super().__init__("validation_agent")
            self.received: list[bytes] = []

        async def handle_message(self, subject: str, payload: bytes, reply_to: str = "") -> None:
            self.received.append(payload)

    class Sender(BaseAgent):
        async def handle_message(self, subject: str, payload: bytes, reply_to: str = "") -> None:
            return None

    sender = Sender("generator_coord")
    receiver = Receiver()
    envelope = AgentMessage(
        trace_id="trace-missing-type-url",
        message_id="message-missing-type-url",
        sender="generator_coord",
        recipient="validation_agent",
        message_type="event",
        payload=b'{"smiles":"CCO"}',
        ttl=4,
    )
    envelope.signature = sender._sign_agent_message(envelope)

    with pytest.raises(ValueError, match="agent payload_type_url is required"):
        await receiver.handle_bus_message(
            "mf.agent.validation",
            envelope.SerializeToString(),
            "",
        )

    assert receiver.received == []


async def _oracle_grpc_call(module, servicer, request, method: str = "Evaluate"):
    server = grpc.aio.server()
    module.oracle_pb2_grpc.add_OracleServiceServicer_to_server(servicer, server)
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    channel = grpc.aio.insecure_channel(f"127.0.0.1:{port}")
    try:
        stub = module.oracle_pb2_grpc.OracleServiceStub(channel)
        return await getattr(stub, method)(request)
    finally:
        await channel.close()
        await server.stop(None)


async def _supply_grpc_call(module, servicer, request, method: str):
    server = grpc.aio.server()
    module.supply_pb2_grpc.add_SupplyOracleServiceServicer_to_server(servicer, server)
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    channel = grpc.aio.insecure_channel(f"127.0.0.1:{port}")
    try:
        stub = module.supply_pb2_grpc.SupplyOracleServiceStub(channel)
        return await getattr(stub, method)(request)
    finally:
        await channel.close()
        await server.stop(None)


@pytest.mark.asyncio
async def test_admet_oracle_aio_contract_is_strict_and_does_not_zero_fill_uncertainty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mf_core.proto_gen.moleculeforge.v1.oracle import oracle_pb2

    module = _load_module(
        "admet_strict_oracle_contract_test",
        ROOT / "services/admet-svc/src/admet_svc/main.py",
    )
    monkeypatch.setattr(module, "_artifact_status_objects", lambda: [])

    class ADMETService:
        async def Screen(self, request, context):
            return SimpleNamespace(
                result={
                    request.smiles: {
                        "clearance": {"value": 1.5},
                    }
                }
            )

    request = oracle_pb2.OracleBatchRequest(
        project_id="project-1",
        request_id="request-1",
        molecule_smiles=["CCO"],
        requested_properties=["clearance"],
        level=oracle_pb2.L1_ML_SURROGATE,
        return_uncertainty=True,
    )
    response = await _oracle_grpc_call(
        module,
        module.ADMETOracleServicer(service=ADMETService()),
        request,
        "PredictWithUncertainty",
    )

    assert response.batch_id == "request-1"
    assert [item.molecule_smiles for item in response.evaluations] == ["CCO"]
    evaluation = response.evaluations[0]
    assert evaluation.level == oracle_pb2.L1_ML_SURROGATE
    assert evaluation.outcome == oracle_pb2.ORACLE_OUTCOME_PASS
    assert evaluation.success is True
    assert evaluation.scores == {"clearance": 1.5}
    assert evaluation.uncertainties == {}
    assert len(evaluation.metrics) == 1
    assert evaluation.metrics[0].property == "clearance"
    assert evaluation.metrics[0].HasField("uncertainty") is False
    assert evaluation.evidence_id == "request-1:admet_ai:0"
    assert evaluation.oracle_version == ""
    assert evaluation.model_version == ""
    assert evaluation.artifact_refs == []


@pytest.mark.asyncio
async def test_admet_oracle_missing_metric_and_computation_failure_are_error_outcomes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mf_core.proto_gen.moleculeforge.v1.oracle import oracle_pb2

    module = _load_module(
        "admet_oracle_error_outcome_test",
        ROOT / "services/admet-svc/src/admet_svc/main.py",
    )
    monkeypatch.setattr(module, "_status_objects", lambda: [])

    class ADMETService:
        async def Predict(self, request, context):
            if request.smiles == "CCN":
                raise RuntimeError("model execution failed")
            return SimpleNamespace(predictions={"qed": 0.8}, elapsed_ms=4)

    response = await _oracle_grpc_call(
        module,
        module.ADMETOracleServicer(service=ADMETService()),
        oracle_pb2.OracleBatchRequest(
            project_id="project-1",
            request_id="request-1",
            molecule_smiles=["CCO", "CCN"],
            requested_properties=["clearance"],
            level=oracle_pb2.L1_ML_SURROGATE,
        ),
    )

    assert [item.molecule_smiles for item in response.evaluations] == ["CCO", "CCN"]
    assert [item.outcome for item in response.evaluations] == [
        oracle_pb2.ORACLE_OUTCOME_ERROR,
        oracle_pb2.ORACLE_OUTCOME_ERROR,
    ]
    assert [item.error_code for item in response.evaluations] == [
        "MISSING_METRIC",
        "COMPUTATION_ERROR",
    ]
    assert all(not item.scores and not item.metrics for item in response.evaluations)


@pytest.mark.asyncio
async def test_boltz_oracle_aio_contract_uses_request_protein_and_fixed_l1() -> None:
    from mf_core.proto_gen.moleculeforge.v1.oracle import boltz2_pb2, oracle_pb2

    module = _load_module(
        "boltz_strict_oracle_contract_test",
        ROOT / "services/boltz2-svc/src/boltz2_svc/main.py",
    )

    class BoltzService:
        def __init__(self) -> None:
            self.request = None

        async def PredictAffinity(self, request, context):
            self.request = request
            return boltz2_pb2.Boltz2BatchResponse(
                protein_pdb_id=request.protein_pdb_id,
                affinities=[
                    boltz2_pb2.Boltz2BindingAffinity(
                        protein_pdb_id=request.protein_pdb_id,
                        ligand_smiles=smiles,
                        delta_g_kcal_mol=-8.0 - index,
                        uncertainty=0.2,
                        ki_nm=12.0,
                        ensemble_size=request.ensemble_size,
                        per_member_dg=[
                            -8.0 - index,
                            -8.1 - index,
                        ],
                    )
                    for index, smiles in enumerate(request.ligand_smiles)
                ],
                elapsed_ms=21,
            )

    service = BoltzService()
    response = await _oracle_grpc_call(
        module,
        module.Boltz2OracleServicer(service=service),
        oracle_pb2.OracleBatchRequest(
            project_id="project-1",
            request_id="request-1",
            molecule_smiles=["CCO", "CCN"],
            requested_properties=["affinity"],
            level=oracle_pb2.L1_ML_SURROGATE,
            protein_pdb_id="6OIM",
            oracle_parameters={"ensemble_size": "2"},
        ),
    )

    assert service.request.protein_pdb_id == "6OIM"
    assert service.request.ensemble_size == 2
    assert [item.molecule_smiles for item in response.evaluations] == ["CCO", "CCN"]
    assert all(item.level == oracle_pb2.L1_ML_SURROGATE for item in response.evaluations)
    assert all(item.outcome == oracle_pb2.ORACLE_OUTCOME_PASS for item in response.evaluations)
    assert [item.metrics[0].property for item in response.evaluations] == [
        "affinity",
        "affinity",
    ]


@pytest.mark.asyncio
async def test_dock_oracle_aio_contract_uses_receptor_and_parameter_engine() -> None:
    from mf_core.proto_gen.moleculeforge.v1.oracle import oracle_pb2

    module = _load_module(
        "dock_strict_oracle_contract_test",
        ROOT / "services/dock-svc/src/dock_svc/main.py",
    )

    class DockService:
        def __init__(self) -> None:
            self.requests = []

        async def Dock(self, request, context):
            self.requests.append(request)
            return SimpleNamespace(
                smiles=request.smiles,
                receptor_uri=request.protein_pdb,
                engine=request.engine,
                scores={"docking_score": -7.5},
                uncertainties={},
                elapsed_ms=19,
            )

    service = DockService()
    response = await _oracle_grpc_call(
        module,
        module.DockOracleServicer(service=service),
        oracle_pb2.OracleBatchRequest(
            project_id="project-1",
            request_id="request-1",
            molecule_smiles=["CCO"],
            requested_properties=["docking_score"],
            level=oracle_pb2.L2_DOCKING,
            receptor_uri="/models/receptor.pdb",
            oracle_parameters={"engine": "gnina"},
        ),
    )

    assert service.requests[0].protein_pdb == "/models/receptor.pdb"
    assert service.requests[0].engine == "gnina"
    assert response.evaluations[0].level == oracle_pb2.L2_DOCKING
    assert response.evaluations[0].outcome == oracle_pb2.ORACLE_OUTCOME_PASS


@pytest.mark.asyncio
async def test_fep_oracle_aio_contract_uses_request_inputs_and_nonconvergence_is_error() -> None:
    from mf_core.proto_gen.moleculeforge.v1.oracle import fep_pb2, oracle_pb2

    module = _load_module(
        "fep_strict_oracle_contract_test",
        ROOT / "services/fep-svc/src/fep_svc/main.py",
    )

    class FEPService:
        def __init__(self) -> None:
            self.request = None

        async def RunFEP(self, request, context):
            self.request = request
            return fep_pb2.FEPBatchResponse(
                results=[
                    fep_pb2.FEPResult(
                        ligand_a_smiles=request.reference_ligand_smiles,
                        ligand_b_smiles=request.test_ligand_smiles[0],
                        ddg_kcal_mol=-1.2,
                        ddg_uncertainty=0.3,
                        n_repeats=request.n_repeats,
                        method=request.method,
                        per_repeat_ddg={
                            f"repeat_{index}": -1.2 for index in range(1, request.n_repeats + 1)
                        },
                        converged=False,
                    )
                ],
                request_id=request.request_id,
                batch_id=request.batch_id,
                total_elapsed_ms=33,
                project_id=request.project_id,
                protein_pdb_id=request.protein_pdb_id,
                reference_ligand_smiles=request.reference_ligand_smiles,
                test_ligand_smiles=request.test_ligand_smiles,
                method=request.method,
                n_repeats=request.n_repeats,
            )

    service = FEPService()
    response = await _oracle_grpc_call(
        module,
        module.FEPOracleServicer(service=service),
        oracle_pb2.OracleBatchRequest(
            project_id="project-1",
            request_id="request-1",
            molecule_smiles=["CCN"],
            requested_properties=["rbfe"],
            level=oracle_pb2.L3_FEP,
            protein_pdb_id="7ABC",
            reference_ligand_smiles="CCO",
            oracle_parameters={"method": "openfe", "n_repeats": "2"},
        ),
    )

    assert service.request.protein_pdb_id == "7ABC"
    assert service.request.reference_ligand_smiles == "CCO"
    assert service.request.method == "openfe"
    assert service.request.n_repeats == 2
    evaluation = response.evaluations[0]
    assert evaluation.outcome == oracle_pb2.ORACLE_OUTCOME_ERROR
    assert evaluation.success is False
    assert evaluation.error_code == "NOT_CONVERGED"
    assert evaluation.scores == {}
    assert evaluation.metrics == []


@pytest.mark.asyncio
async def test_oracle_aio_contract_maps_invalid_timeout_unavailable_and_data_loss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mf_core.plugins.oracle import OracleUnavailableError
    from mf_core.proto_gen.moleculeforge.v1.oracle import boltz2_pb2, oracle_pb2

    admet = _load_module(
        "admet_oracle_error_mapping_test",
        ROOT / "services/admet-svc/src/admet_svc/main.py",
    )
    with pytest.raises(grpc.aio.AioRpcError) as invalid:
        await _oracle_grpc_call(
            admet,
            admet.ADMETOracleServicer(service=object()),
            oracle_pb2.OracleBatchRequest(
                request_id="request-1",
                molecule_smiles=["CCO"],
                requested_properties=["clearance"],
                level=oracle_pb2.L1_ML_SURROGATE,
            ),
        )
    assert invalid.value.code() == grpc.StatusCode.INVALID_ARGUMENT

    class BooleanADMET:
        async def Predict(self, request, context):
            return SimpleNamespace(
                predictions={"clearance": True},
                elapsed_ms=1,
            )

    with pytest.raises(grpc.aio.AioRpcError) as invalid_metric:
        await _oracle_grpc_call(
            admet,
            admet.ADMETOracleServicer(service=BooleanADMET()),
            oracle_pb2.OracleBatchRequest(
                project_id="project-1",
                request_id="request-1",
                molecule_smiles=["CCO"],
                requested_properties=["clearance"],
                level=oracle_pb2.L1_ML_SURROGATE,
            ),
        )
    assert invalid_metric.value.code() == grpc.StatusCode.DATA_LOSS

    monkeypatch.delenv("ADMET_MODEL_PATH", raising=False)
    monkeypatch.delenv("ADMET_ORACLE_COMMAND", raising=False)
    with pytest.raises(grpc.aio.AioRpcError) as actual_unavailable:
        await _oracle_grpc_call(
            admet,
            admet.ADMETOracleServicer(),
            oracle_pb2.OracleBatchRequest(
                project_id="project-1",
                request_id="request-1",
                molecule_smiles=["CCO"],
                requested_properties=["clearance"],
                level=oracle_pb2.L1_ML_SURROGATE,
            ),
        )
    assert actual_unavailable.value.code() == grpc.StatusCode.FAILED_PRECONDITION

    dock = _load_module(
        "dock_oracle_error_mapping_test",
        ROOT / "services/dock-svc/src/dock_svc/main.py",
    )

    class TimedOutDock:
        async def Dock(self, request, context):
            raise TimeoutError

    with pytest.raises(grpc.aio.AioRpcError) as timed_out:
        await _oracle_grpc_call(
            dock,
            dock.DockOracleServicer(service=TimedOutDock()),
            oracle_pb2.OracleBatchRequest(
                project_id="project-1",
                request_id="request-1",
                molecule_smiles=["CCO"],
                requested_properties=["docking_score"],
                level=oracle_pb2.L2_DOCKING,
                receptor_uri="/models/receptor.pdb",
                oracle_parameters={"engine": "gnina"},
            ),
        )
    assert timed_out.value.code() == grpc.StatusCode.DEADLINE_EXCEEDED

    fep = _load_module(
        "fep_oracle_error_mapping_test",
        ROOT / "services/fep-svc/src/fep_svc/main.py",
    )

    class UnavailableFEP:
        async def RunFEP(self, request, context):
            raise OracleUnavailableError("FEP runtime unavailable")

    with pytest.raises(grpc.aio.AioRpcError) as unavailable:
        await _oracle_grpc_call(
            fep,
            fep.FEPOracleServicer(service=UnavailableFEP()),
            oracle_pb2.OracleBatchRequest(
                project_id="project-1",
                request_id="request-1",
                molecule_smiles=["CCN"],
                requested_properties=["rbfe"],
                level=oracle_pb2.L3_FEP,
                protein_pdb_id="7ABC",
                reference_ligand_smiles="CCO",
                oracle_parameters={"method": "openfe", "n_repeats": "2"},
            ),
        )
    assert unavailable.value.code() == grpc.StatusCode.FAILED_PRECONDITION

    boltz = _load_module(
        "boltz_oracle_error_mapping_test",
        ROOT / "services/boltz2-svc/src/boltz2_svc/main.py",
    )

    class ReorderedBoltz:
        async def PredictAffinity(self, request, context):
            return boltz2_pb2.Boltz2BatchResponse(
                affinities=[
                    boltz2_pb2.Boltz2BindingAffinity(
                        protein_pdb_id=request.protein_pdb_id,
                        ligand_smiles=smiles,
                        delta_g_kcal_mol=-8.0,
                        uncertainty=0.2,
                    )
                    for smiles in reversed(request.ligand_smiles)
                ]
            )

    with pytest.raises(grpc.aio.AioRpcError) as data_loss:
        await _oracle_grpc_call(
            boltz,
            boltz.Boltz2OracleServicer(service=ReorderedBoltz()),
            oracle_pb2.OracleBatchRequest(
                project_id="project-1",
                request_id="request-1",
                molecule_smiles=["CCO", "CCN"],
                requested_properties=["affinity"],
                level=oracle_pb2.L1_ML_SURROGATE,
                protein_pdb_id="6OIM",
                oracle_parameters={"ensemble_size": "2"},
            ),
        )
    assert data_loss.value.code() == grpc.StatusCode.DATA_LOSS


@pytest.mark.asyncio
async def test_oracle_artifact_checksums_run_once_in_worker_thread(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from mf_core.artifacts import RequirementStatus
    from mf_core.plugins import oracle as oracle_plugin

    artifact = tmp_path / "oracle.bin"
    artifact.write_bytes(b"oracle")
    event_loop_thread = threading.get_ident()
    checksum_threads = []

    def checksum(status):
        checksum_threads.append(threading.get_ident())
        return "sha256:worker"

    monkeypatch.setattr(oracle_plugin, "_artifact_checksum", checksum)
    statuses = [
        RequirementStatus(
            name="oracle_model",
            configured=True,
            available=True,
            required=True,
            path=str(artifact),
            source="ORACLE_MODEL_PATH",
            message="available",
        )
    ]

    refs = await oracle_plugin.resolve_oracle_artifact_refs(statuses)

    assert refs[0].checksum == "sha256:worker"
    assert checksum_threads == [checksum_threads[0]]
    assert checksum_threads[0] != event_loop_thread


@pytest.mark.asyncio
async def test_oracle_artifacts_resolve_once_and_reuse_for_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mf_core.proto_gen.moleculeforge.v1.core import audit_pb2
    from mf_core.proto_gen.moleculeforge.v1.oracle import oracle_pb2

    module = _load_module(
        "admet_artifact_batch_reuse_test",
        ROOT / "services/admet-svc/src/admet_svc/main.py",
    )
    resolution_calls = []

    async def resolve(statuses):
        resolution_calls.append(list(statuses))
        return [
            audit_pb2.ArtifactRef(
                name="admet_command",
                checksum="sha256:batch",
                required=True,
            )
        ]

    monkeypatch.setattr(module, "resolve_oracle_artifact_refs", resolve)

    class ADMETService:
        async def Predict(self, request, context):
            return SimpleNamespace(
                predictions={"clearance": 1.5},
                elapsed_ms=1,
            )

    response = await _oracle_grpc_call(
        module,
        module.ADMETOracleServicer(service=ADMETService()),
        oracle_pb2.OracleBatchRequest(
            project_id="project-1",
            request_id="request-1",
            molecule_smiles=["CCO", "CCN"],
            requested_properties=["clearance"],
            level=oracle_pb2.L1_ML_SURROGATE,
        ),
    )

    assert resolution_calls == [[]]
    assert [evaluation.artifact_refs[0].checksum for evaluation in response.evaluations] == [
        "sha256:batch",
        "sha256:batch",
    ]


@pytest.mark.asyncio
async def test_oracle_command_checksum_covers_runner_script(tmp_path: Path) -> None:
    from mf_core.artifacts import RequirementStatus
    from mf_core.plugins.oracle import resolve_oracle_artifact_refs

    runner = tmp_path / "oracle_runner.py"
    runner.write_text("print('first')\n", encoding="utf-8")
    status = RequirementStatus(
        name="oracle_command",
        configured=True,
        available=True,
        required=True,
        path=f"{sys.executable} {runner}",
        source="ORACLE_COMMAND",
        message="available",
    )

    first = (await resolve_oracle_artifact_refs([status]))[0].checksum
    runner.write_text("print('second version')\n", encoding="utf-8")
    second = (await resolve_oracle_artifact_refs([status]))[0].checksum

    assert first.startswith("sha256:")
    assert second.startswith("sha256:")
    assert second != first


@pytest.mark.asyncio
async def test_negative_oracle_uncertainty_is_data_loss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mf_core.proto_gen.moleculeforge.v1.oracle import oracle_pb2

    module = _load_module(
        "admet_negative_uncertainty_test",
        ROOT / "services/admet-svc/src/admet_svc/main.py",
    )
    monkeypatch.setattr(module, "_status_objects", lambda: [])

    class ADMETService:
        async def Screen(self, request, context):
            return SimpleNamespace(
                result={
                    request.smiles: {
                        "clearance": {
                            "value": 1.5,
                            "uncertainty": -0.1,
                        }
                    }
                }
            )

    with pytest.raises(grpc.aio.AioRpcError) as error:
        await _oracle_grpc_call(
            module,
            module.ADMETOracleServicer(service=ADMETService()),
            oracle_pb2.OracleBatchRequest(
                project_id="project-1",
                request_id="request-1",
                molecule_smiles=["CCO"],
                requested_properties=["clearance"],
                level=oracle_pb2.L1_ML_SURROGATE,
            ),
            "PredictWithUncertainty",
        )

    assert error.value.code() == grpc.StatusCode.DATA_LOSS


@pytest.mark.asyncio
async def test_boltz_inference_runtime_error_becomes_computation_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mf_core.proto_gen.moleculeforge.v1.oracle import oracle_pb2

    module = _load_module(
        "boltz_inference_computation_error_test",
        ROOT / "services/boltz2-svc/src/boltz2_svc/main.py",
    )
    monkeypatch.setattr(module, "_status_objects", lambda: [])

    class FailingRunner:
        def predict_affinity(self, protein_pdb_id, ligand_smiles, ensemble_size):
            raise RuntimeError("inference failed")

    response = await _oracle_grpc_call(
        module,
        module.Boltz2OracleServicer(service=module.Boltz2Servicer(runner=FailingRunner())),
        oracle_pb2.OracleBatchRequest(
            project_id="project-1",
            request_id="request-1",
            molecule_smiles=["CCO"],
            requested_properties=["affinity"],
            level=oracle_pb2.L1_ML_SURROGATE,
            protein_pdb_id="6OIM",
            oracle_parameters={"ensemble_size": "2"},
        ),
    )

    assert response.evaluations[0].outcome == oracle_pb2.ORACLE_OUTCOME_ERROR
    assert response.evaluations[0].error_code == "COMPUTATION_ERROR"
    assert "inference failed" in response.evaluations[0].error_message


@pytest.mark.asyncio
async def test_dock_response_engine_mismatch_is_data_loss() -> None:
    from mf_core.proto_gen.moleculeforge.v1.oracle import oracle_pb2

    module = _load_module(
        "dock_response_engine_mismatch_test",
        ROOT / "services/dock-svc/src/dock_svc/main.py",
    )

    class DockService:
        async def Dock(self, request, context):
            return SimpleNamespace(
                smiles=request.smiles,
                receptor_uri=request.protein_pdb,
                engine="diffdock",
                scores={"docking_score": -7.5},
                uncertainties={},
                elapsed_ms=10,
            )

    with pytest.raises(grpc.aio.AioRpcError) as error:
        await _oracle_grpc_call(
            module,
            module.DockOracleServicer(service=DockService()),
            oracle_pb2.OracleBatchRequest(
                project_id="project-1",
                request_id="request-1",
                molecule_smiles=["CCO"],
                requested_properties=["docking_score"],
                level=oracle_pb2.L2_DOCKING,
                receptor_uri="/models/receptor.pdb",
                oracle_parameters={"engine": "gnina"},
            ),
        )

    assert error.value.code() == grpc.StatusCode.DATA_LOSS


@pytest.mark.asyncio
async def test_dock_diffdock_runner_alias_uses_logical_engine_name() -> None:
    from mf_core.proto_gen.moleculeforge.v1.oracle import oracle_pb2

    module = _load_module(
        "dock_diffdock_alias_test",
        ROOT / "services/dock-svc/src/dock_svc/main.py",
    )

    class DockService:
        async def Dock(self, request, context):
            return SimpleNamespace(
                smiles=request.smiles,
                receptor_uri=request.protein_pdb,
                engine="diffdock_l",
                scores={"docking_score": -7.5},
                uncertainties={},
                elapsed_ms=10,
            )

    response = await _oracle_grpc_call(
        module,
        module.DockOracleServicer(service=DockService()),
        oracle_pb2.OracleBatchRequest(
            project_id="project-1",
            request_id="request-1",
            molecule_smiles=["CCO"],
            requested_properties=["docking_score"],
            level=oracle_pb2.L2_DOCKING,
            receptor_uri="/models/receptor.pdb",
            oracle_parameters={"engine": "diffdock"},
        ),
    )

    assert response.evaluations[0].oracle_name == "diffdock"
    assert response.evaluations[0].outcome == oracle_pb2.ORACLE_OUTCOME_PASS


@pytest.mark.asyncio
async def test_dock_command_success_proves_selected_artifact_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from mf_core.proto_gen.moleculeforge.v1.oracle import oracle_pb2

    module = _load_module(
        "dock_command_artifact_provenance_test",
        ROOT / "services/dock-svc/src/dock_svc/main.py",
    )
    receptor = tmp_path / "receptor.pdb"
    receptor.write_text("HEADER TEST\nEND\n", encoding="utf-8")
    runner = tmp_path / "dock_runner.py"
    runner.write_text(
        "import json, sys\n"
        "request = json.load(sys.stdin)\n"
        "print(json.dumps({"
        "'smiles': request['smiles'], "
        "'receptor_uri': request['protein_pdb'], "
        "'engine': request['engine'], "
        "'scores': {'docking_score': -7.5}, "
        "'elapsed_ms': 10"
        "}))\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("DOCK_ORACLE_COMMAND", f"{sys.executable} {runner}")
    monkeypatch.setenv("GNINA_BINARY", "/missing/unused-gnina")
    monkeypatch.setenv("DIFFDOCK_MODEL_PATH", "/missing/unused-diffdock")

    response = await _oracle_grpc_call(
        module,
        module.DockOracleServicer(),
        oracle_pb2.OracleBatchRequest(
            project_id="project-1",
            request_id="request-1",
            molecule_smiles=["CCO"],
            requested_properties=["docking_score"],
            level=oracle_pb2.L2_DOCKING,
            receptor_uri=str(receptor),
            oracle_parameters={"engine": "gnina"},
        ),
    )

    artifacts = response.evaluations[0].artifact_refs
    assert [artifact.name for artifact in artifacts] == [
        "dock_oracle_command",
        "dock_receptor",
    ]
    assert all(artifact.required for artifact in artifacts)
    assert all(artifact.checksum.startswith("sha256:") for artifact in artifacts)


@pytest.mark.asyncio
async def test_boltz_response_ensemble_mismatch_is_data_loss() -> None:
    from mf_core.proto_gen.moleculeforge.v1.oracle import boltz2_pb2, oracle_pb2

    module = _load_module(
        "boltz_response_ensemble_mismatch_test",
        ROOT / "services/boltz2-svc/src/boltz2_svc/main.py",
    )

    class BoltzService:
        async def PredictAffinity(self, request, context):
            return boltz2_pb2.Boltz2BatchResponse(
                protein_pdb_id=request.protein_pdb_id,
                affinities=[
                    boltz2_pb2.Boltz2BindingAffinity(
                        protein_pdb_id=request.protein_pdb_id,
                        ligand_smiles=request.ligand_smiles[0],
                        delta_g_kcal_mol=-8.0,
                        uncertainty=0.2,
                        ki_nm=12.0,
                        ensemble_size=request.ensemble_size + 1,
                    )
                ],
            )

    with pytest.raises(grpc.aio.AioRpcError) as error:
        await _oracle_grpc_call(
            module,
            module.Boltz2OracleServicer(service=BoltzService()),
            oracle_pb2.OracleBatchRequest(
                project_id="project-1",
                request_id="request-1",
                molecule_smiles=["CCO"],
                requested_properties=["affinity"],
                level=oracle_pb2.L1_ML_SURROGATE,
                protein_pdb_id="6OIM",
                oracle_parameters={"ensemble_size": "2"},
            ),
        )

    assert error.value.code() == grpc.StatusCode.DATA_LOSS


@pytest.mark.asyncio
async def test_oracle_parameters_reject_unknown_keys() -> None:
    from mf_core.proto_gen.moleculeforge.v1.oracle import oracle_pb2

    module = _load_module(
        "boltz_unknown_parameter_test",
        ROOT / "services/boltz2-svc/src/boltz2_svc/main.py",
    )

    with pytest.raises(grpc.aio.AioRpcError) as error:
        await _oracle_grpc_call(
            module,
            module.Boltz2OracleServicer(service=object()),
            oracle_pb2.OracleBatchRequest(
                project_id="project-1",
                request_id="request-1",
                molecule_smiles=["CCO"],
                requested_properties=["affinity"],
                level=oracle_pb2.L1_ML_SURROGATE,
                protein_pdb_id="6OIM",
                oracle_parameters={"ensemble_szie": "2"},
            ),
        )

    assert error.value.code() == grpc.StatusCode.INVALID_ARGUMENT
    assert "ensemble_szie" in error.value.details()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response_request_id", "response_batch_id", "response_repeats"),
    [
        ("wrong-request", "request-1", 2),
        ("request-1", "wrong-batch", 2),
        ("request-1", "request-1", 3),
    ],
)
async def test_fep_response_identity_mismatch_is_data_loss(
    response_request_id: str,
    response_batch_id: str,
    response_repeats: int,
) -> None:
    from mf_core.proto_gen.moleculeforge.v1.oracle import fep_pb2, oracle_pb2

    module = _load_module(
        "fep_response_identity_mismatch_"
        f"{response_request_id}_{response_batch_id}_{response_repeats}",
        ROOT / "services/fep-svc/src/fep_svc/main.py",
    )

    class FEPService:
        async def RunFEP(self, request, context):
            return fep_pb2.FEPBatchResponse(
                results=[
                    fep_pb2.FEPResult(
                        ligand_a_smiles=request.reference_ligand_smiles,
                        ligand_b_smiles=request.test_ligand_smiles[0],
                        ddg_kcal_mol=-1.2,
                        ddg_uncertainty=0.3,
                        n_repeats=response_repeats,
                        method=request.method,
                        per_repeat_ddg={
                            f"repeat_{index}": -1.2 for index in range(1, response_repeats + 1)
                        },
                        converged=True,
                    )
                ],
                request_id=response_request_id,
                batch_id=response_batch_id,
                total_elapsed_ms=33,
                project_id=request.project_id,
                protein_pdb_id=request.protein_pdb_id,
                reference_ligand_smiles=request.reference_ligand_smiles,
                test_ligand_smiles=request.test_ligand_smiles,
                method=request.method,
                n_repeats=response_repeats,
            )

    with pytest.raises(grpc.aio.AioRpcError) as error:
        await _oracle_grpc_call(
            module,
            module.FEPOracleServicer(service=FEPService()),
            oracle_pb2.OracleBatchRequest(
                project_id="project-1",
                request_id="request-1",
                molecule_smiles=["CCN"],
                requested_properties=["rbfe"],
                level=oracle_pb2.L3_FEP,
                protein_pdb_id="7ABC",
                reference_ligand_smiles="CCO",
                oracle_parameters={"method": "openfe", "n_repeats": "2"},
            ),
        )

    assert error.value.code() == grpc.StatusCode.DATA_LOSS


@pytest.mark.asyncio
async def test_oracle_command_checksum_binds_non_path_argv(tmp_path: Path) -> None:
    from mf_core.artifacts import RequirementStatus
    from mf_core.plugins.oracle import resolve_oracle_artifact_refs

    runner = tmp_path / "oracle_runner.py"
    runner.write_text("print('stable runner')\n", encoding="utf-8")

    async def checksum(arguments: str) -> str:
        status = RequirementStatus(
            name="oracle_command",
            configured=True,
            available=True,
            required=True,
            path=f"{sys.executable} {runner} {arguments}",
            source="ORACLE_COMMAND",
            message="available",
        )
        return (await resolve_oracle_artifact_refs([status]))[0].checksum

    first = await checksum("--device cpu --ensemble-size 2")
    same = await checksum("--device cpu --ensemble-size 2")
    changed = await checksum("--device cuda --ensemble-size 2")

    assert first == same
    assert changed != first


@pytest.mark.asyncio
async def test_oracle_checksum_reads_content_even_when_stat_signature_is_unchanged(
    tmp_path: Path,
) -> None:
    import os

    from mf_core.artifacts import RequirementStatus
    from mf_core.plugins.oracle import resolve_oracle_artifact_refs

    artifact = tmp_path / "oracle.bin"
    artifact.write_bytes(b"first-model")
    initial_stat = artifact.stat()
    status = RequirementStatus(
        name="oracle_model",
        configured=True,
        available=True,
        required=True,
        path=str(artifact),
        source="ORACLE_MODEL_PATH",
        message="available",
    )

    first = (await resolve_oracle_artifact_refs([status]))[0].checksum
    artifact.write_bytes(b"other-model")
    os.utime(
        artifact,
        ns=(initial_stat.st_atime_ns, initial_stat.st_mtime_ns),
    )
    second = (await resolve_oracle_artifact_refs([status]))[0].checksum

    assert artifact.stat().st_size == initial_stat.st_size
    assert artifact.stat().st_mtime_ns == initial_stat.st_mtime_ns
    assert second != first


@pytest.mark.asyncio
async def test_oracle_directory_checksum_frames_each_file_content(tmp_path: Path) -> None:
    import struct

    from mf_core.artifacts import RequirementStatus
    from mf_core.plugins.oracle import resolve_oracle_artifact_refs

    artifact_dir = tmp_path / "oracle-model"
    artifact_dir.mkdir()
    first_file = artifact_dir / "a"
    second_file = artifact_dir / "b"
    payload = b"model"
    status = RequirementStatus(
        name="oracle_model",
        configured=True,
        available=True,
        required=True,
        path=str(artifact_dir),
        source="ORACLE_MODEL_PATH",
        message="available",
    )

    first_file.write_bytes(struct.pack("<Q", 1) + b"b" + payload)
    embedded_boundary = (await resolve_oracle_artifact_refs([status]))[0].checksum

    first_file.write_bytes(b"")
    second_file.write_bytes(payload)
    separate_files = (await resolve_oracle_artifact_refs([status]))[0].checksum

    assert separate_files != embedded_boundary


@pytest.mark.asyncio
async def test_admet_http_oracle_records_remote_model_artifact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mf_core.proto_gen.moleculeforge.v1.oracle import oracle_pb2
    from mf_oracles.admet_ai.oracle import ADMETHTTPRunner

    module = _load_module(
        "admet_http_artifact_contract_test",
        ROOT / "services/admet-svc/src/admet_svc/main.py",
    )
    monkeypatch.delenv("ADMET_ORACLE_COMMAND", raising=False)
    monkeypatch.setenv("ADMET_SERVICE_URL", "https://admet.local")
    monkeypatch.setenv("ADMET_TARGETS", "clearance")
    runner = ADMETHTTPRunner(
        service_url="https://admet.local",
        targets=["clearance"],
        post_json=lambda _url, payload, _timeout: {
            "model_version": "chemprop-2026-07",
            "artifact": {
                "name": "admet-chemprop",
                "sha256": "e" * 64,
            },
            "results": [
                {
                    "smiles": payload["smiles"][0],
                    "predictions": {"clearance": 1.5},
                }
            ],
        },
    )
    monkeypatch.setattr(ADMETHTTPRunner, "from_env", classmethod(lambda cls: runner))

    response = await _oracle_grpc_call(
        module,
        module.ADMETOracleServicer(),
        oracle_pb2.OracleBatchRequest(
            project_id="project-1",
            request_id="request-1",
            molecule_smiles=["C(C)O"],
            requested_properties=["clearance"],
            level=oracle_pb2.L1_ML_SURROGATE,
        ),
    )

    evaluation = response.evaluations[0]
    assert evaluation.molecule_smiles == "C(C)O"
    assert evaluation.model_version == "chemprop-2026-07"
    assert len(evaluation.artifact_refs) == 1
    assert evaluation.artifact_refs[0].name == "admet-chemprop"
    assert evaluation.artifact_refs[0].version == "chemprop-2026-07"
    assert evaluation.artifact_refs[0].checksum == f"sha256:{'e' * 64}"
    assert evaluation.artifact_refs[0].required is True


@pytest.mark.asyncio
async def test_admet_http_uncertainty_survives_real_http_and_grpc_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from mf_core.proto_gen.moleculeforge.v1.oracle import oracle_pb2

    module = _load_module(
        "admet_http_uncertainty_contract_test",
        ROOT / "services/admet-svc/src/admet_svc/main.py",
    )
    requests: list[dict] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            body = self.rfile.read(int(self.headers["Content-Length"]))
            payload = json.loads(body)
            requests.append(payload)
            response = json.dumps(
                {
                    "model_version": "chemprop-2026-07",
                    "artifact": {
                        "name": "admet-chemprop",
                        "sha256": "e" * 64,
                    },
                    "results": [
                        {
                            "smiles": payload["smiles"][0],
                            "predictions": {"clearance": 1.5},
                            "uncertainties": {"clearance": 0.25},
                        }
                    ],
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

        def log_message(self, format: str, *args) -> None:
            return None

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    monkeypatch.delenv("ADMET_ORACLE_COMMAND", raising=False)
    monkeypatch.setenv(
        "ADMET_SERVICE_URL",
        f"http://127.0.0.1:{server.server_address[1]}",
    )
    monkeypatch.setenv("ADMET_TARGETS", "clearance")

    try:
        response = await _oracle_grpc_call(
            module,
            module.ADMETOracleServicer(),
            oracle_pb2.OracleBatchRequest(
                project_id="project-1",
                request_id="request-1",
                molecule_smiles=["C(C)O"],
                requested_properties=["clearance"],
                level=oracle_pb2.L1_ML_SURROGATE,
                return_uncertainty=True,
            ),
            "PredictWithUncertainty",
        )
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=5)

    assert requests == [
        {
            "smiles": ["CCO"],
            "endpoints": ["clearance"],
            "batch_size": 64,
            "return_uncertainty": True,
        }
    ]
    evaluation = response.evaluations[0]
    assert evaluation.molecule_smiles == "C(C)O"
    assert evaluation.outcome == oracle_pb2.ORACLE_OUTCOME_PASS
    assert evaluation.success is True
    assert evaluation.scores == {"clearance": 1.5}
    assert evaluation.uncertainties == {"clearance": 0.25}
    assert len(evaluation.metrics) == 1
    assert evaluation.metrics[0].property == "clearance"
    assert evaluation.metrics[0].value == 1.5
    assert evaluation.metrics[0].uncertainty == 0.25
    assert evaluation.model_version == "chemprop-2026-07"
    assert len(evaluation.artifact_refs) == 1
    assert evaluation.artifact_refs[0].checksum == f"sha256:{'e' * 64}"


@pytest.mark.asyncio
async def test_admet_http_oracle_missing_artifact_metadata_is_data_loss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mf_core.proto_gen.moleculeforge.v1.oracle import oracle_pb2
    from mf_oracles.admet_ai.oracle import ADMETHTTPRunner

    module = _load_module(
        "admet_http_missing_artifact_contract_test",
        ROOT / "services/admet-svc/src/admet_svc/main.py",
    )
    monkeypatch.delenv("ADMET_ORACLE_COMMAND", raising=False)
    monkeypatch.setenv("ADMET_SERVICE_URL", "https://admet.local")
    monkeypatch.setenv("ADMET_TARGETS", "clearance")
    runner = ADMETHTTPRunner(
        service_url="https://admet.local",
        targets=["clearance"],
        post_json=lambda _url, payload, _timeout: {
            "results": [
                {
                    "smiles": payload["smiles"][0],
                    "predictions": {"clearance": 1.5},
                }
            ],
        },
    )
    monkeypatch.setattr(ADMETHTTPRunner, "from_env", classmethod(lambda cls: runner))

    with pytest.raises(grpc.aio.AioRpcError) as error:
        await _oracle_grpc_call(
            module,
            module.ADMETOracleServicer(),
            oracle_pb2.OracleBatchRequest(
                project_id="project-1",
                request_id="request-1",
                molecule_smiles=["CCO"],
                requested_properties=["clearance"],
                level=oracle_pb2.L1_ML_SURROGATE,
            ),
        )

    assert error.value.code() == grpc.StatusCode.DATA_LOSS


@pytest.mark.asyncio
async def test_dock_command_requires_exact_input_identity_and_consistent_scores(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_module(
        "dock_command_identity_contract_test",
        ROOT / "services/dock-svc/src/dock_svc/main.py",
    )
    receptor = tmp_path / "receptor.pdb"
    receptor.write_text("HEADER TEST\nEND\n", encoding="utf-8")
    runner = tmp_path / "dock_runner.py"
    runner.write_text(
        "import json, sys\n"
        "request = json.load(sys.stdin)\n"
        "print(json.dumps({"
        "'smiles': request['smiles'], "
        "'receptor_uri': request['protein_pdb'], "
        "'engine': request['engine'], "
        "'score': -7.0, "
        "'scores': {'docking_score': -8.0}, "
        "'elapsed_ms': 10"
        "}))\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("DOCK_ORACLE_COMMAND", f"{sys.executable} {runner}")

    with pytest.raises(module.OracleDataError, match="contradict"):
        await module.DockServicer().Dock(
            SimpleNamespace(
                smiles="CCO",
                protein_pdb=str(receptor),
                engine="gnina",
            ),
            None,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response_smiles", "response_receptor"),
    [
        ("WRONG", "/models/receptor.pdb"),
        ("CCO", "/models/wrong-receptor.pdb"),
    ],
)
async def test_dock_oracle_rejects_injected_response_identity_mismatch(
    response_smiles: str,
    response_receptor: str,
) -> None:
    from mf_core.proto_gen.moleculeforge.v1.oracle import oracle_pb2

    module = _load_module(
        f"dock_response_identity_{response_smiles}_{response_receptor}",
        ROOT / "services/dock-svc/src/dock_svc/main.py",
    )

    class DockService:
        async def Dock(self, request, context):
            return SimpleNamespace(
                smiles=response_smiles,
                receptor_uri=response_receptor,
                engine=request.engine,
                scores={"docking_score": -7.5},
                uncertainties={},
                elapsed_ms=10,
            )

    with pytest.raises(grpc.aio.AioRpcError) as error:
        await _oracle_grpc_call(
            module,
            module.DockOracleServicer(service=DockService()),
            oracle_pb2.OracleBatchRequest(
                project_id="project-1",
                request_id="request-1",
                molecule_smiles=["CCO"],
                requested_properties=["docking_score"],
                level=oracle_pb2.L2_DOCKING,
                receptor_uri="/models/receptor.pdb",
                oracle_parameters={"engine": "gnina"},
            ),
        )

    assert error.value.code() == grpc.StatusCode.DATA_LOSS


@pytest.mark.asyncio
async def test_dock_oracle_evidence_includes_receptor_content(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from mf_core.proto_gen.moleculeforge.v1.oracle import oracle_pb2

    module = _load_module(
        "dock_receptor_artifact_contract_test",
        ROOT / "services/dock-svc/src/dock_svc/main.py",
    )
    receptor = tmp_path / "receptor.pdb"
    receptor.write_text("HEADER FIRST\nEND\n", encoding="utf-8")
    runner = tmp_path / "dock_runner.py"
    runner.write_text(
        "import json, sys\n"
        "request = json.load(sys.stdin)\n"
        "print(json.dumps({"
        "'smiles': request['smiles'], "
        "'receptor_uri': request['protein_pdb'], "
        "'engine': request['engine'], "
        "'scores': {'docking_score': -7.5}, "
        "'elapsed_ms': 10"
        "}))\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("DOCK_ORACLE_COMMAND", f"{sys.executable} {runner}")

    async def evaluate() -> str:
        response = await _oracle_grpc_call(
            module,
            module.DockOracleServicer(),
            oracle_pb2.OracleBatchRequest(
                project_id="project-1",
                request_id="request-1",
                molecule_smiles=["CCO"],
                requested_properties=["docking_score"],
                level=oracle_pb2.L2_DOCKING,
                receptor_uri=str(receptor),
                oracle_parameters={"engine": "gnina"},
            ),
        )
        artifacts = response.evaluations[0].artifact_refs
        assert [artifact.name for artifact in artifacts] == [
            "dock_oracle_command",
            "dock_receptor",
        ]
        return artifacts[1].checksum

    first = await evaluate()
    receptor.write_text("HEADER SECOND VERSION\nEND\n", encoding="utf-8")
    second = await evaluate()

    assert first.startswith("sha256:")
    assert second != first


def test_boltz_cli_timeout_is_finite_and_terminates_process_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import subprocess

    module = _load_module(
        "boltz_cli_timeout_contract_test",
        ROOT / "services/boltz2-svc/src/boltz2_svc/main.py",
    )
    events: list[object] = []

    class Process:
        pid = 321
        returncode = None

        def __init__(self, command, **kwargs):
            events.append((command, kwargs))

        def communicate(self, input=None, timeout=None):
            events.append(("communicate", input, timeout))
            if timeout is not None:
                raise subprocess.TimeoutExpired(["boltz"], timeout)
            self.returncode = -9
            return "", ""

    monkeypatch.setattr(module.subprocess, "Popen", Process)
    monkeypatch.setattr(
        module.os,
        "killpg",
        lambda pid, signal: events.append(("killpg", pid, signal)),
    )

    with pytest.raises(subprocess.TimeoutExpired):
        module._run_process_group(
            ["boltz", "predict"],
            timeout=0.25,
        )

    assert events[0][1]["start_new_session"] is True
    assert any(
        event[0] == "killpg" and event[1] == 321 for event in events if isinstance(event, tuple)
    )

    monkeypatch.setenv("BOLTZ2_ORACLE_TIMEOUT_SECONDS", "inf")
    with pytest.raises(RuntimeError, match="finite positive"):
        module.BoltzCommandRunner("boltz")
    monkeypatch.setenv("BOLTZ2_ORACLE_COMMAND", sys.executable)
    with pytest.raises(RuntimeError, match="boltz2_oracle_timeout"):
        module._require_runtime()


@pytest.mark.asyncio
async def test_fep_command_requires_complete_request_identity_and_repeat_count(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from mf_core.proto_gen.moleculeforge.v1.oracle import fep_pb2

    module = _load_module(
        "fep_command_identity_contract_test",
        ROOT / "services/fep-svc/src/fep_svc/main.py",
    )
    runner = tmp_path / "fep_runner.py"
    runner.write_text(
        "import json, sys\n"
        "request = json.load(sys.stdin)\n"
        "print(json.dumps({"
        "'request_id': request['request_id'], "
        "'batch_id': request['batch_id'], "
        "'project_id': request['project_id'], "
        "'protein_pdb_id': request['protein_pdb_id'], "
        "'reference_ligand_smiles': request['reference_ligand_smiles'], "
        "'test_ligand_smiles': request['test_ligand_smiles'], "
        "'method': request['method'], "
        "'n_repeats': request['n_repeats'], "
        "'total_elapsed_ms': 10, "
        "'results': [{"
        "'ligand_a_smiles': request['reference_ligand_smiles'], "
        "'ligand_b_smiles': request['test_ligand_smiles'][0], "
        "'ddg_kcal_mol': -1.0, "
        "'ddg_uncertainty': 0.2, "
        "'n_repeats': request['n_repeats'], "
        "'method': request['method'], "
        "'per_repeat_ddg': {'repeat_1': -1.0}, "
        "'converged': True"
        "}]"
        "}))\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("FEP_ORACLE_COMMAND", f"{sys.executable} {runner}")

    with pytest.raises(module.OracleDataError, match="per_repeat_ddg"):
        await module.FEPServicer().RunFEP(
            fep_pb2.FEPBatchRequest(
                project_id="project-1",
                request_id="request-1",
                batch_id="batch-1",
                protein_pdb_id="7ABC",
                reference_ligand_smiles="CCO",
                test_ligand_smiles=["CCN"],
                method="openfe",
                n_repeats=2,
            ),
            None,
        )


def test_fep_command_rejects_missing_top_level_repeat_identity() -> None:
    module = _load_module(
        "fep_missing_top_level_repeat_identity_test",
        ROOT / "services/fep-svc/src/fep_svc/main.py",
    )
    request = SimpleNamespace(
        project_id="project-1",
        request_id="request-1",
        batch_id="batch-1",
        protein_pdb_id="7ABC",
        reference_ligand_smiles="CCO",
        test_ligand_smiles=["CCN"],
        method="openfe",
        n_repeats=2,
    )
    response = {
        "request_id": "request-1",
        "batch_id": "batch-1",
        "project_id": "project-1",
        "protein_pdb_id": "7ABC",
        "reference_ligand_smiles": "CCO",
        "test_ligand_smiles": ["CCN"],
        "method": "openfe",
    }

    with pytest.raises(module.OracleDataError, match="n_repeats"):
        module._validate_fep_command_identity(response, request)


@pytest.mark.asyncio
async def test_fep_oracle_rejects_missing_per_repeat_evidence() -> None:
    from mf_core.proto_gen.moleculeforge.v1.oracle import fep_pb2, oracle_pb2

    module = _load_module(
        "fep_missing_repeat_evidence_contract_test",
        ROOT / "services/fep-svc/src/fep_svc/main.py",
    )

    class FEPService:
        async def RunFEP(self, request, context):
            return fep_pb2.FEPBatchResponse(
                results=[
                    fep_pb2.FEPResult(
                        ligand_a_smiles=request.reference_ligand_smiles,
                        ligand_b_smiles=request.test_ligand_smiles[0],
                        ddg_kcal_mol=-1.2,
                        ddg_uncertainty=0.3,
                        n_repeats=request.n_repeats,
                        method=request.method,
                        per_repeat_ddg={"repeat_1": -1.2},
                        converged=True,
                    )
                ],
                request_id=request.request_id,
                batch_id=request.batch_id,
                total_elapsed_ms=33,
                project_id=request.project_id,
                protein_pdb_id=request.protein_pdb_id,
                reference_ligand_smiles=request.reference_ligand_smiles,
                test_ligand_smiles=request.test_ligand_smiles,
                method=request.method,
                n_repeats=request.n_repeats,
            )

    with pytest.raises(grpc.aio.AioRpcError) as error:
        await _oracle_grpc_call(
            module,
            module.FEPOracleServicer(service=FEPService()),
            oracle_pb2.OracleBatchRequest(
                project_id="project-1",
                request_id="request-1",
                molecule_smiles=["CCN"],
                requested_properties=["rbfe"],
                level=oracle_pb2.L3_FEP,
                protein_pdb_id="7ABC",
                reference_ligand_smiles="CCO",
                oracle_parameters={"method": "openfe", "n_repeats": "2"},
            ),
        )

    assert error.value.code() == grpc.StatusCode.DATA_LOSS


@pytest.mark.asyncio
async def test_fep_identity_fields_survive_real_grpc_round_trip(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from mf_core.proto_gen.moleculeforge.v1.oracle import fep_pb2

    module = _load_module(
        "fep_identity_grpc_round_trip_test",
        ROOT / "services/fep-svc/src/fep_svc/main.py",
    )
    runner = tmp_path / "fep_runner.py"
    runner.write_text(
        "import json, sys\n"
        "request = json.load(sys.stdin)\n"
        "print(json.dumps({"
        "'request_id': request['request_id'], "
        "'batch_id': request['batch_id'], "
        "'project_id': request['project_id'], "
        "'protein_pdb_id': request['protein_pdb_id'], "
        "'reference_ligand_smiles': request['reference_ligand_smiles'], "
        "'test_ligand_smiles': request['test_ligand_smiles'], "
        "'method': request['method'], "
        "'n_repeats': request['n_repeats'], "
        "'total_elapsed_ms': 10, "
        "'results': [{"
        "'ligand_a_smiles': request['reference_ligand_smiles'], "
        "'ligand_b_smiles': request['test_ligand_smiles'][0], "
        "'ddg_kcal_mol': -1.0, "
        "'ddg_uncertainty': 0.2, "
        "'n_repeats': request['n_repeats'], "
        "'method': request['method'], "
        "'per_repeat_ddg': {'repeat_1': -1.0}, "
        "'converged': True"
        "}]"
        "}))\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("FEP_ORACLE_COMMAND", f"{sys.executable} {runner}")
    request = fep_pb2.FEPBatchRequest(
        project_id="project-1",
        request_id="request-1",
        batch_id="batch-1",
        protein_pdb_id="7ABC",
        reference_ligand_smiles="CCO",
        test_ligand_smiles=["CCN"],
        method="openfe",
        n_repeats=1,
    )
    server = grpc.aio.server()
    module.fep_pb2_grpc.add_FEPServiceServicer_to_server(module.FEPServicer(), server)
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    channel = grpc.aio.insecure_channel(f"127.0.0.1:{port}")
    try:
        response = await module.fep_pb2_grpc.FEPServiceStub(channel).RunFEP(request)
    finally:
        await channel.close()
        await server.stop(None)

    assert response.request_id == request.request_id
    assert response.batch_id == request.batch_id
    assert response.project_id == request.project_id
    assert response.protein_pdb_id == request.protein_pdb_id
    assert response.reference_ligand_smiles == request.reference_ligand_smiles
    assert list(response.test_ligand_smiles) == list(request.test_ligand_smiles)
    assert response.method == request.method
    assert response.n_repeats == request.n_repeats


@pytest.mark.asyncio
async def test_fep_oracle_rejects_legacy_response_without_identity_fields() -> None:
    from mf_core.proto_gen.moleculeforge.v1.oracle import fep_pb2, oracle_pb2

    module = _load_module(
        "fep_legacy_response_identity_test",
        ROOT / "services/fep-svc/src/fep_svc/main.py",
    )

    class LegacyFEPService:
        async def RunFEP(self, request, context):
            return fep_pb2.FEPBatchResponse(
                results=[
                    fep_pb2.FEPResult(
                        ligand_a_smiles=request.reference_ligand_smiles,
                        ligand_b_smiles=request.test_ligand_smiles[0],
                        ddg_kcal_mol=-1.2,
                        ddg_uncertainty=0.3,
                        n_repeats=request.n_repeats,
                        method=request.method,
                        per_repeat_ddg={"repeat_1": -1.2},
                        converged=True,
                    )
                ],
                batch_id=request.project_id,
                total_elapsed_ms=33,
            )

    with pytest.raises(grpc.aio.AioRpcError) as error:
        await _oracle_grpc_call(
            module,
            module.FEPOracleServicer(service=LegacyFEPService()),
            oracle_pb2.OracleBatchRequest(
                project_id="project-1",
                request_id="request-1",
                molecule_smiles=["CCN"],
                requested_properties=["rbfe"],
                level=oracle_pb2.L3_FEP,
                protein_pdb_id="7ABC",
                reference_ligand_smiles="CCO",
                oracle_parameters={"method": "openfe", "n_repeats": "1"},
            ),
        )

    assert error.value.code() == grpc.StatusCode.DATA_LOSS


@pytest.mark.asyncio
async def test_boltz_timeout_maps_to_deadline_exceeded() -> None:
    import subprocess

    from mf_core.proto_gen.moleculeforge.v1.oracle import oracle_pb2

    module = _load_module(
        "boltz_timeout_error_mapping_test",
        ROOT / "services/boltz2-svc/src/boltz2_svc/main.py",
    )

    class TimedOutBoltzService:
        async def PredictAffinity(self, request, context):
            raise subprocess.TimeoutExpired(["boltz"], 0.1)

    with pytest.raises(grpc.aio.AioRpcError) as error:
        await _oracle_grpc_call(
            module,
            module.Boltz2OracleServicer(service=TimedOutBoltzService()),
            oracle_pb2.OracleBatchRequest(
                project_id="project-1",
                request_id="request-1",
                molecule_smiles=["CCO"],
                requested_properties=["affinity"],
                level=oracle_pb2.L1_ML_SURROGATE,
                protein_pdb_id="6OIM",
                oracle_parameters={"ensemble_size": "2"},
            ),
        )

    assert error.value.code() == grpc.StatusCode.DEADLINE_EXCEEDED


@pytest.mark.asyncio
async def test_boltz_oracle_rejects_incomplete_ensemble_members() -> None:
    from mf_core.proto_gen.moleculeforge.v1.oracle import boltz2_pb2, oracle_pb2

    module = _load_module(
        "boltz_incomplete_ensemble_contract_test",
        ROOT / "services/boltz2-svc/src/boltz2_svc/main.py",
    )

    class BoltzService:
        async def PredictAffinity(self, request, context):
            return boltz2_pb2.Boltz2BatchResponse(
                protein_pdb_id=request.protein_pdb_id,
                affinities=[
                    boltz2_pb2.Boltz2BindingAffinity(
                        protein_pdb_id=request.protein_pdb_id,
                        ligand_smiles=request.ligand_smiles[0],
                        delta_g_kcal_mol=-8.0,
                        uncertainty=0.2,
                        ki_nm=12.0,
                        ensemble_size=request.ensemble_size,
                        per_member_dg=[-8.0],
                    )
                ],
                elapsed_ms=10,
            )

    with pytest.raises(grpc.aio.AioRpcError) as error:
        await _oracle_grpc_call(
            module,
            module.Boltz2OracleServicer(service=BoltzService()),
            oracle_pb2.OracleBatchRequest(
                project_id="project-1",
                request_id="request-1",
                molecule_smiles=["CCO"],
                requested_properties=["affinity"],
                level=oracle_pb2.L1_ML_SURROGATE,
                protein_pdb_id="6OIM",
                oracle_parameters={"ensemble_size": "2"},
            ),
        )

    assert error.value.code() == grpc.StatusCode.DATA_LOSS


def test_boltz_command_row_rejects_incomplete_ensemble_members() -> None:
    module = _load_module(
        "boltz_command_incomplete_ensemble_contract_test",
        ROOT / "services/boltz2-svc/src/boltz2_svc/main.py",
    )
    row = {
        "protein_pdb_id": "6OIM",
        "ligand_smiles": "CCO",
        "delta_g_kcal_mol": -8.0,
        "uncertainty": 0.2,
        "ki_nm": 12.0,
        "ensemble_size": 2,
        "per_member_dg": [-8.0],
    }

    with pytest.raises(module.OracleDataError, match="per_member_dg"):
        module._require_affinity_row(row)


def test_agents_network_policy_allows_deployed_service_ports() -> None:
    import yaml

    documents = list(
        yaml.safe_load_all(
            (ROOT / "infra/kubernetes/namespaces/mf-agents-ns.yaml").read_text(encoding="utf-8")
        )
    )
    policy = next(
        document
        for document in documents
        if document.get("kind") == "NetworkPolicy"
        and document["metadata"]["name"] == "agents-netpol"
    )
    ingress_ports = {
        int(port["port"])
        for rule in policy["spec"]["ingress"]
        for port in rule.get("ports", [])
        if port.get("protocol") == "TCP"
    }

    assert {8000, 8010, 8011, 50051, 50071} <= ingress_ports


def test_generators_network_policy_allows_deployed_grpc_ports() -> None:
    import yaml

    documents = list(
        yaml.safe_load_all(
            (ROOT / "infra/kubernetes/namespaces/mf-generators-ns.yaml").read_text(
                encoding="utf-8"
            )
        )
    )
    policy = next(
        document
        for document in documents
        if document.get("kind") == "NetworkPolicy"
        and document["metadata"]["name"] == "generators-netpol"
    )
    ingress_ports = {
        int(port["port"])
        for rule in policy["spec"]["ingress"]
        for port in rule.get("ports", [])
        if port.get("protocol") == "TCP"
    }

    assert {50051, 50062, 50065, 50066, 50067, 50069} <= ingress_ports
