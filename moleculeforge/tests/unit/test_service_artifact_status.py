"""Service artifact status reporting."""
from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import re
import sys
import uuid
from pathlib import Path
from types import ModuleType, SimpleNamespace

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
    helm_values = (ROOT / "infra/helm/moleculeforge/values.yaml").read_text(
        encoding="utf-8"
    )

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
        "CIG_SEMANTIC_PARSER_TIMEOUT_SECONDS: "
        "${CIG_SEMANTIC_PARSER_TIMEOUT_SECONDS:-30}"
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

    parser_status = next(
        item for item in status if item["name"] == "cig_semantic_parser_command"
    )
    refiner_status = next(
        item for item in status if item["name"] == "cig_refinement_command"
    )
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
    runner = tmp_path / "dock_runner.py"
    runner.write_text(
        "import json, sys\n"
        "request = json.load(sys.stdin)\n"
        "assert request['smiles'] == 'CCO'\n"
        "assert request['engine'] == 'diffdock'\n"
        "print(json.dumps({"
        "'engine': 'diffdock_l', "
        "'scores': {'docking_score': -8.5}, "
        "'uncertainties': {'docking_score': 0.2}, "
        "'elapsed_ms': 17"
        "}))\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("DOCK_ORACLE_COMMAND", f"{sys.executable} {runner}")

    response = await module.DockServicer().Dock(
        SimpleNamespace(smiles="CCO", engine="diffdock"),
        None,
    )

    assert response.engine == "diffdock_l"
    assert response.scores == {"docking_score": -8.5}
    assert response.uncertainties == {"docking_score": 0.2}
    assert response.elapsed_ms == 17


@pytest.mark.asyncio
async def test_dock_oracle_uses_default_receptor_for_oracle_requests(
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
        "'engine': 'gnina', "
        "'scores': {'docking_score': -6.5}, "
        "'elapsed_ms': 19"
        "}))\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("DOCK_ORACLE_COMMAND", f"{sys.executable} {runner}")
    monkeypatch.setenv("DOCK_ORACLE_RECEPTOR_PDB", str(receptor))

    from mf_core.proto_gen.moleculeforge.v1.oracle import oracle_pb2

    response = await module.DockOracleServicer().Evaluate(
        oracle_pb2.OracleBatchRequest(
            molecule_smiles=["CCO"],
            level=oracle_pb2.L2_DOCKING,
            requested_properties=["docking_score"],
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
    helm_values = (ROOT / "infra/helm/moleculeforge/values.yaml").read_text(
        encoding="utf-8"
    )

    assert "FEAST_REPO_PATH" in compose
    assert "FEAST_REPO_PATH" in k8s
    assert "FEAST_REPO_PATH" in helm_values
    assert "name: feature-store-config" in k8s
    assert "configMapKeyRef:" in k8s
    assert "envValueFrom:" in helm_values


def test_admet_runtime_status_reports_missing_model(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module(
        "admet_status_test",
        ROOT / "services/admet-svc/src/admet_svc/main.py",
    )
    monkeypatch.delenv("ADMET_MODEL_PATH", raising=False)

    statuses = module.runtime_status()

    assert statuses[0]["name"] == "admet_model"
    assert statuses[0]["available"] is False


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
    assert generator.humu_latent_sampler is not None


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
    helm_values = (ROOT / "infra/helm/moleculeforge/values.yaml").read_text(
        encoding="utf-8"
    )

    for env_name in (
        "FRAGFM_VOCAB_PATH",
        "FRAGFM_CHECKPOINT_PATH",
        "FRAGFM_RATE_MATRIX_PATH",
        "FRAGFM_HUMU_CURVATURE",
    ):
        assert env_name in compose
        assert env_name in k8s
        assert env_name in helm_values

    assert "FRAGFM_HUMU_CURVATURE: ${FRAGFM_HUMU_CURVATURE:-1.0}" in compose
    assert (
        "FRAGFM_VOCAB_PATH: ${FRAGFM_VOCAB_PATH:-checkpoints/fragfm_humu_5k/vocab.json}"
        in compose
    )
    assert (
        "FRAGFM_CHECKPOINT_PATH: "
        "${FRAGFM_CHECKPOINT_PATH:-checkpoints/fragfm_humu_5k/best_model.pt}"
        in compose
    )
    assert (
        "FRAGFM_RATE_MATRIX_PATH: "
        "${FRAGFM_RATE_MATRIX_PATH:-checkpoints/fragfm_humu_5k/rate_matrix.pt}"
        in compose
    )
    assert (ROOT / "checkpoints/fragfm_humu_5k/vocab.json").is_file()
    assert (ROOT / "checkpoints/fragfm_humu_5k/best_model.pt").is_file()
    assert (ROOT / "checkpoints/fragfm_humu_5k/rate_matrix.pt").is_file()
    quality_report = json.loads(
        (ROOT / "checkpoints/fragfm_humu_5k/quality_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert quality_report["status"] == "pass"
    assert quality_report["humu_embedding_coverage"] == pytest.approx(1.0)
    assert "name: fragfm-generator-config" in k8s
    assert "configMapKeyRef:" in k8s
    assert "envValueFrom:" in helm_values
    for config in (
        _k8s_configmap_data(k8s, "mf-generators", "fragfm-generator-config"),
        _helm_configmap_data(helm_values, "mf-generators", "fragfm-generator-config"),
    ):
        assert config["vocab-path"] == "checkpoints/fragfm_humu_5k/vocab.json"
        assert config["checkpoint-path"] == "checkpoints/fragfm_humu_5k/best_model.pt"
        assert config["rate-matrix-path"] == "checkpoints/fragfm_humu_5k/rate_matrix.pt"
        assert config["humu-curvature"] == "1.0"


def test_fragfm_deployment_default_artifact_loads_and_generates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module(
        "fragfm_service_humu_5k_runtime_smoke_test",
        ROOT / "services/fragfm-generator-svc/src/fragfm_generator_svc/main.py",
    )
    monkeypatch.setenv(
        "FRAGFM_VOCAB_PATH",
        str(ROOT / "checkpoints/fragfm_humu_5k/vocab.json"),
    )
    monkeypatch.setenv(
        "FRAGFM_CHECKPOINT_PATH",
        str(ROOT / "checkpoints/fragfm_humu_5k/best_model.pt"),
    )
    monkeypatch.setenv(
        "FRAGFM_RATE_MATRIX_PATH",
        str(ROOT / "checkpoints/fragfm_humu_5k/rate_matrix.pt"),
    )
    monkeypatch.setenv("FRAGFM_HUMU_CURVATURE", "1.0")

    generator = module._build_generator()
    molecules = asyncio.run(generator.generate(batch_size=1))

    assert len(molecules) == 1
    assert molecules[0].smiles
    from rdkit import Chem

    assert Chem.MolFromSmiles(molecules[0].smiles) is not None
    assert molecules[0].metadata["generator_name"] == "fragfm"
    assert molecules[0].metadata["fragment_vocabulary"].endswith(
        "checkpoints/fragfm_humu_5k/vocab.json"
    )
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


def test_mmpt_deployment_wires_index_rag_and_decoder_env() -> None:
    compose = (ROOT / "infra/docker/docker-compose.dev.yml").read_text(encoding="utf-8")
    k8s = (ROOT / "infra/kubernetes/deployments/moleculeforge-services.yaml").read_text(
        encoding="utf-8"
    )
    helm_values = (ROOT / "infra/helm/moleculeforge/values.yaml").read_text(
        encoding="utf-8"
    )

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

    assert "MMPT_PATENT_RAG_TIMEOUT_SECONDS: ${MMPT_PATENT_RAG_TIMEOUT_SECONDS:-300}" in compose
    assert (
        "MMPT_INDEX_URI: ${MMPT_INDEX_URI:-file:///workspace/models/artifacts/mmpt/mmpt_index.json}"
        in compose
    )
    assert (ROOT / "models/artifacts/mmpt/mmpt_index.json").is_file()
    assert (
        "MMPT_SEQ2SEQ_DECODER_TIMEOUT_SECONDS: "
        "${MMPT_SEQ2SEQ_DECODER_TIMEOUT_SECONDS:-300}"
    ) in compose
    assert "name: mmpt-generator-config" in k8s
    assert "configMapKeyRef:" in k8s
    assert "envValueFrom:" in helm_values
    for config in (
        _k8s_configmap_data(k8s, "mf-generators", "mmpt-generator-config"),
        _helm_configmap_data(helm_values, "mf-generators", "mmpt-generator-config"),
    ):
        assert config["index-uri"] == "file:///workspace/models/artifacts/mmpt/mmpt_index.json"
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

    rag_status = next(
        item for item in status if item["name"] == "mmpt_patent_rag_command"
    )
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
    helm_values = (ROOT / "infra/helm/moleculeforge/values.yaml").read_text(
        encoding="utf-8"
    )

    for env_name in ("HUMU_CHECKPOINT_PATH", "HUMU_DEVICE"):
        assert env_name in compose
        assert env_name in k8s
        assert env_name in helm_values

    assert "HUMU_DEVICE: ${HUMU_DEVICE:-cpu}" in compose
    assert (
        "HUMU_CHECKPOINT_PATH: ${HUMU_CHECKPOINT_PATH:-checkpoints/humu/best_model.pt}"
        in compose
    )
    assert (ROOT / "checkpoints/humu/best_model.pt").is_file()
    assert "name: humu-encoder-config" in k8s
    assert "configMapKeyRef:" in k8s
    assert "envValueFrom:" in helm_values
    for config in (
        _k8s_configmap_data(k8s, "mf-generators", "humu-encoder-config"),
        _helm_configmap_data(helm_values, "mf-generators", "humu-encoder-config"),
    ):
        assert config["checkpoint-path"] == "checkpoints/humu/best_model.pt"
        assert config["device"] == "cpu"


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
        (item["namespace"], item["name"])
        for item in helm_values.get("configMaps", {}).values()
    }
    helm_secrets = {
        (item["namespace"], item["name"])
        for item in helm_values.get("secrets", {}).values()
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
                    missing_helm_targets.append(
                        (service_name, env_name, namespace, target_service)
                    )
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


class _RecordingAbortContext:
    def __init__(self) -> None:
        self.code = None
        self.message = ""

    async def abort(self, code, message: str) -> None:
        self.code = code
        self.message = message


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
        SimpleNamespace(
            project_id="project-1",
            batch_size=1,
            intent_cone={"axis": [1.0] + [0.0] * 128, "half_angle": 0.2},
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

    with pytest.raises(ValueError, match="intent_cone"):
        asyncio.run(
            service.Generate(
                SimpleNamespace(
                    project_id="project-1",
                    batch_size=1,
                    intent_cone={"axis": "not-a-vector", "half_angle": 0.2},
                ),
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
        SimpleNamespace(
            project_id="project-1",
            batch_size=1,
            intent_cone={"axis": [1.0] + [0.0] * 128, "half_angle": 0.2},
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
            [
                {
                    "smiles": "CCO",
                    "catalog_id": "CAT-1",
                    "source": "local_catalog",
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
    assert response.catalog_id == "CAT-1"
    assert response.catalog_source == "local_catalog"
    assert response.source_timestamp == "2026-05-01T00:00:00Z"
    assert response.price == 12.5
    assert response.lead_time_days == 3


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
    monkeypatch.setenv("SUPPLY_CATALOG_URI", stock_path.as_uri())
    module = _load_module(
        "supply_catalog_hdf5_test",
        ROOT / "services/supply-oracle-svc/src/supply_oracle_svc/main.py",
    )
    service = module.SupplyOracleServicer()

    response = await service.CheckAvailability(SimpleNamespace(smiles="CCO"), None)

    assert response.available is True
    assert response.catalog_id == inchi_key
    assert response.catalog_source == "aizynth_stock"
    assert response.source_timestamp
    assert response.price is None


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


def test_deployment_declares_remaining_runtime_config_data() -> None:
    k8s = (ROOT / "infra/kubernetes/deployments/moleculeforge-services.yaml").read_text(
        encoding="utf-8"
    )
    helm_values = (ROOT / "infra/helm/moleculeforge/values.yaml").read_text(
        encoding="utf-8"
    )

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
        assert config["sign-command"] == ""
        assert config["verify-command"] == ""
        assert config["expected-identity"] == ""
        assert config["rekor-url"] == "https://rekor.sigstore.dev"
        assert config["command-timeout-seconds"] == "30"

    for secret in (
        _k8s_secret_string_data(k8s, "mf-agents", "sigstore-provenance"),
        _helm_secret_string_data(helm_values, "mf-agents", "sigstore-provenance"),
    ):
        assert secret["identity-token"] == ""


@pytest.mark.asyncio
async def test_supply_agent_aggregates_catalog_availability() -> None:
    module = _load_module(
        "supply_agent_catalog_aggregation_test",
        ROOT / "agents/supply_agent/src/supply_agent/agent.py",
    )

    class CatalogClient:
        def __init__(self) -> None:
            self.calls: list[str] = []

        async def check_availability(self, smiles: str) -> dict:
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
                },
            }
            return records[smiles]

    catalog_client = CatalogClient()
    agent = module.SupplyAgent(supply_client=catalog_client)

    result = await agent.process(
        {
            "smiles": "CCOON",
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


@pytest.mark.asyncio
async def test_supply_agent_persists_supply_feasibility_belief() -> None:
    module = _load_module(
        "supply_agent_crg_repository_test",
        ROOT / "agents/supply_agent/src/supply_agent/agent.py",
    )

    class CatalogClient:
        async def check_availability(self, smiles: str) -> dict:
            return {
                "smiles": smiles,
                "available": True,
                "catalog_id": "CAT-1",
                "source": "local_catalog",
                "source_timestamp": "2026-05-01T00:00:00Z",
                "price": 10.0,
                "currency": "USD",
                "lead_time_days": 2,
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
            "smiles": "CCO",
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
    assert belief["evidence_ids"] == ["CAT-1"]


@pytest.mark.asyncio
async def test_supply_agent_uses_zero_retrosyn_routes_belief_from_shared_crg() -> None:
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

    repository = CRGRepository()
    agent = module.SupplyAgent(supply_client=None, crg_repository=repository)

    result = await agent.process(
        {
            "project_id": "project-1",
            "run_id": "run-1",
            "smiles": "CCO",
            "building_blocks": ["CCO"],
        }
    )

    assert result["status"] == "assessed"
    assert result["supply_assessment"]["overall_feasibility"] == "unavailable"
    assert result["block_assessments"][0]["catalog_id"] == "crg_retrosyn_routes"
    assert repository.beliefs[0]["predicate"] == "supply_feasibility"
    assert repository.beliefs[0]["object_value"] == "unavailable"
    assert repository.beliefs[0]["evidence_ids"] == ["crg_retrosyn_routes"]


@pytest.mark.asyncio
async def test_supply_agent_uses_existing_supply_feasibility_from_shared_crg() -> None:
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

    repository = CRGRepository()
    agent = module.SupplyAgent(supply_client=None, crg_repository=repository)

    result = await agent.process(
        {
            "project_id": "project-1",
            "run_id": "run-1",
            "smiles": "CCO",
            "building_blocks": ["CCO", "CCN"],
        }
    )

    assert result["status"] == "assessed"
    assert result["cache_source"] == "shared_crg"
    assert result["supply_assessment"]["overall_feasibility"] == "available"
    assert result["supply_assessment"]["commercially_available"] == 2
    assert result["block_assessments"][0]["catalog_id"] == "crg_supply_feasibility"
    assert repository.beliefs == []


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
        await agent.process({"smiles": "CCO", "building_blocks": ["CCO"]})


def test_fep_runtime_uses_openfe_executable_without_importing_python_package(
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

    assert status == [
        {
            "name": "openfe_runner",
            "configured": True,
            "available": True,
            "required": True,
            "path": str(runner),
            "source": "OPENFE_RUNNER_PATH",
            "message": "openfe_runner executable is available",
        }
    ]


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
        "'total_elapsed_ms': 33, "
        "'results': [{"
        "'ligand_a_smiles': 'CCO', "
        "'ligand_b_smiles': 'CCN', "
        "'ddg_kcal_mol': -1.2, "
        "'ddg_uncertainty': 0.3, "
        "'n_repeats': 2, "
        "'method': 'openfe', "
        "'per_repeat_ddg': {'repeat_1': -1.1}, "
        "'converged': True"
        "}]"
        "}))\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("FEP_ORACLE_COMMAND", f"{sys.executable} {runner}")
    request = fep_pb2.FEPBatchRequest(
        project_id="project-1",
        protein_pdb_id="7abc",
        reference_ligand_smiles="CCO",
        test_ligand_smiles=["CCN"],
        method="openfe",
        n_repeats=2,
    )

    response = await module.FEPServicer().RunFEP(request, None)

    assert response.batch_id == "project-1"
    assert response.total_elapsed_ms == 33
    assert response.results[0].ligand_a_smiles == "CCO"
    assert response.results[0].ligand_b_smiles == "CCN"
    assert response.results[0].ddg_kcal_mol == pytest.approx(-1.2)
    assert response.results[0].ddg_uncertainty == pytest.approx(0.3)
    assert response.results[0].per_repeat_ddg["repeat_1"] == pytest.approx(-1.1)
    assert response.results[0].converged is True


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
        protein_pdb_id="7abc",
        reference_ligand_smiles="CCO",
        test_ligand_smiles=["CCN"],
        method="openfe",
        n_repeats=1,
    )

    service = module.FEPServicer()
    submitted = await service.SubmitFEP(request, None)

    assert submitted.job_id
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
    assert status.response.batch_id == "project-async"
    assert status.response.total_elapsed_ms == 44
    assert status.response.results[0].ddg_kcal_mol == pytest.approx(-1.4)
    assert status.response.results[0].converged is True


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
                        converged=True,
                    )
                ],
                batch_id=request.project_id,
                total_elapsed_ms=33,
            )

    monkeypatch.setenv("FEP_REFERENCE_LIGAND_SMILES", "CCO")
    service = FEPService()
    oracle = module.FEPOracleServicer(service=service)

    response = await oracle.PredictWithUncertainty(
        oracle_pb2.OracleBatchRequest(
            project_id="project-1",
            molecule_smiles=["CCN"],
            requested_properties=["rbfe"],
            level=oracle_pb2.L3_FEP,
            return_uncertainty=True,
        ),
        None,
    )

    assert service.requests[0].reference_ligand_smiles == "CCO"
    assert list(service.requests[0].test_ligand_smiles) == ["CCN"]
    assert response.batch_id == "project-1"
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


def test_iclm_deployment_wires_model_and_update_runner_env() -> None:
    compose = (ROOT / "infra/docker/docker-compose.dev.yml").read_text(encoding="utf-8")
    k8s = (ROOT / "infra/kubernetes/deployments/moleculeforge-services.yaml").read_text(
        encoding="utf-8"
    )
    helm_values = (ROOT / "infra/helm/moleculeforge/values.yaml").read_text(
        encoding="utf-8"
    )

    for env_name in (
        "ICLM_MODEL_PATH",
        "ICLM_DEVICE",
        "ICLM_UPDATE_COMMAND",
        "ICLM_UPDATE_TIMEOUT_SECONDS",
    ):
        assert env_name in compose
        assert env_name in k8s
        assert env_name in helm_values

    assert "ICLM_DEVICE: ${ICLM_DEVICE:-cpu}" in compose
    assert (
        "ICLM_MODEL_PATH: "
        "${ICLM_MODEL_PATH:-models/artifacts/iclm/novomolgen_157m_smiles_bpe}"
    ) in compose
    assert (ROOT / "models/artifacts/iclm/novomolgen_157m_smiles_bpe").is_dir()
    assert (ROOT / "models/artifacts/iclm/novomolgen_157m_smiles_bpe/model.safetensors").is_file()
    assert "ICLM_UPDATE_TIMEOUT_SECONDS: ${ICLM_UPDATE_TIMEOUT_SECONDS:-300}" in compose
    assert "name: iclm-generator-config" in k8s
    assert "configMapKeyRef:" in k8s
    assert "envValueFrom:" in helm_values
    for config in (
        _k8s_configmap_data(k8s, "mf-generators", "iclm-generator-config"),
        _helm_configmap_data(helm_values, "mf-generators", "iclm-generator-config"),
    ):
        assert config["model-path"] == "models/artifacts/iclm/novomolgen_157m_smiles_bpe"
        assert config["device"] == "cpu"
        assert config["update-command"] == ""
        assert config["update-timeout-seconds"] == "300"


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


def test_iclm_update_command_preflight_rejects_missing_executable(
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
        module._run_update_command(
            SimpleNamespace(project_id="project-iclm", training_samples=["CCO"])
        )


def test_generator_router_deployment_wires_hypseek_teacher_env() -> None:
    import yaml

    compose = (ROOT / "infra/docker/docker-compose.dev.yml").read_text(encoding="utf-8")
    compose_config = yaml.safe_load(compose)
    k8s = (ROOT / "infra/kubernetes/deployments/moleculeforge-services.yaml").read_text(
        encoding="utf-8"
    )
    k8s_docs = list(yaml.safe_load_all(k8s))
    helm_values = (ROOT / "infra/helm/moleculeforge/values.yaml").read_text(
        encoding="utf-8"
    )
    helm_config = yaml.safe_load(helm_values)
    helm_template = (ROOT / "infra/helm/moleculeforge/templates/services.yaml").read_text(
        encoding="utf-8"
    )

    for env_name in (
        "HYPSEEK_TEACHER_URL",
        "HYPSEEK_TEACHER_COMMAND",
        "HYPSEEK_TEACHER_TIMEOUT_SECONDS",
    ):
        assert env_name in compose
        assert env_name in k8s
        assert env_name in helm_values

    assert "HYPSEEK_TEACHER_URL: http://hypseek-teacher-svc:8012/teacher" in compose
    assert (
        "HYPSEEK_TEACHER_TIMEOUT_SECONDS: ${HYPSEEK_TEACHER_TIMEOUT_SECONDS:-60}"
        in compose
    )
    assert "name: hypseek-teacher-config" in k8s
    assert "envValueFrom:" in helm_values

    compose_healthcheck = compose_config["services"]["hypseek-teacher-svc"]["healthcheck"]
    assert "http://localhost:8012/healthz" in " ".join(compose_healthcheck["test"])
    hypseek_deployment = next(
        item
        for item in k8s_docs
        if item
        and item.get("kind") == "Deployment"
        and item.get("metadata", {}).get("name") == "hypseek-teacher-svc"
    )
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


def test_generator_router_runtime_rejects_missing_external_commands(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HYPSEEK_TEACHER_COMMAND", "missing-hypseek-teacher --json")
    monkeypatch.setenv("TAR_PROXYLESS_SEARCH_COMMAND", "missing-tar-search --json")
    module = _load_module(
        "generator_router_missing_external_commands_test",
        ROOT / "services/generator-router-svc/src/generator_router_svc/main.py",
    )

    status = module.runtime_status()

    hypseek_status = next(item for item in status if item["name"] == "hypseek_teacher_command")
    tar_status = next(item for item in status if item["name"] == "tar_proxyless_search_command")
    assert hypseek_status["configured"] is True
    assert hypseek_status["available"] is False
    assert hypseek_status["source"] == "HYPSEEK_TEACHER_COMMAND"
    assert "not found" in hypseek_status["message"]
    assert tar_status["configured"] is True
    assert tar_status["available"] is False
    assert tar_status["source"] == "TAR_PROXYLESS_SEARCH_COMMAND"
    assert "not found" in tar_status["message"]


def test_generator_router_external_command_preflight_rejects_missing_executable() -> None:
    module = _load_module(
        "generator_router_external_command_preflight_test",
        ROOT / "services/generator-router-svc/src/generator_router_svc/main.py",
    )

    with pytest.raises(RuntimeError, match="not found"):
        module._hypseek_feedback_from_command(
            "missing-hypseek-teacher --json",
            generator_name="hfm_3d",
            reward=0.7,
            oracle_feedback=[],
        )
    with pytest.raises(RuntimeError, match="not found"):
        module._proxyless_search_from_command(
            "missing-tar-search --json",
            {
                "reward_batches_by_dataset": {"dataset": []},
                "generator_costs": {},
                "cost_weight": 0.0,
                "learning_rate": 0.1,
                "temperature": 1.0,
            },
            timeout_seconds=1.0,
        )


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
        "payload = json.load(sys.stdin)\n"
        "assert payload['model_path'].endswith('iclm_model')\n"
        "assert payload['device'] == 'cpu'\n"
        "assert payload['project_id'] == 'project-iclm'\n"
        "assert payload['training_samples'] == ['CCO', 'CCN']\n"
        "assert payload['kd_teacher_embeddings'] == [[0.1, 0.2, 0.3, 0.4]]\n"
        "assert payload['kd_weight'] == 0.25\n"
        "print(json.dumps({"
        "'checkpoint_path': payload['model_path'] + '/updated', "
        "'updated_samples': 2, "
        "'ewc_loss': 0.125, "
        "'kd_loss': 0.375, "
        "'metadata': {'teacher': 'hypseek'}"
        "}))\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("ICLM_MODEL_PATH", str(model_path))
    monkeypatch.setenv("ICLM_UPDATE_COMMAND", f"{sys.executable} {runner}")

    response = await module.ICLMServicer(generator=object()).UpdateModel(
        SimpleNamespace(
            project_id="project-iclm",
            training_samples=["CCO", "CCN"],
            kd_teacher_embeddings=[[0.1, 0.2, 0.3, 0.4]],
            kd_weight=0.25,
        ),
        None,
    )

    assert response.checkpoint_path == str(model_path / "updated")
    assert response.updated_samples == 2
    assert response.ewc_loss == pytest.approx(0.125)
    assert response.kd_loss == pytest.approx(0.375)
    assert response.metadata == {"teacher": "hypseek"}


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
    calls: list[dict] = []

    class Learner:
        def update(self, payload):
            calls.append(payload)
            return {
                "checkpoint_path": str(model_path / "online-updated"),
                "updated_samples": 2,
                "ewc_loss": 0.2,
                "kd_loss": 0.4,
                "metadata": {"mode": "online_learner"},
            }

    response = await module.ICLMServicer(
        generator=SimpleNamespace(online_learner=Learner())
    ).UpdateModel(
        SimpleNamespace(
            project_id="project-iclm",
            training_samples=["CCO", "CCN"],
            kd_teacher_embeddings=[[0.1, 0.2]],
            kd_weight=0.5,
        ),
        None,
    )

    assert calls == [
        {
            "project_id": "project-iclm",
            "training_samples": ["CCO", "CCN"],
            "kd_teacher_embeddings": [[0.1, 0.2]],
            "kd_weight": 0.5,
        }
    ]
    assert response.checkpoint_path == str(model_path / "online-updated")
    assert response.updated_samples == 2
    assert response.ewc_loss == pytest.approx(0.2)
    assert response.kd_loss == pytest.approx(0.4)
    assert response.metadata == {"mode": "online_learner"}


@pytest.mark.asyncio
async def test_iclm_service_update_model_reports_online_learner_kd_metrics(
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

    class Learner:
        last_task_loss = 0.25
        last_kd_loss = 4.0

        def update(self, payload):
            assert payload["kd_weight"] == 0.5
            return 2.25

    response = await module.ICLMServicer(
        generator=SimpleNamespace(
            checkpoint_path=str(model_path),
            online_learner=Learner(),
        )
    ).UpdateModel(
        SimpleNamespace(
            project_id="project-iclm",
            training_samples=["CCO"],
            kd_teacher_embeddings=[[0.0]],
            kd_weight=0.5,
        ),
        None,
    )

    assert response.checkpoint_path == str(model_path)
    assert response.ewc_loss == pytest.approx(0.25)
    assert response.kd_loss == pytest.approx(4.0)
    assert response.metadata == {"mode": "online_learner"}


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

    assert started["status"] == "escalated"
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
async def test_orchestrator_workflow_runs_supply_and_srb_hooks_after_retrosyn() -> None:
    module = _load_module(
        "orchestrator_supply_srb_hooks_test",
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
            return {"supply_assessment": {"overall_feasibility": "available"}}

        async def compile_synthesis(self, state):
            assert state["supply"]["supply_assessment"]["overall_feasibility"] == "available"
            return {"status": "compiled", "protocols": [{"ssp_id": "ssp-1"}]}

        async def review_candidates(self, state):
            assert state["srb"]["protocols"][0]["ssp_id"] == "ssp-1"
            return {"verdict": "pass", "total_rules": 1}

    started = await module.start_design(
        {
            "nl_input": "Design KRAS G12C inhibitor",
            "workflow_scope": "engineering",
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
    ]
    assert started["state"]["supply"]["supply_assessment"]["overall_feasibility"] == "available"
    assert started["state"]["srb"]["protocols"][0]["ssp_id"] == "ssp-1"


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
        "RETROSYN",
        "CRITIC",
    ]
    assert all(belief["subject"] == "run-orch-crg-1" for belief in beliefs)
    assert all(belief["predicate"] == "workflow_stage" for belief in beliefs)
    assert all(belief["source_agent"] == "orchestrator" for belief in beliefs)
    assert [edge["relation"] for edge in edges] == [
        "derives_from",
        "derives_from",
        "derives_from",
        "derives_from",
    ]
    assert edges[0]["source_belief_id"] == beliefs[0]["id"]
    assert edges[0]["target_belief_id"] == beliefs[1]["id"]


@pytest.mark.asyncio
async def test_orchestrator_full_workflow_uses_runtime_clients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module(
        "orchestrator_full_workflow_test",
        ROOT / "services/orchestrator-svc/src/orchestrator_svc/main.py",
    )

    class Clients:
        async def compile_intent(self, state):
            return {"cig": {"source": state["nl_input"]}, "hciv": {}, "intent_cone": {}}

        async def generate_candidates(self, state):
            return [{"smiles": "CCO", "canonical_smiles": "CCO"}]

        async def validate_candidates(self, state):
            return {
                "passed": True,
                "results": [{"smiles": "CCO", "delta_g_kcal_mol": -8.0}],
            }

        async def plan_routes(self, state):
            return {"skipped": False, "routes": [{"route_id": "route-1"}]}

        async def assess_supply(self, state):
            return {"supply_assessment": {"overall_feasibility": "available"}}

        async def compile_synthesis(self, state):
            return {"status": "compiled", "protocols": [{"ssp_id": "ssp-1"}]}

        async def review_candidates(self, state):
            return {"verdict": "pass", "total_rules": 1}

    monkeypatch.setattr(module, "FullWorkflowClients", Clients, raising=False)

    started = await module.start_design(
        {
            "nl_input": "Design KRAS G12C inhibitor",
            "workflow_scope": "full",
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
    ]
    assert started["state"]["candidates"][0]["canonical_smiles"] == "CCO"
    assert started["state"]["validation"]["results"][0]["delta_g_kcal_mol"] == -8.0
    assert started["state"]["retrosyn"]["routes"][0]["route_id"] == "route-1"
    assert started["state"]["critic"]["verdict"] == "pass"


@pytest.mark.asyncio
async def test_full_workflow_generator_receives_intent_cone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module(
        "orchestrator_full_workflow_intent_cone_test",
        ROOT / "services/orchestrator-svc/src/orchestrator_svc/main.py",
    )
    calls: list[object] = []

    class HFMGeneratorServicer:
        async def Generate(self, request, context):
            calls.append(request)
            return SimpleNamespace(
                molecules=[
                    json.dumps(
                        {
                            "smiles": "CCO",
                            "canonical_smiles": "CCO",
                        }
                    ).encode("utf-8")
                ]
            )

    fake_hfm_module = ModuleType("hfm_generator_svc.main")
    fake_hfm_module.HFMGeneratorServicer = HFMGeneratorServicer
    monkeypatch.setitem(sys.modules, "hfm_generator_svc.main", fake_hfm_module)

    cone = {"axis": [1.0] + [0.0] * 128, "half_angle": 0.25}
    candidates = await module.FullWorkflowClients().generate_candidates(
        {
            "run_id": "run-full-intent",
            "intent_cone": cone,
            "request": {"n_samples": 1, "seed": 7},
        }
    )

    assert candidates[0]["canonical_smiles"] == "CCO"
    assert calls[0].intent_cone == cone
    assert calls[0].generator_params["sampling_seed"] == 7


@pytest.mark.asyncio
async def test_full_workflow_generator_receives_generation_feedback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module(
        "orchestrator_full_workflow_generation_feedback_test",
        ROOT / "services/orchestrator-svc/src/orchestrator_svc/main.py",
    )
    calls: list[object] = []

    class HFMGeneratorServicer:
        async def Generate(self, request, context):
            calls.append(request)
            return SimpleNamespace(
                molecules=[
                    json.dumps(
                        {
                            "smiles": "CCO",
                            "canonical_smiles": "CCO",
                        }
                    ).encode("utf-8")
                ]
            )

    fake_hfm_module = ModuleType("hfm_generator_svc.main")
    fake_hfm_module.HFMGeneratorServicer = HFMGeneratorServicer
    monkeypatch.setitem(sys.modules, "hfm_generator_svc.main", fake_hfm_module)

    feedback = [
        {
            "source": "validation",
            "reason": "affinity gate failed",
            "passed": False,
            "evidence_ids": "validation-belief-1",
        }
    ]
    candidates = await module.FullWorkflowClients().generate_candidates(
        {
            "run_id": "run-full-feedback",
            "intent_cone": {"axis": [1.0] + [0.0] * 128, "half_angle": 0.25},
            "generation_feedback": feedback,
            "request": {"n_samples": 1, "seed": 7},
        }
    )

    assert candidates[0]["canonical_smiles"] == "CCO"
    assert calls[0].generator_params["sampling_seed"] == 7
    assert json.loads(calls[0].generator_params["generation_feedback"]) == feedback
    jmcg_feedback = json.loads(calls[0].generator_params["jmcg_feedback"])
    assert jmcg_feedback["schema"] == "moleculeforge.jmcg.feedback.v1"
    assert jmcg_feedback["run_id"] == "run-full-feedback"
    assert [record["kind"] for record in jmcg_feedback["records"]] == [
        "intent",
        "property",
    ]
    assert len(jmcg_feedback["records"][0]["humu_embedding"]) == 129
    assert (
        jmcg_feedback["records"][0]["metadata"]["embedding_source"]
        == "intent_cone.axis"
    )
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
async def test_full_workflow_generator_receives_non_steering_intent_and_pocket_feedback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module(
        "orchestrator_full_workflow_intent_pocket_feedback_test",
        ROOT / "services/orchestrator-svc/src/orchestrator_svc/main.py",
    )
    calls: list[object] = []

    class HFMGeneratorServicer:
        async def Generate(self, request, context):
            calls.append(request)
            return SimpleNamespace(
                molecules=[
                    json.dumps(
                        {
                            "smiles": "CCO",
                            "canonical_smiles": "CCO",
                        }
                    ).encode("utf-8")
                ]
            )

    fake_hfm_module = ModuleType("hfm_generator_svc.main")
    fake_hfm_module.HFMGeneratorServicer = HFMGeneratorServicer
    monkeypatch.setitem(sys.modules, "hfm_generator_svc.main", fake_hfm_module)

    candidates = await module.FullWorkflowClients().generate_candidates(
        {
            "run_id": "run-intent-pocket-feedback",
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
    jmcg_feedback = json.loads(calls[0].generator_params["jmcg_feedback"])
    assert [record["kind"] for record in jmcg_feedback["records"]] == [
        "intent",
        "pocket",
    ]
    assert jmcg_feedback["records"][0]["subject"] == {
        "type": "intent",
        "id": "run-intent-pocket-feedback",
    }
    assert len(jmcg_feedback["records"][0]["humu_embedding"]) == 129
    assert (
        jmcg_feedback["records"][0]["metadata"]["embedding_source"]
        == "intent_cone.axis"
    )
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
    calls: list[object] = []

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

    class HFMGeneratorServicer:
        async def Generate(self, request, context):
            calls.append(request)
            return SimpleNamespace(
                molecules=[
                    json.dumps(
                        {
                            "smiles": "CCO",
                            "canonical_smiles": "CCO",
                        }
                    ).encode("utf-8")
                ]
            )

    fake_hfm_module = ModuleType("hfm_generator_svc.main")
    fake_hfm_module.HFMGeneratorServicer = HFMGeneratorServicer
    monkeypatch.setitem(sys.modules, "hfm_generator_svc.main", fake_hfm_module)
    monkeypatch.setattr(module, "_encode_pocket_humu_feedback", encode_pocket)

    candidates = await module.FullWorkflowClients().generate_candidates(
        {
            "run_id": "run-pocket-embedding",
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
    jmcg_feedback = json.loads(calls[0].generator_params["jmcg_feedback"])
    pocket_record = next(
        record for record in jmcg_feedback["records"] if record["kind"] == "pocket"
    )
    assert len(pocket_record["humu_embedding"]) == 129
    assert pocket_record["curvature"] == 1.0
    assert pocket_record["source"] == "humu_encoder_svc"
    assert pocket_record["evidence_ids"] == ["pocket-geometry"]
    assert pocket_record["metadata"]["pocket_id"] == "switch-ii"


@pytest.mark.asyncio
async def test_full_workflow_metadata_only_pocket_feedback_stays_non_steering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module(
        "orchestrator_full_workflow_metadata_only_pocket_feedback_test",
        ROOT / "services/orchestrator-svc/src/orchestrator_svc/main.py",
    )
    calls: list[object] = []

    class HFMGeneratorServicer:
        async def Generate(self, request, context):
            calls.append(request)
            return SimpleNamespace(
                molecules=[
                    json.dumps(
                        {
                            "smiles": "CCO",
                            "canonical_smiles": "CCO",
                        }
                    ).encode("utf-8")
                ]
            )

    fake_hfm_module = ModuleType("hfm_generator_svc.main")
    fake_hfm_module.HFMGeneratorServicer = HFMGeneratorServicer
    monkeypatch.setitem(sys.modules, "hfm_generator_svc.main", fake_hfm_module)

    await module.FullWorkflowClients().generate_candidates(
        {
            "run_id": "run-pocket-metadata-only",
            "cig": {
                "target_context": {
                    "pdb_id": "6OIM",
                    "pocket_id": "switch-ii",
                },
            },
            "request": {"n_samples": 1, "seed": 7},
        }
    )

    jmcg_feedback = json.loads(calls[0].generator_params["jmcg_feedback"])
    pocket_record = next(
        record for record in jmcg_feedback["records"] if record["kind"] == "pocket"
    )
    assert "humu_embedding" not in pocket_record
    assert pocket_record["metadata"] == {
        "pdb_id": "6OIM",
        "pocket_id": "switch-ii",
    }


@pytest.mark.asyncio
async def test_full_workflow_intent_axis_embedding_becomes_steering_capable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module(
        "orchestrator_full_workflow_intent_axis_embedding_test",
        ROOT / "services/orchestrator-svc/src/orchestrator_svc/main.py",
    )
    calls: list[object] = []

    class HFMGeneratorServicer:
        async def Generate(self, request, context):
            calls.append(request)
            return SimpleNamespace(
                molecules=[
                    json.dumps(
                        {
                            "smiles": "CCO",
                            "canonical_smiles": "CCO",
                        }
                    ).encode("utf-8")
                ]
            )

    fake_hfm_module = ModuleType("hfm_generator_svc.main")
    fake_hfm_module.HFMGeneratorServicer = HFMGeneratorServicer
    monkeypatch.setitem(sys.modules, "hfm_generator_svc.main", fake_hfm_module)

    axis = [1.0] + [0.0] * 128
    await module.FullWorkflowClients().generate_candidates(
        {
            "run_id": "run-intent-axis",
            "intent_cone": {"axis": axis, "half_angle": 0.25},
            "request": {"n_samples": 1, "seed": 7},
        }
    )

    jmcg_feedback = json.loads(calls[0].generator_params["jmcg_feedback"])
    intent_record = next(
        record for record in jmcg_feedback["records"] if record["kind"] == "intent"
    )
    assert intent_record["humu_embedding"] == axis
    assert intent_record["metadata"]["embedding_source"] == "intent_cone.axis"


@pytest.mark.asyncio
async def test_full_workflow_invalid_lorentz_intent_axis_stays_non_steering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module(
        "orchestrator_full_workflow_invalid_lorentz_intent_axis_test",
        ROOT / "services/orchestrator-svc/src/orchestrator_svc/main.py",
    )
    calls: list[object] = []

    class HFMGeneratorServicer:
        async def Generate(self, request, context):
            calls.append(request)
            return SimpleNamespace(
                molecules=[
                    json.dumps(
                        {
                            "smiles": "CCO",
                            "canonical_smiles": "CCO",
                        }
                    ).encode("utf-8")
                ]
            )

    fake_hfm_module = ModuleType("hfm_generator_svc.main")
    fake_hfm_module.HFMGeneratorServicer = HFMGeneratorServicer
    monkeypatch.setitem(sys.modules, "hfm_generator_svc.main", fake_hfm_module)

    await module.FullWorkflowClients().generate_candidates(
        {
            "run_id": "run-invalid-intent-axis",
            "intent_cone": {"axis": [0.0] * 129, "half_angle": 0.25},
            "request": {"n_samples": 1, "seed": 7},
        }
    )

    jmcg_feedback = json.loads(calls[0].generator_params["jmcg_feedback"])
    intent_record = next(
        record for record in jmcg_feedback["records"] if record["kind"] == "intent"
    )
    assert "humu_embedding" not in intent_record


@pytest.mark.asyncio
async def test_full_workflow_hciv_vector_does_not_become_humu_embedding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module(
        "orchestrator_full_workflow_hciv_non_embedding_test",
        ROOT / "services/orchestrator-svc/src/orchestrator_svc/main.py",
    )
    calls: list[object] = []

    class HFMGeneratorServicer:
        async def Generate(self, request, context):
            calls.append(request)
            return SimpleNamespace(
                molecules=[
                    json.dumps(
                        {
                            "smiles": "CCO",
                            "canonical_smiles": "CCO",
                        }
                    ).encode("utf-8")
                ]
            )

    fake_hfm_module = ModuleType("hfm_generator_svc.main")
    fake_hfm_module.HFMGeneratorServicer = HFMGeneratorServicer
    monkeypatch.setitem(sys.modules, "hfm_generator_svc.main", fake_hfm_module)

    await module.FullWorkflowClients().generate_candidates(
        {
            "run_id": "run-hciv-non-embedding",
            "hciv": {"coordinates": [0.0] * 128, "curvature": 1.0},
            "intent_cone": {"axis": [0.0] * 128, "half_angle": 0.25},
            "request": {"n_samples": 1, "seed": 7},
        }
    )

    jmcg_feedback = json.loads(calls[0].generator_params["jmcg_feedback"])
    intent_record = next(
        record for record in jmcg_feedback["records"] if record["kind"] == "intent"
    )
    assert "humu_embedding" not in intent_record
    assert intent_record["metadata"]["has_hciv"] is True


@pytest.mark.asyncio
async def test_full_workflow_generation_delegates_to_generator_coord_strategy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module(
        "orchestrator_full_workflow_generator_coord_test",
        ROOT / "services/orchestrator-svc/src/orchestrator_svc/main.py",
    )
    calls: list[dict] = []

    class GeneratorCoordAgent:
        async def process(self, payload):
            calls.append(payload)
            return {
                "status": "dispatched",
                "selected_generators": ["hfm_3d", "fragfm"],
                "candidates": [{"smiles": "CCN"}],
            }

    fake_generator_coord_module = ModuleType("generator_coord.agent")
    fake_generator_coord_module.GeneratorCoordAgent = GeneratorCoordAgent
    monkeypatch.setitem(sys.modules, "generator_coord.agent", fake_generator_coord_module)

    cone = {"axis": [1.0] + [0.0] * 128, "half_angle": 0.25}
    candidates = await module.FullWorkflowClients().generate_candidates(
        {
            "run_id": "run-generator-coord",
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
    assert calls == [
        {
            "project_id": "project-1",
            "run_id": "run-generator-coord",
            "request_id": "run-generator-coord",
            "generation_strategy": "auto",
            "objectives": {"complexity": "high"},
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
    ]


@pytest.mark.asyncio
async def test_full_workflow_generator_coord_receives_generation_feedback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module(
        "orchestrator_full_workflow_generator_coord_feedback_test",
        ROOT / "services/orchestrator-svc/src/orchestrator_svc/main.py",
    )
    calls: list[dict] = []

    class GeneratorCoordAgent:
        async def process(self, payload):
            calls.append(payload)
            return {
                "status": "dispatched",
                "selected_generators": ["hfm_3d"],
                "candidates": [{"smiles": "CCN"}],
            }

    fake_generator_coord_module = ModuleType("generator_coord.agent")
    fake_generator_coord_module.GeneratorCoordAgent = GeneratorCoordAgent
    monkeypatch.setitem(sys.modules, "generator_coord.agent", fake_generator_coord_module)

    feedback = [
        {
            "source": "critic",
            "verdict": "fail",
            "reason": "synthetic accessibility failed",
        }
    ]
    candidates = await module.FullWorkflowClients().generate_candidates(
        {
            "run_id": "run-generator-coord-feedback",
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
    generator_params = calls[0]["generator_params"]
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
async def test_full_workflow_validation_applies_affinity_quality_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module(
        "orchestrator_full_workflow_quality_gate_test",
        ROOT / "services/orchestrator-svc/src/orchestrator_svc/main.py",
    )
    from mf_core.proto_gen.moleculeforge.v1.oracle import boltz2_pb2

    class Boltz2Servicer:
        async def PredictAffinity(self, request, context):
            return boltz2_pb2.Boltz2BatchResponse(
                protein_pdb_id=request.protein_pdb_id,
                affinities=[
                    boltz2_pb2.Boltz2BindingAffinity(
                        protein_pdb_id=request.protein_pdb_id,
                        ligand_smiles=request.ligand_smiles[0],
                        delta_g_kcal_mol=-4.7,
                        uncertainty=1.5,
                        ki_nm=316154.0,
                        ensemble_size=request.ensemble_size,
                    )
                ],
            )

    fake_boltz_module = ModuleType("boltz2_svc.main")
    fake_boltz_module.Boltz2Servicer = Boltz2Servicer
    monkeypatch.setitem(sys.modules, "boltz2_svc.main", fake_boltz_module)

    state = {
        "run_id": "run-quality-gate",
        "candidates": [{"canonical_smiles": "CCO"}],
        "request": {
            "protein_pdb_id": "6OIM",
            "boltz_ensemble_size": 1,
            "boltz_max_ki_nm": 10.0,
        },
    }

    result = await module.FullWorkflowClients().validate_candidates(state)

    assert result["passed"] is False
    assert result["quality_gate"]["max_ki_nm"] == 10.0
    assert result["results"][0]["passes_affinity_gate"] is False


@pytest.mark.asyncio
async def test_full_workflow_validation_delegates_to_validation_agent_for_oracle_cascade(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module(
        "orchestrator_full_workflow_validation_agent_test",
        ROOT / "services/orchestrator-svc/src/orchestrator_svc/main.py",
    )
    calls: list[dict] = []

    class ValidationAgent:
        async def process(self, payload):
            calls.append(payload)
            return {
                "status": "validated",
                "overall_passed": True,
                "max_oracle_level": payload["oracle_level"],
                "cascade": {
                    "L4_quantum": {
                        "completed": True,
                        "passed": True,
                        "result": {"quantum_correction": -0.1},
                    }
                },
                "upgrade_path": ["L0", "L1", "L2", "L3", "L4"],
            }

    fake_validation_module = ModuleType("validation_agent.agent")
    fake_validation_module.ValidationAgent = ValidationAgent
    monkeypatch.setitem(sys.modules, "validation_agent.agent", fake_validation_module)

    result = await module.FullWorkflowClients().validate_candidates(
        {
            "run_id": "run-validation-cascade",
            "candidates": [{"canonical_smiles": "CCO"}],
            "request": {
                "project_id": "project-1",
                "oracle_level": 4,
                "l4_max_quantum_correction": 0.0,
            },
        }
    )

    assert result["passed"] is True
    assert result["validation_mode"] == "adaptive_oracle_cascade"
    assert result["results"][0]["cascade"]["L4_quantum"]["result"] == {
        "quantum_correction": -0.1
    }
    assert calls == [
        {
            "project_id": "project-1",
            "run_id": "run-validation-cascade",
            "smiles": "CCO",
            "oracle_level": 4,
            "l4_max_quantum_correction": 0.0,
        }
    ]


@pytest.mark.asyncio
async def test_full_workflow_clients_plan_routes_delegates_to_retrosyn_agent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module(
        "orchestrator_full_retrosyn_client_test",
        ROOT / "services/orchestrator-svc/src/orchestrator_svc/main.py",
    )
    calls: list[dict] = []

    class RetroSynAgent:
        async def process(self, payload):
            calls.append(payload)
            return {"status": "planned", "routes": [{"route_id": "route-1"}]}

    fake_retrosyn_module = ModuleType("retrosyn_agent.agent")
    fake_retrosyn_module.RetroSynAgent = RetroSynAgent
    monkeypatch.setitem(sys.modules, "retrosyn_agent.agent", fake_retrosyn_module)
    monkeypatch.delenv("AIZYNTH_CONFIG_PATH", raising=False)
    state = {
        "run_id": "run-1",
        "request": {"project_id": "project-1", "retrosyn_max_routes": 2},
        "candidates": [{"canonical_smiles": "CCO"}],
    }

    result = await module.FullWorkflowClients().plan_routes(state)

    assert result["routes"][0]["route_id"] == "route-1"
    assert calls == [
        {
            "project_id": "project-1",
            "run_id": "run-1",
            "smiles": "CCO",
            "max_routes": 2,
        }
    ]


@pytest.mark.asyncio
async def test_full_workflow_clients_assess_supply_delegates_to_supply_agent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module(
        "orchestrator_full_supply_client_test",
        ROOT / "services/orchestrator-svc/src/orchestrator_svc/main.py",
    )
    calls: list[dict] = []

    class SupplyAgent:
        async def process(self, payload):
            calls.append(payload)
            return {"status": "assessed", "supply_assessment": {"overall_feasibility": "available"}}

    fake_supply_module = ModuleType("supply_agent.agent")
    fake_supply_module.SupplyAgent = SupplyAgent
    monkeypatch.setitem(sys.modules, "supply_agent.agent", fake_supply_module)
    state = {
        "run_id": "run-1",
        "request": {"project_id": "project-1"},
        "candidates": [{"canonical_smiles": "CCO"}],
        "retrosyn": {
            "routes": [
                {
                    "route_id": "route-1",
                    "building_blocks": [{"smiles": "CC"}, {"smiles": "CO"}],
                }
            ]
        },
    }

    result = await module.FullWorkflowClients().assess_supply(state)

    assert result["supply_assessment"]["overall_feasibility"] == "available"
    assert calls == [
        {
            "project_id": "project-1",
            "run_id": "run-1",
            "smiles": "CCO",
            "building_blocks": [{"smiles": "CC"}, {"smiles": "CO"}],
        }
    ]


@pytest.mark.asyncio
async def test_full_workflow_clients_assess_supply_uses_local_catalog_without_grpc_target(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    catalog_path = tmp_path / "supply_catalog.json"
    catalog_path.write_text(
        json.dumps(
            [
                {
                    "smiles": "CCO",
                    "catalog_id": "CAT-1",
                    "source": "local_catalog",
                    "source_timestamp": "2026-06-09T00:00:00Z",
                    "available": True,
                }
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv("SUPPLY_ORACLE_TARGET", raising=False)
    monkeypatch.setenv("SUPPLY_CATALOG_URI", catalog_path.as_uri())
    module = _load_module(
        "orchestrator_full_supply_local_catalog_test",
        ROOT / "services/orchestrator-svc/src/orchestrator_svc/main.py",
    )
    state = {
        "run_id": "run-1",
        "request": {"project_id": "project-1"},
        "candidates": [{"canonical_smiles": "CCO"}],
        "retrosyn": {
            "routes": [
                {
                    "route_id": "route-1",
                    "building_blocks": [{"smiles": "CCO"}],
                }
            ]
        },
    }

    result = await module.FullWorkflowClients().assess_supply(state)

    assert result["supply_assessment"]["overall_feasibility"] == "available"
    assert result["block_assessments"][0]["catalog_id"] == "CAT-1"
    assert result["block_assessments"][0]["catalog_source"] == "local_catalog"


@pytest.mark.asyncio
async def test_full_workflow_clients_assess_supply_marks_unavailable_without_routes() -> None:
    module = _load_module(
        "orchestrator_full_supply_no_routes_test",
        ROOT / "services/orchestrator-svc/src/orchestrator_svc/main.py",
    )
    state = {
        "run_id": "run-1",
        "request": {"project_id": "project-1"},
        "candidates": [{"canonical_smiles": "CCO"}],
        "retrosyn": {"routes": []},
    }

    result = await module.FullWorkflowClients().assess_supply(state)

    assert result["status"] == "assessed"
    assert result["smiles"] == "CCO"
    assert result["skip_reason"] == "retrosyn.routes is empty"
    assert result["supply_assessment"]["overall_feasibility"] == "unavailable"
    assert result["block_assessments"] == []


@pytest.mark.asyncio
async def test_full_workflow_clients_compile_synthesis_delegates_to_srb_agent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module(
        "orchestrator_full_srb_client_test",
        ROOT / "services/orchestrator-svc/src/orchestrator_svc/main.py",
    )
    calls: list[dict] = []

    class SRBAgent:
        async def process(self, payload):
            calls.append(payload)
            return {"status": "compiled", "protocols": [{"ssp_id": "ssp-1"}]}

    fake_srb_module = ModuleType("srb_agent.agent")
    fake_srb_module.SRBAgent = SRBAgent
    monkeypatch.setitem(sys.modules, "srb_agent.agent", fake_srb_module)
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
    state = {
        "run_id": "run-1",
        "request": {"project_id": "project-1"},
        "candidates": [{"canonical_smiles": "CCO"}],
        "retrosyn": {"routes": [route]},
    }

    result = await module.FullWorkflowClients().compile_synthesis(state)

    assert result["protocols"][0]["ssp_id"] == "ssp-1"
    assert calls == [
        {
            "project_id": "project-1",
            "run_id": "run-1",
            "molecule": {"smiles": "CCO"},
            "retrosyn_route": route,
        }
    ]


def test_orchestrator_deployment_wires_sila2_adapter_env() -> None:
    compose = (ROOT / "infra/docker/docker-compose.dev.yml").read_text(encoding="utf-8")
    k8s = (ROOT / "infra/kubernetes/deployments/moleculeforge-services.yaml").read_text(
        encoding="utf-8"
    )
    helm_values = (ROOT / "infra/helm/moleculeforge/values.yaml").read_text(
        encoding="utf-8"
    )

    for env_name in ("SILA2_PLAN_COMMAND", "SILA2_PLAN_TIMEOUT_SECONDS"):
        assert env_name in compose
        assert env_name in k8s
        assert env_name in helm_values

    assert "SILA2_PLAN_TIMEOUT_SECONDS: ${SILA2_PLAN_TIMEOUT_SECONDS:-120}" in compose
    assert "name: sila2-adapter-config" in k8s
    assert "configMapKeyRef:" in k8s
    assert "envValueFrom:" in helm_values


def test_orchestrator_deployment_wires_full_workflow_dependency_env() -> None:
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

    compose_env = set(compose_config["services"]["orchestrator-svc"]["environment"])
    deployment = next(
        item
        for item in k8s_docs
        if item
        and item.get("kind") == "Deployment"
        and item.get("metadata", {}).get("name") == "orchestrator-svc"
    )
    k8s_env = {
        item["name"]
        for item in deployment["spec"]["template"]["spec"]["containers"][0].get("env", [])
    }
    helm_service = helm_values["services"]["orchestrator-svc"]
    helm_env = set(helm_service.get("env", {})) | set(helm_service.get("envValueFrom", {}))
    required_env = {
        "BOLTZ2_ORACLE_COMMAND",
        "BOLTZ2_ORACLE_TIMEOUT_SECONDS",
        "BOLTZ_MODEL_PATH",
        "BOLTZ_INPUT_TEMPLATE_DIR",
        "BOLTZ_WORK_DIR",
        "BOLTZ_BINARY",
        "L4_QUANTUM_ORACLE_TARGET",
        "L4_QUANTUM_ORACLE_COMMAND",
        "L4_QUANTUM_ENGINE",
        "L4_GPU4PYSCF_COMMAND",
        "L4_ORCA_COMMAND",
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
        "RETROSYN_PLANNER_COMMAND",
        "RETROSYN_PLANNER_COMMANDS_JSON",
        "RASCORE_PLANNER_COMMAND",
        "RSGPT_PLANNER_COMMAND",
        "UALIGN_PLANNER_COMMAND",
        "AIZYNTH_PLANNER_COMMAND",
        "AIZYNTH_CONFIG_PATH",
        "RETROSYN_PLANNER_COMMAND_TIMEOUT_SECONDS",
        "HUMU_ENCODER_TARGET",
        "SUPPLY_ORACLE_TARGET",
        "SILA2_PLAN_COMMAND",
        "SILA2_PLAN_TIMEOUT_SECONDS",
    }

    assert required_env <= compose_env
    assert required_env <= k8s_env
    assert required_env <= helm_env
    assert "SUPPLY_ORACLE_TARGET: ${SUPPLY_ORACLE_TARGET:-supply-oracle-svc:50059}" in (
        ROOT / "infra/docker/docker-compose.dev.yml"
    ).read_text(encoding="utf-8")
    assert "HUMU_ENCODER_TARGET: ${HUMU_ENCODER_TARGET:-humu-encoder-svc:50051}" in (
        ROOT / "infra/docker/docker-compose.dev.yml"
    ).read_text(encoding="utf-8")
    assert (
        "UAS_GENERATOR_TARGET: "
        "${UAS_GENERATOR_TARGET:-python://generator_coord.agent:create_uas_generator_client}"
    ) in (ROOT / "infra/docker/docker-compose.dev.yml").read_text(encoding="utf-8")


def test_oracle_deployments_wire_external_runner_env() -> None:
    compose = (ROOT / "infra/docker/docker-compose.dev.yml").read_text(encoding="utf-8")
    k8s = (ROOT / "infra/kubernetes/deployments/moleculeforge-services.yaml").read_text(
        encoding="utf-8"
    )
    helm_values = (ROOT / "infra/helm/moleculeforge/values.yaml").read_text(
        encoding="utf-8"
    )

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
    assert (ROOT / "models/artifacts/gnina/gnina.1.3.2.cuda12.8").is_file()
    assert os.access(ROOT / "models/artifacts/gnina/gnina.1.3.2.cuda12.8", os.X_OK)
    assert (ROOT / "models/artifacts/diffdock").is_dir()
    assert (ROOT / "models/artifacts/boltz-2").is_dir()
    assert (ROOT / "models/artifacts/boltz-input-templates").is_dir()
    assert "BOLTZ2_ORACLE_TIMEOUT_SECONDS: ${BOLTZ2_ORACLE_TIMEOUT_SECONDS:-300}" in compose
    assert "BOLTZ2_ENSEMBLE_SIZE: ${BOLTZ2_ENSEMBLE_SIZE:-5}" in compose
    assert "BOLTZ_WORK_DIR: ${BOLTZ_WORK_DIR:-runs/boltz2}" in compose
    assert "BOLTZ_BINARY: ${BOLTZ_BINARY:-boltz}" in compose
    assert "FEP_ORACLE_TIMEOUT_SECONDS: ${FEP_ORACLE_TIMEOUT_SECONDS:-120}" in compose
    assert "FEP_METHOD: ${FEP_METHOD:-openfe}" in compose
    assert "FEP_N_REPEATS: ${FEP_N_REPEATS:-1}" in compose
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
        assert config["dock-oracle-command"] == ""
        assert config["dock-oracle-timeout-seconds"] == "120"
        assert config["gnina-binary"] == "models/artifacts/gnina/gnina.1.3.2.cuda12.8"
        assert config["diffdock-model-path"] == "models/artifacts/diffdock"
        assert config["boltz2-oracle-command"] == ""
        assert config["boltz2-oracle-timeout-seconds"] == "300"
        assert config["boltz2-ensemble-size"] == "5"
        assert config["boltz-model-path"] == "models/artifacts/boltz-2"
        assert config["boltz-input-template-dir"] == "models/artifacts/boltz-input-templates"
        assert config["boltz-work-dir"] == "runs/boltz2"
        assert config["boltz-binary"] == "boltz"
        assert config["fep-oracle-command"] == ""
        assert config["fep-oracle-timeout-seconds"] == "120"
        assert config["fep-method"] == "openfe"
        assert config["fep-n-repeats"] == "1"
        assert config["openfe-runner-path"] == ""
        assert config["l4-quantum-oracle-command"] == ""
        assert config["l4-quantum-engine"] == "quantum"
        assert config["l4-gpu4pyscf-command"] == ""
        assert config["l4-orca-command"] == ""


@pytest.mark.asyncio
async def test_full_workflow_clients_skip_synthesis_when_supply_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module(
        "orchestrator_full_srb_supply_skip_test",
        ROOT / "services/orchestrator-svc/src/orchestrator_svc/main.py",
    )

    class SRBAgent:
        async def process(self, payload):
            raise AssertionError("SRB compile must not run when supply is unavailable")

    fake_srb_module = ModuleType("srb_agent.agent")
    fake_srb_module.SRBAgent = SRBAgent
    monkeypatch.setitem(sys.modules, "srb_agent.agent", fake_srb_module)
    state = {
        "run_id": "run-1",
        "request": {"project_id": "project-1"},
        "candidates": [{"canonical_smiles": "CCO"}],
        "retrosyn": {"routes": [{"route_id": "route-1"}]},
        "supply": {"supply_assessment": {"overall_feasibility": "unavailable"}},
    }

    result = await module.FullWorkflowClients().compile_synthesis(state)

    assert result == {
        "status": "skipped",
        "protocols": [],
        "skip_reason": "supply feasibility is unavailable",
    }


@pytest.mark.asyncio
async def test_full_workflow_clients_compile_synthesis_skips_without_routes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module(
        "orchestrator_full_srb_no_routes_test",
        ROOT / "services/orchestrator-svc/src/orchestrator_svc/main.py",
    )

    class SRBAgent:
        async def process(self, payload):
            raise AssertionError("SRB compile must not run without retrosyn routes")

    fake_srb_module = ModuleType("srb_agent.agent")
    fake_srb_module.SRBAgent = SRBAgent
    monkeypatch.setitem(sys.modules, "srb_agent.agent", fake_srb_module)
    state = {
        "run_id": "run-1",
        "request": {"project_id": "project-1"},
        "candidates": [{"canonical_smiles": "CCO"}],
        "retrosyn": {"routes": []},
    }

    result = await module.FullWorkflowClients().compile_synthesis(state)

    assert result == {
        "status": "skipped",
        "protocols": [],
        "skip_reason": "retrosyn.routes is empty",
    }


@pytest.mark.asyncio
async def test_full_workflow_records_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module(
        "orchestrator_full_workflow_provenance_test",
        ROOT / "services/orchestrator-svc/src/orchestrator_svc/main.py",
    )
    records: list[object] = []

    class ProvenanceRecord:
        def __init__(self, **kwargs) -> None:
            self.__dict__.update(kwargs)

    async def create_record(record):
        records.append(record)
        return {
            "artifact_id": record.artifact_id,
            "signature": "sig-test",
            "recorded_at": "2026-05-30T00:00:00+00:00",
        }

    fake_provenance_module = ModuleType("provenance_svc.main")
    fake_provenance_module.ProvenanceRecord = ProvenanceRecord
    fake_provenance_module.create_record = create_record
    monkeypatch.setitem(sys.modules, "provenance_svc.main", fake_provenance_module)

    class Clients:
        async def compile_intent(self, state):
            return {"cig": {"source": state["nl_input"]}, "hciv": {}, "intent_cone": {}}

        async def generate_candidates(self, state):
            return [{"smiles": "CCO", "canonical_smiles": "CCO"}]

        async def validate_candidates(self, state):
            return {"passed": True, "results": [{"smiles": "CCO", "ki_nm": 5.0}]}

        async def plan_routes(self, state):
            return {"skipped": False, "routes": [{"route_id": "route-1"}]}

        async def assess_supply(self, state):
            return {"supply_assessment": {"overall_feasibility": "available"}}

        async def compile_synthesis(self, state):
            return {"status": "compiled", "protocols": [{"ssp_id": "ssp-1"}]}

        async def review_candidates(self, state):
            return {"verdict": "pass", "total_rules": 1}

    started = await module.start_design(
        {
            "nl_input": "Design KRAS G12C inhibitor",
            "workflow_scope": "full",
            "clients": Clients(),
            "run_id": "run-provenance-1",
            "trace_id": "trace-provenance-1",
            "artifact_ids": ["artifact-input"],
        }
    )

    assert records
    assert records[0].artifact_type == "workflow_state"
    assert records[0].parent_ids == ["artifact-input"]
    assert records[0].metadata["crg"]["project_id"] == "run-provenance-1"
    assert records[0].metadata["supply_feasibility"] == "available"
    assert records[0].metadata["srb_protocol_count"] == 1
    assert len(records[0].metadata["crg"]["beliefs"]) == 5
    assert len(records[0].metadata["crg"]["edges"]) == 4
    assert records[0].metadata["crg_belief_count"] == 5
    assert records[0].metadata["crg_edge_count"] == 4
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
                    "steps": [{"reaction": f"{smiles}>>{self.route_id}"}],
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
                    "steps": [{"reaction": "CCO>>route-rsgpt"}],
                }
            ),
        },
    )
    request = SimpleNamespace(molecule_smiles="CCO", max_routes=1, engine="ensemble")

    response = await service.FindRoutes(request, None)

    assert response.total_routes_found == 1
    assert [route.route_id for route in response.routes] == ["route-rsgpt"]


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
                "    {'routes': [{'route_id': route_id, 'score': score,"
                " 'steps': [{'reaction': 'CCO>>' + route_id}]}]},",
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
                "    {'routes': [{'route_id': route_id, 'score': score,"
                " 'steps': [{'reaction': 'CCO>>' + route_id}]}]},",
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
        "\"import json,sys; "
        "payload=json.load(sys.stdin); "
        "assert payload['smiles'] == 'CCO'; "
        "assert payload['max_routes'] == 1; "
        "print(json.dumps({'routes':[{'route_id':'route-command',"
        "'score':0.8,'predicted_yield':0.6,"
        "'steps':[{'reaction':'CCO>>CC=O'}]}]}))\""
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
        f"{sys.executable} -c \"import json,sys;"
        "json.dump({'routes': []}, sys.stdout)\"",
    )

    statuses = module._require_planner_runtime()
    planner_status = next(
        status
        for status in statuses
        if status.name == "retrosyn_planner_command"
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
        status
        for status in statuses
        if status.name == "retrosyn_rsgpt_planner_command"
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
    helm_values = (ROOT / "infra/helm/moleculeforge/values.yaml").read_text(
        encoding="utf-8"
    )

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
        "RETROSYN_PLANNER_COMMAND_TIMEOUT_SECONDS: "
        "${RETROSYN_PLANNER_COMMAND_TIMEOUT_SECONDS:-300}"
    ) in compose
    assert (
        "AIZYNTH_CONFIG_PATH: "
        "${AIZYNTH_CONFIG_PATH:-models/artifacts/aizynthfinder/config.yml}"
    ) in compose
    assert (ROOT / "models/artifacts/aizynthfinder/config.yml").is_file()
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
        helm_retrosyn_config = helm_configmaps[
            (namespace, "retrosyn-planner-config")
        ]["data"]
        assert helm_retrosyn_config["aizynth-config-path"] == (
            "models/artifacts/aizynthfinder/config.yml"
        )

    helm_template = (ROOT / "infra/helm/moleculeforge/templates/services.yaml").read_text(
        encoding="utf-8"
    )
    assert "kind: ConfigMap" in helm_template
    assert ".Values.configMaps" in helm_template


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
        SimpleNamespace(
            project_id="hfm-intent",
            batch_size=1,
            intent_cone={"axis": [1.0] + [0.0] * 128, "half_angle": 0.2},
            generator_params={"sampling_seed": 7},
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


def test_hfm_deployment_wires_checkpoint_and_decoder_env() -> None:
    compose = (ROOT / "infra/docker/docker-compose.dev.yml").read_text(encoding="utf-8")
    k8s = (ROOT / "infra/kubernetes/deployments/moleculeforge-services.yaml").read_text(
        encoding="utf-8"
    )
    helm_values = (ROOT / "infra/helm/moleculeforge/values.yaml").read_text(
        encoding="utf-8"
    )

    for env_name in (
        "HFM_CHECKPOINT_PATH",
        "HFM_DECODER_PATH",
        "HFM_MOLECULAR_DECODER_COMMAND",
    ):
        assert env_name in compose
        assert env_name in k8s
        assert env_name in helm_values

    assert "name: hfm-generator-config" in k8s
    assert "configMapKeyRef:" in k8s
    assert "envValueFrom:" in helm_values
    assert (
        "HFM_CHECKPOINT_PATH: ${HFM_CHECKPOINT_PATH:-checkpoints/hfm3d_4h200/best_model.pt}"
        in compose
    )
    assert (
        "HFM_DECODER_PATH: ${HFM_DECODER_PATH:-checkpoints/hfm3d_4h200/decoder.json}"
        in compose
    )
    assert (ROOT / "checkpoints/hfm3d_4h200/best_model.pt").is_file()
    assert (ROOT / "checkpoints/hfm3d_4h200/decoder.json").is_file()
    decoder_payload = json.loads(
        (ROOT / "checkpoints/hfm3d_4h200/decoder.json").read_text(encoding="utf-8")
    )
    decoder_entry = decoder_payload["entries"][0]
    assert isinstance(decoder_entry["sdf"], str)
    from rdkit import Chem

    assert Chem.MolFromMolBlock(
        decoder_entry["sdf"],
        sanitize=False,
        removeHs=False,
    ) is not None
    for config in (
        _k8s_configmap_data(k8s, "mf-generators", "hfm-generator-config"),
        _helm_configmap_data(helm_values, "mf-generators", "hfm-generator-config"),
    ):
        assert config["checkpoint-path"] == "checkpoints/hfm3d_4h200/best_model.pt"
        assert config["decoder-path"] == "checkpoints/hfm3d_4h200/decoder.json"
        assert config["molecular-decoder-command"] == ""


def test_crem_deployment_wires_mmp_and_external_scorer_env() -> None:
    compose = (ROOT / "infra/docker/docker-compose.dev.yml").read_text(encoding="utf-8")
    k8s = (ROOT / "infra/kubernetes/deployments/moleculeforge-services.yaml").read_text(
        encoding="utf-8"
    )
    helm_values = (ROOT / "infra/helm/moleculeforge/values.yaml").read_text(
        encoding="utf-8"
    )

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

    assert (
        "CREM_SCORER_COMMAND_TIMEOUT_SECONDS: "
        "${CREM_SCORER_COMMAND_TIMEOUT_SECONDS:-120}"
    ) in compose
    assert (
        "CREM_MMP_DB_PATH: ${CREM_MMP_DB_PATH:-models/artifacts/crem/crem_mmp_database.json}"
        in compose
    )
    assert (ROOT / "models/artifacts/crem/crem_mmp_database.json").is_file()
    assert "name: crem-generator-config" in k8s
    assert "configMapKeyRef:" in k8s
    assert "envValueFrom:" in helm_values
    for config in (
        _k8s_configmap_data(k8s, "mf-generators", "crem-generator-config"),
        _helm_configmap_data(helm_values, "mf-generators", "crem-generator-config"),
    ):
        assert config["mmp-db-path"] == "models/artifacts/crem/crem_mmp_database.json"
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
async def test_aizynth_retrosyn_ignores_empty_step_routes() -> None:
    from mf_retrosyn.aizynth.retrosyn import AiZynthRetrosyn

    class Runner:
        def find_routes(self, smiles: str, max_routes: int = 10) -> list[dict]:
            return [{"route_id": "aizynth-1", "smiles": smiles, "steps": []}]

    routes = await AiZynthRetrosyn(runner=Runner()).find_routes("CCO", max_routes=1)

    assert routes == []


@pytest.mark.asyncio
async def test_aizynth_retrosyn_completes_reactant_only_steps() -> None:
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

    routes = await AiZynthRetrosyn(runner=Runner()).find_routes("CCOO", max_routes=1)

    assert routes[0]["steps"][0]["reaction"] == "CCO.O=O>>CCOO"
    assert routes[0]["steps"][0]["conditions"] == {"source": "aizynthfinder"}
    assert routes[0]["steps"][0]["building_blocks"] == [
        {"smiles": "CCO"},
        {"smiles": "O=O"},
    ]


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

    monkeypatch.setenv("BOLTZ2_PROTEIN_PDB_ID", "6OIM")
    monkeypatch.setenv("BOLTZ2_ENSEMBLE_SIZE", "2")
    service = Boltz2Service()
    oracle = module.Boltz2OracleServicer(service=service)

    response = await oracle.PredictWithUncertainty(
        oracle_pb2.OracleBatchRequest(
            project_id="project-1",
            molecule_smiles=["CCO"],
            requested_properties=["affinity"],
            level=oracle_pb2.L2_DOCKING,
            return_uncertainty=True,
        ),
        None,
    )

    assert service.requests[0].project_id == "project-1"
    assert service.requests[0].protein_pdb_id == "6OIM"
    assert list(service.requests[0].ligand_smiles) == ["CCO"]
    assert service.requests[0].ensemble_size == 2
    assert response.batch_id == "project-1"
    assert response.total_elapsed_ms == 21
    assert response.evaluations[0].oracle_name == "boltz2"
    assert response.evaluations[0].molecule_smiles == "CCO"
    assert response.evaluations[0].scores == {"affinity": -8.2, "ki_nm": 12.0}
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
            "steps": [
                {
                    "reaction": "CCO>>CC=O",
                    "building_blocks": [{"smiles": "CCO"}],
                }
            ],
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

    response = await module.CriticServicer().Evaluate(
        critic_pb2.CriticBatchResult(
            molecule_smiles="CCO",
            project_id="critic-test",
        ),
        None,
    )

    assert response is not None
    assert response.molecule_smiles == "CCO"
    assert response.rules_evaluated > 0
    assert response.rule_results


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

    assert len(repository.beliefs) == 1
    belief = repository.beliefs[0]
    assert belief["project_id"] == "project-1"
    assert belief["run_id"] == "run-1"
    assert belief["subject"] == "CCO"
    assert belief["predicate"] == "critic_verdict"
    assert belief["object_value"] == "pass"
    assert belief["source_agent"] == "critic_agent"


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
async def test_critic_agent_uses_existing_critic_verdict_from_shared_crg() -> None:
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
        rule_id = "must_not_run"
        name = "Must not run"

        def evaluate(self, smiles, properties):
            raise AssertionError("critic rules must not run when shared CRG has verdict")

    repository = CRGRepository()
    agent = module.ScientificCriticAgent(crg_repository=repository)
    agent.rules = [Rule()]

    result = await agent.evaluate_molecule(
        {
            "project_id": "project-1",
            "run_id": "run-1",
            "smiles": "CCO",
            "properties": {},
        }
    )

    assert result["cache_source"] == "shared_crg"
    assert result["verdict"] == "pass"
    assert result["passed"] == 1
    assert result["failed"] == 0
    assert result["rule_results"][0]["rule_id"] == "crg_critic_verdict"
    assert repository.beliefs == []


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
                    "steps": [{"reaction": "CCO>>CC=O"}],
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
                    "steps": [{"reaction": "CCO>>CC=O"}],
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
                    "steps": [{"reaction": f"{smiles}>>{self.route_id}"}],
                }
            ]

    aizynth = Planner("route-aizynth", 0.4)
    rsgpt = Planner("route-rsgpt", 0.9)
    agent = module.RetroSynAgent(
        route_planners={"aizynth": aizynth, "rsgpt": rsgpt},
        crg_repository=None,
    )

    result = await agent.process({"smiles": "CCO", "max_routes": 2})

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
                    "steps": [{"reaction": "CCO>>route-rsgpt"}],
                }
            ),
        },
        crg_repository=None,
    )

    result = await agent.process({"smiles": "CCO", "max_routes": 1})

    assert [route["route_id"] for route in result["routes"]] == ["route-rsgpt"]


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
                "    {'routes': [{'route_id': route_id, 'score': score,"
                " 'steps': [{'reaction': 'CCO>>' + route_id}]}]},",
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

    result = await agent.process({"smiles": "CCO", "max_routes": 2})

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
                "    {'routes': [{'route_id': route_id, 'score': score,"
                " 'steps': [{'reaction': 'CCO>>' + route_id}]}]},",
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

    result = await agent.process({"smiles": "CCO", "max_routes": 3})

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
        "\"import json,sys; "
        "payload=json.load(sys.stdin); "
        "assert payload['smiles'] == 'CCO'; "
        "assert payload['max_routes'] == 1; "
        "print(json.dumps({'routes':[{'route_id':'route-command',"
        "'score':0.8,'steps':[{'reaction':'CCO>>CC=O'}]}]}))\""
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
async def test_retrosyn_agent_uses_failed_validation_belief_from_shared_crg() -> None:
    module = _load_module(
        "retrosyn_agent_crg_readback_test",
        ROOT / "agents/retrosyn_agent/src/retrosyn_agent/agent.py",
    )

    class Planner:
        async def find_routes(self, smiles: str, max_routes: int) -> list[dict]:
            raise AssertionError("planner should not run after failed validation")

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
    agent = module.RetroSynAgent(planner=Planner(), crg_repository=repository)

    result = await agent.process(
        {
            "project_id": "project-1",
            "run_id": "run-1",
            "smiles": "CCO",
            "max_routes": 1,
        }
    )

    assert result["status"] == "skipped"
    assert result["routes"] == []
    assert result["skip_reason"] == "shared CRG contains failed validation_status"
    assert repository.beliefs[0]["predicate"] == "retrosyn_routes"
    assert repository.beliefs[0]["object_value"] == "0"
    assert repository.beliefs[0]["evidence_ids"] == ["crg_validation_status"]


@pytest.mark.asyncio
async def test_retrosyn_agent_uses_existing_zero_routes_from_shared_crg() -> None:
    module = _load_module(
        "retrosyn_agent_zero_routes_crg_readback_test",
        ROOT / "agents/retrosyn_agent/src/retrosyn_agent/agent.py",
    )

    class Planner:
        async def find_routes(self, smiles: str, max_routes: int) -> list[dict]:
            raise AssertionError("planner should not run when shared CRG has zero routes")

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
    agent = module.RetroSynAgent(planner=Planner(), crg_repository=repository)

    result = await agent.process(
        {
            "project_id": "project-1",
            "run_id": "run-1",
            "smiles": "CCO",
            "max_routes": 1,
        }
    )

    assert result["status"] == "skipped"
    assert result["routes"] == []
    assert result["cache_source"] == "shared_crg"
    assert result["skip_reason"] == "shared CRG contains zero retrosyn_routes"
    assert repository.beliefs == []


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


@pytest.mark.asyncio
async def test_orchestrator_agent_uses_completed_workflow_status_from_shared_crg() -> None:
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
        }
    )

    assert repository.reads == ["run-1"]
    assert result["status"] == "completed"
    assert result["cached"] is True
    assert result["visited_nodes"] == ["nl2obj", "generate", "critic"]
    assert agent.cycle_count == 0
    assert repository.beliefs[0]["evidence_ids"] == ["crg_workflow_status"]


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
async def test_base_agent_publishes_signed_agent_message_envelope() -> None:
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
        envelope.payload_type_url
        == "type.googleapis.com/moleculeforge.v1.agent.ValidationResult"
    )
    assert envelope.lineage["parent_trace"] == "trace-0"
    assert envelope.ttl == 4
    assert envelope.signature
    assert agent.verify_agent_message(envelope) is True

    envelope.payload = b'{"smiles":"CCN"}'
    assert agent.verify_agent_message(envelope) is False


@pytest.mark.asyncio
async def test_base_agent_generates_uuidv7_message_id_by_default() -> None:
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
        f"{sys.executable} -c \"import json,sys;"
        "req=json.load(sys.stdin);"
        "sig='agent-sig-'+req['payload_hash'][:8];"
        "print(json.dumps({'signature':sig,'signature_type':'sigstore_rekor',"
        "'rekor_entry':{'uuid':'agent-rekor'}}))\""
    )
    verify_command = (
        f"{sys.executable} -c \"import json,sys;"
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
        f"{sys.executable} -c \"import json,sys;"
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
        f"{sys.executable} -c \"import json,sys;"
        "req=json.load(sys.stdin);"
        "sig='agent-sig-'+req['payload_hash'][:8];"
        "print(json.dumps({'signature':sig}))\""
    )
    verify_command = (
        f"{sys.executable} -c \"import json,sys;"
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
async def test_base_agent_encodes_jsonld_payload_before_signing() -> None:
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
async def test_base_agent_verifies_messages_signed_by_sender_identity() -> None:
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
async def test_base_agent_start_dispatches_verified_agent_message_payload() -> None:
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
async def test_base_agent_rejects_expired_agent_message_ttl() -> None:
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
async def test_base_agent_rejects_invalid_received_agent_message_type() -> None:
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
async def test_base_agent_rejects_missing_received_recipient() -> None:
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
async def test_base_agent_rejects_missing_received_payload_type_url() -> None:
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
