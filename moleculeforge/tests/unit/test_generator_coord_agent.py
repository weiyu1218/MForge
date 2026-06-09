import json
import sys
from types import ModuleType

import pytest
from generator_coord.agent import GeneratorCoordAgent
from mf_core.routing.task_router import GENERATOR_NAMES


def test_available_generators_match_task_router_names():
    agent = GeneratorCoordAgent()

    assert agent.generators == list(GENERATOR_NAMES)
    assert set(agent.generators).issubset(GENERATOR_NAMES)


def test_select_all_returns_task_router_names():
    agent = GeneratorCoordAgent()

    selected = agent._select_generators("all", {})

    assert selected == list(GENERATOR_NAMES)
    assert set(selected).issubset(GENERATOR_NAMES)


def test_select_real_generator_strategy_returns_that_generator():
    agent = GeneratorCoordAgent()

    selected = agent._select_generators("hfm_3d", {})

    assert selected == ["hfm_3d"]
    assert set(selected).issubset(GENERATOR_NAMES)


def test_auto_selected_generators_are_task_router_names():
    agent = GeneratorCoordAgent()

    for objectives in (
        {"complexity": "high"},
        {"complexity": "low"},
        {"complexity": "medium"},
        {},
    ):
        selected = agent._select_generators("auto", objectives)
        assert set(selected).issubset(GENERATOR_NAMES)


def test_unknown_strategy_uses_stable_real_generator_defaults():
    agent = GeneratorCoordAgent()

    selected = agent._select_generators("unknown", {})

    assert selected == ["hfm_3d", "fragfm"]
    assert set(selected).issubset(GENERATOR_NAMES)


def test_builds_generator_clients_from_target_map(monkeypatch):
    import generator_coord.agent as module

    created_targets: list[str] = []

    class GeneratorClient:
        def __init__(self, target: str) -> None:
            created_targets.append(target)
            self.target = target

    monkeypatch.setattr(module, "GeneratorGrpcClient", GeneratorClient)

    agent = module.GeneratorCoordAgent(
        generator_targets={"hfm_3d": "localhost:50066"}
    )

    assert created_targets == ["localhost:50066"]
    assert agent.generator_clients["hfm_3d"].target == "localhost:50066"


def test_builds_generator_clients_from_python_target(monkeypatch):
    import generator_coord.agent as module

    class GeneratorClient:
        async def generate(self, request: dict) -> dict:
            return {"candidates": [{"smiles": "CCO"}]}

    provider_module = ModuleType("test_generator_coord_python_client")
    provider_module.create_client = lambda: GeneratorClient()
    monkeypatch.setitem(sys.modules, provider_module.__name__, provider_module)

    agent = module.GeneratorCoordAgent(
        generator_targets={
            "uas": f"python://{provider_module.__name__}:create_client",
        }
    )

    assert isinstance(agent.generator_clients["uas"], GeneratorClient)


@pytest.mark.asyncio
async def test_uas_python_client_uses_runner_command(monkeypatch):
    import generator_coord.agent as module

    command = (
        f"{sys.executable} -c \"import json,sys;"
        "req=json.load(sys.stdin);"
        "assert req['generator']=='uas';"
        "assert req['hciv']=={'coordinates':[1.0,0.0]};"
        "assert req['cone']=={'axis':[1.0,0.0],'half_angle':0.25};"
        "assert req['n_samples']==1;"
        "assert req['seed']==7;"
        "print(json.dumps({'candidates':[{'id':'uas-1','smiles':'CCO',"
        "'canonical_smiles':'CCO'}]}))\""
    )
    monkeypatch.setenv("UAS_RUNNER_COMMAND", command)

    client = module.create_uas_generator_client()
    health = await client.health_check()
    result = await client.generate(
        {
            "hciv": {"coordinates": [1.0, 0.0]},
            "intent_cone": {"axis": [1.0, 0.0], "half_angle": 0.25},
            "n_samples": 1,
            "generator_params": {"sampling_seed": 7},
        }
    )

    assert health == {"healthy": True, "generator_name": "uas", "version": "0.1.0"}
    assert result["generator_name"] == "uas"
    assert result["candidates"][0]["smiles"] == "CCO"
    assert result["candidates"][0]["generator_name"] == "uas"


@pytest.mark.asyncio
async def test_uas_python_client_rejects_missing_runner_command(monkeypatch):
    import generator_coord.agent as module

    monkeypatch.setenv("UAS_RUNNER_COMMAND", "missing-uas-runner --json")
    client = module.create_uas_generator_client()

    health = await client.health_check()

    assert health["healthy"] is False
    assert "not found" in health["reason"]
    with pytest.raises(RuntimeError, match="not found"):
        await client.generate(
            {
                "hciv": {"coordinates": [1.0, 0.0]},
                "intent_cone": {"axis": [1.0, 0.0], "half_angle": 0.25},
                "n_samples": 1,
            }
        )


def test_generator_targets_from_http_discovery_uri(monkeypatch):
    import json

    import generator_coord.agent as module

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return json.dumps(
                {
                    "targets": {
                        "hfm_3d": "hfm-discovered:50051",
                        "fragfm": "fragfm-discovered:50052",
                    }
                }
            ).encode("utf-8")

    calls = []

    def fake_urlopen(request, timeout):
        calls.append((request.full_url, timeout))
        return FakeResponse()

    monkeypatch.setenv("GENERATOR_DISCOVERY_URI", "https://registry.example/generators")
    monkeypatch.setenv("HFM_3D_GENERATOR_TARGET", "hfm-override:50061")
    monkeypatch.delenv("GENERATOR_CLIENT_TARGETS", raising=False)
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    targets = module._generator_targets_from_env()

    assert calls == [("https://registry.example/generators", 30.0)]
    assert targets["hfm_3d"] == "hfm-override:50061"
    assert targets["fragfm"] == "fragfm-discovered:50052"


@pytest.mark.asyncio
async def test_process_dispatches_to_selected_generator_client():
    class GeneratorClient:
        def __init__(self) -> None:
            self.requests: list[dict] = []

        async def generate(self, request: dict) -> dict:
            self.requests.append(request)
            return {"candidates": [{"smiles": "CCO"}]}

    client = GeneratorClient()
    agent = GeneratorCoordAgent(generator_clients={"hfm_3d": client})

    result = await agent.process(
        {
            "generation_strategy": "hfm_3d",
            "batch_size": 2,
            "objectives": {"complexity": "low"},
            "intent_cone": {"axis": [1.0], "half_angle": 0.2},
        }
    )

    assert client.requests == [
        {
            "generator": "hfm_3d",
            "batch_size": 2,
            "objectives": {"complexity": "low"},
            "intent_cone": {"axis": [1.0], "half_angle": 0.2},
        }
    ]
    assert result["status"] == "dispatched"
    assert result["dispatch_results"][0]["generator"] == "hfm_3d"
    assert result["candidates"] == [{"smiles": "CCO", "generator": "hfm_3d"}]


@pytest.mark.asyncio
async def test_process_checks_generator_health_before_dispatch():
    class GeneratorClient:
        def __init__(self) -> None:
            self.events: list[str] = []

        async def health_check(self) -> dict:
            self.events.append("health")
            return {"healthy": True, "generator_name": "hfm_3d"}

        async def generate(self, request: dict) -> dict:
            self.events.append("generate")
            return {"candidates": [{"smiles": "CCO"}]}

    client = GeneratorClient()
    agent = GeneratorCoordAgent(generator_clients={"hfm_3d": client})

    result = await agent.process({"generation_strategy": "hfm_3d", "objectives": {}})

    assert client.events == ["health", "generate"]
    assert result["dispatch_results"][0]["health_status"] == "healthy"


@pytest.mark.asyncio
async def test_process_rejects_unhealthy_generator_client():
    class GeneratorClient:
        async def health_check(self) -> dict:
            return {
                "healthy": False,
                "generator_name": "hfm_3d",
                "reason": "not serving",
            }

        async def generate(self, request: dict) -> dict:
            raise AssertionError("unhealthy generator must not be invoked")

    agent = GeneratorCoordAgent(generator_clients={"hfm_3d": GeneratorClient()})

    with pytest.raises(RuntimeError, match="not serving"):
        await agent.process({"generation_strategy": "hfm_3d", "objectives": {}})


@pytest.mark.asyncio
async def test_process_persists_selected_generator_belief_to_crg_repository():
    class CRGRepository:
        def __init__(self) -> None:
            self.beliefs: list[dict] = []

        async def write_workflow_belief(self, **kwargs) -> None:
            self.beliefs.append(kwargs)

    repository = CRGRepository()
    agent = GeneratorCoordAgent(crg_repository=repository)

    await agent.process(
        {
            "project_id": "project-1",
            "run_id": "run-1",
            "generation_strategy": "hfm_3d",
            "objectives": {},
        }
    )

    assert len(repository.beliefs) == 1
    belief = repository.beliefs[0]
    assert belief["project_id"] == "project-1"
    assert belief["run_id"] == "run-1"
    assert belief["predicate"] == "selected_generators"
    assert belief["object_value"] == "hfm_3d"
    assert belief["source_agent"] == "generator_coord"


@pytest.mark.asyncio
async def test_auto_strategy_uses_failed_validation_belief_from_shared_crg():
    class CRGRepository:
        def __init__(self) -> None:
            self.reads: list[str] = []
            self.beliefs: list[dict] = []

        async def get_run_crg(self, run_id: str) -> dict:
            self.reads.append(run_id)
            return {
                "project_id": "project-1",
                "beliefs": [
                    {
                        "predicate": "validation_status",
                        "object": "failed",
                    }
                ],
                "edges": [],
            }

        async def write_workflow_belief(self, **kwargs) -> None:
            self.beliefs.append(kwargs)

    repository = CRGRepository()
    agent = GeneratorCoordAgent(crg_repository=repository)

    result = await agent.process(
        {
            "project_id": "project-1",
            "run_id": "run-1",
            "generation_strategy": "auto",
            "objectives": {},
        }
    )

    assert repository.reads == ["run-1"]
    assert result["selected_generators"] == ["mmpt_rag", "fragfm"]
    assert repository.beliefs[0]["object_value"] == "mmpt_rag,fragfm"


@pytest.mark.asyncio
async def test_auto_strategy_uses_existing_selected_generators_from_shared_crg():
    class CRGRepository:
        def __init__(self) -> None:
            self.reads: list[str] = []
            self.beliefs: list[dict] = []

        async def get_run_crg(self, run_id: str) -> dict:
            self.reads.append(run_id)
            return {
                "project_id": "project-1",
                "beliefs": [
                    {
                        "predicate": "selected_generators",
                        "object_value": "iclm,uas",
                    },
                    {
                        "predicate": "validation_status",
                        "object_value": "failed",
                    },
                ],
                "edges": [],
            }

        async def write_workflow_belief(self, **kwargs) -> None:
            self.beliefs.append(kwargs)

    repository = CRGRepository()
    agent = GeneratorCoordAgent(crg_repository=repository)

    result = await agent.process(
        {
            "project_id": "project-1",
            "run_id": "run-1",
            "generation_strategy": "auto",
            "objectives": {},
        }
    )

    assert repository.reads == ["run-1"]
    assert result["cache_source"] == "shared_crg"
    assert result["selected_generators"] == ["iclm", "uas"]
    assert repository.beliefs == []


@pytest.mark.asyncio
async def test_process_passes_route_humu_feedback_from_shared_crg_to_generators():
    class CRGRepository:
        def __init__(self) -> None:
            self.reads: list[str] = []
            self.beliefs: list[dict] = []

        async def get_run_crg(self, run_id: str) -> dict:
            self.reads.append(run_id)
            return {
                "project_id": "project-1",
                "beliefs": [
                    {
                        "subject": "CCO",
                        "predicate": "route_humu_embedding",
                        "object_value": json.dumps(
                            {
                                "route_id": "route-1",
                                "source": "retrosyn_agent",
                                "curvature": 1.0,
                                "humu_embedding": [1.0, 0.0],
                                "evidence_ids": "belief-1",
                                "metadata": {"route_rank": 1},
                            },
                            sort_keys=True,
                        ),
                    }
                ],
                "edges": [],
            }

        async def write_workflow_belief(self, **kwargs) -> None:
            self.beliefs.append(kwargs)

    class GeneratorClient:
        def __init__(self) -> None:
            self.requests: list[dict] = []

        async def health_check(self) -> dict:
            return {"healthy": True}

        async def generate(self, request: dict) -> dict:
            self.requests.append(request)
            return {"candidates": []}

    hfm_client = GeneratorClient()
    fragfm_client = GeneratorClient()
    repository = CRGRepository()
    agent = GeneratorCoordAgent(
        generator_clients={"hfm_3d": hfm_client, "fragfm": fragfm_client},
        crg_repository=repository,
    )

    result = await agent.process(
        {
            "project_id": "project-1",
            "run_id": "run-1",
            "generation_strategy": "auto",
            "objectives": {},
        }
    )

    assert repository.reads == ["run-1"]
    assert result["route_humu_feedback"] == [
        {
            "route_id": "route-1",
            "source": "retrosyn_agent",
            "curvature": 1.0,
            "humu_embedding": [1.0, 0.0],
            "evidence_ids": "belief-1",
            "metadata": {"route_rank": 1},
        }
    ]
    feedback = json.loads(hfm_client.requests[0]["generator_params"]["route_humu_feedback"])
    assert feedback == result["route_humu_feedback"]
    jmcg_feedback = json.loads(hfm_client.requests[0]["generator_params"]["jmcg_feedback"])
    assert jmcg_feedback == result["jmcg_feedback"]
    assert jmcg_feedback["schema"] == "moleculeforge.jmcg.feedback.v1"
    assert jmcg_feedback["run_id"] == "run-1"
    assert jmcg_feedback["project_id"] == "project-1"
    assert jmcg_feedback["records"] == [
        {
            "kind": "route",
            "source": "retrosyn_agent",
            "run_id": "run-1",
            "subject": {"type": "route", "id": "route-1"},
            "humu_embedding": [1.0, 0.0],
            "curvature": 1.0,
            "weight": 1.0,
            "polarity": "attract",
            "confidence": 1.0,
            "evidence_ids": ["belief-1"],
            "metadata": {"route_rank": 1},
        }
    ]
    fragfm_feedback = json.loads(
        fragfm_client.requests[0]["generator_params"]["route_humu_feedback"]
    )
    assert fragfm_feedback == feedback
    fragfm_jmcg_feedback = json.loads(
        fragfm_client.requests[0]["generator_params"]["jmcg_feedback"]
    )
    assert fragfm_jmcg_feedback == jmcg_feedback


@pytest.mark.asyncio
async def test_process_merges_existing_jmcg_property_feedback_with_route_feedback():
    class CRGRepository:
        async def get_run_crg(self, run_id: str) -> dict:
            return {
                "project_id": "project-1",
                "beliefs": [
                    {
                        "subject": "CCO",
                        "predicate": "route_humu_embedding",
                        "object_value": json.dumps(
                            {
                                "route_id": "route-1",
                                "curvature": 1.0,
                                "humu_embedding": [1.0, 0.0],
                            },
                            sort_keys=True,
                        ),
                    }
                ],
                "edges": [],
            }

        async def write_workflow_belief(self, **kwargs) -> None:
            return None

    class GeneratorClient:
        def __init__(self) -> None:
            self.requests: list[dict] = []

        async def health_check(self) -> dict:
            return {"healthy": True}

        async def generate(self, request: dict) -> dict:
            self.requests.append(request)
            return {"candidates": []}

    client = GeneratorClient()
    property_feedback = {
        "schema": "moleculeforge.jmcg.feedback.v1",
        "run_id": "run-1",
        "project_id": "project-1",
        "records": [
            {
                "kind": "property",
                "source": "validation",
                "run_id": "run-1",
                "subject": {"type": "workflow_feedback", "id": "validation-0"},
                "weight": 1.0,
                "polarity": "repel",
                "confidence": 1.0,
                "evidence_ids": [],
                "metadata": {"reason": "affinity gate failed"},
            }
        ],
    }
    agent = GeneratorCoordAgent(
        generator_clients={"hfm_3d": client, "fragfm": GeneratorClient()},
        crg_repository=CRGRepository(),
    )

    result = await agent.process(
        {
            "project_id": "project-1",
            "run_id": "run-1",
            "generation_strategy": "auto",
            "objectives": {},
            "generator_params": {
                "jmcg_feedback": json.dumps(property_feedback, sort_keys=True),
            },
        }
    )

    merged = json.loads(client.requests[0]["generator_params"]["jmcg_feedback"])
    assert merged == result["jmcg_feedback"]
    assert [record["kind"] for record in merged["records"]] == ["property", "route"]
    assert merged["records"][0] == property_feedback["records"][0]
    assert merged["records"][1]["subject"] == {"type": "route", "id": "route-1"}


@pytest.mark.asyncio
async def test_process_preserves_existing_intent_and_pocket_jmcg_feedback_records():
    class CRGRepository:
        async def get_run_crg(self, run_id: str) -> dict:
            return {
                "project_id": "project-1",
                "beliefs": [
                    {
                        "subject": "CCO",
                        "predicate": "route_humu_embedding",
                        "object_value": json.dumps(
                            {
                                "route_id": "route-1",
                                "curvature": 1.0,
                                "humu_embedding": [1.0, 0.0],
                            },
                            sort_keys=True,
                        ),
                    }
                ],
                "edges": [],
            }

        async def write_workflow_belief(self, **kwargs) -> None:
            return None

    class GeneratorClient:
        def __init__(self) -> None:
            self.requests: list[dict] = []

        async def health_check(self) -> dict:
            return {"healthy": True}

        async def generate(self, request: dict) -> dict:
            self.requests.append(request)
            return {"candidates": []}

    existing_feedback = {
        "schema": "moleculeforge.jmcg.feedback.v1",
        "run_id": "run-1",
        "project_id": "project-1",
        "records": [
            {
                "kind": "intent",
                "source": "orchestrator_svc",
                "run_id": "run-1",
                "subject": {"type": "intent", "id": "run-1"},
                "weight": 1.0,
                "polarity": "attract",
                "confidence": 1.0,
                "evidence_ids": [],
                "metadata": {"has_hciv": True},
            },
            {
                "kind": "pocket",
                "source": "orchestrator_svc",
                "run_id": "run-1",
                "subject": {"type": "pocket", "id": "switch-ii"},
                "weight": 1.0,
                "polarity": "attract",
                "confidence": 1.0,
                "evidence_ids": [],
                "metadata": {"pocket_id": "switch-ii"},
            },
        ],
    }
    client = GeneratorClient()
    agent = GeneratorCoordAgent(
        generator_clients={"hfm_3d": client, "fragfm": GeneratorClient()},
        crg_repository=CRGRepository(),
    )

    result = await agent.process(
        {
            "project_id": "project-1",
            "run_id": "run-1",
            "generation_strategy": "auto",
            "objectives": {},
            "generator_params": {
                "jmcg_feedback": json.dumps(existing_feedback, sort_keys=True),
            },
        }
    )

    merged = json.loads(client.requests[0]["generator_params"]["jmcg_feedback"])
    assert merged == result["jmcg_feedback"]
    assert [record["kind"] for record in merged["records"]] == [
        "intent",
        "pocket",
        "route",
    ]
    assert merged["records"][0] == existing_feedback["records"][0]
    assert merged["records"][1] == existing_feedback["records"][1]
    assert merged["records"][2]["subject"] == {"type": "route", "id": "route-1"}


@pytest.mark.asyncio
async def test_process_fails_when_selected_generator_client_missing():
    agent = GeneratorCoordAgent(generator_clients={"fragfm": object()})

    with pytest.raises(RuntimeError, match="hfm_3d"):
        await agent.process({"generation_strategy": "hfm_3d", "objectives": {}})
