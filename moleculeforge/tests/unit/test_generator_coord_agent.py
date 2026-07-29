import asyncio
import json
import struct
import sys
from collections.abc import Callable
from types import ModuleType

import generator_coord.agent as coordinator_module
import grpc
import pytest
from google.protobuf.message import Message
from mf_core.proto_gen.moleculeforge.v1.core import audit_pb2, cig_pb2, humu_pb2
from mf_core.proto_gen.moleculeforge.v1.generator import (
    generator_pb2,
    router_pb2,
    router_pb2_grpc,
)
from mf_core.routing.task_router import GENERATOR_NAMES


def _artifact(generator_name: str) -> audit_pb2.ArtifactRef:
    return audit_pb2.ArtifactRef(
        name=f"{generator_name}_checkpoint",
        version="test-v1",
        checksum=f"sha256:{generator_name}",
        required=True,
    )


def _info(
    generator_name: str,
    *,
    max_batch_size: int = 8,
    runtime_status: int = audit_pb2.GENERATOR_RUNTIME_STATUS_READY,
) -> generator_pb2.GeneratorInfo:
    return generator_pb2.GeneratorInfo(
        name=generator_name,
        version="0.1.0",
        max_batch_size=max_batch_size,
        runtime_status=runtime_status,
        status_message=(
            "ready" if runtime_status == audit_pb2.GENERATOR_RUNTIME_STATUS_READY else "off"
        ),
        artifacts=[_artifact(generator_name)],
    )


def _cig_dict() -> dict:
    return {
        "project_id": "project-1",
        "objectives": [
            {
                "id": "qed",
                "name": "QED",
                "type": "MAXIMIZE",
                "target_value": 0.8,
                "target_min": None,
                "target_max": None,
                "property": "qed",
                "weight": 1.0,
                "pareto_tier": 1,
            }
        ],
        "edges": [],
        "hyperedges": [],
        "constraints": {"mw": "<500"},
        "created_by": "test",
    }


def _cig_proto() -> cig_pb2.CIG:
    return cig_pb2.CIG(
        project_id="project-1",
        objectives=[
            cig_pb2.ObjectiveNode(
                id="qed",
                name="QED",
                type=cig_pb2.MAXIMIZE,
                target_value=0.8,
                property="qed",
                weight=1.0,
                pareto_tier=1,
            )
        ],
        constraints={"mw": "<500"},
        created_by="test",
    )


def _hciv_dict() -> dict:
    return {
        "coordinates": [1.0, *([0.0] * 128)],
        "curvature": 1.0,
        "molecule_smiles": "",
    }


def _cone_dict() -> dict:
    return {
        "axis": [1.0, *([0.0] * 128)],
        "half_angle": 0.25,
        "curvature": 1.0,
        "property_weights": {"qed": 1.0},
    }


def _generation_payload(*, n_samples: int = 3, n_select: int = 2) -> dict:
    return {
        "project_id": "project-1",
        "run_id": "run-1",
        "request_id": "request-1",
        "generation_strategy": "auto",
        "objectives": {"complexity": "high", "qed": 0.8},
        "task_profile": {
            "target_family": "kinase",
            "stage": "lead_opt",
            "data_richness": 25.0,
            "novelty_demand": 0.8,
            "multi_target": True,
            "sa_constraint": 3.0,
            "task_complexity": "high",
        },
        "cig": _cig_dict(),
        "hciv": _hciv_dict(),
        "intent_cone": _cone_dict(),
        "n_samples": n_samples,
        "batch_size": n_samples,
        "n_select": n_select,
        "generator_params": {"sampling_seed": "11"},
    }


def _molecule_payload(
    smiles: str,
    *,
    extra: dict | None = None,
) -> bytes:
    payload = {
        "smiles": smiles,
        "canonical_smiles": smiles,
        "properties": {},
    }
    payload.update(extra or {})
    return json.dumps(payload, sort_keys=True).encode("utf-8")


def _response_for_request(
    generator_name: str,
    request: generator_pb2.GenerateRequest,
    artifacts: list[audit_pb2.ArtifactRef],
    *,
    molecule_count: int | None = None,
    payload_factory: Callable[[int], bytes] | None = None,
    embeddings: list[bytes] | None = None,
) -> generator_pb2.GenerateResponse:
    count = request.batch_size if molecule_count is None else molecule_count
    payload_factory = payload_factory or (lambda index: _molecule_payload(f"C{'C' * (index + 1)}O"))
    return generator_pb2.GenerateResponse(
        generator_name=generator_name,
        generation_id=request.request_id,
        request_id=request.request_id,
        molecules=[payload_factory(index) for index in range(count)],
        humu_embeddings=list(embeddings or []),
        artifacts=artifacts,
        molecule_payload_schema="molecule.v1",
        embedding_payload_schema="humu.float32.v1",
    )


class RecordingGenerator:
    def __init__(
        self,
        info: generator_pb2.GeneratorInfo,
        *,
        events: list[str] | None = None,
        response_factory: Callable[[generator_pb2.GenerateRequest], generator_pb2.GenerateResponse]
        | None = None,
    ) -> None:
        self.info_response = info
        self.events = events if events is not None else []
        self.requests: list[generator_pb2.GenerateRequest] = []
        self.response_factory = response_factory

    async def info(self) -> generator_pb2.GeneratorInfo:
        self.events.append(f"info:{self.info_response.name}")
        return self.info_response

    async def generate(
        self,
        request: generator_pb2.GenerateRequest,
    ) -> generator_pb2.GenerateResponse:
        self.events.append(f"generate:{self.info_response.name}")
        self.requests.append(request)
        if self.response_factory is not None:
            return self.response_factory(request)
        return _response_for_request(
            self.info_response.name,
            request,
            list(self.info_response.artifacts),
        )


class RecordingRouter:
    def __init__(
        self,
        allocations: list[tuple[str, int]],
        *,
        events: list[str] | None = None,
    ) -> None:
        self.allocations = allocations
        self.events = events if events is not None else []
        self.route_requests: list[router_pb2.RouterRequest] = []
        self.feedback_requests: list[router_pb2.RouterFeedbackRequest] = []
        self.feedback_failures = 0

    async def route(self, request: router_pb2.RouterRequest) -> router_pb2.RouterResponse:
        self.events.append("route")
        self.route_requests.append(request)
        weights = [1.0 / len(self.allocations)] * len(self.allocations)
        rewards = [0.5] * len(self.allocations)
        return router_pb2.RouterResponse(
            selected_generators=[name for name, _count in self.allocations],
            selection_weights=weights,
            expected_rewards=rewards,
            allocations=[
                router_pb2.GeneratorAllocation(
                    generator_name=name,
                    n_samples=count,
                    normalized_weight=weight,
                    expected_reward=reward,
                )
                for (name, count), weight, reward in zip(
                    self.allocations,
                    weights,
                    rewards,
                    strict=True,
                )
            ],
            strategy="task_aware_router",
            state_version=7,
        )

    async def submit_feedback(
        self,
        request: router_pb2.RouterFeedbackRequest,
    ) -> router_pb2.RouterFeedbackResponse:
        self.feedback_requests.append(request)
        if self.feedback_failures:
            self.feedback_failures -= 1
            raise RuntimeError("router unavailable")
        return router_pb2.RouterFeedbackResponse(
            acknowledged=True,
            duplicate=False,
            state_version=len(self.feedback_requests),
        )


class NoopRepository:
    async def write_workflow_belief(self, **kwargs) -> None:
        return None

    async def get_run_crg(self, run_id: str) -> dict:
        return {}


class GeneratorInfoStub:
    def __init__(self, response: generator_pb2.GeneratorInfo) -> None:
        self.response = response

    async def Info(
        self,
        request: generator_pb2.GeneratorInfo,
        timeout: float | None = None,
    ) -> generator_pb2.GeneratorInfo:
        return self.response


def _agent(
    generator_clients: dict[str, object],
    router: RecordingRouter,
    **kwargs,
):
    return coordinator_module.GeneratorCoordAgent(
        generator_clients=generator_clients,
        router_client=router,
        crg_repository=NoopRepository(),
        **kwargs,
    )


def test_available_generators_match_task_router_names() -> None:
    agent = coordinator_module.GeneratorCoordAgent(
        router_client=RecordingRouter([("hfm_3d", 1)]),
        crg_repository=NoopRepository(),
    )

    assert agent.generators == list(GENERATOR_NAMES)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "info",
    [
        _info(
            "hfm_3d",
            runtime_status=audit_pb2.GENERATOR_RUNTIME_STATUS_UNAVAILABLE,
        ),
        generator_pb2.GeneratorInfo(
            name="hfm_3d",
            version="0.1.0",
            max_batch_size=0,
            runtime_status=audit_pb2.GENERATOR_RUNTIME_STATUS_READY,
            artifacts=[_artifact("hfm_3d")],
        ),
        generator_pb2.GeneratorInfo(
            name="hfm_3d",
            version="0.1.0",
            max_batch_size=1,
            runtime_status=audit_pb2.GENERATOR_RUNTIME_STATUS_READY,
        ),
        _info("fragfm"),
    ],
)
async def test_grpc_health_check_requires_strict_expected_info(
    info: generator_pb2.GeneratorInfo,
) -> None:
    client = coordinator_module.GeneratorGrpcClient.__new__(coordinator_module.GeneratorGrpcClient)
    client.generator_name = "hfm_3d"
    client.stub = GeneratorInfoStub(info)

    health = await client.health_check()

    assert health["healthy"] is False


@pytest.mark.asyncio
async def test_grpc_dict_generate_builds_complete_typed_request() -> None:
    seen: list[generator_pb2.GenerateRequest] = []

    class GenerateStub:
        async def Generate(
            self,
            request: generator_pb2.GenerateRequest,
            timeout: float | None = None,
        ) -> generator_pb2.GenerateResponse:
            seen.append(request)
            return generator_pb2.GenerateResponse()

    client = coordinator_module.GeneratorGrpcClient.__new__(coordinator_module.GeneratorGrpcClient)
    client.stub = GenerateStub()
    payload = _generation_payload(n_samples=2, n_select=1)

    await client.generate(payload)

    assert len(seen) == 1
    request = seen[0]
    assert request.project_id == payload["project_id"]
    assert request.request_id == payload["request_id"]
    assert request.cig == _cig_proto()
    assert list(request.hciv.coordinates) == _hciv_dict()["coordinates"]
    assert request.intent_cone == humu_pb2.IntentCone(
        axis=_cone_dict()["axis"],
        half_angle=0.25,
        curvature=1.0,
        property_weights={"qed": 1.0},
    ).SerializeToString(deterministic=True)
    assert request.context_schema_version == "generator_context.v1"


@pytest.mark.asyncio
async def test_info_barrier_precedes_single_route_and_excludes_unavailable_backend() -> None:
    events: list[str] = []
    hfm = RecordingGenerator(_info("hfm_3d", max_batch_size=4), events=events)
    frag = RecordingGenerator(
        _info(
            "fragfm",
            max_batch_size=2,
            runtime_status=audit_pb2.GENERATOR_RUNTIME_STATUS_UNAVAILABLE,
        ),
        events=events,
    )
    router = RecordingRouter([("hfm_3d", 3)], events=events)
    agent = _agent({"hfm_3d": hfm, "fragfm": frag}, router)

    result = await agent.process(_generation_payload(n_samples=3, n_select=1))

    assert events[:3] == ["info:hfm_3d", "info:fragfm", "route"]
    assert len(router.route_requests) == 1
    route_request = router.route_requests[0]
    assert list(route_request.available_generator_names) == ["hfm_3d"]
    assert route_request.n_samples == 3
    assert route_request.n_select == 1
    assert route_request.task_complexity == router_pb2.TASK_COMPLEXITY_HIGH
    assert list(route_request.hciv) == _hciv_dict()["coordinates"]
    assert route_request.cig == _cig_proto().SerializeToString(deterministic=True)
    assert not frag.requests
    assert len(result["candidates"]) == 3


@pytest.mark.asyncio
async def test_allocation_is_chunked_with_stable_ids_seeds_and_typed_context() -> None:
    hfm = RecordingGenerator(_info("hfm_3d", max_batch_size=2))
    router = RecordingRouter([("hfm_3d", 5)])
    agent = _agent({"hfm_3d": hfm}, router)

    first = await agent.process(_generation_payload(n_samples=5, n_select=1))

    assert [request.batch_size for request in hfm.requests] == [2, 2, 1]
    assert [request.request_id for request in hfm.requests] == [
        "request-1:hfm_3d:chunk-0000",
        "request-1:hfm_3d:chunk-0001",
        "request-1:hfm_3d:chunk-0002",
    ]
    assert [request.generator_params["chunk_seed"] for request in hfm.requests] == [
        "1000014",
        "1097423",
        "1194832",
    ]
    expected_cig = _cig_proto()
    expected_hciv = humu_pb2.HCIV(
        coordinates=_hciv_dict()["coordinates"],
        curvature=1.0,
    )
    expected_cone = humu_pb2.IntentCone(
        axis=_cone_dict()["axis"],
        half_angle=0.25,
        curvature=1.0,
        property_weights={"qed": 1.0},
    )
    for request in hfm.requests:
        assert request.context_schema_version == "generator_context.v1"
        assert request.cig == expected_cig
        assert request.hciv == expected_hciv
        assert request.intent_cone == expected_cone.SerializeToString(deterministic=True)
    assert len(first["candidates"]) == 5
    assert [candidate["chunk_id"] for candidate in first["candidates"]] == [
        "request-1:hfm_3d:chunk-0000",
        "request-1:hfm_3d:chunk-0000",
        "request-1:hfm_3d:chunk-0001",
        "request-1:hfm_3d:chunk-0001",
        "request-1:hfm_3d:chunk-0002",
    ]
    assert [candidate["chunk_seed"] for candidate in first["candidates"]] == [
        1000014,
        1000014,
        1097423,
        1097423,
        1194832,
    ]

    second_hfm = RecordingGenerator(_info("hfm_3d", max_batch_size=2))
    second = await _agent(
        {"hfm_3d": second_hfm},
        RecordingRouter([("hfm_3d", 5)]),
    ).process(_generation_payload(n_samples=5, n_select=1))

    assert [candidate["chunk_seed"] for candidate in second["candidates"]] == [
        candidate["chunk_seed"] for candidate in first["candidates"]
    ]


@pytest.mark.asyncio
async def test_different_generator_allocations_execute_concurrently() -> None:
    started: set[str] = set()
    both_started = asyncio.Event()

    class ConcurrentGenerator(RecordingGenerator):
        async def generate(
            self,
            request: generator_pb2.GenerateRequest,
        ) -> generator_pb2.GenerateResponse:
            started.add(self.info_response.name)
            if started == {"hfm_3d", "fragfm"}:
                both_started.set()
            await asyncio.wait_for(both_started.wait(), timeout=0.2)
            return await super().generate(request)

    hfm = ConcurrentGenerator(_info("hfm_3d"))
    frag = ConcurrentGenerator(_info("fragfm"))
    agent = _agent(
        {"hfm_3d": hfm, "fragfm": frag},
        RecordingRouter([("hfm_3d", 1), ("fragfm", 1)]),
    )

    result = await agent.process(_generation_payload(n_samples=2, n_select=2))

    assert started == {"hfm_3d", "fragfm"}
    assert len(result["candidates"]) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("allocations", "message"),
    [
        ([("hfm_3d", 1), ("hfm_3d", 1)], "duplicate"),
        ([("hfm_3d", 1)], "sum"),
        ([("iclm", 2)], "available"),
    ],
)
async def test_invalid_router_allocations_fail_before_generation(
    allocations: list[tuple[str, int]],
    message: str,
) -> None:
    hfm = RecordingGenerator(_info("hfm_3d"))
    agent = _agent({"hfm_3d": hfm}, RecordingRouter(allocations))

    with pytest.raises(RuntimeError, match=message):
        await agent.process(_generation_payload(n_samples=2, n_select=1))

    assert hfm.requests == []


@pytest.mark.asyncio
async def test_router_allocation_count_must_match_requested_n_select() -> None:
    hfm = RecordingGenerator(_info("hfm_3d"))
    frag = RecordingGenerator(_info("fragfm"))
    agent = _agent(
        {"hfm_3d": hfm, "fragfm": frag},
        RecordingRouter([("hfm_3d", 1), ("fragfm", 1)]),
    )

    with pytest.raises(RuntimeError, match="n_select"):
        await agent.process(_generation_payload(n_samples=2, n_select=1))

    assert hfm.requests == []
    assert frag.requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response", "message"),
    [
        (
            router_pb2.RouterResponse(
                selected_generators=["hfm_3d"],
                selection_weights=[],
                expected_rewards=[0.5],
                allocations=[
                    router_pb2.GeneratorAllocation(
                        generator_name="hfm_3d",
                        n_samples=2,
                        normalized_weight=1.0,
                        expected_reward=0.5,
                    )
                ],
            ),
            "selection_weights",
        ),
        (
            router_pb2.RouterResponse(
                selected_generators=["hfm_3d"],
                selection_weights=[1.0],
                expected_rewards=[],
                allocations=[
                    router_pb2.GeneratorAllocation(
                        generator_name="hfm_3d",
                        n_samples=2,
                        normalized_weight=1.0,
                        expected_reward=0.5,
                    )
                ],
            ),
            "expected_rewards",
        ),
        (
            router_pb2.RouterResponse(
                selected_generators=["hfm_3d"],
                selection_weights=[float("nan")],
                expected_rewards=[0.5],
                allocations=[
                    router_pb2.GeneratorAllocation(
                        generator_name="hfm_3d",
                        n_samples=2,
                        normalized_weight=float("nan"),
                        expected_reward=0.5,
                    )
                ],
            ),
            "finite",
        ),
        (
            router_pb2.RouterResponse(
                selected_generators=["hfm_3d", "fragfm"],
                selection_weights=[0.8, 0.3],
                expected_rewards=[0.5, 0.5],
                allocations=[
                    router_pb2.GeneratorAllocation(
                        generator_name="hfm_3d",
                        n_samples=1,
                        normalized_weight=0.8,
                        expected_reward=0.5,
                    ),
                    router_pb2.GeneratorAllocation(
                        generator_name="fragfm",
                        n_samples=1,
                        normalized_weight=0.3,
                        expected_reward=0.5,
                    ),
                ],
            ),
            "sum to one",
        ),
        (
            router_pb2.RouterResponse(
                selected_generators=["hfm_3d"],
                selection_weights=[1.0],
                expected_rewards=[0.5],
                allocations=[
                    router_pb2.GeneratorAllocation(
                        generator_name="hfm_3d",
                        n_samples=2,
                        normalized_weight=0.75,
                        expected_reward=0.5,
                    )
                ],
            ),
            "do not match",
        ),
    ],
)
async def test_router_response_requires_complete_consistent_numeric_contract(
    response: router_pb2.RouterResponse,
    message: str,
) -> None:
    class StaticRouter:
        async def route(self, _request: router_pb2.RouterRequest) -> router_pb2.RouterResponse:
            return response

    hfm = RecordingGenerator(_info("hfm_3d"))
    frag = RecordingGenerator(_info("fragfm"))
    agent = _agent({"hfm_3d": hfm, "fragfm": frag}, StaticRouter())

    with pytest.raises(RuntimeError, match=message):
        await agent.process(_generation_payload(n_samples=2, n_select=2))

    assert hfm.requests == []
    assert frag.requests == []


def _corrupt_response(
    mode: str,
    request: generator_pb2.GenerateRequest,
) -> generator_pb2.GenerateResponse:
    response = _response_for_request("hfm_3d", request, [_artifact("hfm_3d")])
    if mode == "request_id":
        response.request_id = "wrong"
    elif mode == "generator_name":
        response.generator_name = "fragfm"
    elif mode == "molecule_schema":
        response.molecule_payload_schema = "unknown"
    elif mode == "embedding_schema":
        response.embedding_payload_schema = "unknown"
    elif mode == "artifacts":
        response.artifacts[0].checksum = "sha256:wrong"
    elif mode == "json":
        response.molecules[0] = b"not-json"
    elif mode == "count":
        del response.molecules[-1]
    elif mode == "embedding_count":
        response.humu_embeddings.append(struct.pack("<129f", *([0.0] * 129)))
    elif mode == "embedding_size":
        response.humu_embeddings.extend([b"short"] * len(response.molecules))
    elif mode == "embedding_finite":
        invalid = struct.pack("<129f", float("nan"), *([0.0] * 128))
        response.humu_embeddings.extend([invalid] * len(response.molecules))
    else:
        raise AssertionError(mode)
    return response


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "message"),
    [
        ("request_id", "request_id"),
        ("generator_name", "generator_name"),
        ("molecule_schema", "molecule payload schema"),
        ("embedding_schema", "embedding payload schema"),
        ("artifacts", "artifacts"),
        ("json", "JSON"),
        ("count", "count"),
        ("embedding_count", "embedding count"),
        ("embedding_size", "516"),
        ("embedding_finite", "finite"),
    ],
)
async def test_strict_generate_response_rejects_contract_violation(
    mode: str,
    message: str,
) -> None:
    hfm = RecordingGenerator(
        _info("hfm_3d"),
        response_factory=lambda request: _corrupt_response(mode, request),
    )
    agent = _agent({"hfm_3d": hfm}, RecordingRouter([("hfm_3d", 2)]))

    with pytest.raises(RuntimeError, match=message):
        await agent.process(_generation_payload(n_samples=2, n_select=1))


@pytest.mark.asyncio
async def test_candidate_provenance_overrides_untrusted_payload_fields() -> None:
    packed_embedding = struct.pack("<129f", 1.0, *([0.0] * 128))

    def response(request: generator_pb2.GenerateRequest) -> generator_pb2.GenerateResponse:
        return _response_for_request(
            "hfm_3d",
            request,
            [_artifact("hfm_3d")],
            payload_factory=lambda _index: _molecule_payload(
                "CCO",
                extra={
                    "generator": "spoofed",
                    "generator_name": "spoofed",
                    "chunk_id": "spoofed",
                    "chunk_seed": -1,
                    "artifact_refs": [{"name": "spoofed"}],
                },
            ),
            embeddings=[packed_embedding] * request.batch_size,
        )

    hfm = RecordingGenerator(_info("hfm_3d"), response_factory=response)
    result = await _agent(
        {"hfm_3d": hfm},
        RecordingRouter([("hfm_3d", 1)]),
    ).process(_generation_payload(n_samples=1, n_select=1))

    candidate = result["candidates"][0]
    assert candidate["generator"] == "hfm_3d"
    assert candidate["generator_name"] == "hfm_3d"
    assert candidate["chunk_id"] == "request-1:hfm_3d:chunk-0000"
    assert candidate["chunk_seed"] == 1000014
    assert candidate["artifact_refs"] == [
        {
            "name": "hfm_3d_checkpoint",
            "version": "test-v1",
            "checksum": "sha256:hfm_3d",
            "required": True,
        }
    ]
    assert candidate["humu_embedding"] == pytest.approx([1.0, *([0.0] * 128)])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda payload: payload.pop("cig"), "cig"),
        (
            lambda payload: payload["hciv"].update({"coordinates": [1.0, 0.0]}),
            "129",
        ),
        (
            lambda payload: payload["hciv"].update({"coordinates": [float("nan"), *([0.0] * 128)]}),
            "finite",
        ),
        (
            lambda payload: payload["intent_cone"].update({"curvature": 2.0}),
            "curvature",
        ),
        (
            lambda payload: payload["hciv"].update({"coordinates": [2.0, *([0.0] * 128)]}),
            "Lorentz",
        ),
    ],
)
async def test_invalid_typed_context_fails_before_info(
    mutate: Callable[[dict], object],
    message: str,
) -> None:
    payload = _generation_payload(n_samples=1, n_select=1)
    mutate(payload)
    hfm = RecordingGenerator(_info("hfm_3d"))
    router = RecordingRouter([("hfm_3d", 1)])
    agent = _agent({"hfm_3d": hfm}, router)

    with pytest.raises(ValueError, match=message):
        await agent.process(payload)

    assert hfm.events == []
    assert router.route_requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mutate",
    [
        lambda payload: payload.pop("project_id"),
        lambda payload: payload.pop("request_id"),
        lambda payload: payload["cig"].update({"project_id": "other-project"}),
    ],
)
async def test_outer_identity_and_cig_project_fail_before_info(
    mutate: Callable[[dict], object],
) -> None:
    payload = _generation_payload(n_samples=1, n_select=1)
    mutate(payload)
    hfm = RecordingGenerator(_info("hfm_3d"))
    router = RecordingRouter([("hfm_3d", 1)])
    agent = _agent({"hfm_3d": hfm}, router)

    with pytest.raises(ValueError):
        await agent.process(payload)

    assert hfm.events == []
    assert router.route_requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_type", [1.5, "1"])
async def test_generation_rejects_ambiguous_integer_inputs_before_info(
    invalid_type: object,
) -> None:
    payload = _generation_payload(n_samples=1, n_select=1)
    payload["n_samples"] = invalid_type
    hfm = RecordingGenerator(_info("hfm_3d"))
    router = RecordingRouter([("hfm_3d", 1)])
    agent = _agent({"hfm_3d": hfm}, router)

    with pytest.raises(ValueError, match="positive integer"):
        await agent.process(payload)

    assert hfm.events == []
    assert router.route_requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_iteration", [1.5, "1"])
async def test_feedback_rejects_ambiguous_iteration(
    invalid_iteration: object,
) -> None:
    router = RecordingRouter([("hfm_3d", 1)])
    payload = _feedback_payload([_feedback_group()])
    payload["iteration"] = invalid_iteration
    agent = _agent(
        {},
        router,
        teacher_adapter=RecordingTeacherAdapter(
            {
                "teacher_score": 0.5,
                "teacher_source": "teacher",
                "teacher_version": "v1",
                "synthetic": False,
            }
        ),
    )

    with pytest.raises(ValueError, match="non-negative integer"):
        await agent.process(payload)

    assert router.feedback_requests == []


@pytest.mark.asyncio
async def test_task_profile_multi_target_must_be_boolean() -> None:
    payload = _generation_payload(n_samples=1, n_select=1)
    payload["task_profile"]["multi_target"] = "false"
    hfm = RecordingGenerator(_info("hfm_3d"))
    router = RecordingRouter([("hfm_3d", 1)])
    agent = _agent({"hfm_3d": hfm}, router)

    with pytest.raises(ValueError, match="multi_target"):
        await agent.process(payload)

    assert router.route_requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["nonfinite", "unknown_enum"])
async def test_direct_cig_proto_requires_finite_known_objectives_before_info(
    mode: str,
) -> None:
    payload = _generation_payload(n_samples=1, n_select=1)
    cig = _cig_proto()
    if mode == "nonfinite":
        cig.objectives[0].weight = float("nan")
    else:
        cig.objectives[0].type = 99
    payload["cig"] = cig
    hfm = RecordingGenerator(_info("hfm_3d"))
    router = RecordingRouter([("hfm_3d", 1)])
    agent = _agent({"hfm_3d": hfm}, router)

    with pytest.raises(ValueError):
        await agent.process(payload)

    assert hfm.events == []
    assert router.route_requests == []


class RecordingTeacherAdapter:
    def __init__(self, result: dict) -> None:
        self.result = result
        self.groups: list[dict] = []

    async def adapt(self, group: dict) -> dict:
        self.groups.append(group)
        return dict(self.result)


def _feedback_payload(groups: list[dict]) -> dict:
    return {
        "action": "generator_coord/feedback/v1",
        "run_id": "run-feedback",
        "request_id": "request-feedback",
        "iteration": 2,
        "groups": groups,
    }


def _feedback_group(
    generator_name: str = "hfm_3d",
    *,
    phase: str = "validation",
) -> dict:
    return {
        "phase": phase,
        "generator_name": generator_name,
        "canonical_smiles": "CCO",
        "candidate_ids": ["candidate-1"],
        "evidence_ids": ["evidence-1"],
        "records": [{"source": phase, "passed": True}],
    }


@pytest.mark.asyncio
async def test_feedback_accepts_zero_score_and_deduplicates_only_after_router_ack() -> None:
    router = RecordingRouter([("hfm_3d", 1)])
    adapter = RecordingTeacherAdapter(
        {
            "teacher_score": 0.0,
            "teacher_source": "test-teacher",
            "teacher_version": "v1",
            "synthetic": True,
        }
    )
    agent = _agent({}, router, teacher_adapter=adapter)
    group = _feedback_group()

    first = await agent.process(_feedback_payload([group, dict(group)]))
    second = await agent.process(_feedback_payload([group]))

    assert first["submitted"] == 1
    assert first["duplicates"] == 1
    assert second["submitted"] == 0
    assert second["duplicates"] == 1
    assert len(router.feedback_requests) == 1
    request = router.feedback_requests[0]
    assert request.HasField("teacher_score")
    assert request.teacher_score == 0.0
    assert request.phase == router_pb2.ROUTER_FEEDBACK_PHASE_VALIDATION
    assert request.synthetic is True
    assert request.teacher_source == "test-teacher"
    assert request.teacher_version == "v1"


@pytest.mark.asyncio
async def test_feedback_failure_remains_retryable_and_dedup_is_generator_scoped() -> None:
    router = RecordingRouter([("hfm_3d", 1)])
    router.feedback_failures = 1
    adapter = RecordingTeacherAdapter(
        {
            "teacher_score": 0.75,
            "teacher_source": "test-teacher",
            "teacher_version": "v1",
            "synthetic": False,
        }
    )
    agent = _agent({}, router, teacher_adapter=adapter)

    with pytest.raises(RuntimeError, match="router unavailable"):
        await agent.process(_feedback_payload([_feedback_group()]))

    retry = await agent.process(_feedback_payload([_feedback_group()]))
    other_generator = await agent.process(_feedback_payload([_feedback_group("fragfm")]))

    assert retry["submitted"] == 1
    assert other_generator["submitted"] == 1
    assert [request.generator_name for request in router.feedback_requests] == [
        "hfm_3d",
        "hfm_3d",
        "fragfm",
    ]


@pytest.mark.asyncio
async def test_router_reported_duplicate_is_counted_as_duplicate_and_remembered() -> None:
    class DuplicateRouter(RecordingRouter):
        async def submit_feedback(
            self,
            request: router_pb2.RouterFeedbackRequest,
        ) -> router_pb2.RouterFeedbackResponse:
            self.feedback_requests.append(request)
            return router_pb2.RouterFeedbackResponse(
                acknowledged=True,
                duplicate=True,
                state_version=7,
            )

    router = DuplicateRouter([("hfm_3d", 1)])
    adapter = RecordingTeacherAdapter(
        {
            "teacher_score": 0.75,
            "teacher_source": "test-teacher",
            "teacher_version": "v1",
            "synthetic": False,
        }
    )
    agent = _agent({}, router, teacher_adapter=adapter)

    first = await agent.process(_feedback_payload([_feedback_group()]))
    retry = await agent.process(_feedback_payload([_feedback_group()]))

    assert first["submitted"] == 0
    assert first["duplicates"] == 1
    assert retry["submitted"] == 0
    assert retry["duplicates"] == 1
    assert len(router.feedback_requests) == 1


@pytest.mark.asyncio
async def test_feedback_deduplication_is_bound_to_run_request_and_iteration() -> None:
    router = RecordingRouter([("hfm_3d", 1)])
    adapter = RecordingTeacherAdapter(
        {
            "teacher_score": 0.75,
            "teacher_source": "test-teacher",
            "teacher_version": "v1",
            "synthetic": False,
        }
    )
    agent = _agent({}, router, teacher_adapter=adapter)
    payloads = [
        _feedback_payload([_feedback_group()]),
        {**_feedback_payload([_feedback_group()]), "run_id": "other-run"},
        {**_feedback_payload([_feedback_group()]), "request_id": "other-request"},
        {**_feedback_payload([_feedback_group()]), "iteration": 3},
    ]

    results = [await agent.process(payload) for payload in payloads]

    assert [result["submitted"] for result in results] == [1, 1, 1, 1]
    assert len({request.feedback_id for request in router.feedback_requests}) == 4


@pytest.mark.asyncio
async def test_feedback_rejects_changed_content_for_the_same_identity() -> None:
    router = RecordingRouter([("hfm_3d", 1)])
    adapter = RecordingTeacherAdapter(
        {
            "teacher_score": 0.75,
            "teacher_source": "test-teacher",
            "teacher_version": "v1",
            "synthetic": False,
        }
    )
    agent = _agent({}, router, teacher_adapter=adapter)
    await agent.process(_feedback_payload([_feedback_group()]))
    changed = _feedback_group()
    changed["evidence_ids"] = ["different-evidence"]

    with pytest.raises(ValueError, match="different content"):
        await agent.process(_feedback_payload([changed]))

    assert len(router.feedback_requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("adapter_result", "message"),
    [
        (
            {
                "teacher_score": float("nan"),
                "teacher_source": "teacher",
                "teacher_version": "v1",
                "synthetic": False,
            },
            "finite",
        ),
        (
            {
                "teacher_score": 1.1,
                "teacher_source": "teacher",
                "teacher_version": "v1",
                "synthetic": False,
            },
            "\\[0, 1\\]",
        ),
        (
            {
                "teacher_score": 0.5,
                "teacher_source": "",
                "teacher_version": "v1",
                "synthetic": False,
            },
            "source",
        ),
        (
            {
                "teacher_score": 0.5,
                "teacher_source": "teacher",
                "teacher_version": "",
                "synthetic": False,
            },
            "version",
        ),
    ],
)
async def test_feedback_rejects_invalid_teacher_output(
    adapter_result: dict,
    message: str,
) -> None:
    router = RecordingRouter([("hfm_3d", 1)])
    agent = _agent(
        {},
        router,
        teacher_adapter=RecordingTeacherAdapter(adapter_result),
    )

    with pytest.raises(ValueError, match=message):
        await agent.process(_feedback_payload([_feedback_group(phase="critic")]))

    assert router.feedback_requests == []


@pytest.mark.asyncio
async def test_router_grpc_client_round_trips_route_and_feedback() -> None:
    seen: list[Message] = []

    class RouterService(router_pb2_grpc.GeneratorRouterServiceServicer):
        async def Route(self, request, context):  # noqa: N802
            seen.append(request)
            return router_pb2.RouterResponse(
                selected_generators=["hfm_3d"],
                selection_weights=[1.0],
                expected_rewards=[0.5],
                allocations=[
                    router_pb2.GeneratorAllocation(
                        generator_name="hfm_3d",
                        n_samples=1,
                    )
                ],
            )

        async def SubmitFeedback(self, request, context):  # noqa: N802
            seen.append(request)
            return router_pb2.RouterFeedbackResponse(acknowledged=True)

    server = grpc.aio.server()
    router_pb2_grpc.add_GeneratorRouterServiceServicer_to_server(RouterService(), server)
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    client = None
    try:
        client = coordinator_module.GeneratorRouterGrpcClient(f"127.0.0.1:{port}")
        route_response = await client.route(
            router_pb2.RouterRequest(
                request_id="route-1",
                n_select=1,
                n_samples=1,
            )
        )
        feedback_response = await client.submit_feedback(
            router_pb2.RouterFeedbackRequest(
                feedback_id="feedback-1",
                run_id="run-1",
                request_id="route-1",
                phase=router_pb2.ROUTER_FEEDBACK_PHASE_VALIDATION,
                generator_name="hfm_3d",
                canonical_smiles="CCO",
                teacher_score=0.5,
                teacher_source="teacher",
                teacher_version="v1",
            )
        )
    finally:
        if client is not None:
            await client.close()
        await server.stop(None)

    assert route_response.allocations[0].n_samples == 1
    assert feedback_response.acknowledged is True
    assert isinstance(seen[0], router_pb2.RouterRequest)
    assert isinstance(seen[1], router_pb2.RouterFeedbackRequest)


@pytest.mark.asyncio
async def test_local_uas_compatibility_is_never_ready_but_can_generate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command = (
        f'{sys.executable} -c "import json,sys;'
        "req=json.load(sys.stdin);"
        "assert req['generator']=='uas';"
        "print(json.dumps({'candidates':"
        " [{'id':'uas-1','smiles':'CCO','canonical_smiles':'CCO'}]}))\""
    )
    monkeypatch.setenv("UAS_RUNNER_COMMAND", command)

    client = coordinator_module.create_uas_generator_client()
    health = await client.health_check()
    result = await client.generate(
        {
            "cig": _cig_dict(),
            "hciv": _hciv_dict(),
            "intent_cone": _cone_dict(),
            "n_samples": 1,
            "generator_params": {"sampling_seed": 7},
        }
    )

    assert health["healthy"] is False
    assert "compatibility" in health["reason"]
    assert result["candidates"][0]["canonical_smiles"] == "CCO"


def test_builds_router_client_from_explicit_target(monkeypatch: pytest.MonkeyPatch) -> None:
    created_targets: list[str] = []

    class RouterClient:
        def __init__(self, target: str) -> None:
            created_targets.append(target)

    monkeypatch.setattr(
        coordinator_module,
        "GeneratorRouterGrpcClient",
        RouterClient,
        raising=False,
    )

    coordinator_module.GeneratorCoordAgent(
        router_target="router:50052",
        crg_repository=object(),
    )

    assert created_targets == ["router:50052"]


def test_builds_grpc_generator_client_with_expected_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[tuple[str, str]] = []

    class GeneratorClient:
        def __init__(self, target: str, generator_name: str) -> None:
            created.append((target, generator_name))

    monkeypatch.setattr(
        coordinator_module,
        "GeneratorGrpcClient",
        GeneratorClient,
    )

    clients = coordinator_module._build_generator_clients({"hfm_3d": "hfm:50051"})

    assert isinstance(clients["hfm_3d"], GeneratorClient)
    assert created == [("hfm:50051", "hfm_3d")]


def test_builds_generator_clients_from_python_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class GeneratorClient:
        async def generate(self, request: dict) -> dict:
            return {"candidates": [{"smiles": "CCO"}]}

    provider_module = ModuleType("test_generator_coord_python_client")
    provider_module.create_client = lambda: GeneratorClient()
    monkeypatch.setitem(sys.modules, provider_module.__name__, provider_module)

    agent = coordinator_module.GeneratorCoordAgent(
        generator_targets={
            "uas": f"python://{provider_module.__name__}:create_client",
        },
        router_client=RecordingRouter([("uas", 1)]),
        crg_repository=object(),
    )

    assert isinstance(agent.generator_clients["uas"], GeneratorClient)
