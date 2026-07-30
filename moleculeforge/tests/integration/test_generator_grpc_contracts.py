from __future__ import annotations

import importlib
import json
import math
import struct
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import grpc
import pytest
from mf_core.artifacts import RequirementStatus
from mf_core.plugins.generator import artifact_refs
from mf_core.proto_gen.moleculeforge.v1.core import audit_pb2, cig_pb2, humu_pb2
from mf_core.proto_gen.moleculeforge.v1.generator import generator_pb2, generator_pb2_grpc
from mf_core.types.molecule import Molecule


@dataclass(frozen=True)
class ServiceCase:
    module_name: str
    servicer_name: str
    generator_name: str
    max_batch_size: int


SERVICE_CASES = (
    ServiceCase("hfm_generator_svc.main", "HFMGeneratorServicer", "hfm_3d", 1024),
    ServiceCase("fragfm_generator_svc.main", "FragFMGeneratorServicer", "fragfm", 512),
    ServiceCase("crem_generator_svc.main", "CReMGeneratorServicer", "crem_3d", 256),
    ServiceCase("mmpt_generator_svc.main", "MMPTGeneratorServicer", "mmpt_rag", 256),
    ServiceCase("iclm_svc.main", "ICLMServicer", "iclm", 64),
)


class RecordingGenerator:
    def __init__(self, *, count: int = 2, smiles: str = "OCC") -> None:
        self.count = count
        self.smiles = smiles
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
        self.embedding = struct.pack("<129f", 1.0, *([0.0] * 128))

    async def generate(self, *args: object, **kwargs: object) -> list[Molecule]:
        self.calls.append((args, kwargs))
        return [
            Molecule(
                smiles=self.smiles,
                humu_embedding=self.embedding,
                metadata={"candidate_index": str(index)},
            )
            for index in range(self.count)
        ]

    async def info(self) -> dict[str, object]:
        return {
            "name": "recording",
            "version": "test",
            "description": "recording generator",
            "supported_properties": ["qed"],
            "max_batch_size": 7,
            "supports_streaming": True,
            "requires_gpu": False,
        }


def _valid_request(*, batch_size: int = 2) -> generator_pb2.GenerateRequest:
    cig = cig_pb2.CIG(
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
        hyperedges=[
            cig_pb2.ObjectiveHyperedge(
                source_ids=["qed"],
                target_ids=["qed"],
                relation="supports",
                strength=0.5,
            )
        ],
        created_by="test",
    )
    hciv = humu_pb2.HCIV(
        coordinates=[1.0, *([0.0] * 128)],
        curvature=1.0,
    )
    cone = humu_pb2.IntentCone(
        axis=[1.0, *([0.0] * 128)],
        half_angle=0.5,
        curvature=1.0,
        property_weights={"qed": 1.0},
    )
    return generator_pb2.GenerateRequest(
        project_id="project-1",
        request_id="request-1",
        batch_size=batch_size,
        total_molecules=batch_size,
        intent_cone=cone.SerializeToString(),
        cig=cig,
        hciv=hciv,
        context_schema_version="generator_context.v1",
        generator_params={"seed": "17"},
    )


def _available_status(path: Path) -> RequirementStatus:
    return RequirementStatus(
        name="test_checkpoint",
        configured=True,
        available=True,
        required=True,
        path=str(path),
        source="test",
        message="available",
    )


async def _start_service(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    case: ServiceCase,
    *,
    generated_count: int = 2,
    generator: object | None = None,
) -> tuple[
    grpc.aio.Server,
    grpc.aio.Channel,
    generator_pb2_grpc.GeneratorServiceStub,
    object,
]:
    artifact_path = tmp_path / f"{case.generator_name}.bin"
    artifact_path.write_bytes(b"artifact")
    statuses = [_available_status(artifact_path)]
    module = importlib.import_module(case.module_name)
    monkeypatch.setattr(module, "_require_runtime", lambda *args, **kwargs: statuses)
    monkeypatch.setattr(
        module,
        "_runtime_statuses",
        lambda *args, **kwargs: statuses,
        raising=False,
    )
    generator = generator or RecordingGenerator(count=generated_count)
    servicer = getattr(module, case.servicer_name)(generator=generator)
    server = grpc.aio.server()
    generator_pb2_grpc.add_GeneratorServiceServicer_to_server(servicer, server)
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    channel = grpc.aio.insecure_channel(f"127.0.0.1:{port}")
    return server, channel, generator_pb2_grpc.GeneratorServiceStub(channel), generator


@pytest.mark.parametrize("case", SERVICE_CASES, ids=lambda case: case.generator_name)
async def test_generator_service_real_grpc_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    case: ServiceCase,
) -> None:
    server, channel, stub, generator = await _start_service(monkeypatch, tmp_path, case)
    try:
        info = await stub.Info(generator_pb2.GeneratorInfo())
        response = await stub.Generate(_valid_request())
    finally:
        await channel.close()
        await server.stop(None)

    assert info.name == case.generator_name
    assert info.runtime_status == audit_pb2.GENERATOR_RUNTIME_STATUS_READY
    assert info.max_batch_size == 7
    assert len(info.artifacts) == 1
    assert info.artifacts[0].name == "test_checkpoint"
    assert info.artifacts[0].checksum.startswith("sha256:")

    assert isinstance(response, generator_pb2.GenerateResponse)
    assert response.generator_name == case.generator_name
    assert response.generation_id == "project-1"
    assert response.request_id == "request-1"
    assert response.molecule_payload_schema == "molecule.v1"
    assert response.embedding_payload_schema == "humu.float32.v1"
    assert len(response.molecules) == 2
    assert len(response.humu_embeddings) == 2
    assert list(response.artifacts) == list(info.artifacts)
    for payload in response.molecules:
        decoded = json.loads(payload.decode("utf-8"))
        assert isinstance(decoded, dict)
        assert decoded["smiles"] == "CCO"
        assert decoded["canonical_smiles"] == "CCO"
    for embedding in response.humu_embeddings:
        assert len(embedding) == 516
        assert all(math.isfinite(value) for value in struct.unpack("<129f", embedding))

    assert len(generator.calls) == 1
    if case.generator_name == "mmpt_rag":
        args, _kwargs = generator.calls[0]
        assert args[0] is not None
        assert args[1] is not None
        assert args[2] is not None
        assert len(args[2].hyperedges) == 1
        assert args[2].hyperedges[0].source_ids == ["qed"]
        assert args[2].hyperedges[0].target_ids == ["qed"]


def _clear_cig(request: generator_pb2.GenerateRequest) -> None:
    request.ClearField("cig")


def _clear_hciv(request: generator_pb2.GenerateRequest) -> None:
    request.ClearField("hciv")


def _clear_cone(request: generator_pb2.GenerateRequest) -> None:
    request.intent_cone = b""


def _short_hciv(request: generator_pb2.GenerateRequest) -> None:
    request.hciv.coordinates[:] = [1.0, 0.0]


def _nonfinite_hciv(request: generator_pb2.GenerateRequest) -> None:
    request.hciv.coordinates[1] = math.nan


def _off_manifold_hciv(request: generator_pb2.GenerateRequest) -> None:
    request.hciv.coordinates[0] = 2.0


def _mismatched_curvature(request: generator_pb2.GenerateRequest) -> None:
    cone = humu_pb2.IntentCone.FromString(request.intent_cone)
    cone.curvature = 2.0
    request.intent_cone = cone.SerializeToString()


def _short_cone(request: generator_pb2.GenerateRequest) -> None:
    cone = humu_pb2.IntentCone.FromString(request.intent_cone)
    cone.axis[:] = [1.0, 0.0]
    request.intent_cone = cone.SerializeToString()


def _nonfinite_cone(request: generator_pb2.GenerateRequest) -> None:
    cone = humu_pb2.IntentCone.FromString(request.intent_cone)
    cone.axis[1] = math.inf
    request.intent_cone = cone.SerializeToString()


def _off_manifold_cone(request: generator_pb2.GenerateRequest) -> None:
    cone = humu_pb2.IntentCone.FromString(request.intent_cone)
    cone.axis[0] = 2.0
    request.intent_cone = cone.SerializeToString()


def _invalid_context_schema(request: generator_pb2.GenerateRequest) -> None:
    request.context_schema_version = "generator_context.invalid"


def _invalid_cig(request: generator_pb2.GenerateRequest) -> None:
    request.cig.ClearField("objectives")


def _invalid_cig_hyperedge(request: generator_pb2.GenerateRequest) -> None:
    request.cig.hyperedges[0].target_ids[:] = ["missing-objective"]


INVALID_REQUESTS: tuple[tuple[str, Callable[[generator_pb2.GenerateRequest], None]], ...] = (
    ("missing-cig", _clear_cig),
    ("missing-hciv", _clear_hciv),
    ("missing-cone", _clear_cone),
    ("short-hciv", _short_hciv),
    ("nonfinite-hciv", _nonfinite_hciv),
    ("off-manifold-hciv", _off_manifold_hciv),
    ("curvature-mismatch", _mismatched_curvature),
    ("short-cone", _short_cone),
    ("nonfinite-cone", _nonfinite_cone),
    ("off-manifold-cone", _off_manifold_cone),
    ("invalid-context-schema", _invalid_context_schema),
    ("invalid-cig", _invalid_cig),
    ("invalid-cig-hyperedge", _invalid_cig_hyperedge),
)


@pytest.mark.parametrize("case", SERVICE_CASES, ids=lambda case: case.generator_name)
@pytest.mark.parametrize(
    "invalid_name,mutate",
    INVALID_REQUESTS,
    ids=[name for name, _mutate in INVALID_REQUESTS],
)
async def test_generator_service_rejects_invalid_context_before_generation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    case: ServiceCase,
    invalid_name: str,
    mutate: Callable[[generator_pb2.GenerateRequest], None],
) -> None:
    del invalid_name
    server, channel, stub, generator = await _start_service(monkeypatch, tmp_path, case)
    request = _valid_request()
    mutate(request)
    try:
        with pytest.raises(grpc.aio.AioRpcError) as exc_info:
            await stub.Generate(request)
    finally:
        await channel.close()
        await server.stop(None)

    assert exc_info.value.code() == grpc.StatusCode.INVALID_ARGUMENT
    assert generator.calls == []


@pytest.mark.parametrize("case", SERVICE_CASES, ids=lambda case: case.generator_name)
async def test_generator_service_rejects_partial_result_count(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    case: ServiceCase,
) -> None:
    server, channel, stub, _generator = await _start_service(
        monkeypatch,
        tmp_path,
        case,
        generated_count=1,
    )
    try:
        with pytest.raises(grpc.aio.AioRpcError) as exc_info:
            await stub.Generate(_valid_request(batch_size=2))
    finally:
        await channel.close()
        await server.stop(None)

    assert exc_info.value.code() == grpc.StatusCode.INTERNAL


@pytest.mark.parametrize("case", SERVICE_CASES, ids=lambda case: case.generator_name)
async def test_generator_service_rejects_batch_larger_than_service_limit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    case: ServiceCase,
) -> None:
    server, channel, stub, generator = await _start_service(monkeypatch, tmp_path, case)
    try:
        with pytest.raises(grpc.aio.AioRpcError) as exc_info:
            await stub.Generate(_valid_request(batch_size=case.max_batch_size + 1))
    finally:
        await channel.close()
        await server.stop(None)

    assert exc_info.value.code() == grpc.StatusCode.INVALID_ARGUMENT
    assert generator.calls == []


class NonIterableGenerator(RecordingGenerator):
    async def generate(self, *args: object, **kwargs: object) -> object:
        self.calls.append((args, kwargs))
        return object()


@pytest.mark.parametrize("case", SERVICE_CASES, ids=lambda case: case.generator_name)
async def test_generator_service_maps_non_iterable_model_result_to_internal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    case: ServiceCase,
) -> None:
    server, channel, stub, _generator = await _start_service(
        monkeypatch,
        tmp_path,
        case,
        generator=NonIterableGenerator(),
    )
    try:
        with pytest.raises(grpc.aio.AioRpcError) as exc_info:
            await stub.Generate(_valid_request())
    finally:
        await channel.close()
        await server.stop(None)

    assert exc_info.value.code() == grpc.StatusCode.INTERNAL


@pytest.mark.parametrize("case", SERVICE_CASES, ids=lambda case: case.generator_name)
async def test_generator_service_rejects_invalid_smiles_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    case: ServiceCase,
) -> None:
    server, channel, stub, _generator = await _start_service(
        monkeypatch,
        tmp_path,
        case,
        generator=RecordingGenerator(smiles="not-a-smiles"),
    )
    try:
        with pytest.raises(grpc.aio.AioRpcError) as exc_info:
            await stub.Generate(_valid_request())
    finally:
        await channel.close()
        await server.stop(None)

    assert exc_info.value.code() == grpc.StatusCode.INTERNAL


def test_artifact_checksum_changes_when_file_content_changes(tmp_path: Path) -> None:
    artifact_path = tmp_path / "checkpoint.bin"
    artifact_path.write_bytes(b"first")
    status = _available_status(artifact_path)

    first = artifact_refs([status])[0].checksum
    artifact_path.write_bytes(b"second-and-longer")
    second = artifact_refs([status])[0].checksum

    assert first != second


def _valid_model_update_request() -> generator_pb2.ModelUpdateRequest:
    return generator_pb2.ModelUpdateRequest(
        run_id="run-1",
        request_id="update-1",
        training_batch_json=json.dumps(
            {
                "schema_version": "training-batch.v1",
                "samples": [
                    {"candidate_id": "candidate-1", "smiles": "CCO", "reward": 0.8},
                    {"candidate_id": "candidate-2", "smiles": "CCN", "reward": 0.6},
                ],
                "kd_weight": 0.5,
            },
            sort_keys=True,
        ),
        teacher_embeddings=struct.pack("<4f", 0.1, 0.2, 0.3, 0.4),
        rows=2,
        dim=2,
        teacher_source="teacher",
        teacher_version="teacher-v1",
        target_checkpoint_version="iclm-v2",
    )


class RecordingOnlineLearner:
    def __init__(self, checkpoint_path: Path) -> None:
        self.checkpoint_path = checkpoint_path
        self.calls: list[dict[str, object]] = []

    def update(self, payload: dict[str, object]) -> dict[str, object]:
        self.calls.append(payload)
        return {
            "checkpoint_path": str(self.checkpoint_path),
            "updated_samples": 2,
        }


async def _start_incremental_service(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> tuple[
    grpc.aio.Server,
    grpc.aio.Channel,
    generator_pb2_grpc.IncrementalGeneratorServiceStub,
    RecordingOnlineLearner,
]:
    module = importlib.import_module("iclm_svc.main")
    active_checkpoint_path = tmp_path / "iclm-active"
    active_checkpoint_path.write_bytes(b"active-checkpoint")
    checkpoint_path = tmp_path / "iclm-checkpoint"
    checkpoint_path.write_bytes(b"updated-checkpoint")
    monkeypatch.setenv("ICLM_CHECKPOINT_DIRECTORY", str(tmp_path))
    monkeypatch.setenv("INTERNAL_SERVICE_TOKEN", "iclm-service-token")
    status = _available_status(checkpoint_path)
    monkeypatch.setattr(module, "_require_runtime", lambda *args, **kwargs: [status])
    learner = RecordingOnlineLearner(checkpoint_path)
    active_generator = type(
        "Generator",
        (),
        {
            "checkpoint_path": str(active_checkpoint_path),
        },
    )()
    training_generator = type(
        "TrainingGenerator",
        (),
        {
            "online_learner": learner,
            "checkpoint_path": str(active_checkpoint_path),
        },
    )()
    activated_generator = type(
        "ActivatedGenerator",
        (),
        {
            "checkpoint_path": str(checkpoint_path),
            "validate_checkpoint": lambda self: None,
        },
    )()
    factory_results = iter((training_generator, activated_generator))
    servicer = module.ICLMServicer(
        generator=active_generator,
        generator_factory=lambda checkpoint: next(factory_results),
    )
    server = grpc.aio.server()
    generator_pb2_grpc.add_IncrementalGeneratorServiceServicer_to_server(servicer, server)
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    channel = grpc.aio.insecure_channel(f"127.0.0.1:{port}")
    return (
        server,
        channel,
        generator_pb2_grpc.IncrementalGeneratorServiceStub(channel),
        learner,
    )


async def test_iclm_incremental_service_real_grpc_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    server, channel, stub, learner = await _start_incremental_service(
        monkeypatch,
        tmp_path,
    )
    try:
        response = await stub.UpdateModel(
            _valid_model_update_request(),
            metadata=(
                ("x-moleculeforge-service-token", "iclm-service-token"),
            ),
        )
    finally:
        await channel.close()
        await server.stop(None)

    assert isinstance(response, generator_pb2.ModelUpdateResponse)
    assert response.acknowledged is True
    assert response.active_version == "iclm-v2"
    assert response.updated_samples == 2
    assert [(item.name, item.version) for item in response.artifacts] == [
        ("iclm_checkpoint", "iclm-v2"),
        ("teacher", "teacher-v1"),
    ]
    assert all(item.checksum.startswith("sha256:") for item in response.artifacts)
    assert learner.calls == [
        {
            "schema_version": "training-batch.v1",
            "samples": [
                {
                    "candidate_id": "candidate-1",
                    "smiles": "CCO",
                    "reward": 0.8,
                },
                {
                    "candidate_id": "candidate-2",
                    "smiles": "CCN",
                    "reward": 0.6,
                },
            ],
            "teacher_weight": 0.5,
            "run_id": "run-1",
            "request_id": "update-1",
            "teacher_embeddings": [
                [pytest.approx(0.1), pytest.approx(0.2)],
                [pytest.approx(0.3), pytest.approx(0.4)],
            ],
            "teacher_source": "teacher",
            "teacher_version": "teacher-v1",
            "target_checkpoint_version": "iclm-v2",
        }
    ]


async def test_iclm_incremental_service_rejects_unauthenticated_real_grpc_update(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    server, channel, stub, learner = await _start_incremental_service(
        monkeypatch,
        tmp_path,
    )
    try:
        with pytest.raises(grpc.aio.AioRpcError) as missing:
            await stub.UpdateModel(_valid_model_update_request())
        with pytest.raises(grpc.aio.AioRpcError) as invalid:
            await stub.UpdateModel(
                _valid_model_update_request(),
                metadata=(
                    ("x-moleculeforge-service-token", "wrong-token"),
                ),
            )
    finally:
        await channel.close()
        await server.stop(None)

    assert missing.value.code() is grpc.StatusCode.UNAUTHENTICATED
    assert invalid.value.code() is grpc.StatusCode.UNAUTHENTICATED
    assert learner.calls == []


def _empty_update_run_id(request: generator_pb2.ModelUpdateRequest) -> None:
    request.run_id = ""


def _zero_update_rows(request: generator_pb2.ModelUpdateRequest) -> None:
    request.rows = 0


def _short_teacher_embeddings(request: generator_pb2.ModelUpdateRequest) -> None:
    request.teacher_embeddings = b"short"


def _nonfinite_teacher_embeddings(request: generator_pb2.ModelUpdateRequest) -> None:
    request.teacher_embeddings = struct.pack("<4f", math.nan, 0.2, 0.3, 0.4)


def _invalid_training_batch_json(request: generator_pb2.ModelUpdateRequest) -> None:
    request.training_batch_json = "not-json"


def _mismatched_training_rows(request: generator_pb2.ModelUpdateRequest) -> None:
    payload = json.loads(request.training_batch_json)
    payload["samples"].pop()
    request.training_batch_json = json.dumps(payload)


@pytest.mark.parametrize(
    "mutate",
    (
        _empty_update_run_id,
        _zero_update_rows,
        _short_teacher_embeddings,
        _nonfinite_teacher_embeddings,
        _invalid_training_batch_json,
        _mismatched_training_rows,
    ),
)
async def test_iclm_incremental_service_rejects_invalid_request_before_update(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mutate: Callable[[generator_pb2.ModelUpdateRequest], None],
) -> None:
    server, channel, stub, learner = await _start_incremental_service(
        monkeypatch,
        tmp_path,
    )
    request = _valid_model_update_request()
    mutate(request)
    try:
        with pytest.raises(grpc.aio.AioRpcError) as exc_info:
            await stub.UpdateModel(
                request,
                metadata=(
                    ("x-moleculeforge-service-token", "iclm-service-token"),
                ),
            )
    finally:
        await channel.close()
        await server.stop(None)

    assert exc_info.value.code() == grpc.StatusCode.INVALID_ARGUMENT
    assert learner.calls == []
