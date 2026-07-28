from __future__ import annotations

import importlib
import json
import math
import struct
from contextlib import asynccontextmanager
from types import SimpleNamespace

import grpc
import pytest
from google.protobuf.message import Message
from mf_core.proto_gen.moleculeforge.v1.core import (
    audit_pb2,
    cig_pb2,
    molecule_pb2,
)
from mf_core.proto_gen.moleculeforge.v1.core import (
    humu_pb2 as core_humu_pb2,
)
from mf_core.proto_gen.moleculeforge.v1.generator import (
    generator_pb2,
    generator_pb2_grpc,
    router_pb2,
)
from mf_core.proto_gen.moleculeforge.v1.humu import encoder_pb2
from mf_core.proto_gen.moleculeforge.v1.oracle import oracle_pb2

_PROTO_MODULE_ROOT = "mf_core.proto_gen.moleculeforge.v1"


def _supply_modules():
    return (
        importlib.import_module(f"{_PROTO_MODULE_ROOT}.oracle.supply_pb2"),
        importlib.import_module(f"{_PROTO_MODULE_ROOT}.oracle.supply_pb2_grpc"),
    )


def _field_kind(field) -> str:
    if field.message_type is not None:
        return field.message_type.full_name
    if field.enum_type is not None:
        return field.enum_type.full_name
    return {
        field.TYPE_BOOL: "bool",
        field.TYPE_BYTES: "bytes",
        field.TYPE_DOUBLE: "double",
        field.TYPE_FLOAT: "float",
        field.TYPE_INT32: "int32",
        field.TYPE_INT64: "int64",
        field.TYPE_STRING: "string",
        field.TYPE_UINT32: "uint32",
        field.TYPE_UINT64: "uint64",
    }[field.type]


def _assert_fields(message_type, expected: dict[str, tuple[int, str, bool]]) -> None:
    actual = {
        field.name: (field.number, _field_kind(field), field.is_repeated)
        for field in message_type.DESCRIPTOR.fields
    }
    assert actual == expected


def _assert_rpc_contract(
    file_descriptor,
    service_name: str,
    expected: dict[str, tuple[str, str, bool, bool]],
) -> None:
    service = file_descriptor.services_by_name[service_name]
    actual = {
        method.name: (
            method.input_type.full_name,
            method.output_type.full_name,
            method.client_streaming,
            method.server_streaming,
        )
        for method in service.methods
    }
    assert actual == expected


def _protobuf_round_trip(message):
    restored = type(message)()
    restored.ParseFromString(message.SerializeToString())
    return restored


@asynccontextmanager
async def _running_server(register):
    server = grpc.aio.server()
    register(server)
    port = server.add_insecure_port("127.0.0.1:0")
    assert port > 0
    await server.start()
    channel = grpc.aio.insecure_channel(f"127.0.0.1:{port}")
    await channel.channel_ready()
    try:
        yield channel
    finally:
        await channel.close()
        await server.stop(None)


def _validate_model_update_request(request) -> dict:
    rows = int(request.rows)
    dim = int(request.dim)
    if rows <= 0:
        raise ValueError("rows must be greater than zero")
    if dim <= 0:
        raise ValueError("dim must be greater than zero")
    teacher_embeddings = bytes(request.teacher_embeddings)
    if len(teacher_embeddings) != rows * dim * 4:
        raise ValueError("teacher_embeddings byte length must equal rows * dim * 4")
    raw_json = request.training_batch_json
    if isinstance(raw_json, bytes):
        try:
            raw_json = raw_json.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("training_batch_json must be UTF-8") from exc
    try:
        payload = json.loads(raw_json)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("training_batch_json must be valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("training_batch_json must contain a JSON object")
    schema_version = payload.get("schema_version")
    if not isinstance(schema_version, str) or not schema_version.strip():
        raise ValueError("training_batch_json requires a nonempty schema_version")
    return payload


def _consume_humu_response(response) -> tuple[float, ...]:
    if response.embedding_dimension != 129:
        raise ValueError("embedding_dimension must be 129")
    if len(response.humu_embedding) != 129 * 4:
        raise ValueError("humu_embedding must contain 129 float32 values")
    coordinates = struct.unpack("<129f", response.humu_embedding)
    if not all(math.isfinite(value) for value in coordinates):
        raise ValueError("humu_embedding must contain only finite values")
    return coordinates


def test_supply_generated_contract_is_importable() -> None:
    supply_pb2, supply_pb2_grpc = _supply_modules()

    assert issubclass(supply_pb2.AvailabilityResponse, Message)
    assert hasattr(supply_pb2_grpc, "SupplyOracleServiceStub")


def test_legacy_field_numbers_remain_unchanged() -> None:
    expected = (
        (
            molecule_pb2.MolecularProperties,
            {
                "mw": 1,
                "logp": 2,
                "hbd": 3,
                "hba": 4,
                "tpsa": 5,
                "qed": 6,
                "sa_score": 7,
                "ecfp4": 8,
            },
        ),
        (
            molecule_pb2.Molecule,
            {
                "smiles": 1,
                "properties": 2,
                "inchi_key": 3,
                "sdf_bytes": 4,
                "humu_embedding": 5,
                "metadata": 6,
            },
        ),
        (molecule_pb2.MoleculeBatch, {"molecules": 1, "batch_id": 2}),
        (
            audit_pb2.AuditEvent,
            {
                "trace_id": 1,
                "event_id": 2,
                "timestamp": 3,
                "actor": 4,
                "action": 5,
                "target": 6,
                "outcome": 7,
                "signature": 8,
                "signature_uri": 9,
                "lineage": 10,
            },
        ),
        (
            audit_pb2.AuditQuery,
            {
                "trace_id": 1,
                "project_id": 2,
                "start_time_ns": 3,
                "end_time_ns": 4,
                "actors": 5,
                "limit": 6,
            },
        ),
        (
            audit_pb2.AuditReport,
            {"events": 1, "total_count": 2, "has_more": 3, "next_page_token": 4},
        ),
        (
            generator_pb2.GenerateRequest,
            {
                "project_id": 1,
                "batch_size": 2,
                "total_molecules": 3,
                "intent_cone": 4,
                "target_properties": 5,
                "property_targets": 6,
                "checkpoint_version": 7,
                "generator_params": 8,
                "timeout_seconds": 9,
            },
        ),
        (
            generator_pb2.GenerateResponse,
            {
                "generator_name": 1,
                "generation_id": 2,
                "molecules": 3,
                "humu_embeddings": 4,
                "aggregate_stats": 5,
                "elapsed_ms": 6,
            },
        ),
        (
            generator_pb2.GeneratorInfo,
            {
                "name": 1,
                "version": 2,
                "description": 3,
                "supported_properties": 4,
                "max_batch_size": 5,
                "supports_streaming": 6,
                "requires_gpu": 7,
                "default_params": 8,
            },
        ),
        (
            router_pb2.RouterRequest,
            {
                "project_id": 1,
                "cig": 2,
                "generator_weights": 3,
                "generator_performance": 4,
                "n_select": 5,
                "hciv": 6,
                "target_family": 7,
                "stage": 8,
                "data_richness": 9,
                "novelty_demand": 10,
                "multi_target": 11,
                "sa_constraint": 12,
                "n_samples": 13,
            },
        ),
        (
            router_pb2.RouterResponse,
            {
                "selected_generators": 1,
                "selection_weights": 2,
                "strategy": 3,
                "expected_rewards": 4,
            },
        ),
        (
            router_pb2.RouterProxylessSearchRequest,
            {
                "reward_batches_json": 1,
                "generator_costs_json": 2,
                "cost_weight": 3,
                "learning_rate": 4,
                "temperature": 5,
            },
        ),
        (
            router_pb2.RouterProxylessSearchResponse,
            {
                "acknowledged": 1,
                "result_json": 2,
                "generator_names": 3,
                "architecture_probabilities": 4,
                "round_count": 5,
            },
        ),
        (
            oracle_pb2.OracleEvaluation,
            {
                "oracle_name": 1,
                "molecule_smiles": 2,
                "level": 3,
                "scores": 4,
                "uncertainties": 5,
                "elapsed_ms": 6,
                "success": 7,
                "error_message": 8,
            },
        ),
        (
            oracle_pb2.OracleBatchRequest,
            {
                "project_id": 1,
                "molecule_smiles": 2,
                "level": 3,
                "requested_properties": 4,
                "return_uncertainty": 5,
            },
        ),
        (
            oracle_pb2.OracleBatchResponse,
            {"evaluations": 1, "batch_id": 2, "total_elapsed_ms": 3},
        ),
        (
            encoder_pb2.EncodeRequest,
            {
                "entity_type": 1,
                "input_data": 2,
                "params": 3,
                "checkpoint_version": 4,
            },
        ),
        (
            encoder_pb2.EncodeResponse,
            {"humu_embedding": 1, "curvature": 2, "elapsed_ms": 3},
        ),
        (
            encoder_pb2.BatchEncodeRequest,
            {"requests": 1, "batch_id": 2},
        ),
        (
            encoder_pb2.BatchEncodeResponse,
            {"responses": 1, "batch_id": 2, "total_elapsed_ms": 3},
        ),
    )

    for message_type, field_numbers in expected:
        assert {
            field.name: field.number
            for field in message_type.DESCRIPTOR.fields
            if field.number in field_numbers.values()
        } == field_numbers


def test_shared_audit_types_have_exact_contract() -> None:
    _assert_fields(
        audit_pb2.ArtifactRef,
        {
            "name": (1, "string", False),
            "version": (2, "string", False),
            "checksum": (3, "string", False),
            "required": (4, "bool", False),
        },
    )
    assert {
        value.name: value.number for value in audit_pb2.GeneratorRuntimeStatus.DESCRIPTOR.values
    } == {
        "GENERATOR_RUNTIME_STATUS_UNSPECIFIED": 0,
        "GENERATOR_RUNTIME_STATUS_READY": 1,
        "GENERATOR_RUNTIME_STATUS_UNAVAILABLE": 2,
    }


def test_generator_descriptors_have_exact_appended_contract() -> None:
    _assert_fields(
        generator_pb2.GenerateRequest,
        {
            "project_id": (1, "string", False),
            "batch_size": (2, "int32", False),
            "total_molecules": (3, "int32", False),
            "intent_cone": (4, "bytes", False),
            "target_properties": (5, "string", True),
            "property_targets": (
                6,
                "moleculeforge.v1.generator.GenerateRequest.PropertyTargetsEntry",
                True,
            ),
            "checkpoint_version": (7, "string", False),
            "generator_params": (
                8,
                "moleculeforge.v1.generator.GenerateRequest.GeneratorParamsEntry",
                True,
            ),
            "timeout_seconds": (9, "int64", False),
            "request_id": (10, "string", False),
            "cig": (11, "moleculeforge.v1.core.CIG", False),
            "hciv": (12, "moleculeforge.v1.core.HCIV", False),
            "context_schema_version": (13, "string", False),
        },
    )
    _assert_fields(
        generator_pb2.GenerateResponse,
        {
            "generator_name": (1, "string", False),
            "generation_id": (2, "string", False),
            "molecules": (3, "bytes", True),
            "humu_embeddings": (4, "bytes", True),
            "aggregate_stats": (
                5,
                "moleculeforge.v1.generator.GenerateResponse.AggregateStatsEntry",
                True,
            ),
            "elapsed_ms": (6, "int64", False),
            "request_id": (7, "string", False),
            "artifacts": (8, "moleculeforge.v1.core.ArtifactRef", True),
            "molecule_payload_schema": (9, "string", False),
            "embedding_payload_schema": (10, "string", False),
        },
    )
    _assert_fields(
        generator_pb2.GeneratorInfo,
        {
            "name": (1, "string", False),
            "version": (2, "string", False),
            "description": (3, "string", False),
            "supported_properties": (4, "string", True),
            "max_batch_size": (5, "int32", False),
            "supports_streaming": (6, "bool", False),
            "requires_gpu": (7, "bool", False),
            "default_params": (
                8,
                "moleculeforge.v1.generator.GeneratorInfo.DefaultParamsEntry",
                True,
            ),
            "runtime_status": (
                9,
                "moleculeforge.v1.core.GeneratorRuntimeStatus",
                False,
            ),
            "status_message": (10, "string", False),
            "artifacts": (11, "moleculeforge.v1.core.ArtifactRef", True),
        },
    )
    _assert_fields(
        generator_pb2.ModelUpdateRequest,
        {
            "run_id": (1, "string", False),
            "request_id": (2, "string", False),
            "training_batch_json": (3, "string", False),
            "teacher_embeddings": (4, "bytes", False),
            "rows": (5, "uint32", False),
            "dim": (6, "uint32", False),
            "teacher_source": (7, "string", False),
            "teacher_version": (8, "string", False),
            "target_checkpoint_version": (9, "string", False),
        },
    )
    _assert_fields(
        generator_pb2.ModelUpdateResponse,
        {
            "acknowledged": (1, "bool", False),
            "active_version": (2, "string", False),
            "artifacts": (3, "moleculeforge.v1.core.ArtifactRef", True),
            "updated_samples": (4, "uint32", False),
        },
    )


def test_generator_services_preserve_old_rpcs_and_add_incremental_update() -> None:
    _assert_rpc_contract(
        generator_pb2.DESCRIPTOR,
        "GeneratorService",
        {
            "Generate": (
                "moleculeforge.v1.generator.GenerateRequest",
                "moleculeforge.v1.generator.GenerateResponse",
                False,
                False,
            ),
            "GenerateStream": (
                "moleculeforge.v1.generator.GenerateRequest",
                "moleculeforge.v1.generator.GenerateResponse",
                True,
                True,
            ),
            "BatchGenerate": (
                "moleculeforge.v1.generator.GenerateRequest",
                "moleculeforge.v1.generator.GenerateResponse",
                True,
                True,
            ),
            "Info": (
                "moleculeforge.v1.generator.GeneratorInfo",
                "moleculeforge.v1.generator.GeneratorInfo",
                False,
                False,
            ),
        },
    )
    _assert_rpc_contract(
        generator_pb2.DESCRIPTOR,
        "IncrementalGeneratorService",
        {
            "UpdateModel": (
                "moleculeforge.v1.generator.ModelUpdateRequest",
                "moleculeforge.v1.generator.ModelUpdateResponse",
                False,
                False,
            )
        },
    )


def test_generator_context_and_artifacts_survive_serialization() -> None:
    request = generator_pb2.GenerateRequest(
        project_id="project-1",
        request_id="request-1",
        cig=cig_pb2.CIG(project_id="project-1", constraints={"mw": "<500"}),
        hciv=core_humu_pb2.HCIV(
            coordinates=[1.0, *([0.0] * 128)],
            curvature=1.0,
        ),
        context_schema_version="generator_context.v1",
    )
    response = generator_pb2.GenerateResponse(
        request_id="request-1",
        artifacts=[
            audit_pb2.ArtifactRef(
                name="checkpoint",
                version="checkpoint-v1",
                checksum="sha256:abc",
                required=True,
            )
        ],
        molecule_payload_schema="molecule.v1",
        embedding_payload_schema="humu.float32.v1",
    )

    restored_request = _protobuf_round_trip(request)
    restored_response = _protobuf_round_trip(response)

    assert restored_request.cig.project_id == "project-1"
    assert len(restored_request.hciv.coordinates) == 129
    assert restored_request.context_schema_version == "generator_context.v1"
    assert restored_response.artifacts[0] == response.artifacts[0]
    assert isinstance(restored_response, Message)


def test_router_descriptors_have_exact_appended_contract() -> None:
    assert {value.name: value.number for value in router_pb2.TaskComplexity.DESCRIPTOR.values} == {
        "TASK_COMPLEXITY_UNSPECIFIED": 0,
        "TASK_COMPLEXITY_LOW": 1,
        "TASK_COMPLEXITY_MEDIUM": 2,
        "TASK_COMPLEXITY_HIGH": 3,
    }
    assert {
        value.name: value.number for value in router_pb2.RouterFeedbackPhase.DESCRIPTOR.values
    } == {
        "ROUTER_FEEDBACK_PHASE_UNSPECIFIED": 0,
        "ROUTER_FEEDBACK_PHASE_VALIDATION": 1,
        "ROUTER_FEEDBACK_PHASE_CRITIC": 2,
    }
    _assert_fields(
        router_pb2.RouterRequest,
        {
            "project_id": (1, "string", False),
            "cig": (2, "bytes", False),
            "generator_weights": (3, "double", True),
            "generator_performance": (4, "double", True),
            "n_select": (5, "int32", False),
            "hciv": (6, "double", True),
            "target_family": (7, "string", False),
            "stage": (8, "string", False),
            "data_richness": (9, "double", False),
            "novelty_demand": (10, "double", False),
            "multi_target": (11, "bool", False),
            "sa_constraint": (12, "double", False),
            "n_samples": (13, "int32", False),
            "request_id": (14, "string", False),
            "available_generator_names": (15, "string", True),
            "task_complexity": (
                16,
                "moleculeforge.v1.generator.TaskComplexity",
                False,
            ),
        },
    )
    _assert_fields(
        router_pb2.GeneratorAllocation,
        {
            "generator_name": (1, "string", False),
            "n_samples": (2, "uint32", False),
            "normalized_weight": (3, "double", False),
            "expected_reward": (4, "double", False),
        },
    )
    _assert_fields(
        router_pb2.RouterResponse,
        {
            "selected_generators": (1, "string", True),
            "selection_weights": (2, "double", True),
            "strategy": (3, "string", False),
            "expected_rewards": (4, "double", True),
            "allocations": (
                5,
                "moleculeforge.v1.generator.GeneratorAllocation",
                True,
            ),
            "warnings": (6, "string", True),
            "state_version": (7, "uint64", False),
        },
    )
    _assert_fields(
        router_pb2.RouterFeedbackRequest,
        {
            "feedback_id": (1, "string", False),
            "run_id": (2, "string", False),
            "request_id": (3, "string", False),
            "iteration": (4, "uint32", False),
            "phase": (
                5,
                "moleculeforge.v1.generator.RouterFeedbackPhase",
                False,
            ),
            "generator_name": (6, "string", False),
            "candidate_ids": (7, "string", True),
            "canonical_smiles": (8, "string", False),
            "evidence_ids": (9, "string", True),
            "teacher_score": (10, "double", False),
            "teacher_source": (11, "string", False),
            "teacher_version": (12, "string", False),
            "synthetic": (13, "bool", False),
        },
    )
    _assert_fields(
        router_pb2.RouterFeedbackResponse,
        {
            "acknowledged": (1, "bool", False),
            "duplicate": (2, "bool", False),
            "state_version": (3, "uint64", False),
        },
    )
    _assert_fields(
        router_pb2.RouterWeightsResponse,
        {
            "generator_names": (1, "string", True),
            "weights": (2, "double", True),
            "state_version": (3, "uint64", False),
        },
    )
    absent_score = router_pb2.RouterFeedbackRequest()
    literal_zero_score = router_pb2.RouterFeedbackRequest(teacher_score=0.0)
    assert not absent_score.HasField("teacher_score")
    assert literal_zero_score.HasField("teacher_score")
    assert _protobuf_round_trip(literal_zero_score).HasField("teacher_score")


def test_router_service_preserves_old_rpcs_and_adds_feedback_and_weights() -> None:
    _assert_rpc_contract(
        router_pb2.DESCRIPTOR,
        "GeneratorRouterService",
        {
            "Route": (
                "moleculeforge.v1.generator.RouterRequest",
                "moleculeforge.v1.generator.RouterResponse",
                False,
                False,
            ),
            "RunProxylessSearch": (
                "moleculeforge.v1.generator.RouterProxylessSearchRequest",
                "moleculeforge.v1.generator.RouterProxylessSearchResponse",
                False,
                False,
            ),
            "SubmitFeedback": (
                "moleculeforge.v1.generator.RouterFeedbackRequest",
                "moleculeforge.v1.generator.RouterFeedbackResponse",
                False,
                False,
            ),
            "GetWeights": (
                "moleculeforge.v1.generator.RouterRequest",
                "moleculeforge.v1.generator.RouterWeightsResponse",
                False,
                False,
            ),
        },
    )


def test_oracle_descriptors_have_exact_appended_contract() -> None:
    assert {value.name: value.number for value in oracle_pb2.OracleOutcome.DESCRIPTOR.values} == {
        "ORACLE_OUTCOME_UNSPECIFIED": 0,
        "ORACLE_OUTCOME_PASS": 1,
        "ORACLE_OUTCOME_FAIL": 2,
        "ORACLE_OUTCOME_SKIPPED": 3,
        "ORACLE_OUTCOME_ERROR": 4,
    }
    _assert_fields(
        oracle_pb2.OracleMetric,
        {
            "property": (1, "string", False),
            "value": (2, "double", False),
            "unit": (3, "string", False),
            "uncertainty": (4, "double", False),
        },
    )
    _assert_fields(
        oracle_pb2.OracleEvaluation,
        {
            "oracle_name": (1, "string", False),
            "molecule_smiles": (2, "string", False),
            "level": (3, "moleculeforge.v1.oracle.OracleLevel", False),
            "scores": (
                4,
                "moleculeforge.v1.oracle.OracleEvaluation.ScoresEntry",
                True,
            ),
            "uncertainties": (
                5,
                "moleculeforge.v1.oracle.OracleEvaluation.UncertaintiesEntry",
                True,
            ),
            "elapsed_ms": (6, "int64", False),
            "success": (7, "bool", False),
            "error_message": (8, "string", False),
            "outcome": (9, "moleculeforge.v1.oracle.OracleOutcome", False),
            "oracle_version": (10, "string", False),
            "model_version": (11, "string", False),
            "artifact_refs": (12, "moleculeforge.v1.core.ArtifactRef", True),
            "evidence_id": (13, "string", False),
            "metrics": (14, "moleculeforge.v1.oracle.OracleMetric", True),
            "error_code": (15, "string", False),
        },
    )
    _assert_fields(
        oracle_pb2.OracleBatchRequest,
        {
            "project_id": (1, "string", False),
            "molecule_smiles": (2, "string", True),
            "level": (3, "moleculeforge.v1.oracle.OracleLevel", False),
            "requested_properties": (4, "string", True),
            "return_uncertainty": (5, "bool", False),
            "receptor_uri": (6, "string", False),
            "protein_pdb_id": (7, "string", False),
            "reference_ligand_smiles": (8, "string", False),
            "oracle_parameters": (
                9,
                "moleculeforge.v1.oracle.OracleBatchRequest.OracleParametersEntry",
                True,
            ),
            "request_id": (10, "string", False),
        },
    )
    oracle_parameters = oracle_pb2.OracleBatchRequest.DESCRIPTOR.fields_by_name["oracle_parameters"]
    assert oracle_parameters.message_type.GetOptions().map_entry


def test_oracle_optional_uncertainty_distinguishes_absent_and_zero() -> None:
    absent = oracle_pb2.OracleMetric(property="qed", value=0.8)
    literal_zero = oracle_pb2.OracleMetric(property="qed", value=0.8, uncertainty=0.0)

    assert not absent.HasField("uncertainty")
    assert literal_zero.HasField("uncertainty")
    assert _protobuf_round_trip(literal_zero).HasField("uncertainty")


def test_humu_response_metadata_and_129_float32_values_round_trip() -> None:
    _assert_fields(
        encoder_pb2.EncodeResponse,
        {
            "humu_embedding": (1, "bytes", False),
            "curvature": (2, "double", False),
            "elapsed_ms": (3, "int64", False),
            "checkpoint_version": (4, "string", False),
            "checkpoint_checksum": (5, "string", False),
            "embedding_dimension": (6, "uint32", False),
        },
    )
    embedding = struct.pack("<129f", 1.0, *([0.0] * 128))
    response = encoder_pb2.EncodeResponse(
        humu_embedding=embedding,
        curvature=1.0,
        elapsed_ms=7,
        checkpoint_version="humu-v1",
        checkpoint_checksum="sha256:def",
        embedding_dimension=129,
    )

    restored = _protobuf_round_trip(response)

    assert restored.checkpoint_version == "humu-v1"
    assert restored.checkpoint_checksum == "sha256:def"
    assert restored.embedding_dimension == 129
    assert _consume_humu_response(restored) == pytest.approx((1.0, *([0.0] * 128)))


@pytest.mark.parametrize(
    ("embedding", "dimension"),
    [
        (struct.pack("<128f", *([0.0] * 128)), 129),
        (struct.pack("<130f", *([0.0] * 130)), 129),
        (struct.pack("<129f", math.nan, *([0.0] * 128)), 129),
        (struct.pack("<129f", math.inf, *([0.0] * 128)), 129),
        (struct.pack("<129f", 1.0, *([0.0] * 128)), 128),
    ],
)
def test_humu_consuming_boundary_rejects_invalid_payloads(
    embedding: bytes,
    dimension: int,
) -> None:
    response = SimpleNamespace(
        humu_embedding=embedding,
        embedding_dimension=dimension,
    )

    with pytest.raises(ValueError):
        _consume_humu_response(response)


def test_supply_descriptors_have_exact_contract_and_optional_presence() -> None:
    supply_pb2, _ = _supply_modules()

    _assert_fields(
        supply_pb2.AvailabilityRequest,
        {
            "smiles": (1, "string", False),
            "request_id": (2, "string", False),
        },
    )
    _assert_fields(
        supply_pb2.AvailabilityResponse,
        {
            "smiles": (1, "string", False),
            "available": (2, "bool", False),
            "catalog_id": (3, "string", False),
            "catalog_source": (4, "string", False),
            "source_timestamp": (5, "string", False),
            "price": (6, "double", False),
            "currency": (7, "string", False),
            "lead_time_days": (8, "uint32", False),
            "evidence_id": (9, "string", False),
            "catalog_version": (10, "string", False),
            "catalog_checksum": (11, "string", False),
        },
    )
    _assert_fields(
        supply_pb2.BatchAvailabilityRequest,
        {
            "requests": (1, "moleculeforge.v1.oracle.AvailabilityRequest", True),
            "request_id": (2, "string", False),
        },
    )
    _assert_fields(
        supply_pb2.BatchAvailabilityResponse,
        {
            "results": (1, "moleculeforge.v1.oracle.AvailabilityResponse", True),
            "total_elapsed_ms": (2, "int64", False),
            "request_id": (3, "string", False),
        },
    )
    _assert_fields(
        supply_pb2.CatalogPriceRequest,
        {
            "smiles": (1, "string", False),
            "catalog_id": (2, "string", False),
            "request_id": (3, "string", False),
        },
    )
    _assert_rpc_contract(
        supply_pb2.DESCRIPTOR,
        "SupplyOracleService",
        {
            "CheckAvailability": (
                "moleculeforge.v1.oracle.AvailabilityRequest",
                "moleculeforge.v1.oracle.AvailabilityResponse",
                False,
                False,
            ),
            "BatchCheck": (
                "moleculeforge.v1.oracle.BatchAvailabilityRequest",
                "moleculeforge.v1.oracle.BatchAvailabilityResponse",
                False,
                False,
            ),
            "GetCatalogPrice": (
                "moleculeforge.v1.oracle.CatalogPriceRequest",
                "moleculeforge.v1.oracle.AvailabilityResponse",
                False,
                False,
            ),
        },
    )

    absent = supply_pb2.AvailabilityResponse()
    literal_zero = supply_pb2.AvailabilityResponse(price=0.0, lead_time_days=0)
    restored = _protobuf_round_trip(literal_zero)

    assert not absent.HasField("price")
    assert not absent.HasField("lead_time_days")
    assert restored.HasField("price")
    assert restored.HasField("lead_time_days")
    assert restored.price == 0.0
    assert restored.lead_time_days == 0


@pytest.mark.asyncio
async def test_supply_generated_stub_round_trips_all_rpc_messages() -> None:
    supply_pb2, supply_pb2_grpc = _supply_modules()

    class SupplyServicer(supply_pb2_grpc.SupplyOracleServiceServicer):
        async def CheckAvailability(self, request, context):
            return supply_pb2.AvailabilityResponse(
                smiles=request.smiles,
                available=True,
                catalog_id="catalog-1",
                catalog_source="local-catalog",
                source_timestamp="2026-07-27T00:00:00Z",
                price=0.0,
                currency="USD",
                lead_time_days=0,
                evidence_id=f"evidence:{request.request_id}",
                catalog_version="catalog-v1",
                catalog_checksum="sha256:catalog",
            )

        async def BatchCheck(self, request, context):
            return supply_pb2.BatchAvailabilityResponse(
                results=[await self.CheckAvailability(item, context) for item in request.requests],
                total_elapsed_ms=1,
                request_id=request.request_id,
            )

        async def GetCatalogPrice(self, request, context):
            return supply_pb2.AvailabilityResponse(
                smiles=request.smiles,
                available=True,
                catalog_id=request.catalog_id,
                catalog_source="local-catalog",
                price=3.5,
                currency="USD",
                evidence_id=f"evidence:{request.request_id}",
                catalog_version="catalog-v1",
                catalog_checksum="sha256:catalog",
            )

    async with _running_server(
        lambda server: supply_pb2_grpc.add_SupplyOracleServiceServicer_to_server(
            SupplyServicer(),
            server,
        )
    ) as channel:
        stub = supply_pb2_grpc.SupplyOracleServiceStub(channel)
        availability = await stub.CheckAvailability(
            supply_pb2.AvailabilityRequest(smiles="CCO", request_id="supply-1")
        )
        batch = await stub.BatchCheck(
            supply_pb2.BatchAvailabilityRequest(
                requests=[
                    supply_pb2.AvailabilityRequest(
                        smiles="CCO",
                        request_id="supply-2-item",
                    )
                ],
                request_id="supply-2",
            )
        )
        price = await stub.GetCatalogPrice(
            supply_pb2.CatalogPriceRequest(
                smiles="CCO",
                catalog_id="catalog-2",
                request_id="supply-3",
            )
        )

    assert isinstance(availability, supply_pb2.AvailabilityResponse)
    assert isinstance(availability, Message)
    assert availability.evidence_id == "evidence:supply-1"
    assert availability.HasField("price")
    assert availability.HasField("lead_time_days")
    assert batch.request_id == "supply-2"
    assert batch.results[0].catalog_checksum == "sha256:catalog"
    assert price.catalog_id == "catalog-2"
    assert price.price == pytest.approx(3.5)


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"rows": 0}, "rows"),
        ({"dim": 0}, "dim"),
        ({"teacher_embeddings": b"short"}, "byte length"),
        ({"training_batch_json": b"\xff"}, "UTF-8"),
        ({"training_batch_json": "not-json"}, "valid JSON"),
        ({"training_batch_json": "[]"}, "JSON object"),
        ({"training_batch_json": "{}"}, "schema_version"),
        ({"training_batch_json": '{"schema_version": ""}'}, "schema_version"),
        ({"training_batch_json": '{"schema_version": 1}'}, "schema_version"),
    ],
)
def test_incremental_update_boundary_rejects_invalid_requests(
    overrides: dict,
    error: str,
) -> None:
    values = {
        "rows": 1,
        "dim": 1,
        "teacher_embeddings": struct.pack("<f", 0.5),
        "training_batch_json": '{"schema_version": "training-batch.v1"}',
    }
    values.update(overrides)

    with pytest.raises(ValueError, match=error):
        _validate_model_update_request(SimpleNamespace(**values))


@pytest.mark.asyncio
async def test_incremental_generator_generated_stub_preserves_update_payload() -> None:
    training_batch_json = json.dumps(
        {
            "schema_version": "training-batch.v1",
            "samples": [{"smiles": "CCO", "reward": 0.75}],
        },
        sort_keys=True,
    )
    teacher_embeddings = struct.pack("<4f", 0.1, 0.2, 0.3, 0.4)

    class IncrementalServicer(generator_pb2_grpc.IncrementalGeneratorServiceServicer):
        received = None

        async def UpdateModel(self, request, context):
            try:
                _validate_model_update_request(request)
            except ValueError as exc:
                await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
            self.received = generator_pb2.ModelUpdateRequest.FromString(request.SerializeToString())
            return generator_pb2.ModelUpdateResponse(
                acknowledged=True,
                active_version=request.target_checkpoint_version,
                artifacts=[
                    audit_pb2.ArtifactRef(
                        name="checkpoint",
                        version=request.target_checkpoint_version,
                        checksum="sha256:checkpoint",
                        required=True,
                    ),
                    audit_pb2.ArtifactRef(
                        name="teacher",
                        version=request.teacher_version,
                        checksum="sha256:teacher",
                        required=True,
                    ),
                ],
                updated_samples=request.rows,
            )

    servicer = IncrementalServicer()
    async with _running_server(
        lambda server: generator_pb2_grpc.add_IncrementalGeneratorServiceServicer_to_server(
            servicer,
            server,
        )
    ) as channel:
        stub = generator_pb2_grpc.IncrementalGeneratorServiceStub(channel)
        response = await stub.UpdateModel(
            generator_pb2.ModelUpdateRequest(
                run_id="run-1",
                request_id="update-1",
                training_batch_json=training_batch_json,
                teacher_embeddings=teacher_embeddings,
                rows=2,
                dim=2,
                teacher_source="humu-teacher",
                teacher_version="teacher-v1",
                target_checkpoint_version="iclm-v2",
            )
        )

    assert isinstance(response, generator_pb2.ModelUpdateResponse)
    assert isinstance(response, Message)
    assert response.acknowledged
    assert response.active_version == "iclm-v2"
    assert response.updated_samples == 2
    assert [(artifact.name, artifact.version) for artifact in response.artifacts] == [
        ("checkpoint", "iclm-v2"),
        ("teacher", "teacher-v1"),
    ]
    assert servicer.received is not None
    assert servicer.received.training_batch_json == training_batch_json
    assert json.loads(servicer.received.training_batch_json)["schema_version"] == (
        "training-batch.v1"
    )
    assert servicer.received.teacher_embeddings == teacher_embeddings
