from __future__ import annotations

import json
import math
from copy import deepcopy
from typing import Any

import grpc
import pytest
from fastapi import HTTPException
from orchestrator.agent import (
    OrchestratorAgent,
    _FullAgentWorkflowClients,
    _initial_validation_state,
)
from orchestrator.pipeline import ReasoningPipeline
from orchestrator.workflow import graph_builder
from orchestrator.workflow.graph_builder import (
    WorkflowGraph,
    create_initial_state,
    ensure_candidate_identities,
)
from orchestrator_svc import main as orchestrator_main
from validation_agent.agent import ValidationAgent


def test_workflow_graph_has_one_full_scope() -> None:
    state = create_initial_state(
        "Design a candidate",
        run_id="run-full",
        trace_id="trace-full",
    )

    assert state["workflow_scope"] == "full"
    assert state["validation_passed"] is False
    with pytest.raises(TypeError):
        WorkflowGraph(workflow_scope="engineering")


def _validation_policy(*, oracle_level: int = 3) -> dict[str, Any]:
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
            "max_uncertainty": 1.0,
        },
        {
            "level": 4,
            "oracle": "external",
            "metric": "experimental_activity",
            "direction": "maximize",
            "value": 0.5,
        },
    ]
    return {
        "oracle_level": oracle_level,
        "batch_size": 8,
        "max_concurrency": 2,
        "thresholds": [threshold for threshold in thresholds if threshold["level"] <= oracle_level],
        "oracle_inputs": {
            "dock": {
                "receptor_uri": "file:///models/receptor.pdbqt",
                "oracle_parameters": {"engine": "gnina"},
            },
            "fep": {
                "protein_pdb_id": "1ABC",
                "reference_ligand_smiles": "CCN",
                "oracle_parameters": {
                    "method": "relative",
                    "n_repeats": 3,
                },
            },
        },
    }


def _teacher_policy() -> dict[str, Any]:
    return {
        "teacher_source": "hypseek",
        "teacher_version": "2026-07-29",
        "allow_synthetic": False,
        "kd_weight": 0.25,
    }


def test_full_policy_preserves_explicit_positive_kd_weight() -> None:
    request = _full_request()
    request["teacher_policy"]["kd_weight"] = 0.25

    policies = graph_builder.validate_full_workflow_policies(request)

    assert policies["teacher_policy"] == {
        "teacher_source": "hypseek",
        "teacher_version": "2026-07-29",
        "allow_synthetic": False,
        "kd_weight": 0.25,
    }


def test_full_policy_requires_kd_weight_before_dynamic_iclm_routing() -> None:
    request = _full_request()
    request["teacher_policy"].pop("kd_weight")

    with pytest.raises(ValueError, match="kd_weight"):
        graph_builder.validate_full_workflow_policies(request)


@pytest.mark.parametrize("kd_weight", [True, 0.0, -0.5, float("inf"), float("nan")])
def test_full_policy_rejects_invalid_explicit_kd_weight(kd_weight: object) -> None:
    request = _full_request()
    request["teacher_policy"]["kd_weight"] = kd_weight

    with pytest.raises(ValueError, match="kd_weight"):
        graph_builder.validate_full_workflow_policies(request)


def _selection_policy() -> dict[str, Any]:
    return {
        "criteria": [
            {"metric": "admet_score", "direction": "maximize"},
            {"metric": "rbfe", "direction": "minimize"},
        ]
    }


def _full_request() -> dict[str, Any]:
    return {
        "project_id": "project-full",
        "nl_input": "Design a validated molecule",
        "workflow_scope": "full",
        "validation_passed": True,
        "max_refinements": 1,
        "validation_policy": _validation_policy(),
        "teacher_policy": _teacher_policy(),
        "selection_policy": _selection_policy(),
    }


@pytest.mark.parametrize(
    "missing_policy",
    ["validation_policy", "teacher_policy", "selection_policy"],
)
async def test_full_policy_is_rejected_before_runtime_or_run_creation(
    monkeypatch: pytest.MonkeyPatch,
    missing_policy: str,
) -> None:
    request = _full_request()
    request.pop(missing_policy)
    runtime_called = False

    async def forbidden_runtime():
        nonlocal runtime_called
        runtime_called = True
        raise AssertionError("runtime must not be opened for an invalid policy")

    monkeypatch.setattr(orchestrator_main, "_runtime", forbidden_runtime)

    with pytest.raises(HTTPException) as exc_info:
        await orchestrator_main.create_design_run(request)

    assert exc_info.value.status_code == 422
    assert missing_policy in str(exc_info.value.detail)
    assert runtime_called is False


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("project_id", None, "project_id"),
        ("project_id", "   ", "project_id"),
        ("external_evidence", {}, "external_evidence"),
        ("external_evidence", [1], "evidence resume endpoint"),
    ],
)
async def test_rest_full_context_is_rejected_before_runtime_or_run_creation(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
    message: str,
) -> None:
    request = _full_request()
    if value is None:
        request.pop(field)
    else:
        request[field] = value
    runtime_called = False

    async def forbidden_runtime():
        nonlocal runtime_called
        runtime_called = True
        raise AssertionError("runtime must not be opened for invalid full-workflow context")

    monkeypatch.setattr(orchestrator_main, "_runtime", forbidden_runtime)

    with pytest.raises(HTTPException) as exc_info:
        await orchestrator_main.create_design_run(request)

    assert exc_info.value.status_code == 422
    assert message in str(exc_info.value.detail)
    assert runtime_called is False


@pytest.mark.parametrize(
    ("project_id", "external_evidence", "message"),
    [
        ("", None, "project_id"),
        ("   ", None, "project_id"),
        ("project-full", [1], "evidence resume endpoint"),
    ],
)
async def test_grpc_full_context_is_rejected_before_runtime(
    monkeypatch: pytest.MonkeyPatch,
    project_id: str,
    external_evidence: object,
    message: str,
) -> None:
    from mf_core.proto_gen.moleculeforge.v1.agent import orchestrator_pb2

    full_request = _full_request()
    runtime_called = False

    async def forbidden_runtime():
        nonlocal runtime_called
        runtime_called = True
        raise AssertionError("runtime must not be opened for invalid full-workflow context")

    monkeypatch.setattr(orchestrator_main, "_runtime", forbidden_runtime)
    request = orchestrator_pb2.StartPipelineRequest(
        project_id=project_id,
        nl_input=str(full_request["nl_input"]),
        workflow_scope="full",
        max_refinements=1,
        validation_policy_json=json.dumps(full_request["validation_policy"]),
        teacher_policy_json=json.dumps(full_request["teacher_policy"]),
        selection_policy_json=json.dumps(full_request["selection_policy"]),
        **(
            {"external_evidence_json": json.dumps(external_evidence)}
            if external_evidence is not None
            else {}
        ),
    )

    with pytest.raises(HTTPException) as exc_info:
        await orchestrator_main.OrchestratorServicer().StartPipeline(request, None)

    assert exc_info.value.status_code == 422
    assert message in str(exc_info.value.detail)
    assert runtime_called is False


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("project_id", "   ", "project_id"),
        ("external_evidence", {}, "external_evidence"),
        ("external_evidence", ["invalid"], "external_evidence"),
    ],
)
async def test_direct_full_context_is_rejected_before_agent_mesh_calls(
    field: str,
    value: object,
    message: str,
) -> None:
    request = {
        **_full_request(),
        "run_id": "run-context",
        "trace_id": "trace-context",
        field: value,
    }
    agent = OrchestratorAgent(message_bus=None, crg_repository=object())

    with pytest.raises(ValueError, match=message):
        await agent.run_design_workflow(request)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("validation_policy_json", "[]", "JSON object"),
        ("teacher_policy_json", "[]", "JSON object"),
        ("selection_policy_json", "[]", "JSON object"),
        ("external_evidence_json", "{}", "JSON list"),
        ("teacher_policy_json", "", "valid JSON"),
        ("selection_policy_json", "NaN", "valid JSON"),
    ],
)
async def test_grpc_json_policy_fields_are_strictly_typed_before_start(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: str,
    message: str,
) -> None:
    from mf_core.proto_gen.moleculeforge.v1.agent import orchestrator_pb2

    async def forbidden_start(_request: dict[str, Any]) -> dict[str, Any]:
        raise AssertionError("invalid gRPC JSON must not start a run")

    monkeypatch.setattr(orchestrator_main, "start_design", forbidden_start)
    request = orchestrator_pb2.StartPipelineRequest(
        nl_input="Design a validated molecule",
        workflow_scope="full",
        max_refinements=1,
        **{field: value},
    )

    with pytest.raises(HTTPException) as exc_info:
        await orchestrator_main.OrchestratorServicer().StartPipeline(request, None)

    assert exc_info.value.status_code == 422
    assert message in str(exc_info.value.detail)


async def test_real_grpc_maps_http_policy_errors_to_invalid_argument() -> None:
    from mf_core.proto_gen.moleculeforge.v1.agent import (
        orchestrator_pb2,
        orchestrator_pb2_grpc,
    )

    server = grpc.aio.server()
    orchestrator_pb2_grpc.add_OrchestratorServiceServicer_to_server(
        orchestrator_main.OrchestratorGrpcServicer(),
        server,
    )
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    channel = grpc.aio.insecure_channel(f"127.0.0.1:{port}")
    await channel.channel_ready()
    stub = orchestrator_pb2_grpc.OrchestratorServiceStub(channel)
    try:
        requests = [
            orchestrator_pb2.StartPipelineRequest(
                nl_input="Design an engineering molecule",
                workflow_scope="engineering",
            ),
            orchestrator_pb2.StartPipelineRequest(
                nl_input="Design a validated molecule",
                workflow_scope="full",
                max_refinements=1,
            ),
        ]
        for request in requests:
            with pytest.raises(grpc.aio.AioRpcError) as exc_info:
                await stub.StartPipeline(request)
            assert exc_info.value.code() == grpc.StatusCode.INVALID_ARGUMENT
    finally:
        await channel.close()
        await server.stop(None)


class _FailingGrpcService:
    def __init__(self, error: Exception) -> None:
        self.error = error

    async def StartPipeline(self, request, context):
        raise self.error

    async def GetPipelineState(self, request, context):
        raise self.error


def _duplicate_run_http_error() -> HTTPException:
    cause = orchestrator_main.RunAlreadyExistsError("run run-existing already exists")
    error = HTTPException(status_code=409, detail=str(cause))
    error.__cause__ = cause
    return error


@pytest.mark.parametrize(
    ("rpc_name", "error", "expected_status"),
    [
        (
            "StartPipeline",
            HTTPException(status_code=400, detail="bad request"),
            grpc.StatusCode.INVALID_ARGUMENT,
        ),
        (
            "StartPipeline",
            HTTPException(status_code=422, detail="invalid policy"),
            grpc.StatusCode.INVALID_ARGUMENT,
        ),
        (
            "GetPipelineState",
            HTTPException(status_code=404, detail="missing run"),
            grpc.StatusCode.NOT_FOUND,
        ),
        (
            "StartPipeline",
            _duplicate_run_http_error(),
            grpc.StatusCode.ALREADY_EXISTS,
        ),
        (
            "StartPipeline",
            HTTPException(status_code=409, detail="run is not resumable"),
            grpc.StatusCode.FAILED_PRECONDITION,
        ),
    ],
)
async def test_real_grpc_maps_expected_http_errors(
    rpc_name: str,
    error: Exception,
    expected_status: grpc.StatusCode,
) -> None:
    from mf_core.proto_gen.moleculeforge.v1.agent import (
        orchestrator_pb2,
        orchestrator_pb2_grpc,
    )

    server = grpc.aio.server()
    orchestrator_pb2_grpc.add_OrchestratorServiceServicer_to_server(
        orchestrator_main.OrchestratorGrpcServicer(_FailingGrpcService(error)),
        server,
    )
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    channel = grpc.aio.insecure_channel(f"127.0.0.1:{port}")
    await channel.channel_ready()
    stub = orchestrator_pb2_grpc.OrchestratorServiceStub(channel)
    try:
        request = (
            orchestrator_pb2.StartPipelineRequest()
            if rpc_name == "StartPipeline"
            else orchestrator_pb2.PipelineStateRequest(design_id="run-missing")
        )
        with pytest.raises(grpc.aio.AioRpcError) as exc_info:
            await getattr(stub, rpc_name)(request)
        assert exc_info.value.code() == expected_status
    finally:
        await channel.close()
        await server.stop(None)


async def test_get_pipeline_state_serializes_canonical_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_state = {
        "z": [True, None],
        "a": {"score": 0.75},
    }

    async def fake_status(_design_id: str) -> dict[str, Any]:
        return {
            "current_stage": "validating",
            "state": expected_state,
        }

    monkeypatch.setattr(orchestrator_main, "get_design_status", fake_status)

    response = await orchestrator_main.OrchestratorServicer().GetPipelineState(
        type("Request", (), {"design_id": "run-state-json"})(),
        None,
    )

    assert json.loads(response.state_json) == expected_state
    assert response.state_json == '{"a":{"score":0.75},"z":[true,null]}'


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda request: request["validation_policy"]["thresholds"].pop(0),
            "rdkit",
        ),
        (
            lambda request: request["validation_policy"].update({"oracle_level": 5}),
            "oracle_level",
        ),
        (
            lambda request: request["validation_policy"].update({"batch_size": True}),
            "batch_size",
        ),
        (
            lambda request: request["validation_policy"]["thresholds"][0].update(
                {"value": math.inf}
            ),
            "value",
        ),
        (
            lambda request: request["validation_policy"]["thresholds"][0].update(
                {"direction": "MAXIMIZE"}
            ),
            "maximize or minimize",
        ),
        (
            lambda request: request["validation_policy"]["thresholds"].append(
                {
                    **request["validation_policy"]["thresholds"][0],
                    "value": 0.6,
                }
            ),
            "unique",
        ),
        (
            lambda request: request["validation_policy"]["thresholds"][2].update(
                {"metric": "affinity"}
            ),
            "unsupported for dock",
        ),
        (
            lambda request: request["validation_policy"]["thresholds"][3].update(
                {"metric": "docking_score"}
            ),
            "unsupported for fep",
        ),
        (
            lambda request: (
                request["validation_policy"]["thresholds"].append(
                    {
                        "level": 1,
                        "oracle": "boltz2",
                        "metric": "docking_score",
                        "direction": "minimize",
                        "value": -6.0,
                    }
                ),
                request["validation_policy"]["oracle_inputs"].update(
                    {"boltz2": {"protein_pdb_id": "1ABC"}}
                ),
            ),
            "unsupported for boltz2",
        ),
        (
            lambda request: request["validation_policy"]["oracle_inputs"]["dock"].pop(
                "receptor_uri"
            ),
            "receptor_uri",
        ),
        (
            lambda request: request["validation_policy"]["oracle_inputs"]["dock"][
                "oracle_parameters"
            ].pop("engine"),
            "engine",
        ),
        (
            lambda request: request["validation_policy"]["oracle_inputs"]["dock"][
                "oracle_parameters"
            ].update({"engine": "vina"}),
            "gnina",
        ),
        (
            lambda request: request["validation_policy"]["oracle_inputs"]["fep"].pop(
                "reference_ligand_smiles"
            ),
            "reference_ligand_smiles",
        ),
        (
            lambda request: request["validation_policy"]["oracle_inputs"]["fep"][
                "oracle_parameters"
            ].pop("method"),
            "method",
        ),
        (
            lambda request: request["validation_policy"]["oracle_inputs"]["fep"][
                "oracle_parameters"
            ].update({"n_repeats": 0}),
            "n_repeats",
        ),
        (
            lambda request: request["teacher_policy"].update({"allow_synthetic": 1}),
            "allow_synthetic",
        ),
        (
            lambda request: request["teacher_policy"].update({"teacher_source": "teacher-model"}),
            "hypseek",
        ),
        (
            lambda request: request["validation_policy"]["oracle_inputs"]["dock"].update(
                {"receptor_url": "file:///models/receptor.pdbqt"}
            ),
            "receptor_url",
        ),
        (
            lambda request: request["validation_policy"]["oracle_inputs"]["dock"][
                "oracle_parameters"
            ].update({"engin": "gnina"}),
            "engin",
        ),
        (
            lambda request: request["validation_policy"]["oracle_inputs"]["fep"].update(
                {"protein_id": "1ABC"}
            ),
            "protein_id",
        ),
        (
            lambda request: request["validation_policy"]["oracle_inputs"]["fep"][
                "oracle_parameters"
            ].update({"repeats": 3}),
            "repeats",
        ),
        (
            lambda request: request["selection_policy"]["criteria"].append(
                {"metric": "admet_score", "direction": "minimize"}
            ),
            "unique",
        ),
        (
            lambda request: request["selection_policy"]["criteria"][0].update(
                {"metric": "clearance"}
            ),
            "exactly one",
        ),
        (
            lambda request: request["validation_policy"]["thresholds"][0].update(
                {"metric": "admet_score"}
            ),
            "exactly one",
        ),
        (
            lambda request: request["selection_policy"]["criteria"][0].update(
                {"direction": "minimize"}
            ),
            "direction",
        ),
        (
            lambda request: request["selection_policy"]["criteria"][0].update(
                {"direction": "MAXIMIZE"}
            ),
            "maximize or minimize",
        ),
    ],
)
def test_full_policy_schema_is_strict_and_level_complete(
    mutator,
    message: str,
) -> None:
    request = _full_request()
    mutator(request)

    with pytest.raises(HTTPException) as exc_info:
        orchestrator_main._validated_policy(request)

    assert exc_info.value.status_code == 422
    assert message in str(exc_info.value.detail)


def test_full_policy_is_the_only_scope_and_default() -> None:
    request = _full_request()
    request.pop("workflow_scope")

    policy = orchestrator_main._validated_policy(request)

    assert policy["workflow_scope"] == "full"
    assert policy["validation_passed"] is False


@pytest.mark.parametrize("workflow_scope", ["state_only", "engineering"])
def test_non_full_policy_scope_is_rejected(workflow_scope: str) -> None:
    request = _full_request()
    request["workflow_scope"] = workflow_scope

    with pytest.raises(HTTPException) as exc_info:
        orchestrator_main._validated_policy(request)

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "workflow_scope must be full"


def test_full_policy_starts_unvalidated_without_legacy_shortcut() -> None:
    request = _full_request()
    request.pop("validation_passed")

    policy = orchestrator_main._validated_policy(request)

    assert policy["validation_passed"] is False
    assert policy["validation_policy"] == _validation_policy()


def test_full_policy_ignores_legacy_validation_passed_true() -> None:
    request = _full_request()
    request["validation_passed"] = True

    policy = orchestrator_main._validated_policy(request)

    assert policy["validation_passed"] is False


def test_direct_full_workflow_starts_unvalidated() -> None:
    assert _initial_validation_state({"validation_passed": True}, "full") is False


def test_full_graph_initial_state_starts_unvalidated() -> None:
    state = graph_builder.create_initial_state("Design a molecule")

    assert state["validation_passed"] is False


def test_shared_full_policy_validator_rejects_unsupported_teacher_source() -> None:
    request = _full_request()
    request["teacher_policy"]["teacher_source"] = "teacher-model"
    validator = getattr(graph_builder, "validate_full_workflow_policies", None)

    assert callable(validator)
    with pytest.raises(ValueError, match="hypseek"):
        validator(request)


async def test_direct_full_policy_is_validated_before_agent_mesh_calls() -> None:
    request = {
        **_full_request(),
        "project_id": "project-policy",
        "run_id": "run-policy",
        "trace_id": "trace-policy",
    }
    request["teacher_policy"]["teacher_source"] = "teacher-model"
    agent = OrchestratorAgent(message_bus=None, crg_repository=object())

    with pytest.raises(ValueError, match="hypseek"):
        await agent.run_design_workflow(request)


@pytest.mark.parametrize(
    ("oracle_inputs", "message"),
    [
        (
            {
                "protein_pdb_id": "1ABC",
                "oracle_parameters": {"ensemble_size": 0},
            },
            "ensemble_size",
        ),
        (
            {
                "protein_pdb_id": "1ABC",
                "oracle_parameters": {
                    "ensemble_size": 2,
                    "ensembl_size": 2,
                },
            },
            "ensembl_size",
        ),
    ],
)
def test_boltz2_oracle_inputs_are_exact(
    oracle_inputs: dict[str, Any],
    message: str,
) -> None:
    request = _full_request()
    request["validation_policy"]["thresholds"].append(
        {
            "level": 1,
            "oracle": "boltz2",
            "metric": "affinity",
            "direction": "minimize",
            "value": -7.0,
        }
    )
    request["validation_policy"]["oracle_inputs"]["boltz2"] = oracle_inputs

    with pytest.raises(HTTPException) as exc_info:
        orchestrator_main._validated_policy(request)

    assert exc_info.value.status_code == 422
    assert message in str(exc_info.value.detail)


def test_missing_candidate_ids_are_stable_unique_and_preserve_existing_ids() -> None:
    candidates = [
        {
            "canonical_smiles": "CCO",
            "generator_name": "hfm_3d",
        },
        {
            "canonical_smiles": "CCO",
            "generator_name": "hfm_3d",
        },
        {
            "candidate_id": "provided-id",
            "canonical_smiles": "CCC",
            "generator_name": "fragfm",
        },
    ]

    first = ensure_candidate_identities(deepcopy(candidates))
    second = ensure_candidate_identities(deepcopy(candidates))

    assert [candidate["candidate_id"] for candidate in first] == [
        candidate["candidate_id"] for candidate in second
    ]
    assert len({candidate["candidate_id"] for candidate in first}) == 3
    assert first[2]["candidate_id"] == "provided-id"


def test_duplicate_provided_candidate_ids_are_rejected() -> None:
    with pytest.raises(RuntimeError, match="candidate_id values must be unique"):
        ensure_candidate_identities(
            [
                {"candidate_id": "duplicate", "canonical_smiles": "CCO"},
                {"candidate_id": "duplicate", "canonical_smiles": "CCC"},
            ]
        )


@pytest.mark.parametrize(
    "candidate_id",
    [True, 7, "   ", " candidate-with-padding "],
)
def test_provided_candidate_id_must_be_a_nonempty_trimmed_string(
    candidate_id: object,
) -> None:
    with pytest.raises(RuntimeError, match="non-empty trimmed string"):
        ensure_candidate_identities(
            [
                {
                    "candidate_id": candidate_id,
                    "canonical_smiles": "CCO",
                }
            ]
        )


@pytest.mark.parametrize(
    ("field", "candidate_value", "record_value"),
    [
        ("candidate_id", "1", 1),
        ("canonical_smiles", "1", 1),
    ],
)
def test_validation_record_identity_requires_native_trimmed_strings(
    field: str,
    candidate_value: str,
    record_value: object,
) -> None:
    candidate = {
        "candidate_id": "candidate-a",
        "canonical_smiles": "CCO",
        "generator_name": "hfm_3d",
    }
    candidate[field] = candidate_value
    record = _record(
        str(candidate["candidate_id"]),
        str(candidate["canonical_smiles"]),
        outcome="PASS",
        admet_score=0.8,
        rbfe=-8.0,
        evidence_id="evidence-a",
    )
    record[field] = record_value

    with pytest.raises(RuntimeError, match="non-empty trimmed string"):
        graph_builder.validation_feedback_groups(
            [candidate],
            [record],
            teacher_policy=_teacher_policy(),
            validation_policy=_validation_policy(),
        )


def test_validation_feedback_records_bind_passed_to_outcome() -> None:
    candidate = {
        "candidate_id": "candidate-a",
        "canonical_smiles": "CCO",
        "generator_name": "hfm_3d",
    }
    record = _record(
        "candidate-a",
        "CCO",
        outcome="PASS",
        admet_score=0.8,
        rbfe=-8.0,
        evidence_id="evidence-a",
    )

    groups = graph_builder.validation_feedback_groups(
        [candidate],
        [record],
        teacher_policy=_teacher_policy(),
        validation_policy=_validation_policy(),
    )

    assert groups[0]["records"][0]["outcome"] == "PASS"
    assert groups[0]["records"][0]["passed"] is True


@pytest.mark.parametrize("evidence_id", [1, True, "", " evidence-a "])
@pytest.mark.parametrize("source", ["evidence_ids", "evidence"])
def test_validation_record_evidence_ids_require_native_trimmed_strings(
    evidence_id: object,
    source: str,
) -> None:
    record: dict[str, object]
    if source == "evidence_ids":
        record = {"evidence_ids": [evidence_id]}
    else:
        record = {"evidence": [{"evidence_id": evidence_id}]}

    with pytest.raises(RuntimeError, match="non-empty trimmed string"):
        graph_builder.validation_record_evidence_ids(record)


@pytest.mark.parametrize(
    ("field", "candidate_value", "record_value"),
    [
        ("candidate_id", "1", 1),
        ("canonical_smiles", "1", 1),
    ],
)
def test_selection_rejects_non_native_validation_record_identity(
    field: str,
    candidate_value: str,
    record_value: object,
) -> None:
    candidate = {
        "candidate_id": "candidate-a",
        "canonical_smiles": "CCO",
        "generator_name": "hfm_3d",
    }
    candidate[field] = candidate_value
    record = _record(
        str(candidate["candidate_id"]),
        str(candidate["canonical_smiles"]),
        outcome="PASS",
        admet_score=0.8,
        rbfe=-8.0,
        evidence_id="evidence-a",
    )
    record[field] = record_value

    with pytest.raises(RuntimeError, match="non-empty trimmed string"):
        graph_builder.select_full_candidate(
            [candidate],
            {
                "outcome": "PASS",
                "validation_policy": _validation_policy(),
                "records": [record],
            },
            _selection_policy(),
        )


@pytest.mark.parametrize("outcome", ["pass", "Pass", " fail", 1, None])
def test_validation_outcomes_require_exact_uppercase_strings(outcome: object) -> None:
    with pytest.raises(RuntimeError, match="must be PASS"):
        graph_builder.strict_validation_outcome(outcome)


def test_validation_contract_rejects_fail_with_only_passing_metrics_and_no_reason() -> None:
    candidate = {
        "candidate_id": "candidate-a",
        "canonical_smiles": "CCO",
        "generator_name": "hfm_3d",
    }
    record = _record(
        "candidate-a",
        "CCO",
        outcome="PASS",
        admet_score=0.8,
        rbfe=-8.0,
        evidence_id="evidence-a",
    )
    record["outcome"] = "FAIL"
    record["levels"][-1]["outcome"] = "FAIL"
    record["levels"][-1]["oracles"][0]["outcome"] = "FAIL"

    with pytest.raises(RuntimeError, match="requires a failed metric"):
        graph_builder.require_validation_records_contract(
            [record],
            [candidate],
            _validation_policy(),
        )


class _BatchRequestClient:
    def __init__(
        self,
        records: list[dict[str, Any]],
        *,
        outcome: str = "PASS",
        acknowledge: bool = True,
        response_overrides: dict[str, Any] | None = None,
    ) -> None:
        self.records = records
        self.outcome = outcome
        self.acknowledge = acknowledge
        self.response_overrides = dict(response_overrides or {})
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def request(
        self,
        subject: str,
        payload: dict[str, Any],
        *,
        payload_type_url: str,
        timeout: float,
    ) -> dict[str, Any]:
        self.calls.append((subject, deepcopy(payload)))
        if payload.get("action") == "generator_coord/feedback/v1":
            if self.acknowledge:
                result = {
                    "action": "generator_coord/feedback/v1",
                    "status": "feedback_submitted",
                    "submitted": len(payload["groups"]),
                    "duplicates": 0,
                }
            else:
                result = {
                    "action": "generator_coord/feedback/v1",
                    "status": "feedback_rejected",
                    "submitted": 0,
                    "duplicates": 0,
                }
        else:
            result = {
                "validation_schema_version": "validation.batch.v1",
                "agent": "validation_agent",
                "project_id": payload["project_id"],
                "outcome": self.outcome,
                "validation_policy": deepcopy(payload["validation_policy"]),
                "records": deepcopy(self.records),
            }
        return {
            **result,
            "run_id": payload["run_id"],
            "request_id": payload["request_id"],
            "schema_version": payload["schema_version"],
            **self.response_overrides,
        }


def _record(
    candidate_id: str,
    smiles: str,
    *,
    outcome: str,
    admet_score: float,
    rbfe: float,
    evidence_id: str,
    oracle_level: int = 3,
) -> dict[str, Any]:
    metrics_by_level = {
        0: [
            {
                "level": 0,
                "oracle": "rdkit",
                "metric": "qed",
                "value": 0.8,
                "direction": "maximize",
                "threshold": 0.5,
                "passed": True,
            }
        ],
        1: [
            {
                "level": 1,
                "oracle": "admet",
                "metric": "admet_score",
                "value": admet_score,
                "direction": "maximize",
                "threshold": 0.5,
                "passed": admet_score >= 0.5,
            }
        ],
        2: [
            {
                "level": 2,
                "oracle": "dock",
                "metric": "docking_score",
                "value": -7.0,
                "direction": "minimize",
                "threshold": -6.0,
                "passed": True,
            }
        ],
        3: [
            {
                "level": 3,
                "oracle": "fep",
                "metric": "rbfe",
                "value": rbfe,
                "direction": "minimize",
                "threshold": -7.0,
                "uncertainty": 0.1,
                "max_uncertainty": 1.0,
                "passed": rbfe <= -7.0,
            }
        ],
        4: [
            {
                "level": 4,
                "oracle": "external",
                "metric": "experimental_activity",
                "value": 0.8,
                "direction": "maximize",
                "threshold": 0.5,
                "passed": True,
            }
        ],
    }
    oracle_by_level = {
        0: "rdkit",
        1: "admet",
        2: "dock",
        3: "fep",
        4: "external",
    }
    levels = []
    flat_metrics = []
    for level in range(oracle_level + 1):
        level_outcome = "PASS" if level < oracle_level else outcome
        level_metrics = deepcopy(metrics_by_level[level])
        if level_outcome in {"ERROR", "AWAITING_EVIDENCE"}:
            level_metrics = []
        flat_metrics.extend(deepcopy(level_metrics))
        oracle_record = {
            "oracle": oracle_by_level[level],
            "outcome": level_outcome,
            "metrics": deepcopy(level_metrics),
            "evidence_ids": [],
        }
        if level_outcome == "ERROR":
            oracle_record["error"] = {
                "code": "ORACLE_ERROR",
                "message": "oracle evaluation failed",
            }
        elif level_outcome == "AWAITING_EVIDENCE":
            oracle_record["reason"] = "external evidence is required"
        levels.append(
            {
                "level": level,
                "outcome": level_outcome,
                "oracles": [oracle_record],
            }
        )
    return {
        "schema_version": "validation.record.v1",
        "candidate_id": candidate_id,
        "canonical_smiles": smiles,
        "outcome": outcome,
        "metrics": flat_metrics,
        "evidence": [
            {
                "evidence_id": evidence_id,
                "level": oracle_level,
                "oracle": "validation_agent",
            }
        ],
        "levels": levels,
    }


def _full_state(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "run_id": "run-policy",
        "trace_id": "trace-policy",
        "refinement_count": 0,
        "candidates": candidates,
        "request": {
            "project_id": "project-policy",
            "request_id": "request-policy",
            "agent_request_timeout_seconds": 1.0,
            "retrosyn_engine": "rsgpt",
            "validation_policy": _validation_policy(),
            "teacher_policy": _teacher_policy(),
            "selection_policy": _selection_policy(),
        },
    }


async def test_full_workflow_sends_one_validation_batch_and_acknowledged_feedback() -> None:
    candidates = [
        {
            "candidate_id": "candidate-a",
            "canonical_smiles": "CCO",
            "generator_name": "hfm_3d",
        },
        {
            "candidate_id": "candidate-b",
            "canonical_smiles": "CCO",
            "generator_name": "hfm_3d",
        },
    ]
    records = [
        _record(
            "candidate-a",
            "CCO",
            outcome="PASS",
            admet_score=0.8,
            rbfe=-8.0,
            evidence_id="evidence-a",
        ),
        _record(
            "candidate-b",
            "CCO",
            outcome="FAIL",
            admet_score=0.7,
            rbfe=-6.5,
            evidence_id="evidence-b",
        ),
    ]
    client = _BatchRequestClient(records)

    result = await orchestrator_main.FullWorkflowClients(client).validate_candidates(
        _full_state(candidates)
    )

    assert result["outcome"] == "PASS"
    assert result["passed"] is True
    assert result["results"] == records
    assert len(client.calls) == 2
    validation_payload = client.calls[0][1]
    assert validation_payload["candidates"] == [
        {
            "candidate_id": "candidate-a",
            "canonical_smiles": "CCO",
            "generator_name": "hfm_3d",
        },
        {
            "candidate_id": "candidate-b",
            "canonical_smiles": "CCO",
            "generator_name": "hfm_3d",
        },
    ]
    assert validation_payload["validation_policy"] == _validation_policy()
    assert validation_payload["teacher_policy"] == _teacher_policy()
    assert validation_payload["selection_policy"] == _selection_policy()
    assert validation_payload["external_evidence"] is None
    feedback_payload = client.calls[1][1]
    assert feedback_payload["action"] == "generator_coord/feedback/v1"
    assert feedback_payload["route_request_id"] == "run-policy:generator_coord:0"
    assert feedback_payload["iteration"] == 0
    assert feedback_payload["groups"] == [
        {
            "phase": "validation",
            "generator_name": "hfm_3d",
            "canonical_smiles": "CCO",
            "candidate_ids": ["candidate-a", "candidate-b"],
            "evidence_ids": ["evidence-a", "evidence-b"],
            "records": [
                {**records[0], "passed": True},
                {**records[1], "passed": False},
            ],
            "teacher_policy": _teacher_policy(),
        },
    ]


async def test_full_workflow_defers_feedback_until_external_evidence_is_resolved() -> None:
    candidate = {
        "candidate_id": "candidate-a",
        "canonical_smiles": "CCO",
        "generator_name": "hfm_3d",
    }
    awaiting_record = _record(
        "candidate-a",
        "CCO",
        outcome="AWAITING_EVIDENCE",
        admet_score=0.8,
        rbfe=-8.0,
        evidence_id="evidence-pending",
        oracle_level=4,
    )
    client = _BatchRequestClient(
        [awaiting_record],
        outcome="AWAITING_EVIDENCE",
    )
    state = _full_state([candidate])
    state["request"]["validation_policy"] = _validation_policy(oracle_level=4)

    awaiting_result = await orchestrator_main.FullWorkflowClients(client).validate_candidates(state)

    assert awaiting_result["outcome"] == "AWAITING_EVIDENCE"
    assert "feedback" not in awaiting_result
    assert [subject for subject, _payload in client.calls] == [
        "agent.validation.request",
    ]

    resolved_record = _record(
        "candidate-a",
        "CCO",
        outcome="PASS",
        admet_score=0.8,
        rbfe=-8.0,
        evidence_id="evidence-external",
        oracle_level=4,
    )
    client.records = [resolved_record]
    client.outcome = "PASS"
    state["validation"] = awaiting_result
    state["external_evidence_resume_verified"] = True
    state["request"]["external_evidence"] = [
        {
            "candidate_id": "candidate-a",
            "evidence_ids": ["evidence-external"],
        }
    ]

    resolved_result = await orchestrator_main.FullWorkflowClients(client).validate_candidates(state)

    assert resolved_result["outcome"] == "PASS"
    assert resolved_result["feedback"]["status"] == "feedback_submitted"
    assert [subject for subject, _payload in client.calls] == [
        "agent.validation.request",
        "agent.validation.request",
        "agent.generator_coord.request",
    ]


@pytest.mark.parametrize("consumer", ["service", "agent"])
@pytest.mark.parametrize("outcome", ["ERROR", "AWAITING_EVIDENCE"])
async def test_non_supervisory_validation_outcomes_never_submit_feedback(
    consumer: str,
    outcome: str,
) -> None:
    candidate = {
        "candidate_id": "candidate-a",
        "canonical_smiles": "CCO",
        "generator_name": "hfm_3d",
    }
    record = _record(
        "candidate-a",
        "CCO",
        outcome=outcome,
        admet_score=0.8,
        rbfe=-8.0,
        evidence_id="evidence-a",
        oracle_level=4 if outcome == "AWAITING_EVIDENCE" else 3,
    )
    client = _BatchRequestClient([record], outcome=outcome)
    state = _full_state([candidate])
    if outcome == "AWAITING_EVIDENCE":
        state["request"]["validation_policy"] = _validation_policy(oracle_level=4)
    workflow_clients = (
        orchestrator_main.FullWorkflowClients(client)
        if consumer == "service"
        else _FullAgentWorkflowClients(client)
    )

    result = await workflow_clients.validate_candidates(state)

    assert result["outcome"] == outcome
    assert "feedback" not in result
    assert [subject for subject, _payload in client.calls] == [
        "agent.validation.request",
    ]


@pytest.mark.parametrize("consumer", ["service", "agent"])
async def test_mixed_validation_batch_feedback_contains_only_pass_or_fail_records(
    consumer: str,
) -> None:
    candidates = [
        {
            "candidate_id": "candidate-pass",
            "canonical_smiles": "CCO",
            "generator_name": "hfm_3d",
        },
        {
            "candidate_id": "candidate-awaiting",
            "canonical_smiles": "CCO",
            "generator_name": "hfm_3d",
        },
    ]
    passed = _record(
        "candidate-pass",
        "CCO",
        outcome="PASS",
        admet_score=0.8,
        rbfe=-8.0,
        evidence_id="evidence-pass",
        oracle_level=4,
    )
    awaiting = _record(
        "candidate-awaiting",
        "CCO",
        outcome="AWAITING_EVIDENCE",
        admet_score=0.8,
        rbfe=-8.0,
        evidence_id="evidence-awaiting",
        oracle_level=4,
    )
    client = _BatchRequestClient([passed, awaiting], outcome="PASS")
    state = _full_state(candidates)
    state["request"]["validation_policy"] = _validation_policy(oracle_level=4)
    workflow_clients = (
        orchestrator_main.FullWorkflowClients(client)
        if consumer == "service"
        else _FullAgentWorkflowClients(client)
    )

    result = await workflow_clients.validate_candidates(state)

    assert result["outcome"] == "PASS"
    assert result["feedback"]["status"] == "feedback_submitted"
    feedback_group = client.calls[1][1]["groups"][0]
    assert feedback_group["candidate_ids"] == ["candidate-pass"]
    assert feedback_group["evidence_ids"] == ["evidence-pass"]
    assert feedback_group["records"] == [{**passed, "passed": True}]


@pytest.mark.parametrize("consumer", ["service", "agent"])
@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("validation_schema_version", "validation.batch.v0"),
        ("agent", "other-agent"),
        ("project_id", "other-project"),
        ("run_id", "other-run"),
        ("request_id", "other-request"),
        ("validation_policy", {"oracle_level": 0}),
    ],
)
async def test_validation_batch_response_echo_must_match_request(
    consumer: str,
    field: str,
    invalid_value: object,
) -> None:
    candidate = {
        "candidate_id": "candidate-a",
        "canonical_smiles": "CCO",
        "generator_name": "hfm_3d",
    }
    client = _BatchRequestClient(
        [
            _record(
                "candidate-a",
                "CCO",
                outcome="PASS",
                admet_score=0.8,
                rbfe=-8.0,
                evidence_id="evidence-a",
            )
        ],
        response_overrides={field: invalid_value},
    )
    workflow_clients = (
        orchestrator_main.FullWorkflowClients(client)
        if consumer == "service"
        else _FullAgentWorkflowClients(client)
    )

    with pytest.raises(RuntimeError, match="ValidationAgent batch response"):
        await workflow_clients.validate_candidates(_full_state([candidate]))

    assert len(client.calls) == 1


def _validation_batch_response(
    record: dict[str, Any],
    validation_policy: dict[str, Any],
) -> dict[str, Any]:
    return {
        "validation_schema_version": "validation.batch.v1",
        "agent": "validation_agent",
        "project_id": "project-policy",
        "run_id": "run-policy",
        "request_id": "request-policy:validation:0",
        "outcome": record["outcome"],
        "validation_policy": deepcopy(validation_policy),
        "records": [record],
    }


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda record: record.__setitem__("schema_version", "validation.record.v0"),
            "schema_version",
        ),
        (
            lambda record: record.__setitem__("canonical_smiles", "CCC"),
            "candidate",
        ),
        (
            lambda record: record["levels"].pop(1),
            "continuous",
        ),
        (
            lambda record: record["levels"][0].__setitem__("level", False),
            "continuous",
        ),
        (
            lambda record: record["levels"][2]["oracles"][0].__setitem__("oracle", "fep"),
            "oracle",
        ),
        (
            lambda record: (
                record["metrics"][0].__setitem__("threshold", 0.7),
                record["levels"][0]["oracles"][0]["metrics"][0].__setitem__(
                    "threshold",
                    0.7,
                ),
            ),
            "threshold",
        ),
        (
            lambda record: (
                record["metrics"][1].__setitem__("direction", "minimize"),
                record["levels"][1]["oracles"][0]["metrics"][0].__setitem__(
                    "direction",
                    "minimize",
                ),
            ),
            "direction",
        ),
        (
            lambda record: (
                record["metrics"][0].__setitem__("passed", False),
                record["levels"][0]["oracles"][0]["metrics"][0].__setitem__(
                    "passed",
                    False,
                ),
            ),
            "passed",
        ),
        (
            lambda record: record["levels"][0]["oracles"][0]["metrics"][0].__setitem__(
                "value",
                0.7,
            ),
            "metrics",
        ),
        (
            lambda record: record["evidence"][0].__setitem__("level", 9),
            "evidence",
        ),
    ],
)
def test_validation_batch_contract_rejects_malformed_candidate_records(
    mutate: Any,
    message: str,
) -> None:
    candidate = {
        "candidate_id": "candidate-a",
        "canonical_smiles": "CCO",
        "generator_name": "hfm_3d",
    }
    policy = _validation_policy()
    record = _record(
        "candidate-a",
        "CCO",
        outcome="PASS",
        admet_score=0.8,
        rbfe=-8.0,
        evidence_id="evidence-a",
    )
    mutate(record)

    with pytest.raises(RuntimeError, match=message):
        graph_builder.require_validation_batch_contract(
            _validation_batch_response(record, policy),
            project_id="project-policy",
            run_id="run-policy",
            request_id="request-policy:validation:0",
            validation_policy=policy,
            candidates=[candidate],
        )


@pytest.mark.parametrize("consumer", ["selection", "feedback"])
def test_selection_and_feedback_reject_records_that_bypass_the_batch_contract(
    consumer: str,
) -> None:
    candidate = {
        "candidate_id": "candidate-a",
        "canonical_smiles": "CCO",
        "generator_name": "hfm_3d",
    }
    policy = _validation_policy()
    record = _record(
        "candidate-a",
        "CCO",
        outcome="PASS",
        admet_score=0.8,
        rbfe=-8.0,
        evidence_id="evidence-a",
    )
    record["metrics"][0]["threshold"] = 0.7
    record["levels"][0]["oracles"][0]["metrics"][0]["threshold"] = 0.7

    with pytest.raises(RuntimeError, match="threshold"):
        if consumer == "selection":
            graph_builder.select_full_candidate(
                [candidate],
                {
                    "outcome": "PASS",
                    "validation_policy": policy,
                    "records": [record],
                },
                _selection_policy(),
            )
        else:
            graph_builder.validation_feedback_groups(
                [candidate],
                [record],
                teacher_policy=_teacher_policy(),
                validation_policy=policy,
            )


async def test_direct_full_workflow_feedback_targets_generation_route_request() -> None:
    candidate = {
        "candidate_id": "candidate-a",
        "canonical_smiles": "CCO",
        "generator_name": "hfm_3d",
    }
    client = _BatchRequestClient(
        [
            _record(
                "candidate-a",
                "CCO",
                outcome="PASS",
                admet_score=0.8,
                rbfe=-8.0,
                evidence_id="evidence-a",
            )
        ]
    )

    await _FullAgentWorkflowClients(client).validate_candidates(_full_state([candidate]))

    assert client.calls[1][1]["route_request_id"] == "request-policy:generator_coord:0"


@pytest.mark.parametrize(
    ("feedback", "expected_groups"),
    [
        ({"acknowledged": True}, 1),
        (
            {
                "status": "feedback_submitted",
                "submitted": 1,
                "duplicates": 0,
            },
            1,
        ),
        (
            {
                "action": "other-action",
                "status": "feedback_submitted",
                "submitted": 1,
                "duplicates": 0,
            },
            1,
        ),
        (
            {
                "action": "generator_coord/feedback/v1",
                "status": "feedback_submitted",
                "submitted": -1,
                "duplicates": 2,
            },
            1,
        ),
        (
            {
                "action": "generator_coord/feedback/v1",
                "status": "feedback_submitted",
                "submitted": 2,
                "duplicates": -1,
            },
            1,
        ),
    ],
)
def test_feedback_acknowledgement_requires_exact_non_negative_contract(
    feedback: dict[str, Any],
    expected_groups: int,
) -> None:
    with pytest.raises(RuntimeError, match="acknowledge"):
        graph_builder.require_feedback_acknowledgement(
            feedback,
            expected_groups=expected_groups,
        )


def test_feedback_acknowledgement_accepts_exact_contract() -> None:
    graph_builder.require_feedback_acknowledgement(
        {
            "action": "generator_coord/feedback/v1",
            "status": "feedback_submitted",
            "submitted": 1,
            "duplicates": 1,
        },
        expected_groups=2,
    )


async def test_full_workflow_stops_when_generator_feedback_is_not_acknowledged() -> None:
    candidate = {
        "candidate_id": "candidate-a",
        "canonical_smiles": "CCO",
        "generator_name": "hfm_3d",
    }
    client = _BatchRequestClient(
        [
            _record(
                "candidate-a",
                "CCO",
                outcome="PASS",
                admet_score=0.8,
                rbfe=-8.0,
                evidence_id="evidence-a",
            )
        ],
        acknowledge=False,
    )

    with pytest.raises(RuntimeError, match="acknowledge"):
        await orchestrator_main.FullWorkflowClients(client).validate_candidates(
            _full_state([candidate])
        )


class _StaticOracle:
    def __init__(self, values: dict[str, float]) -> None:
        self.values = values

    async def evaluate(
        self,
        molecules: list[str],
        properties: list[str],
        *,
        request_context: dict[str, Any] | None = None,
    ) -> dict[str, dict[str, float]]:
        return {
            smiles: {metric: self.values[metric] for metric in properties} for smiles in molecules
        }

    async def predict_with_uncertainty(
        self,
        molecules: list[str],
        properties: list[str],
        *,
        request_context: dict[str, Any] | None = None,
    ) -> dict[str, tuple[dict[str, float], dict[str, float]]]:
        return {
            smiles: (
                {metric: self.values[metric] for metric in properties},
                {metric: 0.1 for metric in properties},
            )
            for smiles in molecules
        }


class _ErrorOracle:
    async def evaluate(
        self,
        molecules: list[str],
        properties: list[str],
        *,
        request_context: dict[str, Any] | None = None,
    ) -> dict[str, dict[str, float]]:
        raise RuntimeError("oracle unavailable")


class _CrgRepository:
    async def write_workflow_belief(self, **_kwargs: object) -> None:
        return None


class _ValidationAgentBridgeClient:
    def __init__(self, validation_agent: ValidationAgent) -> None:
        self.validation_agent = validation_agent
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def request(
        self,
        subject: str,
        payload: dict[str, Any],
        *,
        payload_type_url: str,
        timeout: float,
    ) -> dict[str, Any]:
        self.calls.append((subject, deepcopy(payload)))
        if payload.get("action") == "generator_coord/feedback/v1":
            result = {
                "action": "generator_coord/feedback/v1",
                "status": "feedback_submitted",
                "submitted": len(payload["groups"]),
                "duplicates": 0,
            }
        else:
            result = await self.validation_agent.process(payload)
        return {
            **result,
            "run_id": payload["run_id"],
            "request_id": payload["request_id"],
            "schema_version": payload["schema_version"],
        }


@pytest.mark.parametrize(
    "expected_outcome",
    ["AWAITING_EVIDENCE", "ERROR"],
)
async def test_real_validation_agent_non_supervisory_outcomes_preserve_evidence_without_feedback(
    expected_outcome: str,
) -> None:
    if expected_outcome == "AWAITING_EVIDENCE":
        validation_agent = ValidationAgent(
            oracles={
                "rdkit": _StaticOracle({"qed": 0.9}),
                "admet": _StaticOracle({"admet_score": 0.9}),
                "dock": _StaticOracle({"docking_score": -7.0}),
                "fep": _StaticOracle({"rbfe": -8.0}),
            },
            crg_repository=_CrgRepository(),
        )
        validation_policy = _validation_policy(oracle_level=4)
        external_evidence: list[dict[str, Any]] = []
    else:
        validation_agent = ValidationAgent(
            oracles={"rdkit": _ErrorOracle()},
            crg_repository=_CrgRepository(),
        )
        validation_policy = _validation_policy(oracle_level=0)
        external_evidence = []
    candidate = {
        "candidate_id": "candidate-real-validation",
        "canonical_smiles": "CCO",
        "generator_name": "hfm_3d",
    }
    state = _full_state([candidate])
    state["request"]["validation_policy"] = validation_policy
    state["request"]["external_evidence"] = external_evidence
    client = _ValidationAgentBridgeClient(validation_agent)

    result = await orchestrator_main.FullWorkflowClients(client).validate_candidates(state)

    assert result["outcome"] == expected_outcome
    assert result["records"][0]["outcome"] == expected_outcome
    evidence_ids = [item["evidence_id"] for item in result["records"][0]["evidence"]]
    assert evidence_ids
    if expected_outcome == "AWAITING_EVIDENCE":
        assert len(client.calls) == 1
        assert "feedback" not in result
        return
    assert len(client.calls) == 1
    assert "feedback" not in result


async def test_batch_outcome_must_match_record_aggregation_before_feedback() -> None:
    candidate = {
        "candidate_id": "candidate-a",
        "canonical_smiles": "CCO",
        "generator_name": "hfm_3d",
    }
    client = _BatchRequestClient(
        [
            _record(
                "candidate-a",
                "CCO",
                outcome="PASS",
                admet_score=0.8,
                rbfe=-8.0,
                evidence_id="evidence-a",
            )
        ],
        outcome="FAIL",
    )

    with pytest.raises(RuntimeError, match="does not match"):
        await orchestrator_main.FullWorkflowClients(client).validate_candidates(
            _full_state([candidate])
        )

    assert len(client.calls) == 1


@pytest.mark.parametrize(
    ("outcomes", "expected"),
    [
        (["FAIL"], "FAIL"),
        (["AWAITING_EVIDENCE", "FAIL"], "AWAITING_EVIDENCE"),
        (["PASS", "AWAITING_EVIDENCE", "FAIL"], "PASS"),
        (["ERROR", "PASS", "AWAITING_EVIDENCE", "FAIL"], "ERROR"),
    ],
)
def test_validation_record_outcome_priority(
    outcomes: list[str],
    expected: str,
) -> None:
    aggregator = getattr(graph_builder, "validation_records_outcome", None)

    assert callable(aggregator)
    assert aggregator([{"outcome": outcome} for outcome in outcomes]) == expected


def test_full_selection_uses_policy_metrics_then_canonical_smiles_and_candidate_id() -> None:
    candidates = [
        {
            "candidate_id": "candidate-z",
            "canonical_smiles": "CCC",
            "generator_name": "fragfm",
        },
        {
            "candidate_id": "candidate-a",
            "canonical_smiles": "CCC",
            "generator_name": "hfm_3d",
        },
        {
            "candidate_id": "candidate-nonpassing",
            "canonical_smiles": "CCN",
            "generator_name": "crem",
        },
    ]
    records = [
        _record(
            "candidate-z",
            "CCC",
            outcome="PASS",
            admet_score=0.8,
            rbfe=-8.0,
            evidence_id="evidence-z",
        ),
        _record(
            "candidate-a",
            "CCC",
            outcome="PASS",
            admet_score=0.8,
            rbfe=-8.0,
            evidence_id="evidence-a",
        ),
        _record(
            "candidate-nonpassing",
            "CCN",
            outcome="FAIL",
            admet_score=0.8,
            rbfe=-6.5,
            evidence_id="evidence-nonpassing",
        ),
    ]
    state = _full_state(candidates)
    state["validation"] = {
        "outcome": "PASS",
        "records": records,
        "results": records,
    }

    candidate, record, candidate_index = orchestrator_main._selected_full_candidate(state)

    assert candidate is candidates[1]
    assert record is records[1]
    assert candidate_index == 1


class _GenerationRequestClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def request(
        self,
        _subject: str,
        payload: dict[str, Any],
        *,
        payload_type_url: str,
        timeout: float,
    ) -> dict[str, Any]:
        self.calls.append(deepcopy(payload))
        return {
            "run_id": payload["run_id"],
            "request_id": payload["request_id"],
            "schema_version": payload["schema_version"],
            "status": "dispatched",
            "candidates": [
                {
                    "candidate_id": "candidate-a",
                    "canonical_smiles": "CCO",
                    "generator_name": "hfm_3d",
                }
            ],
        }


@pytest.mark.parametrize(
    ("request_controls", "n_samples", "sampling_seed"),
    [
        ({}, 4, 42),
        (
            {
                "batch_size": 3,
                "generator_params": {"sampling_seed": 7},
            },
            3,
            7,
        ),
        (
            {
                "n_samples": 2,
                "batch_size": 9,
                "sampling_seed": 11,
                "generator_params": {"sampling_seed": 7},
            },
            2,
            11,
        ),
        ({"n_samples": 2, "seed": 0}, 2, 0),
    ],
)
async def test_agent_and_rest_generation_controls_are_identical(
    request_controls: dict[str, Any],
    n_samples: int,
    sampling_seed: int,
) -> None:
    request = {
        "project_id": "project-generation",
        "request_id": "request-generation",
        **request_controls,
    }
    state = {
        "run_id": "run-generation",
        "trace_id": "trace-generation",
        "refinement_count": 0,
        "request": request,
    }
    agent_client = _GenerationRequestClient()
    rest_client = _GenerationRequestClient()

    await _FullAgentWorkflowClients(agent_client).generate_candidates(deepcopy(state))
    await orchestrator_main.FullWorkflowClients(rest_client).generate_candidates(deepcopy(state))

    for payload in (agent_client.calls[0], rest_client.calls[0]):
        assert payload["n_samples"] == n_samples
        assert payload["batch_size"] == n_samples
        assert payload["generator_params"]["sampling_seed"] == sampling_seed


@pytest.mark.parametrize(
    "controls",
    [
        {"n_samples": 0},
        {"n_samples": True},
        {"n_samples": "4"},
        {"batch_size": -1},
        {"sampling_seed": True},
        {"sampling_seed": -1},
        {"generator_params": []},
        {"generator_params": {"sampling_seed": "42"}},
    ],
)
def test_generation_controls_reject_invalid_values(
    controls: dict[str, Any],
) -> None:
    validator = getattr(graph_builder, "generation_controls", None)

    assert callable(validator)
    with pytest.raises(ValueError):
        validator(controls)


@pytest.mark.parametrize(
    ("outcome", "expected_route"),
    [
        ("PASS", "done"),
        ("FAIL", "refine"),
        ("ERROR", "error"),
        ("AWAITING_EVIDENCE", "await"),
    ],
)
def test_full_graph_routes_validation_outcomes_without_refining_technical_states(
    outcome: str,
    expected_route: str,
) -> None:
    graph = WorkflowGraph()
    state = {
        "validation": {"outcome": outcome},
        "validation_passed": outcome == "PASS",
        "refinement_count": 0,
        "max_refinements": 1,
    }

    assert graph._route_after_validation(state) == expected_route


async def test_cig_policy_direction_conflict_rejects_before_generation() -> None:
    calls: list[str] = []

    class _Clients:
        async def compile_intent(self, state: dict[str, Any]) -> dict[str, Any]:
            calls.append("compile")
            return {
                "cig": {
                    "objectives": [
                        {
                            "id": "objective-affinity",
                            "property": "rbfe",
                            "type": "MAXIMIZE",
                        }
                    ]
                }
            }

        async def generate_candidates(self, state: dict[str, Any]) -> list[dict[str, Any]]:
            calls.append("generate")
            return []

        async def validate_candidates(self, state: dict[str, Any]) -> dict[str, Any]:
            calls.append("validate")
            return {"outcome": "FAIL", "passed": False, "records": []}

    state = {
        "nl_input": "Design an affinity candidate",
        "run_id": "run-conflict",
        "trace_id": "trace-conflict",
        "artifact_ids": [],
        "history": [],
        "events": [],
        "workflow_scope": "full",
        "refinement_count": 0,
        "max_refinements": 0,
        "validation_passed": True,
        "request": {
            "validation_policy": _validation_policy(),
            "teacher_policy": _teacher_policy(),
            "selection_policy": _selection_policy(),
        },
    }

    result = await WorkflowGraph(clients=_Clients()).build().ainvoke(state)

    assert result["status"] == "ESCALATING"
    assert result["invalid_policy"]["conflicts"]
    assert calls == ["compile"]


async def test_full_awaiting_evidence_finishes_in_stable_state_without_retry() -> None:
    calls: list[str] = []

    class _Clients:
        async def compile_intent(self, state: dict[str, Any]) -> dict[str, Any]:
            calls.append("compile")
            return {"cig": {"objectives": []}}

        async def generate_candidates(self, state: dict[str, Any]) -> list[dict[str, Any]]:
            calls.append("generate")
            return [
                {
                    "candidate_id": "candidate-a",
                    "canonical_smiles": "CCO",
                    "generator_name": "hfm_3d",
                }
            ]

        async def validate_candidates(self, state: dict[str, Any]) -> dict[str, Any]:
            calls.append("validate")
            return {
                "outcome": "AWAITING_EVIDENCE",
                "passed": False,
                "records": [],
                "results": [],
            }

        async def plan_routes(self, state: dict[str, Any]) -> dict[str, Any]:
            calls.append("retrosyn")
            return {}

    state = {
        "nl_input": "Design an evidence-backed candidate",
        "run_id": "run-awaiting",
        "trace_id": "trace-awaiting",
        "artifact_ids": [],
        "history": [],
        "events": [],
        "workflow_scope": "full",
        "refinement_count": 0,
        "max_refinements": 1,
        "validation_passed": True,
        "request": {
            "validation_policy": _validation_policy(oracle_level=4),
            "teacher_policy": _teacher_policy(),
            "selection_policy": _selection_policy(),
        },
    }
    result = await WorkflowGraph(clients=_Clients()).build().ainvoke(state)

    assert result["status"] == "AWAITING_EVIDENCE"
    assert result["validation_outcome"] == "AWAITING_EVIDENCE"
    assert calls == ["compile", "generate", "validate"]
    assert orchestrator_main._workflow_terminal_status(result).value == "awaiting_evidence"


async def test_full_external_evidence_resume_reenters_at_validation() -> None:
    calls: list[str] = []

    class _Clients:
        async def compile_intent(self, state: dict[str, Any]) -> dict[str, Any]:
            raise AssertionError("resume must not recompile intent")

        async def generate_candidates(self, state: dict[str, Any]) -> list[dict[str, Any]]:
            raise AssertionError("resume must not regenerate candidates")

        async def validate_candidates(self, state: dict[str, Any]) -> dict[str, Any]:
            calls.append("validate")
            assert state["request"]["external_evidence"][0]["candidate_id"] == "candidate-a"
            return {
                "outcome": "PASS",
                "passed": True,
                "records": [
                    {
                        "candidate_id": "candidate-a",
                        "canonical_smiles": "CCO",
                        "outcome": "PASS",
                        "metrics": [],
                    }
                ],
                "results": [],
            }

        async def plan_routes(self, state: dict[str, Any]) -> dict[str, Any]:
            calls.append("retrosyn")
            return {"routes": [{"route_id": "route-a"}]}

        async def assess_supply(self, state: dict[str, Any]) -> dict[str, Any]:
            calls.append("supply")
            return {
                "route_id": "route-a",
                "supply_assessment": {"overall_feasibility": "available"},
            }

        async def compile_synthesis(self, state: dict[str, Any]) -> dict[str, Any]:
            calls.append("srb")
            return {
                "status": "compiled",
                "route_id": "route-a",
                "protocols": [{"route_id": "route-a", "ssp_id": "ssp-a"}],
            }

        async def review_candidates(self, state: dict[str, Any]) -> dict[str, Any]:
            calls.append("critic")
            return {"verdict": "pass"}

        async def execute_synthesis(self, state: dict[str, Any]) -> dict[str, Any]:
            calls.append("execute")
            return {
                "status": "executed",
                "route_id": "route-a",
                "protocols": [{"route_id": "route-a", "ssp_id": "ssp-a"}],
            }

    state = {
        "nl_input": "Design an evidence-backed candidate",
        "run_id": "run-resume",
        "trace_id": "trace-resume",
        "artifact_ids": [],
        "history": ["PLANNING", "GENERATING", "VALIDATING", "AWAITING_EVIDENCE"],
        "events": [],
        "workflow_scope": "full",
        "refinement_count": 0,
        "max_refinements": 1,
        "validation_passed": False,
        "candidates": [
            {
                "candidate_id": "candidate-a",
                "canonical_smiles": "CCO",
                "generator_name": "hfm_3d",
            }
        ],
        "request": {
            "validation_policy": _validation_policy(oracle_level=4),
            "teacher_policy": _teacher_policy(),
            "selection_policy": _selection_policy(),
            "external_evidence": [
                {
                    "candidate_id": "candidate-a",
                    "metrics": {"activity": 0.8},
                    "evidence_ids": ["artifact:measurement-1"],
                }
            ],
        },
    }

    result = await (
        WorkflowGraph(clients=_Clients())
        .build(entry_point="validating")
        .ainvoke(state)
    )

    assert result["status"] == "EXECUTING"
    assert calls == ["validate", "retrosyn", "supply", "srb", "critic", "execute"]
    assert result["history"].count("PLANNING") == 1
    assert result["history"].count("GENERATING") == 1


@pytest.mark.parametrize(
    ("critic_verdict", "expected_status", "expected_execution_calls"),
    [
        ("pass", "EXECUTING", 1),
        ("fail", "ESCALATING", 0),
    ],
)
async def test_full_workflow_executes_sila2_only_after_critic_pass(
    critic_verdict: str,
    expected_status: str,
    expected_execution_calls: int,
) -> None:
    calls: list[str] = []

    class _Clients:
        async def compile_intent(self, _state):
            calls.append("planning")
            return {"cig": {"objectives": []}}

        async def generate_candidates(self, _state):
            calls.append("generating")
            return [
                {
                    "candidate_id": "candidate-a",
                    "canonical_smiles": "CCO",
                    "generator_name": "hfm_3d",
                }
            ]

        async def validate_candidates(self, _state):
            calls.append("validating")
            record = _record(
                "candidate-a",
                "CCO",
                outcome="PASS",
                admet_score=0.8,
                rbfe=-8.0,
                evidence_id="validation-a",
            )
            return {
                "outcome": "PASS",
                "passed": True,
                "records": [record],
                "results": [record],
            }

        async def plan_routes(self, _state):
            calls.append("retrosyn")
            return {"routes": [{"route_id": "route-a"}]}

        async def assess_supply(self, _state):
            calls.append("supply")
            return {
                "route_id": "route-a",
                "supply_assessment": {"overall_feasibility": "available"},
            }

        async def compile_synthesis(self, _state):
            calls.append("compile")
            return {
                "route_id": "route-a",
                "protocols": [{"ssp_id": "ssp-a", "route_id": "route-a"}],
            }

        async def review_candidates(self, _state):
            calls.append("critic")
            return {
                "verdict": critic_verdict,
                "rule_results": (
                    []
                    if critic_verdict == "pass"
                    else [
                        {
                            "rule_id": "workflow_srb_protocols",
                            "verdict": "fail",
                            "blocking": True,
                        }
                    ]
                ),
            }

        async def submit_critic_feedback(self, _state):
            calls.append("feedback")
            return {
                "action": "generator_coord/feedback/v1",
                "status": "feedback_submitted",
                "submitted": 1,
                "duplicates": 0,
            }

        async def execute_synthesis(self, state):
            calls.append("execute")
            assert state["critic"]["verdict"] == "pass"
            return {
                "status": "executed",
                "route_id": "route-a",
                "protocols": state["srb"]["protocols"],
            }

    state = {
        **_full_state([]),
        "nl_input": "Design a molecule",
        "history": [],
        "events": [],
        "artifact_ids": [],
        "workflow_scope": "full",
        "max_refinements": 0,
        "validation_passed": False,
    }

    result = await WorkflowGraph(clients=_Clients()).build().ainvoke(state)

    assert result["status"] == expected_status
    assert calls.count("execute") == expected_execution_calls
    if critic_verdict == "pass":
        assert calls[-2:] == ["critic", "execute"]
        assert result["srb_execution"]["route_id"] == "route-a"
    else:
        assert calls[-2:] == ["critic", "feedback"]
        assert "srb_execution" not in result


async def test_full_agent_clients_assess_all_routes_and_bind_selected_route_through_execution(
) -> None:
    candidate = {
        "candidate_id": "candidate-a",
        "canonical_smiles": "CCO",
        "generator_name": "hfm_3d",
    }
    record = _record(
        "candidate-a",
        "CCO",
        outcome="PASS",
        admet_score=0.8,
        rbfe=-8.0,
        evidence_id="validation-a",
    )
    state = {
        **_full_state([candidate]),
        "workflow_scope": "full",
        "validation": {
            "outcome": "PASS",
            "records": [record],
            "results": [record],
        },
    }
    state["request"]["retrosyn_engine"] = "rsgpt"
    payloads: list[dict[str, Any]] = []

    class _Client:
        async def request(self, subject, payload, **_kwargs):
            payloads.append(dict(payload))
            identity = {
                field: payload[field]
                for field in (
                    "project_id",
                    "candidate_id",
                    "candidate_index",
                    "canonical_smiles",
                )
            }
            correlation = {
                field: payload[field] for field in ("run_id", "request_id", "schema_version")
            }
            if "retrosyn" in subject:
                return {
                    **identity,
                    **correlation,
                    "routes": [
                        {
                            "route_id": "route-a",
                            "building_blocks": [{"smiles": "CC"}],
                        },
                        {
                            "route_id": "route-b",
                            "building_blocks": [{"smiles": "CN"}],
                        },
                    ],
                }
            if "supply" in subject:
                return {
                    **identity,
                    **correlation,
                    "route_id": payload["route_id"],
                    "supply_assessment": {"overall_feasibility": "available"},
                }
            if "srb" in subject:
                protocol = dict((payload.get("protocols") or [{"ssp_id": "ssp-a"}])[0])
                protocol["route_id"] = payload["route_id"]
                return {
                    **identity,
                    **correlation,
                    "status": "executed" if payload.get("action") == "execute" else "compiled",
                    "route_id": payload["route_id"],
                    "protocols": [protocol],
                }
            if "critic" in subject:
                return {
                    **identity,
                    **correlation,
                    "verdict": "pass",
                    "rule_results": [],
                }
            raise AssertionError(subject)

    clients = _FullAgentWorkflowClients(_Client())
    state["retrosyn"] = await clients.plan_routes(state)
    state["supply"] = await clients.assess_supply(state)
    state["srb"] = await clients.compile_synthesis(state)
    state["critic"] = await clients.review_candidates(state)
    state["srb_execution"] = await clients.execute_synthesis(state)

    (
        retrosyn_payload,
        first_supply_payload,
        second_supply_payload,
        compile_payload,
        critic_payload,
        execute_payload,
    ) = payloads
    assert retrosyn_payload["engine"] == "rsgpt"
    assert first_supply_payload["workflow_scope"] == "full"
    assert first_supply_payload["route_id"] == "route-a"
    assert first_supply_payload["building_blocks"] == [{"smiles": "CC"}]
    assert second_supply_payload["route_id"] == "route-b"
    assert second_supply_payload["building_blocks"] == [{"smiles": "CN"}]
    assert [route["route_id"] for route in compile_payload["pathways"]] == ["route-a"]
    assert compile_payload["route_id"] == "route-a"
    assert critic_payload["properties"]["selected_route_id"] == "route-a"
    assert execute_payload["action"] == "execute"
    assert execute_payload["route_id"] == "route-a"
    assert [protocol["route_id"] for protocol in execute_payload["protocols"]] == ["route-a"]
    assert compile_payload["request_id"] != execute_payload["request_id"]


async def test_full_agent_client_rejects_missing_retrosyn_engine() -> None:
    candidate = {
        "candidate_id": "candidate-a",
        "canonical_smiles": "CCO",
        "generator_name": "hfm_3d",
    }
    record = _record(
        "candidate-a",
        "CCO",
        outcome="PASS",
        admet_score=0.8,
        rbfe=-8.0,
        evidence_id="validation-a",
    )
    state = {
        **_full_state([candidate]),
        "workflow_scope": "full",
        "validation": {
            "outcome": "PASS",
            "records": [record],
            "results": [record],
        },
    }
    state["request"].pop("retrosyn_engine")

    class _Client:
        async def request(self, *_args, **_kwargs):
            raise AssertionError("retrosyn request must not be sent without an engine")

    with pytest.raises(ValueError, match="retrosyn_engine"):
        await _FullAgentWorkflowClients(_Client()).plan_routes(state)


async def test_full_agent_client_does_not_compile_an_unavailable_route() -> None:
    candidate = {
        "candidate_id": "candidate-a",
        "canonical_smiles": "CCO",
        "generator_name": "hfm_3d",
    }
    record = _record(
        "candidate-a",
        "CCO",
        outcome="PASS",
        admet_score=0.8,
        rbfe=-8.0,
        evidence_id="validation-a",
    )
    state = {
        **_full_state([candidate]),
        "workflow_scope": "full",
        "validation": {
            "outcome": "PASS",
            "records": [record],
            "results": [record],
        },
        "retrosyn": {
            "routes": [
                {
                    "route_id": "route-a",
                    "building_blocks": [{"smiles": "CC"}],
                }
            ]
        },
        "supply": {
            "route_id": "route-a",
            "supply_assessment": {"overall_feasibility": "unavailable"},
        },
    }

    class _Client:
        async def request(self, *_args, **_kwargs):
            raise AssertionError("unavailable route must not be sent to SRB")

    result = await _FullAgentWorkflowClients(_Client()).compile_synthesis(state)

    assert result["status"] == "not_compiled"
    assert result["route_id"] == "route-a"
    assert result["protocols"] == []
    assert result["blocking_evidence"] == [
        {
            "rule_id": "workflow_supply_feasibility",
            "reason": "selected route supply feasibility is unavailable",
        }
    ]


async def test_reasoning_pipeline_forwards_explicit_full_policies() -> None:
    calls: list[dict[str, Any]] = []

    class _Pipeline(ReasoningPipeline):
        async def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
            calls.append({"path": path, "payload": payload})
            return {"run_id": "run-policy"}

    pipeline = _Pipeline("http://orchestrator.test")
    run_id = await pipeline.submit(
        "Design a molecule",
        workflow_scope="full",
        max_refinements=1,
        validation_policy=_validation_policy(),
        teacher_policy=_teacher_policy(),
        selection_policy=_selection_policy(),
    )

    assert run_id == "run-policy"
    assert calls[0]["payload"]["validation_policy"] == _validation_policy()
    assert calls[0]["payload"]["teacher_policy"] == _teacher_policy()
    assert calls[0]["payload"]["selection_policy"] == _selection_policy()
    assert "validation_passed" not in calls[0]["payload"]


@pytest.mark.parametrize(
    "clients_type",
    [_FullAgentWorkflowClients, orchestrator_main.FullWorkflowClients],
)
@pytest.mark.parametrize(
    "method_name",
    ["plan_routes", "assess_supply", "compile_synthesis", "review_candidates"],
)
async def test_full_downstream_clients_reject_selected_identity_mismatch(
    clients_type,
    method_name: str,
) -> None:
    candidate = {
        "candidate_id": "candidate-a",
        "canonical_smiles": "CCO",
        "generator_name": "hfm_3d",
    }
    state = _full_state([candidate])
    record = _record(
        "candidate-a",
        "CCO",
        outcome="PASS",
        admet_score=0.8,
        rbfe=-8.0,
        evidence_id="evidence-a",
    )
    state.update(
        {
            "workflow_scope": "full",
            "validation": {
                "outcome": "PASS",
                "records": [record],
                "results": [record],
            },
            "retrosyn": {
                "project_id": "project-policy",
                "candidate_id": "candidate-a",
                "candidate_index": 0,
                "canonical_smiles": "CCO",
                "routes": [
                    {
                        "route_id": "route-a",
                        "building_blocks": [{"smiles": "CC"}],
                        "steps": [],
                    }
                ],
            },
            "supply": {
                "project_id": "project-policy",
                "candidate_id": "candidate-a",
                "candidate_index": 0,
                "canonical_smiles": "CCO",
                "route_id": "route-a",
                "supply_assessment": {"overall_feasibility": "available"},
            },
            "srb": {
                "project_id": "project-policy",
                "candidate_id": "candidate-a",
                "candidate_index": 0,
                "canonical_smiles": "CCO",
                "route_id": "route-a",
                "protocols": [{"ssp_id": "ssp-a", "route_id": "route-a", "steps": []}],
            },
        }
    )

    class _Client:
        async def request(self, _subject, payload, **_kwargs):
            return {
                "project_id": payload["project_id"],
                "candidate_id": "wrong-candidate",
                "candidate_index": payload["candidate_index"],
                "canonical_smiles": payload["canonical_smiles"],
                "status": "ok",
                "routes": [],
                "supply_assessment": {"overall_feasibility": "available"},
                "protocols": [],
                "verdict": "pass",
                "run_id": payload["run_id"],
                "request_id": payload["request_id"],
                "schema_version": payload["schema_version"],
            }

    with pytest.raises(RuntimeError, match="candidate_id"):
        await getattr(clients_type(_Client()), method_name)(state)


@pytest.mark.parametrize("max_refinements", [0, 1])
async def test_full_critic_failure_is_acknowledged_before_reject_or_refine(
    max_refinements: int,
) -> None:
    calls: list[str] = []

    class _Clients:
        def __init__(self) -> None:
            self.critic_calls = 0

        async def compile_intent(self, _state):
            return {"cig": {"objectives": []}}

        async def generate_candidates(self, _state):
            calls.append("generate")
            return [
                {
                    "candidate_id": "candidate-a",
                    "canonical_smiles": "CCO",
                    "generator_name": "hfm_3d",
                }
            ]

        async def validate_candidates(self, _state):
            record = _record(
                "candidate-a",
                "CCO",
                outcome="PASS",
                admet_score=0.8,
                rbfe=-8.0,
                evidence_id="validation-a",
            )
            return {
                "outcome": "PASS",
                "passed": True,
                "records": [record],
                "results": [record],
            }

        async def plan_routes(self, _state):
            return {"routes": [{"route_id": "route-a"}]}

        async def assess_supply(self, _state):
            return {
                "route_id": "route-a",
                "supply_assessment": {"overall_feasibility": "available"},
            }

        async def compile_synthesis(self, _state):
            return {
                "route_id": "route-a",
                "protocols": [{"ssp_id": "ssp-a", "route_id": "route-a"}],
            }

        async def review_candidates(self, _state):
            self.critic_calls += 1
            calls.append("critic")
            if self.critic_calls == 1:
                return {
                    "verdict": "fail",
                    "rule_results": [
                        {
                            "rule_id": "workflow_srb_protocols",
                            "verdict": "fail",
                            "blocking": True,
                        }
                    ],
                }
            return {"verdict": "pass", "rule_results": []}

        async def submit_critic_feedback(self, state):
            assert state["critic"]["verdict"] == "fail"
            calls.append("feedback")
            return {
                "action": "generator_coord/feedback/v1",
                "status": "feedback_submitted",
                "submitted": 1,
                "duplicates": 0,
            }

        async def execute_synthesis(self, state):
            calls.append("execute")
            return {
                "status": "executed",
                "route_id": "route-a",
                "protocols": state["srb"]["protocols"],
            }

    state = {
        **_full_state([]),
        "nl_input": "Design a molecule",
        "history": [],
        "events": [],
        "artifact_ids": [],
        "workflow_scope": "full",
        "max_refinements": max_refinements,
        "validation_passed": True,
    }

    result = (
        await WorkflowGraph(
            clients=_Clients(),
        )
        .build()
        .ainvoke(state)
    )

    assert calls[:4] == [
        "generate",
        "critic",
        "feedback",
        *(["generate"] if max_refinements else []),
    ]
    assert result["status"] == ("EXECUTING" if max_refinements else "ESCALATING")


def test_full_critic_properties_include_explicit_downstream_blockers() -> None:
    properties = graph_builder.full_workflow_critic_properties(
        {
            "retrosyn": {"routes": []},
            "supply": {
                "supply_assessment": {
                    "overall_feasibility": "unavailable",
                    "total_blocks": 0,
                    "commercially_available": 0,
                }
            },
            "srb": {"protocols": []},
            "request": {},
        },
        {
            "candidate_id": "candidate-a",
            "canonical_smiles": "CCO",
        },
    )

    assert properties["retrosyn_route_count"] == 0
    assert properties["supply_feasibility"] == "unavailable"
    assert properties["building_block_availability"] == 0.0
    assert properties["srb_protocol_count"] == 0
    assert {
        "workflow_retrosyn_routes",
        "workflow_supply_feasibility",
        "workflow_srb_protocols",
        "rule_081",
    } <= set(properties["_critic_blocking_rule_ids"])


async def test_critic_agent_emits_structured_blocking_downstream_evidence() -> None:
    from critic_agent.agent import ScientificCriticAgent

    agent = ScientificCriticAgent(crg_repository=None)
    agent.rules = []
    properties = {
        "retrosyn_route_count": 0,
        "supply_feasibility": "unavailable",
        "srb_protocol_count": 0,
        "_critic_blocking_rule_ids": [
            "workflow_retrosyn_routes",
            "workflow_supply_feasibility",
            "workflow_srb_protocols",
        ],
    }

    result = await agent.evaluate_molecule(
        {
            "smiles": "CCO",
            "properties": properties,
        }
    )

    assert result["verdict"] == "fail"
    assert result["blocking_failed"] == 3
    assert {row["rule_id"] for row in result["rule_results"] if row["blocking"]} == {
        "workflow_retrosyn_routes",
        "workflow_supply_feasibility",
        "workflow_srb_protocols",
    }


async def test_critic_agent_full_process_echoes_exact_candidate_identity() -> None:
    from critic_agent.agent import ScientificCriticAgent

    result = await ScientificCriticAgent(crg_repository=None).process(
        {
            "workflow_scope": "full",
            "project_id": "project-a",
            "run_id": "run-a",
            "request_id": "request-a",
            "schema_version": "critic.request.v1",
            "candidate_id": "candidate-a",
            "candidate_index": 0,
            "canonical_smiles": "CCO",
            "smiles": "CCO",
            "properties": {},
        }
    )

    assert {
        field: result[field]
        for field in (
            "project_id",
            "candidate_id",
            "candidate_index",
            "canonical_smiles",
        )
    } == {
        "project_id": "project-a",
        "candidate_id": "candidate-a",
        "candidate_index": 0,
        "canonical_smiles": "CCO",
    }
