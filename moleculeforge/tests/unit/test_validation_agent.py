"""Validation agent policy, batching, and outcome contracts."""

from __future__ import annotations

import asyncio
import importlib.util
import math
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import grpc
import pytest
from mf_core.proto_gen.moleculeforge.v1.core import audit_pb2
from mf_core.proto_gen.moleculeforge.v1.oracle import oracle_pb2

ROOT = Path(__file__).resolve().parents[2]
VALID_ARTIFACT_CHECKSUM = f"sha256:{'a' * 64}"


def _artifact_ref(
    *,
    name: str = "oracle-model",
    checksum: str = VALID_ARTIFACT_CHECKSUM,
    required: bool = True,
) -> audit_pb2.ArtifactRef:
    return audit_pb2.ArtifactRef(
        name=name,
        version="1",
        checksum=checksum,
        required=required,
    )


def _successful_wire_evaluation(
    *,
    molecule_smiles: str = "CCO",
    evidence_id: str = "request-1:rdkit:0",
) -> oracle_pb2.OracleEvaluation:
    return oracle_pb2.OracleEvaluation(
        oracle_name="rdkit",
        molecule_smiles=molecule_smiles,
        level=oracle_pb2.L0_RDKIT,
        scores={"admet_score": 0.9},
        success=True,
        outcome=oracle_pb2.ORACLE_OUTCOME_PASS,
        artifact_refs=[_artifact_ref()],
        evidence_id=evidence_id,
        metrics=[
            oracle_pb2.OracleMetric(
                property="admet_score",
                value=0.9,
            )
        ],
    )


def _consume_wire_evaluations(
    module: ModuleType,
    evaluations: list[oracle_pb2.OracleEvaluation],
    *,
    molecules: list[str] | None = None,
    properties: list[str] | None = None,
    level: int = 0,
    oracle_name: str = "rdkit",
    request_context: dict | None = None,
) -> dict[str, dict]:
    return module._evaluations_by_smiles(
        oracle_pb2.OracleBatchResponse(
            batch_id="request-1",
            evaluations=evaluations,
        ),
        molecules or ["CCO"],
        properties or ["admet_score"],
        expected_level=level,
        expected_oracle_name=oracle_name,
        request_context=request_context or {},
        request_id="request-1",
    )


def _load_source_module(module_name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_validation_module() -> ModuleType:
    return _load_source_module(
        "validation_agent_test",
        ROOT / "agents/validation_agent/src/validation_agent/agent.py",
    )


async def _oracle_grpc_call(
    module: ModuleType,
    servicer: object,
    request: oracle_pb2.OracleBatchRequest,
) -> oracle_pb2.OracleBatchResponse:
    server = grpc.aio.server()
    module.oracle_pb2_grpc.add_OracleServiceServicer_to_server(servicer, server)
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    channel = grpc.aio.insecure_channel(f"127.0.0.1:{port}")
    try:
        stub = module.oracle_pb2_grpc.OracleServiceStub(channel)
        return await stub.Evaluate(request)
    finally:
        await channel.close()
        await server.stop(None)


def _threshold(
    level: int,
    oracle: str,
    metric: str,
    direction: str,
    value: float,
    *,
    max_uncertainty: float | None = None,
) -> dict:
    result = {
        "level": level,
        "oracle": oracle,
        "metric": metric,
        "direction": direction,
        "value": value,
    }
    if max_uncertainty is not None:
        result["max_uncertainty"] = max_uncertainty
    return result


def _policy(level: int, *, boltz2: bool = False) -> dict:
    thresholds = [
        _threshold(0, "rdkit", "admet_score", "maximize", 0.5),
    ]
    if level >= 1:
        thresholds.append(_threshold(1, "admet", "clearance", "minimize", 1.0))
        if boltz2:
            thresholds.append(
                _threshold(
                    1,
                    "boltz2",
                    "affinity",
                    "minimize",
                    -7.0,
                    max_uncertainty=0.5,
                )
            )
    if level >= 2:
        thresholds.append(_threshold(2, "dock", "docking_score", "minimize", -6.0))
    if level >= 3:
        thresholds.append(_threshold(3, "fep", "rbfe", "minimize", 1.0))
    if level >= 4:
        thresholds.append(_threshold(4, "external", "activity", "maximize", 0.75))
    return {
        "oracle_level": level,
        "batch_size": 2,
        "max_concurrency": 2,
        "thresholds": thresholds,
        "oracle_inputs": {
            "boltz2": {"protein_pdb_id": "8ABC"},
            "dock": {
                "receptor_uri": "file:///receptor.pdb",
                "oracle_parameters": {"engine": "gnina"},
            },
            "fep": {
                "protein_pdb_id": "8ABC",
                "reference_ligand_smiles": "CC",
                "oracle_parameters": {
                    "method": "openfe",
                    "n_repeats": 3,
                },
            },
        },
    }


def _payload(
    level: int,
    *,
    candidates: list[dict] | None = None,
    boltz2: bool = False,
    external_evidence: list[dict] | None = None,
) -> dict:
    result = {
        "project_id": "project-1",
        "run_id": "run-1",
        "request_id": "request-1",
        "candidates": candidates or [{"candidate_id": "candidate-1", "canonical_smiles": "CCO"}],
        "validation_policy": _policy(level, boltz2=boltz2),
    }
    if external_evidence is not None:
        result["external_evidence"] = external_evidence
    return result


class _CrgRepository:
    def __init__(self) -> None:
        self.writes: list[dict] = []

    async def write_workflow_belief(self, **kwargs: object) -> None:
        self.writes.append(dict(kwargs))


_DEFAULT_CRG_REPOSITORY = object()


def _validation_agent(
    module: ModuleType,
    *,
    crg_repository: object = _DEFAULT_CRG_REPOSITORY,
    **kwargs: object,
) -> object:
    repository = _CrgRepository() if crg_repository is _DEFAULT_CRG_REPOSITORY else crg_repository
    return module.ValidationAgent(
        crg_repository=repository,
        **kwargs,
    )


def test_fep_chunk_timeout_covers_openfe_repeats_and_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENFE_QUICKRUN_TIMEOUT_SECONDS", "10")
    monkeypatch.setenv("OPENFE_GATHER_TIMEOUT_SECONDS", "2")
    monkeypatch.setenv("OPENFE_MAX_TRANSFORMATIONS_PER_PAIR", "2")
    module = _load_validation_module()

    timeout = module._oracle_chunk_timeout_seconds(
        "fep",
        ["CCO", "CCN"],
        {
            "oracle_parameters": {
                "method": "openfe",
                "n_repeats": "3",
            }
        },
        default_timeout_seconds=7.0,
    )

    assert timeout == 222.0


class _BatchOracle:
    def __init__(
        self,
        values: dict[str, float] | None = None,
        *,
        values_by_smiles: dict[str, dict] | None = None,
        delay: float = 0.0,
        tracker: dict | None = None,
        reverse: bool = False,
        drop_last: bool = False,
    ) -> None:
        self.values = dict(values or {})
        self.values_by_smiles = values_by_smiles
        self.delay = delay
        self.tracker = tracker
        self.reverse = reverse
        self.drop_last = drop_last
        self.calls: list[dict] = []

    async def evaluate(
        self,
        molecules: list[str],
        properties: list[str],
        *,
        request_context: dict | None = None,
    ) -> dict:
        self.calls.append(
            {
                "molecules": list(molecules),
                "properties": list(properties),
                "request_context": dict(request_context or {}),
            }
        )
        if self.tracker is not None:
            self.tracker["active"] += 1
            self.tracker["maximum"] = max(
                self.tracker["maximum"],
                self.tracker["active"],
            )
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
            order = list(reversed(molecules)) if self.reverse else list(molecules)
            if self.drop_last:
                order = order[:-1]
            return {
                smiles: dict((self.values_by_smiles or {}).get(smiles, self.values))
                for smiles in order
            }
        finally:
            if self.tracker is not None:
                self.tracker["active"] -= 1


class _UncertaintyOracle(_BatchOracle):
    def __init__(
        self,
        scores: dict[str, float],
        uncertainty: dict[str, float],
        **kwargs: object,
    ) -> None:
        super().__init__(scores, **kwargs)
        self.uncertainty = dict(uncertainty)

    async def predict_with_uncertainty(
        self,
        molecules: list[str],
        properties: list[str],
        *,
        request_context: dict | None = None,
    ) -> dict:
        await super().evaluate(
            molecules,
            properties,
            request_context=request_context,
        )
        return {smiles: (dict(self.values), dict(self.uncertainty)) for smiles in molecules}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda policy: policy.pop("oracle_level"), "oracle_level is required"),
        (
            lambda policy: policy.__setitem__("oracle_levle", 0),
            "unsupported validation_policy field: oracle_levle",
        ),
        (
            lambda policy: policy.__setitem__("oracle_level", True),
            "oracle_level must be an integer between 0 and 4",
        ),
        (
            lambda policy: policy.__setitem__("oracle_level", 5),
            "oracle_level must be an integer between 0 and 4",
        ),
        (lambda policy: policy.pop("batch_size"), "batch_size is required"),
        (
            lambda policy: policy.__setitem__("batch_size", 0),
            "batch_size must be a positive integer",
        ),
        (
            lambda policy: policy.__setitem__("max_concurrency", False),
            "max_concurrency must be a positive integer",
        ),
        (
            lambda policy: policy.__setitem__("thresholds", {}),
            "thresholds must be a list",
        ),
        (
            lambda policy: policy["thresholds"][0].__setitem__("value", math.nan),
            "threshold value must be a finite number",
        ),
        (
            lambda policy: policy["thresholds"][0].__setitem__("valeu", 0.5),
            "unsupported threshold field: valeu",
        ),
        (
            lambda policy: policy["thresholds"][0].__setitem__(
                "direction",
                "ascending",
            ),
            "threshold direction must be maximize or minimize",
        ),
        (
            lambda policy: policy["thresholds"][0].__setitem__(
                "max_uncertainty",
                -0.1,
            ),
            "threshold max_uncertainty must be non-negative",
        ),
        (
            lambda policy: policy.__setitem__("oracle_inputs", []),
            "oracle_inputs must be an object",
        ),
        (
            lambda policy: policy["oracle_inputs"]["dock"].__setitem__(
                "receptro_uri",
                "file:///typo.pdb",
            ),
            "unsupported oracle_inputs.dock field",
        ),
        (
            lambda policy: policy["oracle_inputs"]["dock"]["oracle_parameters"].__setitem__(
                "engine", "vina"
            ),
            "dock engine must be gnina or diffdock",
        ),
        (
            lambda policy: policy["oracle_inputs"]["fep"]["oracle_parameters"].__setitem__(
                "n_repeats", 0
            ),
            "fep n_repeats must be a positive integer",
        ),
        (
            lambda policy: policy["oracle_inputs"]["fep"]["oracle_parameters"].__setitem__(
                "n_repeats", "3"
            ),
            "fep n_repeats must be a positive integer",
        ),
    ],
)
async def test_validation_policy_is_strict(
    mutate: Callable[[dict], object],
    message: str,
) -> None:
    module = _load_validation_module()
    payload = _payload(0)
    mutate(payload["validation_policy"])
    oracle = _BatchOracle({"admet_score": 0.9})

    with pytest.raises(ValueError, match=message):
        await _validation_agent(module, oracles={"rdkit": oracle}).process(payload)

    assert oracle.calls == []


@pytest.mark.parametrize("oracle_name", ["rdkit", "admet", "external"])
def test_validation_policy_oracle_inputs_only_accept_transport_inputs(
    oracle_name: str,
) -> None:
    module = _load_validation_module()

    with pytest.raises(
        ValueError,
        match=f"unsupported oracle_inputs key: {oracle_name}",
    ):
        module._parse_oracle_inputs({oracle_name: {}})


@pytest.mark.parametrize(
    ("oracle_name", "parameters"),
    [
        ("rdkit", {"typo": "value"}),
        ("admet", {"typo": "value"}),
        ("boltz2", {"ensemble_szie": 3}),
        ("dock", {"engnie": "gnina"}),
        ("fep", {"n_repeat": 3}),
        ("external", {"typo": "value"}),
    ],
)
def test_oracle_parameters_reject_unknown_fields(
    oracle_name: str,
    parameters: dict,
) -> None:
    module = _load_validation_module()

    with pytest.raises(
        ValueError,
        match=rf"unsupported oracle parameter {oracle_name}",
    ):
        module._parse_oracle_parameters(oracle_name, parameters)


@pytest.mark.parametrize(
    ("oracle_name", "parameters", "expected"),
    [
        ("rdkit", {}, {}),
        ("admet", {}, {}),
        ("boltz2", {"ensemble_size": 3}, {"ensemble_size": "3"}),
        ("dock", {"engine": "gnina"}, {"engine": "gnina"}),
        (
            "fep",
            {"method": "openfe", "n_repeats": 3},
            {"method": "openfe", "n_repeats": "3"},
        ),
        ("external", {}, {}),
    ],
)
def test_oracle_parameters_accept_only_exact_fields(
    oracle_name: str,
    parameters: dict,
    expected: dict,
) -> None:
    module = _load_validation_module()

    assert module._parse_oracle_parameters(oracle_name, parameters) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda policy: policy["thresholds"].clear(),
            "L0 requires a rdkit threshold",
        ),
        (
            lambda policy: policy["thresholds"][0].__setitem__("level", 1),
            "threshold level exceeds oracle_level",
        ),
        (
            lambda policy: policy["thresholds"][0].__setitem__("oracle", "dock"),
            "oracle dock belongs to L2",
        ),
        (
            lambda policy: policy["thresholds"].append(dict(policy["thresholds"][0])),
            "duplicate threshold",
        ),
        (
            lambda policy: policy["thresholds"][0].__setitem__(
                "metric",
                "unknown",
            ),
            "unsupported rdkit metric",
        ),
    ],
)
async def test_validation_policy_rejects_incomplete_or_contradictory_thresholds(
    mutate: Callable[[dict], object],
    message: str,
) -> None:
    module = _load_validation_module()
    payload = _payload(0)
    mutate(payload["validation_policy"])

    with pytest.raises(ValueError, match=message):
        await _validation_agent(
            module,
            oracles={"rdkit": _BatchOracle({"admet_score": 0.9})},
        ).process(payload)


@pytest.mark.asyncio
async def test_same_level_oracles_batch_deduplicate_fanout_and_bound_concurrency() -> None:
    module = _load_validation_module()
    tracker = {"active": 0, "maximum": 0}
    rdkit = _BatchOracle({"admet_score": 0.9}, delay=0.01, tracker=tracker)
    admet = _BatchOracle({"clearance": 0.2}, delay=0.01, tracker=tracker)
    boltz2 = _UncertaintyOracle(
        {"affinity": -8.0},
        {"affinity": 0.2},
        delay=0.01,
        tracker=tracker,
    )
    candidates = [
        {"candidate_id": "candidate-a", "canonical_smiles": "CCO"},
        {"candidate_id": "candidate-b", "canonical_smiles": "CCC"},
        {"candidate_id": "candidate-c", "canonical_smiles": "CCO"},
        {"candidate_id": "candidate-d", "canonical_smiles": "CCN"},
    ]

    result = await _validation_agent(
        module,
        oracles={"rdkit": rdkit, "admet": admet, "boltz2": boltz2},
    ).process(_payload(1, candidates=candidates, boltz2=True))

    assert result["validation_schema_version"] == "validation.batch.v1"
    assert result["outcome"] == "PASS"
    assert [record["candidate_id"] for record in result["records"]] == [
        "candidate-a",
        "candidate-b",
        "candidate-c",
        "candidate-d",
    ]
    assert all(record["schema_version"] == "validation.record.v1" for record in result["records"])
    assert all(record["outcome"] == "PASS" for record in result["records"])
    assert rdkit.calls[0]["molecules"] == ["CCO", "CCC"]
    assert rdkit.calls[1]["molecules"] == ["CCN"]
    assert [call["molecules"] for call in admet.calls] == [["CCO", "CCC"], ["CCN"]]
    assert [call["molecules"] for call in boltz2.calls] == [["CCO", "CCC"], ["CCN"]]
    assert [call["request_context"]["request_id"] for call in admet.calls] == [
        "request-1:L1:admet:0",
        "request-1:L1:admet:1",
    ]
    assert [call["request_context"]["request_id"] for call in boltz2.calls] == [
        "request-1:L1:boltz2:0",
        "request-1:L1:boltz2:1",
    ]
    assert tracker["maximum"] == 2
    assert result["records"][0]["metrics"] == result["records"][2]["metrics"]
    assert result["records"][0]["levels"] == result["records"][2]["levels"]
    assert result["records"][0]["evidence"]
    assert result["records"][2]["evidence"]
    assert all(call["request_context"]["protein_pdb_id"] == "8ABC" for call in boltz2.calls)


@pytest.mark.asyncio
async def test_agent_request_reply_preserves_envelope_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mf_agents.messaging.redis_bus import InMemoryBus
    from mf_agents.messaging.request_client import AgentRequestClient

    monkeypatch.setenv("AGENT_MESSAGE_HMAC_SECRET", "validation-agent-test-secret")
    module = _load_validation_module()
    bus = InMemoryBus()
    await bus.connect()
    agent = _validation_agent(
        module,
        message_bus=bus,
        oracles={"rdkit": _BatchOracle({"admet_score": 0.9})},
    )
    await agent.start()
    try:
        result = await AgentRequestClient(bus).request(
            "agent.validation.request",
            {
                **_payload(0),
                "trace_id": "trace-1",
                "parent_id": "parent-1",
                "schema_version": "validation.request.v1",
            },
            payload_type_url="type.moleculeforge.ai/agent/validation/request.v1",
            timeout=1.0,
        )
    finally:
        await agent.stop()
        await bus.close()

    assert result["schema_version"] == "validation.request.v1"
    assert result["validation_schema_version"] == "validation.batch.v1"
    assert result["records"][0]["schema_version"] == "validation.record.v1"


@pytest.mark.asyncio
async def test_lower_level_failure_prevents_higher_oracle_for_that_smiles() -> None:
    module = _load_validation_module()
    rdkit = _BatchOracle(
        values_by_smiles={
            "CCO": {"admet_score": 0.2},
            "CCC": {"admet_score": 0.9},
        }
    )
    admet = _BatchOracle({"clearance": 0.2})

    result = await _validation_agent(
        module,
        oracles={"rdkit": rdkit, "admet": admet},
    ).process(
        _payload(
            1,
            candidates=[
                {"candidate_id": "failed", "canonical_smiles": "CCO"},
                {"candidate_id": "passed", "canonical_smiles": "CCC"},
            ],
        )
    )

    assert [record["outcome"] for record in result["records"]] == ["FAIL", "PASS"]
    assert admet.calls[0]["molecules"] == ["CCC"]
    assert [level["level"] for level in result["records"][0]["levels"]] == [0]
    assert [level["level"] for level in result["records"][1]["levels"]] == [0, 1]


def test_wire_fail_with_passing_metrics_has_structured_failure_reason() -> None:
    module = _load_validation_module()

    result = module._normalize_oracle_item(
        0,
        "rdkit",
        [_threshold(0, "rdkit", "admet_score", "maximize", 0.5)],
        {
            "scores": {"admet_score": 0.9},
            "uncertainties": {},
            "outcome": "ORACLE_OUTCOME_FAIL",
        },
    )

    assert result["outcome"] == "FAIL"
    assert result["metrics"][0]["passed"] is True
    assert result["oracle_record"]["failure"] == {
        "code": "ORACLE_OUTCOME_FAIL",
        "message": "oracle reported a failing outcome",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("oracle", "agent_kwargs"),
    [
        (None, {}),
        (_BatchOracle({"skipped": True}), {}),
        (_BatchOracle({"admet_score": 0.9}, delay=0.03), {"oracle_timeout_seconds": 0.005}),
        (_BatchOracle({"admet_score": 0.9}, drop_last=True), {}),
        (_BatchOracle({"admet_score": 0.9}, reverse=True), {}),
        (_BatchOracle({}), {}),
    ],
)
async def test_required_missing_skipped_timeout_or_protocol_error_maps_to_error(
    oracle: object | None,
    agent_kwargs: dict,
) -> None:
    module = _load_validation_module()
    payload = _payload(
        0,
        candidates=[
            {"candidate_id": "candidate-a", "canonical_smiles": "CCO"},
            {"candidate_id": "candidate-b", "canonical_smiles": "CCC"},
        ],
    )
    oracles = {} if oracle is None else {"rdkit": oracle}

    result = await _validation_agent(module, oracles=oracles, **agent_kwargs).process(payload)

    assert result["outcome"] == "ERROR"
    assert [record["outcome"] for record in result["records"]] == ["ERROR", "ERROR"]
    for record in result["records"]:
        assert record["evidence"]
        assert record["levels"][0]["outcome"] == "ERROR"
        assert record["levels"][0]["oracles"][0]["error"]["message"]


@pytest.mark.asyncio
async def test_real_metric_below_threshold_maps_to_fail() -> None:
    module = _load_validation_module()

    result = await _validation_agent(
        module,
        oracles={"rdkit": _BatchOracle({"admet_score": 0.49})}
    ).process(_payload(0))

    record = result["records"][0]
    assert result["outcome"] == "FAIL"
    assert record["outcome"] == "FAIL"
    assert record["metrics"] == [
        {
            "level": 0,
            "oracle": "rdkit",
            "metric": "admet_score",
            "value": 0.49,
            "direction": "maximize",
            "threshold": 0.5,
            "passed": False,
        }
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [True, math.nan, math.inf, -math.inf])
async def test_non_finite_or_boolean_oracle_metric_maps_to_error(
    value: object,
) -> None:
    module = _load_validation_module()

    result = await _validation_agent(
        module,
        oracles={"rdkit": _BatchOracle({"admet_score": value})}
    ).process(_payload(0))

    assert result["records"][0]["outcome"] == "ERROR"
    assert result["records"][0]["levels"][0]["oracles"][0]["error"]["code"] == "INVALID_METRIC"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "uncertainty",
    [None, True, math.nan, math.inf, -math.inf],
)
async def test_missing_invalid_or_boolean_required_uncertainty_maps_to_error(
    uncertainty: object,
) -> None:
    module = _load_validation_module()
    uncertainty_values = {} if uncertainty is None else {"affinity": uncertainty}
    oracles = {
        "rdkit": _BatchOracle({"admet_score": 0.9}),
        "admet": _BatchOracle({"clearance": 0.2}),
        "boltz2": _UncertaintyOracle(
            {"affinity": -8.0},
            uncertainty_values,
        ),
    }

    result = await _validation_agent(module, oracles=oracles).process(
        _payload(1, boltz2=True)
    )

    assert result["records"][0]["outcome"] == "ERROR"
    assert result["records"][0]["levels"][1]["oracles"][1]["error"]["code"] in {
        "MISSING_UNCERTAINTY",
        "INVALID_UNCERTAINTY",
    }


@pytest.mark.asyncio
async def test_invalid_smiles_fails_only_that_candidate_in_a_shared_l0_batch() -> None:
    module = _load_validation_module()

    result = await _validation_agent(module).process(
        _payload(
            0,
            candidates=[
                {
                    "candidate_id": "invalid",
                    "canonical_smiles": "not_valid!!!",
                },
                {"candidate_id": "valid", "canonical_smiles": "CCO"},
            ],
        )
    )

    assert result["outcome"] == "PASS"
    assert [record["outcome"] for record in result["records"]] == ["FAIL", "PASS"]
    assert result["records"][0]["levels"][0]["oracles"][0]["failure"] == {
        "code": "INVALID_SMILES",
        "message": "invalid SMILES: not_valid!!!",
    }
    assert result["records"][1]["metrics"][0]["metric"] == "admet_score"


@pytest.mark.asyncio
async def test_default_rdkit_batch_reports_zero_uncertainty() -> None:
    module = _load_validation_module()
    payload = _payload(0)
    payload["validation_policy"]["thresholds"][0]["max_uncertainty"] = 0.0

    result = await _validation_agent(module).process(payload)

    assert result["outcome"] == "PASS"
    metric = result["records"][0]["metrics"][0]
    assert metric["uncertainty"] == 0.0
    assert metric["max_uncertainty"] == 0.0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("level", "oracle_name", "missing_field"),
    [
        (1, "boltz2", "protein_pdb_id"),
        (2, "dock", "receptor_uri"),
        (2, "dock", "oracle_parameters"),
        (2, "dock", "oracle_parameters.engine"),
        (3, "fep", "protein_pdb_id"),
        (3, "fep", "reference_ligand_smiles"),
        (3, "fep", "oracle_parameters"),
        (3, "fep", "oracle_parameters.method"),
        (3, "fep", "oracle_parameters.n_repeats"),
    ],
)
async def test_missing_oracle_input_maps_to_error_before_service_call(
    level: int,
    oracle_name: str,
    missing_field: str,
) -> None:
    module = _load_validation_module()
    payload = _payload(level, boltz2=oracle_name == "boltz2")
    if "." in missing_field:
        parent, child = missing_field.split(".", 1)
        payload["validation_policy"]["oracle_inputs"][oracle_name][parent].pop(child)
    else:
        payload["validation_policy"]["oracle_inputs"][oracle_name].pop(missing_field)
    oracles = {
        "rdkit": _BatchOracle({"admet_score": 0.9}),
        "admet": _BatchOracle({"clearance": 0.2}),
        "boltz2": _UncertaintyOracle({"affinity": -8.0}, {"affinity": 0.2}),
        "dock": _BatchOracle({"docking_score": -7.0}),
        "fep": _BatchOracle({"rbfe": 0.1}),
    }

    result = await _validation_agent(module, oracles=oracles).process(payload)

    assert result["records"][0]["outcome"] == "ERROR"
    assert oracles[oracle_name].calls == []
    assert result["records"][0]["levels"][-1]["level"] == level
    assert result["records"][0]["levels"][-1]["outcome"] == "ERROR"


@pytest.mark.asyncio
async def test_l4_without_external_evidence_awaits_and_never_fabricates_metrics() -> None:
    module = _load_validation_module()
    oracles = {
        "rdkit": _BatchOracle({"admet_score": 0.9}),
        "admet": _BatchOracle({"clearance": 0.2}),
        "dock": _BatchOracle({"docking_score": -7.0}),
        "fep": _BatchOracle({"rbfe": 0.1}),
    }

    result = await _validation_agent(module, oracles=oracles).process(_payload(4))

    record = result["records"][0]
    assert result["outcome"] == "AWAITING_EVIDENCE"
    assert record["outcome"] == "AWAITING_EVIDENCE"
    assert all(metric["level"] < 4 for metric in record["metrics"])
    assert record["levels"][-1] == {
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


@pytest.mark.asyncio
async def test_l4_external_evidence_is_thresholded_and_preserves_reference() -> None:
    module = _load_validation_module()
    oracles = {
        "rdkit": _BatchOracle({"admet_score": 0.9}),
        "admet": _BatchOracle({"clearance": 0.2}),
        "dock": _BatchOracle({"docking_score": -7.0}),
        "fep": _BatchOracle({"rbfe": 0.1}),
    }
    evidence = [
        {
            "candidate_id": "candidate-1",
            "metrics": {"activity": 0.8},
            "uncertainties": {"activity": 0.03},
            "evidence_ids": ["artifact:measurement-1"],
        }
    ]

    result = await _validation_agent(module, oracles=oracles).process(
        _payload(4, external_evidence=evidence)
    )

    record = result["records"][0]
    assert record["outcome"] == "PASS"
    assert record["metrics"][-1]["metric"] == "activity"
    assert record["metrics"][-1]["value"] == pytest.approx(0.8)
    assert any(item["evidence_id"] == "artifact:measurement-1" for item in record["evidence"])


@pytest.mark.asyncio
async def test_l4_evidence_resume_reuses_lower_levels_without_oracle_calls() -> None:
    module = _load_validation_module()
    oracles = {
        "rdkit": _BatchOracle({"admet_score": 0.9}),
        "admet": _BatchOracle({"clearance": 0.2}),
        "dock": _BatchOracle({"docking_score": -7.0}),
        "fep": _BatchOracle({"rbfe": 0.1}),
    }
    agent = _validation_agent(module, oracles=oracles)
    initial = await agent.process(_payload(4))
    calls_before_resume = {
        oracle_name: len(oracle.calls) for oracle_name, oracle in oracles.items()
    }
    payload = _payload(
        4,
        external_evidence=[
            {
                "candidate_id": "candidate-1",
                "canonical_smiles": "CCO",
                "metrics": {"activity": 0.8},
                "uncertainties": {},
                "evidence_ids": ["artifact:measurement-1"],
            }
        ],
    )
    payload.update(
        {
            "resume_external_evidence": True,
            "prior_validation_records": initial["records"],
        }
    )

    resumed = await agent.process(payload)

    assert resumed["outcome"] == "PASS"
    assert [level["level"] for level in resumed["records"][0]["levels"]] == [
        0,
        1,
        2,
        3,
        4,
    ]
    assert {
        oracle_name: len(oracle.calls) for oracle_name, oracle in oracles.items()
    } == calls_before_resume


@pytest.mark.asyncio
async def test_batch_outcome_priority_is_error_then_pass_then_awaiting_then_fail() -> None:
    module = _load_validation_module()

    assert module._aggregate_batch_outcome(["PASS", "FAIL"]) == "PASS"
    assert module._aggregate_batch_outcome(["AWAITING_EVIDENCE", "FAIL"]) == ("AWAITING_EVIDENCE")
    assert module._aggregate_batch_outcome(["FAIL", "FAIL"]) == "FAIL"
    assert (
        module._aggregate_batch_outcome(["PASS", "ERROR", "AWAITING_EVIDENCE", "FAIL"]) == "ERROR"
    )


@pytest.mark.parametrize(
    "aggregate_name",
    ["_aggregate_level_outcome", "_aggregate_batch_outcome"],
)
@pytest.mark.parametrize("outcome", ["pass", "Pass", 1, None])
def test_validation_outcomes_require_exact_uppercase_four_state_values(
    aggregate_name: str,
    outcome: object,
) -> None:
    module = _load_validation_module()

    with pytest.raises(ValueError, match="outcomes must be non-empty validation outcomes"):
        getattr(module, aggregate_name)([outcome])


@pytest.mark.asyncio
async def test_crg_is_write_only_evidence_and_every_record_gets_an_evidence_id() -> None:
    module = _load_validation_module()

    class Repository:
        def __init__(self) -> None:
            self.writes: list[dict] = []
            self.reads = 0

        async def get_run_crg(self, run_id: str) -> dict:
            self.reads += 1
            raise AssertionError("validation must not read decision cache")

        async def write_workflow_belief(self, **kwargs: object) -> None:
            self.writes.append(kwargs)

    repository = Repository()
    result = await module.ValidationAgent(
        oracles={"rdkit": _BatchOracle({"admet_score": 0.9})},
        crg_repository=repository,
    ).process(_payload(0))

    assert repository.reads == 0
    assert len(repository.writes) == 1
    assert repository.writes[0]["predicate"] == "validation_record"
    assert repository.writes[0]["subject"] == "candidate-1"
    assert result["records"][0]["evidence"]
    assert result["records"][0]["evidence"][-1]["oracle"] == "validation_agent"
    assert repository.writes[0]["evidence_ids"] == []


@pytest.mark.asyncio
async def test_validation_fails_closed_without_a_crg_repository() -> None:
    module = _load_validation_module()
    agent = _validation_agent(
        module,
        crg_repository=None,
        oracles={"rdkit": _BatchOracle({"admet_score": 0.9})},
    )

    with pytest.raises(RuntimeError, match="CRG repository is required"):
        await agent.process(_payload(0))


@pytest.mark.asyncio
async def test_oracle_grpc_client_sends_literal_context_and_preserves_wire_evidence() -> None:
    module = _load_validation_module()
    captured = []

    class Stub:
        async def Evaluate(  # noqa: N802
            self,
            request: oracle_pb2.OracleBatchRequest,
            timeout: float | None = None,
        ) -> oracle_pb2.OracleBatchResponse:
            captured.append((request, timeout))
            return oracle_pb2.OracleBatchResponse(
                batch_id=request.request_id,
                evaluations=[
                    oracle_pb2.OracleEvaluation(
                        oracle_name="gnina",
                        molecule_smiles="CCO",
                        level=oracle_pb2.L2_DOCKING,
                        scores={"docking_score": -7.2},
                        success=True,
                        outcome=oracle_pb2.ORACLE_OUTCOME_PASS,
                        artifact_refs=[_artifact_ref()],
                        evidence_id="request-1:gnina:0",
                        metrics=[
                            oracle_pb2.OracleMetric(
                                property="docking_score",
                                value=-7.2,
                            )
                        ],
                    )
                ],
            )

    client = module.OracleGrpcClient(
        "unused:50054",
        level=2,
        oracle_name="dock",
    )
    client.stub = Stub()

    result = await client.evaluate(
        ["CCO"],
        ["docking_score"],
        request_context={
            "project_id": "project-1",
            "request_id": "request-1",
            "receptor_uri": "file:///receptor.pdb",
            "oracle_parameters": {"engine": "gnina"},
        },
    )

    request = captured[0][0]
    assert request.project_id == "project-1"
    assert request.request_id == "request-1"
    assert request.level == oracle_pb2.L2_DOCKING
    assert request.receptor_uri == "file:///receptor.pdb"
    assert dict(request.oracle_parameters) == {"engine": "gnina"}
    assert result["CCO"]["scores"] == {"docking_score": pytest.approx(-7.2)}
    assert result["CCO"]["evidence_id"] == "request-1:gnina:0"


@pytest.mark.asyncio
async def test_oracle_grpc_client_rejects_missing_typed_metric_rows() -> None:
    module = _load_validation_module()

    class Stub:
        async def Evaluate(  # noqa: N802
            self,
            request: oracle_pb2.OracleBatchRequest,
            timeout: float | None = None,
        ) -> oracle_pb2.OracleBatchResponse:
            return oracle_pb2.OracleBatchResponse(
                batch_id=request.request_id,
                evaluations=[
                    oracle_pb2.OracleEvaluation(
                        oracle_name="rdkit",
                        molecule_smiles="CCO",
                        level=oracle_pb2.L0_RDKIT,
                        scores={"admet_score": 0.9},
                        success=True,
                        outcome=oracle_pb2.ORACLE_OUTCOME_PASS,
                        artifact_refs=[_artifact_ref()],
                        evidence_id="request-1:rdkit:0",
                    )
                ],
            )

    client = module.OracleGrpcClient("unused:50051", 0, "rdkit")
    client.stub = Stub()

    with pytest.raises(RuntimeError, match="metric order"):
        await client.evaluate(
            ["CCO"],
            ["admet_score"],
            request_context={
                "project_id": "project-1",
                "request_id": "request-1",
            },
        )


@pytest.mark.asyncio
async def test_oracle_grpc_client_rejects_wrong_logical_oracle_identity() -> None:
    module = _load_validation_module()

    class Stub:
        async def Evaluate(  # noqa: N802
            self,
            request: oracle_pb2.OracleBatchRequest,
            timeout: float | None = None,
        ) -> oracle_pb2.OracleBatchResponse:
            return oracle_pb2.OracleBatchResponse(
                batch_id=request.request_id,
                evaluations=[
                    oracle_pb2.OracleEvaluation(
                        oracle_name="admet_ai",
                        molecule_smiles="CCO",
                        level=oracle_pb2.L1_ML_SURROGATE,
                        scores={"affinity": -8.0},
                        success=True,
                        outcome=oracle_pb2.ORACLE_OUTCOME_PASS,
                        artifact_refs=[_artifact_ref()],
                        evidence_id="wrong:evidence:1",
                        metrics=[
                            oracle_pb2.OracleMetric(
                                property="affinity",
                                value=-8.0,
                            )
                        ],
                    )
                ],
            )

    client = module.OracleGrpcClient("unused:50053", 1, "boltz2")
    client.stub = Stub()

    with pytest.raises(RuntimeError, match="logical oracle identity"):
        await client.evaluate(
            ["CCO"],
            ["affinity"],
            request_context={
                "project_id": "project-1",
                "request_id": "request-1",
                "protein_pdb_id": "8ABC",
            },
        )


@pytest.mark.asyncio
async def test_oracle_health_check_only_waits_for_channel_readiness() -> None:
    module = _load_validation_module()

    class Channel:
        def __init__(self) -> None:
            self.ready_calls = 0

        async def channel_ready(self) -> None:
            self.ready_calls += 1

    class Stub:
        async def Evaluate(self, *_args: object, **_kwargs: object) -> None:  # noqa: N802
            raise AssertionError("health checks must not execute Oracle Evaluate")

    channel = Channel()
    client = module.OracleGrpcClient("unused:50055", 3, "fep")
    client.channel = channel
    client.stub = Stub()

    assert await client.health_check() == {"healthy": True}
    assert channel.ready_calls == 1


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda evaluation: evaluation.scores.__setitem__(
                "extra_score",
                math.nan,
            ),
            "score extra_score must be finite",
        ),
        (
            lambda evaluation: evaluation.uncertainties.__setitem__(
                "extra_uncertainty",
                math.inf,
            ),
            "uncertainty extra_uncertainty must be finite",
        ),
        (
            lambda evaluation: evaluation.uncertainties.__setitem__(
                "extra_uncertainty",
                -0.1,
            ),
            "uncertainty extra_uncertainty must be non-negative",
        ),
        (
            lambda evaluation: setattr(
                evaluation.metrics[0],
                "value",
                math.inf,
            ),
            "typed metric admet_score must be finite",
        ),
        (
            lambda evaluation: (
                evaluation.uncertainties.__setitem__("admet_score", 0.2),
                setattr(
                    evaluation.metrics[0],
                    "uncertainty",
                    math.inf,
                ),
            ),
            "typed uncertainty admet_score must be finite",
        ),
        (
            lambda evaluation: (
                evaluation.uncertainties.__setitem__("admet_score", 0.2),
                setattr(
                    evaluation.metrics[0],
                    "uncertainty",
                    -0.1,
                ),
            ),
            "typed uncertainty admet_score must be non-negative",
        ),
    ],
)
def test_oracle_grpc_consumer_rejects_invalid_numeric_values(
    mutate: Callable[[oracle_pb2.OracleEvaluation], object],
    message: str,
) -> None:
    module = _load_validation_module()
    evaluation = _successful_wire_evaluation()
    mutate(evaluation)

    with pytest.raises(RuntimeError, match=message):
        _consume_wire_evaluations(module, [evaluation])


@pytest.mark.parametrize(
    "evidence_id",
    [
        "request-1:rdkit:1",
        "other-request:rdkit:0",
        "request-1:rdkit_oracle_l0:0",
    ],
)
def test_oracle_grpc_consumer_requires_exact_producer_evidence_id(
    evidence_id: str,
) -> None:
    module = _load_validation_module()
    evaluation = _successful_wire_evaluation(evidence_id=evidence_id)

    with pytest.raises(RuntimeError, match="evidence_id does not match request"):
        _consume_wire_evaluations(module, [evaluation])


@pytest.mark.parametrize(
    "mutate",
    [
        lambda evaluation: evaluation.scores.__setitem__("extra_score", 1.0),
        lambda evaluation: evaluation.uncertainties.__setitem__(
            "extra_uncertainty",
            0.1,
        ),
    ],
)
def test_oracle_grpc_consumer_requires_exact_pass_numeric_keys(
    mutate: Callable[[oracle_pb2.OracleEvaluation], object],
) -> None:
    module = _load_validation_module()
    evaluation = _successful_wire_evaluation()
    mutate(evaluation)

    with pytest.raises(
        RuntimeError,
        match="PASS scores and uncertainties do not match requested_properties",
    ):
        _consume_wire_evaluations(module, [evaluation])


@pytest.mark.parametrize(
    "mutate",
    [
        lambda evaluation: evaluation.scores.__setitem__("admet_score", 0.1),
        lambda evaluation: evaluation.uncertainties.__setitem__(
            "admet_score",
            0.1,
        ),
        lambda evaluation: evaluation.metrics.append(
            oracle_pb2.OracleMetric(property="admet_score", value=0.1),
        ),
    ],
)
def test_oracle_grpc_consumer_requires_empty_error_numeric_payloads(
    mutate: Callable[[oracle_pb2.OracleEvaluation], object],
) -> None:
    module = _load_validation_module()
    evaluation = oracle_pb2.OracleEvaluation(
        oracle_name="rdkit",
        molecule_smiles="CCO",
        level=oracle_pb2.L0_RDKIT,
        success=False,
        outcome=oracle_pb2.ORACLE_OUTCOME_ERROR,
        error_code="COMPUTATION_ERROR",
        error_message="failed",
        evidence_id="request-1:rdkit:0",
    )
    mutate(evaluation)

    with pytest.raises(
        RuntimeError,
        match="ERROR scores, uncertainties, and metrics must be empty",
    ):
        _consume_wire_evaluations(module, [evaluation])


def test_oracle_grpc_consumer_accepts_present_empty_error_message() -> None:
    module = _load_validation_module()
    evaluation = oracle_pb2.OracleEvaluation(
        oracle_name="rdkit",
        molecule_smiles="CCO",
        level=oracle_pb2.L0_RDKIT,
        success=False,
        outcome=oracle_pb2.ORACLE_OUTCOME_ERROR,
        error_code="COMPUTATION_ERROR",
        error_message="",
        evidence_id="request-1:rdkit:0",
    )
    assert evaluation.HasField("error_message") is True

    result = _consume_wire_evaluations(module, [evaluation])

    assert result["CCO"]["error_message"] == ""


def test_oracle_grpc_consumer_rejects_present_pass_error_message() -> None:
    module = _load_validation_module()
    evaluation = _successful_wire_evaluation()
    evaluation.error_message = ""

    assert evaluation.HasField("error_message") is True
    with pytest.raises(
        RuntimeError,
        match="success/outcome/error fields are contradictory",
    ):
        _consume_wire_evaluations(module, [evaluation])


def test_oracle_grpc_consumer_rejects_absent_error_message() -> None:
    module = _load_validation_module()
    evaluation = oracle_pb2.OracleEvaluation(
        oracle_name="rdkit",
        molecule_smiles="CCO",
        level=oracle_pb2.L0_RDKIT,
        success=False,
        outcome=oracle_pb2.ORACLE_OUTCOME_ERROR,
        error_code="COMPUTATION_ERROR",
        evidence_id="request-1:rdkit:0",
    )
    assert evaluation.HasField("error_message") is False

    with pytest.raises(
        RuntimeError,
        match="success/outcome/error fields are contradictory",
    ):
        _consume_wire_evaluations(module, [evaluation])


def test_oracle_grpc_consumer_requires_identical_batch_artifact_refs() -> None:
    module = _load_validation_module()
    first = _successful_wire_evaluation(
        molecule_smiles="CCO",
        evidence_id="request-1:rdkit:0",
    )
    second = _successful_wire_evaluation(
        molecule_smiles="CCN",
        evidence_id="request-1:rdkit:1",
    )
    second.artifact_refs[0].checksum = f"sha256:{'b' * 64}"

    with pytest.raises(RuntimeError, match="artifact_refs must match within batch"):
        _consume_wire_evaluations(
            module,
            [first, second],
            molecules=["CCO", "CCN"],
        )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda evaluation: evaluation.ClearField("artifact_refs"),
            "successful oracle response requires artifact_refs",
        ),
        (
            lambda evaluation: setattr(
                evaluation.artifact_refs[0],
                "name",
                "",
            ),
            "artifact name is required",
        ),
        (
            lambda evaluation: evaluation.artifact_refs.append(
                _artifact_ref(),
            ),
            "artifact names must be unique",
        ),
        (
            lambda evaluation: setattr(
                evaluation.artifact_refs[0],
                "required",
                False,
            ),
            "artifact must be required",
        ),
    ],
)
def test_oracle_grpc_consumer_rejects_missing_or_ambiguous_artifacts(
    mutate: Callable[[oracle_pb2.OracleEvaluation], object],
    message: str,
) -> None:
    module = _load_validation_module()
    evaluation = _successful_wire_evaluation()
    mutate(evaluation)

    with pytest.raises(RuntimeError, match=message):
        _consume_wire_evaluations(module, [evaluation])


@pytest.mark.parametrize(
    "checksum",
    [
        "",
        "sha1:" + ("a" * 40),
        "sha256:abcd",
        "sha256:" + ("z" * 64),
        "sha256:" + ("A" * 64),
        f" {VALID_ARTIFACT_CHECKSUM}",
    ],
)
def test_oracle_grpc_consumer_rejects_invalid_required_artifact_checksum(
    checksum: str,
) -> None:
    module = _load_validation_module()
    evaluation = _successful_wire_evaluation()
    evaluation.artifact_refs[0].checksum = checksum

    with pytest.raises(
        RuntimeError,
        match="required artifact checksum must be sha256",
    ):
        _consume_wire_evaluations(module, [evaluation])


@pytest.mark.parametrize(
    (
        "logical_oracle",
        "level",
        "proto_level",
        "reported_oracle",
        "property_name",
        "score",
        "request_context",
    ),
    [
        (
            "dock",
            2,
            oracle_pb2.L2_DOCKING,
            "gnina",
            "docking_score",
            -7.2,
            {
                "receptor_uri": "file:///receptor.pdb",
                "oracle_parameters": {"engine": "gnina"},
            },
        ),
        (
            "fep",
            3,
            oracle_pb2.L3_FEP,
            "openfe",
            "rbfe",
            -1.2,
            {
                "protein_pdb_id": "8ABC",
                "reference_ligand_smiles": "CC",
                "oracle_parameters": {
                    "method": "openfe",
                    "n_repeats": "3",
                },
            },
        ),
    ],
)
def test_dock_and_fep_success_responses_require_artifacts(
    logical_oracle: str,
    level: int,
    proto_level: int,
    reported_oracle: str,
    property_name: str,
    score: float,
    request_context: dict,
) -> None:
    module = _load_validation_module()
    evaluation = oracle_pb2.OracleEvaluation(
        oracle_name=reported_oracle,
        molecule_smiles="CCO",
        level=proto_level,
        scores={property_name: score},
        success=True,
        outcome=oracle_pb2.ORACLE_OUTCOME_PASS,
        evidence_id=f"request-1:{reported_oracle}:0",
        metrics=[
            oracle_pb2.OracleMetric(
                property=property_name,
                value=score,
            )
        ],
    )

    with pytest.raises(
        RuntimeError,
        match="successful oracle response requires artifact_refs",
    ):
        _consume_wire_evaluations(
            module,
            [evaluation],
            properties=[property_name],
            level=level,
            oracle_name=logical_oracle,
            request_context=request_context,
        )


def test_admet_artifacts_remain_strict_when_present() -> None:
    module = _load_validation_module()
    evaluation = oracle_pb2.OracleEvaluation(
        oracle_name="admet_ai",
        molecule_smiles="CCO",
        level=oracle_pb2.L1_ML_SURROGATE,
        scores={"clearance": 1.5},
        success=True,
        outcome=oracle_pb2.ORACLE_OUTCOME_PASS,
        artifact_refs=[
            _artifact_ref(
                name="admet-model",
                required=False,
            )
        ],
        evidence_id="request-1:admet_ai:0",
        metrics=[
            oracle_pb2.OracleMetric(
                property="clearance",
                value=1.5,
            )
        ],
    )

    with pytest.raises(RuntimeError, match="artifact must be required"):
        _consume_wire_evaluations(
            module,
            [evaluation],
            properties=["clearance"],
            level=1,
            oracle_name="admet",
        )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda evaluation: setattr(
            evaluation,
            "outcome",
            oracle_pb2.ORACLE_OUTCOME_ERROR,
        ),
        lambda evaluation: setattr(evaluation, "error_code", "FAILED"),
        lambda evaluation: setattr(evaluation, "error_message", "failed"),
        lambda evaluation: setattr(evaluation, "error_message", " "),
        lambda evaluation: (
            setattr(evaluation, "success", False),
            setattr(evaluation, "error_code", "FAILED"),
            setattr(evaluation, "error_message", "failed"),
        ),
        lambda evaluation: (
            setattr(evaluation, "success", False),
            setattr(evaluation, "outcome", oracle_pb2.ORACLE_OUTCOME_ERROR),
        ),
        lambda evaluation: (
            setattr(evaluation, "success", False),
            setattr(evaluation, "outcome", oracle_pb2.ORACLE_OUTCOME_ERROR),
            setattr(evaluation, "error_code", "FAILED"),
        ),
        lambda evaluation: (
            setattr(evaluation, "success", False),
            setattr(evaluation, "outcome", oracle_pb2.ORACLE_OUTCOME_ERROR),
            setattr(evaluation, "error_message", "failed"),
        ),
    ],
)
def test_oracle_grpc_consumer_rejects_contradictory_status_fields(
    mutate: Callable[[oracle_pb2.OracleEvaluation], object],
) -> None:
    module = _load_validation_module()
    evaluation = _successful_wire_evaluation()
    mutate(evaluation)

    with pytest.raises(
        RuntimeError,
        match="success/outcome/error fields are contradictory",
    ):
        _consume_wire_evaluations(module, [evaluation])


@pytest.mark.parametrize(
    (
        "logical_oracle",
        "level",
        "proto_level",
        "reported_oracle",
        "property_name",
        "score",
        "request_fields",
    ),
    [
        (
            "admet",
            1,
            oracle_pb2.L1_ML_SURROGATE,
            "admet_ai",
            "clearance",
            1.5,
            {},
        ),
        (
            "boltz2",
            1,
            oracle_pb2.L1_ML_SURROGATE,
            "boltz2",
            "affinity",
            -8.0,
            {
                "protein_pdb_id": "8ABC",
                "oracle_parameters": {"ensemble_size": "2"},
            },
        ),
        (
            "dock",
            2,
            oracle_pb2.L2_DOCKING,
            "gnina",
            "docking_score",
            -7.2,
            {
                "receptor_uri": "file:///receptor.pdb",
                "oracle_parameters": {"engine": "gnina"},
            },
        ),
        (
            "fep",
            3,
            oracle_pb2.L3_FEP,
            "openfe",
            "rbfe",
            -1.2,
            {
                "protein_pdb_id": "8ABC",
                "reference_ligand_smiles": "CC",
                "oracle_parameters": {
                    "method": "openfe",
                    "n_repeats": "3",
                },
            },
        ),
    ],
)
def test_real_producer_responses_are_accepted_by_validation_consumer(
    logical_oracle: str,
    level: int,
    proto_level: int,
    reported_oracle: str,
    property_name: str,
    score: float,
    request_fields: dict,
) -> None:
    from mf_core.plugins.oracle import (
        build_oracle_evaluation,
        build_oracle_response,
        validate_oracle_request,
    )

    module = _load_validation_module()
    request = oracle_pb2.OracleBatchRequest(
        project_id="project-1",
        request_id="request-1",
        molecule_smiles=["CCO"],
        requested_properties=[property_name],
        level=proto_level,
        **request_fields,
    )
    producer_context = validate_oracle_request(
        request,
        expected_level=proto_level,
        require_receptor_uri=logical_oracle == "dock",
        require_protein_pdb_id=logical_oracle in {"boltz2", "fep"},
        require_reference_ligand=logical_oracle == "fep",
        required_parameters=(("method", "n_repeats") if logical_oracle == "fep" else ()),
        allowed_parameters=tuple(request_fields.get("oracle_parameters", {})),
    )
    response = build_oracle_response(
        request=producer_context,
        evaluations=[
            build_oracle_evaluation(
                request=producer_context,
                index=0,
                oracle_name=reported_oracle,
                scores={property_name: score},
                uncertainties={},
                elapsed_ms=1,
                artifacts=[_artifact_ref()],
            )
        ],
        total_elapsed_ms=1,
    )

    result = module._evaluations_by_smiles(
        response,
        ["CCO"],
        [property_name],
        expected_level=level,
        expected_oracle_name=logical_oracle,
        request_context=request_fields,
        request_id="request-1",
    )

    assert result["CCO"]["success"] is True
    assert result["CCO"]["scores"] == {property_name: pytest.approx(score)}
    assert result["CCO"]["artifact_refs"] == [
        {
            "name": "oracle-model",
            "version": "1",
            "checksum": VALID_ARTIFACT_CHECKSUM,
            "required": True,
        }
    ]


def test_real_producer_error_response_is_accepted_by_validation_consumer() -> None:
    from mf_core.plugins.oracle import (
        build_oracle_error_evaluation,
        build_oracle_response,
        validate_oracle_request,
    )

    module = _load_validation_module()
    request = oracle_pb2.OracleBatchRequest(
        project_id="project-1",
        request_id="request-1",
        molecule_smiles=["CCO"],
        requested_properties=["clearance"],
        level=oracle_pb2.L1_ML_SURROGATE,
    )
    producer_context = validate_oracle_request(
        request,
        expected_level=oracle_pb2.L1_ML_SURROGATE,
    )
    response = build_oracle_response(
        request=producer_context,
        evaluations=[
            build_oracle_error_evaluation(
                request=producer_context,
                index=0,
                oracle_name="admet_ai",
                elapsed_ms=1,
                artifacts=[_artifact_ref()],
                error_code="COMPUTATION_ERROR",
                error_message="model execution failed",
            )
        ],
        total_elapsed_ms=1,
    )

    result = module._evaluations_by_smiles(
        response,
        ["CCO"],
        ["clearance"],
        expected_level=1,
        expected_oracle_name="admet",
        request_context={},
        request_id="request-1",
    )

    assert result["CCO"]["success"] is False
    assert result["CCO"]["outcome"] == "ORACLE_OUTCOME_ERROR"
    assert result["CCO"]["error_code"] == "COMPUTATION_ERROR"
    assert result["CCO"]["error_message"] == "model execution failed"


def test_admet_response_without_artifacts_is_accepted() -> None:
    evaluation = oracle_pb2.OracleEvaluation(
        oracle_name="admet_ai",
        molecule_smiles="CCO",
        level=oracle_pb2.L1_ML_SURROGATE,
        scores={"clearance": 1.5},
        success=True,
        outcome=oracle_pb2.ORACLE_OUTCOME_PASS,
        evidence_id="request-1:admet_ai:0",
        metrics=[
            oracle_pb2.OracleMetric(
                property="clearance",
                value=1.5,
            )
        ],
    )

    result = _consume_wire_evaluations(
        _load_validation_module(),
        [evaluation],
        properties=["clearance"],
        level=1,
        oracle_name="admet",
    )

    assert result["CCO"]["scores"] == {"clearance": pytest.approx(1.5)}
    assert result["CCO"]["artifact_refs"] == []


@pytest.mark.asyncio
async def test_real_dock_grpc_response_is_accepted_by_validation_consumer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    producer = _load_source_module(
        "validation_consumer_real_dock_producer_test",
        ROOT / "services/dock-svc/src/dock_svc/main.py",
    )
    runner = tmp_path / "dock_runner.py"
    runner.write_text(
        "import json, sys\n"
        "request = json.load(sys.stdin)\n"
        "print(json.dumps({"
        "'engine': request['engine'], "
        "'smiles': request['smiles'], "
        "'receptor_uri': request['protein_pdb'], "
        "'scores': {'docking_score': -7.5}, "
        "'elapsed_ms': 10"
        "}))\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("DOCK_ORACLE_COMMAND", f"{sys.executable} {runner}")
    monkeypatch.setenv("GNINA_BINARY", "/missing/unused-gnina")
    monkeypatch.setenv("DIFFDOCK_MODEL_PATH", "/missing/unused-diffdock")
    receptor = tmp_path / "receptor.pdb"
    receptor.write_text("HEADER TEST RECEPTOR\nEND\n", encoding="utf-8")
    request_context = {
        "receptor_uri": str(receptor),
        "oracle_parameters": {"engine": "gnina"},
    }
    response = await _oracle_grpc_call(
        producer,
        producer.DockOracleServicer(),
        oracle_pb2.OracleBatchRequest(
            project_id="project-1",
            request_id="request-1",
            molecule_smiles=["CCO"],
            requested_properties=["docking_score"],
            level=oracle_pb2.L2_DOCKING,
            **request_context,
        ),
    )

    result = _load_validation_module()._evaluations_by_smiles(
        response,
        ["CCO"],
        ["docking_score"],
        expected_level=2,
        expected_oracle_name="dock",
        request_context=request_context,
        request_id="request-1",
    )

    assert result["CCO"]["scores"] == {"docking_score": pytest.approx(-7.5)}
    assert result["CCO"]["artifact_refs"][0]["name"] == "dock_oracle_command"
    assert result["CCO"]["artifact_refs"][0]["required"] is True
    assert len(result["CCO"]["artifact_refs"][0]["checksum"]) == len(VALID_ARTIFACT_CHECKSUM)


@pytest.mark.asyncio
async def test_oracle_grpc_client_rejects_response_count_and_order() -> None:
    module = _load_validation_module()

    class Stub:
        async def Evaluate(  # noqa: N802
            self,
            request: oracle_pb2.OracleBatchRequest,
            timeout: float | None = None,
        ) -> oracle_pb2.OracleBatchResponse:
            return oracle_pb2.OracleBatchResponse(
                batch_id=request.request_id,
                evaluations=[
                    oracle_pb2.OracleEvaluation(
                        oracle_name="rdkit",
                        molecule_smiles="CCC",
                        level=oracle_pb2.L0_RDKIT,
                        scores={"admet_score": 0.8},
                        success=True,
                        outcome=oracle_pb2.ORACLE_OUTCOME_PASS,
                        artifact_refs=[_artifact_ref()],
                        evidence_id="rdkit:evidence:1",
                    ),
                    oracle_pb2.OracleEvaluation(
                        oracle_name="rdkit",
                        molecule_smiles="CCO",
                        level=oracle_pb2.L0_RDKIT,
                        scores={"admet_score": 0.9},
                        success=True,
                        outcome=oracle_pb2.ORACLE_OUTCOME_PASS,
                        artifact_refs=[_artifact_ref()],
                        evidence_id="rdkit:evidence:2",
                    ),
                ],
            )

    client = module.OracleGrpcClient("unused:50051", 0, "rdkit")
    client.stub = Stub()

    with pytest.raises(RuntimeError, match="order"):
        await client.evaluate(
            ["CCO", "CCC"],
            ["admet_score"],
            request_context={
                "project_id": "project-1",
                "request_id": "request-1",
            },
        )


def test_default_wiring_uses_fixed_level_targets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("L1_ADMET_ORACLE_TARGET", "admet:50056")
    monkeypatch.setenv("L1_BOLTZ2_ORACLE_TARGET", "boltz2:50053")
    monkeypatch.setenv("L2_DOCK_ORACLE_TARGET", "dock:50054")
    monkeypatch.setenv("L3_FEP_ORACLE_TARGET", "fep:50055")
    module = _load_validation_module()

    agent = module.ValidationAgent()

    assert agent.oracles["admet"].level == 1
    assert agent.oracles["boltz2"].level == 1
    assert agent.oracles["dock"].level == 2
    assert agent.oracles["fep"].level == 3
    assert agent.oracles["admet"].target == "admet:50056"
    assert agent.oracles["boltz2"].target == "boltz2:50053"
    assert agent.oracles["dock"].target == "dock:50054"
    assert agent.oracles["fep"].target == "fep:50055"
