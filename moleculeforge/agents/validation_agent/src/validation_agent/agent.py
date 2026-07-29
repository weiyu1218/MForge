"""Validation Agent - explicit, batched L0-L4 validation."""

from __future__ import annotations

import asyncio
import copy
import inspect
import json
import math
import os
import re
from collections.abc import Mapping
from numbers import Real
from typing import Any

from mf_agents.base.agent import (
    BaseAgent,
    agent_health_check_timeout_seconds,
    close_owned_channel,
    ensure_default_event_loop,
    run_health_probe_in_daemon,
)
from mf_agents.crg.graph import ChemicalReasoningGraph
from mf_core.db.repositories import build_shared_crg_repository_from_env
from mf_core.proto_gen.moleculeforge.v1.oracle import oracle_pb2, oracle_pb2_grpc
from mf_core.types.crg import Belief

_OUTCOME_ACCEPTED = "PASS"
_OUTCOME_FAIL = "FAIL"
_OUTCOME_AWAITING = "AWAITING_EVIDENCE"
_OUTCOME_ERROR = "ERROR"
_OUTCOMES = frozenset(
    {
        _OUTCOME_ACCEPTED,
        _OUTCOME_FAIL,
        _OUTCOME_AWAITING,
        _OUTCOME_ERROR,
    }
)
_ORACLE_LEVEL_BY_NAME = {
    "rdkit": 0,
    "admet": 1,
    "boltz2": 1,
    "dock": 2,
    "fep": 3,
    "external": 4,
}
_REQUIRED_ORACLE_BY_LEVEL = {
    0: "rdkit",
    1: "admet",
    2: "dock",
    3: "fep",
    4: "external",
}
_PROTO_LEVEL_BY_HTTP_LEVEL = {
    0: oracle_pb2.L0_RDKIT,
    1: oracle_pb2.L1_ML_SURROGATE,
    2: oracle_pb2.L2_DOCKING,
    3: oracle_pb2.L3_FEP,
    4: oracle_pb2.L4_WETLAB,
}
_RDKIT_METRICS = frozenset(
    {
        "qed",
        "sa_score",
        "logp",
        "lipinski_violations",
        "admet_score",
    }
)
_FIXED_ORACLE_METRICS = {
    "boltz2": frozenset({"affinity"}),
    "dock": frozenset({"docking_score"}),
    "fep": frozenset({"rbfe"}),
}
_ORACLE_INPUT_FIELDS = {
    "boltz2": frozenset({"protein_pdb_id", "oracle_parameters"}),
    "dock": frozenset({"receptor_uri", "oracle_parameters"}),
    "fep": frozenset(
        {
            "protein_pdb_id",
            "reference_ligand_smiles",
            "oracle_parameters",
        }
    ),
}
_ORACLE_PARAMETER_FIELDS = {
    "rdkit": frozenset(),
    "admet": frozenset(),
    "boltz2": frozenset({"ensemble_size"}),
    "dock": frozenset({"engine"}),
    "fep": frozenset({"method", "n_repeats"}),
    "external": frozenset(),
}
_HEALTH_METRIC_BY_ORACLE = {
    "rdkit": "admet_score",
    "admet": "clearance",
    "boltz2": "affinity",
    "dock": "docking_score",
    "fep": "rbfe",
}
_VALIDATION_POLICY_REQUIRED_FIELDS = (
    "oracle_level",
    "batch_size",
    "max_concurrency",
    "thresholds",
    "oracle_inputs",
)
_VALIDATION_POLICY_FIELDS = frozenset(_VALIDATION_POLICY_REQUIRED_FIELDS)
_THRESHOLD_FIELDS = frozenset(
    {
        "level",
        "oracle",
        "metric",
        "direction",
        "value",
        "max_uncertainty",
    }
)
_STATIC_REPORTED_ORACLE_NAMES = {
    "rdkit": frozenset({"rdkit", "rdkit_oracle_l0"}),
    "admet": frozenset({"admet", "admet_ai"}),
    "boltz2": frozenset({"boltz2"}),
    "external": frozenset({"external"}),
}
_SHA256_CHECKSUM = re.compile(r"sha256:[0-9a-f]{64}\Z")


class OracleGrpcClient:
    """Typed client for one fixed logical oracle and fidelity level."""

    def __init__(
        self,
        target: str,
        level: int,
        oracle_name: str,
        *,
        health_level: int | None = None,
    ) -> None:
        if not str(target).strip():
            raise ValueError("oracle target is required")
        if level not in _PROTO_LEVEL_BY_HTTP_LEVEL:
            raise ValueError("oracle level must be an integer between 0 and 4")
        self.target = str(target).strip()
        self.level = level
        self.health_level = level if health_level is None else health_level
        self.oracle_name = str(oracle_name).strip()
        self.channel = None
        self.stub = None
        self._closed = False

    async def evaluate(
        self,
        molecules: list[str],
        properties: list[str],
        *,
        request_context: dict | None = None,
    ) -> dict[str, dict]:
        context = dict(request_context or {})
        response = await self._stub().Evaluate(
            _oracle_batch_request(
                molecules,
                properties,
                self.level,
                context,
            )
        )
        return _evaluations_by_smiles(
            response,
            molecules,
            properties,
            expected_level=self.level,
            expected_oracle_name=self.oracle_name,
            request_context=context,
            request_id=str(context.get("request_id") or ""),
        )

    async def predict_with_uncertainty(
        self,
        molecules: list[str],
        properties: list[str],
        *,
        request_context: dict | None = None,
    ) -> dict[str, dict]:
        context = dict(request_context or {})
        response = await self._stub().PredictWithUncertainty(
            _oracle_batch_request(
                molecules,
                properties,
                self.level,
                context,
                return_uncertainty=True,
            )
        )
        return _evaluations_by_smiles(
            response,
            molecules,
            properties,
            expected_level=self.level,
            expected_oracle_name=self.oracle_name,
            request_context=context,
            request_id=str(context.get("request_id") or ""),
        )

    async def health_check(self) -> dict[str, bool]:
        context = _health_request_context(self.oracle_name)
        if context is None:
            return {"healthy": False}
        property_name = _HEALTH_METRIC_BY_ORACLE.get(self.oracle_name)
        if not property_name:
            return {"healthy": False}
        try:
            response = await self._stub().Evaluate(
                _oracle_batch_request(
                    ["C"],
                    [property_name],
                    self.level,
                    context,
                ),
                timeout=agent_health_check_timeout_seconds(),
            )
            result = _evaluations_by_smiles(
                response,
                ["C"],
                [property_name],
                expected_level=self.level,
                expected_oracle_name=self.oracle_name,
                request_context=context,
                request_id=context["request_id"],
            )
        except Exception:
            return {"healthy": False}
        item = result.get("C")
        return {
            "healthy": bool(
                isinstance(item, Mapping)
                and item.get("success") is True
                and _is_finite_number(_mapping_or_empty(item.get("scores")).get(property_name))
            )
        }

    def _stub(self) -> object:
        if self.stub is None:
            import grpc

            ensure_default_event_loop()
            self.channel = grpc.aio.insecure_channel(self.target)
            self.stub = oracle_pb2_grpc.OracleServiceStub(self.channel)
        return self.stub

    async def close(self) -> None:
        await close_owned_channel(self, self.channel)


class ValidationAgent(BaseAgent):
    """Execute an explicit validation policy for a candidate batch."""

    def __init__(
        self,
        message_bus: object | None = None,
        oracles: dict | None = None,
        crg_repository: object | None = None,
        *,
        oracle_timeout_seconds: float | None = None,
    ) -> None:
        super().__init__("validation_agent", message_bus)
        self._subscription_subjects = [
            "agent.validation.request",
            "orchestrator.validate.check",
        ]
        self.crg = ChemicalReasoningGraph()
        self.oracles = dict(oracles) if oracles is not None else _build_default_oracles()
        if crg_repository is None:
            self.crg_repository = build_shared_crg_repository_from_env()
            self._owns_crg_repository = self.crg_repository is not None
        else:
            self.crg_repository = crg_repository
            self._owns_crg_repository = False
        configured_timeout = (
            oracle_timeout_seconds
            if oracle_timeout_seconds is not None
            else _environment_positive_float(
                "VALIDATION_ORACLE_TIMEOUT_SECONDS",
                300.0,
            )
        )
        if not _is_finite_number(configured_timeout) or float(configured_timeout) <= 0:
            raise ValueError("oracle_timeout_seconds must be a positive finite number")
        self.oracle_timeout_seconds = float(configured_timeout)

    def runtime_targets(self) -> dict[str, object | None]:
        legacy_keys = {f"L{level}" for level in range(5)}
        if legacy_keys.issubset(self.oracles):
            targets = {f"oracle.L{level}": self.oracles[f"L{level}"] for level in range(5)}
        else:
            targets = {
                "oracle.rdkit": self._oracle_for_name("rdkit"),
                "oracle.admet": self._oracle_for_name("admet"),
                "oracle.dock": self._oracle_for_name("dock"),
                "oracle.fep": self._oracle_for_name("fep"),
            }
            boltz2 = self._oracle_for_name("boltz2")
            if boltz2 is not None:
                targets["oracle.boltz2"] = boltz2
        if self._owns_crg_repository:
            targets["crg_repository"] = self.crg_repository
        return targets

    async def process(self, data: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(data, Mapping):
            raise ValueError("validation payload must be an object")
        project_id = _required_text(data, "project_id")
        run_id = _required_text(data, "run_id")
        request_id = _required_text(data, "request_id")
        raw_validation_policy = data.get("validation_policy")
        policy = _parse_validation_policy(raw_validation_policy)
        candidates = _parse_candidates(data.get("candidates"))
        external_evidence = _parse_external_evidence(
            data.get("external_evidence"),
            candidates,
        )
        records = [
            {
                "schema_version": "validation.record.v1",
                "candidate_id": candidate["candidate_id"],
                "canonical_smiles": candidate["canonical_smiles"],
                "outcome": None,
                "metrics": [],
                "evidence": [],
                "levels": [],
            }
            for candidate in candidates
        ]
        records_by_id = {record["candidate_id"]: record for record in records}
        candidate_ids_by_smiles: dict[str, list[str]] = {}
        for candidate in candidates:
            candidate_ids_by_smiles.setdefault(
                candidate["canonical_smiles"],
                [],
            ).append(candidate["candidate_id"])
        context = {
            "project_id": project_id,
            "run_id": run_id,
            "request_id": request_id,
        }
        semaphore = asyncio.Semaphore(policy["max_concurrency"])

        for level in range(min(policy["oracle_level"], 3) + 1):
            active_smiles = [
                smiles
                for smiles, candidate_ids in candidate_ids_by_smiles.items()
                if any(
                    records_by_id[candidate_id]["outcome"] is None for candidate_id in candidate_ids
                )
            ]
            if not active_smiles:
                break
            level_results = await self._run_level(
                level,
                active_smiles,
                policy,
                context,
                semaphore,
            )
            for smiles in active_smiles:
                level_result = level_results[smiles]
                for candidate_id in candidate_ids_by_smiles[smiles]:
                    record = records_by_id[candidate_id]
                    if record["outcome"] is not None:
                        continue
                    record["metrics"].extend(copy.deepcopy(level_result["metrics"]))
                    record["evidence"].extend(copy.deepcopy(level_result["evidence"]))
                    record["levels"].append(copy.deepcopy(level_result["level_record"]))
                    if level_result["outcome"] != _OUTCOME_ACCEPTED:
                        record["outcome"] = level_result["outcome"]

        if policy["oracle_level"] == 4:
            thresholds = _thresholds_for_oracle(policy, "external")
            for record in records:
                if record["outcome"] is not None:
                    continue
                level_result = _external_level_result(
                    thresholds,
                    external_evidence.get(record["candidate_id"]),
                )
                record["metrics"].extend(level_result["metrics"])
                record["evidence"].extend(level_result["evidence"])
                record["levels"].append(level_result["level_record"])
                if level_result["outcome"] != _OUTCOME_ACCEPTED:
                    record["outcome"] = level_result["outcome"]

        for record in records:
            if record["outcome"] is None:
                record["outcome"] = _OUTCOME_ACCEPTED
            _deduplicate_evidence(record)
            await self._persist_record(
                record,
                project_id=project_id,
                run_id=run_id,
            )

        return {
            "validation_schema_version": "validation.batch.v1",
            "agent": self.name,
            "project_id": project_id,
            "run_id": run_id,
            "request_id": request_id,
            "outcome": _aggregate_batch_outcome([record["outcome"] for record in records]),
            "validation_policy": copy.deepcopy(dict(raw_validation_policy)),
            "records": records,
        }

    async def _run_level(
        self,
        level: int,
        smiles: list[str],
        policy: dict[str, Any],
        context: dict[str, str],
        semaphore: asyncio.Semaphore,
    ) -> dict[str, dict]:
        oracle_names = [_REQUIRED_ORACLE_BY_LEVEL[level]]
        if level == 1 and _thresholds_for_oracle(policy, "boltz2"):
            oracle_names.append("boltz2")
        chunks = [
            smiles[index : index + policy["batch_size"]]
            for index in range(0, len(smiles), policy["batch_size"])
        ]
        jobs = []
        for oracle_name in oracle_names:
            for chunk_index, chunk in enumerate(chunks):
                call_context = {
                    **context,
                    "request_id": (f"{context['request_id']}:L{level}:{oracle_name}:{chunk_index}"),
                }
                jobs.append(
                    (
                        oracle_name,
                        chunk,
                        asyncio.create_task(
                            self._run_oracle_chunk(
                                level,
                                oracle_name,
                                chunk,
                                _thresholds_for_oracle(
                                    policy,
                                    oracle_name,
                                ),
                                policy["oracle_inputs"],
                                call_context,
                                semaphore,
                            )
                        ),
                    )
                )
        completed = await asyncio.gather(*(job[2] for job in jobs))
        by_oracle: dict[str, dict[str, dict]] = {oracle_name: {} for oracle_name in oracle_names}
        for (oracle_name, _chunk, _task), chunk_result in zip(
            jobs,
            completed,
            strict=True,
        ):
            by_oracle[oracle_name].update(chunk_result)

        results: dict[str, dict] = {}
        for canonical_smiles in smiles:
            oracle_records = [
                by_oracle[oracle_name][canonical_smiles] for oracle_name in oracle_names
            ]
            outcome = _aggregate_level_outcome([record["outcome"] for record in oracle_records])
            results[canonical_smiles] = {
                "outcome": outcome,
                "metrics": [
                    copy.deepcopy(metric)
                    for oracle_record in oracle_records
                    for metric in oracle_record["metrics"]
                ],
                "evidence": [
                    copy.deepcopy(evidence)
                    for oracle_record in oracle_records
                    for evidence in oracle_record["evidence"]
                ],
                "level_record": {
                    "level": level,
                    "outcome": outcome,
                    "oracles": [
                        copy.deepcopy(record["oracle_record"]) for record in oracle_records
                    ],
                },
            }
        return results

    async def _run_oracle_chunk(
        self,
        level: int,
        oracle_name: str,
        smiles: list[str],
        thresholds: list[dict],
        oracle_inputs: dict[str, dict],
        context: dict[str, str],
        semaphore: asyncio.Semaphore,
    ) -> dict[str, dict]:
        oracle = self._oracle_for_name(oracle_name)
        if oracle is None:
            return _chunk_error_results(
                level,
                oracle_name,
                smiles,
                "MISSING_ORACLE",
                f"Oracle {oracle_name} is not configured",
            )
        invalid_results: dict[str, dict] = {}
        oracle_smiles = list(smiles)
        if oracle_name == "rdkit":
            invalid_results = {
                canonical_smiles: _oracle_error_result(
                    level,
                    oracle_name,
                    "INVALID_SMILES",
                    f"invalid SMILES: {canonical_smiles}",
                )
                for canonical_smiles in smiles
                if not _is_valid_smiles(canonical_smiles)
            }
            oracle_smiles = [
                canonical_smiles
                for canonical_smiles in smiles
                if canonical_smiles not in invalid_results
            ]
            if not oracle_smiles:
                return invalid_results
        try:
            request_context = _oracle_request_context(
                oracle_name,
                oracle_inputs,
                context,
            )
        except RuntimeError as exc:
            errors = _chunk_error_results(
                level,
                oracle_name,
                oracle_smiles,
                "MISSING_INPUT",
                str(exc),
            )
            return _merge_chunk_results(smiles, invalid_results, errors)
        properties = [threshold["metric"] for threshold in thresholds]
        requires_uncertainty = any("max_uncertainty" in threshold for threshold in thresholds)
        try:
            async with semaphore:
                raw_result = await asyncio.wait_for(
                    _invoke_oracle(
                        oracle,
                        oracle_smiles,
                        properties,
                        request_context,
                        requires_uncertainty=requires_uncertainty,
                    ),
                    timeout=self.oracle_timeout_seconds,
                )
            normalized = _normalize_oracle_result(
                level,
                oracle_name,
                oracle_smiles,
                thresholds,
                raw_result,
            )
            return _merge_chunk_results(smiles, invalid_results, normalized)
        except TimeoutError:
            errors = _chunk_error_results(
                level,
                oracle_name,
                oracle_smiles,
                "TIMEOUT",
                f"Oracle {oracle_name} timed out",
            )
            return _merge_chunk_results(smiles, invalid_results, errors)
        except Exception as exc:
            errors = _chunk_error_results(
                level,
                oracle_name,
                oracle_smiles,
                "PROTOCOL_ERROR",
                str(exc) or type(exc).__name__,
            )
            return _merge_chunk_results(smiles, invalid_results, errors)

    def _oracle_for_name(self, oracle_name: str) -> object | None:
        if oracle_name in self.oracles:
            return self.oracles[oracle_name]
        legacy_keys = {
            "rdkit": (0, "L0"),
            "admet": (1, "L1"),
            "dock": (2, "L2"),
            "fep": (3, "L3"),
        }.get(oracle_name, ())
        for key in legacy_keys:
            if key in self.oracles:
                return self.oracles[key]
        return None

    async def _persist_record(
        self,
        record: dict,
        *,
        project_id: str,
        run_id: str,
    ) -> None:
        source_evidence_ids = [
            item["evidence_id"] for item in record["evidence"] if item.get("evidence_id")
        ]
        belief = self.crg.add_belief(
            subject=record["candidate_id"],
            predicate="validation_record",
            obj="",
            confidence=1.0,
            source_agent=self.name,
            evidence_ids=source_evidence_ids,
        )
        record["evidence"].append(
            {
                "evidence_id": belief.id,
                "level": record["levels"][-1]["level"],
                "oracle": self.name,
            }
        )
        belief.object = json.dumps(
            record,
            sort_keys=True,
            separators=(",", ":"),
        )
        await self._persist_belief(
            belief,
            project_id=project_id,
            run_id=run_id,
        )

    async def _persist_belief(
        self,
        belief: Belief,
        project_id: str,
        run_id: str,
    ) -> None:
        if self.crg_repository is None:
            return
        write_belief = getattr(self.crg_repository, "write_workflow_belief", None)
        if not callable(write_belief):
            raise TypeError("crg_repository must expose write_workflow_belief(**kwargs)")
        result = write_belief(
            project_id=project_id,
            run_id=run_id,
            belief_id=belief.id,
            subject=belief.subject,
            predicate=belief.predicate,
            object_value=belief.object,
            confidence=belief.confidence,
            source_agent=belief.source_agent,
            timestamp_ns=belief.timestamp_ns,
            evidence_ids=list(belief.evidence_ids),
        )
        if inspect.isawaitable(result):
            await result


def _parse_validation_policy(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("validation_policy must be an object")
    _reject_unknown_fields(
        value,
        _VALIDATION_POLICY_FIELDS,
        "validation_policy",
    )
    for field in _VALIDATION_POLICY_REQUIRED_FIELDS:
        if field not in value:
            raise ValueError(f"{field} is required")
    oracle_level = value["oracle_level"]
    if (
        isinstance(oracle_level, bool)
        or not isinstance(oracle_level, int)
        or oracle_level not in range(5)
    ):
        raise ValueError("oracle_level must be an integer between 0 and 4")
    batch_size = _positive_integer(value["batch_size"], "batch_size")
    max_concurrency = _positive_integer(
        value["max_concurrency"],
        "max_concurrency",
    )
    raw_thresholds = value["thresholds"]
    if not isinstance(raw_thresholds, list):
        raise ValueError("thresholds must be a list")
    thresholds: list[dict[str, Any]] = []
    threshold_keys: set[tuple[int, str, str]] = set()
    for index, raw_threshold in enumerate(raw_thresholds):
        if not isinstance(raw_threshold, Mapping):
            raise ValueError(f"thresholds[{index}] must be an object")
        threshold = _parse_threshold(raw_threshold)
        if threshold["level"] > oracle_level:
            raise ValueError("threshold level exceeds oracle_level")
        expected_level = _ORACLE_LEVEL_BY_NAME.get(threshold["oracle"])
        if expected_level is None:
            raise ValueError(f"unsupported threshold oracle: {threshold['oracle']}")
        if threshold["level"] != expected_level:
            raise ValueError(f"oracle {threshold['oracle']} belongs to L{expected_level}")
        _validate_oracle_metric(threshold["oracle"], threshold["metric"])
        key = (
            threshold["level"],
            threshold["oracle"],
            threshold["metric"],
        )
        if key in threshold_keys:
            raise ValueError(
                "duplicate threshold for "
                f"L{threshold['level']} {threshold['oracle']} {threshold['metric']}"
            )
        threshold_keys.add(key)
        thresholds.append(threshold)
    for level in range(oracle_level + 1):
        required_oracle = _REQUIRED_ORACLE_BY_LEVEL[level]
        if not any(
            threshold["level"] == level and threshold["oracle"] == required_oracle
            for threshold in thresholds
        ):
            raise ValueError(f"L{level} requires a {required_oracle} threshold")
    oracle_inputs = _parse_oracle_inputs(value["oracle_inputs"])
    return {
        "oracle_level": oracle_level,
        "batch_size": batch_size,
        "max_concurrency": max_concurrency,
        "thresholds": thresholds,
        "oracle_inputs": oracle_inputs,
    }


def _parse_threshold(value: Mapping[str, Any]) -> dict[str, Any]:
    _reject_unknown_fields(value, _THRESHOLD_FIELDS, "threshold")
    for field in ("level", "oracle", "metric", "direction", "value"):
        if field not in value:
            raise ValueError(f"threshold {field} is required")
    level = value["level"]
    if isinstance(level, bool) or not isinstance(level, int) or level not in range(5):
        raise ValueError("threshold level must be an integer between 0 and 4")
    oracle = _nonempty_text(value["oracle"], "threshold oracle")
    metric = _nonempty_text(value["metric"], "threshold metric")
    direction = _nonempty_text(value["direction"], "threshold direction")
    if direction not in {"maximize", "minimize"}:
        raise ValueError("threshold direction must be maximize or minimize")
    threshold = {
        "level": level,
        "oracle": oracle,
        "metric": metric,
        "direction": direction,
        "value": _finite_float(
            value["value"],
            "threshold value",
            error_type=ValueError,
        ),
    }
    if "max_uncertainty" in value:
        max_uncertainty = _finite_float(
            value["max_uncertainty"],
            "threshold max_uncertainty",
            error_type=ValueError,
        )
        if max_uncertainty < 0:
            raise ValueError("threshold max_uncertainty must be non-negative")
        threshold["max_uncertainty"] = max_uncertainty
    return threshold


def _validate_oracle_metric(oracle_name: str, metric: str) -> None:
    if oracle_name == "rdkit" and metric not in _RDKIT_METRICS:
        raise ValueError(f"unsupported rdkit metric: {metric}")
    fixed_metrics = _FIXED_ORACLE_METRICS.get(oracle_name)
    if fixed_metrics is not None and metric not in fixed_metrics:
        raise ValueError(f"unsupported {oracle_name} metric: {metric}")


def _parse_oracle_inputs(value: object) -> dict[str, dict]:
    if not isinstance(value, Mapping):
        raise ValueError("oracle_inputs must be an object")
    result: dict[str, dict] = {}
    for raw_oracle_name, raw_inputs in value.items():
        oracle_name = _nonempty_text(raw_oracle_name, "oracle_inputs key")
        if oracle_name not in _ORACLE_INPUT_FIELDS:
            raise ValueError(f"unsupported oracle_inputs key: {oracle_name}")
        if not isinstance(raw_inputs, Mapping):
            raise ValueError(f"oracle_inputs.{oracle_name} must be an object")
        inputs: dict[str, Any] = {}
        for raw_key, raw_input in raw_inputs.items():
            key = _nonempty_text(
                raw_key,
                f"oracle_inputs.{oracle_name} key",
            )
            if key not in _ORACLE_INPUT_FIELDS[oracle_name]:
                raise ValueError(f"unsupported oracle_inputs.{oracle_name} field: {key}")
            if key == "oracle_parameters":
                if not isinstance(raw_input, Mapping):
                    raise ValueError(
                        f"oracle_inputs.{oracle_name}.oracle_parameters must be an object"
                    )
                inputs[key] = _parse_oracle_parameters(
                    oracle_name,
                    raw_input,
                )
            else:
                inputs[key] = _nonempty_text(
                    raw_input,
                    f"oracle_inputs.{oracle_name}.{key}",
                )
        result[oracle_name] = inputs
    return result


def _parse_oracle_parameters(
    oracle_name: str,
    value: Mapping[object, object],
) -> dict[str, str]:
    allowed_parameters = _ORACLE_PARAMETER_FIELDS.get(oracle_name)
    if allowed_parameters is None:
        raise ValueError(f"unsupported oracle parameters key: {oracle_name}")
    result: dict[str, str] = {}
    for raw_name, raw_value in value.items():
        name = _nonempty_text(raw_name, "oracle parameter name")
        if name not in allowed_parameters:
            raise ValueError(f"unsupported oracle parameter {oracle_name}.{name}")
        if oracle_name == "dock" and name == "engine":
            engine = _nonempty_text(raw_value, "dock engine")
            if engine not in {"gnina", "diffdock"}:
                raise ValueError("dock engine must be gnina or diffdock")
            result[name] = engine
        elif oracle_name == "fep" and name == "method":
            result[name] = _nonempty_text(raw_value, "fep method")
        elif oracle_name == "fep" and name == "n_repeats":
            result[name] = str(_positive_integer(raw_value, "fep n_repeats"))
        elif oracle_name == "boltz2" and name == "ensemble_size":
            result[name] = str(_positive_integer(raw_value, "boltz2 ensemble_size"))
        else:
            raise ValueError(f"unsupported oracle parameter {oracle_name}.{name}")
    return result


def _parse_candidates(value: object) -> list[dict[str, str]]:
    if not isinstance(value, list) or not value:
        raise ValueError("candidates must be a non-empty list")
    result: list[dict[str, str]] = []
    candidate_ids: set[str] = set()
    for index, candidate in enumerate(value):
        if not isinstance(candidate, Mapping):
            raise ValueError(f"candidates[{index}] must be an object")
        candidate_id = _nonempty_text(
            candidate.get("candidate_id"),
            f"candidates[{index}].candidate_id",
        )
        if candidate_id in candidate_ids:
            raise ValueError(f"duplicate candidate_id: {candidate_id}")
        candidate_ids.add(candidate_id)
        result.append(
            {
                "candidate_id": candidate_id,
                "canonical_smiles": _nonempty_text(
                    candidate.get("canonical_smiles"),
                    f"candidates[{index}].canonical_smiles",
                ),
            }
        )
    return result


def _parse_external_evidence(
    value: object,
    candidates: list[dict[str, str]],
) -> dict[str, dict]:
    if value is None:
        return {}
    if not isinstance(value, list):
        raise ValueError("external_evidence must be a list")
    candidates_by_id = {candidate["candidate_id"]: candidate for candidate in candidates}
    result: dict[str, dict] = {}
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ValueError(f"external_evidence[{index}] must be an object")
        candidate_id = _nonempty_text(
            item.get("candidate_id"),
            f"external_evidence[{index}].candidate_id",
        )
        if candidate_id not in candidates_by_id:
            raise ValueError(f"external_evidence references unknown candidate_id: {candidate_id}")
        if candidate_id in result:
            raise ValueError(f"duplicate external_evidence candidate_id: {candidate_id}")
        if (
            "canonical_smiles" in item
            and str(item["canonical_smiles"]).strip()
            != (candidates_by_id[candidate_id]["canonical_smiles"])
        ):
            raise ValueError(f"external_evidence canonical_smiles mismatch for {candidate_id}")
        metrics = item.get("metrics")
        uncertainties = item.get("uncertainties", {})
        evidence_ids = item.get("evidence_ids", [])
        result[candidate_id] = {
            "metrics": dict(metrics) if isinstance(metrics, Mapping) else metrics,
            "uncertainties": (
                dict(uncertainties) if isinstance(uncertainties, Mapping) else uncertainties
            ),
            "evidence_ids": (
                list(evidence_ids) if isinstance(evidence_ids, list) else evidence_ids
            ),
        }
    return result


def _oracle_request_context(
    oracle_name: str,
    oracle_inputs: dict[str, dict],
    context: dict[str, str],
) -> dict[str, Any]:
    inputs = dict(oracle_inputs.get(oracle_name, {}))
    if oracle_name == "boltz2":
        _require_oracle_input(inputs, oracle_name, "protein_pdb_id")
    elif oracle_name == "dock":
        _require_oracle_input(inputs, oracle_name, "receptor_uri")
        _require_oracle_parameters(inputs, oracle_name, ("engine",))
    elif oracle_name == "fep":
        _require_oracle_input(inputs, oracle_name, "protein_pdb_id")
        _require_oracle_input(
            inputs,
            oracle_name,
            "reference_ligand_smiles",
        )
        _require_oracle_parameters(
            inputs,
            oracle_name,
            ("method", "n_repeats"),
        )
    return {
        "project_id": context["project_id"],
        "request_id": context["request_id"],
        **inputs,
    }


def _require_oracle_input(
    inputs: dict[str, Any],
    oracle_name: str,
    field: str,
) -> None:
    if not str(inputs.get(field) or "").strip():
        raise RuntimeError(f"{oracle_name} requires {field}")


def _require_oracle_parameters(
    inputs: dict[str, Any],
    oracle_name: str,
    required: tuple[str, ...],
) -> None:
    parameters = inputs.get("oracle_parameters")
    if not isinstance(parameters, Mapping) or not parameters:
        raise RuntimeError(f"{oracle_name} requires oracle_parameters")
    missing = [
        parameter for parameter in required if not str(parameters.get(parameter) or "").strip()
    ]
    if missing:
        raise RuntimeError(f"{oracle_name} oracle_parameters requires: {', '.join(missing)}")


async def _invoke_oracle(
    oracle: object,
    molecules: list[str],
    properties: list[str],
    request_context: dict[str, Any],
    *,
    requires_uncertainty: bool,
) -> object:
    method_name = "predict_with_uncertainty" if requires_uncertainty else "evaluate"
    method = getattr(oracle, method_name, None)
    if not callable(method):
        raise RuntimeError(f"oracle must expose {method_name}")
    kwargs = {}
    try:
        parameters = inspect.signature(method).parameters.values()
    except (TypeError, ValueError):
        parameters = ()
    if any(
        parameter.name == "request_context" or parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    ):
        kwargs["request_context"] = request_context
    result = method(list(molecules), list(properties), **kwargs)
    if inspect.isawaitable(result):
        result = await result
    return result


def _normalize_oracle_result(
    level: int,
    oracle_name: str,
    molecules: list[str],
    thresholds: list[dict],
    result: object,
) -> dict[str, dict]:
    if not isinstance(result, Mapping):
        raise RuntimeError("oracle batch result must be an object")
    actual_order = [str(key) for key in result]
    if len(result) != len(molecules):
        raise RuntimeError(f"oracle returned {len(result)} rows, expected {len(molecules)}")
    if actual_order != molecules:
        raise RuntimeError("oracle batch result order does not match request")
    normalized: dict[str, dict] = {}
    for smiles in molecules:
        normalized[smiles] = _normalize_oracle_item(
            level,
            oracle_name,
            thresholds,
            result[smiles],
        )
    return normalized


def _normalize_oracle_item(
    level: int,
    oracle_name: str,
    thresholds: list[dict],
    item: object,
) -> dict:
    meta: dict[str, Any] = {}
    if isinstance(item, tuple) and len(item) == 2:
        values, uncertainty = item
    elif isinstance(item, Mapping) and isinstance(item.get("scores"), Mapping):
        meta = dict(item)
        values = item.get("scores")
        uncertainty = item.get("uncertainties", {})
    elif isinstance(item, Mapping):
        values = item
        uncertainty = item.get("uncertainties", {})
    else:
        return _oracle_error_result(
            level,
            oracle_name,
            "PROTOCOL_ERROR",
            "oracle row must be an object or (scores, uncertainty)",
        )
    if not isinstance(values, Mapping):
        return _oracle_error_result(
            level,
            oracle_name,
            "PROTOCOL_ERROR",
            "oracle scores must be an object",
        )
    if values.get("skipped") is True:
        return _oracle_error_result(
            level,
            oracle_name,
            "SKIPPED",
            str(values.get("skip_reason") or "required oracle was skipped"),
            evidence=_evidence_from_meta(level, oracle_name, meta),
        )
    if meta.get("success") is False:
        return _oracle_error_result(
            level,
            oracle_name,
            str(meta.get("error_code") or "ORACLE_ERROR"),
            str(meta.get("error_message") or "oracle evaluation failed"),
            evidence=_evidence_from_meta(level, oracle_name, meta),
        )
    wire_outcome = _wire_outcome(meta.get("outcome"))
    if wire_outcome in {"ORACLE_OUTCOME_SKIPPED", "ORACLE_OUTCOME_ERROR"}:
        return _oracle_error_result(
            level,
            oracle_name,
            str(meta.get("error_code") or wire_outcome),
            str(meta.get("error_message") or f"oracle returned {wire_outcome}"),
            evidence=_evidence_from_meta(level, oracle_name, meta),
        )
    if wire_outcome == "ORACLE_OUTCOME_UNSPECIFIED":
        return _oracle_error_result(
            level,
            oracle_name,
            "PROTOCOL_ERROR",
            "oracle outcome is unspecified",
            evidence=_evidence_from_meta(level, oracle_name, meta),
        )
    if not isinstance(uncertainty, Mapping):
        return _oracle_error_result(
            level,
            oracle_name,
            "PROTOCOL_ERROR",
            "oracle uncertainties must be an object",
            evidence=_evidence_from_meta(level, oracle_name, meta),
        )
    metrics: list[dict] = []
    for threshold in thresholds:
        metric = threshold["metric"]
        if metric not in values:
            return _oracle_error_result(
                level,
                oracle_name,
                "MISSING_METRIC",
                f"Oracle {oracle_name} result requires metric {metric}",
                evidence=_evidence_from_meta(level, oracle_name, meta),
            )
        if not _is_finite_number(values[metric]):
            return _oracle_error_result(
                level,
                oracle_name,
                "INVALID_METRIC",
                f"Oracle {oracle_name} metric {metric} must be finite",
                evidence=_evidence_from_meta(level, oracle_name, meta),
            )
        value = float(values[metric])
        passed = (
            value >= threshold["value"]
            if threshold["direction"] == "maximize"
            else value <= threshold["value"]
        )
        metric_record = {
            "level": level,
            "oracle": oracle_name,
            "metric": metric,
            "value": value,
            "direction": threshold["direction"],
            "threshold": threshold["value"],
        }
        if "max_uncertainty" in threshold:
            if metric not in uncertainty:
                return _oracle_error_result(
                    level,
                    oracle_name,
                    "MISSING_UNCERTAINTY",
                    f"Oracle {oracle_name} result requires uncertainty for {metric}",
                    evidence=_evidence_from_meta(level, oracle_name, meta),
                )
            if not _is_finite_number(uncertainty[metric]) or float(uncertainty[metric]) < 0:
                return _oracle_error_result(
                    level,
                    oracle_name,
                    "INVALID_UNCERTAINTY",
                    (
                        f"Oracle {oracle_name} uncertainty for {metric} "
                        "must be finite and non-negative"
                    ),
                    evidence=_evidence_from_meta(level, oracle_name, meta),
                )
            uncertainty_value = float(uncertainty[metric])
            metric_record["uncertainty"] = uncertainty_value
            metric_record["max_uncertainty"] = threshold["max_uncertainty"]
            passed = passed and uncertainty_value <= threshold["max_uncertainty"]
        metric_record["passed"] = passed
        metrics.append(metric_record)
    outcome = (
        _OUTCOME_ACCEPTED
        if all(metric["passed"] for metric in metrics) and wire_outcome != "ORACLE_OUTCOME_FAIL"
        else _OUTCOME_FAIL
    )
    evidence = _evidence_from_meta(level, oracle_name, meta)
    return {
        "outcome": outcome,
        "metrics": metrics,
        "evidence": evidence,
        "oracle_record": {
            "oracle": oracle_name,
            "outcome": outcome,
            "metrics": copy.deepcopy(metrics),
            "evidence_ids": [item["evidence_id"] for item in evidence],
        },
    }


def _external_level_result(
    thresholds: list[dict],
    evidence_item: dict | None,
) -> dict:
    if evidence_item is None:
        oracle_record = {
            "oracle": "external",
            "outcome": _OUTCOME_AWAITING,
            "metrics": [],
            "evidence_ids": [],
            "reason": "external evidence is required",
        }
        return {
            "outcome": _OUTCOME_AWAITING,
            "metrics": [],
            "evidence": [],
            "level_record": {
                "level": 4,
                "outcome": _OUTCOME_AWAITING,
                "oracles": [oracle_record],
            },
        }
    evidence_ids = evidence_item.get("evidence_ids")
    if (
        not isinstance(evidence_ids, list)
        or not evidence_ids
        or any(not isinstance(item, str) or not item.strip() for item in evidence_ids)
    ):
        oracle_result = _oracle_error_result(
            4,
            "external",
            "MISSING_EVIDENCE_ID",
            "external evidence_ids must be a non-empty string list",
        )
    else:
        evidence = [
            {
                "evidence_id": item.strip(),
                "level": 4,
                "oracle": "external",
            }
            for item in evidence_ids
        ]
        oracle_result = _normalize_oracle_item(
            4,
            "external",
            thresholds,
            (
                evidence_item.get("metrics"),
                evidence_item.get("uncertainties", {}),
            ),
        )
        oracle_result["evidence"] = evidence
        oracle_result["oracle_record"]["evidence_ids"] = [item["evidence_id"] for item in evidence]
    return {
        "outcome": oracle_result["outcome"],
        "metrics": oracle_result["metrics"],
        "evidence": oracle_result["evidence"],
        "level_record": {
            "level": 4,
            "outcome": oracle_result["outcome"],
            "oracles": [oracle_result["oracle_record"]],
        },
    }


def _chunk_error_results(
    level: int,
    oracle_name: str,
    smiles: list[str],
    code: str,
    message: str,
) -> dict[str, dict]:
    return {
        canonical_smiles: _oracle_error_result(
            level,
            oracle_name,
            code,
            message,
        )
        for canonical_smiles in smiles
    }


def _merge_chunk_results(
    order: list[str],
    first: Mapping[str, dict],
    second: Mapping[str, dict],
) -> dict[str, dict]:
    return {
        canonical_smiles: (
            first[canonical_smiles] if canonical_smiles in first else second[canonical_smiles]
        )
        for canonical_smiles in order
    }


def _is_valid_smiles(smiles: str) -> bool:
    from rdkit import Chem, rdBase

    with rdBase.BlockLogs():
        return Chem.MolFromSmiles(smiles) is not None


def _oracle_error_result(
    level: int,
    oracle_name: str,
    code: str,
    message: str,
    *,
    evidence: list[dict] | None = None,
) -> dict:
    references = list(evidence or [])
    return {
        "outcome": _OUTCOME_ERROR,
        "metrics": [],
        "evidence": references,
        "oracle_record": {
            "oracle": oracle_name,
            "outcome": _OUTCOME_ERROR,
            "metrics": [],
            "evidence_ids": [item["evidence_id"] for item in references],
            "error": {
                "code": str(code),
                "message": str(message) or str(code),
            },
        },
    }


def _evidence_from_meta(
    level: int,
    oracle_name: str,
    meta: Mapping[str, Any],
) -> list[dict]:
    evidence_id = str(meta.get("evidence_id") or "").strip()
    if not evidence_id:
        return []
    evidence = {
        "evidence_id": evidence_id,
        "level": level,
        "oracle": oracle_name,
    }
    reported_oracle = str(meta.get("oracle_name") or "").strip()
    if reported_oracle:
        evidence["reported_oracle"] = reported_oracle
    oracle_version = str(meta.get("oracle_version") or "").strip()
    if oracle_version:
        evidence["oracle_version"] = oracle_version
    model_version = str(meta.get("model_version") or "").strip()
    if model_version:
        evidence["model_version"] = model_version
    artifact_refs = meta.get("artifact_refs")
    if isinstance(artifact_refs, list) and artifact_refs:
        evidence["artifact_refs"] = copy.deepcopy(artifact_refs)
    return [evidence]


def _aggregate_level_outcome(outcomes: list[str]) -> str:
    if not outcomes or any(outcome not in _OUTCOMES for outcome in outcomes):
        raise ValueError("level outcomes must be non-empty validation outcomes")
    if _OUTCOME_ERROR in outcomes:
        return _OUTCOME_ERROR
    if _OUTCOME_FAIL in outcomes:
        return _OUTCOME_FAIL
    if _OUTCOME_AWAITING in outcomes:
        return _OUTCOME_AWAITING
    return _OUTCOME_ACCEPTED


def _aggregate_batch_outcome(outcomes: list[str]) -> str:
    if not outcomes or any(outcome not in _OUTCOMES for outcome in outcomes):
        raise ValueError("batch outcomes must be non-empty validation outcomes")
    if _OUTCOME_ERROR in outcomes:
        return _OUTCOME_ERROR
    if _OUTCOME_ACCEPTED in outcomes:
        return _OUTCOME_ACCEPTED
    if _OUTCOME_AWAITING in outcomes:
        return _OUTCOME_AWAITING
    return _OUTCOME_FAIL


def _thresholds_for_oracle(
    policy: Mapping[str, Any],
    oracle_name: str,
) -> list[dict]:
    return [threshold for threshold in policy["thresholds"] if threshold["oracle"] == oracle_name]


def _deduplicate_evidence(record: dict) -> None:
    evidence = []
    seen: set[str] = set()
    for item in record["evidence"]:
        evidence_id = str(item.get("evidence_id") or "")
        if not evidence_id or evidence_id in seen:
            continue
        seen.add(evidence_id)
        evidence.append(item)
    record["evidence"] = evidence


def _required_text(data: Mapping[str, Any], field: str) -> str:
    if field not in data:
        raise ValueError(f"{field} is required")
    return _nonempty_text(data[field], field)


def _nonempty_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _positive_integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _finite_float(
    value: object,
    field: str,
    *,
    error_type: type[Exception] = RuntimeError,
) -> float:
    if not _is_finite_number(value):
        raise error_type(f"{field} must be a finite number")
    return float(value)


def _is_finite_number(value: object) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(float(value))


def _mapping_or_empty(value: object) -> Mapping:
    return value if isinstance(value, Mapping) else {}


def _reject_unknown_fields(
    value: Mapping,
    allowed_fields: frozenset[str],
    object_name: str,
) -> None:
    unknown_fields = sorted(str(field) for field in value if field not in allowed_fields)
    if unknown_fields:
        raise ValueError(f"unsupported {object_name} field: {unknown_fields[0]}")


def _environment_positive_float(name: str, default: float) -> float:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive finite number") from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return value


def _build_default_oracles() -> dict[str, object]:
    from mf_oracles.rdkit_oracle.oracle import RDKitOracle

    oracles: dict[str, object] = {
        "rdkit": _BatchEvaluateOnlyOracle(
            RDKitOracle(),
            oracle_name="rdkit",
        )
    }
    targets = (
        ("admet", "L1_ADMET_ORACLE_TARGET", 1),
        ("boltz2", "L1_BOLTZ2_ORACLE_TARGET", 1),
        ("dock", "L2_DOCK_ORACLE_TARGET", 2),
        ("fep", "L3_FEP_ORACLE_TARGET", 3),
    )
    for oracle_name, environment_name, level in targets:
        target = os.environ.get(environment_name, "").strip()
        if target:
            oracles[oracle_name] = OracleGrpcClient(
                target,
                level=level,
                oracle_name=oracle_name,
            )
    return oracles


class _BatchEvaluateOnlyOracle:
    """Run local synchronous-capable oracles outside the event-loop thread."""

    def __init__(
        self,
        oracle: object,
        *,
        level: int | None = None,
        oracle_name: str | None = None,
    ) -> None:
        self.oracle = oracle
        if oracle_name is None:
            oracle_name = {
                0: "rdkit",
                1: "admet",
                2: "dock",
                3: "fep",
            }.get(level)
        if oracle_name not in _HEALTH_METRIC_BY_ORACLE:
            raise ValueError("local oracle_name is required")
        self.oracle_name = oracle_name

    @property
    def _close_target(self) -> object:
        return self.oracle

    async def evaluate(
        self,
        molecules: list[str],
        properties: list[str],
    ) -> dict:
        return await run_health_probe_in_daemon(
            lambda: _run_oracle_evaluate(
                self.oracle,
                molecules,
                properties,
            )
        )

    async def predict_with_uncertainty(
        self,
        molecules: list[str],
        properties: list[str],
    ) -> dict:
        if self.oracle_name != "rdkit":
            raise RuntimeError(
                f"local {self.oracle_name} oracle does not provide batch uncertainty"
            )
        scores = await self.evaluate(molecules, properties)
        return {
            smiles: (
                values,
                {property_name: 0.0 for property_name in properties},
            )
            for smiles, values in scores.items()
        }

    async def health_check(self) -> dict[str, bool]:
        property_name = _HEALTH_METRIC_BY_ORACLE[self.oracle_name]
        try:
            result = await self.evaluate(["C"], [property_name])
        except Exception:
            return {"healthy": False}
        scores = result.get("C") if isinstance(result, Mapping) else None
        return {
            "healthy": bool(
                isinstance(scores, Mapping) and _is_finite_number(scores.get(property_name))
            )
        }

    async def close(self) -> None:
        close = getattr(self.oracle, "close", None)
        if not callable(close):
            return
        result = close()
        if inspect.isawaitable(result):
            await result


def _run_oracle_evaluate(
    oracle: object,
    molecules: list[str],
    properties: list[str],
) -> dict:
    result = oracle.evaluate(molecules, properties)
    if inspect.isawaitable(result):
        return asyncio.run(result)
    return result


def _oracle_required_properties(level: int) -> list[str]:
    oracle_name = {
        0: "rdkit",
        1: "admet",
        2: "dock",
        3: "fep",
    }.get(level)
    if oracle_name is None:
        raise ValueError(f"unsupported Oracle health level: {level}")
    return [_HEALTH_METRIC_BY_ORACLE[oracle_name]]


def _oracle_batch_request(
    molecules: list[str],
    properties: list[str],
    level: int,
    request_context: Mapping[str, Any],
    return_uncertainty: bool = False,
) -> oracle_pb2.OracleBatchRequest:
    return oracle_pb2.OracleBatchRequest(
        project_id=str(request_context.get("project_id") or ""),
        molecule_smiles=[str(item) for item in molecules],
        level=_oracle_level_proto(level),
        requested_properties=[str(item) for item in properties],
        return_uncertainty=return_uncertainty,
        receptor_uri=str(request_context.get("receptor_uri") or ""),
        protein_pdb_id=str(request_context.get("protein_pdb_id") or ""),
        reference_ligand_smiles=str(request_context.get("reference_ligand_smiles") or ""),
        oracle_parameters={
            str(key): str(value)
            for key, value in _mapping_or_empty(request_context.get("oracle_parameters")).items()
        },
        request_id=str(request_context.get("request_id") or ""),
    )


def _oracle_level_proto(level: int) -> int:
    try:
        return _PROTO_LEVEL_BY_HTTP_LEVEL[level]
    except KeyError as exc:
        raise ValueError("oracle level must be an integer between 0 and 4") from exc


def _evaluations_by_smiles(
    response: oracle_pb2.OracleBatchResponse,
    molecules: list[str],
    properties: list[str],
    *,
    expected_level: int,
    expected_oracle_name: str,
    request_context: Mapping[str, Any],
    request_id: str,
) -> dict[str, dict]:
    if str(response.batch_id) != request_id:
        raise RuntimeError("oracle response batch_id does not match request_id")
    evaluations = list(response.evaluations)
    if len(evaluations) != len(molecules):
        raise RuntimeError(
            f"oracle response count {len(evaluations)} does not match "
            f"request count {len(molecules)}"
        )
    actual_order = [str(evaluation.molecule_smiles) for evaluation in evaluations]
    if actual_order != molecules:
        raise RuntimeError("oracle response molecule order does not match request")
    expected_proto_level = _oracle_level_proto(expected_level)
    expected_reported_oracle_names = _expected_reported_oracle_names(
        expected_oracle_name,
        request_context,
    )
    result: dict[str, dict] = {}
    evidence_ids: set[str] = set()
    batch_artifact_refs: tuple[tuple[str, str, str, bool], ...] | None = None
    for index, evaluation in enumerate(evaluations):
        if int(evaluation.level) != expected_proto_level:
            raise RuntimeError("oracle response level does not match request")
        reported_oracle_name = str(evaluation.oracle_name).strip()
        if not reported_oracle_name:
            raise RuntimeError("oracle response oracle_name is required")
        if reported_oracle_name not in expected_reported_oracle_names:
            raise RuntimeError("oracle response logical oracle identity does not match request")
        evidence_id = str(evaluation.evidence_id)
        expected_evidence_id = f"{request_id}:{reported_oracle_name}:{index}"
        if evidence_id != expected_evidence_id:
            raise RuntimeError("oracle response evidence_id does not match request")
        if evidence_id in evidence_ids:
            raise RuntimeError("oracle response evidence_id must be unique within batch")
        evidence_ids.add(evidence_id)
        outcome = int(evaluation.outcome)
        if outcome == oracle_pb2.ORACLE_OUTCOME_UNSPECIFIED:
            raise RuntimeError("oracle response outcome is unspecified")
        success = bool(evaluation.success)
        error_code = str(evaluation.error_code)
        error_message = str(evaluation.error_message)
        _validate_wire_status(
            success,
            outcome,
            error_code,
            error_message,
            error_message_present=evaluation.HasField("error_message"),
        )
        scores = _validated_wire_numeric_mapping(
            evaluation.scores,
            "score",
        )
        uncertainties = _validated_wire_numeric_mapping(
            evaluation.uncertainties,
            "uncertainty",
            non_negative=True,
        )
        _validate_wire_artifact_refs(
            evaluation.artifact_refs,
            require_nonempty=(success and reported_oracle_name not in {"admet", "admet_ai"}),
        )
        artifact_refs = tuple(
            (
                str(artifact.name),
                str(artifact.version),
                str(artifact.checksum),
                bool(artifact.required),
            )
            for artifact in evaluation.artifact_refs
        )
        if batch_artifact_refs is None:
            batch_artifact_refs = artifact_refs
        elif artifact_refs != batch_artifact_refs:
            raise RuntimeError("oracle response artifact_refs must match within batch")
        if success:
            requested_properties = set(properties)
            if set(scores) != requested_properties or not set(uncertainties).issubset(
                requested_properties
            ):
                raise RuntimeError(
                    "oracle response PASS scores and uncertainties "
                    "do not match requested_properties"
                )
            metric_properties = [str(metric.property) for metric in evaluation.metrics]
            if metric_properties != properties:
                raise RuntimeError("oracle response metric order does not match request")
            for metric in evaluation.metrics:
                property_name = str(metric.property)
                metric_value = float(metric.value)
                if not math.isfinite(metric_value):
                    raise RuntimeError(
                        f"oracle response typed metric {property_name} must be finite"
                    )
                if metric_value != scores[property_name]:
                    raise RuntimeError("oracle response typed metric contradicts score map")
                has_uncertainty = metric.HasField("uncertainty")
                if has_uncertainty != (property_name in uncertainties):
                    raise RuntimeError("oracle response typed metric contradicts uncertainty map")
                if has_uncertainty:
                    typed_uncertainty = float(metric.uncertainty)
                    if not math.isfinite(typed_uncertainty):
                        raise RuntimeError(
                            f"oracle response typed uncertainty {property_name} must be finite"
                        )
                    if typed_uncertainty < 0:
                        raise RuntimeError(
                            f"oracle response typed uncertainty {property_name} "
                            "must be non-negative"
                        )
                    if typed_uncertainty != uncertainties[property_name]:
                        raise RuntimeError("oracle response typed uncertainty contradicts map")
        elif scores or uncertainties or evaluation.metrics:
            raise RuntimeError(
                "oracle response ERROR scores, uncertainties, and metrics must be empty"
            )
        result[str(evaluation.molecule_smiles)] = {
            "scores": scores,
            "uncertainties": uncertainties,
            "success": success,
            "outcome": oracle_pb2.OracleOutcome.Name(outcome),
            "error_message": error_message,
            "error_code": error_code,
            "evidence_id": evidence_id,
            "oracle_name": str(evaluation.oracle_name),
            "oracle_version": str(evaluation.oracle_version),
            "model_version": str(evaluation.model_version),
            "artifact_refs": [
                {
                    "name": str(artifact.name),
                    "version": str(artifact.version),
                    "checksum": str(artifact.checksum),
                    "required": bool(artifact.required),
                }
                for artifact in evaluation.artifact_refs
            ],
        }
    return result


def _validate_wire_status(
    success: bool,
    outcome: int,
    error_code: str,
    error_message: str,
    *,
    error_message_present: bool,
) -> None:
    valid_success = (
        success
        and outcome == oracle_pb2.ORACLE_OUTCOME_PASS
        and not error_code
        and not error_message
        and not error_message_present
    )
    valid_error = (
        not success
        and outcome == oracle_pb2.ORACLE_OUTCOME_ERROR
        and bool(error_code.strip())
        and error_message_present
    )
    if not valid_success and not valid_error:
        raise RuntimeError("oracle response success/outcome/error fields are contradictory")


def _validated_wire_numeric_mapping(
    values: Mapping[str, float],
    field_name: str,
    *,
    non_negative: bool = False,
) -> dict[str, float]:
    output: dict[str, float] = {}
    for raw_name, raw_value in values.items():
        name = str(raw_name)
        number = float(raw_value)
        if not math.isfinite(number):
            raise RuntimeError(f"oracle response {field_name} {name} must be finite")
        if non_negative and number < 0:
            raise RuntimeError(f"oracle response {field_name} {name} must be non-negative")
        output[name] = number
    return output


def _validate_wire_artifact_refs(
    artifact_refs: object,
    *,
    require_nonempty: bool,
) -> None:
    refs = list(artifact_refs)
    if require_nonempty and not refs:
        raise RuntimeError("successful oracle response requires artifact_refs")
    names: set[str] = set()
    for artifact in refs:
        name = str(artifact.name).strip()
        if not name:
            raise RuntimeError("oracle response artifact name is required")
        if name in names:
            raise RuntimeError("oracle response artifact names must be unique")
        names.add(name)
        if not artifact.required:
            raise RuntimeError("oracle response artifact must be required")
        checksum = str(artifact.checksum)
        if _SHA256_CHECKSUM.fullmatch(checksum) is None:
            raise RuntimeError(
                "oracle response required artifact checksum must be sha256: plus 64 hex"
            )


def _expected_reported_oracle_names(
    logical_oracle_name: str,
    request_context: Mapping[str, Any],
) -> frozenset[str]:
    static_names = _STATIC_REPORTED_ORACLE_NAMES.get(logical_oracle_name)
    if static_names is not None:
        return static_names
    parameters = _mapping_or_empty(request_context.get("oracle_parameters"))
    parameter_name = {
        "dock": "engine",
        "fep": "method",
    }.get(logical_oracle_name)
    if parameter_name is None:
        return frozenset({logical_oracle_name})
    reported_name = str(parameters.get(parameter_name) or "").strip()
    if not reported_name:
        raise RuntimeError(
            f"{logical_oracle_name} logical oracle identity requires "
            f"oracle_parameters.{parameter_name}"
        )
    return frozenset({reported_name})


def _wire_outcome(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, int) and not isinstance(value, bool):
        try:
            return oracle_pb2.OracleOutcome.Name(value)
        except ValueError:
            return "INVALID"
    return str(value)


def _health_request_context(oracle_name: str) -> dict[str, Any] | None:
    context: dict[str, Any] = {
        "project_id": "validation-health",
        "request_id": f"validation-health:{oracle_name}",
    }
    if oracle_name == "boltz2":
        protein_pdb_id = os.environ.get(
            "VALIDATION_HEALTH_PROTEIN_PDB_ID",
            "",
        ).strip()
        if not protein_pdb_id:
            return None
        context["protein_pdb_id"] = protein_pdb_id
    elif oracle_name == "dock":
        receptor_uri = os.environ.get(
            "VALIDATION_HEALTH_RECEPTOR_URI",
            "",
        ).strip()
        engine = os.environ.get(
            "VALIDATION_HEALTH_DOCK_ENGINE",
            "",
        ).strip()
        if not receptor_uri or not engine:
            return None
        context["receptor_uri"] = receptor_uri
        context["oracle_parameters"] = {"engine": engine}
    elif oracle_name == "fep":
        protein_pdb_id = os.environ.get(
            "VALIDATION_HEALTH_PROTEIN_PDB_ID",
            "",
        ).strip()
        reference_ligand = os.environ.get(
            "VALIDATION_HEALTH_REFERENCE_LIGAND_SMILES",
            "",
        ).strip()
        method = os.environ.get(
            "VALIDATION_HEALTH_FEP_METHOD",
            "",
        ).strip()
        raw_n_repeats = os.environ.get(
            "VALIDATION_HEALTH_FEP_N_REPEATS",
            "",
        ).strip()
        try:
            n_repeats = int(raw_n_repeats)
        except ValueError:
            return None
        if not protein_pdb_id or not reference_ligand or not method or n_repeats <= 0:
            return None
        context["protein_pdb_id"] = protein_pdb_id
        context["reference_ligand_smiles"] = reference_ligand
        context["oracle_parameters"] = {
            "method": method,
            "n_repeats": str(n_repeats),
        }
    return context
