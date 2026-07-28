"""Validation Agent - Adaptive Oracle Cascade L0-L4 (Agent-4)."""

from __future__ import annotations

import asyncio
import inspect
import json
import math
import os
import shlex
import subprocess
from typing import Any

from mf_agents.base.agent import (
    BaseAgent,
    agent_health_check_timeout_seconds,
    close_owned_channel,
    ensure_default_event_loop,
    run_health_probe_in_daemon,
)
from mf_agents.crg.graph import ChemicalReasoningGraph
from mf_core.artifacts import CommandRequirement, check_command, require_available
from mf_core.db.repositories import build_shared_crg_repository_from_env
from mf_core.proto_gen.moleculeforge.v1.oracle import oracle_pb2, oracle_pb2_grpc

_L4_QUANTUM_COMMAND = CommandRequirement(
    "l4_quantum_oracle_command",
    "L4_QUANTUM_ORACLE_COMMAND",
)
_ORACLE_LEVELS = {
    0: ("filter", ("admet_score",)),
    1: ("docking", ("docking_score",)),
    2: ("affinity", ("affinity",)),
    3: ("rbfe", ("rbfe",)),
    4: ("quantum", ("quantum_correction",)),
}


class OracleGrpcClient:
    def __init__(
        self,
        target: str,
        level: int,
        oracle_name: str,
        *,
        health_level: int | None = None,
    ) -> None:
        self.target = target
        self.level = level
        self.health_level = level if health_level is None else health_level
        self.oracle_name = oracle_name
        self.channel = None
        self.stub = None
        self._closed = False

    async def evaluate(self, molecules: list[str], properties: list[str]) -> dict:
        response = await self._stub().Evaluate(
            _oracle_batch_request(molecules, properties, self.level)
        )
        return _scores_by_smiles(response)

    async def predict_with_uncertainty(
        self,
        molecules: list[str],
        properties: list[str],
    ) -> dict:
        response = await self._stub().PredictWithUncertainty(
            _oracle_batch_request(
                molecules,
                properties,
                self.level,
                return_uncertainty=True,
            )
        )
        return _scores_and_uncertainty_by_smiles(response)

    async def health_check(self) -> dict[str, bool]:
        required_properties = _oracle_required_properties(getattr(self, "health_level", self.level))
        response = await self._stub().Evaluate(
            _oracle_batch_request(["C"], required_properties, self.level),
            timeout=agent_health_check_timeout_seconds(),
        )
        return {
            "healthy": _oracle_result_is_healthy(
                _scores_by_smiles(response),
                required_properties,
            )
        }

    def _stub(self):
        if self.stub is None:
            import grpc

            ensure_default_event_loop()
            self.channel = grpc.aio.insecure_channel(self.target)
            self.stub = oracle_pb2_grpc.OracleServiceStub(self.channel)
        return self.stub

    async def close(self) -> None:
        await close_owned_channel(self, self.channel)


class QuantumCommandOracle:
    def __init__(
        self,
        command: str | list[str],
        engine: str = "quantum",
        run_command=None,
    ) -> None:
        self.command = shlex.split(command) if isinstance(command, str) else list(command)
        if not self.command:
            raise ValueError("L4 quantum oracle command must not be empty")
        self.engine = engine or "quantum"
        self.oracle_name = "quantum"
        self.run_command = run_command or subprocess.run
        self._uses_default_runner = run_command is None
        self.timeout_seconds = float(os.environ.get("L4_QUANTUM_ORACLE_TIMEOUT_SECONDS", "300"))

    async def evaluate(self, molecules: list[str], properties: list[str]) -> dict[str, dict]:
        results = {}
        for smiles in molecules:
            payload = {
                "molecule_smiles": smiles,
                "requested_properties": list(properties),
                "engine": self.engine,
            }
            completed = await self._run(
                payload,
                timeout=self.timeout_seconds,
            )
            if getattr(completed, "returncode", 0) != 0:
                stderr = getattr(completed, "stderr", "")
                raise RuntimeError(f"L4 quantum command failed for {smiles}: {stderr}")
            results[smiles] = _quantum_command_scores(
                getattr(completed, "stdout", ""),
                smiles,
                properties,
            )
        return results

    async def health_check(self) -> dict[str, bool]:
        payload = {
            "molecule_smiles": "C",
            "requested_properties": ["quantum_correction"],
            "engine": self.engine,
        }
        completed = await self._run(
            payload,
            timeout=agent_health_check_timeout_seconds(),
        )
        if getattr(completed, "returncode", 0) != 0:
            return {"healthy": False}
        result = _quantum_command_scores(
            getattr(completed, "stdout", ""),
            "C",
            ["quantum_correction"],
        )
        return {
            "healthy": _oracle_result_is_healthy(
                {"C": result},
                _oracle_required_properties(4),
            )
        }

    async def _run(self, payload: dict, *, timeout: float):
        if self._uses_default_runner:
            _require_command_available(_L4_QUANTUM_COMMAND, shlex.join(self.command))
        kwargs = {
            "check": False,
            "capture_output": True,
            "text": True,
            "input": json.dumps(payload, sort_keys=True),
        }
        if self._uses_default_runner:
            kwargs["timeout"] = timeout
        return await asyncio.wait_for(
            asyncio.to_thread(self.run_command, self.command, **kwargs),
            timeout=timeout,
        )


def _require_command_available(
    requirement: CommandRequirement,
    command: str,
) -> None:
    env = {**os.environ, requirement.env_var: command}
    require_available([check_command(requirement, env=env)])


class ValidationAgent(BaseAgent):
    def __init__(
        self,
        message_bus=None,
        oracles: dict | None = None,
        crg_repository: Any = None,
    ):
        super().__init__("validation_agent", message_bus)
        self._subscription_subjects = ["agent.validation.request", "orchestrator.validate.check"]
        self.crg = ChemicalReasoningGraph()
        self.oracles = dict(oracles) if oracles is not None else _build_default_oracles()
        if crg_repository is None:
            self.crg_repository = build_shared_crg_repository_from_env()
            self._owns_crg_repository = self.crg_repository is not None
        else:
            self.crg_repository = crg_repository
            self._owns_crg_repository = False
        self.oracle_levels = {
            level: (oracle_name, list(properties))
            for level, (oracle_name, properties) in _ORACLE_LEVELS.items()
        }
        self.default_thresholds = {
            1: ("docking_score", "max", -6.0, "l1_max_docking_score"),
            2: ("affinity", "max", -7.0, "l2_max_affinity"),
            3: ("rbfe", "max", 0.0, "l3_max_rbfe"),
            4: ("quantum_correction", "max", 0.0, "l4_max_quantum_correction"),
        }
        self.default_uncertainty_thresholds = {
            1: ("docking_score", 1.0, "l1_max_uncertainty"),
            2: ("affinity", 1.0, "l2_max_uncertainty"),
            3: ("rbfe", 1.0, "l3_max_uncertainty"),
            4: ("quantum_correction", 1.0, "l4_max_uncertainty"),
        }

    def runtime_targets(self) -> dict[str, object | None]:
        targets: dict[str, object | None] = {
            f"oracle.L{level}": (
                self.oracles[level] if level in self.oracles else self.oracles.get(f"L{level}")
            )
            for level in range(5)
        }
        if self._owns_crg_repository:
            targets["crg_repository"] = self.crg_repository
        return targets

    async def process(self, data):
        """Run adaptive oracle cascade from L0 to requested level.

        Progressively validates molecules through increasingly expensive
        oracles. Early termination on failure at any level for efficiency.
        """
        max_level = _requested_oracle_level(data)
        smiles = data.get("smiles", "")
        if not smiles:
            raise ValueError("smiles is required")
        l0_threshold = _finite_float(
            data.get("l0_threshold", 0.0),
            "l0_threshold",
            error_type=ValueError,
        )
        validation_policy = self._validation_policy(data, max_level, l0_threshold)
        cached_result = await self._validation_result_from_shared_crg(
            data,
            smiles,
            validation_policy,
        )
        if cached_result is not None:
            cached_result["cached"] = True
            cached_result["cache_source"] = "shared_crg"
            return cached_result
        cascade_results = {}
        upgrade_path = []
        overall_passed = True
        for level in range(max_level + 1):
            level_config = self.oracle_levels.get(level)
            if not isinstance(level_config, tuple):
                raise RuntimeError(f"Oracle level L{level} is not configured")
            oracle_name, properties = level_config
            oracle = self._oracle_for_level(level)
            values, uncertainty = await self._run_oracle(oracle, smiles, properties)
            thresholds = self._thresholds_for_level(level, data, l0_threshold)
            passed = self._level_passed(level, values, uncertainty, thresholds)
            key = f"L{level}_{oracle_name}"
            cascade_results[key] = {
                "completed": True,
                "passed": passed,
                "result": values,
                "uncertainty": uncertainty,
                "thresholds": thresholds,
            }
            if values.get("skipped") is True:
                cascade_results[key]["skipped"] = True
                if "skip_reason" in values:
                    cascade_results[key]["skip_reason"] = values["skip_reason"]
            upgrade_path.append(f"L{level}")
            if not passed:
                overall_passed = False
                break
        status = "validated" if overall_passed else "failed"
        result = {
            "agent": self.name,
            "status": status,
            "smiles": smiles,
            "max_oracle_level": len(upgrade_path) - 1,
            "requested_oracle_level": max_level,
            "cascade": cascade_results,
            "upgrade_path": upgrade_path,
            "overall_passed": overall_passed,
        }
        status_belief = self.crg.add_belief(
            subject=smiles,
            predicate="validation_status",
            obj=status,
            confidence=1.0,
            source_agent=self.name,
            evidence_ids=list(cascade_results.keys()),
        )
        result_belief = self.crg.add_belief(
            subject=smiles,
            predicate="validation_result",
            obj=json.dumps(
                {
                    "schema_version": "validation_result.v1",
                    "validation_policy": validation_policy,
                    "result": result,
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            confidence=1.0,
            source_agent=self.name,
            evidence_ids=list(cascade_results.keys()),
        )
        project_id = str(data.get("project_id") or "")
        run_id = str(data.get("run_id") or data.get("request_id") or "")
        for belief in (status_belief, result_belief):
            await self._persist_belief(
                belief,
                project_id=project_id,
                run_id=run_id,
            )
        return result

    def _validation_policy(
        self,
        data: dict,
        max_level: int,
        l0_threshold: float,
    ) -> dict[str, Any]:
        return {
            "oracle_level": max_level,
            "thresholds": {
                f"L{level}": self._thresholds_for_level(
                    level,
                    data,
                    l0_threshold,
                )
                for level in range(max_level + 1)
            },
        }

    async def _validation_result_from_shared_crg(
        self,
        data: dict,
        smiles: str,
        validation_policy: dict[str, Any],
    ) -> dict[str, Any] | None:
        run_id = str(data.get("run_id") or data.get("request_id") or "")
        if (
            not run_id
            or self.crg_repository is None
            or not callable(getattr(self.crg_repository, "get_run_crg", None))
        ):
            return None
        crg = await self.read_shared_crg(run_id)
        for belief in reversed(crg.get("beliefs", []) or []):
            if not isinstance(belief, dict):
                continue
            if str(belief.get("subject") or "") != smiles:
                continue
            if str(belief.get("predicate") or "") != "validation_result":
                continue
            if str(belief.get("source_agent") or "") != self.name:
                continue
            raw_contract = belief.get("object_value", belief.get("object"))
            if not isinstance(raw_contract, str):
                continue
            try:
                contract = json.loads(raw_contract)
            except json.JSONDecodeError:
                continue
            if not isinstance(contract, dict):
                continue
            if contract.get("schema_version") != "validation_result.v1":
                continue
            if contract.get("validation_policy") != validation_policy:
                continue
            result = contract.get("result")
            if _is_cached_validation_result(
                result,
                smiles,
                validation_policy,
                self._level_passed,
            ):
                return dict(result)
        return None

    async def _persist_belief(self, belief, project_id: str, run_id: str) -> None:
        if self.crg_repository is None:
            return
        write_belief = getattr(self.crg_repository, "write_workflow_belief", None)
        if not callable(write_belief):
            raise TypeError("crg_repository must expose write_workflow_belief(**kwargs)")
        result = write_belief(
            project_id=project_id,
            run_id=run_id or belief.subject,
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

    def _oracle_for_level(self, level: int):
        for key in (level, f"L{level}"):
            if key in self.oracles:
                return self.oracles[key]
        raise RuntimeError(f"Oracle level L{level} runner is not configured")

    async def _run_oracle(
        self,
        oracle,
        smiles: str,
        properties: list[str],
    ) -> tuple[dict, dict | None]:
        if hasattr(oracle, "predict_with_uncertainty"):
            result = oracle.predict_with_uncertainty([smiles], properties)
            if inspect.isawaitable(result):
                result = await result
            values, uncertainty = _extract_uncertainty_result(result, smiles)
            return values, uncertainty
        result = oracle.evaluate([smiles], properties)
        if inspect.isawaitable(result):
            result = await result
        return result[smiles], None

    def _thresholds_for_level(self, level: int, data: dict, l0_threshold: float) -> dict:
        if level == 0:
            return {"admet_score_min": l0_threshold}
        score_name, mode, default_value, config_key = self.default_thresholds[level]
        uncertainty_name, uncertainty_default, uncertainty_key = (
            self.default_uncertainty_thresholds[level]
        )
        return {
            f"{score_name}_{mode}": _finite_float(
                data.get(config_key, default_value),
                config_key,
                error_type=ValueError,
            ),
            f"{uncertainty_name}_uncertainty_max": _finite_float(
                data.get(uncertainty_key, uncertainty_default),
                uncertainty_key,
                error_type=ValueError,
            ),
        }

    def _level_passed(
        self,
        level: int,
        values: dict,
        uncertainty: dict | None,
        thresholds: dict,
    ) -> bool:
        if not isinstance(values, dict):
            raise RuntimeError(f"L{level} oracle result must be an object")
        if level == 0:
            score_threshold = _finite_float(
                thresholds.get("admet_score_min"),
                "admet_score_min",
            )
            if values.get("skipped") is True:
                return True
            score = _finite_float(
                values.get("admet_score"),
                "L0 admet_score",
            )
            return score >= score_threshold
        score_name, mode, _default_value, _config_key = self.default_thresholds[level]
        score_threshold = _finite_float(
            thresholds.get(f"{score_name}_{mode}"),
            f"{score_name}_{mode}",
        )
        uncertainty_name, _uncertainty_default, _uncertainty_key = (
            self.default_uncertainty_thresholds[level]
        )
        uncertainty_threshold = _finite_float(
            thresholds.get(f"{uncertainty_name}_uncertainty_max"),
            f"{uncertainty_name}_uncertainty_max",
        )
        if values.get("skipped") is True:
            return True
        score = _finite_float(
            values.get(score_name),
            f"L{level} {score_name}",
        )
        if mode == "max" and score > score_threshold:
            return False
        if uncertainty is None:
            return True
        if not isinstance(uncertainty, dict):
            raise RuntimeError(f"L{level} uncertainty must be an object")
        uncertainty_value = uncertainty.get(uncertainty_name)
        if uncertainty_value is None:
            return True
        return (
            _finite_float(
                uncertainty_value,
                f"L{level} {uncertainty_name} uncertainty",
            )
            <= uncertainty_threshold
        )


def _requested_oracle_level(data: dict) -> int:
    level = data.get("oracle_level", 0)
    if isinstance(level, bool) or not isinstance(level, int) or level not in _ORACLE_LEVELS:
        raise ValueError("oracle_level must be an integer between 0 and 4")
    return level


def _finite_float(
    value: object,
    field: str,
    *,
    error_type: type[Exception] = RuntimeError,
) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise error_type(f"{field} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise error_type(f"{field} must be a finite number")
    return number


def _is_cached_validation_result(
    result: object,
    smiles: str,
    validation_policy: dict[str, Any],
    level_evaluator,
) -> bool:
    if not isinstance(result, dict):
        return False
    if result.get("agent") != "validation_agent":
        return False
    if result.get("smiles") != smiles:
        return False
    status = result.get("status")
    if status not in {"validated", "failed"}:
        return False
    overall_passed = result.get("overall_passed")
    if not isinstance(overall_passed, bool):
        return False
    if (status == "validated") != overall_passed:
        return False
    requested_level = validation_policy.get("oracle_level")
    if isinstance(requested_level, bool) or not isinstance(requested_level, int):
        return False
    cached_requested_level = result.get("requested_oracle_level")
    if (
        isinstance(cached_requested_level, bool)
        or not isinstance(cached_requested_level, int)
        or cached_requested_level != requested_level
    ):
        return False
    cascade = result.get("cascade")
    if not isinstance(cascade, dict) or not cascade:
        return False
    upgrade_path = result.get("upgrade_path")
    if not isinstance(upgrade_path, list) or not upgrade_path:
        return False
    executed_level = result.get("max_oracle_level")
    if (
        isinstance(executed_level, bool)
        or not isinstance(executed_level, int)
        or executed_level < 0
        or executed_level > requested_level
    ):
        return False
    if overall_passed and executed_level != requested_level:
        return False
    if upgrade_path != [f"L{level}" for level in range(executed_level + 1)]:
        return False
    expected_keys = [f"L{level}_{_ORACLE_LEVELS[level][0]}" for level in range(executed_level + 1)]
    if set(cascade) != set(expected_keys):
        return False
    policy_thresholds = validation_policy.get("thresholds")
    if not isinstance(policy_thresholds, dict):
        return False
    level_passed = []
    for level, key in enumerate(expected_keys):
        level_result = cascade.get(key)
        if not isinstance(level_result, dict):
            return False
        if level_result.get("completed") is not True:
            return False
        passed = level_result.get("passed")
        if not isinstance(passed, bool):
            return False
        values = level_result.get("result")
        if not isinstance(values, dict):
            return False
        uncertainty = level_result.get("uncertainty")
        if uncertainty is not None and not isinstance(uncertainty, dict):
            return False
        thresholds = level_result.get("thresholds")
        if thresholds != policy_thresholds.get(f"L{level}"):
            return False
        try:
            expected_passed = level_evaluator(
                level,
                values,
                uncertainty,
                thresholds,
            )
        except (KeyError, TypeError, ValueError, RuntimeError):
            return False
        if passed != expected_passed:
            return False
        level_passed.append(passed)
    if not all(level_passed[:-1]):
        return False
    return level_passed[-1] == overall_passed


def _extract_uncertainty_result(result, smiles: str) -> tuple[dict, dict | None]:
    if isinstance(result, dict) and smiles in result:
        item = result[smiles]
    else:
        item = result
    if not isinstance(item, tuple) or len(item) != 2:
        raise RuntimeError("predict_with_uncertainty must return (scores, uncertainty)")
    values, uncertainty = item
    if not isinstance(values, dict):
        raise RuntimeError("predict_with_uncertainty scores must be a dict")
    if uncertainty is not None and not isinstance(uncertainty, dict):
        raise RuntimeError("predict_with_uncertainty uncertainty must be a dict")
    return values, uncertainty


def _build_default_oracles() -> dict[int, object]:
    from mf_oracles.boltz2.oracle import Boltz2Oracle
    from mf_oracles.gnina.oracle import GninaOracle
    from mf_oracles.openfe.oracle import OpenFEOracle
    from mf_oracles.rdkit_oracle.oracle import RDKitOracle

    oracles = {
        0: _BatchEvaluateOnlyOracle(RDKitOracle(), level=0),
        1: _BatchEvaluateOnlyOracle(GninaOracle(), level=1),
        2: _BatchEvaluateOnlyOracle(Boltz2Oracle(), level=2),
        3: _BatchEvaluateOnlyOracle(OpenFEOracle(), level=3),
    }
    l0_admet_target = os.environ.get("L0_ADMET_ORACLE_TARGET", "")
    if l0_admet_target:
        oracles[0] = OracleGrpcClient(
            l0_admet_target,
            level=1,
            oracle_name="admet_ai",
            health_level=0,
        )
    for level, env_var, oracle_name in (
        (1, "L1_DOCKING_ORACLE_TARGET", "docking"),
        (2, "L2_AFFINITY_ORACLE_TARGET", "affinity"),
        (3, "L3_FEP_ORACLE_TARGET", "rbfe"),
    ):
        target = os.environ.get(env_var, "")
        if target:
            oracles[level] = OracleGrpcClient(target, level=level, oracle_name=oracle_name)
    l4_target = os.environ.get("L4_QUANTUM_ORACLE_TARGET", "")
    if l4_target:
        oracles[4] = OracleGrpcClient(l4_target, level=4, oracle_name="quantum")
    else:
        l4_command, l4_engine = _l4_quantum_command_from_env()
        if l4_command:
            oracles[4] = QuantumCommandOracle(
                l4_command,
                engine=l4_engine,
            )
    return oracles


def _l4_quantum_command_from_env() -> tuple[str, str]:
    generic_command = os.environ.get("L4_QUANTUM_ORACLE_COMMAND", "").strip()
    if generic_command:
        return generic_command, os.environ.get("L4_QUANTUM_ENGINE", "quantum")
    for env_var, engine in (
        ("L4_GPU4PYSCF_COMMAND", "gpu4pyscf"),
        ("L4_ORCA_COMMAND", "orca"),
    ):
        command = os.environ.get(env_var, "").strip()
        if command:
            return command, engine
    return "", "quantum"


class _BatchEvaluateOnlyOracle:
    def __init__(self, oracle: object, *, level: int = 0) -> None:
        self.oracle = oracle
        self.level = level

    @property
    def _close_target(self) -> object:
        return self.oracle

    async def evaluate(self, molecules: list[str], properties: list[str]) -> dict:
        return await run_health_probe_in_daemon(
            lambda: _run_oracle_evaluate(self.oracle, molecules, properties)
        )

    async def health_check(self) -> dict[str, bool]:
        required_properties = _oracle_required_properties(self.level)
        result = await self.evaluate(["C"], required_properties)
        return {
            "healthy": _oracle_result_is_healthy(
                result,
                required_properties,
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
    try:
        return list(_ORACLE_LEVELS[level][1])
    except KeyError as exc:
        raise ValueError(f"unsupported Oracle health level: {level}") from exc


def _oracle_result_is_healthy(
    result: object,
    required_properties: list[str],
) -> bool:
    if not isinstance(result, dict):
        return False
    scores = result.get("C")
    if not isinstance(scores, dict):
        return False
    return all(
        isinstance(scores.get(prop), int | float)
        and not isinstance(scores[prop], bool)
        and math.isfinite(float(scores[prop]))
        for prop in required_properties
    )


def _oracle_batch_request(
    molecules: list[str],
    properties: list[str],
    level: int,
    return_uncertainty: bool = False,
) -> oracle_pb2.OracleBatchRequest:
    return oracle_pb2.OracleBatchRequest(
        molecule_smiles=[str(item) for item in molecules],
        requested_properties=[str(item) for item in properties],
        level=_oracle_level_proto(level),
        return_uncertainty=return_uncertainty,
    )


def _oracle_level_proto(level: int) -> int:
    return {
        0: oracle_pb2.L0_RDKIT,
        1: oracle_pb2.L1_ML_SURROGATE,
        2: oracle_pb2.L2_DOCKING,
        3: oracle_pb2.L3_FEP,
        4: oracle_pb2.L4_WETLAB,
    }[level]


def _scores_by_smiles(response) -> dict[str, dict]:
    scores = {}
    for evaluation in response.evaluations:
        if not evaluation.success:
            raise RuntimeError(evaluation.error_message or "oracle evaluation failed")
        scores[str(evaluation.molecule_smiles)] = {
            str(key): float(value) for key, value in evaluation.scores.items()
        }
    return scores


def _scores_and_uncertainty_by_smiles(response) -> dict[str, tuple[dict, dict]]:
    values = {}
    for evaluation in response.evaluations:
        if not evaluation.success:
            raise RuntimeError(evaluation.error_message or "oracle evaluation failed")
        values[str(evaluation.molecule_smiles)] = (
            {str(key): float(value) for key, value in evaluation.scores.items()},
            {str(key): float(value) for key, value in evaluation.uncertainties.items()},
        )
    return values


def _quantum_command_scores(stdout: str, smiles: str, properties: list[str]) -> dict:
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"L4 quantum command returned invalid JSON for {smiles}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"L4 quantum command returned non-object JSON for {smiles}")
    score_payload = payload.get("scores", payload)
    if not isinstance(score_payload, dict):
        raise RuntimeError(f"L4 quantum command scores must be an object for {smiles}")
    values: dict[str, Any] = {}
    for prop in properties:
        if prop not in score_payload:
            raise RuntimeError(f"L4 quantum command result for {smiles} requires {prop}")
        values[prop] = float(score_payload[prop])
    uncertainty = payload.get("uncertainty", payload.get("uncertainties", {}))
    if isinstance(uncertainty, dict):
        for prop, value in uncertainty.items():
            values[f"uncertainty_{prop}"] = float(value)
    if "engine" in payload:
        values["engine"] = str(payload["engine"])
    return values
