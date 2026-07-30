"""Scientific Critic Agent - internal adversary for bias prevention."""

import hashlib
import importlib
import inspect
import json
import logging
import math
import pkgutil
from collections.abc import Mapping
from pathlib import Path
from types import CodeType
from typing import Any

from mf_agents.base.agent import BaseAgent
from mf_agents.crg.graph import ChemicalReasoningGraph
from mf_core.db.repositories import build_shared_crg_repository_from_env

from critic_agent.rules.rule_base import CriticRule

_LOGGER = logging.getLogger(__name__)
_CRITIC_RESULT_SCHEMA_VERSION = "critic_result.v1"
_CRITIC_RESULT_PREDICATE = "critic_result"
_MISSING = object()


def _blocking_rule_ids(properties: dict) -> set[str] | None:
    value = properties.get("_critic_blocking_rule_ids")
    if value is None:
        return None
    if not isinstance(value, list | tuple | set):
        raise TypeError("_critic_blocking_rule_ids must be a list")
    return {str(item) for item in value}


def _is_blocking_rule(rule_id: str, blocking_rule_ids: set[str] | None) -> bool:
    return True if blocking_rule_ids is None else rule_id in blocking_rule_ids


def _critic_input_fingerprint(
    smiles: str,
    properties: dict,
    blocking_rule_ids: set[str] | None,
    rules: list,
    crg: Mapping[str, object],
) -> str:
    normalised_properties = dict(properties)
    if blocking_rule_ids is None:
        normalised_properties.pop("_critic_blocking_rule_ids", None)
    else:
        normalised_properties["_critic_blocking_rule_ids"] = sorted(blocking_rule_ids)
    payload = {
        "smiles": smiles,
        "properties": _canonical_cache_value(normalised_properties),
        "rules": [_rule_semantic_identity(rule) for rule in rules],
        "crg_failure_evidence": _critic_failure_evidence(crg, smiles),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _rule_semantic_identity(rule: object) -> dict:
    rule_type = type(rule)
    identity = {
        "rule_id": str(getattr(rule, "rule_id", "")),
        "rule_name": str(getattr(rule, "name", "")),
        "rule_type": f"{rule_type.__module__}.{rule_type.__qualname__}",
    }
    explicit_identity = getattr(rule, "cache_identity", _MISSING)
    if explicit_identity is not _MISSING:
        if callable(explicit_identity):
            explicit_identity = explicit_identity()
    constructor = rule_type.__init__
    identity["implementation"] = {
        "constructor": (
            "builtins.object.__init__"
            if constructor is object.__init__
            else _callable_implementation_digest(constructor)
        ),
        "evaluate": _callable_implementation_digest(rule.evaluate),
    }
    identity["configuration"] = (
        {"explicit": _canonical_cache_value(explicit_identity)}
        if explicit_identity is not _MISSING
        else _rule_constructor_configuration(rule)
    )
    return identity


def _rule_constructor_configuration(rule: object) -> dict:
    try:
        parameters = inspect.signature(type(rule).__init__).parameters.values()
    except (TypeError, ValueError):
        return {}
    crg = getattr(rule, "crg", _MISSING)
    configuration = {}
    for parameter in parameters:
        if parameter.name == "self" or parameter.kind in {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }:
            continue
        if parameter.name in {
            "calls",
            "call_count",
            "evaluation_count",
            "invocation_count",
            "crg",
        }:
            continue
        value = getattr(rule, parameter.name, _MISSING)
        if value is _MISSING or value is crg:
            continue
        configuration[parameter.name] = _canonical_cache_value(value)
    return configuration


def _callable_implementation_digest(callable_value: object) -> str:
    function = getattr(callable_value, "__func__", callable_value)
    try:
        source = inspect.getsource(function)
    except (OSError, TypeError):
        code = getattr(function, "__code__", None)
        if not isinstance(code, CodeType):
            raise TypeError(
                "critic rule callables must expose Python code or define cache_identity"
            ) from None
        implementation = {"code": _code_identity(code)}
    else:
        implementation = {"source": source}
    implementation["defaults"] = _canonical_cache_value(getattr(function, "__defaults__", None))
    implementation["kwdefaults"] = _canonical_cache_value(getattr(function, "__kwdefaults__", None))
    closure = getattr(function, "__closure__", None)
    if closure:
        free_variables = getattr(getattr(function, "__code__", None), "co_freevars", ())
        implementation["closure"] = {
            name: _implementation_closure_identity(cell.cell_contents)
            for name, cell in zip(free_variables, closure, strict=True)
            if name != "__class__"
        }
    encoded = json.dumps(
        implementation,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _implementation_closure_identity(value: object) -> object:
    if isinstance(value, type):
        return {"type": f"{value.__module__}.{value.__qualname__}"}
    return _canonical_cache_value(value)


def _code_identity(code: CodeType) -> dict:
    return {
        "bytecode": code.co_code.hex(),
        "constants": [_code_constant_identity(value) for value in code.co_consts],
        "names": list(code.co_names),
        "variables": list(code.co_varnames),
        "free_variables": list(code.co_freevars),
        "cell_variables": list(code.co_cellvars),
    }


def _code_constant_identity(value: object) -> object:
    if isinstance(value, CodeType):
        return {"code": _code_identity(value)}
    if value is Ellipsis:
        return {"constant": "ellipsis"}
    if isinstance(value, complex):
        return {"complex": [value.real, value.imag]}
    return _canonical_cache_value(value)


def _canonical_cache_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_cache_value(item)
            for key, item in sorted(
                value.items(),
                key=lambda pair: str(pair[0]),
            )
        }
    if isinstance(value, list | tuple):
        return [_canonical_cache_value(item) for item in value]
    if isinstance(value, set):
        items = [_canonical_cache_value(item) for item in value]
        return sorted(
            items,
            key=lambda item: json.dumps(
                item,
                sort_keys=True,
                separators=(",", ":"),
            ),
        )
    if isinstance(value, float):
        if math.isnan(value):
            return {"number": "nan"}
        if math.isinf(value):
            return {"number": "infinity" if value > 0 else "-infinity"}
        if value == 0:
            return 0.0
        return value
    if isinstance(value, bytes):
        return {"bytes_hex": value.hex()}
    if value is None or isinstance(value, str | int | bool):
        return value
    raise TypeError(f"critic properties contain unsupported value type: {type(value).__name__}")


def _critic_failure_evidence(
    crg: Mapping[str, object],
    smiles: str,
) -> list[dict]:
    evidence = []
    for belief in _latest_critic_evidence(crg, smiles):
        predicate = str(belief.get("predicate") or "")
        object_value = _belief_object_value(belief)
        if (
            (predicate == "validation_status" and object_value == "failed")
            or (predicate == "supply_feasibility" and object_value == "unavailable")
            or (predicate == "retrosyn_routes" and object_value == "0")
        ):
            evidence.append(
                {
                    "predicate": predicate,
                    "object_value": object_value,
                    "confidence": _belief_confidence(belief),
                }
            )
    return sorted(
        evidence,
        key=lambda item: json.dumps(
            item,
            sort_keys=True,
            separators=(",", ":"),
        ),
    )


def _latest_critic_evidence(
    crg: Mapping[str, object],
    smiles: str,
) -> list[Mapping[str, object]]:
    beliefs = crg.get("beliefs")
    latest_by_predicate: dict[str, tuple[tuple[int, str], Mapping[str, object]]] = {}
    for belief in beliefs if isinstance(beliefs, list) else []:
        if not isinstance(belief, Mapping):
            continue
        if str(belief.get("subject") or "") != smiles:
            continue
        predicate = str(belief.get("predicate") or "")
        if predicate not in {
            "validation_status",
            "supply_feasibility",
            "retrosyn_routes",
        }:
            continue
        order_key = _belief_order_key(belief)
        existing = latest_by_predicate.get(predicate)
        if existing is None or order_key > existing[0]:
            latest_by_predicate[predicate] = (order_key, belief)
    return [latest_by_predicate[predicate][1] for predicate in sorted(latest_by_predicate)]


def _belief_order_key(belief: Mapping[str, object]) -> tuple[int, str]:
    raw_timestamp = belief.get("timestamp_ns")
    timestamp_ns = 0 if raw_timestamp is None else int(raw_timestamp)
    deterministic_tiebreaker = json.dumps(
        _canonical_cache_value(
            {
                "id": str(belief.get("id") or ""),
                "object_value": _belief_object_value(belief),
                "confidence": _belief_confidence(belief),
                "source_agent": str(belief.get("source_agent") or ""),
            }
        ),
        sort_keys=True,
        separators=(",", ":"),
    )
    return timestamp_ns, deterministic_tiebreaker


def _belief_object_value(belief: Mapping[str, object]) -> str:
    return str(belief.get("object") or belief.get("object_value") or "").lower()


def _belief_confidence(belief: Mapping[str, object]) -> float:
    value = belief.get("confidence")
    return 1.0 if value is None else float(value)


def _is_cached_critic_result(
    result: object,
    smiles: str,
    expected_total_rules: int,
) -> bool:
    if not isinstance(result, dict):
        return False
    if result.get("smiles") != smiles:
        return False
    verdict = result.get("verdict")
    if verdict not in {"pass", "fail"}:
        return False
    counts = {}
    for key in (
        "passed",
        "failed",
        "blocking_failed",
        "non_blocking_failed",
        "total_rules",
    ):
        value = result.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return False
        counts[key] = value
    if counts["blocking_failed"] > counts["failed"]:
        return False
    if counts["total_rules"] != expected_total_rules:
        return False
    if counts["non_blocking_failed"] != counts["failed"] - counts["blocking_failed"]:
        return False
    rule_results = result.get("rule_results")
    if not isinstance(rule_results, list) or not all(
        isinstance(item, dict) for item in rule_results
    ):
        return False
    if len(rule_results) < counts["total_rules"]:
        return False
    passed = 0
    failed = 0
    blocking_failed = 0
    for rule_result in rule_results:
        if not str(rule_result.get("rule_id") or ""):
            return False
        rule_verdict = rule_result.get("verdict")
        if rule_verdict == "pass":
            passed += 1
            continue
        if rule_verdict not in {"fail", "error"}:
            return False
        blocking = rule_result.get("blocking")
        if not isinstance(blocking, bool):
            return False
        failed += 1
        if blocking:
            blocking_failed += 1
    if counts["passed"] != passed or counts["failed"] != failed:
        return False
    if counts["passed"] > counts["total_rules"]:
        return False
    if counts["blocking_failed"] != blocking_failed:
        return False
    if (verdict == "pass") != (blocking_failed == 0):
        return False
    return counts["non_blocking_failed"] == failed - blocking_failed


class _RuleRegistryTarget:
    def __init__(
        self,
        rules: list[CriticRule],
        load_failures: list[str] | None = None,
    ) -> None:
        self.rules = rules
        self.load_failures = list(load_failures or [])

    async def health_check(self) -> dict[str, bool]:
        rule_ids: list[str] = []
        for rule in self.rules:
            rule_id = getattr(rule, "rule_id", None)
            if (
                not isinstance(rule_id, str)
                or not rule_id.strip()
                or not callable(getattr(rule, "evaluate", None))
            ):
                return {"healthy": False}
            rule_ids.append(rule_id)
        return {
            "healthy": (
                bool(rule_ids) and not self.load_failures and len(rule_ids) == len(set(rule_ids))
            )
        }


class ScientificCriticAgent(BaseAgent):
    def __init__(self, message_bus=None, crg_repository: Any = None):
        super().__init__("critic_agent", message_bus)
        self._subscription_subjects = ["agent.critic.request", "orchestrator.critic.evaluate"]
        self.crg = ChemicalReasoningGraph()
        if crg_repository is None:
            self.crg_repository = build_shared_crg_repository_from_env()
            self._owns_crg_repository = self.crg_repository is not None
        else:
            self.crg_repository = crg_repository
            self._owns_crg_repository = False
        self.rules: list[CriticRule] = []
        self.rule_load_failures: list[str] = []
        self._load_rules()

    def runtime_targets(self) -> dict[str, object]:
        targets: dict[str, object] = {
            "critic_rules": _RuleRegistryTarget(
                self.rules,
                load_failures=self.rule_load_failures,
            )
        }
        if self._owns_crg_repository:
            targets["crg_repository"] = self.crg_repository
        return targets

    def _load_rules(self) -> None:
        """Auto-discover and load all rule classes from the rules package."""
        import critic_agent.rules as rules_pkg

        rules_path = Path(rules_pkg.__path__[0])
        for module_info in pkgutil.iter_modules([str(rules_path)]):
            if module_info.name == "rule_base" or module_info.name.startswith("_"):
                continue
            try:
                module = importlib.import_module(f"critic_agent.rules.{module_info.name}")
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (
                        isinstance(attr, type)
                        and issubclass(attr, CriticRule)
                        and attr is not CriticRule
                    ):
                        self.rules.append(attr(self.crg))
                        break
            except Exception as exc:
                failure = f"Could not load rule module {module_info.name}: {exc}"
                self.rule_load_failures.append(failure)
                _LOGGER.warning("%s", failure)

    async def evaluate_molecule(self, data: dict) -> dict:
        smiles = data.get("smiles", "")
        properties = data.get("properties", {})
        if not isinstance(properties, dict):
            raise TypeError("properties must be an object")
        results = []
        passed = 0
        failed = 0
        blocking_failed = 0
        run_id = str(data.get("run_id") or data.get("request_id") or "")
        blocking_rule_ids = _blocking_rule_ids(properties)
        shared_crg = await self._read_shared_crg_or_empty(run_id)
        input_fingerprint = _critic_input_fingerprint(
            smiles,
            properties,
            blocking_rule_ids,
            self.rules,
            shared_crg,
        )
        cached_result = self._cached_critic_result(
            shared_crg,
            smiles,
            input_fingerprint,
        )
        if cached_result is not None:
            return cached_result

        for rule in self.rules:
            try:
                verdict = rule.evaluate(smiles, properties)
                if verdict.get("verdict") != "pass":
                    verdict["blocking"] = _is_blocking_rule(
                        str(verdict.get("rule_id") or rule.rule_id),
                        blocking_rule_ids,
                    )
                results.append(verdict)
                if verdict.get("verdict") == "pass":
                    passed += 1
                else:
                    failed += 1
                    if bool(verdict.get("blocking", True)):
                        blocking_failed += 1
            except Exception as e:
                results.append(
                    {
                        "rule_id": rule.rule_id,
                        "rule_name": rule.name,
                        "verdict": "error",
                        "score": 0.0,
                        "reasoning": str(e),
                        "blocking": _is_blocking_rule(rule.rule_id, blocking_rule_ids),
                    }
                )
                failed += 1
                if bool(results[-1].get("blocking", True)):
                    blocking_failed += 1

        crg_results = self._shared_crg_failure_results(
            shared_crg,
            smiles,
            blocking_rule_ids=blocking_rule_ids,
        )
        results.extend(crg_results)
        failed += len(crg_results)
        blocking_failed += sum(1 for result in crg_results if bool(result.get("blocking", True)))
        workflow_results = _workflow_failure_results(
            properties,
            blocking_rule_ids,
        )
        results.extend(workflow_results)
        failed += len(workflow_results)
        blocking_failed += sum(
            1 for result in workflow_results if bool(result.get("blocking", True))
        )

        overall_verdict = "pass" if blocking_failed == 0 else "fail"
        total_evidence = len(results)
        result = {
            "smiles": smiles,
            "verdict": overall_verdict,
            "passed": passed,
            "failed": failed,
            "blocking_failed": blocking_failed,
            "non_blocking_failed": max(0, failed - blocking_failed),
            "total_rules": len(self.rules),
            "rule_results": results,
        }
        evidence_ids = [str(item["rule_id"]) for item in results if item.get("rule_id")]
        confidence = passed / total_evidence if total_evidence else 1.0
        verdict_belief = self.crg.add_belief(
            subject=smiles,
            predicate="critic_verdict",
            obj=overall_verdict,
            confidence=confidence,
            source_agent=self.name,
            evidence_ids=evidence_ids,
        )
        result_belief = self.crg.add_belief(
            subject=smiles,
            predicate=_CRITIC_RESULT_PREDICATE,
            obj=json.dumps(
                {
                    "schema_version": _CRITIC_RESULT_SCHEMA_VERSION,
                    "input_fingerprint": input_fingerprint,
                    "result": result,
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            confidence=confidence,
            source_agent=self.name,
            evidence_ids=evidence_ids,
        )
        for belief in (verdict_belief, result_belief):
            await self._persist_belief(
                belief,
                project_id=str(data.get("project_id") or ""),
                run_id=run_id,
            )
        return result

    async def process(self, data):
        identity = _full_workflow_identity(data) if data.get("workflow_scope") == "full" else None
        result = await self.evaluate_molecule(data)
        if identity is None:
            return result
        return {
            **result,
            **identity,
        }

    async def _read_shared_crg_or_empty(
        self,
        run_id: str,
    ) -> dict:
        if not run_id or self.crg_repository is None:
            return {"beliefs": []}
        read_crg = getattr(self.crg_repository, "get_run_crg", None)
        if not callable(read_crg):
            return {"beliefs": []}
        return await self.read_shared_crg(run_id)

    def _cached_critic_result(
        self,
        crg: Mapping[str, object],
        smiles: str,
        input_fingerprint: str,
    ) -> dict[str, Any] | None:
        beliefs = crg.get("beliefs")
        if not isinstance(beliefs, list):
            return None
        for belief in reversed(beliefs):
            if not isinstance(belief, dict):
                continue
            if str(belief.get("subject") or "") != smiles:
                continue
            predicate = str(belief.get("predicate") or "")
            if predicate != _CRITIC_RESULT_PREDICATE:
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
            if contract.get("schema_version") != _CRITIC_RESULT_SCHEMA_VERSION:
                continue
            if contract.get("input_fingerprint") != input_fingerprint:
                continue
            result = contract.get("result")
            if not _is_cached_critic_result(result, smiles, len(self.rules)):
                continue
            return {**result, "cache_source": "shared_crg"}
        return None

    def _shared_crg_failure_results(
        self,
        crg: Mapping[str, object],
        smiles: str,
        *,
        blocking_rule_ids: set[str] | None = None,
    ) -> list[dict]:
        results = []
        for belief in _latest_critic_evidence(crg, smiles):
            predicate = str(belief.get("predicate") or "")
            object_value = _belief_object_value(belief)
            if predicate == "validation_status" and object_value == "failed":
                results.append(
                    {
                        "rule_id": "crg_validation_status",
                        "rule_name": "Shared CRG validation status",
                        "verdict": "fail",
                        "score": _belief_confidence(belief),
                        "reasoning": "shared CRG contains failed validation_status",
                        "blocking": _is_blocking_rule(
                            "crg_validation_status",
                            blocking_rule_ids,
                        ),
                    }
                )
            elif predicate == "supply_feasibility" and object_value == "unavailable":
                results.append(
                    {
                        "rule_id": "crg_supply_feasibility",
                        "rule_name": "Shared CRG supply feasibility",
                        "verdict": "fail",
                        "score": _belief_confidence(belief),
                        "reasoning": "shared CRG contains unavailable supply_feasibility",
                        "blocking": _is_blocking_rule(
                            "crg_supply_feasibility",
                            blocking_rule_ids,
                        ),
                    }
                )
            elif predicate == "retrosyn_routes" and object_value == "0":
                results.append(
                    {
                        "rule_id": "crg_retrosyn_routes",
                        "rule_name": "Shared CRG retrosynthesis routes",
                        "verdict": "fail",
                        "score": _belief_confidence(belief),
                        "reasoning": "shared CRG contains zero retrosyn_routes",
                        "blocking": _is_blocking_rule(
                            "crg_retrosyn_routes",
                            blocking_rule_ids,
                        ),
                    }
                )
        return results

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


def _workflow_failure_results(
    properties: Mapping[str, object],
    blocking_rule_ids: set[str] | None,
) -> list[dict]:
    checks = (
        (
            "retrosyn_route_count",
            "workflow_retrosyn_routes",
            "Retrosynthesis route count",
            "no retrosynthesis route was produced",
        ),
        (
            "srb_protocol_count",
            "workflow_srb_protocols",
            "Synthesis protocol count",
            "no synthesis protocol could be compiled",
        ),
    )
    results = []
    for property_name, rule_id, rule_name, reasoning in checks:
        if property_name not in properties:
            continue
        value = properties[property_name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise TypeError(f"{property_name} must be a non-negative integer")
        if value == 0:
            results.append(
                {
                    "rule_id": rule_id,
                    "rule_name": rule_name,
                    "verdict": "fail",
                    "score": 0.0,
                    "reasoning": reasoning,
                    "blocking": _is_blocking_rule(rule_id, blocking_rule_ids),
                }
            )
    if "supply_feasibility" in properties:
        feasibility = properties["supply_feasibility"]
        if not isinstance(feasibility, str):
            raise TypeError("supply_feasibility must be a string")
        if feasibility.lower() in {"unavailable", "not_assessed"}:
            rule_id = "workflow_supply_feasibility"
            results.append(
                {
                    "rule_id": rule_id,
                    "rule_name": "Supply feasibility",
                    "verdict": "fail",
                    "score": 0.0,
                    "reasoning": f"supply assessment is {feasibility.lower()}",
                    "blocking": _is_blocking_rule(rule_id, blocking_rule_ids),
                }
            )
    return results


def _full_workflow_identity(data: Mapping[str, object]) -> dict[str, object]:
    identity: dict[str, object] = {}
    for field in (
        "project_id",
        "run_id",
        "request_id",
        "schema_version",
        "candidate_id",
        "canonical_smiles",
    ):
        value = data.get(field)
        if not isinstance(value, str) or not value or value != value.strip():
            raise ValueError(f"{field} must be a non-empty trimmed string")
        identity[field] = value
    if identity["schema_version"] != "critic.request.v1":
        raise ValueError("schema_version must be critic.request.v1")
    candidate_index = data.get("candidate_index")
    if (
        isinstance(candidate_index, bool)
        or not isinstance(candidate_index, int)
        or candidate_index < 0
    ):
        raise ValueError("candidate_index must be a non-negative integer")
    identity["candidate_index"] = candidate_index
    if identity["canonical_smiles"] != data.get("smiles"):
        raise ValueError("canonical_smiles must match smiles")
    return {
        field: identity[field]
        for field in (
            "project_id",
            "candidate_id",
            "candidate_index",
            "canonical_smiles",
        )
    }
