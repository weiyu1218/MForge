"""Scientific Critic Agent - internal adversary for bias prevention."""

import importlib
import inspect
import logging
import pkgutil
from pathlib import Path
from typing import Any

from mf_agents.base.agent import BaseAgent
from mf_agents.crg.graph import ChemicalReasoningGraph
from mf_core.db.repositories import build_shared_crg_repository_from_env

from critic_agent.rules.rule_base import CriticRule

_LOGGER = logging.getLogger(__name__)


def _blocking_rule_ids(properties: dict) -> set[str] | None:
    value = properties.get("_critic_blocking_rule_ids")
    if value is None:
        return None
    if not isinstance(value, list | tuple | set):
        raise TypeError("_critic_blocking_rule_ids must be a list")
    return {str(item) for item in value}


def _is_blocking_rule(rule_id: str, blocking_rule_ids: set[str] | None) -> bool:
    return True if blocking_rule_ids is None else rule_id in blocking_rule_ids


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
        self.crg_repository = (
            crg_repository if crg_repository is not None else build_shared_crg_repository_from_env()
        )
        self.rules: list[CriticRule] = []
        self.rule_load_failures: list[str] = []
        self._load_rules()

    def runtime_targets(self) -> dict[str, object]:
        return {
            "critic_rules": _RuleRegistryTarget(
                self.rules,
                load_failures=self.rule_load_failures,
            )
        }

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
        results = []
        passed = 0
        failed = 0
        blocking_failed = 0
        run_id = str(data.get("run_id") or data.get("request_id") or "")
        blocking_rule_ids = _blocking_rule_ids(properties)
        cached_verdict = await self._existing_critic_verdict(run_id, smiles)
        if cached_verdict is not None:
            return cached_verdict

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

        crg_results = await self._shared_crg_failure_results(
            run_id,
            smiles,
            blocking_rule_ids=blocking_rule_ids,
        )
        results.extend(crg_results)
        failed += len(crg_results)
        blocking_failed += sum(1 for result in crg_results if bool(result.get("blocking", True)))

        overall_verdict = "pass" if blocking_failed == 0 else "fail"
        total_evidence = len(results)
        belief = self.crg.add_belief(
            subject=smiles,
            predicate="critic_verdict",
            obj=overall_verdict,
            confidence=(passed / total_evidence if total_evidence else 1.0),
            source_agent=self.name,
            evidence_ids=[str(item["rule_id"]) for item in results if item.get("rule_id")],
        )
        await self._persist_belief(
            belief,
            project_id=str(data.get("project_id") or ""),
            run_id=run_id,
        )
        return {
            "smiles": smiles,
            "verdict": overall_verdict,
            "passed": passed,
            "failed": failed,
            "blocking_failed": blocking_failed,
            "non_blocking_failed": max(0, failed - blocking_failed),
            "total_rules": len(self.rules),
            "rule_results": results,
        }

    async def process(self, data):
        return await self.evaluate_molecule(data)

    async def _existing_critic_verdict(
        self,
        run_id: str,
        smiles: str,
    ) -> dict[str, Any] | None:
        if not run_id or self.crg_repository is None:
            return None
        read_crg = getattr(self.crg_repository, "get_run_crg", None)
        if not callable(read_crg):
            return None
        crg = await self.read_shared_crg(run_id)
        for belief in crg.get("beliefs", []) or []:
            if not isinstance(belief, dict):
                continue
            if str(belief.get("subject") or "") != smiles:
                continue
            predicate = str(belief.get("predicate") or "")
            verdict = str(belief.get("object") or belief.get("object_value") or "").lower()
            if predicate != "critic_verdict" or verdict not in {"pass", "fail"}:
                continue
            rule_result = {
                "rule_id": "crg_critic_verdict",
                "rule_name": "Shared CRG critic verdict",
                "verdict": verdict,
                "score": float(belief.get("confidence") or 1.0),
                "reasoning": "shared CRG contains existing critic_verdict",
            }
            return {
                "smiles": smiles,
                "verdict": verdict,
                "passed": 1 if verdict == "pass" else 0,
                "failed": 1 if verdict == "fail" else 0,
                "total_rules": 0,
                "rule_results": [rule_result],
                "cache_source": "shared_crg",
            }
        return None

    async def _shared_crg_failure_results(
        self,
        run_id: str,
        smiles: str,
        *,
        blocking_rule_ids: set[str] | None = None,
    ) -> list[dict]:
        if not run_id or self.crg_repository is None:
            return []
        read_crg = getattr(self.crg_repository, "get_run_crg", None)
        if not callable(read_crg):
            return []
        crg = await self.read_shared_crg(run_id)
        results = []
        for belief in crg.get("beliefs", []) or []:
            if not isinstance(belief, dict):
                continue
            if str(belief.get("subject") or "") != smiles:
                continue
            predicate = str(belief.get("predicate") or "")
            object_value = str(belief.get("object") or belief.get("object_value") or "").lower()
            if predicate == "validation_status" and object_value == "failed":
                results.append(
                    {
                        "rule_id": "crg_validation_status",
                        "rule_name": "Shared CRG validation status",
                        "verdict": "fail",
                        "score": float(belief.get("confidence") or 1.0),
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
                        "score": float(belief.get("confidence") or 1.0),
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
                        "score": float(belief.get("confidence") or 1.0),
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
