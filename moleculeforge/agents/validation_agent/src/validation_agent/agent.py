"""Validation Agent - Adaptive Oracle Cascade L0-L4 (Agent-4)."""
from __future__ import annotations

import inspect
import json

from mf_agents.base.agent import BaseAgent
from mf_agents.crg.graph import ChemicalReasoningGraph


class ValidationAgent(BaseAgent):
    def __init__(self, message_bus=None, oracles: dict | None = None):
        super().__init__("validation_agent", message_bus)
        self._subscription_subjects = ["agent.validation.request", "orchestrator.validate.check"]
        self.crg = ChemicalReasoningGraph()
        self.oracles = oracles or {}
        self.oracle_levels = {
            0: ("filter", ["admet_score"]),
            1: ("docking", ["docking_score"]),
            2: ("affinity", ["affinity"]),
            3: ("rbfe", ["rbfe"]),
            4: "experimental_assay",
        }
        self.default_thresholds = {
            1: ("docking_score", "max", -6.0, "l1_max_docking_score"),
            2: ("affinity", "max", -7.0, "l2_max_affinity"),
            3: ("rbfe", "max", 0.0, "l3_max_rbfe"),
        }
        self.default_uncertainty_thresholds = {
            1: ("docking_score", 1.0, "l1_max_uncertainty"),
            2: ("affinity", 1.0, "l2_max_uncertainty"),
            3: ("rbfe", 1.0, "l3_max_uncertainty"),
        }

    async def handle_message(self, subject, payload, reply_to=""):
        data = json.loads(payload) if isinstance(payload, bytes) else {"raw": payload}
        result = await self.process(data)
        if reply_to:
            await self.publish(reply_to, json.dumps(result).encode())

    async def process(self, data):
        """Run adaptive oracle cascade from L0 to requested level.

        Progressively validates molecules through increasingly expensive
        oracles. Early termination on failure at any level for efficiency.
        """
        max_level = int(data.get("oracle_level", 0))
        smiles = data.get("smiles", "")
        if not smiles:
            raise ValueError("smiles is required")
        l0_threshold = float(data.get("l0_threshold", 0.0))
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
        return {
            "agent": self.name,
            "status": "validated" if overall_passed else "failed",
            "smiles": smiles,
            "max_oracle_level": max_level,
            "cascade": cascade_results,
            "upgrade_path": upgrade_path,
            "overall_passed": overall_passed,
        }

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
            f"{score_name}_{mode}": float(data.get(config_key, default_value)),
            f"{uncertainty_name}_uncertainty_max": float(
                data.get(uncertainty_key, uncertainty_default)
            ),
        }

    def _level_passed(
        self,
        level: int,
        values: dict,
        uncertainty: dict | None,
        thresholds: dict,
    ) -> bool:
        if values.get("skipped") is True:
            return True
        if level == 0:
            score = values.get("admet_score")
            if not isinstance(score, int | float):
                raise RuntimeError("L0 filter requires admet_score")
            return float(score) >= thresholds["admet_score_min"]
        score_name, mode, _default_value, _config_key = self.default_thresholds[level]
        score = values.get(score_name)
        if not isinstance(score, int | float):
            raise RuntimeError(f"L{level} oracle requires {score_name}")
        score_threshold = thresholds[f"{score_name}_{mode}"]
        if mode == "max" and float(score) > score_threshold:
            return False
        uncertainty_name, _uncertainty_default, _uncertainty_key = (
            self.default_uncertainty_thresholds[level]
        )
        if uncertainty is None:
            return True
        uncertainty_value = uncertainty.get(uncertainty_name)
        if uncertainty_value is None:
            return True
        return float(uncertainty_value) <= thresholds[f"{uncertainty_name}_uncertainty_max"]


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
