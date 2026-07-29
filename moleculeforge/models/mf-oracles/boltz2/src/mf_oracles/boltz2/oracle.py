"""Boltz2 oracle: Ensemble docking and binding free energy prediction."""

import asyncio
import inspect

from mf_core.plugins.oracle import OraclePlugin


class Boltz2Oracle(OraclePlugin):
    def __init__(self, runner=None):
        self.runner = runner

    async def evaluate(
        self,
        molecules: list[str],
        properties: list[str],
    ) -> dict[str, dict[str, float]]:
        if self.runner is None:
            raise RuntimeError("BOLTZ_RUNNER is required")
        evaluate = self.runner.evaluate
        if inspect.iscoroutinefunction(evaluate):
            result = await evaluate(molecules, properties)
        else:
            result = await asyncio.to_thread(evaluate, molecules, properties)
        if inspect.isawaitable(result):
            result = await result
        _require_result_fields(result, ("model_version", "runtime_ms"))
        return result

    async def predict_with_uncertainty(self, molecules, properties):
        if self.runner is None or not hasattr(self.runner, "predict_with_uncertainty"):
            raise RuntimeError("BOLTZ uncertainty runner is required")
        predict = self.runner.predict_with_uncertainty
        if inspect.iscoroutinefunction(predict):
            result = await predict(molecules, properties)
        else:
            result = await asyncio.to_thread(predict, molecules, properties)
        if inspect.isawaitable(result):
            return await result
        return result

    def oracle_level(self) -> int:
        return 1


def _require_result_fields(result: dict, fields: tuple[str, ...]) -> None:
    for smiles, values in result.items():
        for field in fields:
            if field not in values or values[field] in ("", None):
                raise RuntimeError(f"Boltz result for {smiles} requires {field}")
