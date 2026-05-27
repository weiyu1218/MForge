"""Gnina oracle: L2 docking with CNN scoring function."""
import inspect

from mf_core.plugins.oracle import OraclePlugin


class GninaOracle(OraclePlugin):
    def __init__(self, runner=None):
        self.runner = runner

    async def evaluate(
        self,
        molecules: list[str],
        properties: list[str],
    ) -> dict[str, dict[str, float]]:
        if self.runner is None:
            raise RuntimeError("GNINA_RUNNER is required")
        result = self.runner.evaluate(molecules, properties)
        if inspect.isawaitable(result):
            result = await result
        _require_result_fields(result, ("input_artifact_hash", "stderr_path"))
        return result

    async def predict_with_uncertainty(self, molecules, properties):
        if self.runner is None or not hasattr(self.runner, "predict_with_uncertainty"):
            raise RuntimeError("GNINA uncertainty runner is required")
        result = self.runner.predict_with_uncertainty(molecules, properties)
        if inspect.isawaitable(result):
            return await result
        return result

    def oracle_level(self) -> int:
        return 2


def _require_result_fields(result: dict, fields: tuple[str, ...]) -> None:
    for smiles, values in result.items():
        for field in fields:
            if field not in values or values[field] in ("", None):
                raise RuntimeError(
                    f"GNINA result for {smiles} requires {field}"
                )
