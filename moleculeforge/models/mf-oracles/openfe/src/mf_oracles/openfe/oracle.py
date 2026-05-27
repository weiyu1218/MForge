"""OpenFE oracle: L3 Free Energy Perturbation calculations."""
import inspect

from mf_core.plugins.oracle import OraclePlugin


class OpenFEOracle(OraclePlugin):
    def __init__(self, runner=None, skip_when_unavailable: bool = False):
        self.runner = runner
        self.skip_when_unavailable = skip_when_unavailable

    async def evaluate(
        self,
        molecules: list[str],
        properties: list[str],
    ) -> dict[str, dict[str, float]]:
        if self.runner is None:
            if self.skip_when_unavailable:
                return {
                    smiles: {
                        "skipped": True,
                        "skip_reason": "OPENFE_RUNNER is required",
                    }
                    for smiles in molecules
                }
            raise RuntimeError("OPENFE_RUNNER is required")
        result = self.runner.evaluate(molecules, properties)
        if inspect.isawaitable(result):
            return await result
        return result

    async def predict_with_uncertainty(self, molecules, properties):
        if self.runner is None or not hasattr(self.runner, "predict_with_uncertainty"):
            raise RuntimeError("OPENFE uncertainty runner is required")
        result = self.runner.predict_with_uncertainty(molecules, properties)
        if inspect.isawaitable(result):
            return await result
        return result

    def oracle_level(self) -> int:
        return 3
