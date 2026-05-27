"""RSGPT: GPT-based retrosynthetic route planner."""
import inspect

from mf_retrosyn._route_validation import validate_retrosyn_routes


class RSGPTRetrosyn:
    """GPT-based retrosynthetic analysis using transformer models.

    Uses a GPT-style autoregressive model trained on reaction data
    to predict retrosynthetic disconnections step by step.
    """

    def __init__(self, runner=None):
        self.runner = runner

    async def find_routes(self, smiles: str, max_routes: int = 10) -> list[dict]:
        """Find retrosynthetic routes for a target molecule using GPT.

        Args:
            smiles: Target molecule SMILES string.
            max_routes: Maximum number of routes to return.

        Returns:
            List of route dictionaries with keys: 'route_id', 'smiles',
            'steps', 'score', 'reactions', 'intermediates'.
        """
        if self.runner is None:
            raise RuntimeError("RSGPT_RUNNER is required")
        result = self.runner.find_routes(smiles, max_routes=max_routes)
        if inspect.isawaitable(result):
            result = await result
        return validate_retrosyn_routes(result, "RSGPTRetrosyn")
