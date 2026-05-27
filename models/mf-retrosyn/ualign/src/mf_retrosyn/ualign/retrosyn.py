"""UAlign: Sequence alignment-based retrosynthetic route planner."""
import inspect

from mf_retrosyn._route_validation import validate_retrosyn_routes


class UAlignRetrosyn:
    """Retrosynthetic analysis via sequence alignment of molecular fingerprints.

    Aligns target molecules against a database of known synthetic routes
    using local sequence alignment to propose synthetic pathways.
    """

    def __init__(self, runner=None):
        self.runner = runner

    async def find_routes(self, smiles: str, max_routes: int = 10) -> list[dict]:
        """Find retrosynthetic routes via sequence alignment.

        Args:
            smiles: Target molecule SMILES string.
            max_routes: Maximum number of routes to return.

        Returns:
            List of route dictionaries with keys: 'route_id', 'smiles',
            'steps', 'score', 'alignment_score', 'template_reactions'.
        """
        if self.runner is None:
            raise RuntimeError("UALIGN_RUNNER is required")
        result = self.runner.find_routes(smiles, max_routes=max_routes)
        if inspect.isawaitable(result):
            result = await result
        return validate_retrosyn_routes(result, "UAlignRetrosyn")
