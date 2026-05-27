"""GuacaMol v3 benchmark runner.

GuacaMol evaluates molecular generation models on goal-directed tasks:
rediscovery, similarity, isomer, median, and multi-property optimization.

Reference: https://github.com/BenevolentAI/guacamol
"""

import pytest


@pytest.mark.benchmark
@pytest.mark.slow
class TestGuacaMolBenchmark:
    """GuacaMol v3 goal-directed benchmark."""

    def test_celecoxib_rediscovery(self):
        """Goal: rediscover Celecoxib from scratch."""
        pytest.skip("Requires trained generators + GuacaMol v3")

    def test_troglitazone_rediscovery(self):
        """Goal: rediscover Troglitazone."""
        pytest.skip("Requires trained generators")

    def test_thiothixene_rediscovery(self):
        """Goal: rediscover Thiothixene."""
        pytest.skip("Requires trained generators")

    def test_median_molecules_1(self):
        """Goal: generate molecules similar to median molecule 1."""
        pytest.skip("Requires trained generators")

    def test_mpo_fexofenadine(self):
        """Multi-property optimization: Fexofenadine-like + LogP + TPSA."""
        pytest.skip("Requires trained generators + ADMET-AI")
