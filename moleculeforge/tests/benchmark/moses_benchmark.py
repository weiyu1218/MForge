"""MOSES benchmark runner for MoleculeForge generators.

MOSES (Molecular Sets) is a benchmarking platform for molecular generation.
Evaluates: validity, uniqueness, novelty, internal diversity, FCD, SNN.

Reference: https://github.com/molecularsets/moses
"""

import pytest


@pytest.mark.benchmark
@pytest.mark.slow
class TestMosesBenchmark:
    """MOSES distribution-learning benchmark for all generators."""

    def test_hfm3d_moses_validity(self):
        """HFM-3D should achieve > 90% validity on MOSES test set."""
        pytest.skip("Requires trained HFM-3D + MOSES dataset")

    def test_hfm3d_moses_uniqueness(self):
        """HFM-3D should generate > 95% unique molecules."""
        pytest.skip("Requires trained HFM-3D + MOSES dataset")

    def test_hfm3d_moses_novelty(self):
        """HFM-3D should produce > 70% novel molecules (not in training)."""
        pytest.skip("Requires trained HFM-3D + MOSES dataset")

    def test_fragfm_moses_validity(self):
        """FragFM should achieve > 95% validity."""
        pytest.skip("Requires trained FragFM + MOSES dataset")

    def test_lamgen3d_moses_validity(self):
        """LaMGen-3D should achieve > 85% validity."""
        pytest.skip("Requires trained LaMGen-3D + MOSES dataset")
