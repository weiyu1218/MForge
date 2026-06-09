"""MOSES benchmark runner for MoleculeForge generators.

MOSES (Molecular Sets) is a benchmarking platform for molecular generation.
Evaluates: validity, uniqueness, novelty, internal diversity, FCD, SNN.

Reference: https://github.com/molecularsets/moses
"""

import pytest
from mf_eval.molecule.moses import evaluate_moses

from . import (
    env_float,
    env_int,
    generate_hfm_smiles,
    read_smiles_file,
    require_hfm_artifacts,
)


@pytest.mark.benchmark
@pytest.mark.slow
class TestMosesBenchmark:
    """MOSES distribution-learning benchmark for all generators."""

    @pytest.fixture(scope="class")
    def reference_smiles(self) -> list[str]:
        return read_smiles_file("MOSES_REFERENCE_SMILES_PATH")

    @pytest.fixture(scope="class")
    async def hfm3d_moses_metrics(self, reference_smiles: list[str]) -> dict[str, float]:
        require_hfm_artifacts()
        batch_size = env_int("MOSES_BENCHMARK_BATCH_SIZE", 256)
        generated = await generate_hfm_smiles(
            batch_size=batch_size,
            seed=env_int("MOSES_BENCHMARK_SEED", 42),
        )
        return evaluate_moses(generated, reference_smiles)

    async def test_hfm3d_moses_validity(self, hfm3d_moses_metrics: dict[str, float]):
        """HFM-3D should achieve > 90% validity on MOSES test set."""
        assert hfm3d_moses_metrics["validity"] >= env_float("MOSES_MIN_VALIDITY", 0.90)

    async def test_hfm3d_moses_uniqueness(self, hfm3d_moses_metrics: dict[str, float]):
        """HFM-3D should generate > 95% unique molecules."""
        assert hfm3d_moses_metrics["uniqueness"] >= env_float("MOSES_MIN_UNIQUENESS", 0.95)

    async def test_hfm3d_moses_novelty(self, hfm3d_moses_metrics: dict[str, float]):
        """HFM-3D should produce > 70% novel molecules (not in training)."""
        assert hfm3d_moses_metrics["novelty"] >= env_float("MOSES_MIN_NOVELTY", 0.70)

    def test_fragfm_moses_validity(self):
        """FragFM should achieve > 95% validity."""
        generated = read_smiles_file("FRAGFM_MOSES_GENERATED_SMILES_PATH")
        reference = read_smiles_file("MOSES_REFERENCE_SMILES_PATH")
        metrics = evaluate_moses(generated, reference)
        assert metrics["validity"] >= env_float("FRAGFM_MOSES_MIN_VALIDITY", 0.95)
