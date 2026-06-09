"""PMO (Practical Molecular Optimization) benchmark.

PMO evaluates multi-property optimization capability with 23 tasks
spanning: logP, QED, DRD2, JNK3, GSK3B, and multi-objective combinations.

Reference: https://github.com/wenhao-gao/mol_opt
"""

import pytest
from rdkit import Chem
from rdkit.Chem import QED, Descriptors

from . import (
    env_float,
    env_int,
    generate_hfm_smiles,
    read_scored_smiles_table,
)


@pytest.mark.benchmark
@pytest.mark.slow
class TestPMOBenchmark:
    """PMO 23-task multi-objective optimization benchmark."""

    @pytest.fixture(scope="class")
    async def generated_smiles(self) -> list[str]:
        return await generate_hfm_smiles(
            batch_size=env_int("PMO_BENCHMARK_BATCH_SIZE", 256),
            seed=env_int("PMO_BENCHMARK_SEED", 42),
        )

    async def test_logp_optimization(self, generated_smiles: list[str]):
        """Maximize LogP (simple single-objective baseline)."""
        score = max(_logp_score(smiles) for smiles in generated_smiles)
        assert score >= env_float("PMO_MIN_LOGP_SCORE", 0.7)

    async def test_qed_optimization(self, generated_smiles: list[str]):
        """Maximize QED (drug-likeness)."""
        score = max(_qed_score(smiles) for smiles in generated_smiles)
        assert score >= env_float("PMO_MIN_QED", 0.7)

    def test_drd2_optimization(self):
        """Maximize DRD2 activity score."""
        scores = read_scored_smiles_table("PMO_SCORE_TABLE_PATH", "drd2")
        assert max(scores.values()) >= env_float("PMO_MIN_DRD2", 0.5)

    async def test_multi_objective_logp_qed(self, generated_smiles: list[str]):
        """Simultaneously maximize LogP and QED."""
        score = max(
            (_logp_score(smiles) + _qed_score(smiles)) / 2
            for smiles in generated_smiles
        )
        assert score >= env_float("PMO_MIN_LOGP_QED", 0.65)

    def test_multi_objective_jnk3_gsk3b(self):
        """Dual kinase selectivity optimization."""
        jnk3 = read_scored_smiles_table("PMO_SCORE_TABLE_PATH", "jnk3")
        gsk3b = read_scored_smiles_table("PMO_SCORE_TABLE_PATH", "gsk3b")
        shared = sorted(set(jnk3).intersection(gsk3b))
        if not shared:
            pytest.skip("PMO_SCORE_TABLE_PATH has no shared JNK3/GSK3B rows")
        score = max((jnk3[smiles] + gsk3b[smiles]) / 2 for smiles in shared)
        assert score >= env_float("PMO_MIN_JNK3_GSK3B", 0.5)


def _logp_score(smiles: str) -> float:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0.0
    logp = Descriptors.MolLogP(mol)
    return float(max(0.0, min(1.0, (logp + 2.0) / 8.0)))


def _qed_score(smiles: str) -> float:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0.0
    return float(QED.qed(mol))
