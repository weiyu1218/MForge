"""GuacaMol v3 benchmark runner.

GuacaMol evaluates molecular generation models on goal-directed tasks:
rediscovery, similarity, isomer, median, and multi-property optimization.

Reference: https://github.com/BenevolentAI/guacamol
"""

import pytest
from rdkit import Chem
from rdkit.Chem import QED, Descriptors

from . import env_float, env_int, generate_hfm_smiles, tanimoto_similarity


@pytest.mark.benchmark
@pytest.mark.slow
class TestGuacaMolBenchmark:
    """GuacaMol v3 goal-directed benchmark."""

    @pytest.fixture(scope="class")
    async def generated_smiles(self) -> list[str]:
        return await generate_hfm_smiles(
            batch_size=env_int("GUACAMOL_BENCHMARK_BATCH_SIZE", 256),
            seed=env_int("GUACAMOL_BENCHMARK_SEED", 42),
        )

    async def test_celecoxib_rediscovery(self, generated_smiles: list[str]):
        """Goal: rediscover Celecoxib from scratch."""
        score = _max_similarity(generated_smiles, "Cc1ccc(S(=O)(=O)Nc2ccc(C(F)(F)F)cc2)cc1")
        assert score >= env_float("GUACAMOL_MIN_CELECOXIB_SIMILARITY", 0.75)

    async def test_troglitazone_rediscovery(self, generated_smiles: list[str]):
        """Goal: rediscover Troglitazone."""
        score = _max_similarity(
            generated_smiles,
            "Cc1c(C)c2c(c(C)c1O)CCC(C)(COc1ccc(CC3SC(=O)NC3=O)cc1)O2",
        )
        assert score >= env_float("GUACAMOL_MIN_TROGLITAZONE_SIMILARITY", 0.75)

    async def test_thiothixene_rediscovery(self, generated_smiles: list[str]):
        """Goal: rediscover Thiothixene."""
        score = _max_similarity(generated_smiles, "CN1CCN(CC/C=C2/c3ccccc3Sc3ccc(S(C)=O)cc32)CC1")
        assert score >= env_float("GUACAMOL_MIN_THIOTHIXENE_SIMILARITY", 0.75)

    async def test_median_molecules_1(self, generated_smiles: list[str]):
        """Goal: generate molecules similar to median molecule 1."""
        target_a = "CC(C)C1=CC=C(C=C1)C(C)C(=O)O"
        target_b = "CC1=C(C(=CC=C1)C)C2=CC=CC=C2"
        score = max(
            (tanimoto_similarity(smiles, target_a) + tanimoto_similarity(smiles, target_b)) / 2
            for smiles in generated_smiles
        )
        assert score >= env_float("GUACAMOL_MIN_MEDIAN1_SIMILARITY", 0.45)

    async def test_mpo_fexofenadine(self, generated_smiles: list[str]):
        """Multi-property optimization: Fexofenadine-like + LogP + TPSA."""
        target = "CC(C)(C(=O)O)c1ccc(cc1)C(O)CCCN1CCC(CC1)(c1ccccc1)c1ccccc1"
        score = max(_fexofenadine_mpo_score(smiles, target) for smiles in generated_smiles)
        assert score >= env_float("GUACAMOL_MIN_FEXOFENADINE_MPO", 0.45)


def _max_similarity(smiles_list: list[str], target: str) -> float:
    return max(tanimoto_similarity(smiles, target) for smiles in smiles_list)


def _fexofenadine_mpo_score(smiles: str, target: str) -> float:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0.0
    similarity = tanimoto_similarity(smiles, target)
    logp = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    qed = QED.qed(mol)
    logp_score = max(0.0, 1.0 - abs(logp - 3.5) / 3.5)
    tpsa_score = max(0.0, 1.0 - abs(tpsa - 90.0) / 90.0)
    return float((similarity + logp_score + tpsa_score + qed) / 4.0)
