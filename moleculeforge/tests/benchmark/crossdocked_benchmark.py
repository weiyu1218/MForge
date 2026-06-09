"""CrossDocked benchmark runner for pocket-conditioned evaluation records."""

from __future__ import annotations

import pytest
from rdkit import Chem

from . import env_float, read_jsonl_records


@pytest.mark.benchmark
@pytest.mark.slow
class TestCrossDockedBenchmark:
    """CrossDocked 2020-style validation from real pocket-ligand records."""

    @pytest.fixture(scope="class")
    def records(self) -> list[dict]:
        return read_jsonl_records("CROSSDOCKED_BENCHMARK_JSONL")

    def test_crossdocked_records_have_pocket_and_ligand(self, records: list[dict]) -> None:
        missing = [
            idx
            for idx, record in enumerate(records, 1)
            if not record.get("pocket_id") or not record.get("ligand_smiles")
        ]
        assert not missing, f"records missing pocket_id or ligand_smiles: {missing[:10]}"

    def test_crossdocked_ligands_are_valid_smiles(self, records: list[dict]) -> None:
        valid = sum(
            1
            for record in records
            if Chem.MolFromSmiles(str(record.get("ligand_smiles", ""))) is not None
        )
        validity = valid / len(records)
        assert validity >= env_float("CROSSDOCKED_MIN_VALIDITY", 0.95)

    def test_crossdocked_contains_test_split(self, records: list[dict]) -> None:
        split_count = sum(1 for record in records if str(record.get("split", "")) == "test")
        coverage = split_count / len(records)
        assert coverage >= env_float("CROSSDOCKED_MIN_TEST_SPLIT_FRACTION", 0.05)

    def test_crossdocked_docking_score_gate(self, records: list[dict]) -> None:
        scored = [
            float(record["docking_score"])
            for record in records
            if record.get("docking_score") is not None
        ]
        if not scored:
            pytest.skip("CROSSDOCKED_BENCHMARK_JSONL contains no docking_score values")
        best_score = min(scored)
        assert best_score <= env_float("CROSSDOCKED_MAX_BEST_DOCKING_SCORE", -6.0)
