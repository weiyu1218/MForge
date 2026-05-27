"""Unit tests for Cross-Paradigm Knowledge Distillation layer."""

from __future__ import annotations

import pytest
import torch
from mf_core.routing.cross_paradigm_kd import CrossParadigmKDLayer, WeakTeacher


class TestWeakTeacher:
    def test_score_valid_molecule(self) -> None:
        teacher = WeakTeacher()
        score = teacher.score("c1ccccc1")  # benzene
        assert 0.0 <= score <= 1.0

    def test_score_invalid_smiles(self) -> None:
        teacher = WeakTeacher()
        score = teacher.score("not_a_smiles")
        assert score == 0.0

    def test_score_batch(self) -> None:
        teacher = WeakTeacher()
        scores = teacher.score_batch(["c1ccccc1", "CC(=O)O", "invalid"])
        assert len(scores) == 3
        assert scores[2] == 0.0

    def test_drug_like_scores_higher(self) -> None:
        teacher = WeakTeacher()
        # Aspirin-like molecule should score higher than a large polymer
        drug_score = teacher.score("CC(=O)Oc1ccccc1C(=O)O")  # aspirin
        bad_score = teacher.score("C" * 50)  # invalid/long chain
        assert drug_score >= bad_score


class TestCrossParadigmKDLayer:
    def test_initialization(self) -> None:
        kd = CrossParadigmKDLayer(n_generators=8)
        assert kd.n_generators == 8
        assert kd.running_means.shape == (8,)

    def test_update_teacher_scores_rejects_smiles_in_production(self) -> None:
        kd = CrossParadigmKDLayer(n_generators=8)

        with pytest.raises(TypeError, match="oracle feedback"):
            kd.update_teacher_scores("hfm_3d", 0, ["c1ccccc1", "CC(=O)O"])

        assert kd.running_counts[0].item() == 0.0

    def test_update_teacher_scores_uses_oracle_feedback(self) -> None:
        kd = CrossParadigmKDLayer(n_generators=8)

        score = kd.update_teacher_scores(
            "hfm_3d",
            0,
            [
                {"oracle_name": "rdkit_oracle_l0", "normalized_score": 0.8},
                {"oracle_name": "dock_l1", "normalized_score": 0.4},
            ],
        )
        assert score == pytest.approx(0.6)
        assert kd.running_counts[0].item() == 2.0

    def test_compute_distillation_loss(self) -> None:
        kd = CrossParadigmKDLayer(n_generators=8)

        # Create some fake embeddings
        embeddings = [torch.randn(4, 128), torch.randn(4, 128)]
        indices = [0, 1]

        loss = kd.compute_distillation_loss(embeddings, indices)
        assert loss.shape == ()
        assert loss.item() >= 0.0
        assert loss.requires_grad

    def test_quality_ranking(self) -> None:
        kd = CrossParadigmKDLayer(n_generators=8)

        # Update with different scores
        kd.update_teacher_scores(
            "hfm_3d",
            0,
            [{"oracle_name": "rdkit_oracle_l0", "normalized_score": 0.9}],
        )
        kd.update_teacher_scores(
            "iclm",
            3,
            [{"oracle_name": "rdkit_oracle_l0", "normalized_score": 0.2}],
        )

        ranking = kd.get_generator_quality_ranking()
        assert "generator_0" in ranking
        assert "generator_3" in ranking
        assert ranking["generator_0"] == pytest.approx(0.9)
        assert ranking["generator_3"] == pytest.approx(0.2)
