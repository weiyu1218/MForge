"""Unit tests for Cross-Paradigm Knowledge Distillation layer."""

from __future__ import annotations

import json

import pytest
import torch
from mf_core.routing.cross_paradigm_kd import (
    CrossParadigmKDLayer,
    WeakTeacher,
    boltz2_affinity_teacher_distribution,
    boltz2_teacher_feedback,
    hypseek_teacher_distribution,
    hypseek_teacher_feedback,
)


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

    def test_update_teacher_scores_accepts_external_teacher_distribution(self) -> None:
        kd = CrossParadigmKDLayer(n_generators=8)

        score = kd.update_teacher_scores(
            "hfm_3d",
            0,
            [
                {
                    "oracle_name": "boltz2",
                    "teacher_distribution": [0.2, 0.6, 1.0],
                }
            ],
        )

        assert score == pytest.approx(0.6)
        assert kd.running_counts[0].item() == 3.0

    def test_boltz2_affinity_adapter_builds_teacher_distribution(self) -> None:
        distribution = boltz2_affinity_teacher_distribution(
            [
                {
                    "delta_g_kcal_mol": -8.0,
                    "per_member_dg": [-12.0, -6.0, 0.0],
                }
            ],
            favorable_delta_g=-12.0,
            unfavorable_delta_g=0.0,
        )

        assert distribution == pytest.approx([1.0, 0.5, 0.0])

    def test_boltz2_teacher_feedback_updates_kd_scores(self) -> None:
        kd = CrossParadigmKDLayer(n_generators=8)

        score = kd.update_teacher_scores(
            "hfm_3d",
            0,
            [
                boltz2_teacher_feedback(
                    [
                        {
                            "delta_g_kcal_mol": -8.0,
                            "per_member_dg": [-12.0, -6.0, 0.0],
                        }
                    ],
                    favorable_delta_g=-12.0,
                    unfavorable_delta_g=0.0,
                )
            ],
        )

        assert score == pytest.approx(0.5)
        assert kd.running_counts[0].item() == 3.0

    def test_hypseek_adapter_builds_teacher_distribution_from_explicit_field(self) -> None:
        distribution = hypseek_teacher_distribution(
            [
                {"hypothesis_confidence": 0.1},
                {"hypothesis_confidence": 0.6},
                {"hypothesis_confidence": 1.1},
            ],
            score_field="hypothesis_confidence",
            min_score=0.1,
            max_score=1.1,
        )

        assert distribution == pytest.approx([0.0, 0.5, 1.0])

    def test_hypseek_adapter_supports_lower_is_better_scores(self) -> None:
        distribution = hypseek_teacher_distribution(
            [
                {"risk_score": 0.0},
                {"risk_score": 5.0},
                {"risk_score": 10.0},
            ],
            score_field="risk_score",
            min_score=0.0,
            max_score=10.0,
            higher_is_better=False,
        )

        assert distribution == pytest.approx([1.0, 0.5, 0.0])

    def test_hypseek_teacher_feedback_updates_kd_scores(self) -> None:
        kd = CrossParadigmKDLayer(n_generators=8)

        score = kd.update_teacher_scores(
            "fragfm",
            1,
            [
                hypseek_teacher_feedback(
                    [
                        {"hypothesis_confidence": 0.0},
                        {"hypothesis_confidence": 1.0},
                    ],
                    score_field="hypothesis_confidence",
                    min_score=0.0,
                    max_score=1.0,
                )
            ],
        )

        assert score == pytest.approx(0.5)
        assert kd.running_counts[1].item() == 2.0

    def test_compute_distillation_loss(self) -> None:
        kd = CrossParadigmKDLayer(n_generators=8)

        embeddings = [torch.randn(4, 128), torch.randn(4, 128)]
        indices = [0, 1]

        loss = kd.compute_distillation_loss(embeddings, indices)
        assert loss.shape == ()
        assert loss.item() >= 0.0
        assert loss.requires_grad

    def test_compute_distillation_loss_uses_teacher_embedding_targets(self) -> None:
        kd = CrossParadigmKDLayer(n_generators=8)
        kd.update_teacher_embedding_targets(
            0,
            torch.tensor(
                [
                    [1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                ]
            ),
        )
        embeddings = [torch.tensor([[1.0, 0.0, 0.0]], requires_grad=True)]

        loss = kd.compute_distillation_loss(embeddings, [0])

        assert loss.item() == pytest.approx(0.0)
        loss.backward()
        assert embeddings[0].grad is not None

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


def test_export_teacher_embeddings_artifact_from_jsonl(tmp_path) -> None:
    from mf_core.routing.kd_artifacts import export_teacher_embeddings_artifact

    records_path = tmp_path / "teacher_records.jsonl"
    records_path.write_text(
        "\n".join(
            [
                json.dumps({"id": "hfm", "teacher_embedding": [0.1, 0.2]}),
                json.dumps({"id": "fragfm", "teacher_embedding": [0.3, 0.4]}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "teacher_embeddings.json"

    report = export_teacher_embeddings_artifact(
        records_path,
        output_path,
        expected_dim=2,
    )

    assert report["status"] == "pass"
    assert report["embedding_count"] == 2
    assert report["embedding_dim"] == 2
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload == {
        "schema_version": "cross_paradigm_teacher_embeddings.v1",
        "embedding_count": 2,
        "embedding_dim": 2,
        "teacher_embeddings": [[0.1, 0.2], [0.3, 0.4]],
    }


def test_teacher_embeddings_report_fails_dimension_mismatch(tmp_path) -> None:
    from mf_core.routing.kd_artifacts import build_teacher_embeddings_report

    artifact_path = tmp_path / "teacher_embeddings.json"
    artifact_path.write_text(
        json.dumps({"teacher_embeddings": [[0.1, 0.2]]}),
        encoding="utf-8",
    )

    report = build_teacher_embeddings_report(
        artifact_path,
        expected_dim=3,
    )

    assert report["status"] == "fail"
    assert report["embedding_count"] == 1
    assert report["embedding_dim"] == 2
    assert any("dimension" in message for message in report["messages"])
