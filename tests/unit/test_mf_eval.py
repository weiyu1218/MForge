from __future__ import annotations

import pytest
import torch


def test_pairwise_distance_distortion_metrics() -> None:
    from mf_eval.distortion import pairwise_distance_distortion

    metrics = pairwise_distance_distortion(
        torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
        torch.tensor([[0.0, 2.0], [2.0, 0.0]]),
    )

    assert metrics["n_pairs"] == 1
    assert metrics["mean_absolute_error"] == pytest.approx(1.0)
    assert metrics["mean_relative_error"] == pytest.approx(1.0)
    assert metrics["spearman_r"] == 1.0


def test_pairwise_distance_distortion_rejects_shape_mismatch() -> None:
    from mf_eval.distortion import pairwise_distance_distortion

    with pytest.raises(ValueError, match="shape"):
        pairwise_distance_distortion(torch.zeros(2, 2), torch.zeros(3, 3))


def test_find_activity_cliffs_and_separation_metric() -> None:
    from mf_eval.cliff_analysis import cliff_separation_auroc, find_activity_cliffs

    cliffs = find_activity_cliffs(
        ["CCO", "CCN", "c1ccccc1"],
        [1.0, 3.0, 1.1],
        similarity_threshold=0.2,
        activity_delta_threshold=1.0,
    )
    assert cliffs
    assert {"i", "j", "similarity", "activity_delta"} <= set(cliffs[0])

    score = cliff_separation_auroc(
        torch.tensor([[0.0, 0.0], [0.1, 0.0], [5.0, 5.0]], dtype=torch.float32),
        [True, False, False],
    )
    assert score is not None
    assert 0.0 <= score <= 1.0

    assert cliff_separation_auroc(torch.zeros(2, 2), [True, True]) is None


def test_hypervolume_2d_and_improvement() -> None:
    from mf_eval.hv_evaluator import (
        filter_non_dominated,
        hypervolume_2d,
        hypervolume_improvement,
    )

    front = filter_non_dominated([[1.0, 1.0], [2.0, 2.0], [3.0, 1.0]])

    assert front.tolist() == [[2.0, 2.0], [3.0, 1.0]]
    assert hypervolume_2d(front, reference=[0.0, 0.0]) == pytest.approx(5.0)
    assert hypervolume_improvement(
        [3.0, 2.0],
        front,
        reference=[0.0, 0.0],
    ) == pytest.approx(1.0)


def test_hypervolume_rejects_non_2d_contract() -> None:
    from mf_eval.hv_evaluator import hypervolume_2d

    with pytest.raises(ValueError, match="2D"):
        hypervolume_2d([[1.0, 1.0, 1.0]], reference=[0.0, 0.0, 0.0])
