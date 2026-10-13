import numpy as np
import pytest

from tb_outcomes.calibration import CalibrationConfig
from tb_outcomes.model_selection import Candidate, declared_candidates, select_candidate, three_way_group_splits


def test_every_scored_row_is_unseen_by_model_and_calibrator():
    groups = np.repeat(np.arange(15), 12)
    scored = []
    for fit, cal, score in three_way_group_splits(groups, 5):
        assert not set(groups[fit]) & set(groups[cal])
        assert not set(groups[fit]) & set(groups[score])
        assert not set(groups[cal]) & set(groups[score])
        assert len(fit) + len(cal) + len(score) == len(groups)
        scored.extend(score)
    assert sorted(scored) == list(range(len(groups)))


def test_selection_trains_once_per_candidate_fold_and_keeps_metrics_separate():
    groups = np.repeat(np.arange(15), 12)
    y = np.tile([0, 1, 2], 60)
    calls = []

    def fit_predict(candidate, fit, predict):
        assert not set(groups[fit]) & set(groups[predict])
        calls.append((candidate.key, len(fit)))
        # Synthetic scores only to exercise the selection protocol, no real model.
        return 0.1 + 0.7 * np.eye(3)[y[predict]]

    cfg = CalibrationConfig(method="sigmoid", epsilon=1e-6, sum_tolerance=1e-6,
                            isotonic_min_class_events=100, ece_bins=10, objective="f1_macro")
    winner, table, ranking = select_candidate([Candidate("stub", "none", {})], y, groups,
                                              fit_predict, cfg, calibration_methods=("sigmoid",))
    assert len(calls) == 5
    assert len(table) == 15
    assert winner["selection_objective"] == "argmax_f1_macro"
    assert winner["argmax_f1_macro"] == 1.0
    assert set(ranking.policy) == {"argmax", "ovr_f1", "joint_macro"}


def test_insufficient_groups_and_missing_classes_fail_explicitly():
    with pytest.raises(ValueError, match="group folds"):
        list(three_way_group_splits([0, 1], 5))
    with pytest.raises(ValueError, match="contiguous"):
        select_candidate([Candidate("stub", "none", {})], [0, 2] * 5, np.arange(10),
                         lambda *args: None, None)


def test_declared_search_budget_does_not_depend_on_observed_scores():
    grids = {"tree": {"depth": [2, 4, 8], "rate": [0.01, 0.1]}}
    first = declared_candidates("tree", "none", grids, 3)
    second = declared_candidates("tree", "none", grids, 3)
    assert first == second and len(first) == 3
    assert first[0].params == {"depth": 2, "rate": 0.01}
    assert first[-1].params == {"depth": 8, "rate": 0.1}
    assert declared_candidates("baseline", "none", grids)[0].params == {}
