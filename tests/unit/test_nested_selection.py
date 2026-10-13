import numpy as np
import pytest

from tb_outcomes.calibration import CalibrationConfig
from tb_outcomes.model_selection import Candidate
from tb_outcomes.nested_selection import fit_selected_pipeline


@pytest.mark.parametrize("fold_method", ["binpack", "random_group"])
def test_external_outcomes_are_not_needed_and_final_fit_occurs_once(fold_method):
    y = np.tile([0, 1, 2], 60)
    groups = np.repeat(np.arange(15), 12)
    calls = []

    def development(candidate, fit, pred):
        assert not set(groups[fit]) & set(groups[pred])
        calls.append("development")
        return np.eye(3)[y[pred]] * 0.8 + 0.1

    def evaluation(candidate):
        calls.append("evaluation")
        return np.array([[0.8, 0.1, 0.1], [0.1, 0.1, 0.8]])

    cfg = CalibrationConfig(method="sigmoid", epsilon=1e-6, sum_tolerance=1e-6,
                            isotonic_min_class_events=100, ece_bins=10, objective="f1_macro")
    result = fit_selected_pipeline([Candidate("stub", "none", {})], y, groups,
                                    development, evaluation, cfg,
                                    n_selection_folds=3, n_calibration_folds=3,
                                    calibration_methods=("sigmoid",), fold_method=fold_method)
    assert calls == ["development"] * 6 + ["evaluation"]
    assert result["pred_argmax"].tolist() == [0, 2]
    assert result["probabilities"].shape == (2, 3)
    assert np.allclose(result["probabilities"].sum(axis=1), 1)
