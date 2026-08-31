import numpy as np

from tb_outcomes.calibration import (
    calibration_metrics,
    expected_calibration_error,
    load_calibration_config,
)

CFG = load_calibration_config("configs/calibration.yaml")


def test_ece_is_zero_for_perfectly_calibrated():
    # p == frequência real por bin => ECE ~ 0
    rng = np.random.RandomState(0)
    p = rng.rand(5000)
    y = (rng.rand(5000) < p).astype(int)
    assert expected_calibration_error(y, p, n_bins=10) < 0.05


def test_metrics_have_both_stages_per_class():
    rng = np.random.RandomState(0)
    y = rng.choice([0, 1, 2], size=1000)
    probs = rng.dirichlet([1, 1, 1], size=1000)
    pre = calibration_metrics(y, probs, CFG, stage="pre_renorm")
    post = calibration_metrics(y, probs, CFG, stage="post_renorm")
    assert all(r["stage"] == "pre_renorm" for r in pre)
    assert all(r["stage"] == "post_renorm" for r in post)
    assert any(r["class"] == "__global__" for r in pre)


def test_proper_scores_present():
    rng = np.random.RandomState(0)
    y = rng.choice([0, 1], size=500)
    probs = rng.dirichlet([1, 1], size=500)
    g = [r for r in calibration_metrics(y, probs, CFG, stage="post_renorm") if r["class"] == "__global__"][0]
    assert "log_loss" in g and "brier_multiclass" in g
