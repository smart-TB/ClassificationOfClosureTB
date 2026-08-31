import numpy as np
import pytest

from tb_outcomes.calibration import (
    calibration_manifest,
    calibration_metrics,
    classify_argmax,
    classify_policy,
    fit_calibrator,
    fit_thresholds,
    load_calibration_config,
)

pytestmark = pytest.mark.integration
CFG = load_calibration_config("configs/calibration.yaml")


def test_full_calibration_flow_on_synthetic_oof():
    rng = np.random.RandomState(3)
    n = 4000
    y = rng.choice([0, 1, 2], size=n, p=[0.6, 0.3, 0.1])
    raw = np.zeros((n, 3))
    for c in range(3):
        raw[:, c] = (y == c) * 0.6 + rng.rand(n) * 0.4

    cal = fit_calibrator(raw, y, CFG)          # ajusta só no OOF
    probs = cal.transform(raw)                 # calibra + clip + renormaliza + valida
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)

    thr = fit_thresholds(probs, y, CFG)        # limiares só no OOF
    pred_a = classify_argmax(probs, cal.classes)
    pred_p = classify_policy(probs, thr, cal.classes)
    assert len(pred_a) == n and len(pred_p) == n

    pre = calibration_metrics(y, raw / raw.sum(axis=1, keepdims=True), CFG, stage="pre_renorm")
    post = calibration_metrics(y, probs, CFG, stage="post_renorm")
    assert {r["stage"] for r in pre} == {"pre_renorm"}
    assert {r["stage"] for r in post} == {"post_renorm"}

    man = calibration_manifest(cal)
    assert man["n_classes"] == 3
