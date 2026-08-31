import numpy as np

from tb_outcomes.calibration import fit_calibrator, load_calibration_config

CFG = load_calibration_config("configs/calibration.yaml")


def _synthetic(n=2000, seed=0):
    rng = np.random.RandomState(seed)
    y = rng.choice([0, 1, 2], size=n, p=[0.6, 0.3, 0.1])
    # escores OOF ruidosos mas informativos, um por classe
    raw = np.zeros((n, 3))
    for c in range(3):
        raw[:, c] = (y == c) * 0.6 + rng.rand(n) * 0.4
    return raw, y


def test_transform_returns_valid_probabilities():
    raw, y = _synthetic()
    cal = fit_calibrator(raw, y, CFG)
    p = cal.transform(raw)
    assert p.shape == raw.shape
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-6)


def test_calibration_improves_or_keeps_brier():
    # calibrar não deve piorar grosseiramente o Brier no próprio OOF
    from sklearn.metrics import brier_score_loss

    raw, y = _synthetic()
    cal = fit_calibrator(raw, y, CFG)
    p = cal.transform(raw)
    before = brier_score_loss((y == 2).astype(int), raw[:, 2] / raw.sum(axis=1))
    after = brier_score_loss((y == 2).astype(int), p[:, 2])
    assert after <= before + 0.05
