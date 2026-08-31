import numpy as np
import pytest

from tb_outcomes.calibration import (
    assert_calibrated_before_metrics,
    calibration_manifest,
    fit_calibrator,
    load_calibration_config,
)
from tb_outcomes.models import build_registry, load_models_config

CFG = load_calibration_config("configs/calibration.yaml")
MREG = build_registry(load_models_config("configs/models.yaml"))


def _raw(y):
    rng = np.random.RandomState(1)
    raw = np.zeros((len(y), 2))
    for c in range(2):
        raw[:, c] = (y == c) * 0.6 + rng.rand(len(y)) * 0.4
    return raw


def test_rich_class_uses_isotonic():
    y = np.array([0] * 900 + [1] * 900)  # 900 eventos > 100
    cal = fit_calibrator(_raw(y), y, CFG)
    assert cal.method_per_class[1] == "isotonic"


def test_rare_class_falls_back_to_platt():
    y = np.array([0] * 1950 + [1] * 50)  # 50 eventos < 100 -> Platt
    cal = fit_calibrator(_raw(y), y, CFG)
    assert cal.method_per_class[1] == "sigmoid"


def test_method_isotonic_forces_isotonic_even_when_rare():
    y = np.array([0] * 1950 + [1] * 50)
    cal = fit_calibrator(_raw(y), y, CFG, method="isotonic")
    assert cal.method_per_class[1] == "isotonic"


def test_uncalibrated_calibrator_required_model_is_refused():
    caps = MREG["ridge_classifier"].capabilities  # calibrator_required=True
    with pytest.raises(ValueError, match="calibr"):
        assert_calibrated_before_metrics(caps, calibrated=False)
    assert_calibrated_before_metrics(caps, calibrated=True)  # não levanta


def test_native_proba_model_is_allowed_uncalibrated():
    caps = MREG["random_forest"].capabilities  # calibrator_required=False
    assert_calibrated_before_metrics(caps, calibrated=False)  # não levanta


def test_manifest_reports_method_fractions():
    y = np.array([0] * 1950 + [1] * 50)  # classe 1 rara -> Platt
    cal = fit_calibrator(_raw(y), y, CFG)
    man = calibration_manifest(cal)
    assert man["n_classes"] == 2
    assert man["n_sigmoid"] >= 1  # a guarda registrou o fallback
    assert 0.0 <= man["frac_isotonic"] <= 1.0
