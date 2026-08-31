import pytest

from tb_outcomes.calibration import CalibrationConfig, load_calibration_config

CFG = load_calibration_config("configs/calibration.yaml")


def test_loads_pre_registered_guard_threshold():
    assert CFG.method == "auto"
    assert CFG.isotonic_min_class_events == 100
    assert 0 < CFG.epsilon < 1e-3


def test_method_must_be_a_known_value():
    with pytest.raises(Exception):
        CalibrationConfig(
            method="banana",
            epsilon=1e-6,
            sum_tolerance=1e-6,
            isotonic_min_class_events=100,
            ece_bins=10,
            objective="f1_macro",
        )
