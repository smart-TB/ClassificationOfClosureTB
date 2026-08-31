import numpy as np
import pytest

from tb_outcomes.calibration import postprocess_probabilities


def test_output_rows_sum_to_one_and_are_clipped():
    p = np.array([[0.0, 0.2, 0.8], [0.5, 0.5, 0.0]])
    out = postprocess_probabilities(p, epsilon=1e-6, tol=1e-6)
    np.testing.assert_allclose(out.sum(axis=1), 1.0, atol=1e-6)
    # a renormalização após o clip pode nudgar o mínimo marginalmente abaixo de
    # epsilon (ponto-flutuante); mesma folga 1e-9 que o limite superior já usa.
    assert out.min() >= 1e-6 - 1e-9 and out.max() <= 1 - 1e-6 + 1e-9


def test_nan_input_raises():
    p = np.array([[np.nan, 0.5, 0.5]])
    with pytest.raises(ValueError, match="NaN"):
        postprocess_probabilities(p, epsilon=1e-6, tol=1e-6)


def test_negative_input_raises():
    p = np.array([[-0.1, 0.6, 0.5]])
    with pytest.raises(ValueError, match="negativ"):
        postprocess_probabilities(p, epsilon=1e-6, tol=1e-6)


def test_all_zero_row_raises_because_it_cannot_renormalize():
    p = np.array([[0.0, 0.0, 0.0]])
    with pytest.raises(ValueError):
        postprocess_probabilities(p, epsilon=1e-6, tol=1e-6)
