import numpy as np
import pytest

from tb_outcomes.executor import as_score_matrix


def test_proba_matrix_passes_through():
    p = np.array([[0.2, 0.8], [0.6, 0.4]])
    out = as_score_matrix(p, n_classes=2)
    np.testing.assert_allclose(out, p)


def test_binary_decision_1d_becomes_two_columns():
    d = np.array([-1.5, 0.0, 2.0])
    out = as_score_matrix(d, n_classes=2)
    assert out.shape == (3, 2)
    np.testing.assert_allclose(out[:, 1], d)
    np.testing.assert_allclose(out[:, 0], -d)


def test_multiclass_matrix_passes_through():
    m = np.arange(12, dtype=float).reshape(4, 3)
    out = as_score_matrix(m, n_classes=3)
    assert out.shape == (4, 3)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        as_score_matrix(np.zeros((4, 2)), n_classes=3)
