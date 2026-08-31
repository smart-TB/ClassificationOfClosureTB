import numpy as np
import pandas as pd
import pytest

from tb_outcomes.leakage import (
    assert_no_group_overlap,
    assert_outcome_count_coherence,
    assert_reproducible,
    assert_strata_out_of_features,
    assert_transformers_fit_on_train_only,
)


def test_group_overlap_detected():
    with pytest.raises(ValueError, match="cluster"):
        assert_no_group_overlap([1, 2, 3], [3, 4])
    assert_no_group_overlap([1, 2], [3, 4])  # não levanta


def test_strata_columns_refused_in_features():
    with pytest.raises(ValueError, match="NU_ANO"):
        assert_strata_out_of_features(["idade", "NU_ANO", "sexo"])
    with pytest.raises(ValueError, match="ID_MN_RESI"):
        assert_strata_out_of_features(["ID_MN_RESI", "sexo"])
    assert_strata_out_of_features(["idade", "sexo"])  # não levanta


def test_count_coherence():
    y = pd.Series([0, 1, 2, 1, 0])
    assert_outcome_count_coherence(y, expected_n=5)
    with pytest.raises(ValueError):
        assert_outcome_count_coherence(y, expected_n=6)


def test_reproducibility_check():
    a = [{"metric": "f1", "value": 0.5}]
    assert_reproducible(a, [{"metric": "f1", "value": 0.5}])
    with pytest.raises(ValueError, match="reprodut"):
        assert_reproducible(a, [{"metric": "f1", "value": 0.51}])


def test_fit_indices_must_not_touch_eval():
    assert_transformers_fit_on_train_only(np.array([0, 1, 2]), np.array([3, 4]))
    with pytest.raises(ValueError, match="avaliação"):
        assert_transformers_fit_on_train_only(np.array([0, 1, 3]), np.array([3, 4]))
