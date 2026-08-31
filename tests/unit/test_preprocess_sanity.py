import numpy as np
import pandas as pd
import pytest

from tb_outcomes.preprocess import ColumnTypes, PreprocessError, assert_sane_columns


def test_constant_column_raises_naming_it():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "const": [5.0, 5.0, 5.0]})
    t = ColumnTypes(categorical=[], numeric=["a", "const"])
    with pytest.raises(PreprocessError, match="const"):
        assert_sane_columns(X, t, max_cardinality=40)


def test_all_na_column_raises():
    X = pd.DataFrame({"a": [1.0, 2.0], "vazia": [np.nan, np.nan]})
    t = ColumnTypes(categorical=[], numeric=["a", "vazia"])
    with pytest.raises(PreprocessError, match="vazia"):
        assert_sane_columns(X, t, max_cardinality=40)


def test_explosive_cardinality_raises():
    X = pd.DataFrame({"txt": [f"v{i}" for i in range(100)]})
    t = ColumnTypes(categorical=["txt"], numeric=[])
    with pytest.raises(PreprocessError, match="txt"):
        assert_sane_columns(X, t, max_cardinality=40)


def test_clean_columns_pass():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "cat": ["x", "y", "x"]})
    t = ColumnTypes(categorical=["cat"], numeric=["a"])
    assert_sane_columns(X, t, max_cardinality=40)
