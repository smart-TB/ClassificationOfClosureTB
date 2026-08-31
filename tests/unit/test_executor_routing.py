import numpy as np
import pandas as pd

from tb_outcomes.executor import build_model_input
from tb_outcomes.preprocess import infer_column_types, load_preprocess_config

PRE = load_preprocess_config("configs/preprocess.yaml")


def _frame(n=100):
    return pd.DataFrame({
        "idade": np.linspace(0, 80, n),
        "sexo": np.where(np.arange(n) % 2 == 0, "M", "F"),
    })


def test_matrix_kind_returns_numeric_array():
    X = _frame()
    types = infer_column_types(X, [], PRE.max_cardinality)
    tr, va = build_model_input(X, X, types, "onehot_scaled", "matrix", PRE)
    assert np.asarray(tr).dtype.kind in "fi"        # numérico


def test_categorical_frame_marks_category_dtype():
    X = _frame()
    types = infer_column_types(X, [], PRE.max_cardinality)
    tr, va = build_model_input(X, X, types, "native_categorical", "categorical_frame", PRE)
    assert str(tr["sexo"].dtype) == "category"
    assert tr["idade"].dtype.kind == "f"


def test_dl_frame_keeps_raw_columns():
    X = _frame()
    types = infer_column_types(X, [], PRE.max_cardinality)
    tr, va = build_model_input(X, X, types, "native_categorical", "dl_frame", PRE)
    assert set(tr.columns) == {"idade", "sexo"}
