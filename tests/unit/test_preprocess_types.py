import numpy as np
import pandas as pd

from tb_outcomes.preprocess import (
    coerce_for_sklearn,
    infer_column_types,
    load_preprocess_config,
)


class _Spec:
    def __init__(self, raw_name, type_, contextual=False, in_set=True):
        self.raw_name = raw_name
        self.type = type_
        self.contextual = contextual
        self.in_notification_set = in_set


def test_infer_types_splits_categorical_and_numeric():
    X = pd.DataFrame(
        {
            "CS_SEXO": pd.array(["M", "F"], dtype="string"),
            "IDADE": pd.array([30, 45], dtype="Int64"),
            "IPEA_IDHM": [0.8, 0.7],
        }
    )
    specs = [
        _Spec("CS_SEXO", "categorical"),
        _Spec("IDADE", "numeric"),
        _Spec("IPEA_IDHM", "numeric", contextual=True),
    ]
    t = infer_column_types(X, specs, max_cardinality=30)
    assert "CS_SEXO" in t.categorical
    assert "IDADE" in t.numeric and "IPEA_IDHM" in t.numeric


def test_coerce_makes_sklearn_happy_with_nullable_dtypes():
    # OneHotEncoder quebra com Int64/string anuláveis. A conversão é obrigatória.
    from sklearn.preprocessing import OneHotEncoder

    X = pd.DataFrame(
        {
            "sexo": pd.array(["M", None, "F"], dtype="string"),
            "idade": pd.array([30, None, 45], dtype="Int64"),
        }
    )
    types = infer_column_types(
        X, [_Spec("sexo", "categorical"), _Spec("idade", "numeric")], 30
    )
    Xc = coerce_for_sklearn(X, types)

    assert Xc["sexo"].dtype == object
    assert (Xc["sexo"] == "MISSING").sum() == 1
    OneHotEncoder(handle_unknown="ignore").fit(Xc[["sexo"]])  # não levanta

    assert Xc["idade"].dtype == np.float64
    assert Xc["idade"].isna().sum() == 1


def test_load_preprocess_config_reads_real_file():
    cfg = load_preprocess_config("configs/preprocess.yaml")
    assert cfg.max_cardinality > 0
    assert cfg.missing_token == "MISSING"
