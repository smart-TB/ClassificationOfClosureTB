import numpy as np
import pandas as pd

from tb_outcomes.preprocess import categorical_encoder, numeric_pipeline


def test_numeric_pipeline_imputes_with_train_median_and_adds_indicator():
    p = numeric_pipeline(scale=True, nonnegative=False)
    Xtr = pd.DataFrame({"a": [1.0, 3.0, 5.0, np.nan]})  # mediana do treino = 3
    p.fit(Xtr)
    out = p.transform(pd.DataFrame({"a": [np.nan]}))
    assert out.shape[1] == 2, "valor escalado + indicador de ausência"
    assert out[0, 1] == 1.0, "indicador de ausência = 1"


def test_missing_indicator_has_fixed_shape_even_without_na_in_train():
    # A sutileza: MissingIndicator(features='all') dá indicador mesmo sem <NA>
    # no treino, senão o shape variaria entre dobras (quebraria o SP4e).
    p = numeric_pipeline(scale=False, nonnegative=False)
    Xtr = pd.DataFrame({"a": [1.0, 2.0, 3.0]})  # SEM ausência no treino
    p.fit(Xtr)
    out = p.transform(pd.DataFrame({"a": [np.nan]}))
    assert out.shape[1] == 2, "indicador presente mesmo sem <NA> no treino"
    assert out[0, 1] == 1.0


def test_numeric_pipeline_nonnegative_never_negative():
    p = numeric_pipeline(scale=False, nonnegative=True)
    Xtr = pd.DataFrame({"a": [10.0, 20.0, 30.0]})
    p.fit(Xtr)
    out = p.transform(pd.DataFrame({"a": [-100.0, 0.0, 1000.0]}))
    assert (out >= 0).all(), "nonnegative não pode produzir valor negativo"


def test_categorical_onehot_has_missing_level_and_ignores_unknown():
    enc = categorical_encoder(kind="onehot")
    Xtr = pd.DataFrame({"c": ["x", "y", "MISSING"]})
    enc.fit(Xtr)
    out = enc.transform(pd.DataFrame({"c": ["z"]}))
    assert out.sum() == 0, "categoria desconhecida vira vetor zero (handle_unknown=ignore)"
