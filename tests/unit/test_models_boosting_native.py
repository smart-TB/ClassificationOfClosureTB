import numpy as np
import pandas as pd

from tb_outcomes.models import build_adapter, load_models_config

ENTRIES = {**load_models_config("configs/models.yaml").baselines,
           **load_models_config("configs/models.yaml").models}


def _frame(n=200, seed=0):
    rng = np.random.RandomState(seed)
    X = pd.DataFrame({
        "idade": rng.rand(n) * 80,
        "sexo": pd.Series(rng.choice(["M", "F"], n)).astype("category"),
        "raca": pd.Series(rng.choice(["1", "2", "3"], n)).astype("category"),
    })
    y = rng.choice([0, 1, 2], size=n)
    return X, y


def _check_model(name):
    X, y = _frame()
    ad = build_adapter(name, ENTRIES[name])
    ad.fit(X, y)
    proba = ad.predict_raw_scores(X)
    assert proba.shape == (len(X), 3)
    assert np.isfinite(proba).all()


def test_xgboost_native_categorical():
    _check_model("xgboost")


def test_lightgbm_native_categorical():
    _check_model("lightgbm")


def test_histgb_native_categorical():
    _check_model("hist_gradient_boost")


def test_catboost_native_categorical():
    _check_model("catboost")
