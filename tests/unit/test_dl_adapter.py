import numpy as np
import pandas as pd
import pytest

from tb_outcomes.dl_config import DLConfig
from tb_outcomes.models import build_dl_adapter, load_models_config

ENTRIES = {**load_models_config("configs/models.yaml").models}
FAST = DLConfig(max_epochs=2, early_stopping_patience=2, validation_split=0.2,
                batch_size=256, seed=0)


def _frame(n=400, seed=0):
    rng = np.random.RandomState(seed)
    X = pd.DataFrame({
        "idade": rng.rand(n) * 80,
        "sexo": rng.choice(["M", "F"], n),
        "raca": rng.choice(["1", "2", "3"], n),
    })
    y = rng.choice([0, 1, 2], size=n)
    return X, y


@pytest.mark.integration  # treina uma rede pequena; fora dos unitários rápidos
def test_category_embedding_fits_and_predicts():
    X, y = _frame()
    ad = build_dl_adapter("category_embedding", ENTRIES["category_embedding"], FAST)
    ad.fit(X, y)
    proba = ad.predict_raw_scores(X)
    assert proba.shape == (len(X), 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-3)


@pytest.mark.integration
def test_tabnet_fits_and_predicts_with_nan():
    # tabnet quebrava quando NaN chegava ao sparsemax (support_size 0 -> index -1). O
    # DLAdapter agora imputa as contínuas com a mediana do treino. Este teste injeta NaN
    # de propósito e falha se a imputação sumir.
    X, y = _frame()
    X = X.copy()
    X.loc[X.index[:50], "idade"] = np.nan
    ad = build_dl_adapter("tabnet", ENTRIES["tabnet"], FAST)
    ad.fit(X, y)
    proba = ad.predict_raw_scores(X)
    assert proba.shape == (len(X), 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-3)


@pytest.mark.integration
def test_dl_refuses_nontrivial_weight():
    X, y = _frame()
    ad = build_dl_adapter("category_embedding", ENTRIES["category_embedding"], FAST)
    with pytest.raises(ValueError):
        ad.fit(X, y, sample_weight=np.linspace(0.1, 2.0, len(y)))
