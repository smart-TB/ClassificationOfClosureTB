import numpy as np
import pytest

from tb_outcomes.models import build_adapter, build_dl_adapter, load_models_config

CFG = load_models_config("configs/models.yaml")
X = np.random.RandomState(0).rand(60, 4)
y = np.array([0] * 40 + [1] * 20)


def _adapter(name):
    return build_adapter(name, CFG.models.get(name) or CFG.baselines[name])


def test_raw_scores_always_exist_even_without_proba():
    a = _adapter("ridge_classifier")
    a.fit(X, y)
    scores = a.predict_raw_scores(X)
    assert scores.shape[0] == X.shape[0]


def test_proba_if_native_is_none_for_ridge_and_linsvc_and_rbf():
    for name in ("ridge_classifier", "linear_svc", "rbf_svm"):
        a = _adapter(name)
        a.fit(X, y)
        assert a.predict_proba_if_native(X) is None


def test_proba_if_native_returns_array_for_random_forest():
    a = _adapter("random_forest")
    a.fit(X, y)
    p = a.predict_proba_if_native(X)
    assert p is not None and p.shape[0] == X.shape[0]


def test_weight_on_model_without_sample_weight_raises():
    a = _adapter("knn")  # sem sample_weight
    # peso variado (como o local_cost_sensitive produz), não uniforme
    w = np.where(y == 0, 0.25, 1.0)
    with pytest.raises(ValueError, match="sample_weight"):
        a.fit(X, y, sample_weight=w)


def test_trivial_weight_is_accepted_even_without_support():
    # peso None ou todos iguais não é uma tentativa real de ponderar
    a = _adapter("knn")
    a.fit(X, y, sample_weight=None)  # não levanta
    a.fit(X, y, sample_weight=np.ones(len(y)))  # trivial: não levanta


def test_dl_adapters_expose_the_same_interface():
    for name in ("category_embedding", "tabnet", "tab_transformer", "ft_transformer"):
        a = build_dl_adapter(name, CFG.models[name])
        assert hasattr(a, "fit") and hasattr(a, "predict_raw_scores")
        assert hasattr(a, "predict_proba_if_native")
        assert a.capabilities.native_categorical is True
        assert a.capabilities.sample_weight is False
        assert a.capabilities.gpu is True
        # cost é unsupported porque DL não recebe sample_weight aqui
        assert "local_cost_sensitive" not in a.capabilities.supported_imbalance
