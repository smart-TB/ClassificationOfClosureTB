from tb_outcomes.models import build_estimator, load_models_config, probe_capabilities

CFG = load_models_config("configs/models.yaml")


def _cap(name):
    entry = CFG.models[name]
    return probe_capabilities(build_estimator(name, entry), name, entry)


def test_models_without_predict_proba_are_detected():
    for name in ("ridge_classifier", "linear_svc", "rbf_svm"):
        c = _cap(name)
        assert c.native_probability is False
        assert c.calibrator_required is True


def test_models_without_sample_weight_are_detected():
    for name in ("lda", "qda", "knn", "mlp"):
        c = _cap(name)
        assert c.sample_weight is False
        assert "local_cost_sensitive" not in c.supported_imbalance


def test_probaful_models_do_not_require_calibrator():
    for name in ("random_forest", "logistic_regression", "gaussian_nb"):
        c = _cap(name)
        assert c.native_probability is True
        assert c.calibrator_required is False


def test_nonnegative_flag_matches_nb():
    assert (
        probe_capabilities(
            build_estimator("multinomial_nb", CFG.models["multinomial_nb"]),
            "multinomial_nb",
            CFG.models["multinomial_nb"],
        ).nonnegative_input
        is True
    )
    assert _cap("random_forest").nonnegative_input is False


def test_boosting_trio_supports_gpu():
    # XGBoost, LightGBM, CatBoost aceleram em GPU (defeito corrigido: não eram só os DL)
    for name in ("xgboost", "lightgbm", "catboost"):
        assert _cap(name).gpu is True
    # GradientBoosting/HistGB/AdaBoost são sklearn CPU-only
    for name in ("gradient_boosting", "hist_gradient_boost", "adaboost"):
        assert _cap(name).gpu is False


def test_capable_boosters_are_native_categorical():
    # XGBoost, LightGBM, HistGB e CatBoost consomem categóricas sem one-hot
    for name in ("xgboost", "lightgbm", "hist_gradient_boost", "catboost"):
        c = _cap(name)
        assert c.native_categorical is True
        assert c.preprocess_profile == "native_categorical"


def test_capabilities_are_measured_not_hardcoded():
    # a prova real: probe olha o estimador, então concorda com sklearn diretamente
    est = build_estimator("ridge_classifier", CFG.models["ridge_classifier"])
    assert hasattr(est, "predict_proba") == _cap("ridge_classifier").native_probability
