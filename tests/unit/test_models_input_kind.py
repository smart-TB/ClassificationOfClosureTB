from tb_outcomes.models import build_registry, load_models_config

REG = build_registry(load_models_config("configs/models.yaml"))


def test_matrix_kind_for_onehot_models():
    assert REG["logistic_plain"].input_kind == "matrix"
    assert REG["random_forest"].input_kind == "matrix"


def test_categorical_frame_for_native_boosting():
    for m in ("xgboost", "lightgbm", "hist_gradient_boost", "catboost"):
        assert REG[m].input_kind == "categorical_frame"


def test_dl_frame_for_dl_models():
    for m in ("category_embedding", "tabnet", "tab_transformer", "ft_transformer"):
        assert REG[m].input_kind == "dl_frame"
