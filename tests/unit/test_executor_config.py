import pytest

from tb_outcomes.executor import ExecutorConfig, load_executor_config

CFG = load_executor_config("configs/executor.yaml")


def test_strategy_grid_is_three_without_none():
    assert CFG.strategies == [
        "random_undersampling",
        "random_oversampling",
        "local_cost_sensitive",
    ]
    assert "none" not in CFG.strategies


def test_full_matrix_present_including_dl():
    # Grade de produção: os 4 DL presentes (tabnet corrigido). rbf_svm e linear_svc
    # excluídos a priori (SVC/liblinear não escalam) -> 26 modelos.
    assert "majority_class" in CFG.models
    assert "random_forest" in CFG.models
    for dl in ("category_embedding", "tabnet", "tab_transformer", "ft_transformer"):
        assert dl in CFG.models
    assert "rbf_svm" not in CFG.models
    assert "linear_svc" not in CFG.models
    assert len(CFG.models) == 26


def test_unknown_field_type_is_rejected():
    with pytest.raises(Exception):
        ExecutorConfig(
            outcome_schema="revised_4class", strategies=[], n_outer_folds="cinco",
            n_inner_folds=5, fold_method="binpack", k=50, random_state=42,
            sanity_subsample=3000, models=[],
        )
