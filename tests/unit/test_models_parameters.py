import pytest

from tb_outcomes.models import build_adapter, load_models_config

ENTRIES = load_models_config("configs/models.yaml").models


@pytest.mark.parametrize("name", ["random_forest", "extra_trees", "decision_tree",
                                 "mlp", "hist_gradient_boost", "xgboost", "lightgbm"])
def test_classical_seed_is_propagated(name):
    ad = build_adapter(name, ENTRIES[name], random_state=1337)
    assert ad._est.get_params()["random_state"] == 1337


def test_catboost_seed_and_explicit_parameters():
    ad = build_adapter("catboost", ENTRIES["catboost"], random_state=1337,
                       params={"depth": 4, "iterations": 100})
    assert ad._est.get_params()["random_seed"] == 1337
    assert ad._est.get_params()["depth"] == 4


def test_explicit_parameters_are_applied_without_changing_default():
    tuned = build_adapter("lightgbm", ENTRIES["lightgbm"], params={"num_leaves": 15})
    reference = build_adapter("lightgbm", ENTRIES["lightgbm"])
    assert tuned._est.get_params()["num_leaves"] == 15
    assert reference._est.get_params()["num_leaves"] == 31
    with pytest.raises(ValueError, match="desconhecidos"):
        build_adapter("lightgbm", ENTRIES["lightgbm"], params={"num_leafs": 15})
