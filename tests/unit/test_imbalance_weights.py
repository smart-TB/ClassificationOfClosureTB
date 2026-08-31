import numpy as np
import pandas as pd

from tb_outcomes.imbalance import load_imbalance_config, local_cost_weights


def test_weight_formula_matches_briefing_on_majority_class():
    y = pd.Series(["A"] * 8 + ["B"] * 2)
    strata = pd.Series(["s0"] * 10)
    w = local_cost_weights(y, strata, R=4.0, w_min=0.15, w_max=1.0, rescale_mean_one=False)
    # w_maioria = clip(4 * 2/8, .15, 1) = 1.0; minoria = 1.0
    assert np.allclose(w[y.values == "A"], 1.0)
    assert np.allclose(w[y.values == "B"], 1.0)


def test_weight_hits_floor_for_extreme_imbalance():
    y = pd.Series(["A"] * 100 + ["B"] * 1)
    strata = pd.Series(["s0"] * 101)
    w = local_cost_weights(y, strata, R=4.0, w_min=0.15, w_max=1.0, rescale_mean_one=False)
    # clip(4 * 1/100, .15, 1) = 0.15
    assert np.allclose(w[y.values == "A"], 0.15)


def test_degenerate_stratum_single_class_goes_to_floor():
    y = pd.Series(["A", "A", "B", "B"])
    strata = pd.Series(["so_A", "so_A", "mix", "mix"])
    w = local_cost_weights(y, strata, R=4.0, w_min=0.15, w_max=1.0, rescale_mean_one=False)
    assert np.allclose(w[:2], 0.15), "estrato de uma classe -> peso no piso"


def test_rescale_mean_one_normalizes_average_to_one():
    y = pd.Series(["A"] * 8 + ["B"] * 2)
    strata = pd.Series(["s0"] * 10)
    w = local_cost_weights(y, strata, R=4.0, w_min=0.15, w_max=1.0, rescale_mean_one=True)
    assert abs(w.mean() - 1.0) < 1e-9


def test_load_config_reads_real_file():
    cfg = load_imbalance_config("configs/imbalance.yaml")
    assert "none" in cfg.strategies
    assert cfg.cost_stratum == "year_region"
    assert cfg.R == 4.0
