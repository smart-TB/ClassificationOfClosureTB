import numpy as np
import pandas as pd
import pytest

from tb_outcomes.imbalance import (
    ImbalanceResult,
    apply_strategy,
    assert_strategies_differ,
    config_signature,
    imbalance_audit,
    load_imbalance_config,
)

CFG = load_imbalance_config("configs/imbalance.yaml")


def _train():
    return pd.Series(["cure"] * 80 + ["death"] * 20), pd.Series(["2019_SE"] * 100)


def _all_strategies():
    y, s = _train()
    return {
        st: apply_strategy(y, s, st, CFG, 42, supports_weight=True)
        for st in (
            "none",
            "random_oversampling",
            "random_undersampling",
            "local_cost_sensitive",
        )
    }


def test_the_four_strategies_have_distinct_signatures():
    assert_strategies_differ(_all_strategies())


def test_identical_strategies_are_caught():
    y, _ = _train()
    idx = np.arange(len(y))
    results = {
        "over": ImbalanceResult(idx, None, "over"),
        "under": ImbalanceResult(idx, None, "under"),  # idêntica -> defeito
    }
    with pytest.raises(AssertionError, match="idêntica"):
        assert_strategies_differ(results)


def test_signature_captures_rows_and_weights():
    y, s = _train()
    none = apply_strategy(y, s, "none", CFG, 42, True)
    cost = apply_strategy(y, s, "local_cost_sensitive", CFG, 42, True)
    assert config_signature(none) != config_signature(cost)


def test_audit_records_counts_before_and_after():
    a = imbalance_audit(_train()[0], _all_strategies())
    assert a["random_oversampling"]["n_after"] > a["none"]["n_after"]
    assert a["random_undersampling"]["n_after"] < a["none"]["n_after"]
    assert "weight_min" in a["local_cost_sensitive"]
