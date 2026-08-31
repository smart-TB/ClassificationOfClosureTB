import numpy as np
import pandas as pd

from tb_outcomes.imbalance import apply_strategy, load_imbalance_config

CFG = load_imbalance_config("configs/imbalance.yaml")


def _train():
    y = pd.Series(["cure"] * 80 + ["death"] * 20)
    strata = pd.Series(["2019_SE"] * 100)
    return y, strata


def test_none_keeps_the_training_set_intact():
    y, s = _train()
    r = apply_strategy(y, s, "none", CFG, random_state=42, supports_weight=True)
    assert len(r.train_index) == len(y)
    assert r.sample_weight is None


def test_oversampling_duplicates_the_minority():
    y, s = _train()
    r = apply_strategy(y, s, "random_oversampling", CFG, random_state=42, supports_weight=True)
    resampled = y.iloc[r.train_index]
    assert (resampled == "death").sum() > 20, "minoria duplicada"
    assert len(r.train_index) > len(y)


def test_undersampling_cuts_the_majority():
    y, s = _train()
    r = apply_strategy(y, s, "random_undersampling", CFG, random_state=42, supports_weight=True)
    resampled = y.iloc[r.train_index]
    assert (resampled == "cure").sum() < 80, "maioria cortada"
    assert len(r.train_index) < len(y)


def test_cost_passes_weights_without_resampling():
    # Desbalanceamento forte (16:1) para o peso da maioria ficar abaixo de 1:
    # clip(4 * 6/96, .15, 1) = clip(0.25, .15, 1) = 0.25.
    y = pd.Series(["cure"] * 96 + ["death"] * 6)
    s = pd.Series(["2019_SE"] * 102)
    r = apply_strategy(y, s, "local_cost_sensitive", CFG, random_state=42, supports_weight=True)
    assert len(r.train_index) == len(y), "custo não reamostra"
    assert r.sample_weight is not None
    # a maioria (cura) recebe peso < 1; a minoria fica em 1 -> pesos não-triviais
    assert r.sample_weight.min() < r.sample_weight.max(), "pesos não-triviais"


def test_cost_is_unsupported_when_estimator_lacks_sample_weight():
    y, s = _train()
    r = apply_strategy(
        y, s, "local_cost_sensitive", CFG, random_state=42, supports_weight=False
    )
    assert r.strategy == "unsupported", "sem sample_weight -> unsupported, não simulação"
    assert r.sample_weight is None


def test_resampling_is_deterministic_by_seed():
    y, s = _train()
    a = apply_strategy(y, s, "random_oversampling", CFG, 7, True).train_index
    b = apply_strategy(y, s, "random_oversampling", CFG, 7, True).train_index
    np.testing.assert_array_equal(a, b)


def test_oversampling_caps_at_max_ratio():
    # cap 1:R -> a minoritária sobe só até majoritária/R; razão maj:min <= R.
    import numpy as np
    import pandas as pd

    from tb_outcomes.imbalance import apply_strategy, load_imbalance_config

    cfg = load_imbalance_config("configs/imbalance.yaml")
    y = pd.Series(np.r_[np.zeros(9000), np.ones(300), np.full(150, 2)].astype(int))
    strata = pd.Series(np.zeros(len(y)))
    r = apply_strategy(y, strata, "random_oversampling", cfg, 0, supports_weight=False)
    counts = pd.Series(y.to_numpy()[r.train_index]).value_counts()
    assert counts.max() / counts.min() <= cfg.oversample_max_ratio + 1e-9
    assert int(counts.min()) == 9000 // cfg.oversample_max_ratio  # minoritárias no piso
