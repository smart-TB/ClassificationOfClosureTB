import numpy as np
import pandas as pd

from tb_outcomes.executor import inner_oof_scores


def _dev(n=400, seed=0):
    rng = np.random.RandomState(seed)
    y = pd.Series(rng.choice([0, 1, 2], size=n))
    clusters = pd.Series(rng.randint(0, 10, size=n))  # 10 clusters em DEV
    return y, clusters


def test_every_row_is_covered_exactly_once():
    y, clusters = _dev()
    seen = np.zeros(len(y), dtype=int)

    def fit_predict_fn(train_idx, val_idx):
        seen[val_idx] += 1
        return np.tile([0.2, 0.3, 0.5], (len(val_idx), 1))

    oof = inner_oof_scores(y, clusters, n_classes=3, n_inner=5, fit_predict_fn=fit_predict_fn)
    assert oof.shape == (len(y), 3)
    assert (seen == 1).all()               # cobertura total, sem sobreposição
    assert not np.isnan(oof).any()


def test_train_and_val_never_share_a_cluster():
    y, clusters = _dev()
    violations = []

    def fit_predict_fn(train_idx, val_idx):
        tr_cl = set(clusters.iloc[train_idx])
        va_cl = set(clusters.iloc[val_idx])
        if tr_cl & va_cl:
            violations.append((tr_cl & va_cl))
        return np.zeros((len(val_idx), 3)) + [0.2, 0.3, 0.5]

    inner_oof_scores(y, clusters, n_classes=3, n_inner=5, fit_predict_fn=fit_predict_fn)
    assert violations == []
