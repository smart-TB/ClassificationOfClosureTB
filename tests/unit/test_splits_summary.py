import numpy as np
import pandas as pd

from tb_outcomes.splits import cluster_summary, fold_summary, make_inner_folds


def _setup(n_clusters=15, per=20):
    rc = pd.Series(np.repeat(range(n_clusters), per), name="cluster")
    y = pd.Series(np.tile([1, 2, 3], n_clusters * per // 3))
    return rc, y


def test_inner_folds_never_share_a_cluster():
    rc, y = _setup()
    for tr, va in make_inner_folds(rc, y, n_folds=5, method="binpack"):
        assert not (set(rc.iloc[tr]) & set(rc.iloc[va])), "cluster vazou entre dobras internas"


def test_inner_folds_yield_five_splits():
    rc, y = _setup()
    splits = list(make_inner_folds(rc, y, n_folds=5, method="binpack"))
    assert len(splits) == 5


def test_cluster_summary_counts_municipalities_and_events():
    rc = pd.Series([0, 0, 0, 1, 1], name="cluster")
    y = pd.Series([1, 3, 3, 1, 2])
    mun = pd.DataFrame({"cluster": [0, 0, 1]}, index=[10, 11, 12])
    s = cluster_summary(rc, y, mun).set_index("cluster")
    assert s.loc[0, "n_records"] == 3
    assert s.loc[0, "n_municipalities"] == 2
    assert s.loc[0, "events_death"] == 2


def test_fold_summary_totals_match_the_cohort():
    rc, y = _setup()
    folds = pd.Series(np.repeat(range(5), len(y) // 5), name="outer_fold")
    s = fold_summary(folds, rc, y)
    assert int(s.n.sum()) == len(y)
