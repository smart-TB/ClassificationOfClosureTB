import numpy as np
import pandas as pd

from tb_outcomes.splits import assign_clusters_binpack, make_outer_folds


def test_binpack_puts_largest_cluster_in_emptiest_fold():
    # 5 clusters, um gigante. Bin-packing deve isolá-lo e equilibrar o resto.
    sizes = pd.Series({0: 1000, 1: 250, 2: 250, 3: 250, 4: 250}, name="n")
    fa = assign_clusters_binpack(sizes, n_folds=5)
    assert len({fa[0], fa[1], fa[2], fa[3], fa[4]}) == 5


def test_binpack_is_deterministic_with_ties():
    sizes = pd.Series({3: 100, 1: 100, 2: 50, 0: 50}, name="n")
    a = assign_clusters_binpack(sizes, n_folds=5)
    b = assign_clusters_binpack(sizes, n_folds=5)
    assert a == b


def test_binpack_balances_fold_sizes():
    rng = np.random.default_rng(0)
    sizes = pd.Series({i: int(v) for i, v in enumerate(rng.integers(10, 1000, 40))}, name="n")
    fa = assign_clusters_binpack(sizes, n_folds=5)
    por_dobra = pd.Series(sizes.values, index=[fa[c] for c in sizes.index]).groupby(level=0).sum()
    assert por_dobra.max() / por_dobra.min() < 1.3


def test_make_outer_folds_keeps_clusters_whole():
    # Nenhum cluster pode aparecer em duas dobras — é o bloqueio espacial.
    df = pd.DataFrame({"ID_MUNIC_ANALISE": pd.array(list(range(100)), dtype="Int64")})
    rc = pd.Series(np.repeat(range(10), 10), name="cluster")
    y = pd.Series(np.tile([1, 2, 3, 1, 2, 3, 1, 2, 3, 1], 10))
    folds = make_outer_folds(df, rc, y, n_folds=5, method="binpack")
    por_cluster = pd.DataFrame({"cluster": rc, "fold": folds})
    assert (por_cluster.groupby("cluster").fold.nunique() == 1).all()


def test_make_outer_folds_covers_all_classes_per_fold():
    df = pd.DataFrame({"ID_MUNIC_ANALISE": pd.array(list(range(150)), dtype="Int64")})
    rc = pd.Series(np.repeat(range(15), 10), name="cluster")
    y = pd.Series(np.tile([1, 2, 3], 50))
    folds = make_outer_folds(df, rc, y, n_folds=5, method="binpack")
    for f in folds.unique():
        assert y[folds == f].nunique() == 3, f"dobra {f} perdeu uma classe"
