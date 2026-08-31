import numpy as np
import pandas as pd

from tb_outcomes.splits import (
    assign_clusters_to_records,
    assignment_hash,
    fit_spatial_clusters,
)


def _mun_table(n=60, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "lat": rng.uniform(-30, 0, n),
            "lon": rng.uniform(-70, -40, n),
            "n": rng.integers(1, 500, n),
        },
        index=pd.Index(range(n), name="ID_MUNIC_ANALISE"),
    )


def test_clusters_are_deterministic_by_seed():
    t = _mun_table()
    a = fit_spatial_clusters(t, k=5, epsg=5880, random_state=42)
    b = fit_spatial_clusters(t, k=5, epsg=5880, random_state=42)
    pd.testing.assert_frame_equal(a, b)
    assert assignment_hash(a) == assignment_hash(b)


def test_different_seed_changes_the_hash():
    t = _mun_table()
    a = fit_spatial_clusters(t, k=5, epsg=5880, random_state=42)
    c = fit_spatial_clusters(t, k=5, epsg=5880, random_state=11)
    assert assignment_hash(a) != assignment_hash(c)


def test_every_municipality_gets_exactly_one_cluster():
    t = _mun_table()
    a = fit_spatial_clusters(t, k=5, epsg=5880, random_state=42)
    assert len(a) == len(t)
    assert a["cluster"].between(0, 4).all()
    assert a["cluster"].nunique() == 5


def test_assign_clusters_back_to_records():
    t = _mun_table(n=10)
    clusters = fit_spatial_clusters(t, k=3, epsg=5880, random_state=42)
    df = pd.DataFrame({"ID_MUNIC_ANALISE": pd.array([0, 0, 5, 9], dtype="Int64")})
    r = assign_clusters_to_records(df, clusters, key="ID_MUNIC_ANALISE")
    assert r.iloc[0] == r.iloc[1], "mesmo município, mesmo cluster"
    assert r.notna().all()
