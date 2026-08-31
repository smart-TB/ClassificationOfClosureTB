"""Vazamento e equilíbrio das dobras sobre a coorte real."""
from pathlib import Path

import pandas as pd
import pytest

from tb_outcomes import splits as sp
from tb_outcomes.cohort import TargetSchema, build_cohort, load_outcome_rules
from tb_outcomes.config import load_config
from tb_outcomes.data import COLUNA_MUNICIPIO_RESOLVIDO, harmonize, load_raw_inputs

DATA = Path("data")
pytestmark = pytest.mark.regression


@pytest.fixture(scope="module")
def cohort_y():
    if not (DATA / "sinnan.parquet").exists():
        pytest.skip("brutos ausentes; rode o pipeline de aquisição.")
    cfg = load_config(Path("configs/analysis_decisions.yaml"))
    df = harmonize(load_raw_inputs(DATA), cfg)
    regras = load_outcome_rules(Path("configs/outcomes.yaml"))
    r = build_cohort(df, cfg, regras, TargetSchema.LEGACY_3CLASS)
    y = r.analytic["target_legacy_3class"].map(
        {"cure": 1, "treatment_interruption": 2, "tb_attributed_death": 3}
    )
    return r.analytic, y


def _folds(cohort, y, k, method="binpack"):
    mun = sp.build_municipality_table(cohort, key=COLUNA_MUNICIPIO_RESOLVIDO)
    clusters = sp.fit_spatial_clusters(mun, k, 5880, random_state=42)
    rc = sp.assign_clusters_to_records(cohort, clusters, key=COLUNA_MUNICIPIO_RESOLVIDO)
    return rc, sp.make_outer_folds(cohort, rc, y, n_folds=5, method=method)


def test_no_cluster_leaks_at_any_k(cohort_y):
    cohort, y = cohort_y
    for k in (27, 50, 75):
        rc, folds = _folds(cohort, y, k)
        d = pd.DataFrame({"cluster": rc.to_numpy(), "fold": folds.to_numpy()})
        assert (d.groupby("cluster").fold.nunique() == 1).all(), f"vazamento em k={k}"


def test_binpack_beats_random_group_on_balance(cohort_y):
    cohort, y = cohort_y
    _, bp = _folds(cohort, y, 50, "binpack")
    _, rg = _folds(cohort, y, 50, "random_group")
    razao_bp = bp.value_counts().max() / bp.value_counts().min()
    razao_rg = rg.value_counts().max() / rg.value_counts().min()
    assert razao_bp < 1.2, f"bin-packing deveria equilibrar; deu {razao_bp:.1f}x"
    assert razao_bp < razao_rg, "bin-packing deveria ser mais equilibrado que random_group"


def test_all_classes_present_in_every_fold(cohort_y):
    cohort, y = cohort_y
    _, folds = _folds(cohort, y, 50)
    for f in folds.unique():
        assert y[folds == f].nunique() == 3, f"dobra {f} perdeu uma classe"


def test_clusters_are_reproducible(cohort_y):
    cohort, y = cohort_y
    mun = sp.build_municipality_table(cohort, key=COLUNA_MUNICIPIO_RESOLVIDO)
    h1 = sp.assignment_hash(sp.fit_spatial_clusters(mun, 50, 5880, 42))
    h2 = sp.assignment_hash(sp.fit_spatial_clusters(mun, 50, 5880, 42))
    assert h1 == h2
