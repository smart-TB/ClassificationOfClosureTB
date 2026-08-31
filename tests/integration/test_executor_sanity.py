import numpy as np
import pandas as pd
import pytest

from tb_outcomes.calibration import load_calibration_config
from tb_outcomes.executor import make_sklearn_fold_factory, run_benchmark
from tb_outcomes.executor_io import write_benchmark_artifacts
from tb_outcomes.imbalance import load_imbalance_config
from tb_outcomes.models import build_registry, load_models_config
from tb_outcomes.preprocess import load_preprocess_config
from tb_outcomes.splits import make_outer_folds

pytestmark = pytest.mark.integration


def _tiny_cohort(n=600, seed=0):
    rng = np.random.RandomState(seed)
    df = pd.DataFrame({
        "idade": rng.randint(0, 90, n).astype(float),
        "sexo": rng.choice(["M", "F"], n),
        "raca": rng.choice(["1", "2", "3", "4"], n),
        "cluster": rng.randint(0, 12, n),
    })
    y = pd.Series(rng.choice([0, 1, 2], size=n, p=[0.6, 0.3, 0.1]))
    # estrato de custo (year_region) vem POR FORA do X (regra de vazamento)
    strata = pd.Series(rng.choice(["2020_N", "2020_S", "2021_N"], size=n)).rename("stratum")
    return df, y, strata


def test_sanity_run_produces_artifacts_and_passes(tmp_path):
    df, y, strata = _tiny_cohort()
    X = df[["idade", "sexo", "raca"]]
    clusters = df["cluster"]
    outer = make_outer_folds(df, clusters, y, n_folds=5)

    registry = build_registry(load_models_config("configs/models.yaml"))
    factory = make_sklearn_fold_factory(
        X, y, strata, registry,
        imb_cfg=load_imbalance_config("configs/imbalance.yaml"),
        pre_cfg=load_preprocess_config("configs/preprocess.yaml"),
        feature_specs=[], seed=42,
    )
    result = run_benchmark(
        models=["logistic_plain", "random_forest", "knn"],
        strategies=["random_undersampling", "random_oversampling", "local_cost_sensitive"],
        y=y, record_clusters=clusters, outer_folds=outer, n_classes=3, classes=[0, 1, 2],
        make_fold_factory=factory, capabilities_of=lambda m: registry[m].capabilities,
        cal_cfg=load_calibration_config("configs/calibration.yaml"), n_inner=3,
    )
    paths = write_benchmark_artifacts(result, tmp_path)
    assert paths["oof"].exists()
    oof = pd.read_parquet(paths["oof"])
    # o cost em modelo sem peso (knn) é marcado incompatível, não zerado
    status = {(r["model"], r["strategy"]): r["status"] for r in result.status_rows}
    assert status[("knn", "local_cost_sensitive")] == "not_run_incompatible"
    # cada par compatível avalia todo registro uma vez (cobertura completa do EVAL)
    n_ok = sum(1 for v in status.values() if v == "ok")
    assert len(oof) == n_ok * len(y)
