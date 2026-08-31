import numpy as np
import pandas as pd
import pytest

from tb_outcomes.calibration import load_calibration_config
from tb_outcomes.executor import make_sklearn_fold_factory, run_benchmark
from tb_outcomes.imbalance import load_imbalance_config
from tb_outcomes.models import build_registry, load_models_config
from tb_outcomes.preprocess import load_preprocess_config
from tb_outcomes.splits import make_outer_folds

pytestmark = pytest.mark.integration


def _cohort(n=400, seed=0):
    rng = np.random.RandomState(seed)
    X = pd.DataFrame({
        "idade": rng.rand(n) * 80,
        "sexo": rng.choice(["M", "F"], n),
    })
    y = pd.Series(rng.choice([0, 1, 2], size=n, p=[0.5, 0.3, 0.2]))
    clusters = pd.Series(rng.randint(0, 10, n))
    strata = pd.Series(rng.choice(["2020_N", "2021_S"], n))
    return X, y, clusters, strata


def _run(models, strategies, checkpoint_dir, factory, registry, y, clusters, outer):
    return run_benchmark(
        models, strategies, y, clusters, outer, 3, [0, 1, 2],
        factory, lambda m: registry[m].capabilities,
        load_calibration_config("configs/calibration.yaml"), n_inner=3,
        checkpoint_dir=checkpoint_dir)


def test_resume_skips_completed_pairs(tmp_path):
    X, y, clusters, strata = _cohort()
    outer = make_outer_folds(pd.DataFrame(index=y.index), clusters, y, n_folds=5)
    registry = build_registry(load_models_config("configs/models.yaml"))
    imb = load_imbalance_config("configs/imbalance.yaml")
    pre = load_preprocess_config("configs/preprocess.yaml")
    models = ["majority_class", "logistic_plain"]
    strategies = ["random_undersampling"]

    factory1 = make_sklearn_fold_factory(X, y, strata, registry, imb, pre, [], seed=42)
    r1 = _run(models, strategies, tmp_path, factory1, registry, y, clusters, outer)
    n_oof1 = len(r1.oof_rows)
    assert all(s["status"] == "ok" for s in r1.status_rows)

    # segunda rodada: fábrica que EXPLODE se chamada -> prova que nada re-treina
    def exploding_factory(model, strategy):
        raise AssertionError("não deveria treinar: par já tem checkpoint")

    r2 = _run(models, strategies, tmp_path, exploding_factory, registry, y, clusters, outer)
    assert [s["status"] for s in r2.status_rows] == [s["status"] for s in r1.status_rows]
    assert len(r2.oof_rows) == n_oof1
    assert len(r2.metric_rows) == len(r1.metric_rows)
    assert len(r2.aggregate_rows) == len(r1.aggregate_rows)


def test_no_checkpoint_dir_still_works(tmp_path):
    X, y, clusters, strata = _cohort()
    outer = make_outer_folds(pd.DataFrame(index=y.index), clusters, y, n_folds=5)
    registry = build_registry(load_models_config("configs/models.yaml"))
    factory = make_sklearn_fold_factory(
        X, y, strata, registry, load_imbalance_config("configs/imbalance.yaml"),
        load_preprocess_config("configs/preprocess.yaml"), [], seed=42)
    r = run_benchmark(
        ["majority_class"], ["random_undersampling"], y, clusters, outer, 3, [0, 1, 2],
        factory, lambda m: registry[m].capabilities,
        load_calibration_config("configs/calibration.yaml"), n_inner=3)
    assert r.status_rows[0]["status"] == "ok"
