import numpy as np
import pandas as pd
import pytest

from tb_outcomes.calibration import load_calibration_config
from tb_outcomes.dl_config import DLConfig
from tb_outcomes.executor import make_sklearn_fold_factory, run_benchmark
from tb_outcomes.imbalance import load_imbalance_config
from tb_outcomes.models import build_registry, load_models_config
from tb_outcomes.preprocess import load_preprocess_config
from tb_outcomes.splits import make_outer_folds

pytestmark = pytest.mark.integration
FAST = DLConfig(max_epochs=2, early_stopping_patience=2, validation_split=0.2,
                batch_size=256, seed=0)


def _cohort(n=500, seed=0):
    rng = np.random.RandomState(seed)
    X = pd.DataFrame({
        "idade": rng.rand(n) * 80,
        "sexo": rng.choice(["M", "F"], n),
        "raca": rng.choice(["1", "2", "3", "4"], n),
    })
    y = pd.Series(rng.choice([0, 1, 2], size=n, p=[0.5, 0.3, 0.2]))
    clusters = pd.Series(rng.randint(0, 10, n))
    strata = pd.Series(rng.choice(["2020_N", "2021_S"], n))
    return X, y, clusters, strata


def test_all_eight_special_models_run(tmp_path):
    X, y, clusters, strata = _cohort()
    outer = make_outer_folds(pd.DataFrame(index=y.index), clusters, y, n_folds=5)
    registry = build_registry(load_models_config("configs/models.yaml"), dl_cfg=FAST)
    factory = make_sklearn_fold_factory(
        X, y, strata, registry, load_imbalance_config("configs/imbalance.yaml"),
        load_preprocess_config("configs/preprocess.yaml"), [], seed=42, dl_cfg=FAST)
    # os 8 rodam, incluindo tabnet (corrigido: imputação das contínuas antes do sparsemax).
    models = ["xgboost", "lightgbm", "hist_gradient_boost", "catboost",
              "category_embedding", "tabnet", "tab_transformer", "ft_transformer"]
    result = run_benchmark(
        models, ["random_oversampling"], y, clusters, outer, 3, [0, 1, 2],
        factory, lambda m: registry[m].capabilities,
        load_calibration_config("configs/calibration.yaml"), n_inner=3)
    status = {r["model"]: r["status"] for r in result.status_rows}
    assert all(status[m] == "ok" for m in models), status     # os 8 rodam
    assert len(result.oof_rows) == len(models) * len(y)
