import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.integration


def _mini_inputs(k):
    rng = np.random.RandomState(k)
    n = 300
    X = pd.DataFrame({"idade": rng.rand(n) * 80, "sexo": rng.choice(["M", "F"], n)})
    y = pd.Series(rng.choice([0, 1, 2], size=n, p=[0.5, 0.3, 0.2]))
    clusters = pd.Series(rng.randint(0, 12, n))
    strata = pd.Series(rng.choice(["2020_N", "2021_S"], n))
    from tb_outcomes.splits import make_outer_folds
    outer = make_outer_folds(X, clusters, y, n_folds=5)
    return X, y, strata, clusters, outer, [0, 1, 2], [], sorted(y.unique())


def test_run_sweep_writes_per_k_artifacts(tmp_path, monkeypatch):
    from tb_outcomes import cli
    from tb_outcomes.executor import SweepConfig, load_executor_config

    exec_cfg = load_executor_config("configs/executor.yaml")
    exec_cfg.sweep = SweepConfig(
        ks=[27, 75], out_root=str(tmp_path / "sweep"),
        models=["majority_class", "logistic_plain"])

    monkeypatch.setattr(cli, "load_executor_config", lambda _p: exec_cfg)
    monkeypatch.setattr(cli, "_load_benchmark_inputs",
                        lambda cfg, sanity=False, k_override=None: _mini_inputs(k_override))

    cli.run_sweep_cmd()

    for k in (27, 75):
        base = tmp_path / "sweep" / f"k{k}"
        assert (base / "benchmark_metrics.csv").exists()
        assert (base / "oof_predictions.parquet").exists()
        assert (base / "calibration_manifest.json").exists()
        assert (base / "pairs" / "majority_class__random_undersampling.meta.json").exists()
    # o k=50 canônico não foi tocado
    assert not (tmp_path / "sweep" / "k50").exists()
