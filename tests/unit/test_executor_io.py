import pandas as pd

from tb_outcomes.executor import BenchmarkResult
from tb_outcomes.executor_io import write_benchmark_artifacts


def test_writes_three_artifacts(tmp_path):
    result = BenchmarkResult(
        oof_rows=[{"record_pos": 0, "model": "m", "strategy": "s", "y_true": 1,
                   "raw_score_0": 0.1, "proba_0": 0.2}],
        metric_rows=[{"model": "m", "strategy": "s", "outer_fold": 0,
                      "class": "__global__", "axis": "discrimination", "f1_macro": 0.7}],
        aggregate_rows=[{"model": "m", "strategy": "s", "metric": "f1_macro",
                         "class": "__global__", "mean": 0.7, "n_folds": 5}],
        manifest_rows=[{"model": "m", "strategy": "s", "outer_fold": 0, "n_classes": 3}],
        status_rows=[{"model": "m", "strategy": "s", "status": "ok"}],
    )
    paths = write_benchmark_artifacts(result, tmp_path)
    assert paths["oof"].exists() and paths["metrics"].exists() and paths["manifest"].exists()
    oof = pd.read_parquet(paths["oof"])
    assert "proba_0" in oof.columns and "raw_score_0" in oof.columns
    metrics = pd.read_csv(paths["metrics"])
    assert set(metrics["kind"]) == {"per_fold", "aggregate"}
