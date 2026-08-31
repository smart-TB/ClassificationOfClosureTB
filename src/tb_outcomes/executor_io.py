"""Escrita dos artefatos do benchmark (SP4e). Registra TUDO (decisão 4):
escore bruto E proba calibrada; métricas por dobra E agregado; manifesto da guarda.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from tb_outcomes.executor import BenchmarkResult


def write_benchmark_artifacts(result: BenchmarkResult, out_dir: str | Path) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    oof_path = out_dir / "oof_predictions.parquet"
    metrics_path = out_dir / "benchmark_metrics.csv"
    manifest_path = out_dir / "calibration_manifest.json"

    pd.DataFrame(result.oof_rows).to_parquet(oof_path, index=False)

    per_fold = pd.DataFrame(result.metric_rows)
    per_fold["kind"] = "per_fold"
    agg = pd.DataFrame(result.aggregate_rows)
    agg["kind"] = "aggregate"
    pd.concat([per_fold, agg], ignore_index=True).to_csv(metrics_path, index=False)

    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump({"manifest": result.manifest_rows, "status": result.status_rows}, fh,
                  ensure_ascii=False, indent=2, default=str)

    return {"oof": oof_path, "metrics": metrics_path, "manifest": manifest_path}
