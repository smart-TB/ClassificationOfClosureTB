"""Audita decisões já salvas em lotes; não ajusta modelos ou limiares no teste."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from tb_outcomes.models import DL_MODELS


def scores(cm):
    tp = cm.diagonal().astype(float)
    actual, predicted = cm.sum(axis=1), cm.sum(axis=0)
    f1 = np.divide(2 * tp, actual + predicted, out=np.zeros(3), where=actual + predicted > 0)
    recall = np.divide(tp, actual, out=np.zeros(3), where=actual > 0)
    precision = np.divide(tp, predicted, out=np.zeros(3), where=predicted > 0)
    out = {"f1_macro": f1.mean(), "balanced_accuracy": recall.mean(), "n": int(cm.sum())}
    for c in range(3):
        out.update({f"recall_{c}": recall[c], f"precision_{c}": precision[c],
                    f"f1_{c}": f1[c], f"predicted_fraction_{c}": predicted[c] / cm.sum()})
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=Path("data/oof_predictions.parquet"))
    p.add_argument("--out", type=Path, default=Path("data/classical_review"))
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    matrices = defaultdict(lambda: np.zeros((3, 3), dtype=np.int64))
    mismatches, seen = 0, 0
    cols = ["model", "strategy", "outer_fold", "y_true", "pred_argmax", "pred_policy"]
    cols += [f"raw_score_{c}" for c in range(3)] + [f"proba_{c}" for c in range(3)]
    for i, batch in enumerate(pq.ParquetFile(args.input).iter_batches(batch_size=131072, columns=cols)):
        df = batch.to_pandas()
        df = df.loc[~df.model.isin(DL_MODELS)]
        seen += len(df)
        recalculated = df[[f"proba_{c}" for c in range(3)]].to_numpy().argmax(axis=1)
        mismatches += int((recalculated != df.pred_argmax.to_numpy()).sum())
        df["raw_argmax"] = df[[f"raw_score_{c}" for c in range(3)]].to_numpy().argmax(axis=1)
        for (model, strategy, fold), group in df.groupby(["model", "strategy", "outer_fold"]):
            y = group.y_true.to_numpy(dtype=int)
            for rule in ("raw_argmax", "pred_argmax", "pred_policy"):
                pred = group[rule].to_numpy(dtype=int)
                matrices[(model, strategy, int(fold), rule)] += np.bincount(
                    3 * y + pred, minlength=9).reshape(3, 3)
        if i % 50 == 0:
            print(f"batches={i + 1} classical_rows={seen}", flush=True)
    rows = []
    for (model, strategy, fold, rule), cm in matrices.items():
        rows.append(dict(model=model, strategy=strategy, outer_fold=fold, rule=rule, **scores(cm)))
    detail = pd.DataFrame(rows)
    detail.to_csv(args.out / "decision_rules_per_fold.csv", index=False)
    metrics = [c for c in detail if c not in ["model", "strategy", "outer_fold", "rule", "n"]]
    aggregate = detail.groupby(["model", "strategy", "rule"])[metrics].mean().reset_index()
    aggregate.to_csv(args.out / "decision_rules_mean.csv", index=False)
    old = pd.read_csv("data/benchmark_metrics.csv")
    old = old.loc[(old.kind == "aggregate") & (old.metric == "f1_macro"), ["model", "strategy", "mean"]]
    checked = aggregate.loc[aggregate.rule == "pred_argmax"].merge(old, on=["model", "strategy"])
    max_error = float((checked.f1_macro - checked["mean"]).abs().max())
    result = {"classical_prediction_rows": seen, "argmax_mismatches": mismatches,
              "max_f1_difference_from_saved_metrics": max_error,
              "note": "Auditoria retrospectiva da coorte histórica; não é validação independente da coorte corrigida."}
    (args.out / "audit_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
