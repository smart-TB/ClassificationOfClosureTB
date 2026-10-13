"""Paired cluster bootstrap of classification metrics on fixed external predictions.

Intervals quantify variation across sampled geographic clusters, conditional on
fitted predictions. They do not include refitting/model-selection uncertainty.
Pooling records is explicit; these estimates are not means of outer-fold metrics.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def metrics_from_confusion(matrix, classes, min_denominator=1):
    matrix = np.asarray(matrix, dtype=float)
    actual, predicted = matrix.sum(1), matrix.sum(0)
    true_positive = matrix.diagonal()
    total = matrix.sum()
    negative = total - actual
    true_negative = total - actual - predicted + true_positive

    def ratio(a, b):
        return float(a / b) if b >= min_denominator and b > 0 else np.nan

    values = {"accuracy": ratio(true_positive.sum(), total)}
    recalls, f1s = [], []
    for i, label in enumerate(classes):
        recall = ratio(true_positive[i], actual[i])
        # No observed events: class F1 is uninformative, even if false positives exist.
        f1 = (ratio(2 * true_positive[i], actual[i] + predicted[i])
              if actual[i] >= min_denominator and actual[i] > 0 else np.nan)
        values.update({f"sensitivity:{label}": recall,
                       f"specificity:{label}": ratio(true_negative[i], negative[i]),
                       f"ppv:{label}": ratio(true_positive[i], predicted[i]),
                       f"f1:{label}": f1})
        recalls.append(recall)
        f1s.append(f1)
    values["balanced_accuracy"] = float(np.mean(recalls))
    values["f1_macro"] = float(np.mean(f1s))
    return values


def paired_cluster_bootstrap(predictions, cluster_map, classes, *, rule="pred_argmax",
                             reference=None, replicates=2000, seed=42,
                             confidence=.95, min_denominator=1, strata_col=None):
    """predictions maps prespecified comparison names to one prediction per source ID.

    cluster_map has source_row_id, cluster and optionally a stratum (outer_fold).
    Spatial OOF inference should resample clusters within their outer fold;
    temporal inference resamples the same frozen clusters without fold strata.
    The SAME cluster multiplicities are used for every comparator in each draw.
    """
    if not predictions or replicates < 100 or not 0 < confidence < 1 or min_denominator < 1:
        raise ValueError("Invalid bootstrap configuration")
    if len(classes) < 2 or len(set(classes)) != len(classes):
        raise ValueError("Declare distinct classes")
    if reference is not None and reference not in predictions:
        raise ValueError("Reference comparator is absent")
    columns = ["source_row_id", "cluster"] + ([strata_col] if strata_col else [])
    mapping = cluster_map[columns].copy()
    if mapping.isna().any().any() or mapping.source_row_id.duplicated().any():
        raise ValueError("Missing/duplicate cluster mapping")
    if strata_col and mapping.groupby("cluster")[strata_col].nunique().gt(1).any():
        raise ValueError("A cluster spans multiple resampling strata")
    mapping = mapping.sort_values("source_row_id").reset_index(drop=True)
    ids = mapping.source_row_id
    cluster_values = sorted(mapping.cluster.unique())
    if len(cluster_values) < 2:
        raise ValueError("At least two clusters are required")
    cluster_codes = pd.Categorical(mapping.cluster, categories=cluster_values).codes
    class_codes = {c: i for i, c in enumerate(classes)}
    tensors, truth = {}, None
    for name, frame in predictions.items():
        p = frame[["source_row_id", "y_true", rule]].sort_values("source_row_id").reset_index(drop=True)
        if p.source_row_id.duplicated().any() or not p.source_row_id.equals(ids):
            raise ValueError(f"Prediction IDs differ for {name}")
        if p.isna().any().any() or not set(p.y_true) <= set(classes) or not set(p[rule]) <= set(classes):
            raise ValueError("Missing/invalid prediction classes")
        if truth is not None and not p.y_true.equals(truth):
            raise ValueError("Comparator labels disagree")
        truth = p.y_true
        tensor = np.zeros((len(cluster_values), len(classes), len(classes)), dtype=np.int64)
        np.add.at(tensor, (cluster_codes, p.y_true.map(class_codes), p[rule].map(class_codes)), 1)
        tensors[name] = tensor
    if strata_col:
        cluster_strata = mapping.drop_duplicates("cluster").set_index("cluster")[strata_col]
        groups = [np.array([i for i, c in enumerate(cluster_values) if cluster_strata[c] == s])
                  for s in sorted(cluster_strata.unique())]
    else:
        groups = [np.arange(len(cluster_values))]
    if any(len(g) < 2 for g in groups):
        raise ValueError("At least two clusters per resampling stratum are required")
    rng = np.random.default_rng(seed)
    counts = np.zeros((replicates, len(cluster_values)), dtype=np.int64)
    for group in groups:
        counts[:, group] = rng.multinomial(len(group), np.full(len(group), 1 / len(group)), size=replicates)
    points, draws = {}, {}
    for name, tensor in tensors.items():
        points[name] = metrics_from_confusion(tensor.sum(0), classes, min_denominator)
        matrices = (counts @ tensor.reshape(len(cluster_values), -1)).reshape(
            replicates, len(classes), len(classes))
        draws[name] = pd.DataFrame([metrics_from_confusion(m, classes, min_denominator) for m in matrices])
    alpha = (1 - confidence) / 2
    rows = []
    for name in predictions:
        for kind in (["estimate", "paired_difference"] if reference and name != reference else ["estimate"]):
            for metric, point in points[name].items():
                values = draws[name][metric].to_numpy()
                if kind == "paired_difference":
                    point -= points[reference][metric]
                    values = values - draws[reference][metric].to_numpy()
                finite = values[np.isfinite(values)]
                # Missing events/denominators in too many draws must not look precise.
                status = ("ok" if np.isfinite(point) and len(finite) >= .95 * replicates
                          else "not_estimable")
                low, high = np.quantile(finite, [alpha, 1 - alpha]) if status == "ok" else (np.nan, np.nan)
                rows.append({"comparison": name, "kind": kind, "reference": reference if kind != "estimate" else None,
                             "metric": metric, "estimate": point, "ci_low": low, "ci_high": high,
                             "status": status, "valid_replicates": len(finite), "replicates": replicates,
                             "n_records": len(ids), "n_clusters": len(cluster_values), "decision_rule": rule,
                             "aggregation": "pooled_records", "resampling_unit": "geographic_cluster",
                             "stratified_by": strata_col, "confidence": confidence, "seed": seed,
                             "uncertainty": "conditional_on_fixed_external_predictions"})
    return pd.DataFrame(rows)
