"""Selection with distinct fit, calibration and scoring groups inside development.

Model parameters, calibration family and decision policy are compared on held-out
inner scoring groups. These are selection scores, never final performance claims.
The caller supplies development data only; an outer fold/test year is inaccessible.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import ParameterGrid

from tb_outcomes.calibration import classify_argmax, classify_policy, fit_calibrator, fit_thresholds
from tb_outcomes.decision_search import fit_joint_macro_policy
from tb_outcomes.splits import make_outer_folds


@dataclass(frozen=True)
class Candidate:
    model: str
    strategy: str
    params: dict

    @property
    def key(self):
        return hashlib.sha256(json.dumps({"model": self.model, "strategy": self.strategy,
                                          "params": self.params}, sort_keys=True).encode()).hexdigest()[:16]


def declared_candidates(model, strategy, grids, max_configs=3):
    """Deterministic bounded grid, fixed before any score is seen.

    Spread evenly across the lexicographically ordered grid; this is a limited
    search budget, not an exhaustive optimization claim. No grid means defaults.
    """
    if max_configs < 1:
        raise ValueError("max_configs must be positive")
    grid = list(ParameterGrid(grids.get(model, {})))
    take = np.linspace(0, len(grid) - 1, min(max_configs, len(grid)), dtype=int)
    return [Candidate(model, strategy, grid[i]) for i in take]


def three_way_group_splits(groups, n_splits=5, *, y=None, fold_method="binpack"):
    """Rotate distinct score/calibration folds; all other folds fit the model."""
    g = pd.Series(groups).reset_index(drop=True)
    if g.isna().any() or g.nunique() < n_splits or n_splits < 3:
        raise ValueError("At least three nonempty group folds are required")
    if fold_method == "random_group" and y is None:
        raise ValueError("Random group stratification requires development labels")
    labels = pd.Series(0, index=g.index) if y is None else pd.Series(y).reset_index(drop=True)
    if len(labels) != len(g):
        raise ValueError("Labels and groups must align")
    folds = make_outer_folds(pd.DataFrame(index=g.index), g, labels,
                             n_splits, fold_method).to_numpy()
    for score_fold in range(n_splits):
        cal_fold = (score_fold + 1) % n_splits
        fit = np.flatnonzero((folds != score_fold) & (folds != cal_fold))
        cal = np.flatnonzero(folds == cal_fold)
        score = np.flatnonzero(folds == score_fold)
        if not len(fit) or not len(cal) or not len(score):
            raise ValueError("Empty inner fit/calibration/score partition")
        yield fit, cal, score


def _check_scores(scores, n_rows, n_classes):
    a = np.asarray(scores, dtype=float)
    if a.shape != (n_rows, n_classes) or not np.isfinite(a).all():
        raise ValueError(f"Invalid scores: {a.shape}; expected {(n_rows, n_classes)}")
    return a


def select_candidate(candidates, y_development, groups_development, fit_predict,
                     cal_cfg, *, n_splits=5, calibration_methods=("auto", "sigmoid"),
                     objective="argmax", event_callback=None, fold_method="binpack"):
    """fit_predict(candidate, fit_indices, prediction_indices) never sees test data.

    Every expensive fit supplies both calibration and score predictions. Calibration
    methods/policies reuse these scores, avoiding repeated training. Threshold fitting
    on calibration data is permissible training, but only score-fold F1 ranks options.
    Primary argmax and secondary policy scores remain explicitly separate.
    """
    if objective not in {"argmax", "policy"}:
        raise ValueError("objective must identify the decision rule")
    y = np.asarray(y_development)
    groups = np.asarray(groups_development)
    if len(y) != len(groups):
        raise ValueError("Labels and groups must align")
    classes = sorted(np.unique(y).tolist())
    if classes != list(range(len(classes))) or len(classes) < 2:
        raise ValueError("Use contiguous class codes starting at zero")
    candidates = list(candidates)
    if not candidates or len({c.key for c in candidates}) != len(candidates):
        raise ValueError("Distinct candidates are required")
    splits = list(three_way_group_splits(groups, n_splits, y=y, fold_method=fold_method))
    rows = []
    for candidate in candidates:
        for fold, (fit, cal, score) in enumerate(splits):
            for name, indices in [("fit", fit), ("calibration", cal), ("score", score)]:
                if sorted(np.unique(y[indices]).tolist()) != classes:
                    raise ValueError(f"Missing class in {name} fold {fold}; revise declared partition")
            raw = _check_scores(fit_predict(candidate, fit, np.r_[cal, score]),
                                len(cal) + len(score), len(classes))
            raw_cal, raw_score = raw[:len(cal)], raw[len(cal):]
            for method in calibration_methods:
                calibrator = fit_calibrator(raw_cal, y[cal], cal_cfg, method=method)
                p_cal, p_score = calibrator.transform(raw_cal), calibrator.transform(raw_score)
                argmax = classify_argmax(p_score, classes)
                argmax_f1 = float(f1_score(y[score], argmax, labels=classes, average="macro"))
                legacy = fit_thresholds(p_cal, y[cal], cal_cfg)
                policies = {"argmax": {c: 1.0 for c in classes}, "ovr_f1": legacy,
                            "joint_macro": fit_joint_macro_policy(p_cal, y[cal], classes, legacy)}
                for policy, thresholds in policies.items():
                    pred = classify_policy(p_score, thresholds, classes)
                    policy_f1 = float(f1_score(y[score], pred, labels=classes, average="macro"))
                    rows.append({"candidate": candidate.key, "model": candidate.model,
                                 "strategy": candidate.strategy, "fold": fold,
                                 "calibration": method, "policy": policy,
                                 "argmax_f1_macro": argmax_f1, "policy_f1_macro": policy_f1,
                                 "n_fit": len(fit), "n_calibration": len(cal), "n_score": len(score)})
            if event_callback:
                event_callback({"event": "selection_fold_completed", "candidate": candidate.key,
                                "fold": fold, "n_folds": n_splits})
    table = pd.DataFrame(rows)
    ranking = table.groupby(["candidate", "calibration", "policy"], sort=True)[
        ["argmax_f1_macro", "policy_f1_macro"]].mean().reset_index()
    # Stable tie-breaking: primary argmax determines the model/calibrator; policy
    # breaks ties only after the identical model/calibrator primary score.
    primary = f"{objective}_f1_macro"
    ranking = ranking.sort_values([primary, "policy_f1_macro", "candidate", "calibration", "policy"],
                                   ascending=[False, False, True, True, True], kind="stable")
    winner = ranking.iloc[0].to_dict()
    winner["candidate_spec"] = next(c for c in candidates if c.key == winner["candidate"])
    winner["selection_objective"] = primary
    winner["score_interpretation"] = "inner selection score; not independent final performance"
    return winner, table, ranking
