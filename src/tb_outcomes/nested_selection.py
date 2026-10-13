"""Fit a selected pipeline without ever receiving evaluation outcomes."""
from __future__ import annotations

import numpy as np

from tb_outcomes.calibration import (
    calibration_manifest, classify_argmax, classify_policy, fit_calibrator, fit_thresholds,
)
from tb_outcomes.decision_search import fit_joint_macro_policy
from tb_outcomes.executor import as_score_matrix, inner_oof_scores
from tb_outcomes.model_selection import select_candidate


def fit_selected_pipeline(candidates, y_development, groups_development,
                          predict_development, predict_evaluation, cal_cfg, *,
                          n_selection_folds=5, n_calibration_folds=5,
                          calibration_methods=("auto", "sigmoid"), event_callback=None,
                          fold_method="binpack"):
    """Callbacks train fresh estimators with all preprocessing confined to fitting rows.

    predict_development(candidate, fit_indices, prediction_indices) works only on
    development rows. predict_evaluation(candidate) refits on full development and
    predicts external features. Neither this function nor either callback needs
    external outcomes. Selection includes model/parameters/strategy/calibration;
    the decision policy is secondary and never called argmax performance.
    """
    y = np.asarray(y_development)
    classes = sorted(np.unique(y).tolist())
    selected, selection_scores, ranking = select_candidate(
        candidates, y, groups_development, predict_development, cal_cfg,
        n_splits=n_selection_folds, calibration_methods=calibration_methods,
        objective="argmax", event_callback=event_callback, fold_method=fold_method)
    candidate = selected["candidate_spec"]
    oof = inner_oof_scores(
        y, groups_development, len(classes), n_calibration_folds,
        lambda fit, pred: as_score_matrix(predict_development(candidate, fit, pred), len(classes)),
        fold_method=fold_method)
    # These fitted-OOF scores train the final calibrator/policy. They are not
    # scored or presented as another independent validation result.
    calibrator = fit_calibrator(oof, y, cal_cfg, method=selected["calibration"])
    calibrated_oof = calibrator.transform(oof)
    if selected["policy"] == "joint_macro":
        thresholds = fit_joint_macro_policy(calibrated_oof, y, classes,
                                             fit_thresholds(calibrated_oof, y, cal_cfg))
    elif selected["policy"] == "ovr_f1":
        thresholds = fit_thresholds(calibrated_oof, y, cal_cfg)
    else:
        thresholds = {c: 1.0 for c in classes}
    raw = as_score_matrix(predict_evaluation(candidate), len(classes))
    if not np.isfinite(raw).all():
        raise ValueError("Nonfinite external scores")
    probabilities = calibrator.transform(raw)
    return {
        "selection": selected, "inner_scores": selection_scores, "ranking": ranking,
        "calibrator": calibration_manifest(calibrator), "thresholds": thresholds,
        "raw_scores": raw, "probabilities": probabilities,
        "pred_argmax": classify_argmax(probabilities, classes),
        "pred_policy": classify_policy(probabilities, thresholds, classes),
    }
