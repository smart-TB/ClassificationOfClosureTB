"""One nested model/strategy/fold job, shared by CPU and GPU launchers."""
from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd

from tb_outcomes.calibration import calibration_metrics, load_calibration_config
from tb_outcomes.executor import PairResult, _fit_transform_predict, discrimination_metrics, raw_norm
from tb_outcomes.imbalance import load_imbalance_config
from tb_outcomes.model_selection import declared_candidates
from tb_outcomes.models import DL_MODELS, load_models_config
from tb_outcomes.nested_selection import fit_selected_pipeline
from tb_outcomes.preprocess import load_preprocess_config


def evaluate_nested_job(bundle, manifest, job, dev_idx, eval_idx, dl_cfg, run):
    config = load_models_config("configs/models.yaml")
    entries = {**config.baselines, **config.models}
    candidates = declared_candidates(job["model"], job["strategy"], config.hyperparameters,
                                      manifest["max_configs"])
    pre_cfg = load_preprocess_config("configs/preprocess.yaml")
    imb_cfg = load_imbalance_config("configs/imbalance.yaml")
    cal_cfg = load_calibration_config("configs/calibration.yaml")
    X, y, strata = bundle["X"], bundle["y"], bundle["strata"]
    classes = bundle["classes"]
    threads = manifest["threads"]

    def fit_predict(candidate, fit, prediction):
        params = dict(candidate.params)
        if candidate.model == "catboost":
            params["thread_count"] = threads
        elif candidate.model in {"random_forest", "extra_trees", "logistic_plain",
                                 "logistic_regression", "lightgbm", "xgboost", "knn"}:
            params["n_jobs"] = threads
        return _fit_transform_predict(
            X.iloc[fit], y.iloc[fit], X.iloc[prediction], candidate.model, candidate.strategy,
            entries, bundle["specs"], pre_cfg, imb_cfg, strata.iloc[fit], bundle["seed"],
            len(classes), dl_cfg if candidate.model in DL_MODELS else None, params=params)

    if job["model"] == "gradient_boosting":
        import json
        from tb_outcomes.gb_parallel import fit_gradient_boosting

        context = dict(X=X.iloc[dev_idx].reset_index(drop=True),
            y=y.iloc[dev_idx].reset_index(drop=True), strata=strata.iloc[dev_idx].reset_index(drop=True),
            entries=entries, specs=bundle["specs"], pre=pre_cfg, imb=imb_cfg,
            seed=bundle["seed"], n_classes=len(classes))
        fitted = fit_gradient_boosting(context, bundle["clusters"].iloc[dev_idx].to_numpy(),
            X.iloc[eval_idx], candidates, cal_cfg, workers=min(threads, 4),
            n_selection_folds=bundle["n_inner"], n_calibration_folds=bundle["n_inner"],
            fold_method=bundle.get("fold_method", "binpack"),
            event_callback=lambda event: print(json.dumps(event), flush=True))
    else:
        fitted = fit_selected_pipeline(
            candidates, y.iloc[dev_idx].to_numpy(), bundle["clusters"].iloc[dev_idx].to_numpy(),
            lambda c, fit, pred: fit_predict(c, dev_idx[fit], dev_idx[pred]),
            lambda c: fit_predict(c, dev_idx, eval_idx), cal_cfg,
            n_selection_folds=bundle["n_inner"], n_calibration_folds=bundle["n_inner"],
            fold_method=bundle.get("fold_method", "binpack"))
    result = PairResult(job["model"], job["strategy"])
    # External labels are first consumed here, after all choices and predictions.
    truth = y.iloc[eval_idx].to_numpy()
    raw, probabilities = fitted["raw_scores"], fitted["probabilities"]
    rows = calibration_metrics(truth, raw_norm(raw), cal_cfg, stage="pre_renorm")
    rows += calibration_metrics(truth, probabilities, cal_cfg, stage="post_renorm")
    rows += discrimination_metrics(truth, probabilities, fitted["pred_argmax"], classes)
    policy_rows = discrimination_metrics(truth, probabilities, fitted["pred_policy"], classes)
    for r in policy_rows:
        r["axis"] = "secondary_policy"
    for r in rows + policy_rows:
        result.metric_rows.append({"model": job["model"], "strategy": job["strategy"],
                                    "outer_fold": job["fold"], **r})
    selected = fitted["selection"].copy()
    selected["candidate_spec"] = asdict(selected["candidate_spec"])
    result.manifest_rows.append({"model": job["model"], "strategy": job["strategy"],
                                 "outer_fold": job["fold"], "selection": selected,
                                 "thresholds": fitted["thresholds"], **fitted["calibrator"]})
    predictions = pd.DataFrame({"record_pos": eval_idx, "outer_fold": job["fold"],
                                "model": job["model"], "strategy": job["strategy"],
                                "y_true": truth, "pred_argmax": fitted["pred_argmax"],
                                "pred_policy": fitted["pred_policy"]})
    for j, c in enumerate(classes):
        predictions[f"raw_score_{c}"] = raw[:, j]
        predictions[f"proba_{c}"] = probabilities[:, j]
    if not np.isfinite(probabilities).all() or len(predictions) != len(eval_idx):
        raise ValueError("Invalid external predictions")
    result.oof_rows = predictions.to_dict("records")
    fitted["inner_scores"].to_csv(run / "jobs" / (job["id"] + ".inner.csv"), index=False)
    return result
