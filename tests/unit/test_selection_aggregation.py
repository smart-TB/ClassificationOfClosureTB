import json

import pandas as pd
import pytest

from tb_outcomes.dl_parallel import atomic_json, file_hash
from tb_outcomes.selection_aggregation import aggregate_selected_runs, choose_inner_winners


def make_run(root, model, inner_score, truth=(0, 1), experiment="same"):
    run = root / model
    (run / "jobs").mkdir(parents=True)
    jobs = []
    for fold in [0, 1]:
        job = {"id": f"{model}__none__fold{fold}", "model": model, "strategy": "none", "fold": fold}
        jobs.append(job)
        path = run / "jobs" / job["id"]
        frame = pd.DataFrame({"model": [model], "strategy": ["none"], "outer_fold": [fold],
                              "source_row_id": [fold + 100], "y_true": [truth[fold]],
                              "pred_argmax": [0], "pred_policy": [0], "proba_0": [.8], "proba_1": [.2]})
        frame.to_parquet(path.with_suffix(".parquet"))
        path.with_suffix(".inner.csv").write_text("fold,score\n0,0.5\n")
        selection = {"candidate": model, "calibration": "sigmoid", "policy": "argmax",
                     "argmax_f1_macro": inner_score, "policy_f1_macro": inner_score,
                     "selection_objective": "argmax_f1_macro",
                     "candidate_spec": {"model": model, "strategy": "none", "params": {}}}
        atomic_json(path.with_suffix(".json"), {
            "status": "ok", "run_id": model, "job": job,
            "parquet_sha256": file_hash(path.with_suffix(".parquet")),
            "inner_scores_sha256": file_hash(path.with_suffix(".inner.csv")),
            "manifest_rows": [{"outer_fold": fold, "selection": selection}],
            # Deliberately contrary to the inner ranking: external metrics cannot choose.
            "metric_rows": [{"value": 1 - inner_score, "outer_fold": fold}],
        })
    atomic_json(run / "run.json", {"experiment_id": experiment, "protocol": "nested", "mode": "spatial",
                                    "sanity": True, "n_expected_predictions_per_pair": 2, "jobs": jobs})
    atomic_json(run / "complete.json", {"jobs": 2})
    return run


def test_global_selection_uses_inner_scores_even_when_external_scores_disagree(tmp_path):
    a, b = make_run(tmp_path, "a", .8), make_run(tmp_path, "b", .4)
    out = aggregate_selected_runs([b, a], [{"model": m, "strategy": "none"} for m in ["a", "b"]], tmp_path / "out")
    predictions = pd.read_parquet(out / "selected_predictions.parquet")
    assert predictions.selected_model.tolist() == ["a", "a"]
    assert predictions.source_row_id.nunique() == 2
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["reporting_status"] == "not_released"


def test_missing_competitor_is_not_silently_dropped(tmp_path):
    a = make_run(tmp_path, "a", .8)
    with pytest.raises(ValueError, match="universe mismatch"):
        aggregate_selected_runs([a], [{"model": m, "strategy": "none"} for m in ["a", "b"]], tmp_path / "out")


@pytest.mark.parametrize("change,match", [("labels", "labels differ"), ("experiment", "Incompatible"),
                                         ("corrupt", "Invalid checkpoint"), ("duplicate", "Duplicate checkpoint")])
def test_aggregation_refuses_incompatible_or_corrupt_evidence(tmp_path, change, match):
    a = make_run(tmp_path, "a", .8)
    b = make_run(tmp_path, "b", .4, truth=(1, 1) if change == "labels" else (0, 1),
                 experiment="other" if change == "experiment" else "same")
    if change == "corrupt":
        next((b / "jobs").glob("*.parquet")).write_bytes(b"corrupt")
    runs = [a, b, a] if change == "duplicate" else [a, b]
    with pytest.raises(ValueError, match=match):
        aggregate_selected_runs(runs, [{"model": m, "strategy": "none"} for m in ["a", "b"]], tmp_path / "out")


def test_inner_ties_are_deterministic_and_invalid_scores_fail():
    rows = [{"outer_fold": 0, "model": m, "strategy": "none", "candidate": m,
             "calibration": "sigmoid", "policy": "argmax", "argmax_f1_macro": .5,
             "policy_f1_macro": .5} for m in ["b", "a"]]
    assert choose_inner_winners(rows).iloc[0].model == "a"
    rows[0]["argmax_f1_macro"] = float("nan")
    with pytest.raises(ValueError, match="Invalid"):
        choose_inner_winners(rows)
