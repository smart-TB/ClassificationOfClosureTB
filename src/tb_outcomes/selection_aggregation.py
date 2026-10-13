"""Select algorithms from inner scores, then collect their external predictions.

The declared universe is mandatory: a missing/failed competitor must never silently
turn into a smaller, apparently complete comparison. CPU/GPU runs may be combined
only when they represent the same scientific experiment.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tb_outcomes.dl_parallel import atomic_json, completed, file_hash


def choose_inner_winners(records):
    """Only selection metadata enters this function; no external metric or label."""
    table = pd.DataFrame(records)
    required = {"outer_fold", "model", "strategy", "candidate", "calibration", "policy",
                "argmax_f1_macro", "policy_f1_macro"}
    if table.empty or not required <= set(table):
        raise ValueError("Missing inner selection records")
    if table.duplicated(["outer_fold", "model", "strategy"]).any():
        raise ValueError("Duplicate model/strategy/fold")
    scores = table[["argmax_f1_macro", "policy_f1_macro"]].to_numpy(float)
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Invalid inner selection scores")
    return (table.sort_values(
        ["argmax_f1_macro", "policy_f1_macro", "model", "strategy", "candidate", "calibration", "policy"],
        ascending=[False, False, True, True, True, True, True], kind="stable")
        .drop_duplicates("outer_fold").sort_values("outer_fold").reset_index(drop=True))


def aggregate_selected_runs(run_paths, expected_pairs, out_dir):
    expected = {(p["model"], p["strategy"]) for p in expected_pairs}
    if not expected or len(expected) != len(expected_pairs):
        raise ValueError("Declare distinct expected compatible pairs before aggregation")
    records, shards, manifests, inputs = [], {}, [], []
    reference = None
    for path in run_paths:
        run = Path(path)
        m = json.loads((run / "run.json").read_text())
        if m["protocol"] != "nested" or not (run / "complete.json").is_file():
            raise ValueError("Only complete nested runs can be aggregated")
        signature = {k: v for k, v in m.items() if k not in
                     {"compute", "threads", "unsupported", "jobs", "input_sha256"}}
        if reference is not None and signature != reference:
            raise ValueError("Incompatible scientific experiments")
        reference = signature
        manifests.append(m)
        inputs.append({"path": str(run), "manifest_sha256": file_hash(run / "run.json")})
        for job in m["jobs"]:
            if not completed(run, job):
                raise ValueError(f"Invalid checkpoint: {job['id']}")
            key = (job["model"], job["strategy"], job["fold"])
            if key in shards:
                raise ValueError(f"Duplicate checkpoint: {key}")
            meta = json.loads((run / "jobs" / (job["id"] + ".json")).read_text())
            rows = meta["manifest_rows"]
            if len(rows) != 1 or rows[0]["outer_fold"] != job["fold"]:
                raise ValueError("Ambiguous selection manifest")
            selection = rows[0]["selection"]
            spec = selection["candidate_spec"]
            if (spec["model"], spec["strategy"]) != key[:2]:
                raise ValueError("Selection candidate disagrees with job")
            if selection["selection_objective"] != "argmax_f1_macro":
                raise ValueError("Unexpected selection objective")
            record = {k: selection[k] for k in
                      ["candidate", "calibration", "policy", "argmax_f1_macro", "policy_f1_macro"]}
            records.append({**record, "model": key[0], "strategy": key[1], "outer_fold": key[2],
                            "parameters": spec["params"], "run": str(run), "job_id": job["id"]})
            shards[key] = (run / "jobs" / (job["id"] + ".parquet"), meta)
    actual = {(m, s) for m, s, _ in shards}
    if actual != expected:
        raise ValueError(f"Candidate universe mismatch: missing={expected-actual}, extra={actual-expected}")
    folds = {f for _, _, f in shards}
    if any({f for m, s, f in shards if (m, s) == pair} != folds for pair in expected):
        raise ValueError("Incomplete outer folds")

    # Make all choices BEFORE reading external predictions or metric values.
    winners = choose_inner_winners(records)
    winners_by_fold = {row["outer_fold"]: row for row in winners.to_dict("records")}
    expected_count = manifests[0]["n_expected_predictions_per_pair"]
    fold_identity, predictions, metrics, coverage = {}, [], [], []
    for pair in sorted(expected):
        seen = set()
        for fold in sorted(folds):
            path, meta = shards[(*pair, fold)]
            identity = pd.read_parquet(path, columns=["source_row_id", "y_true", "outer_fold"])
            if identity.source_row_id.isna().any() or identity.source_row_id.duplicated().any():
                raise ValueError("Missing/duplicate prediction IDs")
            ids = set(identity.source_row_id)
            if seen.intersection(ids) or not identity.outer_fold.eq(fold).all():
                raise ValueError("Overlapping/misassigned external folds")
            seen.update(ids)
            identity = identity.sort_values("source_row_id").reset_index(drop=True)
            if fold in fold_identity and not identity.equals(fold_identity[fold]):
                raise ValueError("External records/labels differ between competitors")
            fold_identity[fold] = identity
            winner = winners_by_fold[fold]
            if pair == (winner["model"], winner["strategy"]):
                frame = pd.read_parquet(path)
                frame = frame.rename(columns={"model": "selected_model", "strategy": "selected_strategy"})
                frame["model"], frame["strategy"] = "nested_selection_procedure", "selected_in_development"
                predictions.append(frame)
                metrics.extend({**r, "selected_model": pair[0], "selected_strategy": pair[1],
                                "model": "nested_selection_procedure", "strategy": "selected_in_development",
                                "kind": "per_fold"} for r in meta["metric_rows"])
        if len(seen) != expected_count:
            raise ValueError(f"Incomplete record coverage for {pair}")
        coverage.append({"model": pair[0], "strategy": pair[1], "n_unique": len(seen), "folds": len(folds)})
    combined = pd.concat(predictions, ignore_index=True)
    if len(combined) != expected_count or combined.source_row_id.duplicated().any():
        raise ValueError("Incomplete selection procedure predictions")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError("Aggregation output must be empty; preserve previous evidence")
    winners.to_json(out / "selected_by_fold.json", orient="records", indent=2)
    pd.DataFrame(records).to_json(out / "inner_comparison.json", orient="records", indent=2)
    combined.to_parquet(out / "selected_predictions.parquet", index=False)
    pd.DataFrame(metrics).to_csv(out / "selected_metrics_per_fold.csv", index=False)
    pd.DataFrame(coverage).to_csv(out / "coverage.csv", index=False)
    files = {p.name: file_hash(p) for p in sorted(out.iterdir()) if p.is_file()}
    atomic_json(out / "manifest.json", {
        "experiment_id": manifests[0]["experiment_id"], "inputs": inputs, "files": files,
        "mode": manifests[0]["mode"], "sanity": manifests[0]["sanity"],
        "expected_pairs": expected_pairs, "n_predictions": len(combined),
        "selection_objective": "mean inner scoring-fold argmax macro-F1",
        "tie_break": "inner policy macro-F1, then model/strategy/candidate/calibration/policy names",
        "interpretation": "External evaluation of the within-development selection procedure; per-fold metrics",
        "temporal_limitation": "2024 was explored historically; no claim of a pristine confirmatory test",
        "reporting_status": "not_released",
    })
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True, type=Path)
    parser.add_argument("--universe", required=True, type=Path, help="JSON list of compatible model/strategy pairs")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    print(aggregate_selected_runs(args.runs, json.loads(args.universe.read_text()), args.out))


if __name__ == "__main__":
    main()
