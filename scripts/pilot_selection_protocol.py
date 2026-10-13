"""CPU-only plumbing pilot, not manuscript results and not final selection."""
from __future__ import annotations

from dataclasses import asdict
import json
from datetime import datetime, timezone
from pathlib import Path
import time

from tb_outcomes.cli import _load_benchmark_inputs
from tb_outcomes.calibration import load_calibration_config
from tb_outcomes.executor import _fit_transform_predict, load_executor_config
from tb_outcomes.imbalance import load_imbalance_config
from tb_outcomes.models import load_models_config
from tb_outcomes.model_selection import Candidate, select_candidate
from tb_outcomes.preprocess import load_preprocess_config


def main():
    out = Path("data/correction_audit/selection_pilot")
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_executor_config("configs/executor.yaml").model_copy(update={"sanity_subsample": 18000})
    b = _load_benchmark_inputs(cfg, True, with_year=True, with_record_id=True)
    X, y, strata, groups, outer, classes, specs, labels, years, ids = b
    assert years.max() < 2024
    ids.to_frame().to_parquet(out / "source_ids.parquet", index=False)
    models = load_models_config("configs/models.yaml")
    entries = {**models.baselines, **models.models}
    pre_cfg = load_preprocess_config("configs/preprocess.yaml")
    imb_cfg = load_imbalance_config("configs/imbalance.yaml")
    candidates = [Candidate("hist_gradient_boost", "none", {"max_iter": 20, "max_leaf_nodes": 15}),
                  Candidate("hist_gradient_boost", "none", {"max_iter": 30, "max_leaf_nodes": 31,
                                                            "l2_regularization": 2.0})]

    def predict(candidate, fit, prediction):
        return _fit_transform_predict(X.iloc[fit], y.iloc[fit], X.iloc[prediction],
                                      candidate.model, candidate.strategy, entries, specs,
                                      pre_cfg, imb_cfg, strata.iloc[fit], cfg.random_state,
                                      len(classes), params=candidate.params)

    def event(message):
        message["timestamp"] = datetime.now(timezone.utc).isoformat()
        with (out / "events.jsonl").open("a") as f:
            f.write(json.dumps(message) + "\n")
        print(json.dumps(message), flush=True)

    started = time.monotonic()
    winner, scores, ranking = select_candidate(candidates, y, groups, predict,
                                               load_calibration_config("configs/calibration.yaml"),
                                               n_splits=3, event_callback=event)
    winner["candidate_spec"] = asdict(winner["candidate_spec"])
    scores.to_csv(out / "inner_scores.csv", index=False)
    ranking.to_csv(out / "ranking.csv", index=False)
    manifest = {"purpose": "CPU plumbing pilot only; cannot supply final scientific results",
                "n": len(y), "years": sorted(years.unique().tolist()), "classes": labels,
                "n_features": len(X.columns), "wall_seconds": time.monotonic() - started,
                "selection": winner, "partitions": "data/correction_audit/partitions/manifest.json"}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
