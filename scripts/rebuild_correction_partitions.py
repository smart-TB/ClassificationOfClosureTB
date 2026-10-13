"""Rebuild geography from development notifications before outcome selection.

Equal weight per municipality. Future-only municipalities are assigned to the
nearest frozen centroid. No outcome or test-year count participates in fitting.
Outputs are isolated from historical clusters/results.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from tb_outcomes.cohort import build_cohort, load_outcome_rules
from tb_outcomes.config import load_config, config_hash
from tb_outcomes.maturation import assess_maturation
from tb_outcomes.splits import build_municipality_table, make_outer_folds, project_to_metric


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs/analysis_decisions.yaml")
    out = root / "data/correction_audit/partitions"
    out.mkdir(parents=True, exist_ok=True)
    source = root / "data/harmonized.parquet"
    df = pd.read_parquet(source)
    year = pd.to_numeric(df.NU_ANO, errors="coerce")
    test_year = int(cfg.study.temporal_test_period)
    development = df.loc[year.between(cfg.study.cohort_year_min, test_year - 1)]
    # Geography is frozen before any outcome selection; postmortem cases carry
    # no extra weight because each municipality is represented exactly once.
    train_mun = build_municipality_table(development)
    all_mun = build_municipality_table(df)
    xy_train = project_to_metric(train_mun.lat.to_numpy(), train_mun.lon.to_numpy(), 5880)
    xy_all = project_to_metric(all_mun.lat.to_numpy(), all_mun.lon.to_numpy(), 5880)
    cohort = build_cohort(df, cfg, load_outcome_rules(root / "configs/outcomes.yaml"),
                          cfg.study.primary_target_schema)
    analytic = cohort.analytic
    label = f"target_{cfg.study.primary_target_schema}"
    population = pd.DataFrame({"source_row_id": analytic.index.astype(str),
                               "year": analytic.NU_ANO.to_numpy(),
                               "target": analytic[label].to_numpy(),
                               "municipality": analytic.ID_MUNIC_ANALISE.to_numpy()})
    population["role"] = np.where(population.year.lt(test_year), "development",
                                  np.where(population.year.eq(test_year), "temporal_test", "unused_future"))
    artifacts = {}
    for k in [27, 50, 75]:
        km = KMeans(n_clusters=k, n_init=10, random_state=cfg.seeds.numpy).fit(xy_train)
        clusters = pd.DataFrame({"ID_MUNIC_ANALISE": all_mun.index,
                                 "cluster": km.predict(xy_all)})
        path = out / f"municipality_clusters_k{k}.parquet"
        clusters.to_parquet(path, index=False)
        artifacts[path.name] = digest(path)
        (out / f"centroids_k{k}.json").write_text(json.dumps(km.cluster_centers_.tolist()))
        mapped = population.municipality.map(clusters.set_index("ID_MUNIC_ANALISE").cluster)
        if mapped.isna().any():
            raise ValueError(f"Missing coordinates/clusters for {int(mapped.isna().sum())} analytic records")
        population[f"cluster_k{k}"] = mapped
        dev = population.role.eq("development")
        folds = make_outer_folds(population.loc[dev], mapped.loc[dev], population.loc[dev].target,
                                 5, "binpack")
        population[f"outer_fold_k{k}"] = pd.Series(pd.NA, index=population.index, dtype="Int64")
        population.loc[dev, f"outer_fold_k{k}"] = folds.astype("Int64")
        assert folds.nunique() == 5
    if population.source_row_id.duplicated().any() or len(population) != len(analytic):
        raise ValueError("Incomplete or duplicated source IDs")
    path = out / "record_assignments.parquet"
    population.to_parquet(path, index=False)
    artifacts[path.name] = digest(path)
    cohort.flow.to_csv(out / "cohort_flow.csv", index=False)
    cohort.exclusions.to_csv(out / "cohort_exclusions.csv", index=False)
    cohort.outcome_distribution.to_csv(out / "outcome_distribution.csv", index=False)
    counts = population.groupby(["role", "year", "target"]).size().rename("n").reset_index()
    counts.to_csv(out / "population_counts.csv", index=False)
    calendar = assess_maturation(df, cfg.study.data_extraction_date, rule="calendar_by_form").mature
    primary = assess_maturation(df, cfg.study.data_extraction_date, rule="days_by_form").mature
    relevant = year.between(cfg.study.cohort_year_min, test_year)
    changed = int(((calendar != primary) & relevant).sum())
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(), "config_hash": config_hash(cfg),
        "harmonized_sha256": digest(source), "script_sha256": digest(Path(__file__)),
        "id_definition": "harmonized SHA256 plus original source row index",
        "development_years": [cfg.study.cohort_year_min, test_year - 1],
        "test_year": test_year, "test_independence": "historically explored; revised retrospective evaluation",
        "geography_fit": "unique municipalities in development notifications, before outcome selection",
        "geography_test_assignment": "nearest frozen centroid",
        "n_geography_fit_municipalities": len(train_mun), "n_mapped_municipalities": len(all_mun),
        "n_by_role": population.role.value_counts().to_dict(),
        "calendar_sensitivity_changed_maturation_flags_through_test_year": changed,
        "artifacts": artifacts,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
