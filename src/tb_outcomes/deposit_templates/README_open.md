# Tuberculosis treatment closure outcomes in Brazil — results and audit artefacts

Companion data deposit for the study of cure, treatment interruption and death from
tuberculosis in Brazil, using nested spatially blocked validation with a temporal holdout.

Analysis code: <https://github.com/smart-TB/ClassificationOfClosureTB>

**This record contains no patient-level data.** Every file here is aggregated by model,
strategy, fold, group or municipality. Patient-level out-of-fold predictions and SHAP
values are held in a separate, access-restricted record.

`MANIFEST.csv` lists every file with its size and SHA-256, so a reader can verify that what
they downloaded is what was deposited.

---

## Contents

### `results/`

| Directory | What it holds |
|---|---|
| `main/` | Leaderboard of the model grid under the primary spatial protocol (k = 50), with the calibration manifest |
| `sweep/` | Robustness of the leaderboard to spatial granularity: per-k metrics for k = 27 and 75, the combined robustness table, and rank-stability statistics |
| `ablation/` | Territorial ablation — individual, municipal and combined feature sets under the same protocol — plus performance by municipal notification completeness |
| `temporal/` | Temporal holdout: trained before 2024, evaluated on 2024, and compared with the spatial leaderboard |
| `equity/` | Performance by region, sex, race/colour, age band, schooling, municipal vulnerability and spatial fold, with Wilson intervals, AUC, prevalence and small-cell suppression |
| `shap/` | SHAP summary by class with effect direction, stability across seeds, and the sampling manifest |
| `utility/` | Programmatic utility: top-k precision and capture, alerts per 1,000, number needed to screen, decision curves, and the simple clinical baseline for comparison |

Metric files share a long format: one row per model × strategy × fold × metric, with
`kind` distinguishing per-fold rows from aggregates and `class` marking per-class rows
against the `__global__` summary.

### `protocol/`

The frozen protocol, as executed. `configs/` holds the configuration files themselves —
they are the authority on every analytical decision. Alongside them: the municipality →
cluster assignments at each granularity, cluster and fold summaries, the cohort flow and
exclusion reasons, the feature dictionary and availability table, the imbalance audit, the
leakage checks, and the model capability matrix.

### `figures/`

The 26 published figures as PNG, and in `figures/data/` the numerical file behind each one.

Those CSVs are extracted from the rendered figure rather than re-derived from source, so
they describe what was actually drawn. Series longer than 2,000 points are subsampled by
uniform stride with the endpoints preserved; where this happened, the columns
`subamostrada` and `n_pontos_originais` record it.

### `compute/`

Computational cost per model–strategy pair and per model, plus a manifest with seeds,
software environment and hardware. `poetry.lock` and `pyproject.toml` pin the exact
dependency versions.

Two limitations are declared there rather than left implicit: the main k = 50 benchmark ran
before the logging used to build this table, so it is not covered; and the deep learning
results are not bit-for-bit reproducible, because no framework-level deterministic mode is
enabled.

---

## Data sources

The study uses public sources, none of which is redistributed here: SINAN for notification
records, IBGE and IPEA for municipal indicators, CNES for health-service capacity. The
acquisition and harmonization script is in the code repository.

## Licence

Data in this record: CC BY 4.0. The analysis code is separately licensed under MIT.

## Funding

CNPq, grant 445458/2023-2. The funder had no role in study design, data collection,
analysis, interpretation, or in writing the report.
