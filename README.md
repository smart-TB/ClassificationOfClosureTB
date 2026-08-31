# Classification of Tuberculosis Treatment Closure Outcomes in Brazil

Analysis code for the study **"Predicting Cure, Treatment Interruption, and Death from Tuberculosis in Brazil: A Nationwide Benchmark of 25 Models Under Spatially Blocked Validation"**.

The study develops and internally validates multiclass prediction models for the three closure outcomes of tuberculosis treatment, using ten years of Brazil's national notification records harmonized with municipal contextual indicators.

> **Status:** manuscript under preparation; this repository holds the reanalysis pipeline.
> <!-- TODO: substituir por DOI/link do artigo quando publicado -->
> <!-- TODO: o título citado abaixo diz "25 Models Under Spatially Blocked Validation";
>      a reanálise usa 26 na grade e acrescentou holdout temporal. Decisão do PI. -->

---

## What the study does

| | |
|---|---|
| **Design** | Retrospective, population-based cohort; model development with internal validation |
| **Cohort** | 1,070,227 notifications declared for 2015–2025; 827,866 analytic after eligibility and ≥365-day follow-up |
| **Outcomes** | Cure (`SITUA_ENCE` = 1), treatment interruption (= 2 or 10), death attributed to tuberculosis (= 3) |
| **Class balance** | 76.67% cure / 18.57% interruption / 4.76% death |
| **Models** | 26 in the grid (3 baselines, 19 classical machine learning, 4 tabular deep learning); 2 support-vector variants excluded a priori because they do not scale to ~10⁶ records |
| **Imbalance strategies** | Random oversampling (capped at 3:1), random undersampling, cost-sensitive reweighting by year × region |
| **Validation** | Nested and spatially blocked: K-means clusters on municipal coordinates, held intact across outer folds; calibration and thresholds fitted only on the inner out-of-fold of each training partition |
| **Robustness to granularity** | Leaderboard re-run at k ∈ {27, 50, 75} municipality clusters |
| **Temporal holdout** | Trained on 2015–2023, evaluated on 2024, reserved before any tuning |
| **Primary metric** | Macro-averaged F1 |
| **Reporting guideline** | TRIPOD+AI |

There is **no external validation** — no independent surveillance system exists for tuberculosis in Brazil. The study does carry a temporal holdout: 2024 was set aside before any model selection and never entered training, calibration, or threshold fitting.

---

## Data

The study uses three public sources. **None is redistributed here.**

| Source | What it provides | Where |
|---|---|---|
| SINAN | Individual tuberculosis notification records | <http://datasus.saude.gov.br> |
| IBGE Cidades | Municipal demographic and socioeconomic indicators | <https://cidades.ibge.gov.br> |
| IPEA Data | Municipal development, income and inequality indicators | <http://www.ipeadata.gov.br> |
| CNES | Health-service capacity indicators | <http://cnes.datasus.gov.br> |

Records carry no direct identifiers. The finest geographic resolution is the municipality of residence; the latitude and longitude attached to each record are municipal reference coordinates, never patient addresses.

`bot_data_SINNAN_IBGE_CNES.py` downloads and harmonizes all four sources into `data/harmonized.parquet`; `src/tb_outcomes/data.py` is a faithful port of its harmonization steps, guarded by a regression test that requires exact reproduction of the reference parquet. The AIDS ⇒ HIV derivation rule lives in `src/tb_outcomes/features.py` as `hiv_pos_model`.

**What this repository does and does not contain.** Aggregated results — metrics by model, strategy and fold, equity tables, ablation, sweep, temporal holdout and programmatic utility — are versioned here. Individual-level material is not: the harmonized cohort (827,866 records with race/colour, schooling, HIV status, homelessness and incarceration), the out-of-fold predictions and the SHAP values all carry one row per patient and are excluded by `.gitignore`. They belong to the data deposit, under access terms set by the principal investigator.

---

## Pipeline

Every model runs under one protocol: **nested, spatially blocked cross-validation at library defaults**, with no hyperparameter search. There is no separate tuning stage, and no model is promoted on the strength of a first pass — a two-stage design would let selection see the evaluation data.

For each of the 5 outer folds:

1. The outer fold is held out entirely. The remaining data is the development set.
2. Inside the development set, a 5-fold inner cross-validation — blocked on the same municipality clusters — produces out-of-fold scores.
3. The probability calibrator and the per-class decision thresholds are fitted **on those inner out-of-fold scores only**.
4. The model is refitted on the full development set and applied **once** to the held-out fold; the calibrator and thresholds are carried over unchanged.

Class-imbalance handling is applied inside step 4's training data only, so the evaluation fold always retains the real outcome prevalence.

### Commands

The pipeline is a single CLI, `tb-outcomes`. Each command writes its artefacts under `data/`.

| Command | Role |
|---|---|
| `harmonize`, `build-cohort`, `build-features`, `build-splits` | Cohort, feature set and spatial folds |
| `run-benchmark` | Main leaderboard: 26 models × 3 strategies at k = 50 |
| `run-sweep`, `sweep-report` | Robustness of the leaderboard to spatial granularity (k = 27, 75) |
| `run-ablation`, `ablation-report` | Territorial ablation: individual vs municipal vs combined feature sets |
| `run-temporal-holdout`, `temporal-report` | Train before 2024, evaluate on 2024 |
| `equity-report` | Performance by region, sex, race/colour, age band, schooling, municipal vulnerability and spatial fold |
| `shap-report` | SHAP panel for the final model, out-of-fold, stratified by fold × class |
| `run-clinical-baseline`, `utility-report` | Programmatic utility: top-k, alerts per 1,000, capture, NNS, decision curves |

### Key configuration

All protocol decisions live in `configs/` and are versioned with the code; `configs/analysis_decisions.yaml` is the authority when documents disagree.

- `random_state = 42`, propagated to the spatial clustering, the folds, the resampling and the models.
- Spatial groups: K-means on municipal coordinates, `k = 50` for the primary analysis; the sweep repeats the whole leaderboard at `k = 27` and `k = 75`.
- Outer partition: 5 folds, municipality clusters kept intact, balanced by a bin-packing assignment.
- Cost-sensitive weights over strata defined by **year × macroregion** (`cost_stratum: year_region`):

  ```
  w_g = clip(4.0 * n_min / max(1, n_maj), 0.15, 1.0)
  ```

  applied only to the majority class (cure). The historical `year × municipality` stratum is retained as a documented sensitivity analysis: it degenerates, pinning 59.4% of the weights at the floor.
- Oversampling is capped at 3:1 rather than balancing to parity, which on this cohort would inflate the training set to ~1.9M rows.
- Calibration is chosen per class and fold: isotonic when at least 100 out-of-fold events are available, Platt scaling otherwise.

### Leakage controls

Asserted programmatically, not by convention — see `src/tb_outcomes/leakage.py` and `tests/`:

1. Class-imbalance handling never reaches an evaluation fold.
2. Calibration and thresholds are fitted only on inner out-of-fold scores of the training partition.
3. Stratum-defining variables (`NU_ANO`, `ID_MN_RESI`) are excluded from the feature set and asserted absent.
4. Municipality clusters never straddle an outer fold.
5. No feature may derive from `DT_ENCERRA` or `SITUA_ENCE`; three independent barriers enforce this, because the defect passed silently once and cost an entire benchmark.

---

## Reproducing the results

```bash
poetry install
poetry run tb-outcomes validate-config --config configs/analysis_decisions.yaml
poetry run tb-outcomes harmonize          # requires the SINAN/IBGE/IPEA/CNES sources
poetry run tb-outcomes build-cohort
poetry run tb-outcomes build-features
poetry run tb-outcomes build-splits
poetry run tb-outcomes run-benchmark      # add --sanity for a subsample smoke run
poetry run pytest                         # 355 tests
```

The classical models run at fixed seeds and library defaults, so re-execution reproduces the reported numbers.

**The deep learning results are not bit-for-bit reproducible.** No framework-level deterministic mode is set for the tensor library, so repeated execution yields small variations. Two constraints are load-bearing and must not be changed casually: `num_workers: 0` in `configs/deep_learning.yaml`, because a non-zero value deadlocks the dataloader against CUDA in this executor; and the internal metric of `pytorch_tabular` is disabled, because a validation batch of size 1 makes torchmetrics fail on a 0-dimensional tensor.

Runtime, for scale: the main leaderboard and the k-sweep each take days on the hardware below, dominated by the two tabular transformers; the ablation, temporal holdout and equity analyses take hours.

### Environment

Pinned in `poetry.lock`. Python 3.10.12.

| Package | Version |
|---|---|
| scikit-learn | 1.4.2 |
| NumPy | 1.26.2 |
| pandas | 2.1.4 |
| XGBoost | 2.1.0 |
| LightGBM | 4.4.0 |
| CatBoost | 1.2.8 |
| SHAP | 0.49.1 |
| PyTorch | 2.1.2 + CUDA 12.1 |
| pytorch-tabular | 1.1.0 |

Hardware used for the reported runs: 2 × NVIDIA RTX A5500 (24 GB each), 32 CPU threads, 124 GB RAM.

> One model per job. LightGBM's scikit-learn wrapper ignores `OMP_NUM_THREADS` and opens all threads regardless; running two training jobs concurrently degrades small fits by orders of magnitude through OpenMP oversubscription.

---

## Known issues

**Two support-vector variants are excluded a priori.** Exact RBF-SVC is O(n²)–O(n³) with a dense kernel, and LinearSVC/liblinear does not converge at this scale. The linear-margin role in the grid is covered by logistic regression and ridge.

**Deep learning results are not bit-for-bit reproducible**, as described under *Reproducing the results*.

Study findings, limitations and their interpretation are reported in the manuscript, not here.

---

## Ethics

The study analyses secondary, de-identified, publicly available administrative records. Under Resolution 510/2016 of the Brazilian National Health Council, article 1, sole paragraph, items II and III, research using publicly accessible information under Federal Law 12.527/2011 and public-domain information is not registered or evaluated by the national research ethics system. The study was therefore not submitted for ethical review, and no approval number or waiver exists.

---

## Citation

<!-- TODO: substituir pela citacao final quando o artigo for publicado -->

```bibtex
@unpublished{abade_tb_closure,
  author = {Abade, Andre da Silva and Arc\^encio, Ricardo Alexandre and Tavares, Reginaldo Bazon Vaz
            and Alves, Yan Mathias and de Campos, Marco Donisete and Borges, Mara\'isa Delmut
            and Yamamura, Mellina and Lima, Jaqueline Costa and Scholze, Alessandro Rolim
            and Diaz-Quijano, Fredi Alexander and Alves, Josilene Dalia},
  title  = {Predicting Cure, Treatment Interruption, and Death from Tuberculosis in Brazil:
            A Nationwide Benchmark of 25 Models Under Spatially Blocked Validation},
  note   = {Manuscript in preparation},
  year   = {2026}
}
```

## Funding

National Council for Scientific and Technological Development (CNPq), grant 445458/2023-2. The funder had no role in study design, data collection, data analysis, data interpretation, or the writing of the report.

## Acknowledgements

Araguaia Epidemiology and Geoprocessing Research Group (EPiGeo), Federal University of Mato Grosso.

## License

MIT. See [LICENSE](LICENSE).
