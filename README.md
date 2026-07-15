# Classification of Tuberculosis Treatment Closure Outcomes in Brazil

Analysis code for the study **"Predicting Cure, Treatment Interruption, and Death from Tuberculosis in Brazil: A Nationwide Benchmark of 25 Models Under Spatially Blocked Validation"**.

The study develops and internally validates multiclass prediction models for the three closure outcomes of tuberculosis treatment, using ten years of Brazil's national notification records harmonized with municipal contextual indicators.

> **Status:** manuscript under preparation. This repository is being populated ahead of submission.
> <!-- TODO: substituir por DOI/link do artigo quando publicado -->

---

## What the study does

| | |
|---|---|
| **Design** | Retrospective, population-based cohort; model development with internal validation only |
| **Cohort** | 755,550 tuberculosis notifications, Brazil, 2015–2024 (window applied on `DT_NOTIFIC`) |
| **Outcomes** | Cure (`SITUA_ENCE` = 1), treatment interruption (= 2), death attributed to tuberculosis (= 3) |
| **Class balance** | 77.48% / 17.56% / 4.96% |
| **Models** | 25 (21 classical machine learning, 4 tabular deep learning) |
| **Imbalance strategies** | Random oversampling, random undersampling, custom cost-sensitive reweighting |
| **Validation** | Spatially blocked: K-means clusters on municipal coordinates, held intact across folds |
| **Primary metric** | Macro-averaged F1 |
| **Reporting guideline** | TRIPOD+AI |

There is **no external validation and no temporal holdout**. Performance transfers across Brazilian municipalities within the study window; nothing here supports transfer to other countries, other surveillance systems, or future years.

---

## Data

The study uses three public sources. **None is redistributed here.**

| Source | What it provides | Where |
|---|---|---|
| SINAN | Individual tuberculosis notification records | <http://datasus.saude.gov.br> |
| IBGE Cidades | Municipal demographic and socioeconomic indicators | <https://cidades.ibge.gov.br> |
| CNES | Health-service capacity indicators | <http://cnes.datasus.gov.br> |

Records carry no direct identifiers. The finest geographic resolution is the municipality of residence; the latitude and longitude attached to each record are municipal reference coordinates, never patient addresses.

The pipeline consumes a pre-processed file, `data/SINAN_PreProcess.csv`, produced by an upstream harmonization step.
<!-- TODO: descrever ou incluir o passo de pre-processamento que gera SINAN_PreProcess.csv,
     incluindo a regra que deriva HIV positivo a partir do campo de AIDS -->

---

## Pipeline

The comparison runs in **two stages**, and the distinction governs how the results should be read.

**Stage 1 — leaderboard.** All 25 models are fitted under each of the three imbalance strategies. The 21 classical models run at their **library defaults with no hyperparameter search**; the 4 deep learning architectures receive a search restricted to learning rate and weight decay over four trials. No probability calibration is applied. This stage produces Table 2 of the paper, the confusion matrices, and the discrimination curves.

**Stage 2 — calibration and tuning.** The models ranked highest by macro-averaged F1 are carried forward, and grid search plus isotonic probability calibration are applied to them. This stage produces the bootstrap comparison (Figure 6 of the paper).

Because selection into stage 2 is made on untuned performance, the second stage refines the ordering established by the first rather than overturning it.

### Scripts

| File | Role |
|---|---|
| `ML_fit_eval.py` | Fits and evaluates the 21 classical models (stage 1) |
| `DL_fit_eval.py` | Fits and evaluates the 4 tabular deep learning models (stage 1) |
| `calibrate_tune_ml.py` | Selects top-K by macro-averaged F1, then calibrates and tunes (stage 2) |
| `calibrate_tune_dl.py` | Same, for the deep learning models |
| `Figures_PreSelection.py` | Generates figures |
| `Class_Model_Final/` | A production trial. **Not part of the benchmark reported in the paper.** |

<!-- TODO: confirmar a lista acima e acrescentar os scripts que faltarem -->

### Key configuration

- `random_state = 42`, propagated to the spatial clustering, the partitioning, the folds, the resampling, and the search.
- Spatial groups: `KMeans(n_clusters=50)` on `LAT_MUNIC` / `LONG_MUNIC` projected to kilometres.
- Partitioning: `StratifiedGroupKFold(n_splits=5)` with the cluster as the grouping unit, giving a hold-out of approximately 20%. Absence of group leakage is asserted programmatically.
- Cost-sensitive weights, over strata defined by year × municipality:

  ```
  w_g = clip(4.0 * n_min / max(1, n_maj), 0.15, 1.0)
  ```

  applied **only to the majority class** (cure); minority records keep a weight of 1.0. The scheme down-weights the majority rather than up-weighting the minorities, and acts most strongly in the strata where unfavourable outcomes are rarest.
- Variables defining the weighting strata are excluded from the feature set.

### Leakage controls

1. Class-imbalance handling is confined to training folds; the hold-out retains the real outcome prevalence.
2. Thresholds and calibration are fitted on out-of-fold training predictions only and applied unchanged to the hold-out.
3. Stratum-defining variables are excluded from the features.
4. Group leakage across the partition is asserted programmatically.

---

## Reproducing the results

Because stage 1 uses library defaults with a fixed seed, re-running the code regenerates the classical results reported in the paper, including the configurations that the stage-2 search selects — those were not stored separately and do not need to be.

**The deep learning results are not bit-for-bit reproducible.** No framework-level deterministic mode is set and no seed is fixed for the tensor library itself, so repeated execution yields small variations.

### Environment

| Package | Version |
|---|---|
| scikit-learn | 1.7.2 |
| NumPy | 1.26.4 |
| XGBoost | 3.1.1 |
| LightGBM | 4.6.0 |
| CatBoost | 1.2.8 |
| pytorch-tabular | 1.1.1 |
| PyTorch Lightning | 2.4.0 |
| PyTorch | 2.9.1 + CUDA 12.8 |

<!-- TODO: completar com a versao do Python, pandas, imbalanced-learn, shap e RAPIDS (cuML/cuDF),
     que nao constam do meta.json; e informar GPU, CPU e RAM -->

```bash
# TODO: instrucoes de execucao
```

---

## Known issues

**An anomaly in the results table is unresolved.** Four models that failed to learn the task — logistic regression, complement naive Bayes, multinomial naive Bayes, and linear support vector classification — returned identical or near-identical metrics across imbalance strategies that should have produced different fits. A systematic comparison of every pairwise strategy contrast, across all 25 models and 11 metrics, locates the behaviour in these four alone; **no tree-based or deep learning model is affected**. One candidate explanation is that resampling was not applied to models whose grids fix `class_weight='balanced'`, so that two nominally distinct conditions executed the same configuration. The origin has not been traced. The affected models sit at the bottom of the ranking and none of the paper's conclusions rest on them.

**The reported quantities are not all drawn from the same fits.** The leaderboard, confusion matrices, and discrimination curves come from stage 1; the bootstrap intervals come from stage 2. The uncertainty quantified does not attach to the performance ranked.

**Missingness is imputed, and it is not ignorable.** The proportion of records with an absent value roughly doubles among deaths across every comorbidity field. Median and modal imputation replaces that pattern with a central value, discarding information associated with the outcome.

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
