# Tuberculosis treatment closure outcomes in Brazil — patient-level predictions

Access-restricted companion to the open results record. Analysis code:
<https://github.com/smart-TB/ClassificationOfClosureTB>

`MANIFEST.csv` lists every file with its size and SHA-256.

---

## Why access is restricted

These files hold **one row per notification**. They carry no name, no health-card number
and no address, and the finest geography anywhere in the study is the municipality of
residence. They are nonetheless **pseudonymized, not anonymous**: `record_pos` is the row
position in the analytic cohort, and the cohort is rebuilt deterministically from the
public SINAN extract by the code in the repository. A holder of that extract can therefore
map a row back to a notification.

Because the columns include HIV status, homelessness, incarceration, race/colour and
schooling alongside the outcome, that linkage is a real re-identification risk rather than
a theoretical one. Access requests are reviewed by the depositor.

---

## Contents

### `oof/`

Out-of-fold predictions for the **final model only** — the pair at the top of the primary
leaderboard. One row per notification, in the fold where it was held out. Columns:
`record_pos`, `outer_fold`, `model`, `strategy`, `y_true`, the decisions under both rules
(`pred_argmax`, `pred_policy`), and per class the raw score and the calibrated probability.

This is the file behind the equity, calibration, programmatic-utility and SHAP analyses, so
each of those can be recomputed and checked at the individual level.

The other model–strategy pairs of the benchmark are deliberately **not** deposited. Their
aggregate metrics are in the open record, and the predictions themselves are reproducible
from the code and the public sources — depositing all seventy would multiply the exposure
of patient-level data more than fiftyfold for no added verifiability.

### `shap/`

SHAP values for the final model, out-of-fold, on the sample stratified by spatial fold ×
class. One row per explained record per class, with the contribution of each of the 72
features. The sampling manifest is in the open record.

### `folds/`

The cluster and outer-fold assignment of every record — the definition of the spatial
blocking at the individual level. The aggregate equivalents, `cluster_summary.csv` and
`fold_summary.csv`, are in the open record and are sufficient for most purposes.

---

## Licence

CC BY 4.0, subject to the access conditions above.

## Funding

CNPq, grant 445458/2023-2.
