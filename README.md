# Classification of tuberculosis treatment outcomes in Brazil

Code and public data for **Predicting cure, treatment interruption, and death from tuberculosis in Brazil under spatially blocked and temporal validation: a nationwide benchmark**.

## Download the data

The [harmonised dataset](datasets/2026-10-13/harmonized.parquet) is publicly available without a request. Use the **Download raw file** button on GitHub. The [dataset directory](datasets/2026-10-13/) includes the dictionary, baseline municipal context, frozen partitions and checksums. The source snapshot contains 1,326,538 records and 168 columns; its SHA-256 matches the primary execution input. No Git LFS is required.

```python
import pandas as pd
df = pd.read_parquet("datasets/2026-10-13/harmonized.parquet")
```

## Study design

The main analysis uses 707,870 notifications from 2015–2023. Nested spatial validation holds municipality clusters together across five outer folds and selects algorithms within each training partition. The comparison covers 26 algorithm families and 96 executable algorithm/imbalance combinations. A later evaluation uses 92,152 notifications from 2024 and is retrospective and exploratory, not an independent confirmatory test. Notifications from 2025 are excluded from predictive evaluation.

The primary target codes are 0 = cure, 1 = death attributed to tuberculosis, and 2 = treatment interruption. Eligibility uses the declared extraction date, 15 July 2026, and presentation-specific maturation. The primary predictors exclude HIV serology and its derivative; pre-study census context is supplied separately. These distinctions are implemented in `configs/` and `src/tb_outcomes/`.

## Reproduce and verify

Use Python 3.10. The project dependencies and lockfile are in `pyproject.toml` and `poetry.lock`; GPU-enabled model training requires the corresponding CUDA environment. Reading Parquet requires pandas and pyarrow only.

```bash
python scripts/prepare_public_data.py
PYTHONPATH=src python scripts/verify_public_dataset.py
PYTHONPATH=src python -m tb_outcomes.cli --help
```

The first command verifies the published checksums and copies the source snapshot and supporting inputs to the working paths expected by the analysis. It refuses to overwrite a different local file. The second recalculates eligibility, target definitions and population counts without fitting models. Neither command downloads fresh surveillance data nor trains models.

Published aggregate results and their original checksums are in [reproducibility/2026-10-13](reproducibility/2026-10-13/). That directory also retains the primary execution configurations. Frozen assignments should be used to reproduce the reported split. Full model fitting is computationally expensive and should be run deliberately, not as part of data verification. Historical reporting/deposit commands target older runs; use this dated release for the manuscript results.

## Availability and citation

The harmonised snapshot and the accompanying reproduction files are publicly downloadable. No individual request is needed. Version: **2026-10-13**. Cite the repository URL together with this version and the commit identifier, and acknowledge SINAN/DATASUS, IBGE, IPEA and CNES. The manuscript is under preparation; no article DOI is claimed. Software is licensed under MIT; consult the dataset README for source-data attribution and terms.

Contact: André Abade, Department of Computing, IFMT Campus Barra do Garças, andre.abade@ifmt.edu.br. Contact is for support, not a condition of access.
