# Harmonised tuberculosis dataset

This is the exact harmonised source snapshot used by the revised Smart-TB analysis. It contains 1,326,538 notification records and 168 columns before eligibility and outcome restrictions. The SHA-256 checksum matches the source hash stored in the primary execution manifest. The file is approximately 39 MB (37.52 MiB).

The analytic three-outcome cohort contains 842,250 records: 707,870 from 2015–2023 for development, 92,152 from 2024 for retrospective exploratory evaluation, and 42,228 from 2025 excluded from predictive evaluation. Do not train or evaluate directly on the entire source snapshot without the published eligibility and partition rules.

## Files

- `harmonized.parquet`: unchanged source snapshot; column names and types are in `schema.csv`.
- `baseline_context.parquet`: municipal census context used by the primary analysis; the contemporaneous context in the source snapshot is reserved for sensitivity analyses.
- `feature_dictionary.csv`, `feature_availability.csv`: variable roles and availability, also defined in `configs/features.yaml`.
- `record_assignments.parquet`: analytic cohort membership and frozen spatial/temporal assignments, keyed to the row order of the unchanged harmonised file.
- Municipal cluster and centroid files: frozen spatial partitions for k = 27, 50 and 75.
- `manifest.json`: checksums, restoration paths and cohort metadata.

## Access and provenance

Download files directly from this repository; no application or author approval is required. The snapshot is assembled from public SINAN/DATASUS notifications, IBGE, IPEA and CNES sources by the acquisition and harmonisation code in this repository. The extraction date of 15 July 2026 was declared by the investigators for this source copy. Do not replace it with a fresh extraction when reproducing the reported cohort: source databases can change.

The snapshot contains health-related individual records from the public surveillance source, including dates, coded attributes and municipality-level context. Municipal coordinates describe municipalities, not patient addresses. Public provenance and absence of dedicated name/CPF columns do not establish that every record is anonymous. This release does not grant permission to identify or contact individuals. Cite and acknowledge the original data providers and the versioned Smart-TB snapshot. The repository MIT licence covers the software; original source-data terms remain applicable to third-party data.

Original providers: [DATASUS](https://datasus.saude.gov.br/), [IBGE](https://www.ibge.gov.br/), [IPEA](http://www.ipeadata.gov.br/), [CNES](https://cnes.datasus.gov.br/).
