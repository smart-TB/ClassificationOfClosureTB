"""Export R11 descriptive aggregates and confusion counts; no fitting or selection."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from tb_outcomes.cohort import build_cohort, load_outcome_rules
from tb_outcomes.config import load_config

OUT = Path('data/reanalysis/c8_20261005/reporting_supplement')
RUN = Path('data/reanalysis/c6_20260919/primary_cpu/a0a12b4c53f318709c0c')
OLD = Path('data/reanalysis/c7_20260923')
FIELDS = ['CS_SEXO', 'CS_RACA', 'CS_ESCOL_N', 'REGIAO', 'FORMA', 'TRATAMENTO',
          'AGRAVAIDS', 'AGRAVALCOO', 'AGRAVDIABE', 'AGRAVDROGA', 'AGRAVTABAC',
          'POP_RUA', 'POP_LIBER', 'POP_SAUDE', 'POP_IMIG', 'BENEF_GOV',
          'RAIOX_TORA', 'BACILOSC_E', 'BACILOSC_O', 'TEST_MOLEC']
LABELS = {0: 'cure', 1: 'tb_death', 2: 'treatment_interruption'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    upstream = {}

    def verify(path, digest=None):
        got = sha(path)
        if digest is not None:
            assert got == digest, str(path)
        upstream[str(path)] = got

    rm = json.loads((RUN / 'run.json').read_text())
    verify(RUN / 'run.json')
    for name in ['src/tb_outcomes/cohort.py', 'src/tb_outcomes/config.py',
                 'src/tb_outcomes/maturation.py']:
        verify(name, rm['sources'][name])
    verify('data/harmonized.parquet', rm['harmonized_sha256'])
    for name in ['analysis_decisions.yaml', 'outcomes.yaml', 'features.yaml']:
        verify(RUN / 'runtime/configs' / name)
    cfg = load_config(RUN / 'runtime/configs/analysis_decisions.yaml')
    rules = load_outcome_rules(RUN / 'runtime/configs/outcomes.yaml')
    specs = yaml.safe_load((RUN / 'runtime/configs/features.yaml').read_text())['features']
    zero_missing = {s['raw_name'] for s in specs if s.get('zero_means_missing')}
    df = pd.read_parquet('data/harmonized.parquet')
    analytic = build_cohort(df, cfg, rules, cfg.study.primary_target_schema).analytic
    assert analytic.index.is_unique
    analytic.index = analytic.index.astype(str)
    overview = {r['arm']: r for r in json.loads(Path(
        'data/reanalysis/c7_20261004/reporting_release/arm_overview.json').read_text())}
    verify('data/reanalysis/c7_20261004/reporting_release/arm_overview.json')
    category_rows, numeric_rows, confusion_rows, checks = [], [], [], []
    for arm, role, year_mask in [
        ('primary', 'development_2015_2023', analytic.NU_ANO.lt(2024)),
        ('temporal', 'temporal_2024', analytic.NU_ANO.eq(2024)),
    ]:
        directory = OLD / (arm + '_selected')
        manifest = json.loads((directory / 'manifest.json').read_text())
        verify(directory / 'manifest.json')
        pred_path = directory / 'selected_predictions.parquet'
        verify(pred_path, manifest['files']['selected_predictions.parquet'])
        pred = pd.read_parquet(pred_path)
        ids = pred.source_row_id.astype(str)
        assert ids.is_unique and len(pred) == overview[arm]['n']
        assert set(ids) == set(analytic.index[year_mask])
        frame = analytic.loc[ids].copy()
        target_name = 'target_' + cfg.study.primary_target_schema
        expected_labels = frame[target_name].replace({'tb_attributed_death': 'tb_death'})
        assert np.array_equal(expected_labels.to_numpy(), pred.y_true.map(LABELS).to_numpy())
        age = pd.to_numeric(frame.NU_ANO, errors='coerce') - pd.to_numeric(frame.ANO_NASC, errors='coerce')
        frame['age_years'] = age.where(age.between(0, 120))
        frame['age_band'] = pd.cut(frame.age_years, [-1, 14, 24, 34, 44, 59, 64, 120],
            labels=['0-14', '15-24', '25-34', '35-44', '45-59', '60-64', '65+'])
        frame['potential_observation_days'] = (
            pd.Timestamp(cfg.study.data_extraction_date) - pd.to_datetime(frame.DT_NOTIFIC, errors='coerce')).dt.days
        frame['_outcome'] = pred.y_true.to_numpy()
        for c in [None, 0, 1, 2]:
            group = frame if c is None else frame.loc[frame._outcome.eq(c)]
            stratum = 'all' if c is None else LABELS[c]
            for feature in FIELDS + ['age_band']:
                series = group[feature].astype(object)
                missing = series.isna() | series.astype(str).str.strip().eq('')
                if feature in zero_missing:
                    missing |= pd.to_numeric(series, errors='coerce').eq(0)
                def code(v):
                    if isinstance(v, (int, float, np.number)) and float(v).is_integer():
                        return str(int(v))
                    return str(v)
                codes = series.loc[~missing].map(code)
                counts = codes.value_counts().to_dict()
                counts['__missing__'] = int(missing.sum())
                assert sum(counts.values()) == len(group)
                for category, count in counts.items():
                    category_rows.append(dict(role=role, outcome=stratum, feature=feature,
                        category=category, n=int(count), denominator=len(group),
                        percent=100 * count / len(group),
                        missing_definition='null_blank_and_configured_zero; encoded_unknown_retained_as_category'))
            for feature in ['age_years', 'NU_CONTATO', 'potential_observation_days']:
                values = pd.to_numeric(group[feature], errors='coerce')
                valid = values.dropna()
                numeric_rows.append(dict(role=role, outcome=stratum, feature=feature,
                    n_total=len(group), n_observed=len(valid), n_missing=int(values.isna().sum()),
                    missing_percent=100 * values.isna().mean(), mean=float(valid.mean()),
                    q1=float(valid.quantile(.25)), median=float(valid.median()), q3=float(valid.quantile(.75))))
        for rule in ['pred_argmax', 'pred_policy']:
            matrix = pd.crosstab(pred.y_true, pred[rule]).reindex(index=[0, 1, 2], columns=[0, 1, 2], fill_value=0)
            assert int(matrix.to_numpy().sum()) == len(pred)
            f1s = []
            for actual in [0, 1, 2]:
                row_n, col_n = int(matrix.loc[actual].sum()), int(matrix[actual].sum())
                assert row_n == overview[arm]['class_' + str(actual) + '_n']
                f1s.append(2 * matrix.loc[actual, actual] / (row_n + col_n))
                for predicted in [0, 1, 2]:
                    count = int(matrix.loc[actual, predicted])
                    confusion_rows.append(dict(arm=arm, role=role, rule=rule, actual=actual,
                        predicted=predicted, actual_label=LABELS[actual], predicted_label=LABELS[predicted],
                        n=count, actual_denominator=row_n, row_percent=100 * count / row_n,
                        total_n=len(pred), aggregation='pooled_records'))
            f1 = float(np.mean(f1s))
            assert abs(f1 - overview[arm][rule + '_pooled_f1']) < 1e-9
            checks.append(dict(arm=arm, rule=rule, n=len(pred), source_ids_and_targets_match=True,
                confusion_macro_f1=f1, released_macro_f1=overview[arm][rule + '_pooled_f1']))
        print(arm, len(pred), 'verified', flush=True)
    pd.DataFrame(category_rows).to_csv(OUT / 'baseline_categorical.csv', index=False)
    pd.DataFrame(numeric_rows).to_csv(OUT / 'baseline_numeric.csv', index=False)
    pd.DataFrame(confusion_rows).to_csv(OUT / 'selected_confusions.csv', index=False)
    (OUT / 'checks.json').write_text(json.dumps(checks, indent=2))
    manifest = dict(created_at=datetime.now(timezone.utc).isoformat(),
        purpose='R11 manuscript descriptive supplement; no refitting or reselection',
        upstream_sha256=upstream, class_labels=LABELS,
        missingness='Null/blank and configured structural zero; ignored/unknown codes remain explicit categories. This is not identical to complete clinical information.',
        limitations=['Descriptive characteristics conditional on modelled outcomes.',
                     'All category percentages use full role/outcome-stratum denominator, including missing and unknown.',
                     'Numeric summaries exclude missing; age follows notification year minus birth year.',
                     'Potential observation is extraction minus notification, not observed treatment duration.',
                     'No confidence intervals or causal effects from this export.',
                     '2025 excluded. No individual rows released.'],
        script_sha256=sha(__file__), files={p.name: sha(p) for p in OUT.iterdir() if p.is_file()})
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print('R11 exported', len(category_rows), 'categorical rows;', len(confusion_rows), 'confusion cells', flush=True)


if __name__ == '__main__':
    main()
