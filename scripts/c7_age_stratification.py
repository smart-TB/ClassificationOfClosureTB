"""Descriptive race/age stratification of frozen primary external predictions."""
import json
from pathlib import Path
import pickle

import pandas as pd

from tb_outcomes.cluster_inference import paired_cluster_bootstrap
from tb_outcomes.dl_parallel import atomic_json, file_hash
from tb_outcomes.equity import build_subgroup_frame


def main():
    root = Path('data/reanalysis/c7_20261004')
    selected = Path('data/reanalysis/c7_20260923/primary_selected')
    run = Path('data/reanalysis/c6_20260919/primary_cpu/a0a12b4c53f318709c0c')
    manifest = json.loads((selected / 'manifest.json').read_text())
    for name, digest in manifest['files'].items():
        if file_hash(selected / name) != digest:
            raise ValueError('Selected artifact changed')
    m = json.loads((run / 'run.json').read_text())
    if file_hash(run / 'inputs.pkl') != m['input_sha256']:
        raise ValueError('Frozen input changed')
    with (run / 'inputs.pkl').open('rb') as file:
        bundle = pickle.load(file)
    groups = build_subgroup_frame(bundle['X'], bundle['outer'])
    groups['source_row_id'] = bundle['record_ids']
    groups['cluster'] = bundle['clusters'].to_numpy()
    predictions = pd.read_parquet(selected / 'selected_predictions.parquet')
    frame = predictions.merge(groups[['source_row_id', 'cluster', 'raca_cor', 'faixa_etaria']],
                              on='source_row_id', validate='one_to_one')
    valid = frame.raca_cor.isin(['1', '2', '3', '4', '5']) & frame.faixa_etaria.notna()
    eligible = frame.loc[valid].copy()
    out = root / 'age_race_stratification'
    out.mkdir(exist_ok=False)
    coverage = dict(total=len(frame), analyzed=len(eligible), excluded=int((~valid).sum()),
                    missing_race=int(frame.raca_cor.isna().sum()), missing_age=int(frame.faixa_etaria.isna().sum()))
    atomic_json(out / 'coverage.json', coverage)
    tables, counts, omitted = [], [], []
    for (race, age), group in eligible.groupby(['raca_cor', 'faixa_etaria'], observed=True):
        for c in bundle['classes']:
            counts.append(dict(race=race, age=str(age), class_code=c,
                               n=len(group), events=int(group.y_true.eq(c).sum())))
        for rule in ['pred_argmax', 'pred_policy']:
            if group.cluster.nunique() < 2:
                omitted.append(dict(race=race, age=str(age), rule=rule, reason='fewer_than_two_geographic_clusters'))
                continue
            table = paired_cluster_bootstrap({'primary_selected': group}, group,
                    bundle['classes'], rule=rule, replicates=2000, seed=42,
                    min_denominator=30, strata_col=None)
            table['race'], table['age'] = race, str(age)
            tables.append(table)
    pd.concat(tables, ignore_index=True).to_csv(out / 'stratified_metrics.csv', index=False)
    pd.DataFrame(counts).to_csv(out / 'stratum_counts.csv', index=False)
    atomic_json(out / 'omitted_intervals.json', omitted)
    atomic_json(out / 'manifest.json', {
        'status': 'validated_within_scope_not_released', 'script_sha256': file_hash(__file__),
        'selection_manifest_sha256': file_hash(selected / 'manifest.json'),
        'input_sha256': m['input_sha256'], 'age_bins': 'unchanged equity.py age bins',
        'files': {p.name: file_hash(p) for p in out.iterdir() if p.is_file()},
        'limitations': ['Exploratory age-stratified description, not a causal test of ethnic disparities',
                       'No claim that age explains a gap without consistent within-stratum evidence',
                       'Unknown or invalid race and missing age excluded only from this table',
                       'Intervals condition on fixed predictions; small denominators suppressed at 30',
                       'No multiplicity adjustment; correlated unequal geographic clusters limit inference']})
    print(out, flush=True)


if __name__ == '__main__':
    main()
