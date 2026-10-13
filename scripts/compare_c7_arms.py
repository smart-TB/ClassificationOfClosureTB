"""Compare frozen correction arms using a common, declared candidate universe.

Choices use inner scores only. Geographic clusters from the main design define
paired resampling, including comparisons whose validation partitions differ.
"""
import argparse
import json
from pathlib import Path
import pickle

import pandas as pd

from tb_outcomes.cluster_inference import paired_cluster_bootstrap
from tb_outcomes.dl_parallel import atomic_json, completed, file_hash
from tb_outcomes.selection_aggregation import choose_inner_winners
from tb_outcomes.utility import decision_curve


def load_selection(path):
    manifest = json.loads((path / 'manifest.json').read_text())
    for name, digest in manifest['files'].items():
        if file_hash(path / name) != digest:
            raise ValueError('Selection artifact changed: ' + str(path / name))
    records = json.loads((path / 'inner_comparison.json').read_text())
    return manifest, records


def restrict_records(records, pairs):
    selected = [r for r in records if (r['model'], r['strategy']) in pairs]
    folds = {r['outer_fold'] for r in records}
    if {(r['model'], r['strategy'], r['outer_fold']) for r in selected} != {
            (m, s, f) for m, s in pairs for f in folds}:
        raise ValueError('Incomplete common candidate universe')
    return selected


def selected_frame(records, pairs, selection_path=None):
    winners = choose_inner_winners(restrict_records(records, pairs))
    if selection_path is not None and pairs == {(r['model'], r['strategy']) for r in records}:
        saved = pd.read_json(selection_path / 'selected_by_fold.json')
        fields = ['outer_fold', 'model', 'strategy', 'candidate', 'calibration', 'policy']
        if saved[fields].astype(str).to_dict('records') != winners[fields].astype(str).to_dict('records'):
            raise ValueError('Saved selection disagrees with inner-only choices')
        path = selection_path / 'selected_predictions.parquet'
        return pd.read_parquet(path), winners, [{'path': str(path), 'sha256': file_hash(path)}]
    frames, evidence = [], []
    for row in winners.to_dict('records'):
        run = Path(row['run'])
        manifest = json.loads((run / 'run.json').read_text())
        job = next(j for j in manifest['jobs'] if j['id'] == row['job_id'])
        if not completed(run, job):
            raise ValueError('Invalid selected checkpoint: ' + row['job_id'])
        path = run / 'jobs' / (row['job_id'] + '.parquet')
        frame = pd.read_parquet(path)
        if not frame.outer_fold.eq(row['outer_fold']).all():
            raise ValueError('Selected fold disagrees with checkpoint')
        frames.append(frame)
        evidence.append({'path': str(path), 'sha256': file_hash(path)})
    return pd.concat(frames, ignore_index=True), winners, evidence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', required=True)
    args = parser.parse_args()
    if args.arm == 'all_cause':
        raise ValueError('Different endpoint/cohort: do not pair with primary labels')
    root = Path('data/reanalysis/c7_20261004')
    paths = {'primary': Path('data/reanalysis/c7_20260923/primary_selected'),
             args.arm: root / args.arm / 'selected'}
    loaded = {name: load_selection(path) for name, path in paths.items()}
    universes = [{(p['model'], p['strategy']) for p in m['expected_pairs']}
                 for m, _ in loaded.values()]
    common = set.intersection(*universes)
    if not common:
        raise ValueError('No common candidate universe')
    # Intersection is derived from declared universes, before examining performance.
    results = {name: selected_frame(records, common, paths[name])
               for name, (_, records) in loaded.items()}
    run = Path('data/reanalysis/c6_20260919/primary_cpu/a0a12b4c53f318709c0c')
    manifest = json.loads((run / 'run.json').read_text())
    if file_hash(run / 'inputs.pkl') != manifest['input_sha256']:
        raise ValueError('Primary input changed')
    with (run / 'inputs.pkl').open('rb') as file:
        bundle = pickle.load(file)
    mapping = pd.DataFrame({'source_row_id': bundle['record_ids'], 'cluster': bundle['clusters']})
    out = root / 'comparisons' / args.arm
    out.mkdir(parents=True, exist_ok=False)
    frames = {name: result[0] for name, result in results.items()}
    for rule in ['pred_argmax', 'pred_policy']:
        paired_cluster_bootstrap(frames, mapping, bundle['classes'], rule=rule,
                                 reference='primary', replicates=2000, seed=42,
                                 strata_col=None).to_csv(out / ('paired_' + rule + '.csv'), index=False)
    curves = []
    for name, (frame, winners, _) in results.items():
        winners.to_json(out / (name + '_selected_by_fold.json'), orient='records', indent=2)
        for c in bundle['classes']:
            if bundle['labels'][c] == 'cure':
                continue
            curve = decision_curve(frame.y_true.eq(c).astype(int), frame[f'proba_{c}'])
            curve['comparison'], curve['class_code'], curve['class_label'] = name, c, bundle['labels'][c]
            curves.append(curve)
    pd.concat(curves, ignore_index=True).to_csv(out / 'decision_curves.csv', index=False)
    atomic_json(out / 'manifest.json', {
        'status': 'validated_within_scope_not_released', 'arm': args.arm,
        'common_pairs': [dict(model=m, strategy=s) for m, s in sorted(common)],
        'universe_rule': 'intersection of declared candidate universes, not external performance',
        'selection': 'inner argmax macro-F1 with existing deterministic tie breaks',
        'sources': {name: {'selection_manifest_sha256': file_hash(paths[name] / 'manifest.json'),
                           'selected_shards': result[2]} for name, result in results.items()},
        'resampling_clusters': 'primary k50 geographic clusters for both members of each pair',
        'partition_comparison': args.arm in ['k27', 'k75', 'random_group'],
        'script_sha256': file_hash(__file__),
        'files': {p.name: file_hash(p) for p in out.iterdir() if p.is_file()},
        'limitations': ['Conditional on fixed external predictions; no refitting uncertainty',
                       'Unequal geographic cluster sizes limit interval reliability',
                       'Partition comparisons change validation design and cannot isolate a feature effect',
                       'Common-universe primary selection can differ from the full primary procedure',
                       'Exploratory paired differences without multiplicity adjustment',
                       'Decision curves are retrospective hypothetical utility, not intervention effects']})
    print(out, flush=True)


if __name__ == '__main__':
    main()
