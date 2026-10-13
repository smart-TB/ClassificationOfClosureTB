"""Confusion matrices before and after calibration for both declared baselines."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from tb_outcomes.dl_parallel import atomic_json, completed, file_hash


def main():
    run = Path('data/reanalysis/c6_20260919/primary_cpu/a0a12b4c53f318709c0c')
    manifest = json.loads((run / 'run.json').read_text())
    rows, sources = [], []
    for job in manifest['jobs']:
        if job['model'] not in ['majority_class', 'stratified_random']:
            continue
        if not completed(run, job):
            raise ValueError('Baseline checkpoint is invalid')
        path = run / 'jobs' / (job['id'] + '.parquet')
        frame = pd.read_parquet(path)
        columns = sorted(c for c in frame if c.startswith('raw_score_'))
        classes = [int(c.removeprefix('raw_score_')) for c in columns]
        raw = np.asarray(classes)[frame[columns].to_numpy().argmax(axis=1)]
        for rule, values in [('uncalibrated_argmax', raw), ('calibrated_argmax', frame.pred_argmax),
                             ('calibrated_policy', frame.pred_policy)]:
            matrix = confusion_matrix(frame.y_true, values, labels=classes)
            for i, actual in enumerate(classes):
                for j, predicted in enumerate(classes):
                    rows.append(dict(model=job['model'], strategy=job['strategy'], fold=job['fold'],
                                     rule=rule, actual=actual, predicted=predicted, count=int(matrix[i, j])))
        sources.append({'path': str(path), 'sha256': file_hash(path)})
    if not rows:
        raise ValueError('No baseline predictions')
    out = Path('data/reanalysis/c7_20261004/baseline_confusions')
    out.mkdir(exist_ok=False)
    pd.DataFrame(rows).to_csv(out / 'confusions_per_fold.csv', index=False)
    pd.DataFrame(rows).groupby(['model', 'strategy', 'rule', 'actual', 'predicted'], as_index=False)['count'].sum().to_csv(
        out / 'confusions_pooled.csv', index=False)
    atomic_json(out / 'manifest.json', {'status': 'validated_within_scope_not_released',
                'script_sha256': file_hash(__file__), 'sources': sources,
                'files': {p.name: file_hash(p) for p in out.iterdir()},
                'interpretation': 'Saved uncalibrated argmax is distinct from calibrated decision rules; no theoretical 1/3 macro-F1 claim'})
    print(out, flush=True)


if __name__ == '__main__':
    main()
