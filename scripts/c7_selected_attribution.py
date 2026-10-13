"""Native TreeSHAP for the internally selected primary model in each outer fold.

Refits are accepted only if all saved external uncalibrated probabilities reproduce.
Attributions explain raw margins, separately by family/fold, before calibration.
"""
import json
import os
from pathlib import Path
import pickle
import time

import numpy as np
import pandas as pd


def native_contributions(estimator, name, frame, n_classes):
    if name == 'lightgbm':
        values = estimator.predict(frame, pred_contrib=True)
        values = np.asarray(values).reshape(len(frame), n_classes, frame.shape[1] + 1)
        margin = estimator.predict(frame, raw_score=True)
    elif name == 'catboost':
        from catboost import Pool
        cats = [c for c in frame if isinstance(frame[c].dtype, pd.CategoricalDtype)]
        pool = Pool(frame, cat_features=cats)
        values = estimator.get_feature_importance(pool, type='ShapValues', thread_count=4)
        margin = estimator.predict(pool, prediction_type='RawFormulaVal', thread_count=4)
    else:
        raise ValueError('Unsupported selected family; do not substitute another model')
    if values.shape != (len(frame), n_classes, frame.shape[1] + 1):
        raise ValueError('Unexpected native SHAP shape')
    error = float(np.max(np.abs(values.sum(axis=2) - margin)))
    if not np.allclose(values.sum(axis=2), margin, atol=1e-7, rtol=1e-7):
        raise ValueError('SHAP additivity failed')
    return values, error


def main():
    from tb_outcomes.dl_parallel import atomic_json, file_hash
    from tb_outcomes.executor import build_model_input
    from tb_outcomes.models import build_adapter, load_models_config
    from tb_outcomes.preprocess import infer_column_types, load_preprocess_config
    from tb_outcomes.shap_analysis import summarize, stability

    project = Path(os.environ['SMARTTB_PROJECT']).resolve()
    run = project / 'data/reanalysis/c6_20260919/primary_cpu/a0a12b4c53f318709c0c'
    selected = project / 'data/reanalysis/c7_20260923/primary_selected'
    out = project / 'data/reanalysis/c7_20261004/attribution'
    out.mkdir(exist_ok=False)
    manifest = json.loads((run / 'run.json').read_text())
    selected_manifest = json.loads((selected / 'manifest.json').read_text())
    for name, digest in selected_manifest['files'].items():
        if file_hash(selected / name) != digest:
            raise ValueError('Selection artifact changed')
    for name, digest in manifest['sources'].items():
        if file_hash(run / 'runtime' / name) != digest:
            raise ValueError('Frozen runtime changed: ' + name)
    if file_hash(run / 'inputs.pkl') != manifest['input_sha256']:
        raise ValueError('Frozen inputs changed')
    with (run / 'inputs.pkl').open('rb') as f:
        bundle = pickle.load(f)
    winners = json.loads((selected / 'selected_by_fold.json').read_text())
    predictions = pd.read_parquet(selected / 'selected_predictions.parquet')
    config = load_models_config(run / 'runtime/configs/models.yaml')
    pre = load_preprocess_config(run / 'runtime/configs/preprocess.yaml')
    entries = {**config.baselines, **config.models}
    seeds, per_class = [42, 101, 202], 100
    atomic_json(out / 'protocol.json', {
        'selection_manifest_sha256': file_hash(selected / 'manifest.json'),
        'script_sha256': file_hash(__file__), 'seeds': seeds, 'sample_per_class_per_fold': per_class,
        'selection': 'existing inner-only winners; no new model selection',
        'scope': 'external records only; class-balanced descriptive sample, not population weighted',
        'scale': 'raw margin before calibration; separate summaries for every fold/family',
        'background': 'native tree-path-dependent training distribution',
        'refit_acceptance': 'all external saved uncalibrated probabilities equal within atol=rtol=1e-8',
        'limitations': ['Noncausal model attribution; correlated features share credit',
                       'Does not explain calibrated probabilities or threshold policy',
                       'No pooled magnitude across families or claims of intervention benefit']})
    os.chdir(out)  # CatBoost auxiliary files belong here, never in frozen runtime.
    X, y = bundle['X'], bundle['y']
    outer = np.asarray(bundle['outer'])
    ids = np.asarray(bundle['record_ids'])
    completed_folds = []
    atomic_json(out / 'status.json', {'state': 'running', 'completed_folds': []})
    for row in winners:
        fold, name = row['outer_fold'], row['model']
        atomic_json(out / 'status.json', {'state': 'running', 'completed_folds': completed_folds,
                                        'current_fold': fold, 'model': name})
        if name not in ['lightgbm', 'catboost'] or row['strategy'] != 'none':
            raise ValueError('Attribution protocol requires review for this selected procedure')
        started = time.time()
        dev, evaluation = np.flatnonzero(outer != fold), np.flatnonzero(outer == fold)
        params = dict(row['parameters'])
        params['thread_count' if name == 'catboost' else 'n_jobs'] = manifest['threads']
        adapter = build_adapter(name, entries[name], random_state=bundle['seed'], params=params)
        types = infer_column_types(X.iloc[dev], bundle['specs'], pre.max_cardinality)
        train, test = build_model_input(X.iloc[dev], X.iloc[evaluation], types,
                                      adapter.capabilities.preprocess_profile, adapter.input_kind, pre)
        adapter.fit(train, y.iloc[dev].to_numpy())
        scores = adapter.predict_raw_scores(test)
        saved = predictions.set_index('source_row_id').loc[ids[evaluation]]
        original = saved[[f'raw_score_{c}' for c in bundle['classes']]].to_numpy()
        error = float(np.max(np.abs(scores - original)))
        if not np.allclose(scores, original, atol=1e-8, rtol=1e-8):
            raise ValueError(f'Fold {fold} refit did not reproduce saved scores: {error}')
        directory = out / f'fold{fold}'
        directory.mkdir()
        summaries = []
        for seed in seeds:
            rng = np.random.default_rng(seed)
            true = y.iloc[evaluation].to_numpy()
            sample = np.sort(np.concatenate([rng.choice(np.flatnonzero(true == c),
                min(per_class, int(np.sum(true == c))), replace=False) for c in bundle['classes']]))
            values, additivity = native_contributions(adapter._est, name, test.iloc[sample], len(bundle['classes']))
            np.savez_compressed(directory / f'seed{seed}.npz', values=values,
                                source_row_id=ids[evaluation[sample]], feature_names=np.asarray(test.columns, dtype=str))
            summary = summarize(values[:, :, :-1].transpose(1, 0, 2), test.columns,
                                bundle['labels'], extra={'model': name, 'outer_fold': fold, '_run': str(seed)})
            summary.to_csv(directory / f'seed{seed}_summary.csv', index=False)
            summaries.append(summary)
            atomic_json(directory / f'seed{seed}_checks.json', {'n': len(sample),
                        'max_additivity_error': additivity, 'prediction_max_abs_difference': error})
        stability(summaries).to_csv(directory / 'sampling_stability.csv', index=False)
        atomic_json(directory / 'manifest.json', {'model': name, 'fold': fold, 'parameters': params,
                    'external_records_checked': len(evaluation), 'prediction_max_abs_difference': error,
                    'seconds': time.time() - started,
                    'files': {p.name: file_hash(p) for p in directory.iterdir() if p.is_file()}})
        completed_folds.append(fold)
        atomic_json(out / 'status.json', {'state': 'running', 'completed_folds': completed_folds})
        print(json.dumps({'fold': fold, 'model': name, 'seconds': time.time() - started, 'max_delta': error}), flush=True)
        del adapter, train, test
    atomic_json(out / 'status.json', {'state': 'completed', 'completed_folds': completed_folds})
    atomic_json(out / 'manifest.json', {'status': 'validated_within_scope_not_released',
                'files': {str(p.relative_to(out)): file_hash(p) for p in out.rglob('*')
                          if p.is_file() and 'catboost_info' not in p.parts}})


if __name__ == '__main__':
    main()
