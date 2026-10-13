"""Audit complete correction runs and derive C7 tables from external predictions.

No model fitting, selection from outer metrics, or publication occurs here.
Confidence intervals condition on the saved fitted predictions.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from scipy.optimize import minimize, brentq
from scipy.special import expit
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, average_precision_score

from tb_outcomes.cluster_inference import paired_cluster_bootstrap, metrics_from_confusion
from tb_outcomes.dl_parallel import file_hash, atomic_json, completed
from tb_outcomes.equity import build_subgroup_frame
from tb_outcomes.selection_aggregation import aggregate_selected_runs

AUDIT_SCRIPT_SHA256 = file_hash(__file__)


def inference(predictions, mapping, classes, **kwargs):
    # One spatial outer fold has one geographic cluster. Stratifying on that
    # fold cannot estimate its geographic variation. Resample complete clusters
    # without fold strata, conditional on fixed OOF predictions; use paired draws.
    return paired_cluster_bootstrap(predictions, mapping, classes, strata_col=None, **kwargs)


def cluster_influence(frame, mapping, classes, rule):
    joined = frame[['source_row_id', 'y_true', rule]].merge(
        mapping[['source_row_id', 'cluster']], on='source_row_id', validate='one_to_one')
    names = sorted(joined.cluster.unique())
    codes = pd.Categorical(joined.cluster, categories=names).codes
    labels = {c:i for i,c in enumerate(classes)}
    tensor = np.zeros((len(names), len(classes), len(classes)), dtype=np.int64)
    np.add.at(tensor, (codes, joined.y_true.map(labels), joined[rule].map(labels)), 1)
    total = tensor.sum(0)
    baseline = metrics_from_confusion(total, classes)['f1_macro']
    rows = []
    for name, matrix in zip(names, tensor):
        score = metrics_from_confusion(total-matrix, classes)['f1_macro']
        rows.append(dict(cluster=name, n_records=int(matrix.sum()),
            record_fraction=matrix.sum()/total.sum(), rule=rule,
            full_f1_macro=baseline, leave_cluster_out_f1_macro=score,
            change=score-baseline, interpretation='Descriptive influence on fixed predictions; no model refit'))
    return rows


def require(condition, message):
    if not condition:
        raise ValueError(message)


def probability_metrics(y, p, minimum=1):
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    n = len(y)
    events = int(y.sum())
    negatives = n-events
    estimable = events >= minimum and negatives >= minimum
    clipped = np.clip(p, 1e-12, 1-1e-12)
    logit = np.log(clipped/(1-clipped))
    ece = 0.
    bins = np.minimum((p*10).astype(int), 9)
    for i in range(10):
        keep = bins == i
        if keep.any():
            ece += keep.mean()*abs(y[keep].mean()-p[keep].mean())
    intercept, slope, citl = np.nan, np.nan, np.nan
    if estimable:
        citl = brentq(lambda a: expit(logit+a).sum()-events, -100, 100)
        if np.std(logit) > 1e-8:
            def objective(beta):
                z = beta[0]+beta[1]*logit
                return np.mean(np.logaddexp(0, z)-y*z)
            def gradient(beta):
                r = expit(beta[0]+beta[1]*logit)-y
                return np.array([r.mean(), (r*logit).mean()])
            fit = minimize(objective, [0., 1.], jac=gradient, method='BFGS', options={'gtol': 1e-8})
            if fit.success:
                intercept, slope = fit.x
    return dict(n=n, events=events, non_events=negatives, prevalence=events/n if n else np.nan,
        auroc=roc_auc_score(y, p) if estimable else np.nan,
        auprc=average_precision_score(y, p) if estimable else np.nan,
        brier=np.mean((p-y)**2) if estimable else np.nan,
        log_loss=np.mean(np.logaddexp(0, logit)-y*logit) if estimable else np.nan,
        ece_fixed_10=ece if estimable else np.nan, calibration_intercept=intercept,
        calibration_slope=slope, calibration_in_large=citl,
        status='ok' if estimable else 'not_estimable',
        calibration_slope_status='ok' if np.isfinite(slope) else 'not_estimable')


def capacity_rows(y, p, ids, minimum=1):
    """Ties use stable source IDs; selection describes event capture, not benefit."""
    order = np.lexsort((np.asarray(ids, str), -np.asarray(p)))
    events = int(np.sum(y))
    rows = []
    for q in [.01, .05, .10, .20]:
        m = max(1, int(round(q*len(y))))
        chosen = order[:m]
        captured = int(np.asarray(y)[chosen].sum())
        ok = events >= minimum and m >= minimum
        ppv = captured/m if ok else np.nan
        rows.append(dict(capacity_fraction=q, n=len(y), events=events, selected=m,
            captured_events=captured, ppv=ppv, sensitivity=captured/events if ok else np.nan,
            notifications_per_observed_event=1/ppv if ppv > 0 else np.nan,
            status='ok' if ok else 'not_estimable', tie_break='ascending_source_row_id'))
    return rows


def check_frame(frame, expected, classes, job):
    cols = ['source_row_id', 'y_true', 'outer_fold']
    require(not frame.source_row_id.isna().any() and not frame.source_row_id.duplicated().any(), 'Invalid IDs')
    check = frame[cols].sort_values('source_row_id').reset_index(drop=True)
    reference = expected[cols].sort_values('source_row_id').reset_index(drop=True)
    require(check.equals(reference), f'Expected population/fold/truth differs: {job}')
    p = frame[[f'proba_{c}' for c in classes]].to_numpy(float)
    require(np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(), 'Invalid probabilities')
    require(np.allclose(p.sum(1), 1, atol=1e-8, rtol=0), 'Probability sum differs from one')
    require(np.array_equal(np.asarray(classes)[p.argmax(1)], frame.pred_argmax), 'Argmax differs')
    require(frame.pred_policy.isin(classes).all(), 'Invalid policy prediction')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', nargs='+', type=Path, required=True)
    parser.add_argument('--universe', type=Path, required=True)
    parser.add_argument('--selected', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--replicates', type=int, default=2000)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    universe = json.loads(args.universe.read_text())
    pairs = universe['pairs'] if isinstance(universe, dict) else universe
    if not args.selected.exists():
        aggregate_selected_runs(args.runs, pairs, args.selected)
    selection_manifest = json.loads((args.selected/'manifest.json').read_text())
    require(selection_manifest['expected_pairs'] == pairs, 'Selection universe mismatch')
    require({x['manifest_sha256'] for x in selection_manifest['inputs']} ==
            {file_hash(p/'run.json') for p in args.runs}, 'Selection sources mismatch')
    for name, digest in selection_manifest['files'].items():
        require(file_hash(args.selected/name) == digest, 'Selection artifact changed')
    manifests = [json.loads((run/'run.json').read_text()) for run in args.runs]
    reference_manifest = manifests[0]
    for run, m in zip(args.runs, manifests):
        require(not m['sanity'], 'Pilot cannot be reported')
        require(file_hash(run/'inputs.pkl') == m['input_sha256'], 'Input hash mismatch')
        for path, digest in m['sources'].items():
            require(file_hash(run/'runtime'/path) == digest, 'Runtime changed')
    # Trusted, hash-verified bundles written by this project's launcher.
    with (args.runs[0]/'inputs.pkl').open('rb') as f:
        bundle = pickle.load(f)
    classes = bundle['classes']
    mode = reference_manifest['mode']
    expected = pd.DataFrame({'source_row_id': bundle['record_ids'], 'y_true': bundle['y'],
                             'cluster': bundle['clusters'], 'year': bundle['years']})
    if mode == 'spatial':
        expected['outer_fold'] = bundle['outer']
        require(expected.year.lt(bundle['test_year']).all(), 'Future records in development')
    else:
        expected = expected.loc[expected.year.eq(bundle['test_year'])].copy()
        expected['outer_fold'] = f"holdout_{bundle['test_year']}"
    require(len(expected) == reference_manifest['n_expected_predictions_per_pair'], 'Wrong expected count')
    source = pd.read_parquet('data/harmonized.parquet', columns=['TRATAMENTO', 'NU_ANO'])
    require(file_hash('data/harmonized.parquet') == reference_manifest['harmonized_sha256'], 'Source changed')
    actual_source = source.loc[expected.source_row_id.astype(int)]
    require(not pd.to_numeric(actual_source.TRATAMENTO, errors='coerce').eq(6).any(), 'Postmortem entry present')
    del source, actual_source
    selected = pd.read_parquet(args.selected/'selected_predictions.parquet')
    check_frame(selected, expected, classes, 'selection_procedure')
    prediction_columns = ['source_row_id','y_true','pred_argmax','pred_policy']
    comparison = {'nested_selection_procedure': selected[prediction_columns]}
    audit_rows, cost_rows, fold_rows = [], [], []
    pooled_probability, capacity = [], []
    for pair in pairs:
        name = pair['model']+'__'+pair['strategy']
        frames = []
        for run, manifest in zip(args.runs, manifests):
            for job in manifest['jobs']:
                if (job['model'],job['strategy']) != (pair['model'],pair['strategy']):
                    continue
                require(completed(run, job), 'Checkpoint failed integrity')
                p = run/'jobs'/(job['id']+'.parquet')
                frame = pd.read_parquet(p)
                check_frame(frame, expected.loc[expected.outer_fold.eq(job['fold'])], classes, job['id'])
                meta = json.loads(p.with_suffix('.json').read_text())
                for rule, axis in [('pred_argmax','discrimination'), ('pred_policy','secondary_policy')]:
                    value = f1_score(frame.y_true, frame[rule], average='macro', labels=classes, zero_division=0)
                    stored = [r['f1_macro'] for r in meta['metric_rows'] if r.get('class') == '__global__' and r.get('axis') == axis]
                    require(len(stored)==1 and abs(stored[0]-value)<1e-12, 'Metric does not reproduce')
                    fold_rows.append(dict(comparison=name, outer_fold=job['fold'], rule=rule, f1_macro=value))
                cost_rows.append(dict(comparison=name, outer_fold=job['fold'], run_id=run.name,
                    wall_seconds=meta['finished_at']-meta['started_at'], compute=manifest['compute'],
                    peak_gpu_memory_bytes=meta.get('peak_gpu_memory_bytes'),
                    execution_amendment=json.dumps(meta.get('execution_amendment'))))
                audit_rows.append(dict(run_id=run.name, job=job['id'], n=len(frame), checks='passed',
                                       prediction_sha256=file_hash(p)))
                frames.append(frame)
        combined = pd.concat(frames, ignore_index=True)
        check_frame(combined, expected, classes, name)
        comparison[name] = combined[prediction_columns].copy()
        for c in classes:
            y = combined.y_true.eq(c).to_numpy()
            prob = combined[f'proba_{c}'].to_numpy()
            pooled_probability.append(dict(comparison=name, class_code=c, **probability_metrics(y, prob)))
            capacity.extend(dict(comparison=name, class_code=c, **r) for r in capacity_rows(y, prob, combined.source_row_id))
        print(json.dumps({'audited_pair':name,'n':len(combined)}), flush=True)
    pd.DataFrame(audit_rows).to_csv(args.out/'checkpoint_audit.csv', index=False)
    pd.DataFrame(cost_rows).to_csv(args.out/'compute_cost.csv', index=False)
    pd.DataFrame(fold_rows).to_csv(args.out/'recomputed_fold_metrics.csv', index=False)
    cluster_counts = expected[['cluster','outer_fold']].drop_duplicates().groupby('outer_fold').size().to_dict()
    atomic_json(args.out/'bootstrap_design.json', dict(mode=mode, clusters_per_outer_fold=cluster_counts,
        stratified_by=None, revision='2026-09-23: unstratified geographic cluster resampling',
        reason='Spatial fold 0 contains one cluster; within-fold geographic uncertainty is not estimable there.',
        limitation='Unequal cluster sizes can make percentile intervals unstable; inspect leave-one-cluster-out influence.'))
    for rule in ['pred_argmax','pred_policy']:
        ci = inference(comparison, expected, classes, rule=rule,
            reference='nested_selection_procedure', replicates=args.replicates)
        ci.to_csv(args.out/f'cluster_intervals_{rule}.csv', index=False)
    del comparison
    pd.DataFrame([row for rule in ['pred_argmax','pred_policy']
                  for row in cluster_influence(selected, expected, classes, rule)]).to_csv(
                      args.out/'cluster_influence.csv', index=False)
    for c in classes:
        y = selected.y_true.eq(c).to_numpy()
        p = selected[f'proba_{c}'].to_numpy()
        pooled_probability.append(dict(comparison='nested_selection_procedure',class_code=c,**probability_metrics(y,p)))
        capacity.extend(dict(comparison='nested_selection_procedure',class_code=c,**r)
                        for r in capacity_rows(y,p,selected.source_row_id))
    pd.DataFrame(pooled_probability).to_csv(args.out/'probability_metrics_pooled.csv', index=False)
    pd.DataFrame(capacity).to_csv(args.out/'capacity_pooled.csv', index=False)
    # Group labels come from the frozen features, joined by stable IDs.
    folds = bundle['outer'] if mode == 'spatial' else pd.Series(['temporal']*len(bundle['X']))
    groups = build_subgroup_frame(bundle['X'], folds)
    groups['feature_missing_fraction'] = pd.cut(bundle['X'].isna().mean(1), [-.001,.1,.25,.5,1.],
                                               labels=['0-10%','>10-25%','>25-50%','>50-100%'])
    groups['source_row_id'] = bundle['record_ids']
    joined = selected.merge(groups, on='source_row_id', how='left', validate='one_to_one')
    group_rows, group_ci, missing_rows = [], [], []
    for axis in groups.columns.drop('source_row_id'):
        missing_rows.append(dict(axis=axis, total=len(joined), missing=int(joined[axis].isna().sum())))
        for value, frame in joined.groupby(axis, observed=True, dropna=True):
            for rule in ['pred_argmax','pred_policy']:
                rates = metrics_from_confusion(confusion_matrix(frame.y_true,frame[rule],labels=classes),classes,30)
                for c in classes:
                    row = dict(axis=axis,group=str(value),rule=rule,class_code=c,
                               **probability_metrics(frame.y_true.eq(c),frame[f'proba_{c}'],30))
                    row.update({key.split(':')[0]:val for key,val in rates.items() if key.endswith(':'+str(c))})
                    group_rows.append(row)
                mapping = expected.loc[expected.source_row_id.isin(frame.source_row_id)].copy()
                try:
                    ci = inference({'nested_selection_procedure':frame},mapping,classes,
                        rule=rule,replicates=args.replicates,min_denominator=30)
                    ci['axis'],ci['group'] = axis,str(value)
                    group_ci.append(ci)
                except ValueError as exc:
                    if 'clusters' not in str(exc):
                        raise
                    missing_rows.append(dict(axis=axis,group=str(value),rule=rule,ci_status='not_estimable',reason=str(exc)))
    pd.DataFrame(group_rows).to_csv(args.out/'subgroup_metrics.csv',index=False)
    pd.concat(group_ci,ignore_index=True).to_csv(args.out/'subgroup_cluster_intervals.csv',index=False)
    pd.DataFrame(missing_rows).to_csv(args.out/'subgroup_coverage.csv',index=False)
    for rule in ['pred_argmax','pred_policy']:
        values = []
        for fold, frame in selected.groupby('outer_fold'):
            values.append(f1_score(frame.y_true,frame[rule],average='macro',labels=classes,zero_division=0))
        print(json.dumps({'selected_rule':rule,'fold_mean_f1':float(np.mean(values))}),flush=True)
    files = {p.name:file_hash(p) for p in args.out.iterdir() if p.is_file()}
    atomic_json(args.out/'audit_manifest.json',dict(created_at=datetime.now(timezone.utc).isoformat(),
        status='validated_within_scope_not_released', n_expected=len(expected), classes=dict(zip(classes,bundle['labels'])),
        experiment_id=reference_manifest['experiment_id'], runs=[str(r) for r in args.runs],
        audit_script_sha256=AUDIT_SCRIPT_SHA256,
        files=files, replicates=args.replicates, decision_rules=['pred_argmax','pred_policy'],
        probability_aggregation='pooled_records',intervals='paired unstratified geographic cluster bootstrap of fixed external predictions',
        limitations=['Conditional on saved fitted predictions; no refitting uncertainty',
          'Exploratory model comparisons; no multiplicity adjustment or proof of superiority',
          'Spatial fold 0 has one geographic cluster; draws are not stratified by fold and model mixture varies by draw',
          'Strongly unequal cluster sizes limit interval reliability; consult cluster influence diagnostics',
          'Subgroup minimum denominator 30; insufficient-event estimates suppressed',
          'Calibration diagnostics use unpenalized logistic fit and separate offset-only calibration-in-the-large',
          '2024 historically explored; not pristine independent confirmation',
          'Utility is retrospective event capture, not observed intervention benefit',
          'Other correction arms and final editorial integration remain required']))


if __name__ == '__main__':
    main()
