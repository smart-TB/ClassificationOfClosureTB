"""Execution-only acceleration for the declared sklearn Gradient Boosting search.

Inner fits are independent; candidates that differ only in n_estimators share
one trajectory. Selection, calibration and policies still use the original code.
No evaluation labels enter this module.
"""
from __future__ import annotations

import hashlib
import json
import pickle
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config

from tb_outcomes.executor import _fit_transform_predict, _take_rows, as_score_matrix, build_model_input
from tb_outcomes.imbalance import apply_strategy
from tb_outcomes.model_selection import three_way_group_splits
from tb_outcomes.models import build_adapter
from tb_outcomes.nested_selection import fit_selected_pipeline
from tb_outcomes.preprocess import infer_column_types
from tb_outcomes.splits import make_inner_folds

_CONTEXT = None


def _initialize(context_file):
    global _CONTEXT
    # Private temporary pickle written by this process, never an external file.
    with open(context_file, 'rb') as fh:
        _CONTEXT = pickle.load(fh)


def _fit_group(context, candidates, fit, pred):
    """One fresh preprocessing/resampling/estimator pipeline per training fold."""
    c = max(candidates, key=lambda a: a.params['n_estimators'])
    if any(a.model != 'gradient_boosting' or a.strategy != c.strategy for a in candidates):
        raise ValueError('Only same-strategy Gradient Boosting candidates may share stages')
    if any({k: v for k, v in a.params.items() if k != 'n_estimators'} !=
           {k: v for k, v in c.params.items() if k != 'n_estimators'} for a in candidates):
        raise ValueError('All parameters except n_estimators must agree')
    x, y, strata = context['X'], context['y'], context['strata']
    if len(candidates) == 1:
        return {c.key: _fit_transform_predict(x.iloc[fit], y.iloc[fit], x.iloc[pred],
            c.model, c.strategy, context['entries'], context['specs'], context['pre'],
            context['imb'], strata.iloc[fit], context['seed'], context['n_classes'], params=c.params)}
    adapter = build_adapter(c.model, context['entries'][c.model],
                            random_state=context['seed'], params=c.params)
    estimator = adapter._est
    # Early stopping and warm starts invalidate the shared-trajectory contract.
    if (estimator.n_iter_no_change is not None or estimator.warm_start or
            not adapter.capabilities.native_probability):
        raise ValueError('Stage sharing requires fresh native-probability GB without early stopping')
    types = infer_column_types(x.iloc[fit], context['specs'], context['pre'].max_cardinality)
    train, test = build_model_input(x.iloc[fit], x.iloc[pred], types,
        adapter.capabilities.preprocess_profile, adapter.input_kind, context['pre'])
    res = apply_strategy(y.iloc[fit], strata.iloc[fit], c.strategy, context['imb'],
                         context['seed'], supports_weight=adapter.capabilities.sample_weight)
    try:
        adapter.fit(_take_rows(train, res.train_index), y.iloc[fit].to_numpy()[res.train_index],
                    sample_weight=res.sample_weight)
        wanted = {a.params['n_estimators']: a.key for a in candidates}
        scores = {wanted[n]: as_score_matrix(np.array(p, copy=True), context['n_classes'])
                  for n, p in enumerate(estimator.staged_predict_proba(test), 1) if n in wanted}
        if set(scores) != {a.key for a in candidates}:
            raise RuntimeError('Incomplete stage predictions')
        return scores
    finally:
        close = getattr(adapter, 'close', None)
        if close is not None:
            close()


def _process_task(candidates, fit, pred):
    if _CONTEXT is None:
        raise RuntimeError('Missing worker context')
    return _fit_group(_CONTEXT, candidates, fit, pred)


def _key(candidate, fit, pred):
    h = hashlib.sha256()
    for a in (fit, pred):
        h.update(np.asarray(a, dtype='int64').tobytes())
        h.update(b'|')
    return candidate.key, h.hexdigest()


def fit_gradient_boosting(context, groups, evaluation_X, candidates, cal_cfg, *,
                          workers=4, n_selection_folds=5, n_calibration_folds=5,
                          fold_method='binpack', reuse_stages=True, event_callback=None,
                          trace_callback=None):
    """Context contains DEVELOPMENT ONLY; final callback receives evaluation X only.

    CPU budget is the number of processes, with native libraries limited to one
    thread per child. Context is initialized once per process, not once per fit.
    """
    if not 1 <= workers <= 4:
        raise ValueError('Gradient Boosting requires 1..4 budgeted processes')
    if not candidates or any(c.model != 'gradient_boosting' for c in candidates):
        raise ValueError('Gradient Boosting candidates required')
    if any(not isinstance(c.params.get('n_estimators'), int) or c.params['n_estimators'] < 1
           for c in candidates):
        raise ValueError('Explicit positive integer stage counts required')
    y = context['y'].to_numpy()
    groups = np.asarray(groups)
    if len(y) != len(groups):
        raise ValueError('Development groups must align')
    classes = list(range(context['n_classes']))
    splits = list(three_way_group_splits(groups, n_selection_folds, y=y, fold_method=fold_method))
    # Fail before any expensive fit if the declared folds omit a class.
    for split in splits:
        if any(sorted(np.unique(y[idx]).tolist()) != classes for idx in split):
            raise ValueError('Missing class in an inner fold')
    cache = {}
    grouped = {}
    for c in candidates:
        params = {k: v for k, v in c.params.items() if not reuse_stages or k != 'n_estimators'}
        grouped.setdefault((c.strategy, json.dumps(params, sort_keys=True)), []).append(c)
    tasks = [(cs, fit, np.r_[cal, score]) for cs in grouped.values() for fit, cal, score in splits]

    def event(phase, **detail):
        if event_callback:
            event_callback(dict(event='gb_parallel', phase=phase, workers=workers, **detail))

    with tempfile.TemporaryDirectory(prefix='smarttb-gb-') as scratch, parallel_config(
            backend='loky', inner_max_num_threads=1):
        context_file = str(Path(scratch) / 'development.pkl')
        if workers > 1:
            with open(context_file, 'wb') as fh:
                pickle.dump(context, fh, protocol=5)
        with Parallel(n_jobs=workers, initializer=_initialize, initargs=(context_file,),
                      pre_dispatch=workers, batch_size=1) as pool:
            def compute(batch, phase):
                event(phase + '_started', fits=len(batch))
                output = ([_fit_group(context, *t) for t in batch] if workers == 1 else
                          pool(delayed(_process_task)(*t) for t in batch))
                for (cs, fit, pred), scores in zip(batch, output):
                    for c in cs:
                        cache[_key(c, fit, pred)] = scores[c.key]
                event(phase + '_completed', fits=len(batch))

            compute(tasks, 'selection')
            prepared = False

            def predict(c, fit, pred):
                nonlocal prepared
                k = _key(c, fit, pred)
                if k not in cache and not prepared:
                    compute([([c], a, b) for a, b in make_inner_folds(
                        pd.Series(groups), pd.Series(y), n_calibration_folds, fold_method)], 'calibration')
                    prepared = True
                if k not in cache:
                    raise RuntimeError('Unexpected prediction request; declared partitions differ')
                scores = cache.pop(k)
                if trace_callback:
                    trace_callback(c, fit, pred, scores)
                return scores

            def final(c):
                event('final_fit_started')
                raw = _fit_transform_predict(context['X'], context['y'], evaluation_X,
                    c.model, c.strategy, context['entries'], context['specs'], context['pre'],
                    context['imb'], context['strata'], context['seed'], context['n_classes'], params=c.params)
                if trace_callback:
                    trace_callback(c, None, None, raw)
                event('final_fit_completed')
                return raw

            return fit_selected_pipeline(candidates, y, groups, predict, final, cal_cfg,
                n_selection_folds=n_selection_folds, n_calibration_folds=n_calibration_folds,
                fold_method=fold_method)
