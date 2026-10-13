"""Validate acceleration against independent fits, including resampling/weights."""
import numpy as np
import pandas as pd
import pytest

from tb_outcomes.calibration import load_calibration_config
from tb_outcomes.executor import _fit_transform_predict
from tb_outcomes.gb_parallel import _fit_group, fit_gradient_boosting
from tb_outcomes.imbalance import load_imbalance_config
from tb_outcomes.model_selection import Candidate
from tb_outcomes.models import load_models_config
from tb_outcomes.nested_selection import fit_selected_pipeline
from tb_outcomes.preprocess import load_preprocess_config


def context():
    rng = np.random.default_rng(741)
    y = np.tile([0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 1, 2], 30)
    x = pd.DataFrame(rng.normal(size=(len(y), 4)), columns=list('abcd'))
    x.loc[::11, 'a'] = np.nan
    x['category'] = np.where(np.arange(len(y)) % 3, 'yes', 'no')
    cfg = load_models_config('configs/models.yaml')
    return dict(X=x, y=pd.Series(y), strata=pd.Series(np.repeat(np.arange(30), 12)),
        entries={**cfg.baselines, **cfg.models}, specs={},
        pre=load_preprocess_config('configs/preprocess.yaml'),
        imb=load_imbalance_config('configs/imbalance.yaml'), seed=42, n_classes=3)


STRATEGIES = ['none', 'random_undersampling', 'random_oversampling', 'local_cost_sensitive']


@pytest.mark.parametrize('strategy', STRATEGIES)
def test_stages_equal_independent_fits(strategy):
    ctx = context()
    cs = [Candidate('gradient_boosting', strategy, dict(n_estimators=n, learning_rate=0.05)) for n in [5, 13]]
    fit, pred = np.arange(240), np.arange(240, 360)
    joint = _fit_group(ctx, cs, fit, pred)
    for c in cs:
        np.testing.assert_array_equal(joint[c.key], _fit_group(ctx, [c], fit, pred)[c.key])


@pytest.mark.parametrize('strategy', STRATEGIES)
def test_parallel_pipeline_matches_serial_all_raw_scores_and_selection(strategy):
    ctx = context()
    groups = np.repeat(np.arange(30), 12)
    cs = [Candidate('gradient_boosting', strategy, dict(n_estimators=n, learning_rate=lr))
          for lr, n in [(0.05, 5), (0.05, 13), (0.1, 13)]]
    cal = load_calibration_config('configs/calibration.yaml')
    evaluation = ctx['X'].iloc[:17].copy()
    traces = []
    def original(c, fit, pred):
        raw = _fit_group(ctx, [c], fit, pred)[c.key]
        traces.append(raw)
        return raw
    def final(c):
        raw = _fit_transform_predict(ctx['X'], ctx['y'], evaluation, c.model, c.strategy,
            ctx['entries'], ctx['specs'], ctx['pre'], ctx['imb'], ctx['strata'], 42, 3, params=c.params)
        traces.append(raw)
        return raw
    serial = fit_selected_pipeline(cs, ctx['y'].to_numpy(), groups, original, final, cal,
                                   n_selection_folds=3, n_calibration_folds=3)
    observed = []
    parallel = fit_gradient_boosting(ctx, groups, evaluation, cs, cal, workers=2,
        n_selection_folds=3, n_calibration_folds=3,
        trace_callback=lambda c, fit, pred, raw: observed.append(raw))
    assert len(observed) == len(traces) == 13
    for actual, expected in zip(observed, traces):
        np.testing.assert_array_equal(actual, expected)
    for k in ['raw_scores', 'probabilities', 'pred_argmax', 'pred_policy']:
        np.testing.assert_array_equal(parallel[k], serial[k])
    for k in ['candidate', 'calibration', 'policy']:
        assert parallel['selection'][k] == serial['selection'][k]
    assert parallel['thresholds'] == serial['thresholds']
    pd.testing.assert_frame_equal(parallel['inner_scores'], serial['inner_scores'])


def test_rejects_invalid_budget_and_early_stopping():
    ctx = context()
    cs = [Candidate('gradient_boosting', 'none', dict(n_estimators=n, n_iter_no_change=2)) for n in [5, 13]]
    with pytest.raises(ValueError, match='early stopping'):
        _fit_group(ctx, cs, np.arange(240), np.arange(240, 360))
    with pytest.raises(ValueError, match='budgeted'):
        fit_gradient_boosting(ctx, np.arange(360), ctx['X'], cs, None, workers=5)
