import numpy as np
import pandas as pd

from tb_outcomes.calibration import load_calibration_config
from tb_outcomes.executor import PairResult, aggregate_pair, run_benchmark
from tb_outcomes.splits import make_outer_folds

CAL = load_calibration_config("configs/calibration.yaml")


def test_aggregates_five_folds_into_five_stats():
    rows = []
    for f in range(5):
        rows.append({"model": "m", "strategy": "s", "outer_fold": f,
                     "class": "__global__", "axis": "discrimination", "f1_macro": 0.5 + 0.1 * f})
    res = PairResult(model="m", strategy="s", metric_rows=rows)
    agg = aggregate_pair(res)
    f1 = [r for r in agg if r["metric"] == "f1_macro" and r["class"] == "__global__"][0]
    assert f1["n_folds"] == 5
    np.testing.assert_allclose(f1["mean"], 0.7)
    np.testing.assert_allclose(f1["min"], 0.5)
    np.testing.assert_allclose(f1["max"], 0.9)
    np.testing.assert_allclose(f1["median"], 0.7)


class _Caps:
    def __init__(self, name, sw, calreq=False):
        self.name = name
        self.sample_weight = sw
        self.calibrator_required = calreq


def _make_fold_factory(y):
    def factory(model, strategy):
        def make_fold_fn(dev_idx, eval_idx, strategy):
            def inner(tr, va):
                yv = y.iloc[dev_idx].to_numpy()[va]
                s = np.full((len(va), 3), 0.2)
                s[np.arange(len(va)), yv] = 0.8
                return s

            def dev_eval():
                ye = y.iloc[eval_idx].to_numpy()
                s = np.full((len(eval_idx), 3), 0.2)
                s[np.arange(len(eval_idx)), ye] = 0.8
                return s

            return inner, dev_eval
        return make_fold_fn
    return factory


def test_incompatible_cost_cell_is_marked_not_run():
    rng = np.random.RandomState(0)
    y = pd.Series(rng.choice([0, 1, 2], size=900, p=[0.6, 0.3, 0.1]))
    clusters = pd.Series(rng.randint(0, 18, size=900))
    outer = make_outer_folds(pd.DataFrame(index=y.index), clusters, y, n_folds=5)
    caps = {"weighted": _Caps("weighted", sw=True), "unweighted": _Caps("unweighted", sw=False)}

    result = run_benchmark(
        models=["weighted", "unweighted"],
        strategies=["random_oversampling", "local_cost_sensitive"],
        y=y, record_clusters=clusters, outer_folds=outer, n_classes=3, classes=[0, 1, 2],
        make_fold_factory=_make_fold_factory(y), capabilities_of=lambda m: caps[m],
        cal_cfg=CAL, n_inner=5,
    )
    status = {(r["model"], r["strategy"]): r["status"] for r in result.status_rows}
    assert status[("unweighted", "local_cost_sensitive")] == "not_run_incompatible"
    assert status[("weighted", "local_cost_sensitive")] == "ok"
    assert status[("unweighted", "random_oversampling")] == "ok"
    # a célula incompatível não polui as métricas
    bad = [r for r in result.metric_rows
           if r["model"] == "unweighted" and r["strategy"] == "local_cost_sensitive"]
    assert bad == []


def test_a_broken_pair_is_recorded_not_fatal():
    rng = np.random.RandomState(0)
    y = pd.Series(rng.choice([0, 1, 2], size=600, p=[0.6, 0.3, 0.1]))
    clusters = pd.Series(rng.randint(0, 12, size=600))
    outer = make_outer_folds(pd.DataFrame(index=y.index), clusters, y, n_folds=5)
    caps = {"good": _Caps("good", sw=True), "bad": _Caps("bad", sw=True)}

    good_factory = _make_fold_factory(y)

    def factory(model, strategy):
        if model == "bad":
            def make_fold_fn(dev_idx, eval_idx, strategy):
                def boom(*_):
                    raise RuntimeError("modelo quebrado")
                return boom, boom
            return make_fold_fn
        return good_factory(model, strategy)

    result = run_benchmark(
        models=["good", "bad"], strategies=["random_oversampling"],
        y=y, record_clusters=clusters, outer_folds=outer, n_classes=3, classes=[0, 1, 2],
        make_fold_factory=factory, capabilities_of=lambda m: caps[m], cal_cfg=CAL, n_inner=3,
    )
    status = {(r["model"], r["strategy"]): r["status"] for r in result.status_rows}
    assert status[("bad", "random_oversampling")] == "error"
    assert status[("good", "random_oversampling")] == "ok"          # o benchmark continuou
    assert any(r["model"] == "good" for r in result.oof_rows)
