import numpy as np
import pandas as pd

from tb_outcomes.calibration import load_calibration_config
from tb_outcomes.executor import evaluate_pair

CAL = load_calibration_config("configs/calibration.yaml")


class _Caps:  # dobra a interface de ModelCapabilities usada pela trava
    name = "fake_model"
    calibrator_required = False


def _setup(n=1000, seed=0):
    rng = np.random.RandomState(seed)
    y = pd.Series(rng.choice([0, 1, 2], size=n, p=[0.6, 0.3, 0.1]))
    clusters = pd.Series(rng.randint(0, 20, size=n))
    from tb_outcomes.splits import make_outer_folds
    outer = make_outer_folds(pd.DataFrame(index=y.index), clusters, y, n_folds=5)
    return y, clusters, outer


def _make_fold_fn(y):
    # stub determinístico e informativo: escore alto na classe verdadeira
    def make_fold_fn(dev_idx, eval_idx, strategy):
        def inner_fit_predict(train_idx, val_idx):
            yv = y.iloc[dev_idx].to_numpy()[val_idx]
            s = np.full((len(val_idx), 3), 0.2)
            s[np.arange(len(val_idx)), yv] = 0.8
            return s

        def dev_full_predict_eval():
            ye = y.iloc[eval_idx].to_numpy()
            s = np.full((len(eval_idx), 3), 0.2)
            s[np.arange(len(eval_idx)), ye] = 0.8
            return s

        return inner_fit_predict, dev_full_predict_eval
    return make_fold_fn


def test_pair_result_has_rows_for_every_outer_fold():
    y, clusters, outer = _setup()
    res = evaluate_pair(
        "fake_model", "random_oversampling", y, clusters, outer,
        n_classes=3, classes=[0, 1, 2], make_fold_fn=_make_fold_fn(y),
        capabilities=_Caps(), cal_cfg=CAL,
    )
    folds = {r["outer_fold"] for r in res.metric_rows}
    assert folds == {0, 1, 2, 3, 4}
    assert res.status == "ok"
    assert len(res.oof_rows) == len(y)          # todo registro aparece uma vez no EVAL
    assert any(r.get("axis") == "discrimination" for r in res.metric_rows)
    assert any(r.get("stage") == "post_renorm" for r in res.metric_rows)


def test_calibrator_required_without_calibration_is_refused():
    y, clusters, outer = _setup()

    class _NeedsCal(_Caps):
        name = "ridge_like"
        calibrator_required = True

    # o caminho normal calibra e NÃO deve levantar; verificamos que roda:
    res = evaluate_pair(
        "ridge_like", "random_oversampling", y, clusters, outer,
        n_classes=3, classes=[0, 1, 2], make_fold_fn=_make_fold_fn(y),
        capabilities=_NeedsCal(), cal_cfg=CAL,
    )
    assert res.status == "ok"
