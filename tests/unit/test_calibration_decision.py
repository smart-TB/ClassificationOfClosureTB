import numpy as np

from tb_outcomes.calibration import (
    classify_argmax,
    classify_policy,
    fit_thresholds,
    load_calibration_config,
)

CFG = load_calibration_config("configs/calibration.yaml")
CLASSES = [0, 1, 2]


def _brute_force_thresholds(oof, y):
    # referência independente: exatamente a busca em grade original (f1 do sklearn),
    # desempate na MENOR threshold. Guarda a versão vetorizada de fit_thresholds.
    from sklearn.metrics import f1_score

    oof = np.asarray(oof, dtype=float)
    y = np.asarray(y)
    classes = sorted(np.unique(y).tolist())
    grid = np.linspace(0.05, 0.95, 19)
    best = {}
    for j, c in enumerate(classes):
        yb = (y == c).astype(int)
        best_t, best_s = 1.0, -1.0
        for t in grid:
            s = f1_score(yb, (oof[:, j] >= t).astype(int), zero_division=0)
            if s > best_s:
                best_s, best_t = s, t
        best[c] = float(best_t)
    return best


def test_thresholds_match_bruteforce_reference():
    rng = np.random.RandomState(7)
    y = rng.choice(CLASSES, size=3000, p=[0.6, 0.3, 0.1])
    oof = rng.dirichlet([1, 1, 1], size=3000)
    assert fit_thresholds(oof, y, CFG) == _brute_force_thresholds(oof, y)


def test_argmax_is_complete_and_exclusive():
    probs = np.array([[0.1, 0.7, 0.2], [0.5, 0.3, 0.2]])
    pred = classify_argmax(probs, CLASSES)
    assert pred.tolist() == [1, 0]
    assert set(np.unique(pred)) <= set(CLASSES)


def test_policy_never_leaves_a_row_unassigned():
    # mesmo se nenhuma classe supera seu limiar, a razão p/limiar decide (sem vazio)
    probs = np.array([[0.05, 0.05, 0.90], [0.2, 0.2, 0.6]])
    thr = {0: 0.5, 1: 0.5, 2: 0.99}
    pred = classify_policy(probs, thr, CLASSES)
    assert len(pred) == 2
    assert not any(p is None for p in pred)


def test_thresholds_fit_only_on_given_oof():
    rng = np.random.RandomState(0)
    y = rng.choice(CLASSES, size=600)
    oof = rng.dirichlet([1, 1, 1], size=600)
    thr = fit_thresholds(oof, y, CFG)
    assert set(thr.keys()) == set(CLASSES)
    assert all(0 < v <= 1 for v in thr.values())
