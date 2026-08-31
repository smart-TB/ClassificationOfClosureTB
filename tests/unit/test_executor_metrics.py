import numpy as np

from tb_outcomes.executor import discrimination_metrics


def _synthetic(n=800, seed=0):
    rng = np.random.RandomState(seed)
    y = rng.choice([0, 1, 2], size=n, p=[0.6, 0.3, 0.1])
    probs = np.zeros((n, 3))
    for c in range(3):
        probs[:, c] = (y == c) * 0.5 + rng.rand(n) * 0.5
    probs /= probs.sum(axis=1, keepdims=True)
    return y, probs


def test_has_per_class_and_global_rows():
    y, probs = _synthetic()
    pred = probs.argmax(axis=1)
    rows = discrimination_metrics(y, probs, pred, classes=[0, 1, 2])
    per_class = [r for r in rows if r["class"] != "__global__"]
    glob = [r for r in rows if r["class"] == "__global__"][0]
    assert len(per_class) == 3
    assert "f1_macro" in glob and "balanced_accuracy" in glob
    assert all(r["axis"] == "discrimination" for r in rows)


def test_ranking_metrics_are_in_unit_interval():
    y, probs = _synthetic()
    pred = probs.argmax(axis=1)
    for r in discrimination_metrics(y, probs, pred, classes=[0, 1, 2]):
        for key in ("recall", "precision", "auprc", "roc_auc", "f1_macro", "balanced_accuracy"):
            if key in r:
                assert 0.0 <= r[key] <= 1.0
