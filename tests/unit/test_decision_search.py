import numpy as np
from sklearn.metrics import f1_score

from tb_outcomes.calibration import classify_policy, fit_thresholds, load_calibration_config
from tb_outcomes.decision_search import fit_joint_macro_policy


def test_joint_policy_optimizes_the_exclusive_multiclass_rule():
    rng = np.random.RandomState(17)
    y = rng.choice([0, 1, 2], 1800, p=[.75, .05, .20])
    p = rng.dirichlet([3, 1, 1], len(y))
    p[np.arange(len(y)), y] += .25
    p /= p.sum(axis=1, keepdims=True)
    legacy = fit_thresholds(p, y, load_calibration_config('configs/calibration.yaml'))
    joint = fit_joint_macro_policy(p, y, [0, 1, 2], legacy)
    score = f1_score(y, classify_policy(p, joint, [0, 1, 2]), average='macro')
    assert score >= f1_score(y, p.argmax(axis=1), average='macro')
    assert score >= f1_score(y, classify_policy(p, legacy, [0, 1, 2]), average='macro')
    assert joint == fit_joint_macro_policy(p, y, [0, 1, 2], legacy)
