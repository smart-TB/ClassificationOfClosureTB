import numpy as np
import pandas as pd
import pytest

from tb_outcomes.cluster_inference import metrics_from_confusion, paired_cluster_bootstrap


def fixture():
    p = pd.DataFrame({"source_row_id": np.arange(48), "y_true": np.tile([0, 1, 2], 16)})
    p["pred_argmax"] = p.y_true
    groups = p[["source_row_id"]].copy()
    groups["cluster"] = np.repeat(np.arange(8), 6)
    groups["outer_fold"] = groups.cluster % 2
    return p, groups


def test_paired_identical_predictions_have_exact_zero_difference():
    p, groups = fixture()
    result = paired_cluster_bootstrap({"a": p, "b": p.sample(frac=1, random_state=4)}, groups, [0, 1, 2],
                                      reference="a", replicates=100, strata_col="outer_fold")
    differences = result.query("kind == 'paired_difference'")
    assert differences.status.eq("ok").all()
    assert differences[["estimate", "ci_low", "ci_high"]].eq(0).all().all()
    assert result.query("kind == 'estimate'").estimate.eq(1).all()


def test_absent_events_and_no_predictions_are_not_reported_as_perfect_metrics():
    values = metrics_from_confusion([[10, 0], [0, 0]], [0, 1])
    assert np.isnan(values["sensitivity:1"])
    assert np.isnan(values["ppv:1"])
    assert np.isnan(values["f1_macro"])
    assert values["specificity:1"] == 1


@pytest.mark.parametrize("change", ["missing_id", "wrong_label", "missing_cluster", "spanning_cluster"])
def test_pairing_and_resampling_integrity_are_enforced(change):
    p, groups = fixture()
    q = p.copy()
    if change == "missing_id":
        q = q.iloc[:-1]
    elif change == "wrong_label":
        q.loc[0, "y_true"] = 1
    elif change == "missing_cluster":
        groups.loc[0, "cluster"] = np.nan
    else:
        groups.loc[0, "outer_fold"] = 1
    with pytest.raises(ValueError):
        paired_cluster_bootstrap({"a": p, "b": q}, groups, [0, 1, 2], reference="a", replicates=100,
                                 strata_col="outer_fold")
