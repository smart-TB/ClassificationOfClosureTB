import pandas as pd

from tb_outcomes.features import closure_timing_audit


def _df():
    return pd.DataFrame(
        {
            "DT_NOTIFIC": ["2019-01-01"] * 6,
            # óbito por TB fecha cedo; cura fecha tarde
            "DT_ENCERRA": [
                "2019-01-10",
                "2019-01-15",
                "2019-01-20",
                "2019-08-01",
                "2019-09-01",
                "2019-10-01",
            ],
            "SITUA_ENCE": pd.array([3, 3, 3, 1, 1, 1], dtype="Int64"),
        }
    )


def test_audit_reports_median_and_share_closed_early():
    a = closure_timing_audit(_df()).set_index("SITUA_ENCE")
    assert a.loc[3, "pct_closed_lt_30d"] == 100.0
    assert a.loc[1, "pct_closed_lt_30d"] == 0.0
    # Notificação em 01/01; encerramentos em 10, 15 e 20/01 => 9, 14 e 19 dias.
    assert a.loc[3, "median_days"] == 14.0


def test_audit_ignores_impossible_durations():
    df = _df()
    df.loc[0, "DT_ENCERRA"] = "1899-12-30"  # encerramento antes da notificação
    a = closure_timing_audit(df).set_index("SITUA_ENCE")
    assert a.loc[3, "n"] == 2, "duração negativa não entra na estatística"
