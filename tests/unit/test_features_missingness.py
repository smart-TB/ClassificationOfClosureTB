import pandas as pd

from tb_outcomes.features import (
    early_death_missingness_audit,
    missingness_indicators,
    missingness_report,
)


def _df():
    return pd.DataFrame(
        {
            "a": pd.array([1, None, 3, None], dtype="Int64"),
            "b": pd.array([1, 2, 3, 4], dtype="Int64"),
            "SITUA_ENCE": pd.array([1, 3, 1, 3], dtype="Int64"),
        }
    )


def test_missingness_report_overall():
    r = missingness_report(_df(), by=None).set_index("column")
    assert r.loc["a", "n_missing"] == 2
    assert r.loc["a", "pct_missing"] == 50.0
    assert r.loc["b", "n_missing"] == 0


def test_missingness_report_by_outcome_exposes_differential():
    r = missingness_report(_df(), by=["SITUA_ENCE"])
    linha = r[(r.column == "a") & (r.SITUA_ENCE == 3)].iloc[0]
    assert linha.pct_missing == 100.0, "ausência concentrada no óbito"


def test_missingness_indicators_are_added_not_substituted():
    X = pd.DataFrame({"a": pd.array([1, None], dtype="Int64")})
    r = missingness_indicators(X)
    assert list(r.columns) == ["a", "a_is_missing"]
    assert r.a_is_missing.tolist() == [0, 1]
    assert pd.isna(r.a.iloc[1]), "o valor original continua ausente"


def test_early_death_audit_flags_variables_missing_more_in_deaths():
    a = early_death_missingness_audit(_df(), outcome_col="SITUA_ENCE").set_index("column")
    # 'a' está 100% ausente nos óbitos e 0% nas curas: diferença máxima.
    assert a.loc["a", "pct_missing_death"] == 100.0
    assert a.loc["a", "pct_missing_cure"] == 0.0
    assert a.loc["a", "abs_difference"] == 100.0
