from pathlib import Path

import pandas as pd

from tb_outcomes.cohort import TargetSchema, build_cohort, followup_audit, load_outcome_rules

RULES = load_outcome_rules(Path("configs/outcomes.yaml"))


class _Study:
    cohort_year_min = 2015
    cohort_year_max = 2025
    minimum_followup_days = 365
    data_extraction_date = "2026-04-07"


class _Cfg:
    study = _Study()


def _df():
    return pd.DataFrame(
        {
            "DT_NOTIFIC": ["2018-01-01", "2019-01-01", "2020-01-01", "2014-01-01", "2025-12-01"],
            "NU_ANO": pd.array([2018, 2019, 2020, 2014, 2025], dtype="Int64"),
            "ID_MUNIC_ANALISE": pd.array([3500105] * 5, dtype="Int64"),
            "ANO_NASC": pd.array([1980] * 5, dtype="Int64"),
            "SITUA_ENCE": pd.array([1, 2, 5, 1, 1], dtype="Int64"),
        }
    )


def test_flow_reports_three_named_populations():
    r = build_cohort(_df(), _Cfg(), RULES, TargetSchema.LEGACY_3CLASS)
    etapas = r.flow.set_index("stage").n
    assert int(etapas["recuperada"]) == 5
    assert int(etapas["declarada"]) == 4  # 2014 fora da janela
    assert int(etapas["elegivel"]) == 3  # 2025-12 sem seguimento
    assert int(etapas["analitica"]) == 2  # código 5 fora do legacy_3class


def test_flow_counts_sum_to_cohort():
    r = build_cohort(_df(), _Cfg(), RULES, TargetSchema.LEGACY_3CLASS)
    assert int(r.outcome_distribution.n.sum()) == len(r.analytic)


def test_analytic_never_contains_unresolved():
    r = build_cohort(_df(), _Cfg(), RULES, TargetSchema.LEGACY_3CLASS)
    assert "unresolved_or_censored" not in set(r.analytic["target_legacy_3class"])


def test_followup_audit_reports_unresolved_rate_by_year():
    aud = followup_audit(_df(), _Cfg())
    assert {"NU_ANO", "n", "pct_followup_ge_min"} <= set(aud.columns)
