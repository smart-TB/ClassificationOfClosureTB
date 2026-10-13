import pandas as pd
import pytest

from tb_outcomes.maturation import assess_maturation, maturation_profile


@pytest.mark.parametrize("notified,extracted,form,site,expected", [
    ("2025-10-15", "2026-07-15", 1, None, True),
    ("2025-10-16", "2026-07-15", 1, None, False),
    ("2025-05-31", "2026-02-28", 1, None, True),
    ("2023-05-31", "2024-02-28", 1, None, False),
    ("2023-05-31", "2024-02-29", 1, None, True),
    ("2025-04-15", "2026-07-15", 2, 7, True),
    ("2025-04-16", "2026-07-15", 2, 4, False),
    (None, "2026-07-15", 1, None, False),
    ("2026-07-16", "2026-07-15", 1, None, False),
])
def test_calendar_boundaries(notified, extracted, form, site, expected):
    df = pd.DataFrame({"DT_NOTIFIC": [notified], "FORMA": [form], "EXTRAPU1_N": [site]})
    assert bool(assess_maturation(df, extracted).mature.iloc[0]) is expected


def test_uncertain_and_secondary_sites_use_conservative_window():
    df = pd.DataFrame({"FORMA": [2, 3, 2, None, 1, 2, 2],
                       "EXTRAPU1_N": [1, 1, 10, None, 2, "2/", 2],
                       "EXTRAPU2_N": [None, 7, 0, None, None, 0, 99]})
    assert maturation_profile(df).tolist() == [
        "standard_9m", "prolonged_15m", "uncertain_15m", "uncertain_15m",
        "uncertain_15m", "uncertain_15m", "uncertain_15m"]


def test_rule_does_not_use_observed_outcome_or_treatment_start():
    df = pd.DataFrame({"DT_NOTIFIC": ["2025-10-15"] * 3, "FORMA": 1,
                       "SITUA_ENCE": [1, 3, None], "DT_ENCERRA": [None, "2025-10-16", None],
                       "DT_INIC_TR": [None, "2026-02-01", "2025-10-15"]})
    assert assess_maturation(df, "2026-07-15").mature.all()


def test_days_sensitivity_preserves_extended_window_and_index():
    df = pd.DataFrame({"DT_NOTIFIC": ["2025-05-01"] * 2,
                       "FORMA": [1, 2], "EXTRAPU1_N": [None, 7]}, index=[42, 10])
    result = assess_maturation(df, "2026-07-15", rule="days_by_form")
    assert result.index.tolist() == [42, 10]
    assert result.mature.tolist() == [True, False]
    assert assess_maturation(df, "2026-07-15", rule="legacy_days").mature.all()


def test_typo_cannot_silently_change_rule():
    with pytest.raises(ValueError, match="Unknown maturation"):
        assess_maturation(pd.DataFrame(), "2026-07-15", rule="calendar")
