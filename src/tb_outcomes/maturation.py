"""Outcome-independent administrative maturation, measured from notification.

Clinical form is a proxy for the expected regimen, not observed treatment length.
The 15-month rule for osteoarticular/uncertain forms is a conservative protocol
extrapolation; the guide explicitly states 15 months for meningoencephalic TB.
"""
from __future__ import annotations

import pandas as pd


def maturation_profile(df: pd.DataFrame) -> pd.Series:
    def numeric(name):
        return pd.to_numeric(df.get(name, pd.Series(index=df.index, dtype=float)), errors="coerce")

    form = numeric("FORMA")
    # The harmonized historical extract has two sites; the newer dictionary names
    # its primary site EXTRAPU_N. Do not infer codes from malformed text like '2/'.
    first = numeric("EXTRAPU1_N" if "EXTRAPU1_N" in df else "EXTRAPU_N")
    second = numeric("EXTRAPU2_N")
    prolonged = first.isin([4, 7]) | second.isin([4, 7])
    ordinary = [1, 2, 3, 5, 6, 8, 9]
    second_optional = second.isna() | second.eq(0) | second.isin(ordinary)
    pulmonary = form.eq(1) & (first.isna() | first.eq(0)) & (second.isna() | second.eq(0))
    extrapulmonary = form.isin([2, 3]) & first.isin(ordinary) & second_optional
    group = pd.Series("uncertain_15m", index=df.index, name="maturation_group")
    group.loc[pulmonary | extrapulmonary] = "standard_9m"
    group.loc[prolonged] = "prolonged_15m"
    return group


def assess_maturation(df: pd.DataFrame, extraction_date: str, *,
                      rule: str = "calendar_by_form", minimum_days: int = 365) -> pd.DataFrame:
    """No outcome, closure date or treatment start is used to decide eligibility.

    'days_by_form' is the planned sensitivity: standard cases need minimum_days,
    with prolonged/uncertain forms retaining 15 calendar months.
    'legacy_days' reproduces the historical uniform-day rule for comparison only.
    """
    if rule not in {"calendar_by_form", "days_by_form", "legacy_days"}:
        raise ValueError(f"Unknown maturation rule: {rule}")
    extraction = pd.Timestamp(extraction_date)
    if pd.isna(extraction):
        raise ValueError("Extraction date is required")
    notified = pd.to_datetime(df["DT_NOTIFIC"], errors="coerce")
    group = maturation_profile(df)
    deadline = notified + pd.Timedelta(days=minimum_days)
    if rule != "legacy_days":
        extended = group.ne("standard_9m")
        deadline.loc[extended] = notified.loc[extended] + pd.DateOffset(months=15)
        if rule == "calendar_by_form":
            deadline.loc[~extended] = notified.loc[~extended] + pd.DateOffset(months=9)
    return pd.DataFrame({
        "maturation_group": group,
        "deadline": deadline,
        "potential_followup_days": (extraction - notified).dt.days,
        "mature": notified.notna() & notified.le(extraction) & deadline.le(extraction),
    }, index=df.index)
