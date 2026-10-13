"""Municipal context available before cohort entry versus retrospective context."""
from __future__ import annotations

import pandas as pd


def freeze_census_context(df, context):
    if context.ID_MUNICIP.duplicated().any():
        raise ValueError("Duplicate municipality in baseline context")
    result = df.copy()
    lookup = context.copy()
    lookup["ID_MUNICIP"] = pd.to_numeric(lookup.ID_MUNICIP, errors="raise")
    lookup = lookup.set_index("ID_MUNICIP")
    keys = pd.to_numeric(df.ID_MUNIC_ANALISE, errors="coerce")
    for col in lookup.columns:
        if col.startswith("IPEA_"):
            result[col] = keys.map(lookup[col])
    assert result.index.equals(df.index) and len(result) == len(df)
    return result


def assert_pre_study_census(df, columns):
    """Do not silently use 2022 census values in a model advertised as baseline."""
    for col in columns:
        if not col.startswith("IPEA_") or col not in df or df[col].notna().sum() == 0:
            continue
        vintage = f"{col}_ANO_CENSO"
        if vintage not in df:
            raise ValueError(f"Missing census vintage for {col}")
        year = pd.to_numeric(df[vintage], errors="coerce")
        if (df[col].notna() & (year.isna() | ~year.isin([2000, 2010]))).any():
            raise ValueError(f"{col} contains context not frozen before the study")
