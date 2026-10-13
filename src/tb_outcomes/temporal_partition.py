"""Partição temporal explícita: passado, ano de avaliação e anos não utilizados."""
from __future__ import annotations

import pandas as pd


def population_mask(years, test_year: int, role: str):
    """Apply before sampling, model comparisons and constructing spatial folds."""
    year = pd.to_numeric(pd.Series(years), errors="coerce")
    if year.isna().any() or ((year % 1) != 0).any():
        raise ValueError("NU_ANO ausente ou inválido: impossível particionar no tempo")
    if role == "development":
        return (year < test_year).to_numpy(dtype=bool)
    if role == "temporal":
        return (year <= test_year).to_numpy(dtype=bool)
    raise ValueError(f"papel de população desconhecido: {role}")


def temporal_masks(years, test_year: int):
    year = pd.to_numeric(pd.Series(years), errors="coerce")
    if year.isna().any() or ((year % 1) != 0).any():
        raise ValueError("NU_ANO ausente ou inválido: impossível particionar no tempo")
    train = (year < test_year).to_numpy(dtype=bool)
    test = (year == test_year).to_numpy(dtype=bool)
    if not train.any() or not test.any():
        raise ValueError(f"partição temporal degenerada para o ano {test_year}")
    return train, test
