import pandas as pd
import pytest

from tb_outcomes.temporal_partition import population_mask


def test_only_past_is_allowed_in_spatial_development():
    assert population_mask([2015, 2023, 2024, 2025], 2024, "development").tolist() == [
        True, True, False, False]


def test_future_is_out_of_temporal_population_and_indices_are_positional():
    years = pd.Series([2025, 2023, 2024], index=[18, 4, 72])
    assert population_mask(years, 2024, "temporal").tolist() == [False, True, True]


@pytest.mark.parametrize("years", [[2023, None], [2023, 2024.5], ["bad", 2024]])
def test_invalid_year_fails_before_assignment(years):
    with pytest.raises(ValueError, match="NU_ANO"):
        population_mask(years, 2024, "development")


def test_unknown_population_cannot_bypass_reservation():
    with pytest.raises(ValueError, match="desconhecido"):
        population_mask([2023, 2024], 2024, "all")
