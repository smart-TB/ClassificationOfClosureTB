import pandas as pd
import pytest

from tb_outcomes.context_features import assert_pre_study_census, freeze_census_context


def test_new_census_cannot_enter_primary():
    df = pd.DataFrame({"IPEA_POPULACAO_TOTAL": [100, 200],
                       "IPEA_POPULACAO_TOTAL_ANO_CENSO": [2010, 2022]})
    with pytest.raises(ValueError, match="not frozen"):
        assert_pre_study_census(df, ["IPEA_POPULACAO_TOTAL"])


def test_freezing_preserves_rows_and_missing_municipalities():
    df = pd.DataFrame({"ID_MUNIC_ANALISE": [1, 2, 1], "SITUA_ENCE": [1, 3, 2],
                       "IPEA_POPULACAO_TOTAL": [300, 400, 500]}, index=[8, 3, 19])
    context = pd.DataFrame({"ID_MUNICIP": [1], "IPEA_POPULACAO_TOTAL": [100],
                            "IPEA_POPULACAO_TOTAL_ANO_CENSO": [2010]})
    r = freeze_census_context(df, context)
    assert r.index.equals(df.index)
    assert r.SITUA_ENCE.equals(df.SITUA_ENCE)
    assert r.IPEA_POPULACAO_TOTAL.iloc[[0, 2]].tolist() == [100, 100]
    assert pd.isna(r.IPEA_POPULACAO_TOTAL.iloc[1])
    assert_pre_study_census(r, ["IPEA_POPULACAO_TOTAL"])


def test_missing_vintage_does_not_imply_baseline_availability():
    with pytest.raises(ValueError, match="Missing census vintage"):
        assert_pre_study_census(pd.DataFrame({"IPEA_IDHM": [0.5]}), ["IPEA_IDHM"])
