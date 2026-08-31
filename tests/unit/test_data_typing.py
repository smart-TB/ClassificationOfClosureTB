import pandas as pd

from tb_outcomes.data import (
    coerce_numeric_column,
    is_all_numeric,
    normalize_missing,
    treat_sinan,
)


def test_normalize_missing_maps_all_sentinels_to_na():
    s = pd.Series(["1", " 2 ", "", "nan", "None", "<NA>", "-", "3"])
    r = normalize_missing(s)
    assert list(r.isna()) == [False, False, True, True, True, True, True, False]
    assert r.iloc[1] == "2", "strip precisa ser aplicado"


def test_is_all_numeric_rejects_free_text_column():
    # AGRAVOUTDE: texto livre com um número solto. A lógica antiga do bot
    # respondia 'sim' e destruía 57.436 descrições, virando zeros.
    s = normalize_missing(pd.Series(["HIPERTENSAO", "LEUCEMIA", "286.4", ""]))
    assert is_all_numeric(s) is False


def test_is_all_numeric_accepts_pure_numeric_column():
    assert is_all_numeric(normalize_missing(pd.Series(["1", "2", ""]))) is True


def test_coerce_numeric_uses_int64_and_preserves_na():
    r = coerce_numeric_column(normalize_missing(pd.Series(["1", "0", "", "10"])))
    assert r.dtype == "Int64"
    assert r.isna().sum() == 1
    assert r.iloc[1] == 0, "código 0 real não pode virar ausente"


def test_coerce_numeric_uses_float_when_decimals_present():
    assert coerce_numeric_column(normalize_missing(pd.Series(["1.5", "2"]))).dtype == "float64"


def test_treat_sinan_separates_empty_from_literal_zero():
    df = pd.DataFrame({"SITUA_ENCE": ["", "0", "1", "03", "04"], "DT_DIAG": ["2016-01-01"] * 5})
    r = treat_sinan(df)
    assert r.SITUA_ENCE.dtype == "Int64"
    assert r.SITUA_ENCE.isna().sum() == 1  # o vazio
    assert (r.SITUA_ENCE == 0).sum() == 1  # o '0' literal
    assert (r.SITUA_ENCE == 3).sum() == 1  # '03' zero-padded
    assert (r.SITUA_ENCE == 4).sum() == 1


def test_treat_sinan_preserves_free_text():
    df = pd.DataFrame({"AGRAVOUTDE": ["HIPERTENSAO", "286.4", ""], "DT_DIAG": ["2016-01-01"] * 3})
    r = treat_sinan(df)
    assert r.AGRAVOUTDE.dtype == "string"
    assert r.AGRAVOUTDE.iloc[0] == "HIPERTENSAO"


def test_treat_sinan_creates_ano_diag_and_no_numerocasos():
    df = pd.DataFrame({"DT_DIAG": ["2019-05-02"], "SITUA_ENCE": ["1"]})
    r = treat_sinan(df)
    assert r.ANO_DIAG.iloc[0] == "2019"
    assert "NumeroCasos" not in r.columns


def test_treat_sinan_ano_nasc_is_nullable_int_without_zeros():
    df = pd.DataFrame({"ANO_NASC": ["1980", ""], "DT_DIAG": ["2019-01-01"] * 2})
    r = treat_sinan(df)
    assert r.ANO_NASC.dtype == "Int64"
    assert r.ANO_NASC.isna().sum() == 1
    assert (r.ANO_NASC == 0).sum() == 0
