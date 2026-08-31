import pandas as pd

from tests.fixtures.synthetic import make_synthetic_raw


def test_fixture_is_deterministic():
    a = make_synthetic_raw(n=300, seed=7).sinan
    b = make_synthetic_raw(n=300, seed=7).sinan
    pd.testing.assert_frame_equal(a, b)


def test_fixture_covers_both_encoding_eras():
    s = make_synthetic_raw(n=800, seed=1).sinan
    ano = s.DT_DIAG.str[:4].astype(int)
    antigos = s.loc[ano <= 2017, "SITUA_ENCE"]
    novos = s.loc[ano >= 2018, "SITUA_ENCE"]
    assert (antigos == "0").any(), "era antiga precisa de '0' literal"
    assert (novos == "").any(), "era nova precisa de campo vazio"
    assert not (novos == "0").any(), "era nova não usa '0'"


def test_fixture_covers_code_6_absent_from_real_data():
    s = make_synthetic_raw(n=800, seed=1).sinan
    assert (
        s.SITUA_ENCE == "6"
    ).any(), "código 6 é válido no dicionário e precisa ser exercitado"


def test_fixture_has_residence_notification_divergence():
    s = make_synthetic_raw(n=800, seed=1).sinan
    assert (s.ID_MN_RESI != s.ID_MUNICIP).any()


def test_fixture_has_impossible_dates_and_duplicates():
    s = make_synthetic_raw(n=800, seed=1).sinan
    assert (s.DT_ENCERRA == "1899-12-30").any()
    assert s.duplicated().any()


def test_fixture_has_municipality_with_only_2022_census():
    ipea = make_synthetic_raw(n=800, seed=1).ipea
    por_munic = ipea.dropna(subset=["IPEA_IDHM"]).groupby("ID_MUNICIP").ANO_DIAG.min()
    assert (por_munic >= 2022).any(), "precisa de município criado após 2010"
