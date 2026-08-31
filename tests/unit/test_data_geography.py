import pandas as pd

from tb_outcomes.data import (
    COLUNA_MUNICIPIO_RESOLVIDO,
    attach_coordinates,
    attach_names,
    municipality_audit,
    resolve_municipality,
)

MUNICIPIOS = pd.DataFrame(
    [
        {"ID_MUNICIP": "3500105", "NOME_MUNIC": "Gama", "SIGLA_UF": "SP", "REGIAO": "Sudeste"},
        {"ID_MUNICIP": "2300100", "NOME_MUNIC": "Beta", "SIGLA_UF": "CE", "REGIAO": "Nordeste"},
    ]
)
LATLONG = pd.DataFrame(
    {
        "GEOCODIGO_MUNICIPIO": [3500105, 2300100],
        "LATITUDE": [-23.55, -3.71],
        "LONGITUDE": [-46.63, -38.54],
    }
)


def test_resolve_prefers_residence():
    df = pd.DataFrame({"ID_MN_RESI": ["3500105"], "ID_MUNICIP": ["2300100"]})
    r = resolve_municipality(df, "ID_MN_RESI", "ID_MUNICIP", MUNICIPIOS)
    assert r[COLUNA_MUNICIPIO_RESOLVIDO].iloc[0] == "3500105"


def test_resolve_falls_back_when_residence_invalid():
    df = pd.DataFrame({"ID_MN_RESI": ["9999999"], "ID_MUNICIP": ["2300100"]})
    r = resolve_municipality(df, "ID_MN_RESI", "ID_MUNICIP", MUNICIPIOS)
    assert r[COLUNA_MUNICIPIO_RESOLVIDO].iloc[0] == "2300100"


def test_resolve_is_na_when_both_invalid():
    df = pd.DataFrame({"ID_MN_RESI": ["9999999"], "ID_MUNICIP": ["8888888"]})
    r = resolve_municipality(df, "ID_MN_RESI", "ID_MUNICIP", MUNICIPIOS)
    assert pd.isna(r[COLUNA_MUNICIPIO_RESOLVIDO].iloc[0])


def test_names_and_coordinates_follow_the_resolved_key():
    # O defeito estrutural corrigido: nome/coordenada seguiam a residência
    # enquanto o contexto seguia a notificação.
    df = pd.DataFrame({"ID_MN_RESI": ["3500105"], "ID_MUNICIP": ["2300100"]})
    r = resolve_municipality(df, "ID_MN_RESI", "ID_MUNICIP", MUNICIPIOS)
    r = attach_coordinates(attach_names(r, MUNICIPIOS), LATLONG)
    assert r.NOME_MUNIC.iloc[0] == "Gama"
    assert r.SIGLA_UF.iloc[0] == "SP"
    assert r.LAT_MUNIC.iloc[0] == -23.55


def test_municipality_audit_counts_divergence():
    df = pd.DataFrame(
        {
            "ID_MN_RESI": ["3500105", "2300100"],
            "ID_MUNICIP": ["2300100", "2300100"],
        }
    )
    r = resolve_municipality(df, "ID_MN_RESI", "ID_MUNICIP", MUNICIPIOS)
    aud = municipality_audit(r, "ID_MN_RESI").set_index("metrica").valor
    assert int(aud["registros_total"]) == 2
    assert int(aud["registros_residencia_diferente_notificacao"]) == 1
