import pandas as pd
import pytest

from tb_outcomes.data import COLUNA_MUNICIPIO_RESOLVIDO, attach_cnes, attach_ibge_ipea


def _base():
    return pd.DataFrame(
        {
            COLUNA_MUNICIPIO_RESOLVIDO: ["3500105", "3500105", "3500105"],
            "DT_DIAG": ["2015-01-15", "2015-02-10", "2015-03-05"],
            "ANO_DIAG": ["2015", "2015", "2015"],
        }
    )


def test_cnes_ffill_never_uses_future_value():
    # Sem dado em janeiro; fevereiro tem. Janeiro deve permanecer AUSENTE:
    # copiar fevereiro para janeiro é usar informação que não existia.
    prof = pd.DataFrame(
        {"ID_MUNICIP": ["350010"], "data": ["2015-02"], "QT_PROFISSIONAIS_TOTAL": [500.0]}
    )
    estab = pd.DataFrame(
        {"ID_MUNICIP": ["350010"], "data": ["2015-02"], "QT_ESTABELECIMENTOS_TOTAL": [20.0]}
    )
    r = attach_cnes(_base(), prof, estab).sort_values("DT_DIAG").reset_index(drop=True)
    assert pd.isna(r.QT_PROFISSIONAIS_TOTAL.iloc[0]), "janeiro não pode receber valor de fevereiro"
    assert r.QT_PROFISSIONAIS_TOTAL.iloc[1] == 500.0
    assert r.QT_PROFISSIONAIS_TOTAL.iloc[2] == 500.0, "março herda fevereiro por ffill"


def test_ibge_ipea_merge_rejects_duplicated_context():
    ibge = pd.DataFrame(
        {
            "ID_MUNICIP": [3500105, 3500105],
            "ANO_DIAG": [2015, 2015],
            "IBGE_PIB_PER_CAPITA": ["100.0", "200.0"],
        }
    )
    ipea = pd.DataFrame({"ID_MUNICIP": [3500105], "ANO_DIAG": [2015], "IPEA_IDHM": [0.8]})
    with pytest.raises(pd.errors.MergeError):
        attach_ibge_ipea(_base(), ibge, ipea)


def test_context_columns_become_numeric_and_sentinel_becomes_na():
    ibge = pd.DataFrame(
        {
            "ID_MUNICIP": [3500105],
            "ANO_DIAG": [2015],
            "IBGE_PIB_PER_CAPITA": ["13670.26"],
            "IBGE_INTERNACOES_POR_DIARREIA_SUS": ["-"],
            "IBGE_BIOMA": ["Mata Atlântica"],
        }
    )
    ipea = pd.DataFrame({"ID_MUNICIP": [3500105], "ANO_DIAG": [2015], "IPEA_IDHM": [0.805]})
    r = attach_ibge_ipea(_base(), ibge, ipea)
    assert str(r.IBGE_PIB_PER_CAPITA.dtype).startswith(("float", "Float"))
    assert r.IBGE_PIB_PER_CAPITA.iloc[0] == 13670.26
    assert r.IBGE_INTERNACOES_POR_DIARREIA_SUS.isna().all(), "'-' é sentinela de ausência"
    assert r.IBGE_BIOMA.dtype == object, "categórica não vira número"


def test_ibge_ipea_merge_preserves_row_count():
    ibge = pd.DataFrame(
        {"ID_MUNICIP": [3500105], "ANO_DIAG": [2015], "IBGE_PIB_PER_CAPITA": ["1.0"]}
    )
    ipea = pd.DataFrame({"ID_MUNICIP": [3500105], "ANO_DIAG": [2015], "IPEA_IDHM": [0.8]})
    assert len(attach_ibge_ipea(_base(), ibge, ipea)) == 3
