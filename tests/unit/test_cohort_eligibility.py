import pandas as pd

from tb_outcomes.cohort import apply_eligibility, audit_duplicates, compute_followup


class _Study:
    cohort_year_min = 2015
    cohort_year_max = 2025
    minimum_followup_days = 365
    data_extraction_date = "2026-04-07"


class _Cfg:
    study = _Study()


def test_compute_followup_counts_days_to_extraction():
    df = pd.DataFrame({"DT_NOTIFIC": ["2025-04-07", "2024-04-07"]})
    r = compute_followup(df, "2026-04-07")
    assert r.iloc[0] == 365
    assert r.iloc[1] == 730


def test_eligibility_excludes_outside_declared_window():
    df = pd.DataFrame(
        {
            "DT_NOTIFIC": ["2014-06-01", "2016-06-01"],
            "NU_ANO": pd.array([2014, 2016], dtype="Int64"),
            "ID_MUNIC_ANALISE": pd.array([3500105, 3500105], dtype="Int64"),
            "ANO_NASC": pd.array([1980, 1980], dtype="Int64"),
        }
    )
    elegivel, log = apply_eligibility(df, _Cfg())
    assert len(elegivel) == 1
    assert int(log.set_index("reason").n["fora_da_janela_declarada"]) == 1


def test_eligibility_excludes_insufficient_followup():
    # 2025-12-01 tem ~127 dias até a extração: fora da analítica pela regra de 365.
    df = pd.DataFrame(
        {
            "DT_NOTIFIC": ["2025-12-01", "2020-01-01"],
            "NU_ANO": pd.array([2025, 2020], dtype="Int64"),
            "ID_MUNIC_ANALISE": pd.array([3500105, 3500105], dtype="Int64"),
            "ANO_NASC": pd.array([1990, 1990], dtype="Int64"),
        }
    )
    elegivel, log = apply_eligibility(df, _Cfg())
    assert len(elegivel) == 1
    assert int(log.set_index("reason").n["seguimento_insuficiente"]) == 1


def test_eligibility_excludes_missing_municipality():
    df = pd.DataFrame(
        {
            "DT_NOTIFIC": ["2018-01-01"],
            "NU_ANO": pd.array([2018], dtype="Int64"),
            "ID_MUNIC_ANALISE": pd.array([None], dtype="Int64"),
            "ANO_NASC": pd.array([1990], dtype="Int64"),
        }
    )
    elegivel, log = apply_eligibility(df, _Cfg())
    assert len(elegivel) == 0
    assert int(log.set_index("reason").n["municipio_irrecuperavel"]) == 1


def test_eligibility_never_excludes_for_missing_feature():
    # Briefing §9.1: ausência de feature NUNCA exclui.
    df = pd.DataFrame(
        {
            "DT_NOTIFIC": ["2018-01-01"],
            "NU_ANO": pd.array([2018], dtype="Int64"),
            "ID_MUNIC_ANALISE": pd.array([3500105], dtype="Int64"),
            "ANO_NASC": pd.array([1990], dtype="Int64"),
            "HIV": pd.array([None], dtype="Int64"),
            "CS_ESCOL_N": pd.array([None], dtype="Int64"),
        }
    )
    elegivel, _ = apply_eligibility(df, _Cfg())
    assert len(elegivel) == 1


def test_eligibility_log_has_sequential_and_non_sequential_counts():
    # Um registro pode violar duas regras: sequencial conta uma vez (na ordem),
    # não sequencial conta em ambas.
    df = pd.DataFrame(
        {
            "DT_NOTIFIC": ["2014-01-01"],
            "NU_ANO": pd.array([2014], dtype="Int64"),
            "ID_MUNIC_ANALISE": pd.array([None], dtype="Int64"),
            "ANO_NASC": pd.array([1990], dtype="Int64"),
        }
    )
    _, log = apply_eligibility(df, _Cfg())
    idx = log.set_index("reason")
    assert int(idx.n["municipio_irrecuperavel"]) == 1
    assert int(idx.n["fora_da_janela_declarada"]) == 0, "já removido pela regra anterior"
    assert int(idx.n_nao_sequencial["fora_da_janela_declarada"]) == 1


def test_audit_duplicates_counts_exact_groups():
    df = pd.DataFrame({"a": ["x", "x", "y"], "b": [1, 1, 2]})
    aud = audit_duplicates(df, ["a", "b"]).set_index("metrica").valor
    assert int(aud["linhas_em_grupos_duplicados"]) == 2
    assert int(aud["linhas_redundantes"]) == 1
