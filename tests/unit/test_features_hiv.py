import pandas as pd

from tb_outcomes.features import (
    NIVEL_NAO_DISPONIVEL,
    collapse_pending_levels,
    derive_hiv_rule,
    hiv_availability_by_duration,
    hiv_rule_audit,
    outcome_contamination_audit,
)


def test_collapse_makes_pending_and_not_performed_indistinguishable():
    # O SINAN reescreve 3 -> 4 conforme o desfecho. Colapsar torna a feature
    # invariante a essa reescrita.
    df = pd.DataFrame({"HIV": pd.array([1, 2, 3, 4, None], dtype="Int64")})
    r = collapse_pending_levels(df, ["HIV"])
    assert r.HIV.iloc[2] == NIVEL_NAO_DISPONIVEL
    assert r.HIV.iloc[3] == NIVEL_NAO_DISPONIVEL
    assert r.HIV.iloc[0] == 1 and r.HIV.iloc[1] == 2
    assert pd.isna(r.HIV.iloc[4]), "ausência continua ausência"


def test_collapse_is_idempotent():
    df = pd.DataFrame({"HIV": pd.array([3, 4], dtype="Int64")})
    a = collapse_pending_levels(df, ["HIV"])
    b = collapse_pending_levels(a, ["HIV"])
    pd.testing.assert_frame_equal(a, b)


def test_hiv_rule_positive_by_serology_or_aids():
    df = pd.DataFrame(
        {
            "HIV": pd.array([1, 2, 2, 3, 4, None], dtype="Int64"),
            "AGRAVAIDS": pd.array([2, 1, 2, 2, 2, 2], dtype="Int64"),
        }
    )
    r = derive_hiv_rule(df)
    assert r.iloc[0] == 1, "sorologia positiva"
    assert r.iloc[1] == 1, "AIDS registrada"
    assert r.iloc[2] == 0, "sorologia negativa sem AIDS"
    assert pd.isna(r.iloc[3]), "em andamento não é negativo"
    assert pd.isna(r.iloc[4]), "não realizado não é negativo"
    assert pd.isna(r.iloc[5])


def test_hiv_rule_preserves_original_columns():
    df = pd.DataFrame(
        {"HIV": pd.array([2], dtype="Int64"), "AGRAVAIDS": pd.array([1], dtype="Int64")}
    )
    antes = df.copy()
    derive_hiv_rule(df)
    pd.testing.assert_frame_equal(df, antes)


def test_hiv_rule_audit_counts_changed_records():
    df = pd.DataFrame(
        {
            "HIV": pd.array([1, 2, 2], dtype="Int64"),
            "AGRAVAIDS": pd.array([2, 1, 2], dtype="Int64"),
        }
    )
    a = hiv_rule_audit(df).set_index("metrica").valor
    assert int(a["positivos_sorologia_original"]) == 1
    assert int(a["positivos_apos_regra"]) == 2
    assert int(a["registros_alterados_pela_regra"]) == 1


def test_contamination_audit_reports_rate_by_outcome():
    df = pd.DataFrame(
        {
            "HIV": pd.array([3, 3, 1, 3], dtype="Int64"),
            "SITUA_ENCE": pd.array([1, 5, 1, 5], dtype="Int64"),
        }
    )
    a = outcome_contamination_audit(df, ["HIV"])
    linha = a[(a.column == "HIV") & (a.SITUA_ENCE == 5)].iloc[0]
    assert linha.pct_pending == 100.0
    assert bool(linha.affected_by_rule) is False
    linha1 = a[(a.column == "HIV") & (a.SITUA_ENCE == 1)].iloc[0]
    assert linha1.pct_pending == 50.0
    assert bool(linha1.affected_by_rule) is True


def test_hiv_availability_by_duration_separates_baseline_from_artifact():
    # Discriminador da spec §3.1: se a indisponibilidade fosse só artefato de
    # registro, ela sumiria a tempo igual. Aqui, com 0-7 dias para os dois, o
    # óbito tem um terço da disponibilidade da cura — isso é linha de base.
    df = pd.DataFrame(
        {
            "DT_NOTIFIC": ["2019-01-01"] * 8,
            "DT_ENCERRA": ["2019-01-05"] * 8,
            "SITUA_ENCE": pd.array([3, 3, 3, 3, 1, 1, 1, 1], dtype="Int64"),
            "HIV": pd.array([1, 4, 4, 4, 1, 1, 2, 4], dtype="Int64"),
        }
    )
    r = hiv_availability_by_duration(df)
    obito = r[(r.SITUA_ENCE == 3) & (r.duration_bin == "0-7")].iloc[0]
    cura = r[(r.SITUA_ENCE == 1) & (r.duration_bin == "0-7")].iloc[0]
    assert obito.pct_available == 25.0
    assert cura.pct_available == 75.0
