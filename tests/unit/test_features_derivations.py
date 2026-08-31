import pandas as pd

from tb_outcomes.features import (
    apply_all_derivations,
    derive_age,
    derive_interactions,
    normalize_agravos,
    normalize_pop_especiais,
)


def test_derive_age_from_year_of_notification_and_birth():
    df = pd.DataFrame(
        {
            "NU_ANO": pd.array([2019, 2020], dtype="Int64"),
            "ANO_NASC": pd.array([1980, None], dtype="Int64"),
        }
    )
    r = derive_age(df)
    assert r.iloc[0] == 39
    assert pd.isna(r.iloc[1]), "sem ano de nascimento, idade é ausente — nunca 0"


def test_derive_age_rejects_impossible_values():
    df = pd.DataFrame(
        {
            "NU_ANO": pd.array([2019, 2019], dtype="Int64"),
            "ANO_NASC": pd.array([2030, 1850], dtype="Int64"),
        }
    )
    r = derive_age(df)
    assert r.isna().all(), "idade negativa ou > 120 é ausente, não valor"


def test_agravos_become_binary_preserving_ignored_as_missing():
    # Códigos SINAN: 1 Sim, 2 Não, 9 Ignorado. 'Ignorado' NÃO é 'Não'.
    df = pd.DataFrame({"AGRAVAIDS": pd.array([1, 2, 9, None], dtype="Int64")})
    r = normalize_agravos(df, ["AGRAVAIDS"])
    assert r.AGRAVAIDS_BIN.iloc[0] == 1
    assert r.AGRAVAIDS_BIN.iloc[1] == 0
    assert pd.isna(r.AGRAVAIDS_BIN.iloc[2]), "'Ignorado' vira ausente, não 0"
    assert pd.isna(r.AGRAVAIDS_BIN.iloc[3])


def test_pop_especiais_keeps_the_four_populations_separate():
    # O original usava keep_only_combined=True e descartava as originais, o que
    # inviabiliza a equidade do briefing §20.1 (situação de rua e privação de
    # liberdade precisam ser analisadas separadamente).
    df = pd.DataFrame(
        {
            "POP_LIBER": pd.array([1, 2], dtype="Int64"),
            "POP_RUA": pd.array([2, 1], dtype="Int64"),
            "POP_SAUDE": pd.array([2, 2], dtype="Int64"),
            "POP_IMIG": pd.array([2, 2], dtype="Int64"),
        }
    )
    r = normalize_pop_especiais(df)
    for col in ["POP_LIBER_BIN", "POP_RUA_BIN", "POP_SAUDE_BIN", "POP_IMIG_BIN"]:
        assert col in r.columns, f"{col} precisa sobreviver separada"
    assert r.POP_ESPECIAIS_BIN.tolist() == [1, 1], "combinado é adicional, não substituto"


def test_pop_especiais_combined_is_na_only_when_all_are_missing():
    df = pd.DataFrame(
        {
            "POP_LIBER": pd.array([9, 2], dtype="Int64"),
            "POP_RUA": pd.array([9, 2], dtype="Int64"),
            "POP_SAUDE": pd.array([9, 2], dtype="Int64"),
            "POP_IMIG": pd.array([9, 2], dtype="Int64"),
        }
    )
    r = normalize_pop_especiais(df)
    assert pd.isna(r.POP_ESPECIAIS_BIN.iloc[0]), "todas ignoradas => ausente"
    assert r.POP_ESPECIAIS_BIN.iloc[1] == 0, "todas 'não' => 0"


def test_interactions_use_derived_binaries():
    df = pd.DataFrame(
        {
            "IDADE": pd.array([40, 50], dtype="Int64"),
            "AGRAVDIABE_BIN": pd.array([1, 0], dtype="Int64"),
            "AGRAVAIDS_BIN": pd.array([0, 1], dtype="Int64"),
        }
    )
    r = derive_interactions(df)
    assert r.IDADE_X_DIABETES.tolist() == [40, 0]
    assert r.IDADE_X_AIDS.tolist() == [0, 50]


def test_no_derivation_touches_the_outcome():
    # Terceira barreira contra o defeito original: nenhuma derivação pode ler
    # DT_ENCERRA nem SITUA_ENCE, nem sequer se elas estiverem presentes.
    df = pd.DataFrame(
        {
            "NU_ANO": pd.array([2019], dtype="Int64"),
            "ANO_NASC": pd.array([1980], dtype="Int64"),
            "DT_ENCERRA": ["2019-06-01"],
            "SITUA_ENCE": pd.array([1], dtype="Int64"),
        }
    )
    r = apply_all_derivations(df)
    novas = set(r.columns) - set(df.columns)
    assert "TEMPO_TRATAMENTO" not in novas
    assert "FLAG_ENCERRA" not in novas
    assert not any("ENCERRA" in c.upper() for c in novas)
