from pathlib import Path

import pandas as pd

from tb_outcomes.features import (
    build_notification_feature_set,
    feature_availability_table,
    feature_dictionary_table,
    load_feature_config,
)

SPECS = load_feature_config(Path("configs/features.yaml"))


def _cohort():
    return pd.DataFrame(
        {
            "CS_SEXO": ["M", "F"],
            "HIV": pd.array([1, 3], dtype="Int64"),
            "AGRAVAIDS": pd.array([2, 2], dtype="Int64"),
            "ANT_RETRO": pd.array([1, 2], dtype="Int64"),
            "SITUA_ENCE": pd.array([1, 2], dtype="Int64"),
            "DT_ENCERRA": ["2019-06-01", "2019-07-01"],
            "BACILOSC_1": pd.array([1, 2], dtype="Int64"),
            "NU_ANO": pd.array([2019, 2020], dtype="Int64"),
            "ANO_NASC": pd.array([1980, 1990], dtype="Int64"),
            "ID_MUNIC_ANALISE": pd.array([3500105, 3500105], dtype="Int64"),
            "ID_MN_RESI": ["3500105", "3500105"],
            "IPEA_IDHM": [0.8, 0.8],
        }
    )


def test_outcome_and_post_baseline_never_reach_the_feature_set():
    X = build_notification_feature_set(_cohort(), SPECS)
    for proibida in ["SITUA_ENCE", "DT_ENCERRA", "ANT_RETRO", "BACILOSC_1"]:
        assert proibida not in X.columns, f"{proibida} vazou para o X"


def test_no_column_of_the_feature_set_derives_from_closure_date():
    # Nenhuma feature do X pode derivar de DT_ENCERRA nem de SITUA_ENCE: uma
    # feature construída a partir do desfecho (p.ex. tempo até o encerramento) o
    # vaza para dentro do modelo.
    por_nome = {s.raw_name: s for s in SPECS}
    for s in SPECS:
        if not s.in_notification_set:
            continue
        assert "DT_ENCERRA" not in s.derived_from, f"{s.raw_name} deriva de DT_ENCERRA"
        assert "SITUA_ENCE" not in s.derived_from, f"{s.raw_name} deriva do alvo"
    for proibida in ["TEMPO_TRATAMENTO", "FLAG_ENCERRA"]:
        if proibida in por_nome:
            assert not por_nome[proibida].in_notification_set, f"{proibida} no X"


def test_strata_and_keys_never_reach_the_feature_set():
    X = build_notification_feature_set(_cohort(), SPECS)
    for chave in ["NU_ANO", "ID_MUNIC_ANALISE", "ID_MN_RESI"]:
        assert chave not in X.columns


def test_notification_and_contextual_variables_are_kept():
    X = build_notification_feature_set(_cohort(), SPECS)
    assert "CS_SEXO" in X.columns
    assert "IPEA_IDHM" in X.columns


def test_hiv_is_collapsed_and_derived_feature_is_added():
    X = build_notification_feature_set(_cohort(), SPECS)
    assert "hiv_pos_model" in X.columns
    assert 3 not in set(X["HIV"].dropna()), "'Em andamento' deve estar colapsado"


def test_derived_features_are_added():
    X = build_notification_feature_set(_cohort(), SPECS)
    assert "IDADE" in X.columns
    assert X.IDADE.iloc[0] == 39


def test_availability_table_has_one_row_per_spec():
    t = feature_availability_table(SPECS)
    assert len(t) == len(SPECS)
    assert {
        "raw_name",
        "availability",
        "dictionary_evidence",
        "in_notification_set",
    } <= set(t.columns)


def test_dictionary_table_documents_every_feature_in_the_set():
    t = feature_dictionary_table(SPECS)
    no_set = t[t.in_notification_set]
    assert (no_set.dictionary_evidence.str.len() > 0).all()


def test_era_zero_becomes_missing_not_category():
    # Até 2017 o SINAN gravava ausência como 0; da v5 em diante, como campo vazio.
    # Sem o mapeamento, 0 entra como categoria colinear com 'antes de 2018'.
    from tb_outcomes.features import collapse_era_zeros

    specs_min = [s for s in SPECS if s.raw_name in ("CS_GESTANT", "NU_CONTATO")]
    df = pd.DataFrame(
        {
            "CS_GESTANT": pd.array([0, 1, 2], dtype="Int64"),
            "NU_CONTATO": pd.array([0, 1, 2], dtype="Int64"),
        }
    )
    r = collapse_era_zeros(df, specs_min)
    assert pd.isna(r.CS_GESTANT.iloc[0]), "0 de era vira ausente"
    assert r.CS_GESTANT.iloc[1] == 1, "os demais códigos ficam intactos"
    assert r.NU_CONTATO.iloc[0] == 0, "NU_CONTATO tem 0 legítimo: não pode ser mexido"


def test_structurally_absent_fields_are_out_of_the_feature_set():
    # Ausência estrutural: o campo saiu da ficha. A coluna sobrevive no DBF mas a
    # pergunta não é mais feita — ID_OCUPA_N tem 10 valores em 808.889 pós-2018.
    # Nenhum tratamento de ausência recupera isso; a variável sai.
    por_nome = {s.raw_name: s for s in SPECS}
    for col in ["ID_OCUPA_N", "INSTITUCIO", "BACILOS_E2", "EXTRAPU2_N", "TESTE_TUBE"]:
        assert not por_nome[col].in_notification_set, f"{col} é ausência estrutural, sai do X"


def test_columns_with_legitimate_zero_are_not_flagged():
    por_nome = {s.raw_name: s for s in SPECS}
    for col in ["CS_ESCOL_N", "NU_CONTATO", "DOENCA_TRA"]:
        assert not por_nome[col].zero_means_missing, f"{col} tem 0 legítimo, não é era"
