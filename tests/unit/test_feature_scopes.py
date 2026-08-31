"""Escopos de feature da ablação territorial (ESPECIFICACAO §3.3).

A pergunta científica da ablação — "determinante real ou proxy da qualidade da
notificação?" — só tem resposta se o corte individual × municipal for declarado, e não
inferido por heurística de nome. O corte vem do `contextual` do FeatureSpec.
"""
import pandas as pd
import pytest

from tb_outcomes.features import (
    FEATURE_SCOPES,
    classify_feature_scope,
    load_feature_config,
    select_feature_scope,
)


def _specs():
    return load_feature_config("configs/features.yaml")


def test_escopos_declarados():
    assert FEATURE_SCOPES == ("individual", "municipal", "combined", "clinical_baseline")


def test_baseline_clinico_e_subconjunto_nomeado():
    from tb_outcomes.features import CLINICAL_BASELINE_FEATURES

    specs = _specs()
    cols = [s.raw_name for s in specs if s.in_notification_set] + list(
        CLINICAL_BASELINE_FEATURES)
    X = pd.DataFrame({c: [0, 1] for c in dict.fromkeys(cols)})
    b = select_feature_scope(X, specs, "clinical_baseline")
    assert list(b.columns) == list(CLINICAL_BASELINE_FEATURES)
    # idade + HIV + álcool/drogas + situação de rua, como manda a §3.8
    assert set(b.columns) == {"IDADE", "hiv_pos_model", "AGRAVALCOO", "AGRAVDROGA", "POP_RUA"}


def test_baseline_clinico_falha_se_faltar_variavel():
    # devolver um baseline menor em silêncio inflaria a vantagem do modelo completo
    X = pd.DataFrame({"IDADE": [1], "hiv_pos_model": [0], "AGRAVALCOO": [0]})
    with pytest.raises(ValueError, match="baseline clínico sem as colunas"):
        select_feature_scope(X, _specs(), "clinical_baseline")


def test_contextual_manda_no_rotulo():
    specs = _specs()
    ns = [s for s in specs if s.in_notification_set]
    cols = [s.raw_name for s in ns]
    escopo = classify_feature_scope(cols, specs)
    for s in ns:
        esperado = "municipal" if s.contextual else "individual"
        assert escopo[s.raw_name] == esperado, s.raw_name


def test_derivadas_sem_spec_sao_individuais():
    # IDADE, os *_BIN e as interações descrevem a PESSOA; hiv_pos_model idem.
    cols = ["IDADE", "POP_ESPECIAIS_BIN", "IDADE_X_DIABETES", "IDADE_X_AIDS", "hiv_pos_model"]
    escopo = classify_feature_scope(cols, _specs())
    assert set(escopo.values()) == {"individual"}


def test_coluna_desconhecida_falha_em_vez_de_ser_engolida():
    # Uma feature nova sem spec e fora da lista de derivadas não pode cair num
    # braço por omissão — isso enviesaria a ablação em silêncio.
    with pytest.raises(ValueError, match="sem escopo declarado"):
        classify_feature_scope(["FEATURE_NOVA_NAO_DECLARADA"], _specs())


def test_select_particiona_sem_perder_nem_duplicar_coluna():
    specs = _specs()
    cols = [s.raw_name for s in specs if s.in_notification_set] + ["IDADE"]
    X = pd.DataFrame({c: [0, 1] for c in cols})

    ind = select_feature_scope(X, specs, "individual")
    mun = select_feature_scope(X, specs, "municipal")
    comb = select_feature_scope(X, specs, "combined")

    assert list(comb.columns) == list(X.columns)
    assert set(ind.columns) & set(mun.columns) == set()
    assert set(ind.columns) | set(mun.columns) == set(X.columns)
    assert len(ind.columns) and len(mun.columns)
    # a ordem original das colunas é preservada dentro de cada braço
    assert list(ind.columns) == [c for c in X.columns if c in set(ind.columns)]


def test_select_recusa_escopo_invalido():
    with pytest.raises(ValueError, match="escopo desconhecido"):
        select_feature_scope(pd.DataFrame({"IDADE": [1]}), _specs(), "municipais")
