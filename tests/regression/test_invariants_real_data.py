"""Invariantes que os achados da auditoria de 2026-07-15 transformaram em guarda.

Cada um destes falhou, ou teria falhado, contra a versão anterior do pipeline.
"""
from pathlib import Path

import pandas as pd
import pytest

from tb_outcomes.cohort import TargetSchema, load_outcome_rules, map_outcome
from tb_outcomes.config import load_config
from tb_outcomes.data import COLUNA_MUNICIPIO_RESOLVIDO, harmonize, load_raw_inputs

DATA = Path("data")
pytestmark = pytest.mark.regression


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    if not (DATA / "sinnan.parquet").exists():
        pytest.skip("brutos ausentes; rode o pipeline de aquisição (etapas 1-7).")
    return harmonize(load_raw_inputs(DATA), load_config(Path("configs/analysis_decisions.yaml")))


def test_geography_is_coherent_within_the_resolved_key(df):
    # Antes da correção: 5.081 de 5.530 municípios violavam isto.
    for col in ["LAT_MUNIC", "LONG_MUNIC", "NOME_MUNIC", "SIGLA_UF", "REGIAO"]:
        viol = int((df.groupby(COLUNA_MUNICIPIO_RESOLVIDO)[col].nunique(dropna=False) > 1).sum())
        assert viol == 0, f"{col} varia dentro de {COLUNA_MUNICIPIO_RESOLVIDO} em {viol} municípios"


def test_context_is_coherent_within_municipality_and_year(df):
    for col in ["IPEA_IDHM", "IBGE_PIB_PER_CAPITA"]:
        viol = int(
            (
                df.groupby([COLUNA_MUNICIPIO_RESOLVIDO, "ANO_DIAG"])[col].nunique(dropna=False)
                > 1
            ).sum()
        )
        assert viol == 0, f"{col} varia dentro de (município, ano) em {viol} pares"


def test_census_year_is_never_in_the_future(df):
    # Antes da correção: 100 violações, de municípios criados após 2010.
    total = 0
    for col in [c for c in df.columns if c.endswith("_ANO_CENSO")]:
        total += int((df[col] > df.ANO_DIAG).sum())
    assert total == 0, f"{total} registros usam censo posterior ao ano de diagnóstico"


def test_missingness_is_preserved_not_zeroed(df):
    # 66.420 vazios e 13.610 códigos '0' reais: o pipeline antigo fundia os dois.
    assert str(df.SITUA_ENCE.dtype) == "Int64"
    assert int(df.SITUA_ENCE.isna().sum()) == 66_420
    assert int((df.SITUA_ENCE == 0).sum()) == 13_610


def test_free_text_column_survives_typing(df):
    # AGRAVOUTDE: 57.436 descrições de comorbidade que o pipeline antigo zerava.
    assert str(df.AGRAVOUTDE.dtype) == "string"
    assert int(df.AGRAVOUTDE.notna().sum()) == 57_436


def test_code_6_is_absent_from_this_snapshot(df):
    # Código 6 é válido no dicionário SINAN NET 5.0 e tem zero registros aqui.
    # Falhar significa que o snapshot mudou — e que os 2.190 do briefing §2.1
    # talvez passem a ser reproduzíveis. É um sinal, não um defeito.
    assert int((df.SITUA_ENCE == 6).sum()) == 0, (
        "código 6 apareceu no snapshot: reavaliar a divergência declarada contra o briefing §2"
    )


def test_unresolved_rate_has_no_artificial_step_at_2018(df):
    # O SINAN mudou a codificação de 'sem encerramento' em 2018: '0' até 2017,
    # vazio depois. Se as duas eras não forem unificadas, a taxa dá um degrau.
    regras = load_outcome_rules(Path("configs/outcomes.yaml"))
    alvo = map_outcome(df, TargetSchema.LEGACY_3CLASS, regras)
    d = pd.DataFrame({"ano": pd.to_numeric(df.NU_ANO, errors="coerce"), "alvo": alvo})
    d = d[d.ano.between(2015, 2024)]
    taxa = d.groupby("ano").alvo.apply(lambda s: (s == "unresolved_or_censored").mean() * 100)

    antes = taxa.loc[2015:2017].mean()
    depois = taxa.loc[2018:2024].mean()
    assert abs(antes - depois) < 2.0, (
        f"degrau em 2018: {antes:.2f}% até 2017 vs {depois:.2f}% depois. "
        "As duas codificações de 'sem encerramento' não estão unificadas."
    )
    assert taxa.min() > 1.0, f"taxa implausivelmente baixa: {taxa.min():.2f}%"
    assert taxa.max() < 8.0, f"taxa implausivelmente alta: {taxa.max():.2f}%"


def test_no_record_is_excluded_for_missing_feature(df):
    # Briefing §9.1: ausência de feature nunca exclui. HIV e escolaridade têm
    # muita ausência e todos os registros continuam presentes.
    assert len(df) == 1_326_538


def test_no_unflagged_era_zero_survives_in_the_feature_set(df):
    """Nenhuma feature do X pode ter o código 0 da era antiga sem estar marcada.

    Generaliza a lição do SITUA_ENCE, onde o 0 pré-2018 fundia-se com o vazio
    pós-2018. O mesmo padrão está em 23 colunas: o 0 não consta no dicionário
    delas e é quase perfeitamente colinear com 'notificado antes de 2018'.

    Esta guarda existe para a próxima variável — se alguém adicionar uma feature
    com o padrão e esquecer de marcar, isto falha em vez de o modelo aprender a
    data da notificação disfarçada de característica clínica.
    """
    from tb_outcomes.features import load_feature_config

    specs = load_feature_config(Path("configs/features.yaml"))
    marcadas = {s.raw_name for s in specs if s.zero_means_missing}
    no_x = [s.raw_name for s in specs if s.in_notification_set]

    janela = df[df.NU_ANO.between(2015, 2025)]
    esquecidas = []
    for c in no_x:
        if c in marcadas or c not in janela.columns:
            continue
        if not pd.api.types.is_integer_dtype(janela[c]):
            continue
        zeros = janela[c] == 0
        if zeros.sum() == 0:
            continue
        if (janela.loc[zeros, "NU_ANO"] <= 2017).mean() >= 0.90:
            esquecidas.append(c)

    assert not esquecidas, (
        f"features do X com o padrão de era e sem zero_means_missing: {esquecidas}. "
        f"O código 0 dessas colunas é ausência anterior a 2018, não categoria."
    )


def test_structural_absence_is_never_imputable_in_the_feature_set(df):
    """Campos com ausência estrutural não podem estar no X.

    Ausência estrutural = o campo saiu da ficha; a coluna sobrevive no DBF mas a
    pergunta não é mais feita. Não é perda aleatória e nenhuma imputação a
    recupera. Medido pós-2018 em 808.889 notificações: ID_OCUPA_N tem 10 valores,
    INSTITUCIO 80, BACILOS_E2 108, TESTE_TUBE 45 em toda a janela.
    """
    from tb_outcomes.features import load_feature_config

    specs = load_feature_config(Path("configs/features.yaml"))
    no_x = {s.raw_name for s in specs if s.in_notification_set}
    estruturais = {"ID_OCUPA_N", "INSTITUCIO", "BACILOS_E2", "EXTRAPU2_N", "TESTE_TUBE"}
    assert not (no_x & estruturais), f"ausência estrutural no X: {sorted(no_x & estruturais)}"
