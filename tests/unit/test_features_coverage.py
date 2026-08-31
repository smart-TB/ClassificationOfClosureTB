"""O portão: nenhuma coluna da coorte pode ficar sem classificação."""
from pathlib import Path

import pandas as pd
import pytest

from tb_outcomes.features import Availability, assert_covers_all_columns, load_feature_config

CONFIG = Path("configs/features.yaml")
SPECS = load_feature_config(CONFIG)

COORTE = Path("data/sinnan_tratado.parquet")
COLUNAS_REAIS = sorted(pd.read_parquet(COORTE).columns) if COORTE.exists() else []


@pytest.mark.regression
def test_every_column_of_the_real_cohort_is_classified():
    if not COLUNAS_REAIS:
        pytest.skip("coorte harmonizada ausente (microdado, não versionado).")
    assert_covers_all_columns(SPECS, COLUNAS_REAIS)


def test_no_post_baseline_or_outcome_in_notification_set():
    proibidos = {
        Availability.POST_BASELINE,
        Availability.FOLLOWUP,
        Availability.OUTCOME,
        Availability.UNKNOWN,
        Availability.INTERNAL,
    }
    vazando = [s.raw_name for s in SPECS if s.in_notification_set and s.availability in proibidos]
    assert not vazando, f"variáveis indevidas no conjunto primário: {vazando}"


def test_outcome_fields_are_marked_as_outcome():
    por_nome = {s.raw_name: s for s in SPECS}
    for col in ["SITUA_ENCE", "DT_ENCERRA", "TRANSF", "UF_TRANSF", "MUN_TRANSF"]:
        assert por_nome[col].availability == Availability.OUTCOME, col


def test_known_post_baseline_fields_are_excluded():
    por_nome = {s.raw_name: s for s in SPECS}
    for col in ["ANT_RETRO", "CULTURA_ES", "CULTURA_OU", "HISTOPATOL", "TEST_SENSI"]:
        assert por_nome[col].availability == Availability.POST_BASELINE, col
        assert por_nome[col].in_notification_set is False, col


def test_followup_fields_are_excluded():
    por_nome = {s.raw_name: s for s in SPECS}
    for col in ["BACILOSC_1", "BACILOSC_6", "BAC_APOS_6", "SITUA_9_M", "SITUA_12_M"]:
        assert por_nome[col].availability == Availability.FOLLOWUP, col
        assert por_nome[col].in_notification_set is False, col


def test_strata_and_keys_are_never_features():
    por_nome = {s.raw_name: s for s in SPECS}
    for col in ["NU_ANO", "ID_MN_RESI", "ID_MUNIC_ANALISE"]:
        assert por_nome[col].in_notification_set is False, f"{col} é estrato/chave, não feature"


def test_every_spec_has_evidence_or_pending_review():
    sem = [
        s.raw_name for s in SPECS if not s.dictionary_evidence and s.clinical_review != "pending"
    ]
    assert not sem, f"sem evidência do dicionário e sem revisão pendente: {sem}"


def test_ant_retro_is_available_only_as_named_sensitivity():
    spec = {s.raw_name: s for s in SPECS}["ANT_RETRO"]
    assert spec.sensitivity_only == "include_antiretro_postbaseline"


def test_special_populations_stay_separate():
    # O original colapsava as quatro num binário e descartava as originais, o que
    # inviabiliza a equidade do briefing §20.1.
    por_nome = {s.raw_name: s for s in SPECS}
    for col in ["POP_LIBER", "POP_RUA", "POP_SAUDE", "POP_IMIG"]:
        assert por_nome[col].in_notification_set is True, f"{col} precisa entrar separada"
