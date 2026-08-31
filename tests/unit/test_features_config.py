import pytest

from tb_outcomes.features import (
    Availability,
    FeatureConfigError,
    FeatureSpec,
    assert_covers_all_columns,
    assert_no_outcome_in_derivation,
    load_feature_config,
)


def _spec(raw_name, **kw):
    base = dict(
        raw_name=raw_name,
        harmonized_name=raw_name.lower(),
        type="categorical",
        source="SINAN",
        availability=Availability.AT_NOTIFICATION,
        dictionary_evidence="campo X do dicionário SINAN NET 5.0",
        in_notification_set=True,
        protected=False,
        contextual=False,
        clinical_review="approved",
        derived_from=[],
    )
    base.update(kw)
    return FeatureSpec(**base)


def test_covers_all_columns_passes_when_complete():
    specs = [_spec("CS_SEXO"), _spec("CS_RACA")]
    assert_covers_all_columns(specs, ["CS_SEXO", "CS_RACA"])


def test_covers_all_columns_fails_and_names_the_missing_ones():
    specs = [_spec("CS_SEXO")]
    with pytest.raises(FeatureConfigError) as e:
        assert_covers_all_columns(specs, ["CS_SEXO", "CS_RACA", "HIV"])
    assert "CS_RACA" in str(e.value) and "HIV" in str(e.value)


def test_covers_all_columns_fails_on_spec_for_unknown_column():
    specs = [_spec("CS_SEXO"), _spec("COLUNA_FANTASMA")]
    with pytest.raises(FeatureConfigError, match="COLUNA_FANTASMA"):
        assert_covers_all_columns(specs, ["CS_SEXO"])


def test_unknown_availability_cannot_be_in_notification_set():
    with pytest.raises(ValueError, match="unknown"):
        _spec("TESTE_TUBE", availability=Availability.UNKNOWN, in_notification_set=True)


def test_post_baseline_cannot_be_in_notification_set():
    with pytest.raises(ValueError, match="post_baseline"):
        _spec("ANT_RETRO", availability=Availability.POST_BASELINE, in_notification_set=True)


def test_outcome_cannot_be_in_notification_set():
    with pytest.raises(ValueError, match="outcome"):
        _spec("SITUA_ENCE", availability=Availability.OUTCOME, in_notification_set=True)


def test_spec_requires_evidence_unless_pending_review():
    with pytest.raises(ValueError, match="dictionary_evidence"):
        _spec("CS_SEXO", dictionary_evidence="")
    _spec(
        "TESTE_TUBE",
        dictionary_evidence="",
        availability=Availability.UNKNOWN,
        in_notification_set=False,
        clinical_review="pending",
    )


def test_load_feature_config_reads_real_file():
    specs = load_feature_config("configs/features.yaml")
    assert len(specs) > 0
    assert all(isinstance(s, FeatureSpec) for s in specs)


def test_feature_derived_from_outcome_is_rejected_at_spec_level():
    # Uma feature = DT_ENCERRA - DT_NOTIFIC seria o tempo até o desfecho: derivar
    # da data de encerramento vaza o próprio alvo para dentro do X.
    with pytest.raises(ValueError, match="DT_ENCERRA"):
        _spec("TEMPO_TRATAMENTO", derived_from=["DT_ENCERRA", "DT_NOTIFIC"])


def test_feature_derived_from_target_is_rejected_at_spec_level():
    with pytest.raises(ValueError, match="SITUA_ENCE"):
        _spec("DIAS_ATE_DESFECHO", derived_from=["SITUA_ENCE"])


def test_outcome_derived_feature_is_allowed_outside_the_notification_set():
    # Pode existir como artefato descritivo, desde que fora do X.
    _spec(
        "TEMPO_TRATAMENTO",
        derived_from=["DT_ENCERRA", "DT_NOTIFIC"],
        availability=Availability.OUTCOME,
        in_notification_set=False,
    )


def test_assert_no_outcome_in_derivation_catches_it_at_config_level():
    specs = [
        _spec("CS_SEXO"),
        _spec(
            "FLAG_ENCERRA",
            derived_from=["DT_ENCERRA"],
            availability=Availability.OUTCOME,
            in_notification_set=False,
        ),
    ]
    assert_no_outcome_in_derivation(specs)

    # Forçar in_notification_set contornando o validador do FeatureSpec:
    specs[1].in_notification_set = True
    with pytest.raises(FeatureConfigError, match="FLAG_ENCERRA"):
        assert_no_outcome_in_derivation(specs)
