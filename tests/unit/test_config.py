from pathlib import Path

import pytest

from tb_outcomes.config import ConfigBlockedError, config_hash, load_config, require_frozen


def _write(tmp_path: Path, primary_target_schema: str = "legacy_3class") -> Path:
    p = tmp_path / "cfg.yaml"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        f"""
study:
  protocol_status: "retrospective_protocol"
  protocol_uri: null
  data_extraction_date: "2026-04-07"
  source_snapshot_id: "sinan-tb-2026-04-07"
  primary_target_schema: "{primary_target_schema}"
  primary_prediction_time: "notification"
  minimum_followup_days: 365
  temporal_validation_enabled: true
  temporal_test_period: "2024"
  cohort_year_min: 2015
  cohort_year_max: 2025
geography:
  municipality_key: "ID_MN_RESI"
  fallback_key: "ID_MUNICIP"
outcomes:
  dictionary_version: "SINAN-NET-5.0-tuberculose"
seeds:
  python: 42
  numpy: 42
""",
        encoding="utf-8",
    )
    return p


def test_load_config_reads_frozen_decisions(tmp_path):
    cfg = load_config(_write(tmp_path))
    assert cfg.study.cohort_year_min == 2015
    assert cfg.study.cohort_year_max == 2025
    assert cfg.study.minimum_followup_days == 365
    assert cfg.study.temporal_test_period == "2024"
    assert cfg.geography.municipality_key == "ID_MN_RESI"


def test_require_frozen_blocks_on_tbd(tmp_path):
    cfg = load_config(_write(tmp_path, primary_target_schema="TBD"))
    with pytest.raises(ConfigBlockedError) as e:
        require_frozen(cfg, ["study.primary_target_schema"])
    assert "study.primary_target_schema" in str(e.value)


def test_require_frozen_passes_when_field_is_set(tmp_path):
    cfg = load_config(_write(tmp_path))
    require_frozen(cfg, ["study.primary_target_schema"])


def test_tbd_in_unrequested_field_does_not_block(tmp_path):
    # build-cohort não precisa de primary_target_schema: gera todos os esquemas.
    cfg = load_config(_write(tmp_path, primary_target_schema="TBD"))
    require_frozen(cfg, ["study.minimum_followup_days"])


def test_config_hash_is_stable_and_sensitive(tmp_path):
    a = load_config(_write(tmp_path / "a"))
    b = load_config(_write(tmp_path / "b"))
    assert config_hash(a) == config_hash(b)
    c = load_config(_write(tmp_path / "c", primary_target_schema="revised_4class"))
    assert config_hash(a) != config_hash(c)
