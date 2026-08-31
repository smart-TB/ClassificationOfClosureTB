from pathlib import Path

from tb_outcomes.config import load_config
from tb_outcomes.data import (
    COLUNA_MUNICIPIO_RESOLVIDO,
    RawInputs,
    harmonize,
    quality_report,
    quality_report_markdown,
)
from tests.fixtures.synthetic import make_synthetic_raw

CONFIG_YAML = """
study:
  protocol_status: "retrospective_protocol"
  protocol_uri: null
  data_extraction_date: "2026-04-07"
  source_snapshot_id: "sint"
  primary_target_schema: "TBD"
  primary_prediction_time: "TBD"
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
"""


def _cfg(tmp_path: Path):
    p = tmp_path / "cfg.yaml"
    p.write_text(CONFIG_YAML, encoding="utf-8")
    return load_config(p)


def _harmonized(tmp_path):
    s = make_synthetic_raw(n=600, seed=3)
    raw = RawInputs(
        s.sinan,
        s.municipios,
        s.municipios_latlong,
        s.estados,
        s.ibge,
        s.ipea,
        s.cnes_prof,
        s.cnes_estab,
    )
    return harmonize(raw, _cfg(tmp_path))


def test_harmonize_geography_is_coherent_within_resolved_key(tmp_path):
    r = _harmonized(tmp_path)
    for col in ["LAT_MUNIC", "LONG_MUNIC", "NOME_MUNIC", "SIGLA_UF", "REGIAO"]:
        viol = (r.groupby(COLUNA_MUNICIPIO_RESOLVIDO)[col].nunique(dropna=False) > 1).sum()
        assert viol == 0, f"{col} varia dentro do municipio de analise"


def test_harmonize_context_is_coherent_within_municipality_year(tmp_path):
    r = _harmonized(tmp_path)
    viol = (
        r.groupby([COLUNA_MUNICIPIO_RESOLVIDO, "ANO_DIAG"]).IPEA_IDHM.nunique(dropna=False) > 1
    ).sum()
    assert viol == 0


def test_harmonize_census_is_never_in_the_future(tmp_path):
    r = _harmonized(tmp_path)
    assert (r.IPEA_IDHM_ANO_CENSO > r.ANO_DIAG).sum() == 0


def test_harmonize_preserves_missing_outcome(tmp_path):
    r = _harmonized(tmp_path)
    assert r.SITUA_ENCE.dtype == "Int64"
    assert r.SITUA_ENCE.isna().any()


def test_quality_report_counts_divergence_and_impossible_dates(tmp_path):
    rep = quality_report(_harmonized(tmp_path))
    assert rep["n_rows"] > 0
    assert rep["residence_notification_divergence"] >= 0
    assert rep["impossible_dates"]["DT_ENCERRA"] > 0


def test_quality_report_markdown_renders_sections(tmp_path):
    md = quality_report_markdown(quality_report(_harmonized(tmp_path)))
    assert "# Relatório de qualidade dos dados" in md
    assert "## Datas fora do intervalo plausível" in md
    assert "## Ausência por coluna" in md
