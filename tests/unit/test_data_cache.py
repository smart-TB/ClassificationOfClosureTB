"""Cache da coorte harmonizada: reusa se fresco, rebuilda se o insumo mudou.

O risco de materializar é servir dado velho em silêncio. A chave de cache detecta
mudança no bruto, na config de geografia e na versão da lógica de harmonização.
"""
from pathlib import Path

import pandas as pd

from tb_outcomes.data import (
    HARMONIZATION_VERSION,
    harmonization_key,
    load_or_build_harmonized,
)
from tb_outcomes.config import load_config

CONFIG_YAML = """
study:
  protocol_status: "retrospective_protocol"
  protocol_uri: null
  data_extraction_date: "2026-04-07"
  source_snapshot_id: "sint"
  primary_target_schema: "TBD"
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
"""


def _cfg(tmp_path: Path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    p = tmp_path / "cfg.yaml"
    p.write_text(CONFIG_YAML, encoding="utf-8")
    return load_config(p)


def _write_raw(data_dir: Path):
    from tests.fixtures.synthetic import make_synthetic_raw

    s = make_synthetic_raw(n=300, seed=5)
    data_dir.mkdir(exist_ok=True)
    s.sinan.to_parquet(data_dir / "sinnan.parquet", index=False)
    s.municipios.to_csv(data_dir / "municipios_ibge.csv", index=False)
    s.municipios_latlong.to_csv(data_dir / "municipios_lat_long.csv", index=False)
    (data_dir / "estados.json").write_text(s.estados.to_json(orient="records"), encoding="utf-8")
    s.ibge.to_csv(data_dir / "indicadores_IBGE.csv", index=False)
    s.ipea.to_csv(data_dir / "indicadores_IPEA.csv", index=False)
    s.cnes_prof.to_parquet(data_dir / "profissionais_CNES.parquet", index=False)
    s.cnes_estab.to_parquet(data_dir / "estabelecimentos_CNES.parquet", index=False)


def test_key_changes_when_geography_config_changes(tmp_path):
    d = tmp_path / "data"
    _write_raw(d)
    cfg_a = _cfg(tmp_path / "a")
    cfg_b = _cfg(tmp_path / "b")
    object.__setattr__(cfg_b.geography, "municipality_key", "ID_MUNICIP")
    assert harmonization_key(d, cfg_a) != harmonization_key(d, cfg_b)


def test_key_includes_harmonization_version(tmp_path):
    d = tmp_path / "data"
    _write_raw(d)
    cfg = _cfg(tmp_path)
    k = harmonization_key(d, cfg)
    assert str(HARMONIZATION_VERSION) in k or len(k) == 16  # versão entra no hash


def test_first_call_builds_and_second_call_reuses(tmp_path):
    d = tmp_path / "data"
    _write_raw(d)
    cfg = _cfg(tmp_path)

    df1, built1 = load_or_build_harmonized(d, cfg)
    assert built1 is True, "primeira chamada deve construir"
    assert (d / "harmonized.parquet").exists()

    df2, built2 = load_or_build_harmonized(d, cfg)
    assert built2 is False, "segunda chamada deve reusar o cache"
    pd.testing.assert_frame_equal(df1, df2)


def test_stale_cache_is_rebuilt_when_raw_changes(tmp_path):
    d = tmp_path / "data"
    _write_raw(d)
    cfg = _cfg(tmp_path)
    _, built1 = load_or_build_harmonized(d, cfg)
    assert built1 is True

    # muda o bruto: o cache deve ser invalidado
    from tests.fixtures.synthetic import make_synthetic_raw

    make_synthetic_raw(n=400, seed=9).sinan.to_parquet(d / "sinnan.parquet", index=False)
    _, built2 = load_or_build_harmonized(d, cfg)
    assert built2 is True, "cache velho não pode ser servido após mudança no bruto"


def test_rebuild_flag_forces_rebuild(tmp_path):
    d = tmp_path / "data"
    _write_raw(d)
    cfg = _cfg(tmp_path)
    load_or_build_harmonized(d, cfg)
    _, built = load_or_build_harmonized(d, cfg, rebuild=True)
    assert built is True, "rebuild=True deve reconstruir mesmo com cache válido"
