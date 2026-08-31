"""Fixture de projeto sintético, compartilhada pelos testes de integração.

Monta um diretório temporário com a mesma estrutura do projeto real, mas com dado
sintético — nenhum microdado. É o que permite ao revisor rodar o pipeline inteiro.
"""
from pathlib import Path

import pytest

from tests.fixtures.synthetic import make_synthetic_raw

CONFIG_YAML = """
study:
  protocol_status: "retrospective_protocol"
  protocol_uri: null
  data_extraction_date: "2026-04-07"
  source_snapshot_id: "sintetico"
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


def _monta(tmp_path: Path, configs_extra: list[str]) -> Path:
    (tmp_path / "data").mkdir(exist_ok=True)
    (tmp_path / "configs").mkdir(exist_ok=True)

    s = make_synthetic_raw(n=800, seed=11)
    s.sinan.to_parquet(tmp_path / "data" / "sinnan.parquet", index=False)
    s.municipios.to_csv(tmp_path / "data" / "municipios_ibge.csv", index=False)
    s.municipios_latlong.to_csv(tmp_path / "data" / "municipios_lat_long.csv", index=False)
    (tmp_path / "data" / "estados.json").write_text(
        s.estados.to_json(orient="records"), encoding="utf-8"
    )
    s.ibge.to_csv(tmp_path / "data" / "indicadores_IBGE.csv", index=False)
    s.ipea.to_csv(tmp_path / "data" / "indicadores_IPEA.csv", index=False)
    s.cnes_prof.to_parquet(tmp_path / "data" / "profissionais_CNES.parquet", index=False)
    s.cnes_estab.to_parquet(tmp_path / "data" / "estabelecimentos_CNES.parquet", index=False)

    (tmp_path / "configs" / "analysis_decisions.yaml").write_text(CONFIG_YAML, encoding="utf-8")
    for nome in configs_extra:
        (tmp_path / "configs" / nome).write_text(
            Path(f"configs/{nome}").read_text(encoding="utf-8"), encoding="utf-8"
        )
    return tmp_path


@pytest.fixture
def projeto(tmp_path, monkeypatch):
    p = _monta(tmp_path, ["outcomes.yaml"])
    monkeypatch.chdir(p)
    return p


@pytest.fixture
def projeto_features(tmp_path, monkeypatch):
    p = _monta(
        tmp_path,
        ["outcomes.yaml", "features.yaml", "splits.yaml", "preprocess.yaml", "imbalance.yaml", "models.yaml"],
    )
    monkeypatch.chdir(p)
    return p
