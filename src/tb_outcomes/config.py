"""Carrega e valida as decisões de análise.

Nenhuma constante científica vive no código: tudo vem do YAML versionado
(briefing §29.3 item 7). Um campo obrigatório com valor 'TBD' bloqueia a etapa
que o exige, com erro nomeando o campo — nunca um fallback silencioso
(briefing §3 princípio 12).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml
from pydantic import BaseModel

TBD = "TBD"


class ConfigBlockedError(Exception):
    """Uma etapa exige uma decisão que ainda não foi congelada."""


class StudyConfig(BaseModel):
    protocol_status: str
    protocol_uri: str | None = None
    data_extraction_date: str
    source_snapshot_id: str
    primary_target_schema: str
    primary_prediction_time: str
    minimum_followup_days: int
    temporal_validation_enabled: bool
    temporal_test_period: str
    cohort_year_min: int
    cohort_year_max: int


class GeographyConfig(BaseModel):
    municipality_key: str
    fallback_key: str


class OutcomesConfig(BaseModel):
    dictionary_version: str


class SeedsConfig(BaseModel):
    python: int
    numpy: int


class AnalysisConfig(BaseModel):
    study: StudyConfig
    geography: GeographyConfig
    outcomes: OutcomesConfig
    seeds: SeedsConfig


def load_config(path: Path) -> AnalysisConfig:
    """Lê e valida a estrutura do YAML. Não checa campos 'TBD' — ver require_frozen."""
    with Path(path).open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return AnalysisConfig.model_validate(raw)


def _resolve(cfg: AnalysisConfig, dotted: str) -> object:
    valor: object = cfg
    for parte in dotted.split("."):
        valor = getattr(valor, parte)
    return valor


def require_frozen(cfg: AnalysisConfig, fields: list[str]) -> None:
    """Bloqueia se algum dos campos exigidos estiver 'TBD'.

    Cada comando declara o que precisa. 'build-cohort' gera todos os esquemas de
    alvo e portanto não exige primary_target_schema; 'run-full' exige.
    """
    pendentes = [f for f in fields if _resolve(cfg, f) == TBD]
    if pendentes:
        raise ConfigBlockedError(
            "Decisões não congeladas bloqueiam esta etapa: "
            + ", ".join(pendentes)
            + ". Preencha-as em configs/analysis_decisions.yaml (briefing §5.2)."
        )


def config_hash(cfg: AnalysisConfig) -> str:
    """SHA-256 da config canonicalizada, para amarrar resultados à configuração."""
    canonico = json.dumps(cfg.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonico.encode("utf-8")).hexdigest()[:12]
