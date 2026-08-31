"""Elegibilidade, esquemas de alvo, seguimento e fluxo da coorte.

Três populações, nomeadas e nunca intercambiáveis:
  recuperada  -> tudo o que a aquisição entregou
  declarada   -> recuperada ∩ notificação na janela do protocolo
  analítica   -> declarada ∩ elegibilidade ∩ desfecho no esquema
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

UNRESOLVED = "unresolved_or_censored"
EXCLUDED = "excluded"

IDADE_MAXIMA_PLAUSIVEL = 120


class TargetSchema:
    LEGACY_3CLASS = "legacy_3class"
    REVISED_4CLASS = "revised_4class"
    ACTIONABLE_BINARY = "actionable_binary"
    SENSITIVE_TB_3CLASS = "sensitive_tb_3class"  # primário (PI 2026-07-19)

    @classmethod
    def all(cls) -> list[str]:
        return [cls.LEGACY_3CLASS, cls.REVISED_4CLASS, cls.ACTIONABLE_BINARY,
                cls.SENSITIVE_TB_3CLASS]


def load_outcome_rules(path: Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def map_outcome(df: pd.DataFrame, schema: str, rules: dict) -> pd.Series:
    """Mapeia SITUA_ENCE para o rótulo do esquema.

    O <NA> (campo vazio, de 2018 em diante) e o 0 literal (até 2017) são o mesmo
    estado — sem encerramento registrado — e recebem o mesmo rótulo.
    """
    mapa = rules["schemas"][schema]
    rotulo_na = mapa[None]
    conhecidos = {k for k in mapa if k is not None}

    codigos = df["SITUA_ENCE"]
    presentes = set(codigos.dropna().astype(int).unique())
    desconhecidos = presentes - conhecidos
    if desconhecidos:
        raise ValueError(
            f"Códigos de encerramento fora do dicionário {rules['dictionary_version']}: "
            f"{sorted(desconhecidos)}. Verifique a versão do dicionário antes de prosseguir."
        )

    return pd.Series(
        [rotulo_na if pd.isna(c) else mapa[int(c)] for c in codigos],
        index=df.index,
        dtype="object",
        name=f"target_{schema}",
    )


def compute_followup(df: pd.DataFrame, extraction_date: str) -> pd.Series:
    """Dias entre a notificação e a data de extração (briefing §9.3)."""
    notif = pd.to_datetime(df["DT_NOTIFIC"], errors="coerce")
    return (pd.Timestamp(extraction_date) - notif).dt.days


def apply_eligibility(df: pd.DataFrame, cfg) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aplica a elegibilidade e devolve (coorte elegível, log de exclusões).

    Nunca exclui por ausência de feature (briefing §9.1): só por critérios de
    qualidade declarados. O log traz contagem sequencial (quantos cada regra
    tirou, na ordem) e não sequencial (quantos violariam a regra isoladamente).
    """
    estudo = cfg.study
    seguimento = compute_followup(df, estudo.data_extraction_date)
    idade = pd.to_numeric(df["NU_ANO"], errors="coerce") - pd.to_numeric(
        df["ANO_NASC"], errors="coerce"
    )

    regras = {
        "municipio_irrecuperavel": df["ID_MUNIC_ANALISE"].isna(),
        "idade_impossivel": idade.notna() & ((idade < 0) | (idade > IDADE_MAXIMA_PLAUSIVEL)),
        "fora_da_janela_declarada": ~pd.to_numeric(df["NU_ANO"], errors="coerce").between(
            estudo.cohort_year_min, estudo.cohort_year_max
        ),
        "seguimento_insuficiente": seguimento.isna()
        | (seguimento < estudo.minimum_followup_days),
    }

    restantes = pd.Series(True, index=df.index)
    linhas = []
    for motivo, viola in regras.items():
        viola = viola.fillna(False).astype(bool)
        excluidos_agora = int((restantes & viola).sum())
        linhas.append(
            {
                "reason": motivo,
                "n": excluidos_agora,  # sequencial: na ordem das regras
                "n_nao_sequencial": int(viola.sum()),  # isolado: violaria sozinho
            }
        )
        restantes &= ~viola
        logger.info("Exclusão '%s': %d registros.", motivo, excluidos_agora)

    return df.loc[restantes].copy(), pd.DataFrame(linhas)


def audit_duplicates(df: pd.DataFrame, sinan_columns: list[str]) -> pd.DataFrame:
    """Duplicatas exatas sobre os campos brutos do SINAN (briefing §10.1).

    Não remove nada: a política de duplicatas é decisão de protocolo. Só mede.
    """
    chave = df[sinan_columns].astype(str)
    em_grupo = chave.duplicated(keep=False)
    return pd.DataFrame(
        [
            {"metrica": "linhas_total", "valor": len(df)},
            {"metrica": "linhas_em_grupos_duplicados", "valor": int(em_grupo.sum())},
            {"metrica": "linhas_redundantes", "valor": int(chave.duplicated().sum())},
        ]
    )


@dataclass
class CohortResult:
    analytic: pd.DataFrame
    flow: pd.DataFrame
    outcome_distribution: pd.DataFrame
    exclusions: pd.DataFrame
    followup: pd.DataFrame


def followup_audit(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """Seguimento potencial por ano (briefing §9.3).

    É o insumo da análise de censura de 180/270/365 dias. Em 2025 a proporção
    com seguimento suficiente cai a ~27% — motivo pelo qual o ano entra na
    coorte declarada mas não na analítica.
    """
    d = df.copy()
    d["_followup"] = compute_followup(d, cfg.study.data_extraction_date)
    minimo = cfg.study.minimum_followup_days
    return (
        d.groupby("NU_ANO")
        .agg(
            n=("_followup", "size"),
            followup_mediano=("_followup", "median"),
            followup_minimo=("_followup", "min"),
            pct_followup_ge_min=(
                "_followup",
                lambda s: round(float((s >= minimo).mean() * 100), 2),
            ),
        )
        .reset_index()
    )


def build_cohort(df: pd.DataFrame, cfg, rules: dict, schema: str) -> CohortResult:
    """Constrói a coorte analítica e o fluxograma das três populações."""
    n_recuperada = len(df)

    ano = pd.to_numeric(df["NU_ANO"], errors="coerce")
    declarada = df.loc[ano.between(cfg.study.cohort_year_min, cfg.study.cohort_year_max)]

    elegivel, exclusoes = apply_eligibility(df, cfg)

    coluna = f"target_{schema}"
    elegivel = elegivel.copy()
    elegivel[coluna] = map_outcome(elegivel, schema, rules)

    # unresolved_or_censored nunca entra no treino supervisionado (briefing §6.4).
    analitica = elegivel.loc[~elegivel[coluna].isin([UNRESOLVED, EXCLUDED])].copy()

    flow = pd.DataFrame(
        [
            {
                "stage": "recuperada",
                "n": n_recuperada,
                "description": "tudo o que a aquisição entregou",
            },
            {
                "stage": "declarada",
                "n": len(declarada),
                "description": (
                    f"notificação em {cfg.study.cohort_year_min}-{cfg.study.cohort_year_max}"
                ),
            },
            {
                "stage": "elegivel",
                "n": len(elegivel),
                "description": "declarada após critérios de qualidade e seguimento",
            },
            {
                "stage": "analitica",
                "n": len(analitica),
                "description": f"elegível com desfecho no esquema {schema}",
            },
        ]
    )

    distribuicao = analitica[coluna].value_counts().rename_axis("target").reset_index(name="n")
    distribuicao["pct"] = (distribuicao.n / max(len(analitica), 1) * 100).round(2)

    logger.info(
        "Coorte %s: recuperada=%d declarada=%d elegível=%d analítica=%d",
        schema,
        n_recuperada,
        len(declarada),
        len(elegivel),
        len(analitica),
    )

    return CohortResult(analitica, flow, distribuicao, exclusoes, followup_audit(df, cfg))
