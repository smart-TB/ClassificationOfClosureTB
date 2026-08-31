"""Estratégias de desbalanceamento, aplicadas só ao treino da dobra.

Dois princípios (briefing §3): nº 5 — reamostragem/peso nunca alcança a
avaliação; nº 6 — estratégias exclusivas, nunca combinadas. A auditoria (§13.3)
prova que as quatro produzem configurações distintas — o teste que torna
impossível a anomalia histórica dos 4 modelos idênticos.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class ImbalanceSensitivity(BaseModel):
    cost_strata: list[str]
    R: list[float]
    w_min: list[float]


class ImbalanceConfig(BaseModel):
    strategies: list[str]
    cost_stratum: str
    R: float
    w_min: float
    w_max: float
    rescale_mean_one: bool
    sensitivity: ImbalanceSensitivity
    # Cap do oversampling: razão máxima majoritária:minoritária após oversamplear.
    # Balancear até 1:1 na coorte real explodiria o treino (~1,9M); com 3 a minoritária
    # sobe só até majoritária/3.
    oversample_max_ratio: int = 3


def load_imbalance_config(path: Path) -> ImbalanceConfig:
    with Path(path).open("r", encoding="utf-8") as fh:
        return ImbalanceConfig.model_validate(yaml.safe_load(fh))


def local_cost_weights(
    y_train: pd.Series,
    strata_train: pd.Series,
    R: float,
    w_min: float,
    w_max: float,
    rescale_mean_one: bool,
) -> np.ndarray:
    """Peso por estrato espaço-temporal (§13.2), só na classe majoritária do estrato.

    w_g = clip(R * n_min / max(1, n_maj), w_min, w_max) para a maioria do estrato;
    minorias recebem 1. Estrato de uma classe (n_min=0) vai ao piso — a degeneração
    que a granularidade fina (ano×município) produz em 59% dos estratos.
    """
    y = np.asarray(y_train)
    strata = np.asarray(strata_train)
    w = np.ones(len(y), dtype="float64")

    for s in np.unique(strata):
        mask = strata == s
        classes, counts = np.unique(y[mask], return_counts=True)
        n_maj = counts.max()
        n_min = counts.min() if len(counts) > 1 else 0
        peso_maj = float(np.clip(R * n_min / max(1, n_maj), w_min, w_max))
        classe_maj = classes[counts.argmax()]
        w[mask & (y == classe_maj)] = peso_maj

    if rescale_mean_one:
        w = w / w.mean()
    return w


@dataclass
class ImbalanceResult:
    """O que o executor recebe: índices do treino e/ou pesos."""

    train_index: np.ndarray  # índices (na base de treino) após resampling
    sample_weight: np.ndarray | None  # pesos (cost) ou None
    strategy: str  # a estratégia efetiva ('unsupported' se não suportada)


def apply_strategy(
    y_train: pd.Series,
    strata_train: pd.Series,
    strategy: str,
    cfg: ImbalanceConfig,
    random_state: int,
    supports_weight: bool,
) -> ImbalanceResult:
    """Aplica a estratégia SÓ ao treino. Nunca recebe a avaliação.

    Over/under devolvem os índices reamostrados (imblearn.sample_indices_); cost
    devolve os pesos e mantém os índices; none é identidade. Custo sem suporte a
    sample_weight vira 'unsupported', não simulação (§13.1).
    """
    idx = np.arange(len(y_train))
    y = np.asarray(y_train)

    if strategy == "none":
        return ImbalanceResult(train_index=idx, sample_weight=None, strategy="none")

    if strategy == "random_oversampling":
        from imblearn.over_sampling import RandomOverSampler

        # Cap 1:R — a minoritária sobe só até majoritária/R (não balanceia até a
        # majoritária, que na coorte real explodiria o treino para ~1,9M linhas e
        # travava modelos iterativos). Classes já acima do piso ficam inalteradas.
        counts = pd.Series(y).value_counts()
        floor = int(counts.max()) // max(int(cfg.oversample_max_ratio), 1)
        strat = {c: max(int(n), floor) for c, n in counts.items()}
        ros = RandomOverSampler(sampling_strategy=strat, random_state=random_state)
        ros.fit_resample(idx.reshape(-1, 1), y)
        return ImbalanceResult(ros.sample_indices_, None, "random_oversampling")

    if strategy == "random_undersampling":
        from imblearn.under_sampling import RandomUnderSampler

        rus = RandomUnderSampler(random_state=random_state)
        rus.fit_resample(idx.reshape(-1, 1), y)
        return ImbalanceResult(rus.sample_indices_, None, "random_undersampling")

    if strategy == "local_cost_sensitive":
        if not supports_weight:
            return ImbalanceResult(idx, None, "unsupported")
        w = local_cost_weights(
            y_train, strata_train, cfg.R, cfg.w_min, cfg.w_max, cfg.rescale_mean_one
        )
        return ImbalanceResult(idx, w, "local_cost_sensitive")

    raise ValueError(f"estratégia desconhecida: {strategy}")


def config_signature(result: ImbalanceResult) -> str:
    """Assinatura da configuração: hash das linhas selecionadas + dos pesos.

    Duas estratégias com a mesma assinatura produzem exatamente o mesmo treino —
    o que não deveria acontecer entre estratégias distintas (§24.4).
    """
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(np.sort(result.train_index)).tobytes())
    if result.sample_weight is None:
        h.update(b"no_weight")
    else:
        h.update(np.ascontiguousarray(np.round(result.sample_weight, 8)).tobytes())
    return h.hexdigest()[:16]


def assert_strategies_differ(results: dict[str, ImbalanceResult]) -> None:
    """Falha se duas estratégias distintas produzirem a mesma configuração.

    O teste que torna impossível a anomalia da §2.1: 4 modelos com métricas
    idênticas entre estratégias que deveriam diferir.
    """
    assinaturas: dict[str, str] = {}
    for nome, r in results.items():
        if r.strategy == "unsupported":
            continue
        sig = config_signature(r)
        for outro, sig_outro in assinaturas.items():
            if sig == sig_outro:
                raise AssertionError(
                    f"estratégia '{nome}' é idêntica a '{outro}' — mesma configuração "
                    f"de treino. É a anomalia histórica; estratégias devem diferir."
                )
        assinaturas[nome] = sig


def imbalance_audit(y_train: pd.Series, results: dict[str, ImbalanceResult]) -> dict:
    """Contas antes/depois e pesos por classe, por estratégia (§13.1)."""
    y = np.asarray(y_train)
    n_antes = len(y)
    audit = {}
    for nome, r in results.items():
        entrada = {"n_before": n_antes, "n_after": len(r.train_index), "strategy": r.strategy}
        if r.sample_weight is not None:
            entrada.update(
                {
                    "weight_min": float(r.sample_weight.min()),
                    "weight_max": float(r.sample_weight.max()),
                    "weight_mean": float(r.sample_weight.mean()),
                    "weight_sum": float(r.sample_weight.sum()),
                }
            )
        audit[nome] = entrada
    return audit


def build_strata(df: pd.DataFrame, kind: str) -> pd.Series:
    """Estrato espaço-temporal de cada registro (§13.2).

    year_region é o primário (55 estratos, sem degeneração). year_municipality
    reproduz a fórmula histórica com o defeito, para a sensibilidade.
    """
    ano = df["NU_ANO"].astype("Int64").astype(str)
    if kind == "year_region":
        return (ano + "_" + df["REGIAO"].astype(str)).rename("stratum")
    if kind == "year_uf":
        return (ano + "_" + df["SIGLA_UF"].astype(str)).rename("stratum")
    if kind == "year_municipality":
        return (ano + "_" + df["ID_MUNIC_ANALISE"].astype(str)).rename("stratum")
    if kind == "year":
        return ano.rename("stratum")
    raise ValueError(f"estrato desconhecido: {kind}")
