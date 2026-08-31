"""Equidade e desempenho por subgrupo (ESPECIFICACAO §3.2).

Fecha TRIPOD+AI 14, 12d, 23a e 23b numa análise só. Não treina nada: lê o OOF já
gravado pelo executor e o recorta por subgrupo, então todo número aqui vem das MESMAS
predições da tabela principal — não de um refit paralelo que poderia divergir dela.

Duas decisões que o resto do módulo assume:

1. **Métrica por classe, um-contra-resto.** Sensibilidade/especificidade/PPV só têm
   sentido binário; com 3 classes, cada uma vira um problema binário. É também o que
   torna comparável o falso negativo de `treatment_interruption` e o de `tb_death`, que
   a §3.2 pede explicitamente.

2. **Supressão pelo denominador de cada métrica, não pelo tamanho do grupo.** Um grupo
   pode ter 3.000 pacientes e só 6 óbitos: a especificidade é estimável, a sensibilidade
   não. Suprimir o grupo inteiro esconderia informação boa; não suprimir publicaria uma
   sensibilidade calculada sobre 6 pessoas. A regra §20.3 do briefing (<30) é aplicada a
   cada denominador separadamente.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from tb_outcomes.calibration import _cal_slope_intercept, expected_calibration_error

logger = logging.getLogger("tb_outcomes")

# Regra de supressão §20.3 do briefing — a mesma da figura 24 (municípios <30 pacientes).
MIN_CELL = 30

# Faixas etárias: cortes de vigilância de TB (pediátrica, adulto jovem, idoso).
AGE_BINS = [0, 15, 25, 35, 45, 60, 65, 200]
AGE_LABELS = ["0-14", "15-24", "25-34", "35-44", "45-59", "60-64", "65+"]

# Eixos de subgrupo da §3.2. O valor é a coluna do X (as seis primeiras) ou uma coluna
# construída à parte (dobra espacial, que vive no OOF e não no X — por regra de vazamento).
AXES = {
    "regiao": "REGIAO",
    "sexo": "CS_SEXO",
    "raca_cor": "CS_RACA",
    "escolaridade": "CS_ESCOL_N",
    "faixa_etaria": "IDADE",
    "vulnerabilidade_municipal": "IPEA_PCT_VULNERAVEIS_POBREZA",
}

# Níveis "Ignorado" — ausência de informação, NÃO uma população.
#
# Compará-los como se fossem um grupo produz um artefato: "pessoas cuja raça/cor não foi
# registrada" não é um estrato social sobre o qual se possa afirmar iniquidade, e o tamanho
# é grande o bastante para o artefato parecer resultado (o nível 9 de CS_ESCOL_N é 21,7% da
# coorte; o de CS_RACA, 5,6%). Vira ausente e o registro sai DAQUELE eixo — continua contando
# em todos os outros.
#
# O código do Ignorado NÃO é o mesmo em todo campo: é 9 nos categóricos numerados e 'I' em
# CS_SEXO. Por isso o mapa é por coluna e explícito, em vez de um `!= 9` global.
IGNORED_LEVELS = {
    "CS_SEXO": {"I", "9"},
    "CS_RACA": {"9"},
    "CS_ESCOL_N": {"9"},
}

# Rótulos do dicionário SINAN NET 5.0 (agravo Tuberculose), fornecidos pelo PI em 2026-08-03.
# O código permanece como identificador do grupo (`grupo`); o rótulo entra numa coluna à
# parte (`grupo_rotulo`), para a tabela seguir auditável contra o dado bruto.
#
# O DBF pode trazer escolaridade com dois dígitos (00, 01, …); a coorte aqui traz sem zero à
# esquerda, então as duas grafias são aceitas.
#
# ATENÇÃO ao 10 de CS_ESCOL_N: "Não se aplica" é preenchido pelo SISTEMA quando a idade é
# menor que 7 anos (e bloqueado a partir dos 7). NÃO é um estrato de escolaridade — é um
# marcador de idade dentro do eixo de escolaridade. Fica no eixo (não define nenhum extremo
# de disparidade nas duas classes), mas o rótulo diz o que ele é, para ninguém lê-lo como
# nível educacional.
_ESCOL = {
    "0": "Analfabeto",
    "1": "1ª a 4ª série incompleta do Fundamental",
    "2": "4ª série completa do Fundamental",
    "3": "5ª a 8ª série incompleta do Fundamental",
    "4": "Fundamental completo",
    "5": "Médio incompleto",
    "6": "Médio completo",
    "7": "Superior incompleto",
    "8": "Superior completo",
    "9": "Ignorado",
    "10": "Não se aplica (idade < 7 anos)",
}
LEVEL_LABELS = {
    "CS_SEXO": {"M": "Masculino", "F": "Feminino", "I": "Ignorado"},
    "CS_RACA": {"1": "Branca", "2": "Preta", "3": "Amarela", "4": "Parda",
                "5": "Indígena", "9": "Ignorado"},
    "CS_ESCOL_N": {**_ESCOL, **{f"{int(k):02d}": v for k, v in _ESCOL.items()}},
}


def label_for(axis: str, grupo) -> str:
    """Rótulo legível de um grupo; devolve o próprio código quando não há dicionário.

    Código fora do dicionário volta marcado em vez de virar string vazia — `CS_RACA` tem um
    registro com código 6, que não existe no dicionário (válidos: 1–5 e 9), e sujeira assim
    tem de aparecer na tabela, não sumir dela.
    """
    col = AXES.get(axis)
    tabela = LEVEL_LABELS.get(col) if col else None
    if not tabela:
        return str(grupo)
    g = str(grupo)
    return tabela.get(g, f"{g} (código fora do dicionário)")


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """IC de Wilson para uma proporção.

    Wilson e não Wald: com sensibilidade perto de 0 ou de 1 — que é o caso das classes
    minoritárias aqui — o Wald produz limites fora de [0,1] e cobertura ruim.
    """
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1.0 + z * z / n
    centro = p + z * z / (2 * n)
    meio = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (float((centro - meio) / d), float((centro + meio) / d))


def binary_rates(y_bin, pred_bin) -> dict:
    """Contagens e taxas de um problema binário (uma classe contra o resto)."""
    y = np.asarray(y_bin).astype(int)
    p = np.asarray(pred_bin).astype(int)
    tp = int(((y == 1) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum())
    fp = int(((y == 0) & (p == 1)).sum())
    tn = int(((y == 0) & (p == 0)).sum())
    pos, neg, pred_pos = tp + fn, tn + fp, tp + fp
    return {
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "n_positivos": pos, "n_negativos": neg, "n_preditos_positivos": pred_pos,
        "sensibilidade": tp / pos if pos else float("nan"),
        "fn_rate": fn / pos if pos else float("nan"),
        "especificidade": tn / neg if neg else float("nan"),
        "ppv": tp / pred_pos if pred_pos else float("nan"),
    }


def age_bands(idade: pd.Series) -> pd.Series:
    """Faixa etária ordenada. Idade ausente permanece ausente (não vira faixa)."""
    return pd.cut(pd.to_numeric(idade, errors="coerce"), bins=AGE_BINS,
                  labels=AGE_LABELS, right=False, ordered=True)


def quantile_bands(x: pd.Series, n: int = 5, prefix: str = "Q") -> pd.Series:
    """Divide uma contínua municipal em n faixas por quantil, rotuladas e ordenadas."""
    v = pd.to_numeric(x, errors="coerce")
    rotulos = [f"{prefix}{i + 1}" for i in range(n)]
    return pd.qcut(v, q=n, labels=rotulos, duplicates="drop")


def _auc(y_bin, proba) -> float:
    """AUC do grupo — discriminação INDEPENDENTE de prevalência.

    Existe para separar duas coisas que a sensibilidade mistura: a capacidade de ORDENAR
    risco dentro do grupo e o ponto de operação (o limiar, que é global). Sem ela, um grupo
    de baixa prevalência aparece com falso negativo alto e parece vítima de iniquidade
    quando o modelo, na verdade, o ordena tão bem ou melhor que os demais — foi exatamente
    o que acontecia no eixo de escolaridade (investigação de 2026-08-03).
    """
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y_bin).astype(int)
    if y.min() == y.max():  # grupo sem uma das classes: AUC indefinida
        return float("nan")
    try:
        return float(roc_auc_score(y, np.asarray(proba, dtype=float)))
    except Exception:  # noqa: BLE001
        return float("nan")


def _cal_pair(y_bin, proba) -> tuple[float, float]:
    y = np.asarray(y_bin).astype(int)
    p = np.asarray(proba, dtype=float)
    try:
        ece = float(expected_calibration_error(y, p, n_bins=10))
    except Exception:  # noqa: BLE001 — grupo degenerado não derruba a tabela inteira
        ece = float("nan")
    try:
        slope, _ = _cal_slope_intercept(y, p)
    except Exception:  # noqa: BLE001
        slope = float("nan")
    return ece, float(slope)


def subgroup_metrics(df: pd.DataFrame, axis: str, group_col: str, y_col: str,
                     pred_col: str, proba_col: str, min_cell: int = MIN_CELL,
                     extra: dict | None = None) -> pd.DataFrame:
    """Uma linha por grupo do eixo, com métricas binárias, IC de Wilson e supressão.

    `df` já deve estar restrito a UMA classe (y binarizado) e a um par modelo×estratégia.
    """
    linhas = []
    d = df[df[group_col].notna()]
    for grupo, g in d.groupby(group_col, observed=True, sort=True):
        r = binary_rates(g[y_col], g[pred_col])
        ece, slope = _cal_pair(g[y_col], g[proba_col])

        # supressão POR DENOMINADOR (ver docstring do módulo)
        pos_ok = r["n_positivos"] >= min_cell
        neg_ok = r["n_negativos"] >= min_cell
        ppv_ok = r["n_preditos_positivos"] >= min_cell
        suprimido = not (pos_ok and neg_ok and ppv_ok)

        sens_lo, sens_hi = wilson_ci(r["tp"], r["n_positivos"]) if pos_ok else (np.nan, np.nan)
        esp_lo, esp_hi = wilson_ci(r["tn"], r["n_negativos"]) if neg_ok else (np.nan, np.nan)
        ppv_lo, ppv_hi = wilson_ci(r["tp"], r["n_preditos_positivos"]) if ppv_ok else (np.nan, np.nan)
        fn_lo, fn_hi = wilson_ci(r["fn"], r["n_positivos"]) if pos_ok else (np.nan, np.nan)

        linhas.append({
            "eixo": axis, "grupo": str(grupo), "grupo_rotulo": label_for(axis, grupo),
            "n": int(len(g)),
            "n_positivos": r["n_positivos"], "n_negativos": r["n_negativos"],
            "n_preditos_positivos": r["n_preditos_positivos"],
            "sensibilidade": r["sensibilidade"] if pos_ok else np.nan,
            "sensibilidade_lo": sens_lo, "sensibilidade_hi": sens_hi,
            "fn_rate": r["fn_rate"] if pos_ok else np.nan,
            "fn_rate_lo": fn_lo, "fn_rate_hi": fn_hi,
            "especificidade": r["especificidade"] if neg_ok else np.nan,
            "especificidade_lo": esp_lo, "especificidade_hi": esp_hi,
            "ppv": r["ppv"] if ppv_ok else np.nan,
            "ppv_lo": ppv_lo, "ppv_hi": ppv_hi,
            "prevalencia": r["n_positivos"] / len(g) if len(g) else np.nan,
            "auc": _auc(g[y_col], g[proba_col]) if (pos_ok and neg_ok) else np.nan,
            "ece": ece if pos_ok else np.nan,
            "calibration_slope": slope if pos_ok else np.nan,
            "suprimido": bool(suprimido),
            "min_cell": int(min_cell),
            **(extra or {}),
        })
    return pd.DataFrame(linhas)


def build_subgroup_frame(X: pd.DataFrame, outer_fold: pd.Series) -> pd.DataFrame:
    """Monta as colunas de subgrupo a partir do X e da dobra espacial externa.

    A dobra vem de fora do X de propósito: `ID_MN_RESI` e o cluster são proibidos como
    feature (regra de vazamento §1.1/§7), mas a heterogeneidade territorial que a §3.2
    pede é justamente por dobra espacial.
    """
    out = pd.DataFrame(index=X.index)
    for eixo, col in AXES.items():
        if col not in X.columns:
            continue
        if eixo == "faixa_etaria":
            out[eixo] = age_bands(X[col])
        elif eixo == "vulnerabilidade_municipal":
            out[eixo] = quantile_bands(X[col], n=5, prefix="Q")
        else:
            s = X[col].astype("string")
            ignorados = IGNORED_LEVELS.get(col)
            if ignorados:
                n_antes = int(s.notna().sum())
                s = s.where(~s.isin(ignorados), other=pd.NA)
                perdidos = n_antes - int(s.notna().sum())
                if perdidos:
                    logger.info(
                        "Eixo '%s': %d registros (%.1f%%) com nível Ignorado %s tratados como "
                        "ausentes — saem deste eixo, seguem nos demais.",
                        eixo, perdidos, 100 * perdidos / max(n_antes, 1), sorted(ignorados))
            out[eixo] = s.where(s.notna(), other=pd.NA)
    out["dobra_espacial"] = pd.Series(np.asarray(outer_fold), index=X.index).astype("string")
    return out


def equity_table(oof_pair: pd.DataFrame, subgroups: pd.DataFrame, classes,
                 class_names, rule: str = "pred_policy",
                 min_cell: int = MIN_CELL) -> pd.DataFrame:
    """Tabela longa de equidade: eixo × grupo × classe, para uma regra de decisão.

    `oof_pair` é o OOF de UM par modelo×estratégia, com `record_pos` alinhado às linhas
    de `subgroups`. `rule` escolhe entre `pred_policy` (regra operacional, com limiares)
    e `pred_argmax`.
    """
    pos = oof_pair["record_pos"].to_numpy()
    sub = subgroups.iloc[pos].reset_index(drop=True)
    base = oof_pair.reset_index(drop=True)

    quadros = []
    for j, c in enumerate(classes):
        nome = class_names[j]
        d = pd.DataFrame({
            "y": (base["y_true"].to_numpy() == c).astype(int),
            "pred": (base[rule].to_numpy() == c).astype(int),
            "proba": base[f"proba_{c}"].to_numpy(),
        })
        d = pd.concat([d, sub], axis=1)
        for eixo in list(AXES) + ["dobra_espacial"]:
            if eixo not in d.columns:
                continue
            quadros.append(subgroup_metrics(
                d, axis=eixo, group_col=eixo, y_col="y", pred_col="pred",
                proba_col="proba", min_cell=min_cell,
                extra={"classe": nome, "regra": rule}))
    if not quadros:
        return pd.DataFrame()
    return pd.concat(quadros, ignore_index=True)


def disparity_summary(tab: pd.DataFrame, metric: str = "fn_rate") -> pd.DataFrame:
    """Amplitude da métrica entre grupos de cada eixo — o resumo que vai ao manuscrito.

    Só considera grupos NÃO suprimidos: a razão máx/mín perde o sentido se um dos
    extremos foi estimado sobre uma célula pequena.
    """
    d = tab[~tab["suprimido"] & tab[metric].notna()]
    linhas = []
    for (classe, regra, eixo), g in d.groupby(["classe", "regra", "eixo"], sort=True):
        if len(g) < 2:
            continue
        i_max, i_min = g[metric].idxmax(), g[metric].idxmin()
        v_max, v_min = g.loc[i_max, metric], g.loc[i_min, metric]
        linhas.append({
            "classe": classe, "regra": regra, "eixo": eixo, "metrica": metric,
            "n_grupos": int(len(g)),
            "grupo_pior": g.loc[i_max, "grupo"],
            "grupo_pior_rotulo": label_for(eixo, g.loc[i_max, "grupo"]),
            "valor_pior": float(v_max),
            "grupo_melhor": g.loc[i_min, "grupo"],
            "grupo_melhor_rotulo": label_for(eixo, g.loc[i_min, "grupo"]),
            "valor_melhor": float(v_min),
            "amplitude_absoluta": float(v_max - v_min),
            "razao": float(v_max / v_min) if v_min > 0 else float("nan"),
        })
    return pd.DataFrame(linhas).sort_values(
        ["classe", "regra", "amplitude_absoluta"], ascending=[True, True, False]
    ).reset_index(drop=True)
