"""Utilidade programática (ESPECIFICACAO §3.8) — fecha R2 #7.

Discriminação e calibração dizem que o modelo ordena risco. Não dizem se um programa de
controle da tuberculose conseguiria USAR essa ordenação. Este módulo responde à pergunta
operacional: se a equipe priorizar os pacientes de maior risco para busca ativa, quantos
alertas recebe, quanto do que visita é evento de verdade, quanto dos eventos alcança, e se
isso ganha de "intensificar para todos" ou "não fazer nada".

Duas escolhas que o resto assume:

1. **Um problema binário por classe.** O programa monta uma lista de prioridade por desfecho
   (quem pode morrer, quem pode interromper), não um rótulo de três níveis. Então tudo aqui
   é um-contra-resto sobre a probabilidade calibrada da classe.

2. **Top-k por capacidade, não por limiar.** O serviço não escolhe um ponto de corte de
   probabilidade; ele tem equipe para visitar X% dos casos. Por isso o eixo primário é a
   fração da coorte sinalizada, e o limiar entra só na curva de decisão.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Frações de capacidade reportadas por padrão (§3.8: top 5%, 10%, 20%).
DEFAULT_KS = (0.05, 0.10, 0.20)


def alerts_per_1000(pred_bin) -> float:
    """Alertas por 1.000 pacientes — a carga de trabalho que o programa assume."""
    p = np.asarray(pred_bin)
    return float(1000.0 * p.sum() / len(p)) if len(p) else float("nan")


def topk_table(y_bin, proba, ks=DEFAULT_KS) -> pd.DataFrame:
    """PPV, captura de eventos e NNS em cada fração de capacidade.

    `ppv` responde "das pessoas que eu visitar, quantas realmente têm o desfecho";
    `captura` responde "de todos os desfechos, quantos estão na minha lista";
    `nns` (number needed to screen) é 1/PPV — quantos visitar para encontrar um evento.
    Empates na probabilidade são resolvidos pela ordem estável do argsort: o corte é por
    posição, como seria uma fila de prioridade real.
    """
    y = np.asarray(y_bin).astype(int)
    p = np.asarray(proba, dtype=float)
    n, n_pos = len(y), int(y.sum())
    ordem = np.argsort(-p, kind="stable")
    linhas = []
    for k in ks:
        m = int(round(k * n))
        if m <= 0:
            continue
        topo = ordem[:m]
        tp = int(y[topo].sum())
        ppv = tp / m
        linhas.append({
            "k": k, "n_sinalizados": m, "alertas_por_1000": 1000.0 * m / n,
            "eventos_no_topo": tp, "ppv": ppv,
            "captura": tp / n_pos if n_pos else float("nan"),
            "nns": 1.0 / ppv if ppv > 0 else float("inf"),
            "prevalencia": n_pos / n,
            "lift": (ppv / (n_pos / n)) if n_pos else float("nan"),
        })
    return pd.DataFrame(linhas)


def net_benefit(y_bin, proba, threshold: float) -> float:
    """Benefício líquido de Vickers–Elkin no limiar `threshold`.

        NB = TP/n − (FP/n) × (pt / (1 − pt))

    `pt` é a taxa de troca implícita: a probabilidade a partir da qual intervir passa a
    compensar. O termo `pt/(1−pt)` é quantos falsos positivos o programa aceita pagar por
    um verdadeiro positivo — em pt=0,10, nove falsos alarmes valem um evento evitado.
    """
    y = np.asarray(y_bin).astype(int)
    p = np.asarray(proba, dtype=float)
    n = len(y)
    if n == 0 or not (0.0 < threshold < 1.0):
        return float("nan")
    sinalizado = p >= threshold
    tp = int(((sinalizado) & (y == 1)).sum())
    fp = int(((sinalizado) & (y == 0)).sum())
    return float(tp / n - (fp / n) * (threshold / (1.0 - threshold)))


def decision_curve(y_bin, proba, thresholds=None) -> pd.DataFrame:
    """Curva de decisão: modelo contra as duas referências que o programa já tem.

    `intervir em todos` e `não intervir em ninguém` não são formalidades — são as duas
    políticas realmente disponíveis hoje. Se a curva do modelo não estiver acima das duas
    na faixa de limiares plausível, ele não acrescenta nada a elas, por melhor que seja a
    AUC.
    """
    y = np.asarray(y_bin).astype(int)
    p = np.asarray(proba, dtype=float)
    if thresholds is None:
        thresholds = np.round(np.arange(0.01, 0.51, 0.01), 4)
    thresholds = np.asarray(thresholds, dtype=float)
    n = len(y)
    todos = np.ones(n, dtype=float)
    linhas = []
    for pt in thresholds:
        n_alertas = int((p >= pt).sum())
        linhas.append({
            "threshold": float(pt),
            "nb_modelo": net_benefit(y, p, pt),
            "nb_intervir_em_todos": net_benefit(y, todos, pt),
            "nb_nao_intervir": 0.0,
            "n_alertas": n_alertas,
            "alertas_por_1000": 1000.0 * n_alertas / n if n else float("nan"),
        })
    return pd.DataFrame(linhas)


def useful_range(curva: pd.DataFrame, folga: float = 0.0) -> dict:
    """Faixa de limiares em que o modelo bate AS DUAS referências.

    É o resumo que decide a §3.8: se esta faixa for vazia, não há utilidade programática a
    reivindicar, e o briefing proíbe reivindicá-la.
    """
    melhor = ((curva["nb_modelo"] > curva["nb_intervir_em_todos"] + folga)
              & (curva["nb_modelo"] > curva["nb_nao_intervir"] + folga))
    if not melhor.any():
        return {"tem_faixa_util": False, "threshold_min": float("nan"),
                "threshold_max": float("nan"), "n_limiares": 0}
    d = curva.loc[melhor, "threshold"]
    return {"tem_faixa_util": True, "threshold_min": float(d.min()),
            "threshold_max": float(d.max()), "n_limiares": int(melhor.sum())}
