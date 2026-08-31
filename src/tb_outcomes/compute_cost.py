"""Tempo e custo computacional por modelo (ESPECIFICACAO §6, linha "Tempo e custo").

Fecha R3 #2 junto com o ambiente travado e as sementes: um leitor precisa saber o que
custa reproduzir cada linha do leaderboard, e quais pares são caros o bastante para
inviabilizar uma réplica casual.

A fonte é o log de execução, não uma instrumentação nova. Duas razões: as corridas já
aconteceram (reexecutar para cronometrar custaria semanas de GPU), e o log é o registro
primário — cronometrar por fora abriria espaço para divergir dele.

Dois formatos convivem nos logs e ambos são aceitos:
  1. `fim <modelo> × <estratégia>: ok em <N>s`  — o executor escreve o tempo direto;
  2. par `início`/`fim` com timestamps — usado quando a linha não traz o tempo.
O formato 1 tem precedência; o 2 é a reserva. Quando os dois existem, o valor do log
manda, e a diferença fica registrada na coluna `fonte`.
"""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import pandas as pd

# 2026-08-01 20:50:00,949 [INFO] [benchmark] 1/15 fim lightgbm × random_undersampling: ok em 130.4s
_FIM_COM_TEMPO = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ .*?"
    r"\d+/\d+ fim (?P<model>\S+) × (?P<strategy>\S+): (?P<status>\S+) em (?P<segundos>[\d.]+)s"
)
# 2026-08-03 22:55:38,122 [INFO] [temporal] 1/33 início majority_class × random_undersampling
_INICIO = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ .*?"
    r"\d+/\d+ início (?P<model>\S+) × (?P<strategy>\S+)\s*$"
)
# ... fim majority_class × random_undersampling: ok        (sem tempo)
_FIM_SEM_TEMPO = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ .*?"
    r"\d+/\d+ fim (?P<model>\S+) × (?P<strategy>\S+): (?P<status>[^\s:]+)\s*$"
)

_FMT = "%Y-%m-%d %H:%M:%S"


def parse_log(path, arm: str) -> pd.DataFrame:
    """Extrai (modelo, estratégia, segundos) de um log de execução."""
    linhas: list[dict] = []
    abertos: dict[tuple[str, str], datetime] = {}
    with Path(path).open("r", encoding="utf-8", errors="replace") as fh:
        for linha in fh:
            m = _FIM_COM_TEMPO.match(linha)
            if m:
                linhas.append({
                    "arm": arm, "model": m["model"], "strategy": m["strategy"],
                    "status": m["status"], "segundos": float(m["segundos"]),
                    "fim": m["ts"], "fonte": "log_explicito",
                })
                abertos.pop((m["model"], m["strategy"]), None)
                continue
            m = _INICIO.match(linha)
            if m:
                abertos[(m["model"], m["strategy"])] = datetime.strptime(m["ts"], _FMT)
                continue
            m = _FIM_SEM_TEMPO.match(linha)
            if m:
                t0 = abertos.pop((m["model"], m["strategy"]), None)
                if t0 is None:
                    continue
                dt = (datetime.strptime(m["ts"], _FMT) - t0).total_seconds()
                linhas.append({
                    "arm": arm, "model": m["model"], "strategy": m["strategy"],
                    "status": m["status"], "segundos": float(dt),
                    "fim": m["ts"], "fonte": "diferenca_de_timestamp",
                })
    return pd.DataFrame(linhas)


def collect(sources: dict) -> pd.DataFrame:
    """Junta vários logs; `sources` mapeia rótulo do braço -> caminho do log."""
    quadros = [parse_log(p, arm) for arm, p in sources.items() if Path(p).exists()]
    quadros = [q for q in quadros if len(q)]
    if not quadros:
        return pd.DataFrame(columns=["arm", "model", "strategy", "status", "segundos",
                                     "fim", "fonte"])
    d = pd.concat(quadros, ignore_index=True)
    # um par pode reaparecer (retomada após queda de energia, backfill): fica o ÚLTIMO,
    # que é a execução que produziu o artefato vigente.
    d = d.sort_values("fim").drop_duplicates(["arm", "model", "strategy"], keep="last")
    d["minutos"] = (d["segundos"] / 60).round(2)
    d["horas"] = (d["segundos"] / 3600).round(3)
    return d.sort_values(["arm", "segundos"], ascending=[True, False]).reset_index(drop=True)


def summarize_by_model(d: pd.DataFrame) -> pd.DataFrame:
    """Custo agregado por modelo — a tabela que responde 'o que é caro aqui?'."""
    if d.empty:
        return pd.DataFrame()
    g = (d.groupby("model")
           .agg(n_pares=("segundos", "size"),
                total_horas=("segundos", lambda s: round(s.sum() / 3600, 3)),
                media_minutos=("segundos", lambda s: round(s.mean() / 60, 2)),
                min_minutos=("segundos", lambda s: round(s.min() / 60, 2)),
                max_minutos=("segundos", lambda s: round(s.max() / 60, 2)))
           .reset_index()
           .sort_values("total_horas", ascending=False))
    total = g["total_horas"].sum()
    g["pct_do_total"] = (100 * g["total_horas"] / total).round(1) if total else float("nan")
    return g.reset_index(drop=True)
