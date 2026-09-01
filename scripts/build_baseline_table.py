"""Características basais da coorte analítica, por desfecho (S1 Table).

Gera `data/baseline_characteristics.csv` a partir da coorte real. Categóricas entram com
n e percentual dentro do desfecho; contínuas, com mediana e intervalo interquartil. O
percentual tem como denominador os NÃO ausentes da variável, e a ausência é reportada em
linha própria, para não inflar categoria à custa de dado que não existe.

Rótulos vêm do dicionário SINAN quando disponível (`equity.LEVEL_LABELS`); do contrário o
código é mantido, nunca inventado.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.WARNING)

from tb_outcomes.cli import EXECUTOR, _load_benchmark_inputs  # noqa: E402
from tb_outcomes.equity import LEVEL_LABELS, age_bands  # noqa: E402
from tb_outcomes.executor import load_executor_config  # noqa: E402

CLASSES = {0: "Cure", 1: "TB death", 2: "Treatment interruption"}

CATEGORICAS = [
    ("CS_SEXO", "Sex"),
    ("CS_RACA", "Race/colour"),
    ("CS_ESCOL_N", "Schooling"),
    ("REGIAO", "Region"),
    ("FORMA", "Clinical form"),
    ("HIV", "HIV status"),
    ("AGRAVALCOO", "Alcohol use"),
    ("AGRAVDROGA", "Illicit drug use"),
    ("AGRAVDIABE", "Diabetes"),
    ("POP_RUA", "Homelessness"),
    ("POP_LIBER", "Deprivation of liberty"),
    ("RAIOX_TORA", "Chest radiography"),
]
CONTINUAS = [("IDADE", "Age (years)")]


def main() -> None:
    cfg = load_executor_config(EXECUTOR)
    X, y, *_ = _load_benchmark_inputs(cfg, False)
    y = pd.Series(y).reset_index(drop=True)

    linhas: list[dict] = []

    def add(var, nivel, contagens, denominadores):
        linha = {"variable": var, "level": nivel}
        for c, nome in CLASSES.items():
            n = int(contagens.get(c, 0))
            den = int(denominadores.get(c, 0))
            linha[f"{nome} n"] = n
            linha[f"{nome} %"] = round(100 * n / den, 2) if den else np.nan
        linha["Total n"] = int(sum(contagens.get(c, 0) for c in CLASSES))
        linhas.append(linha)

    # n por desfecho, para o cabeçalho
    tot = {c: int((y == c).sum()) for c in CLASSES}
    add("N", "", tot, {c: int(len(y)) for c in CLASSES})

    for col, rotulo in CONTINUAS:
        if col not in X.columns:
            continue
        v = pd.to_numeric(X[col], errors="coerce")
        linha = {"variable": rotulo, "level": "median [IQR]"}
        for c, nome in CLASSES.items():
            s = v[y == c].dropna()
            if len(s):
                linha[f"{nome} n"] = f"{s.median():.0f} [{s.quantile(.25):.0f}-{s.quantile(.75):.0f}]"
            else:
                linha[f"{nome} n"] = ""
            linha[f"{nome} %"] = np.nan
        linha["Total n"] = int(v.notna().sum())
        linhas.append(linha)
        # faixa etária, que é como a análise de equidade estratifica
        faixa = age_bands(v).astype("string")
        den = {c: int(faixa[y == c].notna().sum()) for c in CLASSES}
        for nivel in [x for x in faixa.dropna().unique()]:
            m = faixa == nivel
            add("Age band", str(nivel), {c: int((m & (y == c)).sum()) for c in CLASSES}, den)

    for col, rotulo in CATEGORICAS:
        if col not in X.columns:
            continue
        s = X[col].astype("string")
        rot = LEVEL_LABELS.get(col, {})
        den = {c: int(s[y == c].notna().sum()) for c in CLASSES}
        for nivel in sorted(s.dropna().unique()):
            m = s == nivel
            add(rotulo, rot.get(str(nivel), str(nivel)),
                {c: int((m & (y == c)).sum()) for c in CLASSES}, den)
        falt = s.isna()
        if falt.any():
            add(rotulo, "Missing", {c: int((falt & (y == c)).sum()) for c in CLASSES},
                {c: tot[c] for c in CLASSES})

    d = pd.DataFrame(linhas)
    out = "data/baseline_characteristics.csv"
    d.to_csv(out, index=False)
    print(f"gravado: {out} ({len(d)} linhas)")


if __name__ == "__main__":
    main()
