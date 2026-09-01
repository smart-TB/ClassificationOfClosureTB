"""Injeta no manuscrito em português as tabelas geradas dos artefatos.

Mesmo mecanismo de marcador da versão em inglês, com rótulos em pt-BR e vírgula decimal.
Nenhum número é digitado: o valor vem do mesmo CSV que alimenta a versão em inglês, de modo
que as duas versões não podem divergir.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from tb_outcomes import labels as rot
from tb_outcomes.robustness import leaderboard

D = Path("data")
TEX = Path("manuscrito/PLOS_DigitalHealth/manuscrito_ptbr.tex")
CLS = {0: "Cura", 1: "Óbito por TB", 2: "Interrupção do tratamento"}


def _n(v, casas: int = 3) -> str:
    """Número no formato brasileiro: vírgula decimal, ponto de milhar."""
    if isinstance(v, float):
        return f"{v:.{casas}f}".replace(".", ",")
    return f"{v:,}".replace(",", ".")


def _esc(s) -> str:
    return str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def _wrap(nome: str, corpo: str) -> str:
    return f"% <<TABELAPT:{nome}>>\n{corpo.rstrip()}\n% <<FIMPT:{nome}>>"


def t_byclass() -> str:
    lb = leaderboard(pd.read_csv(D / "benchmark_metrics.csv")).sort_values(
        "f1_macro", ascending=False).iloc[0]
    m = pd.read_csv(D / "benchmark_metrics.csv")
    a = m[(m.kind == "aggregate") & (m.model == lb["model"])
          & (m.strategy == lb["best_strategy"])]

    def v(metric, klass, stage=None):
        g = a[(a.metric == metric) & (a["class"].astype(str) == str(klass))]
        if stage:
            g = g[g.stage == stage]
        return float(g["mean"].iloc[0]) if len(g) else float("nan")

    linhas = [r"\begin{table}[!ht]", r"\centering",
              r"\caption{Desempenho por classe de desfecho no modelo selecionado. LightGBM "
              r"com reponderação sensível ao custo. O erro esperado de calibração é "
              r"apresentado antes e depois do calibrador ajustado por validação cruzada.}",
              r"\label{tab:classe}", r"\begin{tabular}{lccccc}", r"\toprule",
              r"Classe & Precisão & Sensibilidade & AUPRC & ECE antes & ECE depois \\",
              r"\midrule"]
    for c in (0, 1, 2):
        linhas.append(
            f"{CLS[c]} & {_n(v('precision', c))} & {_n(v('recall', c))} & "
            f"{_n(v('auprc', c))} & {_n(v('ece_fixed', c, 'pre_renorm'))} & "
            f"{_n(v('ece_fixed', c, 'post_renorm'))} " + r"\\")
    linhas += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return _wrap("byclass", "\n".join(linhas))


def t_temporal() -> str:
    p = D / "temporal" / "temporal_vs_spatial.csv"
    if not p.exists():
        return _wrap("temporal", "% artefato ausente")
    t = pd.read_csv(p)
    col = [c for c in t.columns if c.startswith("temporal_")][0]
    ano = col.split("_")[1]
    linhas = [r"\begin{table}[!ht]", r"\centering",
              r"\caption{Validação espacial e temporal comparadas. F1 macro sob validação "
              r"aninhada e espacialmente bloqueada e no ano reservado de " + ano +
              r", retido antes de qualquer seleção de modelo.}",
              r"\label{tab:temporal}", r"\begin{tabular}{lccc}", r"\toprule",
              r"Algoritmo & Espacial & Temporal (" + ano + r") & Retido (\%) \\",
              r"\midrule"]
    for r in t.itertuples():
        linhas.append(f"{_esc(rot.algoritmo(r.model))} & {_n(r.espacial_k50, 4)} & "
                      f"{_n(getattr(r, col), 4)} & {_n(r.pct_retido, 1)} " + r"\\")
    linhas += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return _wrap("temporal", "\n".join(linhas))


def t_utility() -> str:
    p = D / "utility_topk.csv"
    if not p.exists():
        return _wrap("utility", "% artefato ausente")
    u = pd.read_csv(p)
    arm = {"modelo_completo": "Modelo", "baseline_clinico": "Regra clínica"}
    linhas = [r"\begin{table}[!ht]", r"\centering",
              r"\caption{Utilidade programática sob restrição de capacidade. Valor "
              r"preditivo positivo, fração de eventos capturada e número necessário para "
              r"rastrear quando se prioriza a fração de maior risco das notificações. A "
              r"regra clínica usa idade, situação sorológica para HIV, uso de álcool ou "
              r"drogas e situação de rua.}",
              r"\label{tab:utilidade}", r"\begin{tabular}{llcccc}", r"\toprule",
              r"Desfecho & Abordagem & Capacidade & VPP & Captura & NNR \\", r"\midrule"]
    for classe in ("tb_death", "treatment_interruption"):
        for k in sorted(u.k.unique()):
            for a in ("modelo_completo", "baseline_clinico"):
                g = u[(u.classe == classe) & (u.k == k) & (u.arm == a)]
                if g.empty:
                    continue
                r = g.iloc[0]
                cap = f"{round(k * 100)}" + r"\%"
                linhas.append(f"{rot.desfecho_pt(classe)} & {arm[a]} & {cap} & "
                              f"{_n(r.ppv)} & {_n(r.captura)} & {_n(r.nns, 2)} " + r"\\")
        linhas.append(r"\midrule")
    linhas[-1] = r"\bottomrule"
    linhas += [r"\end{tabular}", r"\end{table}"]
    return _wrap("utility", "\n".join(linhas))


GERADORES = {"byclass": t_byclass, "temporal": t_temporal, "utility": t_utility}


def main() -> None:
    tex = TEX.read_text(encoding="utf-8")
    feitas = []
    for nome, fn in GERADORES.items():
        marcador = re.compile(rf"% <<TABELAPT:{nome}>>.*?% <<FIMPT:{nome}>>", re.S)
        if not marcador.search(tex):
            continue
        tex = marcador.sub(lambda _: fn(), tex)
        feitas.append(nome)
    TEX.write_text(tex, encoding="utf-8")
    print(f"tabelas pt-BR injetadas: {feitas}")


if __name__ == "__main__":
    main()
