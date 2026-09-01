"""Gera as tabelas do manuscrito a partir dos artefatos e as injeta no .tex.

Duas restrições se cruzam aqui. O invariante 9 do protocolo proíbe copiar número à mão.
A PLOS exige que o fonte LaTeX seja um arquivo único, sem `\\input`. A saída é injeção por
marcador: o script substitui o bloco entre `% <<TABELA:nome>>` e `% <<FIM:nome>>` no próprio
manuscript.tex, de modo que o arquivo continua único e os números continuam gerados.

Idempotente: rodar duas vezes produz o mesmo arquivo.
"""
from __future__ import annotations

import re
from pathlib import Path

import sys

import pandas as pd

sys.path.insert(0, "scripts")
import manuscript_labels as rot
from tb_outcomes.robustness import leaderboard

D = Path("data")
TEX = Path("manuscrito/PLOS_DigitalHealth/manuscript.tex")
CLS = {0: "Cure", 1: "TB death", 2: "Treatment interruption"}


def _esc(s) -> str:
    return str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def _wrap(nome: str, corpo: str) -> str:
    return f"% <<TABELA:{nome}>>\n{corpo.rstrip()}\n% <<FIM:{nome}>>"


def t_cohort() -> str:
    fl = pd.read_csv(D / "cohort_flow_sensitive_tb_3class.csv")
    ex = pd.read_csv(D / "exclusion_reasons_sensitive_tb_3class.csv")
    motivo = {"municipio_irrecuperavel": "Unrecoverable municipality code",
           "idade_impossivel": "Impossible age",
           "fora_da_janela_declarada": "Outside the declared window",
           "seguimento_insuficiente": "Follow-up shorter than 365 days"}
    desc = {"recuperada": "Records returned by the acquisition pipeline",
            "declarada": "Notified between 2015 and 2025",
            "elegivel": "Eligible after quality and follow-up criteria",
            "analitica": "Analytic cohort with a three-class outcome"}
    linhas = [r"\begin{table}[!ht]", r"\caption{{\bf Cohort definition and exclusions.}}",
              r"\label{tab:cohort}", r"\begin{tabular}{lr}", r"\hline",
              r"Stage & n \\ \hline"]
    for r in fl.itertuples():
        linhas.append(f"{desc.get(r.stage, _esc(r.stage))} & {r.n:,} " + r"\\")
    linhas.append(r"\hline")
    linhas.append(r"\multicolumn{2}{l}{\emph{Exclusions applied sequentially}} \\")
    for r in ex.itertuples():
        linhas.append(f"\\quad {motivo.get(r.reason, _esc(r.reason))} & {r.n:,} " + r"\\")
    linhas += [r"\hline", r"\end{tabular}", r"\end{table}"]
    return _wrap("cohort", "\n".join(linhas))


def t_leaderboard() -> str:
    lb = leaderboard(pd.read_csv(D / "benchmark_metrics.csv")).sort_values(
        "f1_macro", ascending=False).reset_index(drop=True)
    est = {"local_cost_sensitive": "Cost-sensitive", "random_oversampling": "Oversampling",
           "random_undersampling": "Undersampling"}
    linhas = [r"\begin{table}[!ht]",
              r"\caption{{\bf Model ranking under nested spatially blocked validation.} "
              r"Macro-averaged F1 for the best-performing imbalance strategy of each "
              r"algorithm, at 50 municipality clusters.}",
              r"\label{tab:leaderboard}", r"\begin{tabular}{rlcr}", r"\hline",
              r"Rank & Algorithm & Best strategy & Macro F1 \\ \hline"]
    for i, r in enumerate(lb.itertuples(), 1):
        linhas.append(f"{i} & {_esc(rot.algoritmo(r.model))} & {est.get(r.best_strategy, _esc(r.best_strategy))} "
                      f"& {r.f1_macro:.4f} " + r"\\")
    linhas += [r"\hline", r"\end{tabular}", r"\end{table}"]
    return _wrap("leaderboard", "\n".join(linhas))


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

    linhas = [r"\begin{table}[!ht]",
              r"\caption{{\bf Performance by outcome class for the selected model.} "
              r"LightGBM with cost-sensitive reweighting. Expected calibration error is "
              r"shown before and after the cross-fitted calibrator.}",
              r"\label{tab:byclass}", r"\begin{tabular}{lccccc}", r"\hline",
              r"Class & Precision & Recall & AUPRC & ECE before & ECE after \\ \hline"]
    for c in (0, 1, 2):
        linhas.append(
            f"{CLS[c]} & {v('precision', c):.3f} & {v('recall', c):.3f} & "
            f"{v('auprc', c):.3f} & {v('ece_fixed', c, 'pre_renorm'):.3f} & "
            f"{v('ece_fixed', c, 'post_renorm'):.3f} " + r"\\")
    linhas += [r"\hline", r"\end{tabular}", r"\end{table}"]
    return _wrap("byclass", "\n".join(linhas))


def t_temporal() -> str:
    p = D / "temporal" / "temporal_vs_spatial.csv"
    if not p.exists():
        return _wrap("temporal", "% artefato ausente")
    t = pd.read_csv(p)
    col = [c for c in t.columns if c.startswith("temporal_")][0]
    ano = col.split("_")[1]
    linhas = [r"\begin{table}[!ht]",
              r"\caption{{\bf Spatial and temporal validation compared.} Macro-averaged F1 "
              r"under nested spatially blocked validation and in the reserved year " + ano +
              r", which was withheld before any model was selected.}",
              r"\label{tab:temporal}", r"\begin{tabular}{lccc}", r"\hline",
              r"Algorithm & Spatial & Temporal (" + ano + r") & Retained (\%) \\ \hline"]
    for r in t.itertuples():
        linhas.append(f"{_esc(rot.algoritmo(r.model))} & {r.espacial_k50:.4f} & {getattr(r, col):.4f} & "
                      f"{r.pct_retido:.1f} " + r"\\")
    linhas += [r"\hline", r"\end{tabular}", r"\end{table}"]
    return _wrap("temporal", "\n".join(linhas))


def t_ablation() -> str:
    p = D / "ablation" / "ablation_leaderboard.csv"
    if not p.exists():
        return _wrap("ablation", "% artefato ausente")
    ab = pd.read_csv(p)
    lb = leaderboard(pd.read_csv(D / "benchmark_metrics.csv"),
                     models=["majority_class", "stratified_random"])
    piso = float(lb["f1_macro"].max())
    linhas = [r"\begin{table}[!ht]",
              r"\caption{{\bf Territorial ablation.} Macro-averaged F1 with individual "
              r"variables only, municipal contextual variables only, and both. The "
              r"majority-class floor is " + f"{piso:.4f}" + r".}",
              r"\label{tab:ablation}", r"\begin{tabular}{lccc}", r"\hline",
              r"Algorithm & Individual & Municipal & Combined \\ \hline"]
    for r in ab.itertuples():
        linhas.append(f"{_esc(rot.algoritmo(r.model))} & {r.individual:.4f} & {r.municipal:.4f} & "
                      f"{r.combined:.4f} " + r"\\")
    linhas += [r"\hline", r"\end{tabular}", r"\end{table}"]
    return _wrap("ablation", "\n".join(linhas))


def t_equity() -> str:
    p = D / "equity_discrimination_by_group.csv"
    if not p.exists():
        return _wrap("equity", "% artefato ausente")
    d = pd.read_csv(p)
    d = d[(d.eixo == "raca_cor") & (d.classe == "tb_death")].sort_values(
        "auc", ascending=False)
    linhas = [r"\begin{table}[!ht]",
              r"\caption{{\bf Discrimination for death attributed to tuberculosis by race "
              r"and colour.} Outcome prevalence is close to flat across categories, so the "
              r"difference in the area under the curve is not a consequence of the decision "
              r"threshold.}",
              r"\label{tab:equity}", r"\begin{tabular}{lrcccc}", r"\hline",
              r"Group & n & Prevalence & AUC & False negative & PPV \\ \hline"]
    for r in d.itertuples():
        linhas.append(f"{_esc(r.rotulo)} & {int(r.n):,} & {r.prevalencia:.3f} & "
                      f"{r.auc:.3f} & {r.fn_rate:.3f} & {r.ppv:.3f} " + r"\\")
    linhas += [r"\hline", r"\end{tabular}", r"\end{table}"]
    return _wrap("equity", "\n".join(linhas))


def t_utility() -> str:
    p = D / "utility_topk.csv"
    if not p.exists():
        return _wrap("utility", "% artefato ausente")
    u = pd.read_csv(p)
    arm = {"modelo_completo": "Model", "baseline_clinico": "Clinical rule"}
    cls = {"tb_death": "TB death", "treatment_interruption": "Treatment interruption"}
    linhas = [r"\begin{table}[!ht]",
              r"\caption{{\bf Programmatic utility under a capacity constraint.} Positive "
              r"predictive value, share of events captured, and number needed to screen when "
              r"the highest-risk fraction of notifications is prioritised. The clinical rule "
              r"uses age, HIV status, alcohol or drug use, and homelessness.}",
              r"\label{tab:utility}", r"\begin{tabular}{llcccc}", r"\hline",
              r"Outcome & Approach & Capacity & PPV & Captured & NNS \\ \hline"]
    for classe in ("tb_death", "treatment_interruption"):
        for k in sorted(u.k.unique()):
            for a in ("modelo_completo", "baseline_clinico"):
                g = u[(u.classe == classe) & (u.k == k) & (u.arm == a)]
                if g.empty:
                    continue
                r = g.iloc[0]
                # o sinal de porcentagem PRECISA sair escapado: `5%` cru inicia um
                # comentário LaTeX e engole o resto da linha, quebrando o alinhamento.
                cap = f"{round(k * 100)}" + r"\%"
                linhas.append(f"{cls[classe]} & {arm[a]} & {cap} & {r.ppv:.3f} & "
                              f"{r.captura:.3f} & {r.nns:.2f} " + r"\\")
        linhas.append(r"\hline")
    linhas += [r"\end{tabular}", r"\end{table}"]
    return _wrap("utility", "\n".join(linhas))


GERADORES = {"cohort": t_cohort, "leaderboard": t_leaderboard, "byclass": t_byclass,
             "temporal": t_temporal, "ablation": t_ablation, "equity": t_equity,
             "utility": t_utility}


def main() -> None:
    tex = TEX.read_text(encoding="utf-8")
    trocadas, ausentes = [], []
    for nome, fn in GERADORES.items():
        marcador = re.compile(rf"% <<TABELA:{nome}>>.*?% <<FIM:{nome}>>", re.S)
        if not marcador.search(tex):
            ausentes.append(nome)
            continue
        tex = marcador.sub(lambda _: fn(), tex)
        trocadas.append(nome)
    TEX.write_text(tex, encoding="utf-8")
    print(f"tabelas injetadas: {trocadas}")
    if ausentes:
        print(f"SEM MARCADOR no .tex: {ausentes}")


if __name__ == "__main__":
    main()
