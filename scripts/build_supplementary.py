"""Monta o material suplementar (S1 File) a partir dos artefatos.

Gera `manuscrito/PLOS_DigitalHealth/S1_File.tex` com as figuras e tabelas que complementam
os resultados do corpo principal, e imprime os textos de legenda a colar na seção
"Supporting information" do manuscrito.

Como o corpo do artigo, nenhum número é digitado: tudo vem dos CSV que o pipeline produziu.
Figuras entram por `\\includegraphics` a partir de `artifacts/figures/`.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from tb_outcomes.robustness import leaderboard

D = Path("data")
FIGS = Path("artifacts/figures")
OUT = Path("manuscrito/PLOS_DigitalHealth/S1_File.tex")

# (arquivo da figura, título, legenda)
FIGURAS = [
    ("01_ranking_f1", "Model ranking by macro-averaged F1",
     "All algorithms under their best-performing imbalance strategy, at 50 municipality "
     "clusters. The dashed line marks the majority-class floor."),
    ("03_strategy_heatmap", "Performance by algorithm and imbalance strategy",
     "Macro-averaged F1 for every algorithm and strategy combination. Cells left blank "
     "denote combinations that are not applicable, such as cost-sensitive weighting for "
     "models that do not accept sample weights."),
    ("04_perclass_recall", "Discrimination by outcome class",
     "Per-class recall and precision for the leading algorithms, showing the asymmetry "
     "between cure and the two unfavourable outcomes."),
    ("06_family_boxplot", "Performance by algorithm family",
     "Distribution of macro-averaged F1 within each algorithm family, pooling strategies."),
    ("09_ece_before_after", "Effect of calibration on expected calibration error",
     "Expected calibration error for each class before and after the cross-fitted "
     "calibrator."),
    ("08_calibration_slope", "Calibration slope by class",
     "Calibration slope before and after calibration. A slope of one denotes agreement "
     "between predicted and observed risk."),
    ("12_fold_variability", "Between-fold variability",
     "Macro-averaged F1 by outer spatial fold, showing how much performance depends on "
     "which territories are held out."),
    ("13_cluster_map", "Spatial clusters used for blocking",
     "Municipalities coloured by K-means cluster at the primary granularity of 50 "
     "clusters. Clusters are held intact across outer folds."),
    ("14_cluster_heterogeneity", "Outcome heterogeneity across clusters",
     "Variation in outcome composition between spatial clusters, which is the dependence "
     "the blocked design controls for."),
    ("17_outcome_by_year", "Outcome composition by notification year",
     "Share of each outcome by year of notification across the study window."),
    ("18_confusion_matrix", "Confusion matrix for the selected model",
     "Row-normalised confusion matrix under the operational decision rule."),
    ("19_proba_distributions", "Predicted probability of death by true class",
     "Distribution of the calibrated probability of death attributed to tuberculosis, "
     "separated by observed outcome."),
    ("21_pr_roc", "Precision-recall and receiver operating characteristic curves",
     "Curves for each outcome class under the selected model."),
    ("26_rank_bump_across_k", "Rank bump chart across spatial granularity",
     "Leaderboard rank of each algorithm at 27, 50, and 75 clusters. Parallel lines "
     "denote a ranking that does not depend on granularity."),
    ("24_small_cells", "Small-cell suppression in subgroup analysis",
     "Distribution of cohort size per municipality, with the threshold below which "
     "subgroup estimates are suppressed."),
]


def _esc(s) -> str:
    return (str(s).replace("\\", "").replace("_", r"\_").replace("%", r"\%")
            .replace("&", r"\&").replace("#", r"\#"))


def _tabela(df: pd.DataFrame, caption: str, label: str, colspec: str | None = None,
            max_linhas: int | None = None) -> str:
    d = df if max_linhas is None else df.head(max_linhas)
    cols = list(d.columns)
    spec = colspec or ("l" + "r" * (len(cols) - 1))
    out = [r"\begin{table}[!ht]", r"\small",
           r"\caption{{\bf " + caption + r"}}", rf"\label{{{label}}}",
           rf"\begin{{tabular}}{{{spec}}}", r"\hline",
           " & ".join(_esc(c) for c in cols) + r" \\ \hline"]
    for r in d.itertuples(index=False):
        vals = []
        for v in r:
            if isinstance(v, float):
                vals.append("" if pd.isna(v) else f"{v:.3f}")
            else:
                vals.append(_esc(v))
        out.append(" & ".join(vals) + r" \\")
    out += [r"\hline", r"\end{tabular}", r"\end{table}"]
    return "\n".join(out)


def main() -> None:
    L: list[str] = []
    add = L.append

    add(r"\documentclass[10pt,letterpaper]{article}")
    add(r"\usepackage[top=1in,left=1in,right=1in,bottom=1in]{geometry}")
    add(r"\usepackage[utf8]{inputenc}")
    add(r"\usepackage{graphicx,booktabs,longtable,caption}")
    add(r"\usepackage[labelfont=bf,labelsep=period]{caption}")
    add(r"\renewcommand{\thefigure}{S\arabic{figure}}")
    add(r"\renewcommand{\thetable}{S\arabic{table}}")
    add(r"\begin{document}")
    add(r"\begin{center}{\Large\bf S1 File. Supporting information}\\[2mm]")
    add(r"Predicting cure, treatment interruption, and death from tuberculosis in Brazil "
        r"under spatially blocked and temporal validation\end{center}")
    add(r"\bigskip")
    add(r"This file collects the figures and tables that complement the main results. "
        r"Every value derives from the analysis artefacts; the numerical file behind each "
        r"figure is deposited with the study data.")
    add(r"\clearpage")

    # ---------------------------------------------------------------- tabelas
    legendas: list[str] = []

    p = D / "baseline_characteristics.csv"
    if p.exists():
        d = pd.read_csv(p)
        cols = ["variable", "level", "Cure n", "Cure %", "TB death n", "TB death %",
                "Treatment interruption n", "Treatment interruption %"]
        d = d[[c for c in cols if c in d.columns]]
        add(_tabela(d, "Baseline characteristics of the analytic cohort, by outcome. "
                       "Percentages use the non-missing total of each variable as "
                       "denominator; missingness is reported as its own row.",
                    "tab:s1", colspec="llrrrrrr"))
        legendas.append(("S1 Table", "Baseline characteristics of the analytic cohort."))
        add(r"\clearpage")

    p = D / "equity_disparity.csv"
    if p.exists():
        d = pd.read_csv(p)
        d = d[(d.regra == "pred_policy") & (d.metrica == "fn_rate")]
        cols = {"classe": "Outcome", "eixo": "Axis", "n_grupos": "Groups",
                "grupo_pior_rotulo": "Worst group", "valor_pior": "Worst",
                "grupo_melhor_rotulo": "Best group", "valor_melhor": "Best",
                "razao": "Ratio"}
        d = d[[c for c in cols if c in d.columns]].rename(columns=cols)
        add(_tabela(d, "False-negative rate by subgroup, under the operational decision "
                       "rule, for every axis examined. Groups whose estimate was "
                       "suppressed for small cell size are excluded from the comparison.",
                    "tab:s2", colspec="llrlrlrr"))
        legendas.append(("S2 Table", "False-negative rate by subgroup and axis."))
        add(r"\clearpage")

    p = D / "sweep" / "rank_stability.csv"
    if p.exists():
        d = pd.read_csv(p)
        add(_tabela(d, "Rank stability across spatial granularity. Kendall's tau and "
                       "top-five overlap between the leaderboards obtained at each pair "
                       "of cluster counts.", "tab:s3"))
        legendas.append(("S3 Table", "Rank stability across spatial granularity."))

    p = D / "compute_cost_by_model.csv"
    if p.exists():
        d = pd.read_csv(p)
        add(_tabela(d, "Computational cost by algorithm, summed over imbalance "
                       "strategies and analytical arms.", "tab:s4"))
        legendas.append(("S4 Table", "Computational cost by algorithm."))
        add(r"\clearpage")

    p = D / "fold_summary.csv"
    if p.exists():
        d = pd.read_csv(p)
        add(_tabela(d, "Composition of the outer spatial folds: number of records, "
                       "clusters, and events per fold.", "tab:s5"))
        legendas.append(("S5 Table", "Composition of the outer spatial folds."))

    p = D / "cluster_summary.csv"
    if p.exists():
        d = pd.read_csv(p)
        add(_tabela(d, "Composition of the spatial clusters at the primary granularity: "
                       "municipalities, records, and events per cluster.", "tab:s6",
                    max_linhas=50))
        legendas.append(("S6 Table", "Composition of the spatial clusters."))
        add(r"\clearpage")

    # ---------------------------------------------------------------- figuras
    for i, (arq, titulo, legenda) in enumerate(FIGURAS, 1):
        caminho = FIGS / f"{arq}.png"
        if not caminho.exists():
            continue
        add(r"\begin{figure}[!ht]")
        add(r"\centering")
        add(rf"\includegraphics[width=\linewidth,height=0.78\textheight,keepaspectratio]"
            rf"{{../../{caminho.as_posix()}}}")
        add(r"\caption{{\bf " + titulo + r".} " + legenda + r"}")
        add(r"\end{figure}")
        add(r"\clearpage")
        legendas.append((f"S{i} Fig", f"{titulo}."))

    add(r"\end{document}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L), encoding="utf-8")
    print(f"gravado: {OUT}")
    print("\n--- legendas para a seção Supporting information do manuscrito ---")
    for rotulo, titulo in legendas:
        print(f"  {rotulo}. {titulo}")


if __name__ == "__main__":
    main()
