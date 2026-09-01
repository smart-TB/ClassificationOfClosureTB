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

from tb_outcomes import labels as rot
from tb_outcomes.robustness import leaderboard

D = Path("data")
FIGS = Path("artifacts/figures")
OUT = Path("manuscrito/PLOS_DigitalHealth/S1_File.tex")

# (arquivo da figura, título, legenda)
FIGURAS = [
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
            max_linhas: int | None = None, longa: bool = False) -> str:
    """Monta a tabela em LaTeX.

    `longa=True` usa `longtable`, que quebra entre páginas. Sem isso uma tabela mais alta
    que a página é SILENCIOSAMENTE CORTADA pelo `table`, e o leitor perde linhas sem
    qualquer aviso. Toda tabela com dezenas de linhas precisa disso.
    """
    d = df if max_linhas is None else df.head(max_linhas)
    cols = list(d.columns)
    spec = colspec or ("l" + "r" * (len(cols) - 1))
    cabecalho = " & ".join(_esc(c) for c in cols) + r" \\ \hline"

    corpo = []
    for r in d.itertuples(index=False):
        vals = []
        for v in r:
            if isinstance(v, float):
                vals.append("" if pd.isna(v) else f"{v:.3f}")
            else:
                vals.append("" if (v is None or str(v) in ("nan", "None")) else _esc(v))
        corpo.append(" & ".join(vals) + r" \\")

    if longa:
        out = [r"\begingroup", r"\small", rf"\begin{{longtable}}{{{spec}}}",
               r"\caption{{\bf " + caption + r"}}", rf"\label{{{label}}}", r"\\",
               r"\hline", cabecalho, r"\endfirsthead",
               r"\hline", cabecalho, r"\endhead",
               r"\hline", r"\endfoot", *corpo,
               r"\end{longtable}", r"\endgroup"]
        return "\n".join(out)

    out = [r"\begin{table}[!ht]", r"\small",
           r"\caption{{\bf " + caption + r"}}", rf"\label{{{label}}}",
           rf"\begin{{tabular}}{{{spec}}}", r"\hline", cabecalho,
           *corpo, r"\hline", r"\end{tabular}", r"\end{table}"]
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

    # fluxo da coorte: saiu do corpo quando a Fig 1 passou a trazer os motivos, mas os
    # números exatos por critério continuam disponíveis aqui.
    fl = pd.read_csv(D / "cohort_flow_sensitive_tb_3class.csv")
    ex = pd.read_csv(D / "exclusion_reasons_sensitive_tb_3class.csv")
    desc = {"recuperada": "Records returned by the acquisition pipeline",
            "declarada": "Notified between 2015 and 2025",
            "elegivel": "Eligible after quality and follow-up criteria",
            "analitica": "Analytic cohort with a three-class outcome"}
    mot = {"municipio_irrecuperavel": "Unrecoverable municipality code",
           "idade_impossivel": "Impossible age",
           "fora_da_janela_declarada": "Outside the declared window",
           "seguimento_insuficiente": "Follow-up shorter than 365 days"}
    linhas = [{"Item": desc.get(r.stage, r.stage), "n": f"{r.n:,}"} for r in fl.itertuples()]
    linhas += [{"Item": "Excluded: " + mot.get(r.reason, r.reason), "n": f"{r.n:,}"}
               for r in ex.itertuples()]
    add(_tabela(pd.DataFrame(linhas), "Cohort definition and exclusions. Stage counts and "
                "the number removed by each criterion. Criteria are applied in sequence. "
                "Values behind Fig 1 of the main text.", "tab:s0", colspec="p{9.0cm}r"))
    legendas.append(("S1 Table", "Cohort definition and exclusions."))
    add(r"\clearpage")

    # tabelas que saíram do corpo quando o achado passou a ser figura: os números
    # continuam disponíveis, só não competem com a ilustração pelo espaço principal.
    from tb_outcomes.robustness import leaderboard as _lb
    d = _lb(pd.read_csv(D / "benchmark_metrics.csv")).sort_values(
        "f1_macro", ascending=False).reset_index(drop=True)
    d.insert(0, "Rank", range(1, len(d) + 1))
    d["model"] = d["model"].map(rot.algoritmo)
    est = {"local_cost_sensitive": "Cost-sensitive", "random_oversampling": "Oversampling",
           "random_undersampling": "Undersampling"}
    d["best_strategy"] = d["best_strategy"].map(lambda x: est.get(x, x))
    d.columns = ["Rank", "Algorithm", "Best strategy", "Macro F1"]
    add(_tabela(d, "Model ranking under nested spatially blocked validation, at 50 "
                   "municipality clusters. Values behind Fig 2 of the main text.",
                "tab:s1", colspec="rp{6.0cm}lr", longa=True))
    legendas.append(("S2 Table", "Model ranking under spatially blocked validation."))
    add(r"\clearpage")

    p = D / "ablation" / "ablation_leaderboard.csv"
    if p.exists():
        d = pd.read_csv(p)
        d["model"] = d["model"].map(rot.algoritmo)
        cols = ["model"] + [c for c in ("individual", "municipal", "combined") if c in d]
        d = d[cols]
        d.columns = ["Algorithm", "Individual", "Municipal", "Both"]
        add(_tabela(d, "Territorial ablation: macro-averaged F1 by feature set. Values "
                       "behind Fig 5 of the main text.", "tab:s2", colspec="p{6.0cm}rrr"))
        legendas.append(("S3 Table", "Territorial ablation by feature set."))
        add(r"\clearpage")

    p = D / "equity_discrimination_by_group.csv"
    if p.exists():
        d = pd.read_csv(p)
        d = d[(d.eixo == "raca_cor") & (d.classe == "tb_death")].sort_values(
            "auc", ascending=False).copy()
        d["rotulo"] = [rot.nivel("Race/colour", x) for x in d["rotulo"]]
        d = d[["rotulo", "n", "prevalencia", "auc", "fn_rate", "ppv"]]
        d.columns = ["Group", "n", "Prevalence", "AUC", "False negative", "PPV"]
        add(_tabela(d, "Discrimination for death attributed to tuberculosis by race and "
                       "colour. Values behind Fig 7 of the main text.", "tab:s3",
                    colspec="p{4.2cm}rrrrr"))
        legendas.append(("S4 Table", "Discrimination by race and colour."))
        add(r"\clearpage")

    p = D / "baseline_characteristics.csv"
    if p.exists():
        d = pd.read_csv(p)
        # combina n e % numa coluna só: com oito colunas a tabela estourava a margem
        for nome in ("Cure", "TB death", "Treatment interruption"):
            cn, cp = f"{nome} n", f"{nome} %"
            if cn in d.columns and cp in d.columns:
                d[nome] = [
                    (f"{int(n):,} ({p:.1f})" if pd.notna(p) and str(n).isdigit() else str(n))
                    for n, p in zip(d[cn], d[cp])
                ]
        d["level"] = [rot.nivel(v, x) for v, x in zip(d["variable"], d["level"])]
        d = d[["variable", "level", "Cure", "TB death", "Treatment interruption"]]
        d.columns = ["Variable", "Level", "Cure", "TB death", "Interruption"]
        sem_dic = ", ".join(rot.SEM_DICIONARIO)
        add(_tabela(d, "Baseline characteristics of the analytic cohort, by outcome, "
                       "as n (\\%). Percentages use the non-missing total of each "
                       "variable as denominator, and missingness is reported as its own "
                       "row. Levels of " + sem_dic + " are given as the SINAN code, since "
                       "the code dictionary for those fields was not available.",
                    "tab:s4", colspec="p{3.0cm}p{5.2cm}rrr", longa=True))
        legendas.append(("S5 Table", "Baseline characteristics of the analytic cohort."))
        add(r"\clearpage")

    p = D / "equity_disparity.csv"
    if p.exists():
        d = pd.read_csv(p)
        d = d[(d.regra == "pred_policy") & (d.metrica == "fn_rate")].copy()
        d["classe"] = d["classe"].map(rot.desfecho)
        d["eixo_en"] = d["eixo"].map(rot.eixo)
        for col in ("grupo_pior_rotulo", "grupo_melhor_rotulo"):
            if col in d.columns:
                d[col] = [rot.nivel(e, x) for e, x in zip(d["eixo_en"], d[col])]
        d = d[["classe", "eixo_en", "grupo_pior_rotulo", "valor_pior",
               "grupo_melhor_rotulo", "valor_melhor", "razao"]]
        d.columns = ["Outcome", "Axis", "Worst group", "Worst", "Best group", "Best",
                     "Ratio"]
        add(_tabela(d, "False-negative rate by subgroup, under the operational decision "
                       "rule, for every axis examined. Groups whose estimate was "
                       "suppressed for small cell size are excluded from the comparison.",
                    "tab:s5", colspec="p{2.4cm}p{2.6cm}p{3.4cm}rp{3.4cm}rr", longa=True))
        legendas.append(("S6 Table", "False-negative rate by subgroup and axis."))
        add(r"\clearpage")

    p = D / "sweep" / "rank_stability.csv"
    if p.exists():
        d = pd.read_csv(p)
        add(_tabela(d, "Rank stability across spatial granularity. Kendall's tau and "
                       "top-five overlap between the leaderboards obtained at each pair "
                       "of cluster counts.", "tab:s6"))
        legendas.append(("S7 Table", "Rank stability across spatial granularity."))

    p = D / "compute_cost_by_model.csv"
    if p.exists():
        d = pd.read_csv(p)
        d["model"] = d["model"].map(rot.algoritmo)
        d = d.rename(columns={"model": "Algorithm", "n_pares": "Pairs",
                              "total_horas": "Hours", "media_minutos": "Mean (min)",
                              "min_minutos": "Min", "max_minutos": "Max",
                              "pct_do_total": "Share (\\%)"})
        add(_tabela(d, "Computational cost by algorithm, summed over imbalance "
                       "strategies and analytical arms.", "tab:s7"))
        legendas.append(("S8 Table", "Computational cost by algorithm."))
        add(r"\clearpage")

    p = D / "fold_summary.csv"
    if p.exists():
        d = pd.read_csv(p)
        add(_tabela(d, "Composition of the outer spatial folds: number of records, "
                       "clusters, and events per fold.", "tab:s8"))
        legendas.append(("S9 Table", "Composition of the outer spatial folds."))

    p = D / "cluster_summary.csv"
    if p.exists():
        d = pd.read_csv(p)
        add(_tabela(d, "Composition of the spatial clusters at the primary granularity: "
                       "municipalities, records, and events per cluster.", "tab:s9",
                    longa=True))
        legendas.append(("S10 Table", "Composition of the spatial clusters."))
        add(r"\clearpage")

    p = D / "feature_dictionary.csv"
    if p.exists():
        d = pd.read_csv(p)
        d = d[d["in_notification_set"] == True]  # noqa: E712
        d = d[["raw_name", "form_field", "type", "source", "contextual"]].copy()
        d["contextual"] = d["contextual"].map({True: "Municipal", False: "Individual"})
        d.columns = ["Variable", "Form field", "Type", "Source", "Level"]
        d = d.sort_values(["Level", "Variable"])
        add(_tabela(d, "Variable dictionary for the feature set. `Form field` is the "
                       "position on the national notification form. Variable names follow "
                       "the SINAN field names, which are the labels shown in Fig 6 of the "
                       "main text.", "tab:s10", colspec="p{4.6cm}rp{2.4cm}p{2.0cm}p{2.2cm}",
                    longa=True))
        legendas.append(("S11 Table", "Variable dictionary for the feature set."))
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
