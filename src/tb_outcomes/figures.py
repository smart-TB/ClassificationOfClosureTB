"""Figuras do manuscrito (SP relatório). Estáticas, prontas para publicação.

Paleta categórica colorblind-safe validada (dataviz skill, validate_palette.js):
azul #2a78d6, verde #008300, magenta #e87ba4, amarelo #eda100, aqua #1baf7a,
laranja #eb6834, violeta #4a3aa7, vermelho #e34948. Cores seguem a ENTIDADE
(mesma classe = mesma cor em todas as figuras), nunca o rank.
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger("tb_outcomes")

FIG_DIR = Path("artifacts/figures")
DATA_DIR = Path("data")
# acompanhamento numérico de cada figura (§6); não confundir com DATA_DIR, que é a
# fonte de dados das figuras.
FIG_DATA_DIR = FIG_DIR / "data"

# paleta categórica validada (light surface)
CAT = ["#2a78d6", "#008300", "#e87ba4", "#eda100", "#1baf7a", "#eb6834", "#4a3aa7", "#e34948"]
# rampa sequencial azul (magnitude / funil ordinal), clara->escura
BLUE_SEQ = ["#86b6ef", "#5598e7", "#3987e5", "#2a78d6", "#1c5cab", "#184f95", "#104281"]
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e6e5e2"

# outcome class -> (English label, fixed colour). Todas as figuras em inglês.
# Esquema primário sensitive_tb_3class (PI 2026-07-19): 3 desfechos fechados.
SCHEMA = "sensitive_tb_3class"
OUTCOME = {
    "cure": ("Cure", "#2a78d6"),
    "treatment_interruption": ("Treatment interruption", "#eda100"),
    "tb_death": ("TB-attributed death", "#e34948"),
}


def _style() -> None:
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.edgecolor": INK2,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelcolor": INK2,
        "text.color": INK,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
    })


def _titles(ax, title: str, subtitle: str) -> None:
    # título acima, subtítulo logo abaixo dele — sem colisão (pad em pontos).
    ax.set_title(title, loc="left", pad=30)
    ax.text(0.0, 1.045, subtitle, transform=ax.transAxes, fontsize=9.5,
            color=INK2, va="bottom", ha="left")


def _save(fig, name: str) -> Path:
    """Salva a figura e, ao lado dela, o CSV com os números que a originaram (§6).

    O acompanhamento numérico é extraído dos artistas do matplotlib, então descreve o que
    foi de fato desenhado. A extração NUNCA derruba a gravação da figura: se falhar, a
    figura sai e o erro fica no log.
    """
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    try:
        from tb_outcomes.figure_data import extract_figure

        dados = extract_figure(fig, name)
        if not dados.empty:
            FIG_DATA_DIR.mkdir(parents=True, exist_ok=True)
            dados.to_csv(FIG_DATA_DIR / f"{name}.csv", index=False)
    except Exception:  # noqa: BLE001 — acompanhamento não pode custar a figura
        logger.warning("não foi possível extrair os dados da figura %s", name, exc_info=True)
    plt.close(fig)
    return path


def _thousands(n: int) -> str:
    return f"{n:,}"  # separador de milhar em inglês (vírgula)


def fig_cohort_funnel() -> Path:
    """Fig 15 — funil de atrito da coorte (recuperada -> analítica)."""
    flow = pd.read_csv(DATA_DIR / f"cohort_flow_{SCHEMA}.csv")
    labels = {"recuperada": "Acquired", "declarada": "Notified (2015–2025)",
              "elegivel": "Eligible", "analitica": "Analytic"}
    flow["label"] = flow["stage"].map(labels)
    base = flow["n"].iloc[0]

    fig, ax = plt.subplots(figsize=(8.4, 3.8))
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", zorder=0)
    y = range(len(flow))[::-1]  # topo = primeira etapa
    colors = BLUE_SEQ[1:1 + len(flow)]
    ax.barh(list(y), flow["n"], color=colors, height=0.62, zorder=3)
    for yi, (_, r) in zip(y, flow.iterrows()):
        pct = 100 * r["n"] / base
        ax.text(r["n"] + base * 0.01, yi, f"{_thousands(int(r['n']))}  ({pct:.1f}%)",
                va="center", ha="left", fontsize=10, color=INK)
    ax.set_yticks(list(y))
    ax.set_yticklabels(flow["label"])
    ax.set_xlim(0, base * 1.18)
    ax.set_xlabel("Number of notifications")
    _titles(ax, "Cohort attrition: from acquisition to the analytic population",
            "TB, Brazil, 2015–2025 · closed 3-outcome benchmark (cure · interruption · TB death)")
    ax.xaxis.set_major_formatter(lambda x, _: f"{x/1e6:.1f}M" if x else "0")
    return _save(fig, "15_cohort_funnel")


def fig_outcome_prevalence() -> Path:
    """Fig 16 — prevalência por classe de desfecho (o desbalanceamento)."""
    dist = pd.read_csv(DATA_DIR / f"outcome_distribution_{SCHEMA}.csv")
    dist = dist.sort_values("n", ascending=True)
    dist["label"] = dist["target"].map(lambda t: OUTCOME[t][0])
    dist["color"] = dist["target"].map(lambda t: OUTCOME[t][1])
    total = dist["n"].sum()

    fig, ax = plt.subplots(figsize=(8.4, 3.8))
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", zorder=0)
    yy = range(len(dist))
    ax.barh(list(yy), dist["n"], color=dist["color"], height=0.62, zorder=3)
    for yi, (_, r) in zip(yy, dist.iterrows()):
        ax.text(r["n"] + total * 0.008, yi, f"{_thousands(int(r['n']))}  ({r['pct']:.1f}%)",
                va="center", ha="left", fontsize=10, color=INK)
    ax.set_yticks(list(yy))
    ax.set_yticklabels(dist["label"])
    ax.set_xlim(0, dist["n"].max() * 1.22)
    ax.set_xlabel("Number of patients")
    _titles(ax, "Outcome prevalence in the analytic cohort",
            f"n = {_thousands(int(total))} patients · imbalanced classes")
    ax.xaxis.set_major_formatter(lambda x, _: f"{x/1e3:.0f}k" if x else "0")
    return _save(fig, "16_outcome_prevalence")


_ANALYTIC = None


def _analytic():
    """Coorte analítica (esquema primário SCHEMA), carregada uma vez e cacheada."""
    global _ANALYTIC
    if _ANALYTIC is None:
        from tb_outcomes import cohort as coh
        from tb_outcomes import data as dat
        from tb_outcomes.config import load_config
        cfg = load_config("configs/analysis_decisions.yaml")
        df, _ = dat.load_or_build_harmonized(DATA_DIR, cfg)
        rules = coh.load_outcome_rules("configs/outcomes.yaml")
        _ANALYTIC = coh.build_cohort(df, cfg, rules, SCHEMA).analytic
    return _ANALYTIC


def _anova_icc(y, groups) -> float:
    """ICC de uma via (ANOVA) de um desfecho binário agrupado por município.

    ICC = fração da variância TOTAL que é ENTRE municípios. Estimador ANOVA
    (corrige o viés de grupos pequenos via n0). Retorna 0..1.
    """
    import numpy as np

    y = np.asarray(y, dtype=float)
    g = pd.Series(groups).to_numpy()
    n = len(y)
    by = pd.Series(y).groupby(g)
    n_j = by.size().to_numpy().astype(float)
    p_j = by.mean().to_numpy()
    k = len(n_j)
    grand = y.mean()
    ssb = float(np.sum(n_j * (p_j - grand) ** 2))
    ssw = float(np.sum(n_j * p_j * (1 - p_j)))  # binário: var intra-grupo = p(1-p)
    msb = ssb / (k - 1)
    msw = ssw / (n - k)
    n0 = (n - np.sum(n_j ** 2) / n) / (k - 1)
    icc = (msb - msw) / (msb + (n0 - 1) * msw)
    return float(max(0.0, icc))


def fig_outcome_by_year() -> Path:
    """Fig 17 — composição do desfecho por ano (com o holdout 2024)."""
    a = _analytic()
    yr = pd.to_numeric(a["NU_ANO"], errors="coerce")
    tab = pd.crosstab(yr, a[f"target_{SCHEMA}"], normalize="index") * 100
    tab = tab[(tab.index >= 2015) & (tab.index <= 2025)]

    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    ax.grid(axis="x", visible=False)
    ax.axvspan(2023.5, 2024.5, color="#f0efec", zorder=0)  # holdout temporal
    ax.text(2024, ax.get_ylim()[1], "2024\nholdout", ha="center", va="top",
            fontsize=8.5, color=INK2)
    years = tab.index.to_numpy()
    for key, (label, color) in OUTCOME.items():
        if key in tab.columns:
            ax.plot(years, tab[key].to_numpy(), color=color, lw=2.2, zorder=3, marker="o", ms=4)
            ax.text(years.max() + 0.1, tab[key].iloc[-1], label, color=color,
                    va="center", fontsize=9.5, fontweight="bold")
    ax.set_xlim(2015, 2027.4)
    ax.set_ylim(0, None)
    ax.set_xticks(range(2015, 2026, 2))
    ax.set_ylabel("Share within the year (%)")
    ax.set_xlabel("Notification year")
    _titles(ax, "Outcome composition by notification year",
            "closed 3-outcome benchmark · shaded band = 2024 temporal holdout")
    return _save(fig, "17_outcome_by_year")


def fig_icc_decomposition() -> Path:
    """Fig 22 — teto territorial: variância ENTRE vs DENTRO de municípios."""
    a = _analytic()
    muni = a["ID_MUNIC_ANALISE"]
    tgt = a[f"target_{SCHEMA}"]
    rows = [
        ("Treatment interruption", (tgt == "treatment_interruption").astype(int), "#eda100"),
        ("TB-attributed death", (tgt == "tb_death").astype(int), "#e34948"),
    ]
    fig, ax = plt.subplots(figsize=(8.6, 3.4))
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", zorder=0)
    for i, (label, yb, color) in enumerate(rows):
        icc = 100 * _anova_icc(yb, muni)
        ax.barh(i, icc, color=color, height=0.5, zorder=3)
        ax.barh(i, 100 - icc, left=icc, color="#e6e5e2", height=0.5, zorder=3)
        ax.text(icc + 1.2, i, f"between municipalities: {icc:.1f}%", va="center",
                ha="left", fontsize=10, color=INK, fontweight="bold")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of total outcome variance (%)")
    _titles(ax, "Territorial ceiling: at most ~4% of the outcome is municipal",
            "one-way ANOVA ICC · the remaining ~96% is within-municipality, "
            "unreachable by any territorial variable")
    return _save(fig, "22_icc_territorial_ceiling")


def fig_effective_dimension() -> Path:
    """Fig 23 — dimensão efetiva das contextuais (quantos componentes p/ 90%)."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    ibge = pd.read_csv(DATA_DIR / "indicadores_IBGE.csv")
    ipea = pd.read_csv(DATA_DIR / "indicadores_IPEA.csv")
    ctx = ibge.merge(ipea, on=["ID_MUNICIP", "ANO_DIAG"], how="outer")
    num = ctx.select_dtypes("number").drop(columns=[c for c in ("ID_MUNICIP", "ANO_DIAG") if c in ctx], errors="ignore")
    num = num.dropna(axis=1, how="all").dropna()
    X = StandardScaler().fit_transform(num.to_numpy())
    pca = PCA().fit(X)
    cum = pca.explained_variance_ratio_.cumsum() * 100
    k90 = int((cum < 90).sum()) + 1
    p = len(cum)

    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    ax.grid(zorder=0)
    ax.plot(range(1, p + 1), cum, color="#2a78d6", lw=2.2, marker="o", ms=3.5, zorder=3)
    ax.axhline(90, color=INK2, lw=1.0, ls="--", zorder=2)
    ax.axvline(k90, color="#eb6834", lw=1.4, zorder=2)
    ax.text(k90 + 0.4, 40, f"{k90} components\nreach 90%", color="#eb6834",
            fontsize=10, fontweight="bold", va="center")
    ax.text(p, 91, "90% of variance", ha="right", va="bottom", fontsize=9, color=INK2)
    ax.set_xlim(1, p)
    ax.set_ylim(0, 101)
    ax.set_xlabel("Number of principal components")
    ax.set_ylabel("Cumulative explained variance (%)")
    _titles(ax, "Effective dimension of the contextual indicators",
            f"{k90} of {p} components explain 90% — counting raw columns overstates territory")
    return _save(fig, "23_effective_dimension")


def fig_small_cells() -> Path:
    """Fig 24 — supressão de células pequenas (municípios com <30 pacientes)."""
    import numpy as np

    a = _analytic()
    counts = a.groupby("ID_MUNIC_ANALISE").size()
    below = 100 * (counts < 30).mean()
    n_muni = len(counts)

    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", zorder=0)
    bins = np.logspace(0, np.log10(counts.max()), 40)
    ax.hist(counts, bins=bins, color="#1baf7a", zorder=3)
    ax.set_xscale("log")
    ax.axvline(30, color="#e34948", lw=1.8, zorder=4)
    ax.annotate(f"< 30 patients:\n{below:.0f}% of municipalities",
                xy=(30, ax.get_ylim()[1] * 0.55), xytext=(120, ax.get_ylim()[1] * 0.78),
                color="#e34948", fontsize=10, fontweight="bold", ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color="#e34948", lw=1.4))
    ax.set_xlabel("Patients per municipality (log scale)")
    ax.set_ylabel("Number of municipalities")
    _titles(ax, "Small-cell suppression will erase much of the equity map",
            f"{n_muni:,} municipalities · the §20.3 suppression rule hits cells with <30 patients")
    return _save(fig, "24_small_cells")


COHORT_ICC_FIGURES = [
    fig_cohort_funnel, fig_outcome_prevalence, fig_outcome_by_year,
    fig_icc_decomposition, fig_effective_dimension, fig_small_cells,
]

# ---------------------------------------------------------------------------
# Benchmark (famílias A/B/C/E) — do benchmark_metrics.csv + oof_predictions.parquet
# ---------------------------------------------------------------------------
# classe codificada -> rótulo (ordem alfabética: cure=0, tb_death=1, interruption=2)
CLS_LABEL = {"0": "Cure", "1": "TB death", "2": "Treatment interruption"}
CLS_COLOR = {"0": "#2a78d6", "1": "#e34948", "2": "#eda100"}
STRAT_LABEL = {"random_undersampling": "under", "random_oversampling": "over",
               "local_cost_sensitive": "cost"}
STRAT_COLOR = {"under": "#2a78d6", "over": "#1baf7a", "cost": "#eb6834"}
FAMILY = {
    "majority_class": "baseline", "stratified_random": "baseline",
    "logistic_plain": "linear", "logistic_regression": "linear", "ridge_classifier": "linear",
    "decision_tree": "tree", "random_forest": "tree", "extra_trees": "tree",
    "gradient_boosting": "boosting", "hist_gradient_boost": "boosting", "adaboost": "boosting",
    "xgboost": "boosting", "lightgbm": "boosting", "catboost": "boosting",
    "gaussian_nb": "naive bayes", "multinomial_nb": "naive bayes",
    "bernoulli_nb": "naive bayes", "complement_nb": "naive bayes",
    "lda": "discriminant", "qda": "discriminant", "knn": "neighbors", "mlp": "neural net",
    "category_embedding": "deep learning", "tabnet": "deep learning",
    "tab_transformer": "deep learning", "ft_transformer": "deep learning",
}
FAMILY_COLOR = {"baseline": "#52514e", "linear": "#2a78d6", "tree": "#1baf7a",
                "boosting": "#008300", "naive bayes": "#eda100", "discriminant": "#e87ba4",
                "neighbors": "#eb6834", "neural net": "#4a3aa7", "deep learning": "#e34948"}

_AGG = None
_PERFOLD = None


def _metrics():
    global _AGG, _PERFOLD
    if _AGG is None:
        m = pd.read_csv(DATA_DIR / "benchmark_metrics.csv", dtype={"class": str})
        _AGG = m[m["kind"] == "aggregate"].copy()
        _PERFOLD = m[m["kind"] == "per_fold"].copy()
    return _AGG, _PERFOLD


def _best_strategy(agg):
    """Melhor estratégia de cada modelo por F1-macro (média entre dobras)."""
    f1 = agg[agg["metric"] == "f1_macro"][["model", "strategy", "mean"]]
    return f1.sort_values("mean", ascending=False).groupby("model", as_index=False).first()


def fig_ranking_f1() -> Path:
    """Fig 1 — forest plot do ranking por F1-macro (melhor estratégia por modelo)."""
    agg, _ = _metrics()
    b = _best_strategy(agg).merge(
        agg[agg["metric"] == "f1_macro"][["model", "strategy", "std", "min", "max"]],
        on=["model", "strategy"]).sort_values("mean")
    fig, ax = plt.subplots(figsize=(8.6, 8.0))
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", zorder=0)
    y = range(len(b))
    colors = [FAMILY_COLOR[FAMILY[m]] for m in b["model"]]
    ax.hlines(list(y), b["min"], b["max"], color=colors, lw=1.5, alpha=0.5, zorder=2)
    ax.scatter(b["mean"], list(y), color=colors, s=42, zorder=3)
    ax.axvline(0.289, color=INK2, ls="--", lw=1.0, zorder=1)
    ax.text(0.289, len(b) - 0.5, " majority\n baseline", fontsize=8, color=INK2, va="top")
    ax.set_yticks(list(y))
    ax.set_yticklabels([f"{m}  [{STRAT_LABEL.get(s, s)}]" for m, s in zip(b["model"], b["strategy"])],
                       fontsize=9)
    ax.set_xlabel("Macro-F1 (mean across 5 outer folds; bar = min–max)")
    _titles(ax, "Model ranking under nested spatial cross-validation",
            "closed 3-outcome benchmark · colour = model family · no winner promoted")
    from matplotlib.patches import Patch
    fams = ["boosting", "tree", "deep learning", "linear", "discriminant", "naive bayes",
            "neighbors", "neural net", "baseline"]
    ax.legend(handles=[Patch(color=FAMILY_COLOR[f], label=f) for f in fams],
              loc="lower right", fontsize=8, frameon=False)
    return _save(fig, "01_ranking_f1")


def fig_ranking_small_multiples() -> Path:
    """Fig 2 — mesmo ranking em balanced accuracy, AUPRC e ROC AUC de óbito-TB."""
    agg, _ = _metrics()
    b = _best_strategy(agg)
    specs = [("balanced_accuracy", "__global__", "Balanced accuracy"),
             ("auprc", "1", "AUPRC · TB death"), ("roc_auc", "1", "ROC AUC · TB death")]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 5.6))
    for ax, (metric, cls, title) in zip(axes, specs):
        d = agg[(agg["metric"] == metric) & (agg["class"] == cls)][["model", "strategy", "mean"]]
        d = b.merge(d, on=["model", "strategy"], suffixes=("_f1", "")).sort_values("mean")
        colors = [FAMILY_COLOR[FAMILY[m]] for m in d["model"]]
        ax.barh(range(len(d)), d["mean"], color=colors, height=0.7, zorder=3)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", zorder=0)
        ax.set_yticks(range(len(d)))
        ax.set_yticklabels(d["model"], fontsize=7)
        ax.set_title(title, fontsize=11, loc="left")
        ax.set_xlim(0, max(d["mean"].max() * 1.1, 0.1))
    fig.suptitle("Ranking is consistent across discrimination metrics", x=0.09, ha="left",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _save(fig, "02_ranking_small_multiples")


def fig_strategy_heatmap() -> Path:
    """Fig 3 — heatmap modelo × estratégia (F1-macro): qual balanceamento ajuda qual."""
    agg, _ = _metrics()
    f1 = agg[agg["metric"] == "f1_macro"]
    piv = f1.pivot_table(index="model", columns="strategy", values="mean")
    piv = piv[["random_undersampling", "random_oversampling", "local_cost_sensitive"]]
    piv = piv.loc[_best_strategy(agg).sort_values("mean", ascending=False)["model"]]
    fig, ax = plt.subplots(figsize=(6.2, 8.2))
    im = ax.imshow(piv.to_numpy(), aspect="auto", cmap="Blues", vmin=0.28, vmax=0.5)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["under", "over", "cost"])
    ax.set_yticks(range(len(piv)))
    ax.set_yticklabels(piv.index, fontsize=8)
    for i in range(len(piv)):
        for j in range(3):
            v = piv.iloc[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if v > 0.42 else INK)
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Macro-F1")
    _titles(ax, "Macro-F1 by model × balancing strategy",
            "cost-sensitive helps most models; blank = not_run (no sample_weight)")
    return _save(fig, "03_strategy_heatmap")


def fig_perclass_discrimination() -> Path:
    """Fig 4 — recall por classe de desfecho, top-10 modelos (melhor estratégia)."""
    agg, _ = _metrics()
    b = _best_strategy(agg).sort_values("mean", ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", zorder=0)
    x = range(len(b))
    w = 0.26
    for k, cls in enumerate(["0", "1", "2"]):
        vals = []
        for _, r in b.iterrows():
            d = agg[(agg["model"] == r["model"]) & (agg["strategy"] == r["strategy"]) &
                    (agg["metric"] == "recall") & (agg["class"] == cls)]
            vals.append(d["mean"].iloc[0] if len(d) else 0)
        ax.bar([xi + (k - 1) * w for xi in x], vals, width=w, color=CLS_COLOR[cls],
               label=CLS_LABEL[cls], zorder=3)
    ax.set_xticks(list(x))
    ax.set_xticklabels(b["model"], rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Recall (argmax)")
    ax.legend(fontsize=9, frameon=False)
    _titles(ax, "Per-class recall: the rare classes are hard even for the leaders",
            "top-10 models · cure is caught, TB death & interruption are under-recovered at argmax")
    return _save(fig, "04_perclass_recall")


def fig_strategy_effect() -> Path:
    """Fig 5 — slopegraph do efeito da estratégia (under→over→cost) por modelo."""
    agg, _ = _metrics()
    f1 = agg[agg["metric"] == "f1_macro"]
    piv = f1.pivot_table(index="model", columns="strategy", values="mean")
    order = ["random_undersampling", "random_oversampling", "local_cost_sensitive"]
    fig, ax = plt.subplots(figsize=(7.6, 6.4))
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", zorder=0)
    for model, row in piv.iterrows():
        ys = [row.get(s, float("nan")) for s in order]
        xs = [0, 1, 2]
        c = FAMILY_COLOR[FAMILY[model]]
        ax.plot(xs, ys, color=c, lw=1.3, marker="o", ms=4, alpha=0.75, zorder=3)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["under", "over", "cost"])
    ax.set_xlim(-0.15, 2.15)
    ax.set_ylabel("Macro-F1")
    _titles(ax, "Effect of the balancing strategy on each model",
            "each line = one model, coloured by family")
    return _save(fig, "05_strategy_effect")


def fig_family_boxplot() -> Path:
    """Fig 6 — F1-macro por família de modelo (melhor estratégia de cada modelo)."""
    agg, _ = _metrics()
    b = _best_strategy(agg).copy()
    b["family"] = b["model"].map(FAMILY)
    order = b.groupby("family")["mean"].median().sort_values(ascending=False).index.tolist()
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", zorder=0)
    data = [b[b["family"] == f]["mean"].to_numpy() for f in order]
    bp = ax.boxplot(data, vert=True, patch_artist=True, widths=0.6, showfliers=False)
    for patch, f in zip(bp["boxes"], order):
        patch.set_facecolor(FAMILY_COLOR[f])
        patch.set_alpha(0.75)
    for med in bp["medians"]:
        med.set_color(INK)
        med.set_linewidth(1.4)
    for f, d, xi in zip(order, data, range(1, len(order) + 1)):
        ax.scatter([xi] * len(d), d, color=INK2, s=12, zorder=3, alpha=0.6)
    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels(order, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Macro-F1 (best strategy per model)")
    _titles(ax, "Performance by model family",
            "boosting & trees lead; do DL/transformers earn their cost?")
    return _save(fig, "06_family_boxplot")


BENCHMARK_A_FIGURES = [
    fig_ranking_f1, fig_ranking_small_multiples, fig_strategy_heatmap,
    fig_perclass_discrimination, fig_strategy_effect, fig_family_boxplot,
]


def _oof(models, strategies, cols):
    """Carrega SÓ os modelos/estratégias/colunas pedidos do oof (1,75GB) via filtro."""
    return pd.read_parquet(
        DATA_DIR / "oof_predictions.parquet", columns=cols,
        filters=[("model", "in", list(models)), ("strategy", "in", list(strategies))])


def _reliability(p, yb, n_bins=10):
    import numpy as np
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    xs, ys = [], []
    for b in range(n_bins):
        mask = idx == b
        if mask.sum() > 50:
            xs.append(p[mask].mean())
            ys.append(yb[mask].mean())
    return np.array(xs), np.array(ys)


def fig_reliability_curves() -> Path:
    """Fig 7 — curvas de confiabilidade (previsto × observado)."""
    import numpy as np
    agg, _ = _metrics()
    b = _best_strategy(agg).sort_values("mean", ascending=False)
    top3 = b.head(3)[["model", "strategy"]].values.tolist()
    lead_model, lead_strat = top3[0]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.2, 5.6))
    # painel 1: modelo líder, 3 classes
    d = _oof([lead_model], [lead_strat], ["y_true", "proba_0", "proba_1", "proba_2"])
    for cls in ["0", "1", "2"]:
        yb = (d["y_true"].to_numpy() == int(cls)).astype(float)
        xs, ys = _reliability(d[f"proba_{cls}"].to_numpy(), yb)
        a1.plot(xs, ys, color=CLS_COLOR[cls], lw=2, marker="o", ms=4, label=CLS_LABEL[cls], zorder=3)
    a1.plot([0, 1], [0, 1], color=INK2, ls="--", lw=1, zorder=1)
    a1.set_title(f"{lead_model} · all outcomes", fontsize=11, loc="left")
    a1.set_xlabel("Predicted probability")
    a1.set_ylabel("Observed frequency")
    a1.legend(fontsize=9, frameon=False)
    a1.grid(zorder=0)
    a1.set_xlim(0, 1)
    a1.set_ylim(0, 1)
    # painel 2: top-3 modelos, óbito-TB
    palette = ["#2a78d6", "#008300", "#e34948"]
    for (mo, st), c in zip(top3, palette):
        d = _oof([mo], [st], ["y_true", "proba_1"])
        yb = (d["y_true"].to_numpy() == 1).astype(float)
        xs, ys = _reliability(d["proba_1"].to_numpy(), yb)
        a2.plot(xs, ys, color=c, lw=2, marker="o", ms=4, label=mo, zorder=3)
    a2.plot([0, 1], [0, 1], color=INK2, ls="--", lw=1, zorder=1)
    a2.set_title("Top-3 models · TB death", fontsize=11, loc="left")
    a2.set_xlabel("Predicted probability")
    a2.grid(zorder=0)
    a2.legend(fontsize=9, frameon=False)
    a2.set_xlim(0, np.nanmax([0.3]))
    a2.set_ylim(0, None)
    fig.suptitle("Calibration reliability after cross-fitted calibration", x=0.07, ha="left",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save(fig, "07_reliability_curves")


def fig_calibration_slope() -> Path:
    """Fig 8 — slope de calibração por modelo (óbito-TB); slope~1 = bem calibrado."""
    agg, _ = _metrics()
    b = _best_strategy(agg)
    d = agg[(agg["metric"] == "cal_slope") & (agg["class"] == "1") & (agg["stage"] == "post_renorm")]
    d = b.merge(d[["model", "strategy", "mean", "std"]], on=["model", "strategy"], suffixes=("_f1", ""))
    d = d.dropna(subset=["mean"]).sort_values("mean")
    fig, ax = plt.subplots(figsize=(8.2, 7.2))
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", zorder=0)
    y = range(len(d))
    colors = [FAMILY_COLOR[FAMILY[m]] for m in d["model"]]
    ax.errorbar(d["mean"], list(y), xerr=d["std"], fmt="o", ecolor="#c9c8c4",
                elinewidth=1.2, ms=6, mfc="none", zorder=3)
    ax.scatter(d["mean"], list(y), color=colors, s=40, zorder=4)
    ax.axvline(1.0, color=INK2, ls="--", lw=1.2)
    ax.text(1.0, len(d) - 0.5, " perfect\n slope = 1", fontsize=8, color=INK2, va="top")
    ax.set_yticks(list(y))
    ax.set_yticklabels(d["model"], fontsize=8)
    ax.set_xlabel("Calibration slope · TB death (mean ± sd across folds)")
    _titles(ax, "Calibration slope by model (TB-death outcome)",
            "slope near 1 = well-calibrated; <1 = over-confident, >1 = under-confident")
    return _save(fig, "08_calibration_slope")


def fig_ece_before_after() -> Path:
    """Fig 9 — ECE antes × depois da renormalização (a distorção vira número)."""
    agg, _ = _metrics()
    b = _best_strategy(agg).sort_values("mean", ascending=False).head(14)
    fig, ax = plt.subplots(figsize=(8.6, 6.4))
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", zorder=0)
    for i, (_, r) in enumerate(b.iterrows()):
        def ece(stage):
            d = agg[(agg["model"] == r["model"]) & (agg["strategy"] == r["strategy"]) &
                    (agg["metric"] == "ece_fixed") & (agg["class"] == "1") & (agg["stage"] == stage)]
            return d["mean"].iloc[0] if len(d) else float("nan")
        pre, post = ece("pre_renorm"), ece("post_renorm")
        ax.plot([pre, post], [i, i], color="#c9c8c4", lw=1.5, zorder=2)
        ax.scatter(pre, i, color="#eb6834", s=38, zorder=3, label="pre-renorm" if i == 0 else "")
        ax.scatter(post, i, color="#2a78d6", s=38, zorder=3, label="post-renorm" if i == 0 else "")
    ax.set_yticks(range(len(b)))
    ax.set_yticklabels(b["model"], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("ECE · TB death")
    ax.legend(fontsize=9, frameon=False, loc="lower right")
    _titles(ax, "Expected calibration error: before vs after renormalization",
            "the SP4d design measures the distortion instead of assuming it away")
    return _save(fig, "09_ece_before_after")


def fig_ece_fixed_adaptive() -> Path:
    """Fig 10 — ECE fixo vs adaptativo por modelo (óbito-TB, pós-renorm)."""
    agg, _ = _metrics()
    b = _best_strategy(agg).sort_values("mean", ascending=False).head(14)
    fig, ax = plt.subplots(figsize=(8.6, 6.0))
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", zorder=0)
    x = range(len(b))
    w = 0.4
    for k, (metric, color, lab) in enumerate([("ece_fixed", "#2a78d6", "fixed bins"),
                                              ("ece_adaptive", "#1baf7a", "adaptive (quantile)")]):
        vals = []
        for _, r in b.iterrows():
            d = agg[(agg["model"] == r["model"]) & (agg["strategy"] == r["strategy"]) &
                    (agg["metric"] == metric) & (agg["class"] == "1") & (agg["stage"] == "post_renorm")]
            vals.append(d["mean"].iloc[0] if len(d) else 0)
        ax.bar([xi + (k - 0.5) * w for xi in x], vals, width=w, color=color, label=lab, zorder=3)
    ax.set_xticks(list(x))
    ax.set_xticklabels(b["model"], rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("ECE · TB death")
    ax.legend(fontsize=9, frameon=False)
    _titles(ax, "ECE: fixed vs adaptive bins (TB death)",
            "descriptive calibration error; log-loss/Brier are the primary proper scores")
    return _save(fig, "10_ece_fixed_adaptive")


def fig_guard_fractions() -> Path:
    """Fig 11 — guarda isotônica↔Platt: fração que caiu para Platt por escassez."""
    import json
    man = json.load(open(DATA_DIR / "calibration_manifest.json"))
    df = pd.DataFrame(man["manifest"])
    g = df.groupby("model")["frac_isotonic"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(8.2, 6.4))
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", zorder=0)
    y = range(len(g))
    ax.barh(list(y), g.to_numpy(), color="#1baf7a", height=0.7, zorder=3, label="isotonic")
    ax.barh(list(y), 1 - g.to_numpy(), left=g.to_numpy(), color="#eda100", height=0.7,
            zorder=3, label="Platt (sparse class)")
    ax.set_yticks(list(y))
    ax.set_yticklabels(g.index, fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Fraction of calibrator fits")
    ax.legend(fontsize=9, frameon=False, loc="lower right")
    _titles(ax, "Isotonic↔Platt guard: where the rare class forced a fallback",
            "the guard turns the data-scarcity decision into a number (SP4d)")
    return _save(fig, "11_guard_fractions")


BENCHMARK_B_FIGURES = [
    fig_reliability_curves, fig_calibration_slope, fig_ece_before_after,
    fig_ece_fixed_adaptive, fig_guard_fractions,
]


def fig_fold_variability() -> Path:
    """Fig 12 — variabilidade do F1-macro entre as 5 dobras espaciais (top-12)."""
    _, pf = _metrics()
    agg, _ = _metrics()
    b = _best_strategy(agg).sort_values("mean", ascending=False).head(12)
    gl = pf[(pf["class"] == "__global__") & (pf["axis"] == "discrimination")]
    fig, ax = plt.subplots(figsize=(8.8, 6.0))
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", zorder=0)
    data, labels, colors = [], [], []
    for _, r in b.iloc[::-1].iterrows():
        vals = gl[(gl["model"] == r["model"]) & (gl["strategy"] == r["strategy"])]["f1_macro"].to_numpy()
        data.append(vals)
        labels.append(r["model"])
        colors.append(FAMILY_COLOR[FAMILY[r["model"]]])
    for i, (vals, c) in enumerate(zip(data, colors)):
        ax.scatter(vals, [i] * len(vals), color=c, s=34, alpha=0.7, zorder=3)
        ax.hlines(i, vals.min(), vals.max(), color=c, lw=1.4, alpha=0.5, zorder=2)
        ax.scatter(vals.mean(), i, color=c, s=90, marker="|", zorder=4)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Macro-F1 per outer fold (dot) · | = mean")
    _titles(ax, "Stability across the 5 spatial folds",
            "spread = sensitivity to which regions are held out (the point of nested spatial CV)")
    return _save(fig, "12_fold_variability")


def _brazil_scatter(ax, g, values, cmap, vmin, vmax, size=3):
    x = g.geometry.x.to_numpy()
    y = g.geometry.y.to_numpy()
    sc = ax.scatter(x, y, c=values, cmap=cmap, s=size, vmin=vmin, vmax=vmax, linewidths=0)
    ax.set_aspect("equal")
    ax.axis("off")
    return sc


def fig_cluster_map() -> Path:
    """Fig 13 — mapa dos 50 clusters espaciais (municípios coloridos por cluster)."""
    import geopandas as gpd
    g = gpd.read_file(DATA_DIR / "municipality_clusters_k50.geojson")
    fig, ax = plt.subplots(figsize=(7.6, 7.6))
    # cor ilustrativa do bloco espacial (50 grupos; não é para identificar cluster individual)
    _brazil_scatter(ax, g, g["cluster"].to_numpy(), "tab20b", 0, 49, size=4)
    _titles(ax, "The 50 spatial validation clusters (KMeans on municipality lat/lon)",
            "colour = cluster (illustrative) · each cluster is kept whole in one CV fold")
    return _save(fig, "13_cluster_map")


def fig_cluster_heterogeneity() -> Path:
    """Fig 14 — mapa da taxa de óbito-TB por cluster (heterogeneidade territorial)."""
    import geopandas as gpd
    g = gpd.read_file(DATA_DIR / "municipality_clusters_k50.geojson")
    a = _analytic()
    a = a.copy()
    a["tb_death"] = (a[f"target_{SCHEMA}"] == "tb_death").astype(float)
    muni_cluster = g.set_index("ID_MUNIC_ANALISE")["cluster"]
    a["cluster"] = a["ID_MUNIC_ANALISE"].map(muni_cluster)
    rate = a.groupby("cluster")["tb_death"].mean() * 100
    g["rate"] = g["cluster"].map(rate)
    fig, ax = plt.subplots(figsize=(7.8, 7.6))
    sc = _brazil_scatter(ax, g, g["rate"].to_numpy(), "YlOrRd", 2, 8, size=4)
    fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02, label="TB-death rate per cluster (%)")
    _titles(ax, "Territorial heterogeneity of TB death is modest",
            "cluster-level rate mostly 3–6% · consistent with the ~1% between-cluster ICC")
    return _save(fig, "14_cluster_heterogeneity")


def fig_confusion() -> Path:
    """Fig 18 — matriz de confusão do líder: argmax vs política de limiar."""
    from sklearn.metrics import confusion_matrix
    agg, _ = _metrics()
    lead = _best_strategy(agg).sort_values("mean", ascending=False).iloc[0]
    d = _oof([lead["model"]], [lead["strategy"]], ["y_true", "pred_argmax", "pred_policy"])
    labs = [0, 1, 2]
    names = ["Cure", "TB death", "Interrupt."]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.2))
    for ax, col, title in zip(axes, ["pred_argmax", "pred_policy"], ["argmax", "threshold policy (§15.5)"]):
        cm = confusion_matrix(d["y_true"], d[col], labels=labs, normalize="true")
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=10,
                        color="white" if cm[i, j] > 0.5 else INK)
        ax.set_xticks(range(3))
        ax.set_xticklabels(names, fontsize=9)
        ax.set_yticks(range(3))
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xlabel("Predicted")
        ax.set_title(title, fontsize=11, loc="left")
        ax.grid(False)
    axes[0].set_ylabel("True outcome")
    fig.suptitle(f"Confusion (row-normalized) · {lead['model']} — the policy recovers rare classes",
                 x=0.06, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save(fig, "18_confusion_matrix")


def fig_proba_distributions() -> Path:
    """Fig 19 — distribuição da proba prevista de óbito-TB por classe verdadeira."""
    import numpy as np
    agg, _ = _metrics()
    lead = _best_strategy(agg).sort_values("mean", ascending=False).iloc[0]
    d = _oof([lead["model"]], [lead["strategy"]], ["y_true", "proba_1"])
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", zorder=0)
    for cls in [0, 2, 1]:
        vals = d[d["y_true"] == cls]["proba_1"].to_numpy()
        ax.hist(vals, bins=np.linspace(0, 0.6, 50), density=True, histtype="stepfilled",
                alpha=0.5, color=CLS_COLOR[str(cls)], label=f"true = {CLS_LABEL[str(cls)]}", zorder=3)
    ax.set_xlim(0, 0.6)
    ax.set_xlabel("Predicted probability of TB death")
    ax.set_ylabel("Density")
    ax.legend(fontsize=9, frameon=False)
    _titles(ax, f"Predicted TB-death probability separates the classes · {lead['model']}",
            "true TB-death cases (red) shifted right, but overlap explains the modest recall")
    return _save(fig, "19_proba_distributions")


def fig_policy_vs_argmax() -> Path:
    """Fig 20 — recall por classe: argmax vs política de limiar (o ganho nas raras)."""
    import numpy as np
    from sklearn.metrics import confusion_matrix
    agg, _ = _metrics()
    lead = _best_strategy(agg).sort_values("mean", ascending=False).iloc[0]
    d = _oof([lead["model"]], [lead["strategy"]], ["y_true", "pred_argmax", "pred_policy"])
    rec = {}
    for col in ["pred_argmax", "pred_policy"]:
        cm = confusion_matrix(d["y_true"], d[col], labels=[0, 1, 2], normalize="true")
        rec[col] = np.diag(cm)
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", zorder=0)
    x = np.arange(3)
    w = 0.38
    ax.bar(x - w / 2, rec["pred_argmax"], w, color="#9ec5f4", label="argmax", zorder=3)
    ax.bar(x + w / 2, rec["pred_policy"], w, color="#eb6834", label="threshold policy", zorder=3)
    for i in range(3):
        ax.annotate("", xy=(i + w / 2, rec["pred_policy"][i]), xytext=(i - w / 2, rec["pred_argmax"][i]),
                    arrowprops=dict(arrowstyle="->", color=INK2, lw=1))
    ax.set_xticks(x)
    ax.set_xticklabels(["Cure", "TB death", "Interruption"])
    ax.set_ylabel("Recall")
    ax.legend(fontsize=9, frameon=False)
    _titles(ax, f"Threshold policy trades cure-recall for the rare classes · {lead['model']}",
            "§15.5 policy raises TB-death & interruption recall — the actionable outcomes")
    return _save(fig, "20_policy_vs_argmax")


def fig_pr_roc() -> Path:
    """Fig 21 — curvas PR e ROC de óbito-TB para os top-3 modelos."""
    from sklearn.metrics import precision_recall_curve, roc_curve
    agg, _ = _metrics()
    top3 = _best_strategy(agg).sort_values("mean", ascending=False).head(3)[["model", "strategy"]].values.tolist()
    palette = ["#2a78d6", "#008300", "#e34948"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.2, 5.4))
    for (mo, st), c in zip(top3, palette):
        d = _oof([mo], [st], ["y_true", "proba_1"])
        yb = (d["y_true"].to_numpy() == 1).astype(int)
        p = d["proba_1"].to_numpy()
        pr, rc, _ = precision_recall_curve(yb, p)
        a1.plot(rc, pr, color=c, lw=2, label=mo, zorder=3)
        fpr, tpr, _ = roc_curve(yb, p)
        a2.plot(fpr, tpr, color=c, lw=2, label=mo, zorder=3)
    a1.axhline(yb.mean(), color=INK2, ls="--", lw=1)
    a1.set_xlabel("Recall")
    a1.set_ylabel("Precision")
    a1.set_title("Precision–Recall · TB death", fontsize=11, loc="left")
    a1.grid(zorder=0)
    a1.legend(fontsize=9, frameon=False)
    a2.plot([0, 1], [0, 1], color=INK2, ls="--", lw=1)
    a2.set_xlabel("False positive rate")
    a2.set_ylabel("True positive rate")
    a2.set_title("ROC · TB death", fontsize=11, loc="left")
    a2.grid(zorder=0)
    a2.legend(fontsize=9, frameon=False)
    fig.suptitle("Discrimination of TB death — top-3 models", x=0.06, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save(fig, "21_pr_roc")


BENCHMARK_CE_FIGURES = [
    fig_fold_variability, fig_cluster_map, fig_cluster_heterogeneity,
    fig_confusion, fig_proba_distributions, fig_policy_vs_argmax, fig_pr_roc,
]


# ---------------------------------------------------------------------------
# Robustez do leaderboard ao k espacial (SP4g) — do robustness_table (Task 6)
# ---------------------------------------------------------------------------


def _family_color(model: str) -> str:
    # FAMILY e FAMILY_COLOR são globais do módulo (definidos acima) -> uso direto.
    return FAMILY_COLOR.get(FAMILY.get(model, "baseline"), INK2)


def _spread_labels(ys, gap):
    """Separa rótulos que se sobrepõem, preservando a ordem vertical.

    Modelos empatados (as duas baselines) ou muito próximos (lightgbm e
    hist_gradient_boost diferem por ~0,001 de F1) imprimiam um texto por cima do
    outro. Empurra cada rótulo para pelo menos `gap` do anterior e recentra o bloco
    no seu centro original, para o deslocamento se repartir nos dois sentidos.
    """
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    out = list(ys)
    for a, b in zip(order, order[1:]):
        if out[b] - out[a] < gap:
            out[b] = out[a] + gap
    shift = (sum(ys) - sum(out)) / len(ys)
    return [v + shift for v in out]


def fig_leaderboard_across_k(robust_long, out_name: str = "25_leaderboard_across_k"):
    """Fig 25 — F1-macro de cada modelo em k∈{27,50,75}; linhas ~planas = robusto."""
    _style()
    df = robust_long.copy()
    ks = sorted(df["k"].unique())
    order = (df.groupby("model")["f1_macro"].max().sort_values(ascending=False).index)

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ax.grid(axis="x", visible=False)
    tags = []
    for model in order:
        d = df[df["model"] == model].sort_values("k")
        c = _family_color(model)
        ax.plot(d["k"].to_numpy(), d["f1_macro"].to_numpy(), color=c, lw=2, marker="o",
                ms=5, zorder=3)
        # cobertura irregular de k (um par que falhou num k some do robustness_table):
        # o rótulo ancora no ÚLTIMO k que o modelo tem, não no último k global.
        tags.append((float(d["k"].iloc[-1]) + 1.5, float(d["f1_macro"].iloc[-1]), model, c))
    ax.set_xticks(ks)
    ax.set_xlim(ks[0] - 2, ks[-1] + 16)

    lo, hi = ax.get_ylim()
    ys = _spread_labels([y for _, y, _, _ in tags], gap=(hi - lo) * 0.045)
    for (x, y0, model, c), y in zip(tags, ys):
        # fio de ligação quando o rótulo teve de sair de cima do fim da linha
        if abs(y - y0) > (hi - lo) * 0.004:
            ax.plot([x - 1.0, x - 0.2], [y0, y], color=c, lw=0.7, alpha=0.55, zorder=2)
        ax.text(x, y, model, color=c, va="center", fontsize=8.5, fontweight="bold")
    pad = (hi - lo) * 0.03
    ax.set_ylim(min(lo, min(ys) - pad), max(hi, max(ys) + pad))
    ax.set_xlabel("Spatial clustering granularity k (number of municipality clusters)")
    ax.set_ylabel("Macro-F1 (best strategy)")
    _titles(ax, "Leaderboard stability across spatial granularity",
            "TB, Brazil · nested spatial CV · flat lines = ranking robust to k")
    return _save(fig, out_name)


def fig_rank_bump(robust_long, out_name: str = "26_rank_bump_across_k"):
    """Fig 26 — bump chart do rank por k; cruzamentos = instabilidade do ranking."""
    _style()
    df = robust_long.copy()
    ks = sorted(df["k"].unique())

    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    ax.grid(visible=False)
    tags = []
    for model in df["model"].unique():
        d = df[df["model"] == model].sort_values("k")
        c = _family_color(model)
        ax.plot(d["k"].to_numpy(), d["rank"].to_numpy(), color=c, lw=2, marker="o",
                ms=6, zorder=3)
        # cobertura irregular de k: ancora no PRIMEIRO k que o modelo tem.
        tags.append((float(d["k"].iloc[0]) - 1.5, float(d["rank"].iloc[0]), model, c))

    # modelos empatados no mesmo rank (as duas baselines) imprimiriam um sobre o outro
    ys = _spread_labels([y for _, y, _, _ in tags], gap=0.42)
    for (x, y0, model, c), y in zip(tags, ys):
        if abs(y - y0) > 0.02:
            ax.plot([x + 0.2, x + 1.0], [y, y0], color=c, lw=0.7, alpha=0.55, zorder=2)
        ax.text(x, y, model, color=c, va="center", ha="right",
                fontsize=8.5, fontweight="bold")
    ax.set_xticks(ks)
    ax.set_xlim(ks[0] - 14, ks[-1] + 2)
    ax.invert_yaxis()  # rank 1 no topo
    # os ticks cobrem a faixa de ranks REALMENTE presente (não o nº de modelos no
    # primeiro k, que subestima quando um modelo só aparece em k posteriores).
    ax.set_yticks(range(1, int(df["rank"].max()) + 1))
    ax.set_xlabel("Spatial clustering granularity k")
    ax.set_ylabel("Leaderboard rank (1 = best)")
    _titles(ax, "Rank bump chart across spatial granularity",
            "TB, Brazil · parallel lines = stable ranks; crossings = rank changes")
    return _save(fig, out_name)


if __name__ == "__main__":
    for fn in COHORT_ICC_FIGURES:
        print(fn())
