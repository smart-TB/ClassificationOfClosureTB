"""Figuras do manuscrito e do suplemento a partir do catálogo liberado (L5).

As funções de `tb_outcomes.figures` leem, por caminho padrão, as tabelas históricas de
`data/`. Aqui nenhuma figura tem caminho padrão: os dados chegam pelo `Catalogo`, que recusa
o que não foi liberado. O estilo e a paleta são os de `figures.py`, para que as figuras novas
continuem lendo como parte do mesmo conjunto.

Cada figura grava, ao lado do PNG, o CSV com exatamente os valores desenhados. As saídas vão
para `manuscrito/PLOS_DigitalHealth/figures_c7/`, sem sobrescrever `artifacts/figures/`, que
permanece como registro da corrida histórica.

Uso:
    python scripts/editorial/figuras_c7.py            # todas
    python scripts/editorial/figuras_c7.py fig2 s1    # só as nomeadas
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Patch  # noqa: E402

from catalogo import C04, C23, REL, RAIZ, Catalogo  # noqa: E402
from dossie_c7 import CLASSES, PROC, Dossie, construir, shap_por_dobra  # noqa: E402

sys.path.insert(0, str(RAIZ / "src"))
from tb_outcomes import labels as rot  # noqa: E402
from tb_outcomes.figures import CAT, GRID, INK, INK2, OUTCOME, _style  # noqa: E402

SAIDA = RAIZ / "manuscrito/PLOS_DigitalHealth/figures_c7"
COR = {k: OUTCOME[v][1] for k, v in CLASSES.items()}
NOME = {0: "Cure", 1: "TB death", 2: "Treatment interruption"}
AZUL, CINZA = CAT[0], "#8a8985"
FAM_CURTA = {"lightgbm": "LightGBM", "catboost": "CatBoost"}
FAM_SIGLA = {"lightgbm": "LGBM", "catboost": "CatB"}


def _salvar(fig, nome: str, dados: pd.DataFrame) -> Path:
    SAIDA.mkdir(parents=True, exist_ok=True)
    (SAIDA / "data").mkdir(exist_ok=True)
    caminho = SAIDA / f"{nome}.png"
    fig.savefig(caminho, dpi=300, bbox_inches="tight", facecolor="white")
    dados.to_csv(SAIDA / "data" / f"{nome}.csv", index=False)
    plt.close(fig)
    return caminho


def _par(comparacao: str) -> tuple[str, str]:
    modelo, estrategia = comparacao.split("__")
    return modelo, estrategia


def _f1_pares(c: Catalogo, braco: str, regra: str = "argmax") -> pd.DataFrame:
    ci = c.csv(f"{c.auditoria(braco)}/cluster_intervals_pred_{regra}.csv")
    g = ci[(ci.metric == "f1_macro") & (ci.kind == "estimate")].copy()
    if not (g.status == "ok").all():
        raise ValueError("estimativa de macro-F1 não estimável no ranking")
    return g


# --------------------------------------------------------------------------- corpo
def fig1(c: Catalogo, d: Dossie) -> Path:
    """Fluxo da coorte, das notificações recuperadas aos três papéis de análise."""
    fig, ax = plt.subplots(figsize=(9.2, 6.6))
    ax.set_axis_off()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    def caixa(x, y, w, h, titulo, n, cor="#eaf2fc", borda=AZUL):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    fc=cor, ec=borda, lw=1.2))
        ax.text(x + w / 2, y + h * 0.63, titulo, ha="center", va="center", fontsize=10, color=INK)
        ax.text(x + w / 2, y + h * 0.27, f"n = {n:,}", ha="center", va="center", fontsize=11,
                color=INK, fontweight="bold")

    def lado(x, y, w, h, linhas):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    fc="#f6f5f2", ec=CINZA, lw=0.9))
        passo = h / (len(linhas) + 0.6)
        for i, (txt, n) in enumerate(linhas):
            yy = y + h - passo * (i + 0.8)
            ax.text(x + 0.12, yy, txt, ha="left", va="center", fontsize=8.6, color=INK2)
            ax.text(x + w - 0.12, yy, f"{n:,}", ha="right", va="center", fontsize=8.6, color=INK)

    def seta(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.1))

    caixa(0.3, 8.5, 4.0, 1.2, "Records returned by acquisition", d["coorte.recuperada"])
    caixa(0.3, 5.75, 4.0, 1.2, "Eligible notifications", d["coorte.elegivel"])
    caixa(0.3, 3.0, 4.0, 1.2, "Analytic cohort, three-class outcome", d["coorte.analitica"])
    seta(2.3, 8.5, 2.3, 6.95)
    seta(2.3, 5.75, 2.3, 4.2)

    motivos = [(rot.motivo_exclusao(k), d[f"coorte.excl.{k}"])
               for k in ("municipio_irrecuperavel", "idade_impossivel", "fora_da_janela_declarada",
                         "notificacao_pos_obito", "seguimento_insuficiente")]
    motivos[2] = ("Outside the 2015–2025 window", motivos[2][1])
    lado(5.0, 6.95, 4.7, 1.9, motivos)
    ax.text(5.0, 8.95, "Excluded, criteria applied in sequence", fontsize=8.8, color=INK2,
            ha="left", va="bottom", style="italic")
    seta(2.3, 7.75, 5.0, 7.75)

    lado(5.0, 4.45, 4.7, 1.0,
         [("Closure outside the three-outcome scheme", d["coorte.fora_esquema.outra_categoria"]),
          ("No closure recorded (unresolved)", d["coorte.fora_esquema.nao_resolvido"])])
    ax.text(5.0, 5.52, "Not in the modelled outcome", fontsize=8.8, color=INK2, ha="left",
            va="bottom", style="italic")
    seta(2.3, 4.95, 5.0, 4.95)

    papeis = [("Development\n2015–2023", d["coorte.dev"], "#eaf2fc", AZUL),
              ("Temporal evaluation\n2024", d["coorte.temporal"], "#eaf2fc", AZUL),
              ("Not used in any\nevaluation, 2025", d["coorte.futuro"], "#f6f5f2", CINZA)]
    for i, (t, n, cor, borda) in enumerate(papeis):
        x = 0.3 + i * 3.2
        caixa(x, 0.3, 2.9, 1.5, t, n, cor, borda)
        seta(2.3, 3.0, x + 1.45, 1.8)

    dados = pd.DataFrame(
        [("recovered", d["coorte.recuperada"]), *[(f"excluded: {m}", n) for m, n in motivos],
         ("eligible", d["coorte.elegivel"]),
         ("outside three-outcome scheme", d["coorte.fora_esquema.outra_categoria"]),
         ("unresolved", d["coorte.fora_esquema.nao_resolvido"]),
         ("analytic", d["coorte.analitica"]), ("development 2015-2023", d["coorte.dev"]),
         ("temporal 2024", d["coorte.temporal"]), ("unused 2025", d["coorte.futuro"])],
        columns=["item", "n"])
    return _salvar(fig, "fig1_cohort_flow", dados)


def _ranking(c: Catalogo, d: Dossie, braco: str, nome: str, titulo_ref: str) -> Path:
    g = _f1_pares(c, braco)
    proc = g[g.comparison == PROC].iloc[0]
    g = g[g.comparison != PROC].copy()
    g[["model", "strategy"]] = [list(_par(x)) for x in g.comparison]
    best = (g.sort_values("estimate", ascending=False).drop_duplicates("model")
             .sort_values("estimate").reset_index(drop=True))
    fig, ax = plt.subplots(figsize=(8.4, 0.30 * len(best) + 1.9))
    ax.grid(axis="y", visible=False)
    ax.axvspan(proc.ci_low, proc.ci_high, color=AZUL, alpha=0.13, lw=0, zorder=1)
    ax.axvline(proc.estimate, color=AZUL, lw=1.4, zorder=2)
    ref = best[best.model == "majority_class"].iloc[0]
    ax.axvline(ref.estimate, color=INK2, lw=1.0, ls=(0, (4, 3)), zorder=2)
    y = np.arange(len(best))
    ax.errorbar(best.estimate, y, xerr=[best.estimate - best.ci_low, best.ci_high - best.estimate],
                fmt="o", ms=4.5, color=INK, ecolor=INK2, elinewidth=1.1, capsize=0, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{rot.algoritmo(m)}  ·  {rot.estrategia(s).lower()}"
                        for m, s in zip(best.model, best.strategy)], fontsize=8.6)
    ax.set_ylim(-0.8, len(best) - 0.2)
    ax.set_xlabel("Macro-averaged F1, arg max rule, pooled records (95% cluster-bootstrap interval)")
    topo = len(best) - 0.2
    ax.text(proc.estimate, topo, " nested selection procedure", color=AZUL, fontsize=8.6,
            ha="left", va="bottom")
    ax.text(ref.estimate, topo, titulo_ref + " ", color=INK2, fontsize=8.6, ha="right", va="bottom")
    dados = best[["model", "strategy", "estimate", "ci_low", "ci_high"]].assign(arm=braco)
    dados.loc[len(dados)] = [PROC, "selected_in_development", proc.estimate, proc.ci_low,
                             proc.ci_high, braco]
    return _salvar(fig, nome, dados)


def fig2(c: Catalogo, d: Dossie) -> Path:
    return _ranking(c, d, "primary", "fig2_exploratory_ranking", "majority-class reference")


def fig3(c: Catalogo, d: Dossie) -> Path:
    b = c.csv(f"{REL}/primary_calibration_bins.csv")
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.7))
    for k, ax in zip((0, 1, 2), axes):
        g = b[b.class_code == k]
        lim = max(g.mean_probability.max(), g.event_rate.max()) * 1.06
        ax.plot([0, lim], [0, lim], color=INK2, lw=0.9, ls=(0, (4, 3)), zorder=1)
        for estagio, kw, rotulo in (
                ("before_calibration", dict(mfc="white", ls="--"), "Before calibration"),
                ("after_calibration", dict(mfc=COR[k], ls="-"), "After calibration")):
            x = g[g.stage == estagio].sort_values("bin")
            ax.plot(x.mean_probability.to_numpy(), x.event_rate.to_numpy(), marker="o", ms=4.5, color=COR[k], lw=1.3,
                    mec=COR[k], label=rotulo, zorder=3, **kw)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_aspect("equal")
        ax.set_title(NOME[k], fontsize=10.5)
        ax.set_xlabel("Mean predicted probability")
        if k == 0:
            ax.set_ylabel("Observed event rate")
        ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    return _salvar(fig, "fig3_calibration", b)


ORDEM_BRACOS = ["individual", "municipal", "contemporary", "without_diagnostics",
                "hiv_uncertain", "k27", "k75", "random_group"]


def fig4(c: Catalogo, d: Dossie) -> Path:
    lin = []
    for b in ORDEM_BRACOS:
        for regra in ("argmax", "policy"):
            k = f"{b}.{regra}.dif"
            lin.append((b, regra, d[k], d[k + ".lo"], d[k + ".hi"]))
    t = pd.DataFrame(lin, columns=["arm", "rule", "difference", "ci_low", "ci_high"])
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.3), sharey=True,
                             gridspec_kw=dict(width_ratios=[1, 1.25], wspace=0.06))
    y0 = {b: i for i, b in enumerate(ORDEM_BRACOS[::-1])}
    for ax, zoom in zip(axes, (False, True)):
        ax.grid(axis="y", visible=False)
        ax.axvline(0, color=INK2, lw=1.0)
        for regra, dy, kw in (("argmax", 0.14, dict(mfc=INK)), ("policy", -0.14, dict(mfc="white"))):
            g = t[t.rule == regra]
            if zoom:
                g = g[g.arm != "municipal"]
            yy = [y0[b] + dy for b in g.arm]
            ax.errorbar(g.difference, yy, xerr=[g.difference - g.ci_low, g.ci_high - g.difference],
                        fmt="o", ms=4.8, color=INK, mec=INK, ecolor=INK2, elinewidth=1.1,
                        label="Arg max rule (primary)" if regra == "argmax" else "Threshold policy (secondary)",
                        **kw)
        ax.set_ylim(-0.6, len(ORDEM_BRACOS) - 0.4)
        ax.axhline(2.5, color=GRID, lw=1.0)
    axes[0].set_yticks(list(y0.values()))
    axes[0].set_yticklabels([rot.braco(b) for b in y0], fontsize=9)
    axes[0].set_title("All arms", fontsize=10.5)
    axes[1].set_title("Detail, municipal-only arm omitted", fontsize=10.5)
    axes[1].set_xlim(-0.016, 0.016)
    axes[1].tick_params(left=False)
    axes[1].legend(frameon=False, fontsize=8.4, loc="lower right")
    fig.supxlabel("Paired difference in pooled macro-averaged F1 against the primary arm "
                  "(95% cluster-bootstrap interval)", fontsize=10, color=INK2, y=-0.02)
    return _salvar(fig, "fig4_arm_differences", t)


def fig5(c: Catalogo, d: Dossie) -> Path:
    bracos = [("primary", "Individual + municipal (primary)", INK),
              ("individual", "Individual only", CAT[0]), ("municipal", "Municipal only", CAT[5])]
    lin = []
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.9), gridspec_kw=dict(width_ratios=[1.05, 1, 1]))
    ax = axes[0]
    ax.grid(axis="x", visible=False)
    for i, (b, nome, cor) in enumerate(bracos):
        v = [d[f"{b}.prob.{CLASSES[k]}.auroc"] for k in (0, 1, 2)]
        ax.plot(np.arange(3) + (i - 1) * 0.16, v, "o", ms=6, color=cor, label=nome)
        lin += [(b, "auroc", CLASSES[k], x) for k, x in zip((0, 1, 2), v)]
    ax.axhline(0.5, color=INK2, lw=1.0, ls=(0, (4, 3)))
    ax.text(2.42, 0.5, "chance", fontsize=8, color=INK2, ha="right", va="bottom")
    ax.set_xticks(range(3))
    ax.set_xticklabels(["Cure", "TB death", "Treatment\ninterruption"], fontsize=9)
    ax.set_xlim(-0.45, 2.45)
    ax.set_ylim(0.45, 0.85)
    ax.set_ylabel("Area under the ROC curve")
    ax.set_title("Discrimination", fontsize=10.5)
    for ax, k in zip(axes[1:], (1, 2)):
        caps = (5, 10, 20)
        for b, nome, cor in bracos:
            v = [100 * d[f"{b}.cap{cp}.{CLASSES[k]}.capturado"] for cp in caps]
            ax.plot(caps, v, "o-", ms=5.5, lw=1.3, color=cor, label=nome)
            lin += [(b, f"captured_pct_cap{cp}", CLASSES[k], x) for cp, x in zip(caps, v)]
        ax.plot(caps, caps, color=INK2, lw=1.0, ls=(0, (4, 3)))
        ax.text(20, 20, "no ranking ", fontsize=8, color=INK2, ha="right", va="bottom")
        ax.set_xticks(caps)
        ax.set_xticklabels([f"{x}%" for x in caps])
        ax.set_xlabel("Highest-risk fraction selected")
        ax.set_ylim(0, 70)
        ax.set_title(f"Share of observed events selected: {NOME[k].lower()}"
                     if k == 1 else NOME[k], fontsize=10.5)
        ax.yaxis.set_major_formatter(lambda x, _: f"{x:.0f}%")
    axes[1].set_title("Events selected: TB death", fontsize=10.5)
    axes[2].set_title("Events selected: treatment interruption", fontsize=10.5)
    axes[0].legend(frameon=False, fontsize=8.2, loc="center left", bbox_to_anchor=(0.0, 0.5))
    fig.tight_layout()
    return _salvar(fig, "fig5_feature_sets",
                   pd.DataFrame(lin, columns=["arm", "quantity", "class", "value"]))


def _painel_postos(ax, g: pd.DataFrame, classe: str, n: int, familias: list[str]) -> pd.DataFrame:
    x = g[g.classe == classe]
    mediana = x.groupby("feature")["rank"].median().sort_values()
    feats = mediana.head(n).index.tolist()
    m = (x[x.feature.isin(feats)].pivot(index="feature", columns="outer_fold", values="rank")
          .loc[feats])
    escopo = x.drop_duplicates("feature").set_index("feature")["scope"]
    im = ax.imshow(m.values, cmap="Blues_r", vmin=1, vmax=25, aspect="auto")
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = int(m.values[i, j])
            ax.text(j, i, str(v), ha="center", va="center", fontsize=8,
                    color="white" if v <= 9 else INK)
    ax.set_xticks(range(5))
    ax.set_xticklabels([f"{k}\n{FAM_SIGLA.get(f, f)}" for k, f in enumerate(familias)], fontsize=7.6)
    ax.set_yticks(range(len(feats)))
    ax.set_yticklabels([rot.preditor(f) + (" (M)" if escopo[f] == "municipal" else "")
                        for f in feats], fontsize=8.3)
    ax.grid(False)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    del im
    return m.reset_index().assign(classe=classe)


def fig6(c: Catalogo, d: Dossie) -> Path:
    g = shap_por_dobra(c)
    fam = d["principal.selecao.familias"]
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 5.4))
    partes = []
    for ax, k in zip(axes, (1, 2, 0)):
        partes.append(_painel_postos(ax, g, CLASSES[k], 15, fam))
        ax.set_title(NOME[k], fontsize=10.5)
        ax.set_xlabel("Outer fold and explained family", fontsize=8.6)
    fig.tight_layout(w_pad=1.2)
    return _salvar(fig, "fig6_attribution_ranks", pd.concat(partes, ignore_index=True))


def fig7(c: Catalogo, d: Dossie) -> Path:
    racas = ["1", "2", "3", "4", "5"]
    lin = []
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.6), sharey=True)
    y = np.arange(len(racas))[::-1]
    for r in racas:
        p = f"sub.raca_cor.{r}.tb_death"
        lin.append((r, rot.raca_cor(r), d[p + ".n"], d[p + ".events"], 100 * d[p + ".prevalence"],
                    d[p + ".auroc"], d[p + ".policy.sensitivity"], d[p + ".policy.sensitivity.lo"],
                    d[p + ".policy.sensitivity.hi"], d[p + ".policy.ppv"], d[p + ".policy.ppv.lo"],
                    d[p + ".policy.ppv.hi"]))
    t = pd.DataFrame(lin, columns=["code", "group", "n", "deaths", "prevalence_pct", "auroc",
                                   "sensitivity", "sens_lo", "sens_hi", "ppv", "ppv_lo", "ppv_hi"])
    ax = axes[0]
    ax.grid(axis="y", visible=False)
    ax.plot(t.auroc.to_numpy(), y, "o", ms=6.5, color=COR[1])
    ax.set_xlim(0.68, 0.84)
    ax.set_xlabel("Area under the ROC curve")
    ax.set_title("Discrimination", fontsize=10.5)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{g}\n(n = {n:,}; {e:,} deaths)" for g, n, e in zip(t.group, t.n, t.deaths)],
                       fontsize=8.4)
    for ax, col, lo, hi, titulo in ((axes[1], "sensitivity", "sens_lo", "sens_hi", "Sensitivity"),
                                    (axes[2], "ppv", "ppv_lo", "ppv_hi", "Positive predictive value")):
        ax.grid(axis="y", visible=False)
        ax.errorbar(t[col], y, xerr=[t[col] - t[lo], t[hi] - t[col]], fmt="o", ms=5.5,
                    color=COR[1], ecolor=INK2, elinewidth=1.1)
        ax.set_xlim(0, 0.42)
        ax.set_xlabel(f"{titulo}, threshold policy")
        ax.set_title(titulo, fontsize=10.5)
    axes[0].set_ylim(-0.6, len(racas) - 0.4)
    fig.tight_layout()
    return _salvar(fig, "fig7_race_colour", t)


def fig8(c: Catalogo, d: Dossie) -> Path:
    dc = c.csv(f"{C23}/primary_vs_clinical/decision_curves.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.9))
    for ax, k, xmax in zip(axes, (1, 2), (0.30, 0.50)):
        g = dc[(dc.class_code == k) & (dc.threshold <= xmax + 1e-9)]
        p = g[g.comparison == "primary"].sort_values("threshold")
        q = g[g.comparison == "clinical"].sort_values("threshold")
        ax.plot(p.threshold.to_numpy(), p.nb_intervir_em_todos.to_numpy(), color=CINZA, lw=1.3, ls=(0, (4, 3)),
                label="Flag every notification")
        ax.plot(p.threshold.to_numpy(), p.nb_nao_intervir.to_numpy(), color=INK2, lw=1.0, label="Flag none")
        ax.plot(q.threshold.to_numpy(), q.nb_modelo.to_numpy(), color=CAT[6], lw=1.6, label="Four-factor comparator")
        ax.plot(p.threshold.to_numpy(), p.nb_modelo.to_numpy(), color=COR[k], lw=1.8, label="Selection procedure")
        topo = max(p.nb_modelo.max(), q.nb_modelo.max())
        ax.set_ylim(-0.25 * topo, topo * 1.08)
        ax.set_xlim(0, xmax)
        ax.set_xlabel("Threshold probability")
        ax.set_title(NOME[k], fontsize=10.5)
        if k == 1:
            ax.set_ylabel("Net benefit")
        ax.legend(frameon=False, fontsize=8.2, loc="upper right")
    fig.tight_layout()
    return _salvar(fig, "fig8_decision_curves", dc)


# --------------------------------------------------------------------------- suplemento
def s1(c: Catalogo, d: Dossie) -> Path:
    g = _f1_pares(c, "primary")
    g = g[g.comparison != PROC].copy()
    g[["model", "strategy"]] = [list(_par(x)) for x in g.comparison]
    est = ["none", "random_oversampling", "random_undersampling", "local_cost_sensitive"]
    m = g.pivot(index="model", columns="strategy", values="estimate")[est]
    m = m.loc[m.max(axis=1).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(7.4, 9.2))
    im = ax.imshow(m.values, cmap="Blues", vmin=0.25, vmax=0.45, aspect="auto")
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = m.values[i, j]
            if np.isnan(v):
                ax.text(j, i, "not run", ha="center", va="center", fontsize=7.6, color=INK2,
                        style="italic")
            else:
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8.2,
                        color="white" if v > 0.39 else INK)
    ax.set_xticks(range(len(est)))
    ax.set_xticklabels([rot.estrategia(e) for e in est], fontsize=9)
    ax.set_yticks(range(m.shape[0]))
    ax.set_yticklabels([rot.algoritmo(x) for x in m.index], fontsize=8.6)
    ax.xaxis.tick_top()
    ax.grid(False)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cb.set_label("Pooled macro-averaged F1, arg max rule", fontsize=9)
    cb.outline.set_visible(False)
    return _salvar(fig, "s1_model_strategy_heatmap", m.reset_index())


def s2(c: Catalogo, d: Dossie) -> Path:
    lin = []
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6), sharey=True)
    for ax, met, titulo in zip(axes, ("sensitivity", "ppv"), ("Sensitivity", "Positive predictive value")):
        ax.grid(axis="x", visible=False)
        for regra, dx, mfc in (("argmax", -0.12, None), ("policy", 0.12, "white")):
            for k in (0, 1, 2):
                p = f"primary.{regra}.{CLASSES[k]}.{met}"
                v, lo, hi = d[p], d[p + ".lo"], d[p + ".hi"]
                ax.errorbar(k + dx, v, yerr=[[v - lo], [hi - v]], fmt="o", ms=6, color=COR[k],
                            mfc=mfc or COR[k], ecolor=INK2, elinewidth=1.1)
                lin.append((regra, CLASSES[k], met, v, lo, hi))
        ax.set_xticks(range(3))
        ax.set_xticklabels(["Cure", "TB death", "Treatment\ninterruption"], fontsize=9)
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylim(0, 1)
        ax.set_title(titulo, fontsize=10.5)
    axes[0].plot([], [], "o", color=INK, label="Arg max rule (primary)")
    axes[0].plot([], [], "o", color=INK, mfc="white", label="Threshold policy (secondary)")
    axes[0].legend(frameon=False, fontsize=8.4, loc="upper right")
    fig.tight_layout()
    return _salvar(fig, "s2_per_class_by_rule",
                   pd.DataFrame(lin, columns=["rule", "class", "metric", "estimate", "ci_low", "ci_high"]))


def metricas_por_dobra(c: Catalogo, braco: str = "primary") -> pd.DataFrame:
    sm = c.csv(f"{c.selecionado(braco)}/selected_metrics_per_fold.csv")
    g = sm[(sm["class"] == "__global__") & sm.axis.notna()]
    t = g.pivot(index="outer_fold", columns="axis", values="f1_macro").rename(
        columns={"discrimination": "argmax", "secondary_policy": "policy"})
    t["family"] = g.drop_duplicates("outer_fold").set_index("outer_fold")["selected_model"]
    return t.reset_index()


def s3(c: Catalogo, d: Dossie) -> Path:
    t = metricas_por_dobra(c)
    for regra in ("argmax", "policy"):  # a média entre dobras tem de ser a do catálogo
        if abs(t[regra].mean() - d[f"primary.{regra}.f1.fold_mean"]) > 1e-9:
            raise ValueError(f"média entre dobras diverge ({regra})")
    fm = c.csv(f"{c.auditoria('primary')}/recomputed_fold_metrics.csv")
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    ax.grid(axis="x", visible=False)
    for regra, cor, nome in (("argmax", INK, "Arg max rule (primary)"),
                             ("policy", AZUL, "Threshold policy (secondary)")):
        todos = fm[fm.rule == f"pred_{regra}"]
        for k in range(5):
            v = todos[todos.outer_fold == k].f1_macro.to_numpy()
            ax.plot(np.full(len(v), k) + (-0.17 if regra == "argmax" else 0.17), v, "_", ms=9,
                    color=cor, alpha=0.22)
        ax.plot(t.outer_fold.to_numpy() + (-0.17 if regra == "argmax" else 0.17), t[regra].to_numpy(), "o", ms=7,
                color=cor, label=nome, zorder=3)
    ax.plot([], [], "_", ms=9, color=INK2, alpha=0.5, label="Each of the 96 candidate pairs")
    ax.set_xticks(range(5))
    cl = d["bootstrap.clusters_por_dobra"]
    ax.set_xticklabels([f"Fold {k}\n{FAM_CURTA.get(f, f)}\n{n} cluster{'s' if n > 1 else ''}"
                        for k, f, n in zip(t.outer_fold, t.family, cl)], fontsize=8.6)
    ax.set_ylabel("Macro-averaged F1 in the held-out fold")
    ax.legend(frameon=False, fontsize=8.4, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)
    fig.tight_layout()
    return _salvar(fig, "s3_between_fold", t)


def s4(c: Catalogo, d: Dossie) -> Path:
    cal = c.csv(f"{REL}/primary_calibration_before_after.csv")
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.5))
    for ax, col, titulo, ref in zip(
            axes, ("ece_fixed_10", "calibration_slope", "calibration_in_large"),
            ("Expected calibration error\n(ten fixed bins)", "Calibration slope",
             "Calibration-in-the-large\n(offset, slope fixed at 1)"), (None, 1.0, 0.0)):
        ax.grid(axis="x", visible=False)
        if ref is not None:
            ax.axhline(ref, color=INK2, lw=1.0, ls=(0, (4, 3)))
        for k in (0, 1, 2):
            a = cal[(cal.stage == "before_calibration") & (cal.class_code == k)][col].iloc[0]
            b = cal[(cal.stage == "after_calibration") & (cal.class_code == k)][col].iloc[0]
            ax.plot([k - 0.13, k + 0.13], [a, b], color=COR[k], lw=1.2)
            ax.plot(k - 0.13, a, "o", ms=6.5, color=COR[k], mfc="white")
            ax.plot(k + 0.13, b, "o", ms=6.5, color=COR[k])
        ax.set_xticks(range(3))
        ax.set_xticklabels(["Cure", "TB death", "Treatment\ninterruption"], fontsize=8.6)
        ax.set_xlim(-0.5, 2.5)
        ax.set_title(titulo, fontsize=10)
    axes[0].set_ylim(0, None)
    axes[0].plot([], [], "o", color=INK, mfc="white", label="Before calibration")
    axes[0].plot([], [], "o", color=INK, label="After calibration")
    axes[0].legend(frameon=False, fontsize=8.4, loc="upper left")
    fig.tight_layout()
    return _salvar(fig, "s4_calibration_summary", cal)


def s5(c: Catalogo, d: Dossie) -> Path:
    inf = c.csv(f"{c.auditoria('primary')}/cluster_influence.csv")
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    ax.axhline(0, color=INK2, lw=1.0)
    for regra, cor, nome, mfc in (("pred_argmax", INK, "Arg max rule (primary)", INK),
                                  ("pred_policy", AZUL, "Threshold policy (secondary)", "white")):
        g = inf[inf.rule == regra]
        ax.plot(100 * g.record_fraction.to_numpy(), g.change.to_numpy(), "o", ms=5, color=cor, mfc=mfc, label=nome)
    ax.set_xscale("log")
    ax.set_xlabel("Share of development records in the cluster (%, logarithmic scale)")
    ax.set_ylabel("Change in pooled macro-averaged F1\nwhen the cluster is removed")
    ax.legend(frameon=False, fontsize=8.4)
    fig.tight_layout()
    return _salvar(fig, "s5_cluster_influence", inf)


def s6(c: Catalogo, d: Dossie) -> Path:
    return _ranking(c, d, "temporal", "s6_temporal_ranking", "majority-class reference")


IDADES = ["0-14", "15-24", "25-34", "35-44", "45-59", "60-64", "65+"]


def s7(c: Catalogo, d: Dossie) -> Path:
    am = c.csv(f"{C04}/age_race_stratification/stratified_metrics.csv")
    x = am[(am.decision_rule == "pred_policy") & (am.metric == "sensitivity:1")].copy()
    x["v"] = np.where(x.status == "ok", x.estimate, np.nan)
    m = x.pivot(index="race", columns="age", values="v").loc[[1, 2, 3, 4, 5], IDADES]
    fig, ax = plt.subplots(figsize=(8.0, 3.4))
    im = ax.imshow(np.ma.masked_invalid(m.values), cmap="Blues", vmin=0, vmax=0.6, aspect="auto")
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = m.values[i, j]
            if np.isnan(v):
                ax.text(j, i, "n/e", ha="center", va="center", fontsize=8, color=INK2, style="italic")
            else:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8.6,
                        color="white" if v > 0.36 else INK)
    ax.set_xticks(range(len(IDADES)))
    ax.set_xticklabels(IDADES)
    ax.set_yticks(range(5))
    ax.set_yticklabels([rot.raca_cor(str(r)) for r in m.index])
    ax.set_xlabel("Age band (years)")
    ax.grid(False)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("Sensitivity for TB death,\nthreshold policy", fontsize=8.6)
    cb.outline.set_visible(False)
    fig.tight_layout()
    return _salvar(fig, "s7_age_by_race",
                   x[["race", "age", "estimate", "ci_low", "ci_high", "status", "n_records"]])


EIXOS_FIG = [("regiao", "Region", ["Norte", "Nordeste", "Centro-Oeste", "Sudeste", "Sul"]),
             ("sexo", "Sex", ["F", "M"]),
             ("faixa_etaria", "Age band", IDADES),
             ("vulnerabilidade_municipal", "Municipal vulnerability", ["Q1", "Q2", "Q3", "Q4", "Q5"]),
             ("raca_cor", "Race/colour", ["1", "2", "3", "4", "5"])]
REG_EN = rot.NIVEIS["Region"]
SEXO_EN = {"F": "Female", "M": "Male"}


def _nivel(eixo: str, g: str) -> str:
    if eixo == "regiao":
        return REG_EN.get(g, g)
    if eixo == "sexo":
        return SEXO_EN.get(g, g)
    if eixo == "raca_cor":
        return rot.raca_cor(g)
    if eixo == "vulnerabilidade_municipal":
        return {"Q1": "Q1 (least)", "Q5": "Q5 (most)"}.get(g, g)
    return g


def s8(c: Catalogo, d: Dossie) -> Path:
    sg = c.csv(f"{c.auditoria('primary')}/subgroup_metrics.csv")
    sg["group"] = sg["group"].astype(str)
    sg = sg[sg.rule == "pred_policy"]
    rotulos, lin = [], []
    for eixo, nome, grupos in EIXOS_FIG:
        for g in grupos:
            r1 = sg[(sg.axis == eixo) & (sg.group == g) & (sg.class_code == 1)].iloc[0]
            r2 = sg[(sg.axis == eixo) & (sg.group == g) & (sg.class_code == 2)].iloc[0]
            rotulos.append((nome, _nivel(eixo, g)))
            lin.append((eixo, g, r1.auroc, r2.auroc, r1.n))
    t = pd.DataFrame(lin, columns=["axis", "group", "auroc_tb_death", "auroc_interruption", "n"])
    fig, ax = plt.subplots(figsize=(7.2, 7.4))
    ax.grid(axis="y", visible=False)
    y = np.arange(len(t))[::-1].astype(float)
    desloc, ant = 0.0, None
    ys = []
    for (nome, _), yy in zip(rotulos, y):
        if ant is not None and nome != ant:
            desloc += 0.9
        ys.append(yy - desloc)
        ant = nome
    ys = np.array(ys)
    ax.plot(t.auroc_tb_death.to_numpy(), ys, "o", ms=6, color=COR[1], label="TB death")
    ax.plot(t.auroc_interruption.to_numpy(), ys, "o", ms=6, color=COR[2], label="Treatment interruption")
    ax.set_yticks(ys)
    ax.set_yticklabels([g for _, g in rotulos], fontsize=8.6)
    ant = None
    for (nome, _), yy in zip(rotulos, ys):
        if nome != ant:
            ax.text(0.642, yy + 0.62, nome, fontsize=8.6, color=INK2, fontweight="bold", ha="left",
                    va="bottom")
        ant = nome
    ax.set_xlim(0.64, 0.86)
    ax.set_ylim(ys.min() - 0.7, ys.max() + 1.5)
    ax.set_xlabel("Area under the ROC curve, pooled records")
    ax.legend(frameon=False, fontsize=8.6, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2)
    fig.tight_layout()
    return _salvar(fig, "s8_subgroup_discrimination", t)


def s9(c: Catalogo, d: Dossie) -> Path:
    g = shap_por_dobra(c)
    x = g[g.classe == "tb_death"]
    fam = d["principal.selecao.familias"]
    fig, grade = plt.subplots(2, 3, figsize=(11.4, 6.6))
    axes = grade.ravel()
    for k, ax in enumerate(axes[:5]):
        t = x[x.outer_fold == k].sort_values("mean_abs_shap", ascending=False).head(10)[::-1]
        ax.grid(axis="y", visible=False)
        ax.barh(range(len(t)), t.mean_abs_shap.to_numpy(),
                color=[CAT[5] if e == "municipal" else CAT[0] for e in t.scope], height=0.62,
                zorder=3)
        ax.set_yticks(range(len(t)))
        ax.set_yticklabels([rot.preditor(f) for f in t.feature], fontsize=8)
        ax.set_title(f"Fold {k} · {FAM_CURTA.get(fam[k], fam[k])}", fontsize=10)
        ax.tick_params(axis="x", labelsize=8)
    axes[5].set_axis_off()
    axes[5].legend(handles=[Patch(color=CAT[0], label="Individual predictor"),
                            Patch(color=CAT[5], label="Municipal predictor")],
                   frameon=False, fontsize=9.4, loc="center")
    fig.supxlabel("Mean absolute Shapley value on the raw margin for TB death "
                  "(scale specific to each fitted model)", fontsize=9.6, color=INK2)
    fig.tight_layout()
    return _salvar(fig, "s9_attribution_magnitude_by_fold",
                   x[["outer_fold", "model", "feature", "scope", "mean_abs_shap", "mean_shap", "rank"]])


def s10(c: Catalogo, d: Dossie) -> Path:
    bracos = [("primary", "Selection procedure, primary arm", INK),
              ("individual", "Individual predictors only", CAT[0]),
              ("clinical", "Four-factor comparator", CAT[6]),
              ("municipal", "Municipal predictors only", CAT[5])]
    lin = []
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.9), sharey=True)
    for ax, k in zip(axes, (1, 2)):
        for b, nome, cor in bracos:
            cp = c.csv(f"{c.auditoria(b)}/capacity_pooled.csv")
            g = cp[(cp.comparison == PROC) & (cp.class_code == k)].sort_values("capacity_fraction")
            ax.plot(100 * g.capacity_fraction.to_numpy(), g.ppv.to_numpy(), "o-", ms=5, lw=1.3, color=cor, label=nome)
            lin += [(b, CLASSES[k], r.capacity_fraction, r.ppv, r.sensitivity) for r in g.itertuples()]
        prev = d[f"primary.prob.{CLASSES[k]}.prevalence"]
        ax.axhline(prev, color=INK2, lw=1.0, ls=(0, (4, 3)))
        ax.text(20, prev, "prevalence ", fontsize=8, color=INK2, ha="right", va="bottom")
        ax.set_xticks([1, 5, 10, 20])
        ax.set_xticklabels(["1%", "5%", "10%", "20%"])
        ax.set_xlabel("Highest-risk fraction selected")
        ax.set_title(NOME[k], fontsize=10.5)
    axes[0].set_ylabel("Positive predictive value")
    axes[1].legend(frameon=False, fontsize=8.2, loc="upper right")
    fig.tight_layout()
    return _salvar(fig, "s10_ppv_by_capacity",
                   pd.DataFrame(lin, columns=["arm", "class", "capacity_fraction", "ppv", "captured"]))


FIGURAS = {"fig1": fig1, "fig2": fig2, "fig3": fig3, "fig4": fig4, "fig5": fig5, "fig6": fig6,
           "fig7": fig7, "fig8": fig8, "s1": s1, "s2": s2, "s3": s3, "s4": s4, "s5": s5, "s6": s6,
           "s7": s7, "s8": s8, "s9": s9, "s10": s10}


def main(nomes: list[str]) -> None:
    _style()
    plt.rcParams.update({"font.size": 9.5, "axes.labelsize": 9.4, "xtick.labelsize": 8.8,
                         "ytick.labelsize": 8.8, "axes.titleweight": "bold"})
    d, c = construir()
    for nome in nomes or list(FIGURAS):
        print("gravado:", FIGURAS[nome](c, d).relative_to(RAIZ))


if __name__ == "__main__":
    main(sys.argv[1:])
