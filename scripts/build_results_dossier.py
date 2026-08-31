"""Dossiê de números do Results, gerado a partir dos artefatos.

Honra o invariante 9 do protocolo: nenhum número de tabela ou figura é copiado à mão.
Cada valor aqui é lido do CSV/JSON que o pipeline produziu, com o caminho de origem ao
lado, de modo que a prosa do manuscrito possa ser conferida contra o artefato e refeita
quando um resultado mudar.

Saída: docs/manuscrito/RESULTS_NUMEROS.md (fora do repositório público).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from tb_outcomes.robustness import leaderboard

D = Path("data")
OUT = Path("docs/manuscrito/RESULTS_NUMEROS.md")
CLASSES = {0: "cure", 1: "tb_death", 2: "treatment_interruption"}


def _linhas() -> list[str]:
    L: list[str] = []
    add = L.append

    add("# Números do Results — gerados dos artefatos")
    add("")
    add("> Gerado por `scripts/build_results_dossier.py`. Não editar à mão: reexecute.")
    add("> Cada seção traz o artefato de origem, para conferência.")
    add("")

    # ---------------------------------------------------------------- coorte
    add("## 1. Coorte")
    add("")
    add("Origem: `data/cohort_flow_sensitive_tb_3class.csv`, "
        "`data/exclusion_reasons_sensitive_tb_3class.csv`")
    add("")
    fl = pd.read_csv(D / "cohort_flow_sensitive_tb_3class.csv")
    add("| Etapa | n |")
    add("|---|---|")
    for r in fl.itertuples():
        add(f"| {r.description} | {r.n:,} |")
    add("")
    ex = pd.read_csv(D / "exclusion_reasons_sensitive_tb_3class.csv")
    add("Exclusões (sequenciais):")
    add("")
    for r in ex.itertuples():
        add(f"- `{r.reason}`: {r.n:,}")
    add("")

    # balanço de classes, do OOF do modelo final
    lb = leaderboard(pd.read_csv(D / "benchmark_metrics.csv")).sort_values(
        "f1_macro", ascending=False)
    final = lb.iloc[0]
    oof = pd.read_parquet(
        D / "oof_predictions.parquet", columns=["y_true"],
        filters=[("model", "==", final["model"]),
                 ("strategy", "==", final["best_strategy"])])
    bal = oof["y_true"].value_counts(normalize=True).sort_index() * 100
    n_cls = oof["y_true"].value_counts().sort_index()
    add("Balanço de classes na coorte analítica:")
    add("")
    add("| Classe | n | % |")
    add("|---|---|---|")
    for c, nome in CLASSES.items():
        add(f"| {nome} | {n_cls[c]:,} | {bal[c]:.2f} |")
    add("")

    # --------------------------------------------------------- leaderboard
    add("## 2. Leaderboard principal (k = 50)")
    add("")
    add("Origem: `data/benchmark_metrics.csv`")
    add("")
    add(f"Modelos avaliados: {len(lb)}. "
        f"Melhor: **{final['model']} × {final['best_strategy']}**, "
        f"F1-macro {final['f1_macro']:.4f}.")
    add("")
    add("| # | Modelo | Melhor estratégia | F1-macro |")
    add("|---|---|---|---|")
    for i, r in enumerate(lb.itertuples(), 1):
        add(f"| {i} | {r.model} | {r.best_strategy} | {r.f1_macro:.4f} |")
    add("")

    # ------------------------------------------- desempenho do modelo final
    add("## 3. Modelo final, por classe")
    add("")
    add("Origem: `data/benchmark_metrics.csv` (linhas `kind=aggregate`)")
    add("")
    m = pd.read_csv(D / "benchmark_metrics.csv")
    a = m[(m.kind == "aggregate") & (m.model == final["model"])
          & (m.strategy == final["best_strategy"])]
    perf = a[a.metric.isin(["precision", "recall", "auprc", "roc_auc"])
             & a["class"].isin(["0", "1", "2"])]
    if not perf.empty:
        piv = perf.pivot_table(index="class", columns="metric", values="mean")
        add("| Classe | Precisão | Recall | AUPRC | AUROC |")
        add("|---|---|---|---|---|")
        for c in piv.index:
            nome = CLASSES.get(int(c), c)
            add(f"| {nome} | {piv.loc[c].get('precision', float('nan')):.4f} "
                f"| {piv.loc[c].get('recall', float('nan')):.4f} "
                f"| {piv.loc[c].get('auprc', float('nan')):.4f} "
                f"| {piv.loc[c].get('roc_auc', float('nan')):.4f} |")
        add("")

    # calibração antes/depois
    cal = a[a.metric.isin(["ece_fixed", "cal_slope"]) & a["class"].isin(["0", "1", "2"])]
    if not cal.empty:
        add("Calibração, antes e depois (ECE fixo e inclinação):")
        add("")
        add("| Classe | ECE pré | ECE pós | Slope pré | Slope pós |")
        add("|---|---|---|---|---|")
        for c in ["0", "1", "2"]:
            g = cal[cal["class"] == c]
            def _v(met, st):
                s = g[(g.metric == met) & (g.stage == st)]["mean"]
                return float(s.iloc[0]) if len(s) else float("nan")
            add(f"| {CLASSES[int(c)]} | {_v('ece_fixed','pre_renorm'):.4f} "
                f"| {_v('ece_fixed','post_renorm'):.4f} "
                f"| {_v('cal_slope','pre_renorm'):.4f} "
                f"| {_v('cal_slope','post_renorm'):.4f} |")
        add("")

    # ------------------------------------------------------------- sweep k
    add("## 4. Robustez à granularidade espacial")
    add("")
    add("Origem: `data/sweep/robustness_table.csv`, `data/sweep/rank_stability.csv`")
    add("")
    rs = pd.read_csv(D / "sweep" / "rank_stability.csv")
    for r in rs.itertuples():
        add(f"- k={r.k_a} vs k={r.k_b}: Kendall tau = {r.kendall_tau:.3f} "
            f"(p = {r.p_value:.2g}), overlap do top-5 = {getattr(r, 'top5_overlap'):.1f}")
    add("")
    rt = pd.read_csv(D / "sweep" / "robustness_table.csv")
    piv = rt.pivot(index="model", columns="k", values="f1_macro")
    amp = (piv.max(axis=1) - piv.min(axis=1)).sort_values(ascending=False)
    add(f"Maior amplitude de F1 entre k: {amp.index[0]} ({amp.iloc[0]:.4f}); "
        f"menor: {amp.index[-1]} ({amp.iloc[-1]:.4f}).")
    add("")

    # ---------------------------------------------------------- holdout temporal
    p = D / "temporal" / "temporal_vs_spatial.csv"
    if p.exists():
        add("## 5. Holdout temporal (treino < 2024, teste 2024)")
        add("")
        add(f"Origem: `{p}`")
        add("")
        t = pd.read_csv(p)
        col = [c for c in t.columns if c.startswith("temporal_")][0]
        add(f"| Modelo | Espacial (k=50) | Temporal ({col.split('_')[1]}) | Δ | % retido |")
        add("|---|---|---|---|---|")
        for r in t.itertuples():
            add(f"| {r.model} | {r.espacial_k50:.4f} | {getattr(r, col):.4f} "
                f"| {r.delta:+.4f} | {r.pct_retido:.1f} |")
        add("")

    # ------------------------------------------------------------- ablação
    p = D / "ablation" / "ablation_leaderboard.csv"
    if p.exists():
        add("## 6. Ablação territorial")
        add("")
        add(f"Origem: `{p}`, `data/ablation/ablation_completeness.csv`")
        add("")
        ab = pd.read_csv(p)
        cols = [c for c in ("individual", "municipal", "combined") if c in ab.columns]
        add("| Modelo | " + " | ".join(cols) + " |")
        add("|---" * (len(cols) + 1) + "|")
        for r in ab.itertuples():
            vals = " | ".join(f"{getattr(r, c):.4f}" for c in cols)
            add(f"| {r.model} | {vals} |")
        add("")
        piso = lb[lb.model.isin(["majority_class", "stratified_random"])]["f1_macro"]
        add(f"Piso das baselines no k=50: {piso.max():.4f}.")
        add("")
        comp = pd.read_csv(D / "ablation" / "ablation_completeness.csv")
        add("Desempenho por completude de notificação do município:")
        add("")
        add("| Classe | Grupo | Sensibilidade | PPV |")
        add("|---|---|---|---|")
        for r in comp.itertuples():
            add(f"| {r.classe} | {r.grupo} | {r.sensibilidade:.4f} | {r.ppv:.4f} |")
        add("")

    # ------------------------------------------------------------ equidade
    p = D / "equity_disparity.csv"
    if p.exists():
        add("## 7. Equidade")
        add("")
        add(f"Origem: `{p}`, `data/equity_table.csv`, "
            "`data/equity_discrimination_by_group.csv`")
        add("")
        eq = pd.read_csv(p)
        eq = eq[(eq.regra == "pred_policy") & (eq.metrica == "fn_rate")]
        for classe in sorted(eq.classe.unique()):
            if classe == "cure":
                continue
            add(f"Falso negativo — {classe} (regra operacional):")
            add("")
            add("| Eixo | Pior | valor | Melhor | valor | Razão |")
            add("|---|---|---|---|---|---|")
            for r in eq[eq.classe == classe].itertuples():
                pior = getattr(r, "grupo_pior_rotulo", r.grupo_pior)
                mel = getattr(r, "grupo_melhor_rotulo", r.grupo_melhor)
                add(f"| {r.eixo} | {pior} | {r.valor_pior:.3f} | {mel} "
                    f"| {r.valor_melhor:.3f} | {r.razao:.3f} |")
            add("")
        pd_ = D / "equity_discrimination_by_group.csv"
        if pd_.exists():
            dg = pd.read_csv(pd_)
            dg = dg[(dg.eixo == "raca_cor") & (dg.classe == "tb_death")]
            add("Discriminação por raça/cor em `tb_death` (AUC independe do limiar):")
            add("")
            add("| Grupo | n | Prevalência | AUC | Falso negativo | PPV |")
            add("|---|---|---|---|---|---|")
            for r in dg.sort_values("auc", ascending=False).itertuples():
                add(f"| {r.rotulo} | {r.n:,} | {r.prevalencia:.4f} | {r.auc:.3f} "
                    f"| {r.fn_rate:.3f} | {r.ppv:.3f} |")
            add("")

    # ------------------------------------------------------------ utilidade
    p = D / "utility_topk.csv"
    if p.exists():
        add("## 8. Utilidade programática")
        add("")
        add(f"Origem: `{p}`, `data/utility_decision_curve.csv`, `data/utility_summary.csv`")
        add("")
        ut = pd.read_csv(p)
        for classe in sorted(ut.classe.unique()):
            add(f"{classe}:")
            add("")
            add("| Capacidade | Braço | PPV | Captura | NNS | Lift |")
            add("|---|---|---|---|---|---|")
            for r in ut[ut.classe == classe].sort_values(["k", "arm"]).itertuples():
                add(f"| top {r.k:.0%} | {r.arm} | {r.ppv:.3f} | {r.captura:.3f} "
                    f"| {r.nns:.2f} | {r.lift:.2f} |")
            add("")

    # ------------------------------------------------------------------ SHAP
    p = D / "shap_summary.csv"
    if p.exists():
        add("## 9. SHAP — top 10 por classe (modelo final)")
        add("")
        add(f"Origem: `{p}`, `data/shap_stability.csv`, `data/shap_manifest.json`")
        add("")
        sh = pd.read_csv(p)
        sh = sh[sh._run == sh._run.iloc[0]]
        for classe in sorted(sh.classe.unique()):
            add(f"{classe}:")
            add("")
            add("| # | Feature | |SHAP| médio | SHAP médio (direção) |")
            add("|---|---|---|---|")
            for r in sh[sh.classe == classe].nsmallest(10, "rank").itertuples():
                add(f"| {r.rank} | `{r.feature}` | {r.mean_abs_shap:.5f} "
                    f"| {r.mean_shap:+.5f} |")
            add("")
        st = pd.read_csv(D / "shap_stability.csv")
        add(f"Estabilidade entre sementes: Spearman {st.spearman.min():.4f}–"
            f"{st.spearman.max():.4f}; overlap do top-15 = "
            f"{st[[c for c in st.columns if 'overlap' in c][0]].min():.1f}.")
        add("")

    # ----------------------------------------------------------------- custo
    p = D / "compute_cost_by_model.csv"
    if p.exists():
        add("## 10. Custo computacional")
        add("")
        add(f"Origem: `{p}`, `data/compute_manifest.json`")
        add("")
        cc = pd.read_csv(p)
        add(f"Total cronometrado: {cc.total_horas.sum():.1f} h em "
            f"{int(cc.n_pares.sum())} pares.")
        add("")
        add("| Modelo | pares | horas | % do total |")
        add("|---|---|---|---|")
        for r in cc.itertuples():
            add(f"| {r.model} | {int(r.n_pares)} | {r.total_horas:.2f} | {r.pct_do_total:.1f} |")
        add("")
        with (D / "compute_manifest.json").open(encoding="utf-8") as fh:
            cm = json.load(fh)
        add(f"Ambiente: Python {cm['ambiente']['python']}, "
            + ", ".join(f"{k} {v}" for k, v in cm["ambiente"].items()
                        if k not in ("python", "so") and v))
        add("")
        add(f"Hardware: {cm['hardware']['cpu_threads']} threads de CPU; "
            f"GPUs: {', '.join(cm['hardware']['gpus']) or 'n/d'}.")
        add("")
        add(f"> Limitação declarada: {cm['cobertura']['limitacao']}")
        add("")
    return L


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(_linhas()), encoding="utf-8")
    print(f"gravado: {OUT} ({OUT.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
