"""Monta o material suplementar (S1 File) a partir do catálogo liberado.

Gera `manuscrito/PLOS_DigitalHealth/S1_File.tex` com as tabelas e figuras que complementam os
resultados do corpo principal, e imprime os títulos a colar na seção "Supporting information".

Versão do L5: todas as entradas chegam pelo `Catalogo` de `scripts/editorial/`, que recusa
artefatos fora do `release_manifest.json`. A versão anterior lia, por caminho padrão, as
tabelas históricas de `data/`; não existe mais caminho padrão. Itens da versão anterior sem
fonte no catálogo (características basais por desfecho, mapa e composição dos clusters,
matriz de confusão, distribuições de probabilidade, curvas PR/ROC) foram retirados em vez de
mantidos com números da corrida substituída.

Uso:
    python scripts/build_supplementary.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "scripts/editorial"))
sys.path.insert(0, str(RAIZ / "src"))

from catalogo import C04, C23, REL, Catalogo  # noqa: E402
from dossie_c7 import CLASSES, PROC, R11, Dossie, construir, shap_por_dobra  # noqa: E402
from figuras_c7 import IDADES, _nivel, metricas_por_dobra  # noqa: E402
from tabelas_c7 import ORDEM_BRACOS, SEM_DIFERENCA, ic, num  # noqa: E402
from tb_outcomes import labels as rot  # noqa: E402

OUT = RAIZ / "manuscrito/PLOS_DigitalHealth/S1_File.tex"
FIGS = "figures_c7"
CONFIGS = "data/reanalysis/c6_20260919/primary_cpu/a0a12b4c53f318709c0c/runtime/configs"
CLS = {0: "Cure", 1: "TB death", 2: "Treatment interruption"}
EST = ["none", "random_oversampling", "random_undersampling", "local_cost_sensitive"]


class Raw(str):
    """Célula já em LaTeX (sinal de menos tipográfico, intervalo): não passa por `_esc`."""


def _esc(s) -> str:
    """Escapa o texto para LaTeX e cria pontos de quebra nos sublinhados.

    Nomes como `IBGE_NUMERO_DE_ESTAB_DE_ENSINO_FUNDAMENTAL` não têm espaço e transbordam a
    coluna, invadindo a seguinte. O `\\allowbreak` após cada sublinhado dá ao LaTeX onde
    quebrar, sem hifenizar nem alterar o texto. A remoção de contrabarras acontece ANTES na
    cadeia, de modo que o comando inserido aqui sobrevive.
    """
    return (str(s).replace("\\", "").replace("_", r"\_\allowbreak ")
            .replace("%", r"\%").replace("&", r"\&").replace("#", r"\#")
            .replace(">", r"$>$").replace("<", r"$<$"))


def _tabela(df: pd.DataFrame, caption: str, label: str, colspec: str | None = None,
            nota: str = "",
            max_linhas: int | None = None, longa: bool = False,
            fonte: str = "small", regua_por_linha: bool = False) -> str:
    """Monta a tabela em LaTeX.

    `longa=True` usa `longtable`, que quebra entre páginas. Sem isso uma tabela mais alta
    que a página é SILENCIOSAMENTE CORTADA pelo `table`, e o leitor perde linhas sem
    qualquer aviso. Toda tabela com dezenas de linhas precisa disso.

    `regua_por_linha=True` desenha um filete entre registros. Serve às tabelas cujas
    células ocupam várias linhas de texto: sem o filete, não se distingue onde termina um
    registro e começa o seguinte.
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
                if pd.isna(v):
                    vals.append("")
                elif float(v).is_integer():
                    vals.append(str(int(v)))
                else:
                    vals.append(f"{v:.3f}")
            else:
                vals.append(v if isinstance(v, Raw) else
                            ("" if (v is None or str(v) in ("nan", "None")) else _esc(v)))
        corpo.append(" & ".join(vals) + r" \\")
        if regua_por_linha:
            corpo.append(r"\hline")

    if longa:
        out = [r"\begingroup", "\\" + fonte, rf"\begin{{longtable}}{{{spec}}}",
               r"\caption{{\bf " + caption + r"} " + nota + "}", rf"\label{{{label}}}", r"\\",
               r"\hline", cabecalho, r"\endfirsthead",
               r"\hline", cabecalho, r"\endhead",
               r"\hline", r"\endfoot", *corpo,
               r"\end{longtable}", r"\endgroup"]
        return "\n".join(out)

    out = [r"\begin{table}[!ht]", "\\" + fonte,
           r"\caption{{\bf " + caption + r"} " + nota + "}", rf"\label{{{label}}}",
           rf"\begin{{tabular}}{{{spec}}}", r"\hline", cabecalho,
           *corpo, r"\hline", r"\end{tabular}", r"\end{table}"]
    return "\n".join(out)


def R(d: Dossie, chave: str, casas: int = 3, sinal: bool = False) -> Raw:
    return Raw(ic(d, chave, casas, sinal=sinal, traco=True))


def _n(v) -> str:
    return f"{int(v):,}"


# (arquivo da figura, título, legenda)
FIGURAS = [
    ("s1_model_strategy_heatmap", "Macro-averaged F1 by algorithm and imbalance strategy",
     "Primary arm, development period 2015 to 2023, arg max rule, outer-fold predictions pooled "
     "by record. Exploratory comparison among the 96 compatible pairs, not the estimate of "
     "performance. Cells marked `not run' are the eight pairs the adapter does not support "
     "under cost-sensitive reweighting; the gap says nothing about those families."),
    ("s2_per_class_by_rule", "Sensitivity and positive predictive value under the two decision rules",
     "Nested selection procedure, primary arm, pooled records, with 95\\% cluster-bootstrap "
     "intervals. The arg max rule is the primary one; the per-class threshold policy trades "
     "precision for sensitivity in the two unfavourable classes."),
    ("s3_between_fold", "Between-fold variability",
     "Macro-averaged F1 in each held-out spatial fold. Large markers are the nested selection "
     "procedure, with the family it chose inside the development data of that fold; the short "
     "dashes are each candidate pair, shown for context and not used to choose. Fold 0 consists "
     "of a single geographic cluster."),
    ("s5_cluster_influence", "Influence of each geographic cluster on the pooled estimate",
     "Change in pooled macro-averaged F1 when one of the 50 clusters is removed from the fixed "
     "predictions. No model is refitted, so this is a description of dependence on individual "
     "territories and not a further validation."),
    ("s4_calibration_summary", "Calibration diagnostics before and after the calibrator",
     "Nested selection procedure, primary arm, pooled records. Expected calibration error is a "
     "magnitude without direction; calibration-in-the-large is the offset estimated with the "
     "slope fixed at one. The two diagnostics move in different directions here: the offset "
     "approaches zero while the binned error rises in the three classes, by a small amount for death."),
    ("s6_temporal_ranking", "Exploratory ranking in the temporal evaluation",
     "Notifications of 2024 (n = 92,152), models fitted on 2015 to 2023; eleven families and 42 "
     "pairs. Macro-averaged F1 under the arg max rule with 95\\% cluster-bootstrap intervals, each "
     "family under its best strategy. This retrospective exploratory temporal analysis "
     "does not constitute an independent confirmatory test."),
    ("s9_attribution_magnitude_by_fold", "Attribution magnitude within each fitted model",
     "Ten largest mean absolute Shapley values for death attributed to tuberculosis in each outer "
     "fold, averaged over three sampling seeds. Values explain the raw margin before calibration "
     "and are on the scale of each fitted model: they are comparable within a panel and not "
     "between panels, in particular not between LightGBM and CatBoost."),
    ("s8_subgroup_discrimination", "Discrimination by subgroup",
     "Area under the receiver operating characteristic curve for the two unfavourable outcomes, "
     "nested selection procedure, primary arm, pooled records. No interval is available for this "
     "quantity. Vulnerability quintiles are of the municipality of residence."),
    ("s7_age_by_race", "Sensitivity for death attributed to tuberculosis by age band and race or colour",
     "Nested selection procedure, primary arm, threshold policy, pooled records. Cells marked "
     "`n/e' were not estimable under the minimum denominator of 30 or for lack of clusters. "
     "Descriptive stratification; it does not test whether age explains a difference between groups."),
    ("s10_ppv_by_capacity", "Positive predictive value by selected fraction",
     "Positive predictive value among the notifications with the highest calibrated probability, "
     "at 1, 5, 10 and 20 per cent of notifications, for four arms. Retrospective ranking of pooled "
     "outer-fold predictions; no intervention was delivered."),
]


def t_modelos(c: Catalogo, d: Dossie) -> pd.DataFrame:
    cfg = yaml.safe_load(c.texto(f"{CONFIGS}/models.yaml"))
    cov = c.csv(f"{REL}/candidate_coverage.csv")
    cov = cov[cov.arm == "primary"]
    fam = {"baseline": "Reference", "tree": "Tree", "boosting": "Boosting", "linear": "Linear",
           "svm": "Support vector", "nb": "Naive Bayes", "discriminant": "Discriminant",
           "neighbors": "Neighbours", "neural": "Neural network", "dl": "Tabular deep learning"}
    rep = {"onehot_unscaled": "One-hot", "onehot_scaled": "One-hot, standardised",
           "native_categorical": "Native categorical", "nonnegative": "One-hot, non-negative",
           "binary": "Binarised"}
    lin = []
    for nome, spec in {**cfg["baselines"], **cfg["models"]}.items():
        grade = cfg["hyperparameters"].get(nome)
        busca = ("; ".join(f"{k}: {', '.join('none' if x is None else str(x) for x in v)}"
                           for k, v in grade.items()) if grade else "single configuration")
        n = int((cov.model == nome).sum())
        lin.append({"Algorithm": rot.algoritmo(nome), "Family": fam[spec["family"]],
                    "Input": rep[spec["preprocess_profile"]], "Declared grid": busca,
                    "Strategies run": f"{n} of 4" if n else "excluded"})
    return pd.DataFrame(lin)


def t_coorte(c: Catalogo, d: Dossie) -> pd.DataFrame:
    lin = [("Records returned by the acquisition pipeline", d["coorte.recuperada"])]
    lin += [("Excluded: " + (rot.motivo_exclusao(k) if k != "fora_da_janela_declarada"
                             else "Outside the 2015-2025 window"), d[f"coorte.excl.{k}"])
            for k in ("municipio_irrecuperavel", "idade_impossivel", "fora_da_janela_declarada",
                      "notificacao_pos_obito", "seguimento_insuficiente")]
    lin += [("Eligible notifications", d["coorte.elegivel"]),
            ("Not modelled: closure outside the three-outcome scheme",
             d["coorte.fora_esquema.outra_categoria"]),
            ("Not modelled: no closure recorded (unresolved)", d["coorte.fora_esquema.nao_resolvido"]),
            ("Analytic cohort with a three-class outcome", d["coorte.analitica"])]
    for papel, nome in (("dev", "Development, 2015-2023"), ("temporal", "Temporal evaluation, 2024"),
                        ("futuro", "Not used in any evaluation, 2025")):
        lin.append((nome, d[f"coorte.{papel}"]))
        for k, cls in CLASSES.items():
            lin.append((f"   {CLS[k]}", Raw(f"{_n(d[f'coorte.{papel}.{cls}'])} "
                                            f"({d[f'coorte.{papel}.{cls}.pct']:.2f}\\%)")))
    return pd.DataFrame([(a, b if isinstance(b, Raw) else _n(b)) for a, b in lin],
                        columns=["Item", "n"])


def t_ranking(c: Catalogo, braco: str) -> pd.DataFrame:
    aud = c.auditoria(braco)
    a = c.csv(f"{aud}/cluster_intervals_pred_argmax.csv")
    a = a[a.metric == "f1_macro"]
    pol = c.csv(f"{aud}/cluster_intervals_pred_policy.csv")
    pol = pol[(pol.metric == "f1_macro") & (pol.kind == "estimate")].set_index("comparison")
    est = a[a.kind == "estimate"].sort_values("estimate", ascending=False)
    dif = a[a.kind == "paired_difference"].set_index("comparison")
    fm = c.csv(f"{aud}/recomputed_fold_metrics.csv")
    fm = fm[fm.rule == "pred_argmax"].groupby("comparison")["f1_macro"].agg(["mean", "std"])
    lin, pos = [], 0
    for r in est.itertuples():
        if r.comparison == PROC:
            alg, estr, rank, media, delta = "Nested selection procedure", "chosen per fold", "", "", ""
        else:
            pos += 1
            m, e = r.comparison.split("__")
            alg, estr, rank = rot.algoritmo(m), rot.estrategia(e), str(pos)
            media = Raw(f"{fm.loc[r.comparison, 'mean']:.3f}"
                        + (f" ({fm.loc[r.comparison, 'std']:.3f})" if braco != "temporal" else ""))
            x = dif.loc[r.comparison]
            delta = Raw(f"{num(x.estimate, 3, sinal=True)} ({num(x.ci_low, 3, sinal=True)} to "
                        f"{num(x.ci_high, 3, sinal=True)})")
        p = pol.loc[r.comparison]
        lin.append({"Rank": rank, "Algorithm": alg, "Strategy": estr,
                    "Arg max, pooled (95% CI)": Raw(f"{r.estimate:.3f} ({r.ci_low:.3f} to {r.ci_high:.3f})"),
                    "Fold mean (SD)" if braco != "temporal" else "Single partition": media,
                    "Difference vs. procedure (95% CI)": delta,
                    "Thresholds, pooled": Raw(f"{p.estimate:.3f}")})
    return pd.DataFrame(lin)


def t_bracos(c: Catalogo, d: Dossie) -> pd.DataFrame:
    lin = []
    for b in ORDEM_BRACOS:
        if b in SEM_DIFERENCA:
            da = dp = "reference" if b == "primary" else "n/c"
            fam = (", ".join(dict.fromkeys(rot.algoritmo(x) for x in d["principal.selecao.familias"]))
                   if b == "primary" else (rot.algoritmo(d["temporal.selecao.familia"]) if b == "temporal" else ""))
        elif b == "clinical":
            # o artefato liberado traz principal menos comparador; a tabela mantém esse sentido
            # e a legenda o declara, em vez de inverter o sinal de um intervalo assimétrico
            da, dp = R(d, "clinical.argmax.dif", 4, True), R(d, "clinical.policy.dif", 4, True)
            fam = "Logistic regression"
        else:
            da, dp = R(d, f"{b}.argmax.dif", 4, True), R(d, f"{b}.policy.dif", 4, True)
            fam = ", ".join(dict.fromkeys(rot.algoritmo(x) for x in d[f"{b}.selecao.familias"]))
        lin.append({"Arm": rot.braco(b), "Arg max, pooled (95% CI)": R(d, f"{b}.argmax.f1.pooled"),
                    "Thresholds, pooled (95% CI)": R(d, f"{b}.policy.f1.pooled"),
                    "Difference, arg max": da, "Difference, thresholds": dp,
                    "Families selected": fam})
    return pd.DataFrame(lin)


def t_bracos_prob(c: Catalogo, d: Dossie) -> pd.DataFrame:
    lin = []
    for b in ("primary", "individual", "municipal", "clinical", "temporal"):
        for k, cls in CLASSES.items():
            p = f"{b}.prob.{cls}"
            cap = (Raw(f"{100 * d[f'{b}.cap5.{cls}.capturado']:.1f}\\%") if k else "")
            lin.append({"Arm": rot.braco(b) if k == 0 else "", "Class": CLS[k],
                        "AUROC": d[p + ".auroc"], "AUPRC": d[p + ".auprc"],
                        "ECE": Raw(f"{d[p + '.ece_fixed_10']:.4f}"),
                        "Slope": d[p + ".calibration_slope"],
                        "Offset": Raw(num(d[p + ".calibration_in_large"], 3, sinal=True)),
                        "Top 5%": cap})
    return pd.DataFrame(lin)


def t_raca(c: Catalogo, d: Dossie) -> pd.DataFrame:
    lin = []
    for r in ("1", "2", "3", "4", "5"):
        p = f"sub.raca_cor.{r}.tb_death"
        ppv_a = (R(d, p + ".argmax.ppv") if (p + ".argmax.ppv") in d.valores else "n/e")
        lin.append({"Group": rot.raca_cor(r), "n": _n(d[p + ".n"]), "Deaths": _n(d[p + ".events"]),
                    "Prevalence": Raw(f"{100 * d[p + '.prevalence']:.2f}\\%"), "AUROC": d[p + ".auroc"],
                    "Slope": d[p + ".calibration_slope"],
                    "Sensitivity, thresholds": R(d, p + ".policy.sensitivity"),
                    "PPV, thresholds": R(d, p + ".policy.ppv"),
                    "Sensitivity, arg max": R(d, p + ".argmax.sensitivity"), "PPV, arg max": ppv_a})
    return pd.DataFrame(lin)


def t_incluidos(c: Catalogo) -> pd.DataFrame:
    ie = c.csv(f"{REL}/included_excluded_by_group.csv")
    ie = ie[(ie.role == "development") & ie.axis.isin(["NU_ANO", "REGIAO", "CS_SEXO", "FORMA"])].copy()
    eixo = {"NU_ANO": "Notification year", "REGIAO": "Region", "CS_SEXO": "Sex", "FORMA": "Clinical form"}
    sexo = {"F": "Female", "M": "Male", "I": "Unknown"}
    lin, ant = [], None
    for r in ie.itertuples():
        g = str(r.group)
        if g.endswith(".0"):
            g = g[:-2]
        if r.axis == "REGIAO":
            g = rot.NIVEIS["Region"].get(g, g)
        elif r.axis == "CS_SEXO":
            g = sexo.get(g, g)
        elif r.axis == "FORMA":
            g = rot.codigo_sinan("FORMA", g)
        lin.append({"Axis": eixo[r.axis] if r.axis != ant else "", "Group": g,
                    "Eligible": _n(r.eligible), "Included": _n(r.included), "Not modelled": _n(r.excluded),
                    "Not modelled (%)": Raw(f"{100 * r.excluded_fraction:.2f}")})
        ant = r.axis
    return pd.DataFrame(lin)


def t_subgrupos(c: Catalogo, d: Dossie) -> pd.DataFrame:
    aud = c.auditoria("primary")
    sg = c.csv(f"{aud}/subgroup_metrics.csv")
    sg["group"] = sg["group"].astype(str)
    si = c.csv(f"{aud}/subgroup_cluster_intervals.csv")
    si["group"] = si["group"].astype(str)
    si = si[si.decision_rule == "pred_policy"]
    eixos = [("regiao", ["Norte", "Nordeste", "Centro-Oeste", "Sudeste", "Sul"]), ("sexo", ["F", "M"]),
             ("faixa_etaria", IDADES), ("vulnerabilidade_municipal", ["Q1", "Q2", "Q3", "Q4", "Q5"]),
             ("raca_cor", ["1", "2", "3", "4", "5"]),
             ("feature_missing_fraction", ["0-10%", ">10-25%", ">25-50%"])]
    nomes = {"regiao": "Region", "sexo": "Sex", "faixa_etaria": "Age band",
             "vulnerabilidade_municipal": "Municipal vulnerability", "raca_cor": "Race/colour",
             "feature_missing_fraction": "Share of predictors missing"}
    lin = []
    for k in (1, 2):
        for eixo, grupos in eixos:
            for i, g in enumerate(grupos):
                r = sg[(sg.axis == eixo) & (sg.group == g) & (sg.rule == "pred_policy") & (sg.class_code == k)].iloc[0]

                def iv(met):
                    x = si[(si.axis == eixo) & (si.group == g) & (si.metric == f"{met}:{k}")].iloc[0]
                    return (Raw(f"{x.estimate:.3f} ({x.ci_low:.3f} to {x.ci_high:.3f})")
                            if x.status == "ok" else "n/e")
                lin.append({"Outcome": CLS[k] if (i == 0 and eixo == "regiao") else "",
                            "Axis": nomes[eixo] if i == 0 else "", "Group": _nivel(eixo, g),
                            "n": _n(r.n), "Events": _n(r.events), "AUROC": r.auroc,
                            "Sensitivity (95% CI)": iv("sensitivity"), "PPV (95% CI)": iv("ppv")})
    return pd.DataFrame(lin)


def t_cobertura(c: Catalogo) -> pd.DataFrame:
    cov = c.csv(f"{REL}/candidate_coverage.csv")
    lin = []
    for b in ORDEM_BRACOS:
        g = cov[cov.arm == b]
        por = g.groupby("strategy").size().reindex(EST, fill_value=0)
        lin.append({"Arm": rot.braco(b), "Families": g.model.nunique(), "Pairs": len(g),
                    **{rot.estrategia(e): int(por[e]) for e in EST},
                    "Records per pair": _n(g.n_unique.iloc[0]),
                    "Complete": "yes" if (g.n_unique.nunique() == 1) else "no"})
    return pd.DataFrame(lin)


def t_custo(c: Catalogo) -> pd.DataFrame:
    cc = c.csv(f"{REL}/compute_cost_all_arms.csv")
    lin = []
    for b in ORDEM_BRACOS:
        g = cc[cc.arm == b]
        cpu, gpu = g[g.compute == "cpu"], g[g.compute == "gpu"]
        lin.append({"Arm": rot.braco(b), "CPU jobs": len(cpu),
                    "CPU hours": Raw(f"{cpu.wall_seconds.sum() / 3600:.1f}"),
                    "GPU jobs": len(gpu), "GPU hours": Raw(f"{gpu.wall_seconds.sum() / 3600:.1f}"),
                    "Total hours": Raw(f"{g.wall_seconds.sum() / 3600:.1f}")})
    lin.append({"Arm": "All arms", "CPU jobs": _n((cc.compute == "cpu").sum()),
                "CPU hours": Raw(f"{cc[cc.compute == 'cpu'].wall_seconds.sum() / 3600:,.1f}"),
                "GPU jobs": int((cc.compute == "gpu").sum()),
                "GPU hours": Raw(f"{cc[cc.compute == 'gpu'].wall_seconds.sum() / 3600:.1f}"),
                "Total hours": Raw(f"{cc.wall_seconds.sum() / 3600:,.1f}")})
    return pd.DataFrame(lin)


def t_dobras(c: Catalogo, d: Dossie) -> pd.DataFrame:
    sg = c.csv(f"{c.auditoria('primary')}/subgroup_metrics.csv")
    g = sg[(sg.axis == "dobra_espacial") & (sg.rule == "pred_argmax")]
    sel = c.json(f"{C23}/primary_selected/selected_by_fold.json")
    fm = metricas_por_dobra(c).set_index("outer_fold")
    lin = []
    for k in range(5):
        x = g[g.group.astype(str) == str(k)].set_index("class_code")
        s = sel[k]
        lin.append({"Fold": k, "Clusters": d["bootstrap.clusters_por_dobra"][k], "Records": _n(x.loc[0, "n"]),
                    "Cure": _n(x.loc[0, "events"]), "TB death": _n(x.loc[1, "events"]),
                    "Interruption": _n(x.loc[2, "events"]),
                    "Selected in development": f"{rot.algoritmo(s['model'])}, {rot.estrategia(s['strategy']).lower()}; "
                    + ", ".join(f"{a.replace('_', ' ')} {b}" for a, b in s["parameters"].items()) + f"; {s['calibration']} calibrator",
                    "F1 arg max": fm.loc[k, "argmax"], "F1 thresholds": fm.loc[k, "policy"]})
    return pd.DataFrame(lin)


def t_influencia(c: Catalogo) -> pd.DataFrame:
    inf = c.csv(f"{c.auditoria('primary')}/cluster_influence.csv")
    m = inf.pivot(index="cluster", columns="rule", values="change")
    base = inf.drop_duplicates("cluster").set_index("cluster")
    t = pd.DataFrame({"Cluster": m.index, "Records": [_n(x) for x in base.loc[m.index, "n_records"]],
                      "Share (%)": [Raw(f"{100 * x:.2f}") for x in base.loc[m.index, "record_fraction"]],
                      "Change, arg max": [Raw(num(x, 5, sinal=True)) for x in m["pred_argmax"]],
                      "Change, thresholds": [Raw(num(x, 5, sinal=True)) for x in m["pred_policy"]]})
    return t.sort_values("Cluster")


def t_dicionario(c: Catalogo) -> pd.DataFrame:
    fd = c.csv(f"{REL}/primary_feature_dictionary.csv")
    nivel = {"individual": "Individual", "municipal": "Municipal"}
    t = pd.DataFrame({"Variable": fd.name, "Label used in figures": fd.name.map(rot.preditor),
                      "Level": fd.scope.map(nivel), "Source": fd.source.map(rot.FONTES_PREDITOR)})
    return t.sort_values(["Level", "Variable"])


def t_baselines(c: Catalogo) -> pd.DataFrame:
    bc = c.csv(f"{C04}/baseline_confusions/confusions_pooled.csv")
    bc = bc[bc.strategy == "none"]
    regras = {"uncalibrated_argmax": "Arg max, before calibration",
              "calibrated_argmax": "Arg max, after calibration",
              "calibrated_policy": "Thresholds, after calibration"}
    lin = []
    for mod in ("majority_class", "stratified_random"):
        for regra, nome in regras.items():
            g = bc[(bc.model == mod) & (bc.rule == regra)]
            m = g.pivot(index="actual", columns="predicted", values="count")
            for k in (0, 1, 2):
                lin.append({"Reference": rot.algoritmo(mod) if (k == 0 and regra == "uncalibrated_argmax") else "",
                            "Rule": nome if k == 0 else "", "Observed": CLS[k],
                            "Predicted cure": _n(m.loc[k, 0]), "Predicted TB death": _n(m.loc[k, 1]),
                            "Predicted interruption": _n(m.loc[k, 2])})
    return pd.DataFrame(lin)


def t_calibracao(c: Catalogo) -> pd.DataFrame:
    cal = c.csv(f"{REL}/primary_calibration_before_after.csv")
    lin = []
    for k in (0, 1, 2):
        for estagio, nome in (("before_calibration", "Before"), ("after_calibration", "After")):
            r = cal[(cal.stage == estagio) & (cal.class_code == k)].iloc[0]
            lin.append({"Class": CLS[k] if estagio == "before_calibration" else "", "Stage": nome,
                        "ECE (10 fixed bins)": Raw(f"{r.ece_fixed_10:.6f}"),
                        "Intercept": Raw(num(r.calibration_intercept, 4, sinal=True)),
                        "Slope": Raw(f"{r.calibration_slope:.4f}"),
                        "Offset (slope 1)": Raw(num(r.calibration_in_large, 4, sinal=True)),
                        "Brier": Raw(f"{r.brier:.5f}"), "AUROC": Raw(f"{r.auroc:.4f}"),
                        "AUPRC": Raw(f"{r.auprc:.4f}")})
    return pd.DataFrame(lin)


def t_idade_raca(c: Catalogo) -> pd.DataFrame:
    sc = c.csv(f"{C04}/age_race_stratification/stratum_counts.csv")
    sc = sc[sc.class_code == 1]
    am = c.csv(f"{C04}/age_race_stratification/stratified_metrics.csv")
    am = am[(am.decision_rule == "pred_policy") & (am.metric == "sensitivity:1")]
    lin = []
    for r in (1, 2, 3, 4, 5):
        for i, faixa in enumerate(IDADES):
            s = sc[(sc.race == r) & (sc.age == faixa)].iloc[0]
            x = am[(am.race == r) & (am.age == faixa)].iloc[0]
            lin.append({"Race/colour": rot.raca_cor(str(r)) if i == 0 else "", "Age band": faixa,
                        "n": _n(s.n), "Deaths": _n(s.events),
                        "Deaths (%)": Raw(f"{100 * s.events / s.n:.2f}"),
                        "Sensitivity, thresholds (95% CI)":
                            (Raw(f"{x.estimate:.3f} ({x.ci_low:.3f} to {x.ci_high:.3f})")
                             if x.status == "ok" else "n/e")})
    return pd.DataFrame(lin)


def t_shap(c: Catalogo) -> pd.DataFrame:
    g = shap_por_dobra(c)
    lin = []
    for k, cls in ((1, "tb_death"), (2, "treatment_interruption"), (0, "cure")):
        x = g[g.classe == cls]
        ordem = x.groupby("feature")["rank"].median().sort_values().head(15).index
        m = x[x.feature.isin(ordem)].pivot(index="feature", columns="outer_fold", values="rank").loc[ordem]
        esc = x.drop_duplicates("feature").set_index("feature")["scope"]
        for i, f in enumerate(ordem):
            lin.append({"Class": CLS[k] if i == 0 else "", "Predictor": rot.preditor(f),
                        "Level": esc[f].capitalize(), **{f"Fold {j}": int(m.loc[f, j]) for j in range(5)}})
    return pd.DataFrame(lin)


def t_shap_estabilidade(c: Catalogo, d: Dossie) -> pd.DataFrame:
    lin = []
    fam = d["principal.selecao.familias"]
    for k in range(5):
        e = c.csv(f"{C04}/attribution/fold{k}/sampling_stability.csv")
        for cls in ("tb_death", "treatment_interruption", "cure"):
            x = e[e.classe == cls]
            lin.append({"Fold": k if cls == "tb_death" else "", "Family": rot.algoritmo(fam[k]) if cls == "tb_death" else "",
                        "Class": CLS[{v: kk for kk, v in CLASSES.items()}[cls]],
                        "Spearman, min": Raw(f"{x.spearman.min():.4f}"),
                        "Spearman, max": Raw(f"{x.spearman.max():.4f}"),
                        "Top-15 overlap, min": Raw(f"{x.top15_overlap.min():.3f}")})
    return pd.DataFrame(lin)


# Campos com dicionário de códigos declarado no projeto. Os demais saem com o código do SINAN:
# rotular por inferência é o erro que `labels.SEM_DICIONARIO` existe para impedir.
VARIAVEIS_BASAIS = [("age_band", "Age band (years)"), ("CS_SEXO", "Sex"), ("CS_RACA", "Race/colour"),
                    ("REGIAO", "Region"), ("CS_ESCOL_N", "Schooling"), ("FORMA", "Clinical form"),
                    ("TRATAMENTO", "Type of entry"), ("AGRAVAIDS", "Declared AIDS"),
                    ("AGRAVALCOO", "Alcohol use"), ("AGRAVDIABE", "Diabetes"),
                    ("AGRAVDROGA", "Illicit drug use"), ("AGRAVTABAC", "Tobacco use"),
                    ("POP_RUA", "Homelessness"), ("POP_LIBER", "Deprived of liberty"),
                    ("POP_SAUDE", "Health worker"), ("POP_IMIG", "Immigrant"),
                    ("BENEF_GOV", "Government cash-transfer beneficiary"),
                    ("RAIOX_TORA", "Chest radiography"), ("BACILOSC_E", "Sputum smear"),
                    ("BACILOSC_O", "Smear of other material"), ("TEST_MOLEC", "Rapid molecular test")]


def _categoria_basal(var: str, cat: str) -> str:
    return rot.codigo_sinan(var, cat)


def t_basal_categorica(c: Catalogo, d: Dossie) -> pd.DataFrame:
    b = c.csv(f"{R11}/baseline_categorical.csv", dtype={"category": str})
    b = b[b.outcome == "all"]
    papeis = ("development_2015_2023", "temporal_2024")
    if set(b.feature) != {v for v, _ in VARIAVEIS_BASAIS}:
        raise ValueError("variáveis basais liberadas divergem da lista da tabela")
    lin = []
    for var, nome in VARIAVEIS_BASAIS:
        g = b[b.feature == var]
        cats = list(dict.fromkeys(g.category))
        ordem = (IDADES if var == "age_band" else
                 sorted((x for x in cats if x != "__missing__"),
                        key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else 0, x)))
        primeira = True
        for cat in [x for x in ordem if x in cats] + ["__missing__"]:
            cel = {}
            for papel in papeis:
                r = g[(g.role == papel) & (g.category == cat)]
                n, den = (int(r.n.iloc[0]), int(r.denominator.iloc[0])) if len(r) else (0, None)
                if den is not None and den != d["coorte.dev" if papel == papeis[0] else "coorte.temporal"]:
                    raise ValueError("denominador basal diverge da coorte")
                cel[papel] = (n, float(r.percent.iloc[0]) if len(r) else 0.0)
            if cat != "__missing__" and all(v[0] == 0 for v in cel.values()):
                continue
            lin.append({"Characteristic": nome if primeira else "", "Level": _categoria_basal(var, cat),
                        "Development 2015-2023, n (%)": Raw(f"{cel[papeis[0]][0]:,} ({cel[papeis[0]][1]:.2f})"),
                        "Evaluation 2024, n (%)": Raw(f"{cel[papeis[1]][0]:,} ({cel[papeis[1]][1]:.2f})")})
            primeira = False
    return pd.DataFrame(lin)


def t_basal_numerica(c: Catalogo) -> pd.DataFrame:
    bn = c.csv(f"{R11}/baseline_numeric.csv")
    nomes = {"age_years": "Age (years)", "NU_CONTATO": "Contacts identified",
             "potential_observation_days": "Potential observation, days"}
    papeis = {"development_2015_2023": "Development 2015-2023", "temporal_2024": "Evaluation 2024"}
    desf = {"all": "All", "cure": "Cure", "tb_death": "TB death",
            "treatment_interruption": "Interruption"}
    lin = []
    for var, nome in nomes.items():
        for papel, pn in papeis.items():
            for o, on in desf.items():
                r = bn[(bn.feature == var) & (bn.role == papel) & (bn.outcome == o)].iloc[0]
                lin.append({"Variable": nome if (o == "all" and papel.startswith("dev")) else "",
                            "Population": pn if o == "all" else "", "Outcome": on, "n": _n(r.n_total),
                            "Missing (%)": Raw(f"{r.missing_percent:.2f}"),
                            "Median (IQR)": Raw(f"{r['median']:,.0f} ({r.q1:,.0f}--{r.q3:,.0f})"),
                            "Mean": Raw(f"{r['mean']:,.1f}")})
    return pd.DataFrame(lin)


def t_matrizes(c: Catalogo) -> pd.DataFrame:
    cf = c.csv(f"{R11}/selected_confusions.csv")
    lin = []
    for braco, pn in (("primary", "Development 2015-2023"), ("temporal", "Evaluation 2024")):
        for regra, rn in (("pred_argmax", "Arg max"), ("pred_policy", "Thresholds")):
            g = cf[(cf.arm == braco) & (cf.rule == regra)]
            for k in (0, 1, 2):
                x = g[g.actual == k].set_index("predicted")
                lin.append({"Population": pn if (k == 0 and regra == "pred_argmax") else "",
                            "Rule": rn if k == 0 else "", "Observed": CLS[k],
                            "n": _n(x.actual_denominator.iloc[0]),
                            **{("Predicted cure", "Predicted TB death", "Predicted interruption")[j]:
                               Raw(f"{int(x.loc[j, 'n']):,} ({x.loc[j, 'row_percent']:.1f})") for j in (0, 1, 2)}})
    return pd.DataFrame(lin)


def t_proveniencia(c: Catalogo, d: Dossie) -> pd.DataFrame:
    arms = {a["arm"]: a for a in c.json(f"{REL}/arm_overview.json")}
    pop = {"temporal": "2024", "all_cause": "2015-2023, all-cause death target"}
    lin = [{"Arm": rot.braco(b), "Source experiment": arms[b]["experiment_id"],
            "Population": pop.get(b, "2015-2023"), "n": _n(arms[b]["n"]),
            "Outer partitions": arms[b]["folds"]} for b in ORDEM_BRACOS]
    return pd.DataFrame(lin)


def main() -> None:
    d, c = construir()
    L: list[str] = []
    add = L.append
    add(r"\documentclass[10pt,letterpaper]{article}")
    add(r"\usepackage[top=1in,left=1in,right=1in,bottom=1in]{geometry}")
    add(r"\usepackage[utf8]{inputenc}")
    add(r"\usepackage{graphicx,booktabs,longtable,caption}")
    add(r"\usepackage[labelfont=bf,labelsep=period]{caption}")
    add(r"\setlength{\tabcolsep}{3.5pt}")
    add(r"\renewcommand{\thefigure}{S\arabic{figure}}")
    add(r"\renewcommand{\thetable}{S\arabic{table}}")
    add(r"\begin{document}")
    add(r"\begin{center}{\Large\bf S1 File. Supporting information}\\[2mm]")
    add(r"Predicting cure, treatment interruption, and death from tuberculosis in Brazil "
        r"under spatially blocked and temporal validation\end{center}")
    add(r"\bigskip")
    add(r"This file collects the tables and figures that complement the main results. Every value "
        r"is read from the artefacts released for reporting; none is typed by hand. Three "
        r"distinctions are stated in each caption because they change the number: the arm, the "
        r"decision rule (the arg max of the calibrated probabilities is the primary rule, the "
        r"per-class threshold policy the secondary one) and the aggregation (pooled over records, "
        r"with a 95\% interval from a bootstrap of geographic clusters with " + _n(d["bootstrap.replicas"])
        + r" replicates, or mean across outer folds, which carries a standard deviation and no "
        r"interval). Intervals are conditional on the fixed predictions and do not include the "
        r"uncertainty of refitting or of selection. `n/e' marks an estimate that was not estimable "
        r"or was suppressed for a denominator below 30; `n/c' marks a comparison that was not computed.")
    add(r"\clearpage")

    legendas: list[tuple[str, str]] = []

    def tabela(titulo, nota, df, **kw):
        n = len([x for x in legendas if "Table" in x[0]]) + 1
        add(_tabela(df, titulo + ".", f"tab:s{n}", nota=nota, **kw))
        add(r"\clearpage")
        legendas.append((f"S{n} Table", titulo + "."))

    tabela("Variable dictionary of the primary feature set",
           str(d["preditores.total"]) + " predictors: " + str(d["preditores.individuais"])
           + " describing the notified person and " + str(d["preditores.municipais"])
           + " describing the municipality of residence. Variable names are the field names of the "
           "source; the label is the one used in Fig 6 and S7 Fig.",
           t_dicionario(c), colspec="p{5.8cm}p{4.6cm}p{1.6cm}p{3.6cm}", longa=True,
           fonte="footnotesize")
    tabela("Coverage of candidate pairs by arm and imbalance strategy",
           "Number of algorithm-strategy pairs run in each arm. In the primary arm, 26 families under "
           "four settings (no balancing and three strategies) give 104 nominal pairs, of which eight "
           "are not supported by the adapter under cost-sensitive reweighting. Every pair listed "
           "covers every record of its arm exactly once across the outer partitions.",
           t_cobertura(c), colspec="p{3.7cm}rrp{1.2cm}p{1.6cm}p{1.9cm}p{1.4cm}p{1.3cm}l", fonte="footnotesize")
    tabela("Algorithms, input representation and declared search grid",
           "Frozen configuration of the primary run. Up to "
           + str(d["grade.max_config"]) + " configurations per algorithm and strategy were taken at "
           "evenly spaced positions of the declared grid, before any score was seen, and compared "
           "inside the development data of each outer fold; this is a bounded search, not an "
           "exhaustive optimisation. `Single configuration' means no grid was declared. The tabular "
           "deep learning architectures were trained with batches of 1,024, at most 50 epochs and "
           "early stopping with a patience of 8, in 32-bit precision. Two support-vector variants "
           "were excluded before the benchmark.",
           t_modelos(c, d), colspec="p{3.4cm}p{2.0cm}p{2.4cm}p{5.6cm}p{1.6cm}", longa=True,
           fonte="footnotesize", regua_por_linha=True)
    tabela("Computational cost by arm and hardware",
           "Sum of the wall-clock time of the jobs that completed. It is not the elapsed duration of "
           "the campaign and does not include failed attempts, pauses, audits or the attribution "
           "analysis.", t_custo(c), colspec="p{5.0cm}rrrrr")
    tabela("Cohort definition, exclusions and analysis roles",
           "Exclusion criteria are applied in sequence, so a record removed by one is not counted "
           "again by the next. Of the records returned, " + _n(d["coorte.declarada"]) + " were notified "
           "between 2015 and 2025; that count is not a step of the sequence. Notifications of 2025 "
           "belong to the analytic cohort and take no part in any predictive evaluation. Values behind "
           "Fig 1 of the main text.", t_coorte(c, d), colspec="p{10.5cm}r")
    tabela("Characteristics at notification of the development and 2024 evaluation records",
           "Records with one of the three modelled outcomes, by analysis role, as n (\\%). Percentages "
           "use every record of the role as denominator, including unknown and missing. `Missing' is a "
           "null or blank field, or a configured structural zero; codes that record an unknown or "
           "not-performed result are kept as levels of their own and are not missing. Labels follow the "
           "SINAN TB dictionary dated 23 July 2020, the individual notification dictionary, and the "
           "January 2010 TB dictionary for smear of other material. Original codes are retained in "
           "parentheses. Codes absent from these dictionaries remain unmapped, including code 3 in "
           "binary fields: a different official v5 PDF calls it unknown, but its applicability to "
           "these records is unverified. No codes are merged or recoded. Age is the "
           "notification year minus the birth year. The characteristics are conditional on the "
           "modelled outcomes; the notifications of 2025 are not shown.",
           t_basal_categorica(c, d), colspec="p{4.1cm}p{4.1cm}rr", longa=True, fonte="footnotesize")
    tabela("Age, contacts identified and potential observation time by role and outcome",
           "Summaries exclude missing values, whose share is given. IQR, interquartile range. "
           "Potential observation is the time from notification to the declared extraction date; it "
           "is not the observed duration of treatment.",
           t_basal_numerica(c), colspec="p{2.6cm}p{3.5cm}p{2.0cm}rrp{3.3cm}r", fonte="footnotesize")
    tabela("Eligible notifications included in and left out of the modelled outcome, development period",
           "Eligible notifications of 2015 to 2023 by whether their closure falls in the three-class "
           "scheme. `Not modelled' gathers closures outside the scheme and unresolved records. Clinical-form "
           "labels retain the SINAN code in parentheses; unmapped codes remain separate.",
           t_incluidos(c), colspec="p{3.0cm}p{4.6cm}rrrr", fonte="footnotesize")
    tabela("Composition of the outer spatial folds and configuration selected in each",
           "Primary arm, 50 geographic clusters. The configuration is the one chosen inside the "
           "development data of the fold; macro-averaged F1 is computed in the held-out fold.",
           t_dobras(c, d), colspec="rrrrrrp{5.0cm}rr", fonte="footnotesize", regua_por_linha=True)
    tabela("Confusion matrices of the two naive references",
           "Primary arm, no balancing, pooled records. Before calibration the stratified random "
           "reference draws labels in proportion to class prevalence; after calibration both "
           "references assign cure to every record.", t_baselines(c),
           colspec="p{3.0cm}p{3.6cm}p{2.9cm}p{1.6cm}p{1.6cm}p{1.9cm}", fonte="footnotesize")
    tabela("Confusion matrices of the nested selection procedure",
           "Primary arm and temporal evaluation, under the two decision rules, pooled records of each "
           "population; cells are n (row \\%). Rows are observed outcomes and columns predicted ones. "
           "The diagonal reproduces the sensitivities of Table 1 of the main text; these are pooled "
           "counts and do not replace the mean across folds.",
           t_matrizes(c), colspec="p{3.2cm}p{1.5cm}p{3.1cm}rp{2.0cm}p{1.9cm}p{2.0cm}", fonte="footnotesize")
    tabela("Geographic clusters and their influence on the pooled estimate",
           "Change in pooled macro-averaged F1 when the cluster is removed from the fixed predictions; "
           "no model is refitted. Values behind S4 Fig.", t_influencia(c), longa=True,
           colspec="rrrrr")
    tabela("Exploratory ranking of the 96 algorithm-strategy pairs in the primary arm",
           "Development period 2015 to 2023, n = " + _n(d["coorte.dev"]) + ". Macro-averaged F1. The "
           "pooled value carries a 95\\% cluster-bootstrap interval; the fold mean carries the standard "
           "deviation across the five outer folds. The difference is paired against the nested "
           "selection procedure on the same bootstrap draws. The ranking uses the evaluation folds and "
           "is therefore exploratory: it is not how the model was chosen and not the estimate of "
           "performance. Values behind Fig 2 and S1 Fig.",
           t_ranking(c, "primary"), colspec="rp{3.2cm}p{1.7cm}p{2.9cm}p{1.6cm}p{3.3cm}p{1.5cm}", longa=True,
           fonte="scriptsize")
    tabela("Calibration of the nested selection procedure before and after the calibrator",
           "Primary arm, pooled records, full precision of the released file. Intercept and slope come "
           "from a joint logistic fit; the offset is estimated separately with the slope fixed at one.",
           t_calibracao(c), colspec="p{3.4cm}lrrrrrrr", fonte="footnotesize")
    tabela("Analysis arms: pooled macro-averaged F1 and paired differences against the primary arm",
           "Nested selection procedure in each arm. Differences are arm minus primary, with selection "
           "repeated on the common universe of candidates, in the eight feature-set and design arms. "
           "For the clinical comparator the difference is primary minus comparator and sets the two "
           "complete procedures side by side on the same records, with no common universe of "
           "candidates. Arms that change the validation design (27 and 75 clusters, "
           "random municipality groups) do not isolate the effect of any variable. Values behind Fig 4.",
           t_bracos(c, d), colspec="p{3.0cm}p{2.3cm}p{2.3cm}p{2.5cm}p{2.5cm}p{2.3cm}",
           fonte="footnotesize", regua_por_linha=True)
    tabela("Exploratory ranking of the 42 pairs in the temporal evaluation",
           "Notifications of 2024, n = " + _n(d["coorte.temporal"]) + ", models fitted on 2015 to 2023. "
           "Columns as in S13 Table; there is a single evaluation partition. Values behind S6 Fig.",
           t_ranking(c, "temporal"), colspec="rp{3.2cm}p{1.7cm}p{2.9cm}p{1.6cm}p{3.3cm}p{1.5cm}", longa=True,
           fonte="scriptsize")
    tabela("Discrimination, calibration and capacity by feature set",
           "Nested selection procedure, calibrated probabilities, pooled records; no interval is "
           "available for these quantities. ECE, expected calibration error on ten fixed bins; slope "
           "and offset from logistic recalibration, the offset estimated with the slope fixed at one. "
           "`Top 5\\%' is the share of observed events among the 5 per cent of notifications with "
           "the highest probability. Values behind Fig 5.",
           t_bracos_prob(c, d), colspec="p{3.4cm}p{3.0cm}rrrrrp{1.3cm}", fonte="footnotesize")
    tabela("Attribution rank of the leading predictors in each outer fold",
           "Rank by mean absolute Shapley value on the raw margin, before calibration, averaged over "
           "three sampling seeds; the 15 predictors with the lowest median rank in each class. Ranks, "
           "not magnitudes, are compared across folds, because fold 2 explains a different family. "
           "Values behind Fig 6.", t_shap(c), colspec="p{3.2cm}p{5.2cm}p{1.8cm}rrrrr", longa=True,
           fonte="footnotesize")
    tabela("Stability of the attribution across sampling seeds",
           "Spearman correlation and overlap of the 15 highest-ranked predictors between pairs of "
           "samples drawn with seeds 42, 101 and 202, within each fold. Stability across samples of "
           "the same fold is not stability across populations.", t_shap_estabilidade(c, d),
           colspec="rp{3.0cm}p{4.0cm}rrr")
    tabela("Subgroup performance for the two unfavourable outcomes",
           "Nested selection procedure, primary arm, pooled records. Sensitivity and positive "
           "predictive value under the threshold policy, with 95\\% cluster-bootstrap intervals; AUROC "
           "has no interval. Values behind S8 Fig.",
           t_subgrupos(c, d), colspec="p{2.0cm}p{2.5cm}p{1.9cm}rrrp{2.7cm}p{2.7cm}", longa=True,
           fonte="scriptsize")

    tabela("Death attributed to tuberculosis by race or colour",
           "Nested selection procedure, primary arm, pooled records. Race or colour was missing or "
           "recorded as unknown in " + _n(d["sub.raca_cor.ausentes"]) + " notifications, which form no "
           "comparison group. Values behind Fig 7.",
           t_raca(c, d), colspec="p{1.4cm}rrrrrp{1.9cm}p{1.9cm}p{1.9cm}p{1.9cm}", fonte="scriptsize")
    tabela("Death attributed to tuberculosis by race or colour and age band",
           "Primary arm, development period. Counts and observed share of deaths in each stratum, and "
           "sensitivity under the threshold policy with 95\\% cluster-bootstrap interval. Notifications "
           "with missing or unknown race or colour, or with missing age, are left out of this table "
           "only. Values behind S9 Fig.", t_idade_raca(c), colspec="p{2.4cm}lrrrp{4.4cm}", longa=True,
           fonte="footnotesize")
    tabela("Source experiment of each arm",
           "The release identifier " + _esc(c.run_id) + " names the editorial bundle; each arm keeps "
           "the identifier of the experiment that produced its fits.", t_proveniencia(c, d),
           colspec="p{4.8cm}p{3.6cm}p{4.2cm}rp{1.4cm}", fonte="footnotesize")

    add(r"\section*{Execution amendments}")
    gb = d["custo.gb_emenda.trabalhos"]
    add(r"Two amendments changed how jobs were executed and are recorded here because they bear on "
        r"reproducibility. First, " + str(gb) + r" gradient boosting jobs of the primary arm "
        r"(three strategies in five folds) ran under an execution extension that fits up to four "
        r"inner models in parallel and reuses boosting stages shared between candidates; "
        r"candidates, splits, seeds, weights, selection and calibration were unchanged. Second, in "
        r"the random municipality groups arm four tabular deep learning jobs were re-run after "
        r"failures. In two of them, FT-Transformer and TabNet under undersampling in fold 2, the "
        r"last training batch held a single record, which batch normalisation cannot process; "
        r"that record was joined to the preceding batch. No record or partition changed, but the "
        r"last batch and the number of optimiser steps did, so no bit-for-bit identity with the "
        r"failed run is claimed. The other two jobs were re-run unchanged after a GPU wait.")
    add(r"\clearpage")

    for i, (arq, titulo, legenda) in enumerate(FIGURAS, 1):
        if not (OUT.parent / FIGS / f"{arq}.png").is_file():
            raise SystemExit(f"figura ausente: {arq}; rode scripts/editorial/figuras_c7.py")
        add(r"\begin{figure}[!ht]")
        add(r"\centering")
        add(rf"\includegraphics[width=\linewidth,height=0.78\textheight,keepaspectratio]"
            rf"{{{FIGS}/{arq}.png}}")
        add(r"\caption{{\bf " + titulo + r".} " + legenda + r"}")
        add(rf"\label{{fig:s{i}}}")
        add(r"\end{figure}")
        add(r"\clearpage")
        legendas.append((f"S{i} Fig", f"{titulo}."))

    add(r"\end{document}")
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"gravado: {OUT.relative_to(RAIZ)}")
    print("\n--- legendas para a seção Supporting information do manuscrito ---")
    for rotulo, titulo in legendas:
        print(f"  {rotulo}. {titulo}")


if __name__ == "__main__":
    main()
