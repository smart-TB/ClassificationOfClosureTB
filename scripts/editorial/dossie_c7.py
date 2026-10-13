"""Dossiê de números do L5: cada valor do manuscrito com arquivo, filtro, regra e agregação.

Substitui, para a reanálise corrigida, o papel que `RESULTS_NUMEROS.md` tinha para a corrida
histórica. Nada aqui é digitado: os valores são lidos pelo `Catalogo`, que só entrega
artefatos listados no `release_manifest.json` e com hash conferido.

Três distinções atravessam todas as chaves, porque foram as que a revisão encontrou
misturadas: a REGRA de decisão (`argmax` é a principal; `policy` é a secundária), a
AGREGAÇÃO (`pooled` sobre registros agrupados, com intervalo por bootstrap de clusters;
`fold_mean`/`fold_sd` entre dobras externas, sem intervalo) e o BRAÇO. O intervalo agrupado
nunca é anexado à média entre dobras: são chaves separadas e o texto cita uma ou outra.

Uso:
    python scripts/editorial/dossie_c7.py            # grava dossie_L5.json e dossie_L5.md
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from catalogo import C04, C23, REL, RAIZ, Catalogo

SAIDA = RAIZ / "docs/manuscrito/colaboracao/claude"
CLASSES = {0: "cure", 1: "tb_death", 2: "treatment_interruption"}
REGRAS = ("argmax", "policy")
BRACOS_COMPARADOS = ("individual", "municipal", "contemporary", "without_diagnostics",
                     "hiv_uncertain", "k27", "k75", "random_group")
PROC = "nested_selection_procedure"
CAPACIDADES = (0.05, 0.10, 0.20)


@dataclass
class Dossie:
    valores: dict = field(default_factory=dict)
    origem: dict = field(default_factory=dict)

    def put(self, chave: str, valor, *, arquivo: str, filtro: str, regra: str = "-",
            agregacao: str = "-", braco: str = "-") -> None:
        if chave in self.valores:
            raise KeyError(f"chave duplicada no dossiê: {chave}")
        if isinstance(valor, (np.floating, np.integer)):
            valor = valor.item()
        self.valores[chave] = valor
        self.origem[chave] = {"arquivo": arquivo, "filtro": filtro, "regra": regra,
                              "agregacao": agregacao, "braco": braco}

    def __getitem__(self, chave: str):
        return self.valores[chave]


def _um(df: pd.DataFrame, **cond) -> pd.Series:
    """Exatamente uma linha; zero ou várias é erro de filtro, nunca escolha silenciosa."""
    m = pd.Series(True, index=df.index)
    for k, v in cond.items():
        m &= df[k].isna() if v is None else (df[k] == v)
    g = df[m]
    if len(g) != 1:
        raise LookupError(f"esperava 1 linha para {cond}, obtive {len(g)}")
    return g.iloc[0]


def _ic(d: Dossie, chave: str, df: pd.DataFrame, arquivo: str, *, regra: str, braco: str,
        opcional: bool = False, **cond) -> None:
    """Estimativa agrupada com intervalo. `opcional=True` registra a não estimabilidade em
    `<chave>.status` em vez de falhar: é o caso de um PPV sem nenhuma predição positiva."""
    r = _um(df, **cond)
    if r["status"] != "ok":
        if not opcional:
            raise ValueError(f"{chave}: estimativa com status {r['status']}")
        d.put(chave + ".status", r["status"], arquivo=arquivo, regra=regra, braco=braco,
              filtro=", ".join(f"{k}={v}" for k, v in cond.items()))
        return
    filtro = ", ".join(f"{k}={v}" for k, v in cond.items())
    kw = dict(arquivo=arquivo, filtro=filtro, regra=regra, braco=braco,
              agregacao="pooled_records; IC 95% por bootstrap de clusters geográficos")
    d.put(chave, r["estimate"], **kw)
    d.put(chave + ".lo", r["ci_low"], **kw)
    d.put(chave + ".hi", r["ci_high"], **kw)


def _coorte(c: Catalogo, d: Dossie) -> None:
    f = f"{REL}/corrected_cohort_flow.csv"
    fluxo = c.csv(f).set_index("stage")["n"]
    for etapa in ("recuperada", "declarada", "elegivel", "analitica"):
        d.put(f"coorte.{etapa}", int(fluxo[etapa]), arquivo=f, filtro=f"stage={etapa}")

    f = f"{REL}/corrected_eligibility_exclusions.csv"
    ex = c.csv(f)
    for r in ex.itertuples():
        d.put(f"coorte.excl.{r.reason}", int(r.n), arquivo=f,
              filtro=f"reason={r.reason}; coluna n (aplicação sequencial)")
    if fluxo["recuperada"] - ex["n"].sum() != fluxo["elegivel"]:
        raise ValueError("exclusões sequenciais não reconciliam recuperada -> elegível")
    d.put("coorte.excl.total", int(ex["n"].sum()), arquivo=f, filtro="soma da coluna n")

    f = f"{REL}/outcome_selection_counts.csv"
    oc = c.csv(f)
    fora = int(oc.loc[~oc.analytic_included, "n"].sum())
    if fluxo["elegivel"] - fora != fluxo["analitica"]:
        raise ValueError("desfechos fora do esquema não reconciliam elegível -> analítica")
    d.put("coorte.fora_esquema", fora, arquivo=f, filtro="analytic_included=False; soma de n")
    nao_resolvido = int(oc.loc[oc.mapped_outcome == "unresolved_or_censored", "n"].sum())
    d.put("coorte.fora_esquema.nao_resolvido", nao_resolvido, arquivo=f,
          filtro="mapped_outcome=unresolved_or_censored; soma de n")
    d.put("coorte.fora_esquema.outra_categoria", fora - nao_resolvido, arquivo=f,
          filtro="mapped_outcome=excluded; soma de n")
    dev = oc[oc.study_role == "development"]
    d.put("coorte.dev.obito_outras_causas", int(_um(dev, SITUA_ENCE=4)["n"]), arquivo=f,
          filtro="study_role=development, SITUA_ENCE=4")
    d.put("coorte.dev.abandono_primario", int(_um(dev, SITUA_ENCE=10)["n"]), arquivo=f,
          filtro="study_role=development, SITUA_ENCE=10")

    f = f"{REL}/population_reconciliation.json"
    pr = c.json(f)
    for chave, campo in (("coorte.dev", "primary_development"),
                         ("coorte.temporal", "temporal_test"),
                         ("coorte.futuro", "unused_future"),
                         ("preditores.total", "primary_features"),
                         ("preditores.individuais", "individual_features"),
                         ("preditores.municipais", "municipal_features"),
                         ("execucao.checkpoints", "checkpoint_count")):
        d.put(chave, pr[campo], arquivo=f, filtro=campo)
    if pr["primary_development"] + pr["temporal_test"] + pr["unused_future"] != pr["total_analytic"]:
        raise ValueError("papéis não somam a coorte analítica")
    if pr["total_analytic"] != fluxo["analitica"]:
        raise ValueError("coorte analítica diverge entre fluxo e reconciliação")

    f = "data/correction_audit/partitions/population_counts.csv"
    pc = c.csv(f)
    for papel, rot in (("development", "dev"), ("temporal_test", "temporal"),
                       ("unused_future", "futuro")):
        g = pc[pc.role == papel].groupby("target")["n"].sum()
        tot = int(g.sum())
        if tot != d[f"coorte.{rot}"]:
            raise ValueError(f"contagem por classe não reconcilia {papel}")
        for cls in CLASSES.values():
            d.put(f"coorte.{rot}.{cls}", int(g[cls]), arquivo=f,
                  filtro=f"role={papel}, target={cls}; soma entre anos")
            d.put(f"coorte.{rot}.{cls}.pct", 100 * g[cls] / tot, arquivo=f,
                  filtro=f"role={papel}, target={cls}; n / total do papel")

    f = "data/correction_audit/partitions/manifest.json"
    pm = c.json(f)
    d.put("particao.municipios_ajuste", pm["n_geography_fit_municipalities"], arquivo=f,
          filtro="n_geography_fit_municipalities")
    d.put("particao.municipios_mapeados", pm["n_mapped_municipalities"], arquivo=f,
          filtro="n_mapped_municipalities")

    f = f"{REL}/included_excluded_by_group.csv"
    ie = c.csv(f)
    ie = ie[ie.role == "development"]
    for eixo, rot in (("NU_ANO", "ano"), ("REGIAO", "regiao")):
        g = ie[ie.axis == eixo].sort_values("excluded_fraction")
        for ponta, r in (("min", g.iloc[0]), ("max", g.iloc[-1])):
            d.put(f"coorte.excluido.{rot}.{ponta}", 100 * r["excluded_fraction"], arquivo=f,
                  filtro=f"role=development, axis={eixo}, group={r['group']}")
            d.put(f"coorte.excluido.{rot}.{ponta}.grupo", str(r["group"]), arquivo=f,
                  filtro=f"role=development, axis={eixo}; grupo de fração {ponta}")
    tot = ie[ie.axis == "NU_ANO"][["eligible", "included"]].sum()
    if tot["included"] != d["coorte.dev"]:
        raise ValueError("incluídos por ano não reconciliam o desenvolvimento")
    d.put("coorte.dev.elegiveis", int(tot["eligible"]), arquivo=f,
          filtro="role=development, axis=NU_ANO; soma de eligible")
    d.put("coorte.dev.excluido.pct", 100 * (1 - tot["included"] / tot["eligible"]), arquivo=f,
          filtro="role=development, axis=NU_ANO; 1 - included/eligible")

    f = f"{REL}/all_cause_population_counts.csv"
    ac = c.csv(f)
    g = ac[ac.NU_ANO <= 2023].groupby("target_all_cause_3class")["n"].sum()
    d.put("todas_causas.dev", int(g.sum()), arquivo=f, filtro="NU_ANO<=2023; soma de n")
    d.put("todas_causas.dev.obito", int(g["death_all_cause"]), arquivo=f,
          filtro="NU_ANO<=2023, target=death_all_cause")


def _grade(c: Catalogo, d: Dossie) -> None:
    f = "data/reanalysis/c6_20260919/declaration.json"
    dec = c.json(f)
    for chave, campo in (("grade.familias", "n_models"), ("grade.pares", "compatible_pairs"),
                         ("grade.trabalhos", "outer_jobs"),
                         ("grade.max_config", "max_configurations_per_pair"),
                         ("grade.dobras_externas", "n_outer"),
                         ("grade.dobras_internas", "n_inner_selection")):
        d.put(chave, dec[campo], arquivo=f, filtro=campo, braco="primary")
    f = f"{REL}/primary_incompatible_pairs.csv"
    inc = c.csv(f)
    if len(inc) != len(dec["unsupported"]):
        raise ValueError("incompatibilidades divergem entre declaração e liberação")
    d.put("grade.incompativeis", len(inc), arquivo=f, filtro="linhas", braco="primary")
    d.put("grade.incompativeis.modelos", sorted(inc.model), arquivo=f, filtro="coluna model",
          braco="primary")
    f = f"{REL}/candidate_coverage.csv"
    cov = c.csv(f)
    p = cov[cov.arm == "primary"]
    if len(p) != dec["compatible_pairs"] or p.model.nunique() != dec["n_models"]:
        raise ValueError("cobertura do principal diverge da declaração")
    if not ((p.n_unique == dec["expected_records_per_pair"]) & (p.folds == 5)).all():
        raise ValueError("par do principal sem cobertura completa")
    d.put("grade.cobertura_completa", int(len(p)), arquivo=f,
          filtro="arm=primary; pares com n_unique=707870 e folds=5", braco="primary")
    for braco, g in cov.groupby("arm"):
        d.put(f"grade.{braco}.pares", int(len(g)), arquivo=f, filtro=f"arm={braco}; linhas",
              braco=braco)
        d.put(f"grade.{braco}.familias", int(g.model.nunique()), arquivo=f,
              filtro=f"arm={braco}; modelos distintos", braco=braco)


def _bracos(c: Catalogo, d: Dossie) -> None:
    f = f"{REL}/arm_overview.json"
    for a in c.json(f):
        b = a["arm"]
        d.put(f"{b}.n", a["n"], arquivo=f, filtro=f"arm={b}, n", braco=b)
        for regra in REGRAS:
            pre = f"pred_{regra}_"
            ag_p = "pooled_records; IC 95% por bootstrap de clusters geográficos"
            for suf, campo, ag in (("pooled", "pooled_f1", ag_p), ("pooled.lo", "pooled_ci_low", ag_p),
                                   ("pooled.hi", "pooled_ci_high", ag_p),
                                   ("fold_mean", "fold_mean_f1", "média não ponderada entre dobras externas"),
                                   ("fold_sd", "fold_sd_f1", "desvio-padrão entre dobras externas")):
                d.put(f"{b}.{regra}.f1.{suf}", a[pre + campo], arquivo=f,
                      filtro=f"arm={b}, {pre + campo}", regra=f"pred_{regra}", agregacao=ag, braco=b)

    f = f"{REL}/independent_f1_checks.csv"
    chk = c.csv(f)
    if not chk.passed.all():
        raise ValueError("conferência independente de macro-F1 com falha")
    for r in chk.itertuples():
        regra = r.rule.replace("pred_", "")
        if abs(r.audit_f1 - d[f"{r.arm}.{regra}.f1.pooled"]) > 1e-9:
            raise ValueError(f"{r.arm}/{r.rule}: visão geral diverge da auditoria")
    d.put("conferencia.f1_independentes", int(len(chk)), arquivo=f, filtro="linhas com passed=True")


def _por_classe(c: Catalogo, d: Dossie, braco: str) -> None:
    """Métricas de classificação do procedimento selecionado, por regra, com intervalo."""
    aud = c.auditoria(braco)
    for regra in REGRAS:
        f = f"{aud}/cluster_intervals_pred_{regra}.csv"
        ci = c.csv(f)
        ci = ci[(ci.comparison == PROC) & (ci.kind == "estimate")]
        for met in ("accuracy", "balanced_accuracy", "f1_macro"):
            _ic(d, f"{braco}.{regra}.{met}", ci, f, regra=f"pred_{regra}", braco=braco, metric=met)
        for k, cls in CLASSES.items():
            for met in ("sensitivity", "specificity", "ppv", "f1"):
                _ic(d, f"{braco}.{regra}.{cls}.{met}", ci, f, regra=f"pred_{regra}", braco=braco,
                    metric=f"{met}:{k}", opcional=True)
        if abs(d[f"{braco}.{regra}.f1_macro"] - d[f"{braco}.{regra}.f1.pooled"]) > 1e-9:
            raise ValueError(f"{braco}/{regra}: intervalo e visão geral divergem")

    f = f"{aud}/probability_metrics_pooled.csv"
    pm = c.csv(f)
    pm = pm[pm.comparison == PROC]
    for k, cls in CLASSES.items():
        r = _um(pm, class_code=k)
        for col in ("prevalence", "auroc", "auprc", "brier", "ece_fixed_10",
                    "calibration_intercept", "calibration_slope", "calibration_in_large"):
            d.put(f"{braco}.prob.{cls}.{col}", r[col], arquivo=f,
                  filtro=f"comparison={PROC}, class_code={k}, {col}",
                  regra="independente de regra (probabilidade calibrada)",
                  agregacao="pooled_records; sem intervalo", braco=braco)

    f = f"{aud}/capacity_pooled.csv"
    cp = c.csv(f)
    cp = cp[cp.comparison == PROC]
    for k in (1, 2):
        for cap in CAPACIDADES:
            r = _um(cp, class_code=k, capacity_fraction=cap)
            for col, rot in (("ppv", "ppv"), ("sensitivity", "capturado"),
                             ("notifications_per_observed_event", "nns"),
                             ("selected", "selecionados"), ("captured_events", "eventos")):
                d.put(f"{braco}.cap{int(cap * 100)}.{CLASSES[k]}.{rot}", r[col], arquivo=f,
                      filtro=f"comparison={PROC}, class_code={k}, capacity_fraction={cap}, {col}",
                      regra="ordenamento pela probabilidade calibrada; sem rótulo",
                      agregacao="pooled_records; sem intervalo", braco=braco)


def _principal(c: Catalogo, d: Dossie) -> None:
    aud = c.auditoria("primary")
    sel_f = f"{C23}/primary_selected/selected_by_fold.json"
    sel = c.json(sel_f)
    fam = [s["model"] for s in sel]
    d.put("principal.selecao.familias", fam, arquivo=sel_f, filtro="model por outer_fold",
          braco="primary")
    d.put("principal.selecao.estrategias", [s["strategy"] for s in sel], arquivo=sel_f,
          filtro="strategy por outer_fold", braco="primary")
    d.put("principal.selecao.calibracao", [s["calibration"] for s in sel], arquivo=sel_f,
          filtro="calibration por outer_fold", braco="primary")
    d.put("principal.selecao.n_lightgbm", fam.count("lightgbm"), arquivo=sel_f,
          filtro="dobras com model=lightgbm", braco="primary")
    d.put("principal.selecao.n_catboost", fam.count("catboost"), arquivo=sel_f,
          filtro="dobras com model=catboost", braco="primary")
    interno = [s["argmax_f1_macro"] for s in sel]
    d.put("principal.selecao.interno.min", min(interno), arquivo=sel_f,
          filtro="argmax_f1_macro interno; mínimo entre dobras", regra="pred_argmax",
          agregacao="pontuação interna da seleção, por dobra", braco="primary")
    d.put("principal.selecao.interno.max", max(interno), arquivo=sel_f,
          filtro="argmax_f1_macro interno; máximo entre dobras", regra="pred_argmax",
          agregacao="pontuação interna da seleção, por dobra", braco="primary")

    f = f"{aud}/bootstrap_design.json"
    bd = c.json(f)
    d.put("bootstrap.clusters_por_dobra", list(bd["clusters_per_outer_fold"].values()), arquivo=f,
          filtro="clusters_per_outer_fold", braco="primary")
    f = f"{aud}/audit_manifest.json"
    d.put("bootstrap.replicas", c.json(f)["replicates"], arquivo=f, filtro="replicates",
          braco="primary")

    # ranking exploratório: estimativas agrupadas por par e diferença pareada contra o
    # procedimento de seleção. Não é a estimativa de desempenho.
    f = f"{aud}/cluster_intervals_pred_argmax.csv"
    ci = c.csv(f)
    est = ci[(ci.metric == "f1_macro") & (ci.kind == "estimate") & (ci.comparison != PROC)]
    est = est.sort_values("estimate", ascending=False).reset_index(drop=True)
    dif = ci[(ci.metric == "f1_macro") & (ci.kind == "paired_difference")]
    for pos, r in enumerate(est.head(6).itertuples(), 1):
        kw = dict(arquivo=f, regra="pred_argmax", braco="primary",
                  agregacao="pooled_records; IC 95% por bootstrap de clusters geográficos")
        d.put(f"ranking.{pos}.par", r.comparison, filtro=f"posição {pos} por estimate, metric=f1_macro",
              **kw)
        d.put(f"ranking.{pos}.f1", r.estimate, filtro=f"comparison={r.comparison}, metric=f1_macro", **kw)
        d.put(f"ranking.{pos}.f1.lo", r.ci_low, filtro=f"comparison={r.comparison}, ci_low", **kw)
        d.put(f"ranking.{pos}.f1.hi", r.ci_high, filtro=f"comparison={r.comparison}, ci_high", **kw)
        x = _um(dif, comparison=r.comparison)
        for suf, col in (("", "estimate"), (".lo", "ci_low"), (".hi", "ci_high")):
            d.put(f"ranking.{pos}.dif{suf}", x[col], filtro=f"comparison={r.comparison}, "
                  f"kind=paired_difference, reference={PROC}, {col}", **kw)
    melhores = est.assign(model=est.comparison.str.split("__").str[0]).drop_duplicates("model")
    sem = est[est.comparison.str.endswith("__none")]
    d.put("ranking.n_sem_balanceamento_no_topo10",
          int(est.head(10).comparison.str.endswith("__none").sum()), arquivo=f,
          filtro="dez maiores estimate de f1_macro; pares com estratégia none",
          regra="pred_argmax", agregacao="pooled_records", braco="primary")
    del sem
    for mod in ("ft_transformer", "tabnet", "category_embedding", "tab_transformer",
                "logistic_regression", "majority_class", "stratified_random"):
        r = _um(melhores, model=mod)
        kw = dict(arquivo=f, regra="pred_argmax", braco="primary",
                  agregacao="pooled_records; IC 95% por bootstrap de clusters geográficos")
        d.put(f"ranking.{mod}.par", r.comparison, filtro=f"melhor par de {mod} por estimate", **kw)
        d.put(f"ranking.{mod}.f1", r.estimate, filtro=f"comparison={r.comparison}, metric=f1_macro", **kw)
        d.put(f"ranking.{mod}.f1.lo", r.ci_low, filtro=f"comparison={r.comparison}, ci_low", **kw)
        d.put(f"ranking.{mod}.f1.hi", r.ci_high, filtro=f"comparison={r.comparison}, ci_high", **kw)
        x = _um(dif, comparison=r.comparison)
        for suf, col in (("", "estimate"), (".lo", "ci_low"), (".hi", "ci_high")):
            d.put(f"ranking.{mod}.dif{suf}", x[col], filtro=f"comparison={r.comparison}, "
                  f"kind=paired_difference, {col}", **kw)
    d.put("ranking.posicao.ft_transformer",
          int(est.index[est.comparison == d["ranking.ft_transformer.par"]][0]) + 1, arquivo=f,
          filtro="posição do melhor par de ft_transformer entre os 96", regra="pred_argmax",
          agregacao="pooled_records", braco="primary")

    kwr = dict(arquivo=f, regra="pred_argmax", agregacao="pooled_records", braco="primary")
    d.put("ranking.n_familias_melhor_sem_balanceamento",
          int(melhores.comparison.str.endswith("__none").sum()),
          filtro="famílias cujo par de maior estimate usa a estratégia none", **kwr)
    d.put("ranking.n_pares_ic_inclui_zero", int(((dif.ci_low <= 0) & (dif.ci_high >= 0)).sum()),
          filtro="kind=paired_difference, metric=f1_macro; intervalos que contêm zero", **kwr)
    d.put("ranking.n_pares_acima_procedimento", int((dif.estimate > 0).sum()),
          filtro="kind=paired_difference, metric=f1_macro; estimate > 0", **kwr)
    ref = d["ranking.majority_class.f1"]
    d.put("ranking.familias_no_nivel_da_referencia",
          sorted(m for m in melhores.loc[(melhores.estimate - ref).abs() < 1e-9, "model"]
                 if m not in ("majority_class", "stratified_random")),
          filtro="famílias não ingênuas cujo melhor par iguala o f1_macro da classe majoritária", **kwr)
    # as duas referências ingênuas: mesma predição em todas as estratégias?
    base = est[est.comparison.str.startswith(("majority_class", "stratified_random"))]
    d.put("baseline.n_pares", int(len(base)), arquivo=f,
          filtro="pares de majority_class e stratified_random", braco="primary")
    d.put("baseline.f1_distintos", int(base.estimate.round(12).nunique()), arquivo=f,
          filtro="valores distintos de f1_macro entre os pares de referência",
          regra="pred_argmax", agregacao="pooled_records", braco="primary")

    f = f"{C04}/baseline_confusions/confusions_pooled.csv"
    bc = c.csv(f)
    for regra in ("uncalibrated_argmax", "calibrated_argmax", "calibrated_policy"):
        for mod in ("majority_class", "stratified_random"):
            g = bc[(bc.rule == regra) & (bc.model == mod) & (bc.strategy == "none")]
            tot = g["count"].sum()
            for k, cls in CLASSES.items():
                d.put(f"baseline.{mod}.{regra}.pred_{cls}.pct",
                      100 * g.loc[g.predicted == k, "count"].sum() / tot, arquivo=f,
                      filtro=f"model={mod}, strategy=none, rule={regra}, predicted={k}; soma/total",
                      regra=regra, agregacao="pooled_records", braco="primary")

    f = f"{REL}/primary_calibration_before_after.csv"
    cal = c.csv(f)
    for estagio, rot in (("before_calibration", "antes"), ("after_calibration", "depois")):
        for k, cls in CLASSES.items():
            r = _um(cal, stage=estagio, class_code=k)
            for col in ("ece_fixed_10", "calibration_intercept", "calibration_slope",
                        "calibration_in_large", "brier", "log_loss", "auroc", "auprc"):
                d.put(f"calibracao.{rot}.{cls}.{col}", r[col], arquivo=f,
                      filtro=f"stage={estagio}, class_code={k}, {col}",
                      regra="independente de regra", agregacao="pooled_records; sem intervalo",
                      braco="primary")
    for k, cls in CLASSES.items():  # o estágio calibrado tem de coincidir com a auditoria
        if abs(d[f"calibracao.depois.{cls}.ece_fixed_10"] - d[f"primary.prob.{cls}.ece_fixed_10"]) > 1e-12:
            raise ValueError("calibração exportada diverge da auditoria principal")

    f = f"{aud}/cluster_influence.csv"
    inf = c.csv(f)
    g = inf[inf.rule == "pred_argmax"]
    maior = g.loc[g.record_fraction.idxmax()]
    d.put("influencia.maior_cluster.pct", 100 * maior["record_fraction"], arquivo=f,
          filtro="rule=pred_argmax; maior record_fraction", braco="primary")
    d.put("influencia.maior_cluster.n", int(maior["n_records"]), arquivo=f,
          filtro="rule=pred_argmax; n_records do maior cluster", braco="primary")
    d.put("influencia.menor_cluster.n", int(g.n_records.min()), arquivo=f,
          filtro="rule=pred_argmax; menor n_records", braco="primary")
    d.put("influencia.variacao.min", g.change.min(), arquivo=f,
          filtro="rule=pred_argmax; mínimo de change", regra="pred_argmax",
          agregacao="pooled_records com um cluster retirado; sem reajuste", braco="primary")
    d.put("influencia.variacao.max", g.change.max(), arquivo=f,
          filtro="rule=pred_argmax; máximo de change", regra="pred_argmax",
          agregacao="pooled_records com um cluster retirado; sem reajuste", braco="primary")


def _comparacoes(c: Catalogo, d: Dossie) -> None:
    completa = [(s["model"], s["strategy"], s["candidate"])
                for s in c.json(f"{C23}/primary_selected/selected_by_fold.json")]
    for b in BRACOS_COMPARADOS:
        base = f"{C04}/comparisons/{b}"
        restr = [(s["model"], s["strategy"], s["candidate"])
                 for s in c.json(f"{base}/primary_selected_by_fold.json")]
        man = c.json(f"{base}/manifest.json")
        d.put(f"{b}.comparacao.pares_comuns", len(man["common_pairs"]), arquivo=f"{base}/manifest.json",
              filtro="common_pairs", braco=b)
        d.put(f"{b}.comparacao.principal_identico", restr == completa,
              arquivo=f"{base}/primary_selected_by_fold.json",
              filtro="seleção restrita ao universo comum igual à seleção principal completa", braco=b)
        d.put(f"{b}.selecao.familias", [s["model"] for s in c.json(f"{base}/{b}_selected_by_fold.json")],
              arquivo=f"{base}/{b}_selected_by_fold.json", filtro="model por outer_fold", braco=b)
        for regra in REGRAS:
            f = f"{base}/paired_pred_{regra}.csv"
            p = c.csv(f)
            _ic(d, f"{b}.{regra}.dif", p, f, regra=f"pred_{regra}", braco=f"{b} - primary",
                comparison=b, kind="paired_difference", metric="f1_macro")
            prim = _um(p, comparison="primary", kind="estimate", metric="f1_macro")["estimate"]
            if restr == completa and abs(prim - d[f"primary.{regra}.f1.pooled"]) > 1e-9:
                raise ValueError(f"{b}/{regra}: principal do universo comum diverge do completo")

    base = f"{C23}/primary_vs_clinical"
    for regra in REGRAS:
        f = f"{base}/paired_pred_{regra}.csv"
        p = c.csv(f)
        for met, rot in (("f1_macro", "dif"), ("sensitivity:1", "dif.tb_death.sensitivity"),
                         ("ppv:1", "dif.tb_death.ppv"),
                         ("sensitivity:2", "dif.treatment_interruption.sensitivity")):
            _ic(d, f"clinical.{regra}.{rot}", p, f, regra=f"pred_{regra}", braco="primary - clinical",
                comparison="primary", kind="paired_difference", metric=met)

    f = f"{base}/decision_curves.csv"
    dc = c.csv(f)
    for b in ("primary", "clinical"):
        for k in (1, 2):
            for t in (0.05, 0.10, 0.20):
                r = _um(dc[np.isclose(dc.threshold, t)], comparison=b, class_code=k)
                kw = dict(arquivo=f, regra=f"intervir quando probabilidade calibrada >= {t}",
                          agregacao="pooled_records; sem intervalo", braco=b)
                tt = int(round(t * 100))
                d.put(f"curva.{b}.{CLASSES[k]}.t{tt}.nb", r["nb_modelo"],
                      filtro=f"comparison={b}, class_code={k}, threshold={t}, nb_modelo", **kw)
                if b == "primary":
                    d.put(f"curva.todos.{CLASSES[k]}.t{tt}.nb", r["nb_intervir_em_todos"],
                          filtro=f"class_code={k}, threshold={t}, nb_intervir_em_todos", **kw)
                d.put(f"curva.{b}.{CLASSES[k]}.t{tt}.alertas_por_1000", r["alertas_por_1000"],
                      filtro=f"comparison={b}, class_code={k}, threshold={t}, alertas_por_1000", **kw)


def _curvas_extra(c: Catalogo, d: Dossie) -> None:
    """Primeiro limiar em que sinalizar todas as notificações tem benefício líquido negativo."""
    f = f"{C23}/primary_vs_clinical/decision_curves.csv"
    dc = c.csv(f)
    p = dc[dc.comparison == "primary"]
    for k in (1, 2):
        g = p[(p.class_code == k) & (p.nb_intervir_em_todos < 0)].sort_values("threshold")
        d.put(f"curva.todos.{CLASSES[k]}.primeiro_negativo", round(float(g.threshold.iloc[0]), 2),
              arquivo=f, filtro=f"comparison=primary, class_code={k}; menor threshold com "
              "nb_intervir_em_todos < 0 (grade de 0,01)", agregacao="pooled_records", braco="primary")
    for b in ("primary", "clinical"):
        for cap in (5,):
            for k in (1, 2):
                d.put(f"{b}.cap{cap}.{CLASSES[k]}.fora.pct",
                      100 * (1 - d[f"{b}.cap{cap}.{CLASSES[k]}.capturado"]),
                      arquivo=d.origem[f"{b}.cap{cap}.{CLASSES[k]}.capturado"]["arquivo"],
                      filtro="100 x (1 - sensitivity) na capacidade indicada",
                      agregacao="pooled_records; sem intervalo", braco=b)


def _temporal(c: Catalogo, d: Dossie) -> None:
    aud = c.auditoria("temporal")
    f = f"{C23}/temporal_selected/selected_by_fold.json"
    s = c.json(f)[0]
    d.put("temporal.selecao.familia", s["model"], arquivo=f, filtro="model", braco="temporal")
    d.put("temporal.selecao.estrategia", s["strategy"], arquivo=f, filtro="strategy", braco="temporal")
    f = f"{aud}/cluster_intervals_pred_argmax.csv"
    ci = c.csv(f)
    est = ci[(ci.metric == "f1_macro") & (ci.kind == "estimate")]
    r = _um(est, comparison="majority_class__none")
    kw = dict(arquivo=f, regra="pred_argmax", braco="temporal",
              agregacao="pooled_records; IC 95% por bootstrap de clusters geográficos")
    d.put("temporal.baseline.f1", r["estimate"], filtro="comparison=majority_class__none, f1_macro", **kw)
    d.put("temporal.baseline.f1.lo", r["ci_low"], filtro="comparison=majority_class__none, ci_low", **kw)
    d.put("temporal.baseline.f1.hi", r["ci_high"], filtro="comparison=majority_class__none, ci_high", **kw)


def _subgrupos(c: Catalogo, d: Dossie) -> None:
    aud = c.auditoria("primary")
    f = f"{aud}/subgroup_metrics.csv"
    sg = c.csv(f)
    sg["group"] = sg["group"].astype(str)
    fi = f"{aud}/subgroup_cluster_intervals.csv"
    si = c.csv(fi)
    si["group"] = si["group"].astype(str)
    eixos = {"regiao": ["Norte", "Nordeste", "Centro-Oeste", "Sudeste", "Sul"],
             "vulnerabilidade_municipal": ["Q1", "Q5"],
             "raca_cor": ["1", "2", "3", "4", "5"],
             "faixa_etaria": ["0-14", "15-24", "25-34", "35-44", "45-59", "60-64", "65+"],
             "sexo": ["F", "M"],
             "feature_missing_fraction": ["0-10%", ">10-25%", ">25-50%"]}
    for eixo, grupos in eixos.items():
        for g in grupos:
            for k in (1, 2):
                cls = CLASSES[k]
                r = _um(sg, axis=eixo, group=g, rule="pred_policy", class_code=k)
                for col in ("n", "events", "prevalence", "auroc", "auprc", "calibration_slope",
                            "ece_fixed_10"):
                    d.put(f"sub.{eixo}.{g}.{cls}.{col}", r[col], arquivo=f,
                          filtro=f"axis={eixo}, group={g}, class_code={k}, {col}",
                          regra="independente de regra", agregacao="pooled_records; sem intervalo",
                          braco="primary")
                for regra in REGRAS:
                    x = si[(si.axis == eixo) & (si.group == g) & (si.decision_rule == f"pred_{regra}")]
                    for met in ("sensitivity", "ppv"):
                        y = _um(x, metric=f"{met}:{k}")
                        if y["status"] != "ok":
                            continue  # estimativa suprimida: fica fora do dossiê e do texto
                        _ic(d, f"sub.{eixo}.{g}.{cls}.{regra}.{met}", x, fi, regra=f"pred_{regra}",
                            braco="primary", metric=f"{met}:{k}")
    f = f"{aud}/subgroup_coverage.csv"
    cv = c.csv(f)
    for eixo in ("raca_cor", "escolaridade", "faixa_etaria", "sexo", "vulnerabilidade_municipal"):
        r = cv[(cv.axis == eixo) & cv.total.notna()].iloc[0]
        d.put(f"sub.{eixo}.ausentes", int(r["missing"]), arquivo=f, filtro=f"axis={eixo}, missing",
              braco="primary")
        d.put(f"sub.{eixo}.ausentes.pct", 100 * r["missing"] / r["total"], arquivo=f,
              filtro=f"axis={eixo}, missing/total", braco="primary")

    # idade x raça/cor: descrição estratificada; denominadores < 30 já vêm suprimidos
    f = f"{C04}/age_race_stratification/stratum_counts.csv"
    sc = c.csv(f)
    ob = sc[sc.class_code == 1]
    for faixa in ("0-14", "65+"):
        ind = _um(ob, race=5, age=faixa)
        resto = ob[(ob.race != 5) & (ob.age == faixa)]
        d.put(f"idade_raca.indigena.{faixa}.n", int(ind["n"]), arquivo=f,
              filtro=f"race=5, age={faixa}, class_code=1, n", braco="primary")
        d.put(f"idade_raca.indigena.{faixa}.obitos", int(ind["events"]), arquivo=f,
              filtro=f"race=5, age={faixa}, class_code=1, events", braco="primary")
        d.put(f"idade_raca.indigena.{faixa}.letalidade.pct", 100 * ind["events"] / ind["n"], arquivo=f,
              filtro=f"race=5, age={faixa}, class_code=1; events/n", braco="primary")
        d.put(f"idade_raca.demais.{faixa}.letalidade.pct", 100 * resto["events"].sum() / resto["n"].sum(),
              arquivo=f, filtro=f"race in 1-4, age={faixa}, class_code=1; soma events / soma n",
              braco="primary")
    f = f"{C04}/age_race_stratification/coverage.json"
    cg = c.json(f)
    for campo in ("analyzed", "excluded", "missing_race", "missing_age"):
        d.put(f"idade_raca.{campo}", cg[campo], arquivo=f, filtro=campo, braco="primary")
    f = f"{C04}/age_race_stratification/stratified_metrics.csv"
    am = c.csv(f)
    x = am[(am.decision_rule == "pred_policy") & (am.metric == "sensitivity:1")]
    d.put("idade_raca.sens_obito.nao_estimaveis", int((x.status != "ok").sum()), arquivo=f,
          filtro="decision_rule=pred_policy, metric=sensitivity:1; status != ok",
          regra="pred_policy", braco="primary")
    for faixa in ("45-59", "65+"):
        for raca, rot in ((5, "indigena"), (1, "branca")):
            _ic(d, f"idade_raca.{rot}.{faixa}.sens_obito", x, f, regra="pred_policy", braco="primary",
                race=raca, age=faixa)


def shap_por_dobra(c: Catalogo) -> pd.DataFrame:
    """Resumo de TreeSHAP por dobra, com média entre as três sementes de amostragem.

    Cada dobra é um modelo e, nas dobras em que a família muda, uma escala diferente. As
    magnitudes só são comparadas DENTRO de uma dobra; entre dobras compara-se o posto.
    """
    dic = c.csv(f"{REL}/primary_feature_dictionary.csv").set_index("name")["scope"]
    partes = []
    for fold in range(5):
        for seed in (42, 101, 202):
            s = c.csv(f"{C04}/attribution/fold{fold}/seed{seed}_summary.csv")
            partes.append(s.assign(seed=seed))
    t = pd.concat(partes, ignore_index=True)
    g = (t.groupby(["outer_fold", "model", "classe", "feature"], as_index=False)
          .agg(mean_abs_shap=("mean_abs_shap", "mean"), mean_shap=("mean_shap", "mean")))
    g["rank"] = g.groupby(["outer_fold", "classe"])["mean_abs_shap"].rank(ascending=False, method="first")
    g["scope"] = g["feature"].map(dic)
    if g["scope"].isna().any():
        raise ValueError("variável do SHAP ausente do dicionário de preditores")
    g["share"] = g["mean_abs_shap"] / g.groupby(["outer_fold", "classe"])["mean_abs_shap"].transform("sum")
    return g


def _atribuicao(c: Catalogo, d: Dossie) -> None:
    f = f"{C04}/attribution/protocol.json"
    pr = c.json(f)
    d.put("shap.amostra_por_classe", pr["sample_per_class_per_fold"], arquivo=f,
          filtro="sample_per_class_per_fold", braco="primary")
    d.put("shap.sementes", pr["seeds"], arquivo=f, filtro="seeds", braco="primary")
    erros, difs, fam = [], [], []
    for fold in range(5):
        m = c.json(f"{C04}/attribution/fold{fold}/manifest.json")
        fam.append(m["model"])
        difs.append(m["prediction_max_abs_difference"])
        for seed in (42, 101, 202):
            erros.append(c.json(f"{C04}/attribution/fold{fold}/seed{seed}_checks.json")["max_additivity_error"])
    sel = d["principal.selecao.familias"]
    if fam != sel:
        raise ValueError("famílias explicadas divergem dos vencedores internos por dobra")
    base = f"{C04}/attribution/fold*/"
    d.put("shap.aditividade.max", max(erros), arquivo=base + "seed*_checks.json",
          filtro="max_additivity_error; máximo entre 15 amostras", braco="primary")
    d.put("shap.reajuste.dif_max", max(difs), arquivo=base + "manifest.json",
          filtro="prediction_max_abs_difference; máximo entre dobras", braco="primary")

    est = pd.concat([c.csv(f"{C04}/attribution/fold{k}/sampling_stability.csv").assign(fold=k)
                     for k in range(5)])
    arq = base + "sampling_stability.csv"
    d.put("shap.estabilidade.spearman.min", est.spearman.min(), arquivo=arq,
          filtro="spearman; mínimo entre dobras, classes e pares de sementes", braco="primary")
    d.put("shap.estabilidade.spearman.max", est.spearman.max(), arquivo=arq,
          filtro="spearman; máximo", braco="primary")
    d.put("shap.estabilidade.top15.min", est.top15_overlap.min(), arquivo=arq,
          filtro="top15_overlap; mínimo", braco="primary")

    g = shap_por_dobra(c)
    arq = base + "seed*_summary.csv"
    kw = dict(arquivo=arq, braco="primary", regra="margem bruta, anterior à calibração",
              agregacao="por dobra e família; média entre três sementes; nunca somado entre famílias")
    for cls in CLASSES.values():
        x = g[g.classe == cls]
        top1 = x[x["rank"] == 1].sort_values("outer_fold")["feature"].tolist()
        d.put(f"shap.{cls}.primeira_por_dobra", top1, filtro=f"classe={cls}; posto 1 em cada dobra", **kw)
        top15 = x[x["rank"] <= 15]
        n_mun = top15[top15.scope == "municipal"].groupby("outer_fold").size().reindex(range(5), fill_value=0)
        d.put(f"shap.{cls}.municipais_top15.min", int(n_mun.min()),
              filtro=f"classe={cls}; variáveis municipais entre os 15 maiores postos; mínimo entre dobras", **kw)
        d.put(f"shap.{cls}.municipais_top15.max", int(n_mun.max()),
              filtro=f"classe={cls}; idem; máximo entre dobras", **kw)
        part = x[x.scope == "municipal"].groupby("outer_fold")["share"].sum()
        d.put(f"shap.{cls}.fracao_municipal.min.pct", 100 * part.min(),
              filtro=f"classe={cls}; soma de mean|SHAP| municipal / total DENTRO de cada dobra; mínimo", **kw)
        d.put(f"shap.{cls}.fracao_municipal.max.pct", 100 * part.max(),
              filtro=f"classe={cls}; idem; máximo entre dobras", **kw)
        med = x.groupby("feature")["rank"].median().sort_values()
        d.put(f"shap.{cls}.top5_posto_mediano", med.head(5).index.tolist(),
              filtro=f"classe={cls}; cinco menores postos medianos entre dobras", **kw)
        comuns = set.intersection(*[set(top15[top15.outer_fold == k].feature) for k in range(5)])
        d.put(f"shap.{cls}.top15_comuns", len(comuns),
              filtro=f"classe={cls}; variáveis entre os 15 maiores postos em todas as cinco dobras", **kw)


def _custo(c: Catalogo, d: Dossie) -> None:
    f = f"{REL}/compute_cost_all_arms.csv"
    cc = c.csv(f)
    kw = dict(arquivo=f, agregacao="soma de wall_seconds dos trabalhos concluídos; não é tempo transcorrido")
    h = cc.groupby(["arm", "compute"])["wall_seconds"].sum() / 3600
    for (braco, hw), v in h.items():
        d.put(f"custo.{braco}.{hw}.horas", v, filtro=f"arm={braco}, compute={hw}", braco=braco, **kw)
    d.put("custo.primary.horas", h["primary"].sum(), filtro="arm=primary", braco="primary", **kw)
    d.put("custo.total.horas", h.sum(), filtro="todos os braços", **kw)
    d.put("custo.total.trabalhos", int(len(cc)), filtro="linhas", **kw)
    d.put("custo.primary.trabalhos", int((cc.arm == "primary").sum()), filtro="arm=primary; linhas",
          braco="primary", **kw)
    p = cc[cc.arm == "primary"].assign(model=lambda x: x.comparison.str.split("__").str[0])
    pm = p.groupby("model")["wall_seconds"].sum().sort_values(ascending=False) / 3600
    d.put("custo.primary.maior.modelo", pm.index[0], filtro="arm=primary; modelo de maior soma",
          braco="primary", **kw)
    d.put("custo.primary.maior.pct", 100 * pm.iloc[0] / pm.sum(), filtro="arm=primary; maior soma / total",
          braco="primary", **kw)
    gb = cc[cc.execution_amendment.notna()]
    d.put("custo.gb_emenda.trabalhos", int(len(gb)), filtro="execution_amendment não nulo; linhas", **kw)
    d.put("custo.gb_emenda.pares", sorted(gb.comparison.unique()),
          filtro="execution_amendment não nulo; comparison", **kw)
    f2 = "data/reanalysis/c6_remaining_20260923/singleton_recovery_20261001/amendment.json"
    am = c.json(f2)
    d.put("custo.lote_unitario.reexecutados", am["jobs"], arquivo=f2, filtro="jobs")
    d.put("custo.lote_unitario.com_juncao", am["singleton_patch_jobs"], arquivo=f2,
          filtro="singleton_patch_jobs")
    rec = cc[cc.recovery_amendment.notna()]
    if len(rec) != len(am["jobs"]) or set(rec.arm) != {"random_group"}:
        raise ValueError("trabalhos recuperados divergem entre custo e emenda")


R11 = "data/reanalysis/c8_20261005/reporting_supplement"


def _r11(c: Catalogo, d: Dossie) -> None:
    """Complementos descritivos da revisão C8: matrizes 3x3 e características basais."""
    f = f"{R11}/checks.json"
    for r in c.json(f):
        regra = r["rule"].replace("pred_", "")
        if not r["source_ids_and_targets_match"]:
            raise ValueError("R11: IDs ou desfechos não coincidem com as predições selecionadas")
        if abs(r["confusion_macro_f1"] - d[f"{r['arm']}.{regra}.f1.pooled"]) > 1e-9:
            raise ValueError(f"R11: macro-F1 da matriz diverge do liberado ({r['arm']}, {regra})")
    f = f"{R11}/selected_confusions.csv"
    cf = c.csv(f)
    for braco in ("primary", "temporal"):
        for regra in REGRAS:
            g = cf[(cf.arm == braco) & (cf.rule == f"pred_{regra}")]
            if g.n.sum() != d[f"{braco}.n"]:
                raise ValueError("R11: matriz não soma a população do braço")
            for k, real in CLASSES.items():
                for j, pred in CLASSES.items():
                    r = _um(g, actual=k, predicted=j)
                    kw = dict(arquivo=f, regra=f"pred_{regra}", braco=braco,
                              agregacao="pooled_records; sem intervalo",
                              filtro=f"arm={braco}, rule=pred_{regra}, actual={k}, predicted={j}")
                    d.put(f"matriz.{braco}.{regra}.{real}.como_{pred}.n", int(r["n"]), **kw)
                    d.put(f"matriz.{braco}.{regra}.{real}.como_{pred}.pct", r["row_percent"], **kw)
                sens = d[f"matriz.{braco}.{regra}.{real}.como_{real}.pct"] / 100
                if abs(sens - d[f"{braco}.{regra}.{real}.sensitivity"]) > 1e-9:
                    raise ValueError("R11: diagonal da matriz diverge da sensibilidade auditada")
    f = f"{R11}/baseline_numeric.csv"
    bn = c.csv(f)
    for papel, rot in (("development_2015_2023", "dev"), ("temporal_2024", "temporal")):
        for var, nome in (("age_years", "idade"), ("NU_CONTATO", "contatos")):
            r = _um(bn, role=papel, outcome="all", feature=var)
            if r["n_total"] != d[f"coorte.{rot}"]:
                raise ValueError("R11: denominador basal diverge da coorte")
            for col in ("median", "q1", "q3", "missing_percent"):
                d.put(f"basal.{rot}.{nome}.{col}", r[col], arquivo=f, braco=rot,
                      filtro=f"role={papel}, outcome=all, feature={var}, {col}",
                      agregacao="registros do papel; ausentes fora do resumo")
    f = f"{R11}/environment_inventory.json"
    runs = c.json(f)["runs"]
    ok = [r for r in runs if r.get("packages")]
    if len({json.dumps(r["packages"], sort_keys=True) for r in ok}) != 1:
        raise ValueError("R11: versões de pacotes divergem entre execuções verificadas")
    d.put("ambiente.execucoes", len(runs), arquivo=f, filtro="runs")
    d.put("ambiente.execucoes_verificadas", len(ok), arquivo=f, filtro="runs com packages registrados")


def construir(c: Catalogo | None = None) -> tuple[Dossie, Catalogo]:
    c = c or Catalogo()
    d = Dossie()
    _coorte(c, d)
    _grade(c, d)
    _bracos(c, d)
    for braco in ("primary", "temporal", "clinical", "individual", "municipal"):
        _por_classe(c, d, braco)
    _principal(c, d)
    _comparacoes(c, d)
    _curvas_extra(c, d)
    _temporal(c, d)
    _subgrupos(c, d)
    _atribuicao(c, d)
    _custo(c, d)
    _r11(c, d)
    return d, c


def gravar(d: Dossie, c: Catalogo) -> None:
    (SAIDA / "dossie_L5.json").write_text(json.dumps(
        {"run_id": c.run_id, "valores": d.valores, "origem": d.origem,
         "artefatos_consultados": c.proveniencia()}, ensure_ascii=False, indent=1), encoding="utf-8")
    L = ["# Dossiê de números do L5", "",
         f"Pacote editorial: `{c.run_id}` — identifica a liberação, não um treinamento único.",
         f"{len(d.valores)} valores, {len(c.usados)} artefatos do catálogo consultados. "
         "Gerado por `scripts/editorial/dossie_c7.py`; não editar à mão.", "",
         "| Chave | Valor | Arquivo | Coluna/filtro | Regra | Agregação | Braço |",
         "|---|---|---|---|---|---|---|"]
    for k, v in d.valores.items():
        o = d.origem[k]
        vs = f"{v:.6g}" if isinstance(v, float) else str(v)
        L.append(f"| `{k}` | {vs} | `{o['arquivo']}` | {o['filtro'].replace('|', '/')} | {o['regra']} | "
                 f"{o['agregacao']} | {o['braco']} |")
    (SAIDA / "dossie_L5.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    dossie, cat = construir()
    gravar(dossie, cat)
    print(f"dossiê: {len(dossie.valores)} valores de {len(cat.usados)} artefatos")
