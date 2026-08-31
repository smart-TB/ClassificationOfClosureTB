"""CLI do tb_outcomes. Nenhuma lógica científica vive aqui — só orquestração.

Nota: este módulo NÃO usa `from __future__ import annotations`. O Typer resolve
os tipos dos parâmetros por introspecção em tempo de execução, e anotações
adiadas (strings) fazem `--config` deixar de receber valor.
"""
import logging
from pathlib import Path

import pandas as pd
import typer

from tb_outcomes import artifacts as art
from tb_outcomes import cohort as coh
from tb_outcomes import data as dat
from tb_outcomes import provenance as prov
from tb_outcomes.config import TBD, config_hash, load_config, require_frozen
from tb_outcomes.executor import load_executor_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("tb_outcomes")

app = typer.Typer(add_completion=False, help="Reanálise de desfechos de tuberculose.")

DATA_DIR = Path("data")
OUTCOMES = Path("configs/outcomes.yaml")
FEATURES = Path("configs/features.yaml")

# Decisões que bloqueiam a execução científica completa, mas não a coorte.
DECISOES_RUN_FULL = ["study.primary_target_schema", "study.primary_prediction_time"]

# Decisões sem as quais a coorte não pode ser construída.
DECISOES_BUILD_COHORT = [
    "study.minimum_followup_days",
    "study.data_extraction_date",
    "study.cohort_year_min",
    "study.cohort_year_max",
]


@app.command("validate-config")
def validate_config(config: Path = typer.Option(..., "--config")) -> None:
    """Valida a estrutura da config e mostra seu hash."""
    cfg = load_config(config)
    typer.echo(f"config válida. config_hash={config_hash(cfg)}")
    pendentes = [
        campo
        for campo in DECISOES_RUN_FULL
        if getattr(cfg.study, campo.split(".", 1)[1]) == TBD
    ]
    if pendentes:
        typer.echo(f"decisões ainda não congeladas (bloqueiam run-full): {', '.join(pendentes)}")


@app.command("harmonize")
def harmonize_cmd(
    config: Path = typer.Option(..., "--config"),
    rebuild: bool = typer.Option(False, "--rebuild", help="reconstrói mesmo com cache válido"),
) -> None:
    """Materializa a coorte harmonizada em data/harmonized.parquet.

    Os demais comandos reusam esse parquet enquanto o bruto e a config de
    geografia não mudarem — evita re-harmonizar 1,3 M de registros a cada um.
    """
    cfg = load_config(config)
    df, construido = dat.load_or_build_harmonized(DATA_DIR, cfg, rebuild=rebuild)
    estado = "materializada" if construido else "já em cache (nada a fazer)"
    typer.echo(f"coorte harmonizada {estado}: {len(df)} linhas, {df.shape[1]} colunas")


@app.command("audit-data")
def audit_data(config: Path = typer.Option(..., "--config")) -> None:
    """Relatório de qualidade sobre a harmonização."""
    cfg = load_config(config)
    df, _ = dat.load_or_build_harmonized(DATA_DIR, cfg)
    relatorio = dat.quality_report(df)
    art.write_json(relatorio, DATA_DIR / "data_quality_report.json")
    art.atomic_write_text(
        DATA_DIR / "data_quality_report.md", dat.quality_report_markdown(relatorio)
    )
    typer.echo("relatório escrito em data/data_quality_report.json e .md")


@app.command("build-cohort")
def build_cohort_cmd(config: Path = typer.Option(..., "--config")) -> None:
    """Constrói a coorte para TODOS os esquemas de alvo e grava os artefatos.

    Não exige primary_target_schema: gera os três. A escolha do primário é
    decisão de protocolo e só bloqueia run-full.
    """
    cfg = load_config(config)
    require_frozen(cfg, DECISOES_BUILD_COHORT)

    art.write_csv(dat.raw_manifest(DATA_DIR), DATA_DIR / "raw_manifest.csv")

    df, _ = dat.load_or_build_harmonized(DATA_DIR, cfg)

    run_id = prov.make_run_id(
        "revised_primary", config_hash(cfg), art.sha256_frame_hash(df)[:12]
    )
    saida = Path("artifacts") / run_id
    manifesto = art.RunManifest(run_id=run_id)

    art.write_json(prov.git_state(), saida / "git_state.json")
    art.write_json(prov.environment(), saida / "environment.json")
    art.write_json(prov.hardware(), saida / "hardware.json")
    art.write_json(cfg.model_dump(mode="json"), saida / "config_resolved.yaml")
    art.write_json({"python": cfg.seeds.python, "numpy": cfg.seeds.numpy}, saida / "seeds.json")
    for nome in (
        "git_state.json",
        "environment.json",
        "hardware.json",
        "config_resolved.yaml",
        "seeds.json",
    ):
        manifesto.record(saida / nome, step="provenance")

    relatorio = dat.quality_report(df)
    art.write_json(relatorio, DATA_DIR / "data_quality_report.json")
    art.atomic_write_text(
        DATA_DIR / "data_quality_report.md", dat.quality_report_markdown(relatorio)
    )
    art.write_csv(
        dat.municipality_audit(df, cfg.geography.municipality_key),
        DATA_DIR / "auditoria_municipio_analise.csv",
    )
    colunas_sinan = [c for c in dat.sinan_column_names(DATA_DIR) if c in df.columns]
    art.write_csv(coh.audit_duplicates(df, colunas_sinan), DATA_DIR / "duplicate_audit.csv")
    for nome in (
        "data_quality_report.json",
        "data_quality_report.md",
        "auditoria_municipio_analise.csv",
        "duplicate_audit.csv",
        "raw_manifest.csv",
    ):
        manifesto.record(DATA_DIR / nome, step="audit-data")

    regras = coh.load_outcome_rules(OUTCOMES)
    for schema in coh.TargetSchema.all():
        r = coh.build_cohort(df, cfg, regras, schema)
        art.write_csv(r.flow, DATA_DIR / f"cohort_flow_{schema}.csv")
        art.write_json(r.flow.to_dict(orient="records"), DATA_DIR / f"cohort_flow_{schema}.json")
        art.write_csv(r.outcome_distribution, DATA_DIR / f"outcome_distribution_{schema}.csv")
        art.write_csv(r.exclusions, DATA_DIR / f"exclusion_reasons_{schema}.csv")
        for nome in (
            f"cohort_flow_{schema}.csv",
            f"cohort_flow_{schema}.json",
            f"outcome_distribution_{schema}.csv",
            f"exclusion_reasons_{schema}.csv",
        ):
            manifesto.record(DATA_DIR / nome, step="build-cohort")

    art.write_csv(coh.followup_audit(df, cfg), DATA_DIR / "followup_audit.csv")
    manifesto.record(DATA_DIR / "followup_audit.csv", step="build-cohort")

    manifesto.save(saida / "run_manifest.json")
    typer.echo(f"coorte construída. run_id={run_id}")



@app.command("build-features")
def build_features_cmd(config: Path = typer.Option(..., "--config")) -> None:
    """Monta o conjunto de features da notificação e as auditorias."""
    from tb_outcomes import features as feat

    cfg = load_config(config)
    require_frozen(cfg, DECISOES_BUILD_COHORT)

    df, _ = dat.load_or_build_harmonized(DATA_DIR, cfg)

    specs = feat.load_feature_config(FEATURES)
    feat.assert_covers_all_columns(specs, list(df.columns))
    feat.assert_no_outcome_in_derivation(specs)

    run_id = prov.make_run_id("revised_primary", config_hash(cfg), art.sha256_frame_hash(df)[:12])
    saida = Path("artifacts") / run_id
    manifesto = art.RunManifest(run_id=run_id)

    art.write_csv(feat.feature_availability_table(specs), DATA_DIR / "feature_availability.csv")
    art.write_csv(feat.feature_dictionary_table(specs), DATA_DIR / "feature_dictionary.csv")
    art.write_csv(feat.hiv_rule_audit(df), DATA_DIR / "hiv_rule_audit.csv")
    art.write_csv(
        feat.outcome_contamination_audit(df, feat.COLUNAS_CONTAMINADAS),
        DATA_DIR / "outcome_contamination_audit.csv",
    )
    art.write_csv(feat.closure_timing_audit(df), DATA_DIR / "closure_timing_audit.csv")
    art.write_csv(
        feat.hiv_availability_by_duration(df), DATA_DIR / "hiv_availability_by_duration.csv"
    )

    X = feat.build_notification_feature_set(df, specs)
    art.write_csv(feat.missingness_report(X, by=None), DATA_DIR / "missingness_overall.csv")

    com_desfecho = X.copy()
    com_desfecho["SITUA_ENCE"] = df["SITUA_ENCE"]
    art.write_csv(
        feat.missingness_report(com_desfecho, by=["SITUA_ENCE"]),
        DATA_DIR / "missingness_by_outcome.csv",
    )
    art.write_csv(
        feat.early_death_missingness_audit(com_desfecho, "SITUA_ENCE"),
        DATA_DIR / "early_death_missingness.csv",
    )

    art.write_csv(feat.correlation_report(X), DATA_DIR / "collinearity.csv")
    art.write_csv(feat.vif_report(X), DATA_DIR / "vif.csv")

    for nome in (
        "feature_availability.csv",
        "feature_dictionary.csv",
        "hiv_rule_audit.csv",
        "outcome_contamination_audit.csv",
        "closure_timing_audit.csv",
        "hiv_availability_by_duration.csv",
        "missingness_overall.csv",
        "missingness_by_outcome.csv",
        "early_death_missingness.csv",
        "collinearity.csv",
        "vif.csv",
    ):
        manifesto.record(DATA_DIR / nome, step="build-features")

    manifesto.save(saida / "run_manifest.json")
    typer.echo(f"features construídas: {X.shape[1]} colunas. run_id={run_id}")


SPLITS = Path("configs/splits.yaml")


@app.command("build-splits")
def build_splits_cmd(
    config: Path = typer.Option(..., "--config"),
    k: int = typer.Option(None, "--k", help="sobrepõe k_primary (para a fixture)"),
) -> None:
    """Constrói clusters espaciais e dobras aninhadas; grava as auditorias."""
    from tb_outcomes import splits as sp

    cfg = load_config(config)
    require_frozen(cfg, DECISOES_BUILD_COHORT)
    scfg = sp.load_splits_config(SPLITS)

    df, _ = dat.load_or_build_harmonized(DATA_DIR, cfg)

    # coorte 3-classes para definir as dobras (o esquema com mais registros)
    regras = coh.load_outcome_rules(OUTCOMES)
    r = coh.build_cohort(df, cfg, regras, coh.TargetSchema.LEGACY_3CLASS)
    cohort = r.analytic
    y = cohort["target_legacy_3class"].map(
        {"cure": 1, "treatment_interruption": 2, "tb_attributed_death": 3}
    )

    # O clustering usa a coluna de município JÁ RESOLVIDA (ID_MUNIC_ANALISE),
    # não a chave de residência do config. São conceitos distintos: a chave do
    # config governa o enriquecimento no SP1; aqui o município já está resolvido.
    mun = sp.build_municipality_table(cohort, key=dat.COLUNA_MUNICIPIO_RESOLVIDO)
    seed = cfg.seeds.python
    run_id = prov.make_run_id(
        "revised_primary", config_hash(cfg), art.sha256_frame_hash(cohort)[:12]
    )
    saida = Path("artifacts") / run_id
    manifesto = art.RunManifest(run_id=run_id)

    ks = [k] if k else [scfg.k_primary, *scfg.k_sensitivity]
    k_principal = k or scfg.k_primary
    checks: dict[str, bool] = {}
    for kk in ks:
        clusters = sp.fit_spatial_clusters(mun, kk, scfg.epsg_metric, seed)
        art.write_parquet(
            clusters.reset_index(), DATA_DIR / f"municipality_clusters_k{kk}.parquet"
        )
        manifesto.record(DATA_DIR / f"municipality_clusters_k{kk}.parquet", step="build-splits")

        rc = sp.assign_clusters_to_records(cohort, clusters, key=dat.COLUNA_MUNICIPIO_RESOLVIDO)
        folds = sp.make_outer_folds(cohort, rc, y, scfg.n_outer_folds, scfg.fold_assignment)

        vaz = pd.DataFrame({"cluster": rc.to_numpy(), "fold": folds.to_numpy()})
        checks[f"k{kk}_no_cluster_leak"] = bool((vaz.groupby("cluster").fold.nunique() == 1).all())
        checks[f"k{kk}_all_classes_per_fold"] = bool(
            all(y[folds == f].nunique() == 3 for f in folds.dropna().unique())
        )
        # O teto de razão é garantia de ESCALA: com bem mais clusters que dobras,
        # o bin-packing equilibra (~1,04x). Com poucos clusters (fixture) o piso é
        # o maior cluster e a razão degenera — não é defeito, é falta de unidades.
        # Só aplicamos o teto quando há clusters suficientes para equilibrar.
        n_clusters = int(rc.nunique())
        if n_clusters >= 2 * scfg.n_outer_folds:
            tam = folds.value_counts()
            checks[f"k{kk}_fold_ratio_ok"] = bool(
                tam.max() / tam.min() <= scfg.max_fold_size_ratio
            )

        if kk == k_principal:
            art.write_parquet(
                pd.DataFrame({"cluster": rc.to_numpy(), "outer_fold": folds.to_numpy()}),
                DATA_DIR / "outer_folds.parquet",
            )
            art.write_csv(sp.fold_summary(folds, rc, y), DATA_DIR / "fold_summary.csv")
            art.write_csv(sp.cluster_summary(rc, y, clusters), DATA_DIR / "cluster_summary.csv")
            sp.clusters_to_geojson(
                mun, clusters, DATA_DIR / f"municipality_clusters_k{kk}.geojson"
            )
            for nome in (
                "outer_folds.parquet",
                "fold_summary.csv",
                "cluster_summary.csv",
                f"municipality_clusters_k{kk}.geojson",
            ):
                manifesto.record(DATA_DIR / nome, step="build-splits")

    art.write_json(checks, DATA_DIR / "leakage_checks.json")
    manifesto.record(DATA_DIR / "leakage_checks.json", step="build-splits")
    manifesto.save(saida / "run_manifest.json")

    if not all(checks.values()):
        raise typer.Exit(code=1)
    typer.echo(f"dobras construídas para k={ks}. run_id={run_id}")


PREPROCESS = Path("configs/preprocess.yaml")


@app.command("preprocess-manifest")
def preprocess_manifest_cmd(config: Path = typer.Option(..., "--config")) -> None:
    """Gera o manifesto de pré-processamento (tipos, perfis, dimensões)."""
    from tb_outcomes import features as feat
    from tb_outcomes import preprocess as pp

    cfg = load_config(config)
    pcfg = pp.load_preprocess_config(PREPROCESS)
    df, _ = dat.load_or_build_harmonized(DATA_DIR, cfg)
    specs = feat.load_feature_config(FEATURES)
    X = feat.build_notification_feature_set(df, specs)

    types = pp.infer_column_types(X, specs, pcfg.max_cardinality)
    manifesto = pp.preprocessing_manifest(types, pcfg)
    art.write_json(manifesto, DATA_DIR / "preprocessing_manifest.json")
    typer.echo(
        f"manifesto: {manifesto['n_categorical']} categóricas, "
        f"{manifesto['n_numeric']} numéricas, {len(manifesto['profiles'])} perfis"
    )


IMBALANCE = Path("configs/imbalance.yaml")


@app.command("imbalance-audit")
def imbalance_audit_cmd(config: Path = typer.Option(..., "--config")) -> None:
    """Aplica as quatro estratégias à coorte e grava a auditoria (§13.1, §13.3)."""
    from tb_outcomes import imbalance as imb

    cfg = load_config(config)
    icfg = imb.load_imbalance_config(IMBALANCE)
    df, _ = dat.load_or_build_harmonized(DATA_DIR, cfg)

    regras = coh.load_outcome_rules(OUTCOMES)
    r = coh.build_cohort(df, cfg, regras, coh.TargetSchema.LEGACY_3CLASS)
    cohort = r.analytic
    y = cohort["target_legacy_3class"]
    strata = imb.build_strata(cohort, icfg.cost_stratum)

    results = {
        st: imb.apply_strategy(y, strata, st, icfg, cfg.seeds.python, supports_weight=True)
        for st in icfg.strategies
    }
    imb.assert_strategies_differ(results)  # prova de não-identidade
    art.write_json(imb.imbalance_audit(y, results), DATA_DIR / "imbalance_audit.json")
    typer.echo(
        f"auditoria de desbalanceamento: {len(icfg.strategies)} estratégias, distintas"
    )


MODELS = Path("configs/models.yaml")


@app.command("capabilities-matrix")
def capabilities_matrix_cmd() -> None:
    """Gera model_capabilities.csv: a matriz MEDIDA (§14.3, §2.2)."""
    from tb_outcomes import models as mdl

    cfg = mdl.load_models_config(MODELS)
    registry = mdl.build_registry(cfg)
    matrix = mdl.capabilities_matrix(registry)
    art.write_csv(matrix, DATA_DIR / "model_capabilities.csv")
    n_cal = int(matrix["calibrator_required"].sum())
    n_nocost = int((matrix["cost_sensitive_status"] == "not_run_incompatible").sum())
    typer.echo(
        f"matriz: {len(matrix)} modelos, {n_cal} exigem calibração (sem proba nativa), "
        f"{n_nocost} sem cost-sensitive (sem sample_weight)"
    )


EXECUTOR = Path("configs/executor.yaml")
CALIBRATION = Path("configs/calibration.yaml")
ANALYSIS = Path("configs/analysis_decisions.yaml")


def _load_benchmark_inputs(exec_cfg, sanity: bool, k_override=None, feature_scope=None,
                           with_municipio: bool = False, with_year: bool = False):
    """Remonta X/y/clusters/estrato/dobras da coorte real (SP4e).

    X e y não são persistidos: X vem de build_notification_feature_set(harmonized, specs)
    e y da coorte analítica (desfecho como coluna). O estrato de custo (year_region) e as
    variáveis de agrupamento (cluster) vêm POR FORA do X — a regra de vazamento (§1.1/§7)
    proíbe NU_ANO/ID_MN_RESI dentro das features.
    """
    from tb_outcomes import features as feat
    from tb_outcomes import leakage
    from tb_outcomes.imbalance import build_strata, load_imbalance_config
    from tb_outcomes.splits import assign_clusters_to_records, make_outer_folds

    cfg = load_config(ANALYSIS)
    df, _ = dat.load_or_build_harmonized(DATA_DIR, cfg)
    specs = feat.load_feature_config(FEATURES)
    rules = coh.load_outcome_rules(OUTCOMES)
    result = coh.build_cohort(df, cfg, rules, exec_cfg.outcome_schema)
    analytic = result.analytic
    if sanity and len(analytic) > exec_cfg.sanity_subsample:
        analytic = analytic.sample(exec_cfg.sanity_subsample, random_state=exec_cfg.random_state)

    X = feat.build_notification_feature_set(df, specs).loc[analytic.index].reset_index(drop=True)
    # ablação territorial (§3.3): o recorte vem DEPOIS da montagem, para os três braços
    # partirem exatamente do mesmo X — nenhuma derivação muda de valor por o braço ser
    # menor. 'combined' devolve o X inteiro (é o braço da rodada de produção).
    if feature_scope is not None:
        n_total = X.shape[1]
        X = feat.select_feature_scope(X, specs, feature_scope)
        logger.info("Braço da ablação '%s': %d das %d colunas.",
                    feature_scope, X.shape[1], n_total)
    leakage.assert_strata_out_of_features(list(X.columns))

    labels = analytic[f"target_{exec_cfg.outcome_schema}"]
    classes = sorted(labels.unique())
    code = {c: i for i, c in enumerate(classes)}
    y = labels.map(code).astype(int).reset_index(drop=True)

    # o parquet guarda ID_MUNIC_ANALISE como COLUNA (índice RangeIndex); assign_clusters
    # espera clusters indexado pelo município para o .map casar por código.
    # k_override (SP4g/run-sweep) remonta os inputs em outro k sem alterar exec_cfg.k.
    k = k_override if k_override is not None else exec_cfg.k
    clusters_df = pd.read_parquet(
        DATA_DIR / f"municipality_clusters_k{k}.parquet"
    ).set_index("ID_MUNIC_ANALISE")
    record_clusters = assign_clusters_to_records(analytic, clusters_df).reset_index(drop=True)
    strata = build_strata(analytic, load_imbalance_config(IMBALANCE).cost_stratum).reset_index(drop=True)

    # Município e ano POR FORA do X (ambos proibidos como feature pela regra de vazamento
    # §1.1/§7), mas necessários para estratificar por unidade notificadora (§3.3) e para o
    # corte do holdout temporal (§4.1).
    municipio = analytic["ID_MUNIC_ANALISE"].reset_index(drop=True)
    ano = pd.to_numeric(analytic["NU_ANO"], errors="coerce").reset_index(drop=True)

    keep = record_clusters.notna()
    if not keep.any():
        raise ValueError(
            "nenhum registro casou com um cluster — chave de município incompatível entre "
            "a coorte e municipality_clusters_k*.parquet"
        )
    if not keep.all():
        logger.warning("%d registros sem cluster removidos do benchmark", int((~keep).sum()))
        X, y = X[keep].reset_index(drop=True), y[keep].reset_index(drop=True)
        record_clusters = record_clusters[keep].reset_index(drop=True)
        strata = strata[keep].reset_index(drop=True)
        municipio = municipio[keep].reset_index(drop=True)
        ano = ano[keep].reset_index(drop=True)

    outer = make_outer_folds(X, record_clusters, y, exec_cfg.n_outer_folds, exec_cfg.fold_method)
    leakage.assert_outcome_count_coherence(y, len(y))
    base = (X, y, strata, record_clusters, outer, list(range(len(classes))), specs, classes)
    # extras na ordem fixa: município, depois ano
    extras = ([municipio] if with_municipio else []) + ([ano] if with_year else [])
    return (*base, *extras) if extras else base


@app.command("run-benchmark")
def run_benchmark_cmd(
    sanity: bool = typer.Option(False, "--sanity", help="rodada de sanidade em subamostra"),
) -> None:
    """Executor aninhado (SP4e): benchmark de defaults, Opção A."""
    from tb_outcomes.calibration import load_calibration_config
    from tb_outcomes.executor import make_sklearn_fold_factory, run_benchmark
    from tb_outcomes.executor_io import write_benchmark_artifacts
    from tb_outcomes.imbalance import load_imbalance_config
    from tb_outcomes.models import build_registry, load_models_config
    from tb_outcomes.preprocess import load_preprocess_config

    exec_cfg = load_executor_config(EXECUTOR)
    X, y, strata, clusters, outer, classes, specs, labels = _load_benchmark_inputs(exec_cfg, sanity)
    registry = build_registry(load_models_config(MODELS))
    factory = make_sklearn_fold_factory(
        X, y, strata, registry, load_imbalance_config(IMBALANCE),
        load_preprocess_config(PREPROCESS), specs, exec_cfg.random_state)
    result = run_benchmark(
        exec_cfg.models, exec_cfg.strategies, y, clusters, outer, len(classes), classes,
        factory, lambda m: registry[m].capabilities,
        load_calibration_config(CALIBRATION), exec_cfg.n_inner_folds)
    paths = write_benchmark_artifacts(result, DATA_DIR)
    n_ok = sum(1 for r in result.status_rows if r["status"] == "ok")
    typer.echo(f"Benchmark: {n_ok}/{len(result.status_rows)} pares ok. Rótulos {labels}. {paths}")


@app.command("run-sweep")
def run_sweep_cmd() -> None:
    """Sweep de granularidade espacial (SP4g): roda o conjunto contendor em k∈sweep.ks,
    com checkpoint por par, gravando em data/sweep/k{K}/. Não toca no k=50."""
    from tb_outcomes.calibration import load_calibration_config
    from tb_outcomes.executor import make_sklearn_fold_factory, run_benchmark
    from tb_outcomes.executor_io import write_benchmark_artifacts
    from tb_outcomes.imbalance import load_imbalance_config
    from tb_outcomes.models import build_registry, load_models_config
    from tb_outcomes.preprocess import load_preprocess_config

    exec_cfg = load_executor_config(EXECUTOR)
    if exec_cfg.sweep is None:
        raise typer.BadParameter("configs/executor.yaml não tem a seção 'sweep'")
    sweep = exec_cfg.sweep
    registry = build_registry(load_models_config(MODELS))

    for k in sweep.ks:
        out_k = Path(sweep.out_root) / f"k{k}"
        ckpt = out_k / "pairs"
        typer.echo(f"[sweep] k={k}: {len(sweep.models)} modelos × {len(exec_cfg.strategies)} "
                   f"estratégias -> {out_k} (checkpoint em {ckpt})")
        X, y, strata, clusters, outer, classes, specs, labels = _load_benchmark_inputs(
            exec_cfg, False, k_override=k)
        factory = make_sklearn_fold_factory(
            X, y, strata, registry, load_imbalance_config(IMBALANCE),
            load_preprocess_config(PREPROCESS), specs, exec_cfg.random_state)
        result = run_benchmark(
            sweep.models, exec_cfg.strategies, y, clusters, outer, len(classes), classes,
            factory, lambda m: registry[m].capabilities,
            load_calibration_config(CALIBRATION), exec_cfg.n_inner_folds,
            checkpoint_dir=ckpt)
        write_benchmark_artifacts(result, out_k)
        n_ok = sum(1 for r in result.status_rows if r["status"] == "ok")
        typer.echo(f"[sweep] k={k}: {n_ok}/{len(result.status_rows)} pares ok -> {out_k}")


@app.command("run-temporal-holdout")
def run_temporal_holdout_cmd() -> None:
    """Holdout temporal (§4.1): treina em tudo ANTES do ano reservado e avalia nele.

    O ano de teste vem de `study.temporal_test_period` (2024). Divergência declarada contra
    a §4.1, que pedia avaliar 2023-2024: 2023 participou de todo o desenvolvimento (sweep,
    seleção de modelo, ablação) e não é holdout limpo — só 2024 foi reservado antes do tuning.
    """
    import numpy as np

    from tb_outcomes.calibration import load_calibration_config
    from tb_outcomes.checkpoint import pair_checkpoint_exists, write_pair_checkpoint
    from tb_outcomes.executor import (
        BenchmarkResult,
        aggregate_pair,
        evaluate_temporal_pair,
        make_sklearn_fold_factory,
    )
    from tb_outcomes.executor_io import write_benchmark_artifacts
    from tb_outcomes.imbalance import load_imbalance_config
    from tb_outcomes.models import build_registry, load_models_config
    from tb_outcomes.preprocess import load_preprocess_config

    exec_cfg = load_executor_config(EXECUTOR)
    if exec_cfg.temporal is None:
        raise typer.BadParameter("configs/executor.yaml não tem a seção 'temporal'")
    temporal = exec_cfg.temporal

    cfg = load_config(ANALYSIS)
    if not cfg.study.temporal_validation_enabled:
        raise typer.BadParameter("study.temporal_validation_enabled é false")
    ano_teste = int(cfg.study.temporal_test_period)

    modelos = temporal.models
    if modelos is None and temporal.models_from_sweep:
        if exec_cfg.sweep is None:
            raise typer.BadParameter("temporal.models_from_sweep=true mas não há seção 'sweep'")
        modelos = exec_cfg.sweep.models
    if not modelos:
        raise typer.BadParameter("a seção 'temporal' não define modelos")

    out = Path(temporal.out_root)
    ckpt = out / "pairs"
    registry = build_registry(load_models_config(MODELS))
    X, y, strata, clusters, outer, classes, specs, labels, ano = _load_benchmark_inputs(
        exec_cfg, False, with_year=True)

    # `ano` é dtype anulável: `(ano < x).to_numpy()` devolveria dtype object, e `~` sobre
    # object faz complemento de dois (True -> -2), não negação. Um ano ausente não pode ser
    # colocado em nenhum dos lados por omissão — falha alto em vez de escolher por acaso.
    if ano.isna().any():
        raise typer.BadParameter(
            f"{int(ano.isna().sum())} registros sem NU_ANO: impossível particionar no tempo")
    treino = (ano < ano_teste).to_numpy(dtype=bool)
    n_tr, n_te = int(treino.sum()), int((~treino).sum())
    if n_te == 0:
        raise typer.BadParameter(
            f"nenhum registro em {ano_teste}; anos presentes: "
            f"{sorted(ano.dropna().unique().tolist())}")
    typer.echo(f"[temporal] treino = anos < {ano_teste} ({n_tr:,} registros) | "
               f"teste = {ano_teste} ({n_te:,}) | {len(modelos)} modelos × "
               f"{len(exec_cfg.strategies)} estratégias -> {out}")
    for c in sorted(pd.Series(y[~treino]).unique()):
        typer.echo(f"           classe {labels[c]}: {int((y[~treino] == c).sum()):,} no teste")

    factory = make_sklearn_fold_factory(
        X, y, strata, registry, load_imbalance_config(IMBALANCE),
        load_preprocess_config(PREPROCESS), specs, exec_cfg.random_state)
    cal_cfg = load_calibration_config(CALIBRATION)

    result = BenchmarkResult()
    total, feito = len(modelos) * len(exec_cfg.strategies), 0
    for model in modelos:
        caps = registry[model].capabilities
        for strategy in exec_cfg.strategies:
            feito += 1
            if pair_checkpoint_exists(ckpt, model, strategy):
                from tb_outcomes.checkpoint import load_pair_checkpoint
                ck = load_pair_checkpoint(ckpt, model, strategy)
                for campo in ("oof_rows", "metric_rows", "manifest_rows", "aggregate_rows"):
                    getattr(result, campo).extend(ck[campo])
                result.status_rows.append(ck["status_row"])
                logger.info("[temporal] %d/%d %s × %s -> checkpoint (pulado: %s)",
                            feito, total, model, strategy, ck["status_row"]["status"])
                continue
            if strategy == "local_cost_sensitive" and not getattr(caps, "sample_weight", False):
                linha = {"model": model, "strategy": strategy, "status": "not_run_incompatible"}
                result.status_rows.append(linha)
                write_pair_checkpoint(ckpt, model, strategy, [], [], [], [], linha)
                continue
            logger.info("[temporal] %d/%d início %s × %s", feito, total, model, strategy)
            try:
                par = evaluate_temporal_pair(
                    model, strategy, y, clusters, treino, len(classes), classes,
                    factory(model, strategy), caps, cal_cfg, n_inner=exec_cfg.n_inner_folds,
                    fold_id=f"holdout_{ano_teste}")
            except Exception as exc:  # noqa: BLE001 — um par que quebra não derruba o resto
                import traceback
                logger.error("[temporal] %s × %s ERRO:\n%s", model, strategy,
                             traceback.format_exc())
                linha = {"model": model, "strategy": strategy, "status": "error",
                         "error": str(exc)[:200]}
                result.status_rows.append(linha)
                write_pair_checkpoint(ckpt, model, strategy, [], [], [], [], linha)
                continue
            agg = aggregate_pair(par)
            linha = {"model": model, "strategy": strategy, "status": par.status}
            result.oof_rows.extend(par.oof_rows)
            result.metric_rows.extend(par.metric_rows)
            result.manifest_rows.extend(par.manifest_rows)
            result.aggregate_rows.extend(agg)
            result.status_rows.append(linha)
            write_pair_checkpoint(ckpt, model, strategy, par.oof_rows, par.metric_rows,
                                  par.manifest_rows, agg, linha)
            logger.info("[temporal] %d/%d fim %s × %s: %s", feito, total, model, strategy,
                        par.status)

    write_benchmark_artifacts(result, out)
    n_ok = sum(1 for r in result.status_rows if r["status"] == "ok")
    typer.echo(f"[temporal] {n_ok}/{len(result.status_rows)} pares ok -> {out}")


@app.command("utility-report")
def utility_report_cmd(
    model: str = typer.Option(None, "--model", help="padrão: o 1º do leaderboard k=50"),
    strategy: str = typer.Option(None, "--strategy", help="padrão: a melhor do modelo"),
) -> None:
    """Utilidade programática (§3.8): top-k por capacidade, alertas por 1.000, captura de
    eventos, NNS e curva de decisão — do modelo E do baseline clínico simples.

    Não treina nada: lê os OOF já gravados. O baseline clínico vem de
    data/clinical_baseline/ (rode `run-clinical-baseline` antes)."""
    from tb_outcomes.robustness import leaderboard
    from tb_outcomes.utility import (
        DEFAULT_KS,
        alerts_per_1000,
        decision_curve,
        topk_table,
        useful_range,
    )

    if model is None or strategy is None:
        lb = leaderboard(pd.read_csv(DATA_DIR / "benchmark_metrics.csv")).sort_values(
            "f1_macro", ascending=False).iloc[0]
        model, strategy = model or str(lb["model"]), strategy or str(lb["best_strategy"])

    base_dir = DATA_DIR / "clinical_baseline"
    fontes = {"modelo_completo": (DATA_DIR / "oof_predictions.parquet", model, strategy)}
    if (base_dir / "benchmark_metrics.csv").exists():
        lb_b = leaderboard(pd.read_csv(base_dir / "benchmark_metrics.csv")).sort_values(
            "f1_macro", ascending=False).iloc[0]
        fontes["baseline_clinico"] = (base_dir / "oof_predictions.parquet",
                                      str(lb_b["model"]), str(lb_b["best_strategy"]))
    else:
        logger.warning("sem data/clinical_baseline/: o comparador da §3.8 ficará de fora")

    # os rótulos de classe têm de sair na MESMA ordem do executor (sorted das classes
    # observadas), senão proba_0/1/2 seriam atribuídos à classe errada.
    cfg = load_config(ANALYSIS)
    rules = coh.load_outcome_rules(OUTCOMES)
    nomes = sorted({v for v in rules["schemas"][cfg.study.primary_target_schema].values()
                    if v not in ("excluded", "unresolved_or_censored")})

    topk, curvas, resumo = [], [], []
    for arm, (caminho, m, s) in fontes.items():
        oof = pd.read_parquet(caminho, filters=[("model", "==", m), ("strategy", "==", s)])
        if oof.empty:
            raise typer.BadParameter(f"{caminho} não tem {m} × {s}")
        for j, classe in enumerate(nomes):
            if classe == "cure":
                continue  # a lista de prioridade do programa é de risco, não de cura
            y_bin = (oof["y_true"].to_numpy() == j).astype(int)
            proba = oof[f"proba_{j}"].to_numpy()

            t = topk_table(y_bin, proba, ks=DEFAULT_KS)
            t.insert(0, "classe", classe); t.insert(0, "arm", arm)
            t["model"], t["strategy"] = m, s
            topk.append(t)

            c = decision_curve(y_bin, proba)
            c.insert(0, "classe", classe); c.insert(0, "arm", arm)
            curvas.append(c)

            faixa = useful_range(c)
            resumo.append({
                "arm": arm, "model": m, "strategy": s, "classe": classe,
                "n": int(len(y_bin)), "prevalencia": float(y_bin.mean()),
                "alertas_por_1000_regra_operacional": alerts_per_1000(
                    (oof["pred_policy"].to_numpy() == j).astype(int)),
                **faixa,
            })

    p1 = DATA_DIR / "utility_topk.csv"
    p2 = DATA_DIR / "utility_decision_curve.csv"
    p3 = DATA_DIR / "utility_summary.csv"
    pd.concat(topk, ignore_index=True).to_csv(p1, index=False)
    pd.concat(curvas, ignore_index=True).to_csv(p2, index=False)
    res = pd.DataFrame(resumo)
    res.to_csv(p3, index=False)
    typer.echo(f"[utility] braços {list(fontes)} | {p1}, {p2}, {p3}")
    typer.echo(res.round(4).to_string(index=False))


@app.command("run-clinical-baseline")
def run_clinical_baseline_cmd() -> None:
    """Baseline clínico simples da §3.8: idade + HIV + álcool/drogas + situação de rua,
    sob o MESMO protocolo espacial aninhado do benchmark principal. É o comparador que
    decide se as 72 variáveis valem a pena — um serviço monta essa lista sem ML."""
    from tb_outcomes.calibration import load_calibration_config
    from tb_outcomes.executor import make_sklearn_fold_factory, run_benchmark
    from tb_outcomes.executor_io import write_benchmark_artifacts
    from tb_outcomes.imbalance import load_imbalance_config
    from tb_outcomes.models import build_registry, load_models_config
    from tb_outcomes.preprocess import load_preprocess_config

    exec_cfg = load_executor_config(EXECUTOR)
    modelos = ["logistic_regression"]
    out = DATA_DIR / "clinical_baseline"
    ckpt = out / "pairs"

    registry = build_registry(load_models_config(MODELS))
    X, y, strata, clusters, outer, classes, specs, labels = _load_benchmark_inputs(
        exec_cfg, False, feature_scope="clinical_baseline")
    typer.echo(f"[clinical] {list(X.columns)} | {len(modelos)} modelo(s) × "
               f"{len(exec_cfg.strategies)} estratégias -> {out}")

    factory = make_sklearn_fold_factory(
        X, y, strata, registry, load_imbalance_config(IMBALANCE),
        load_preprocess_config(PREPROCESS), specs, exec_cfg.random_state)
    result = run_benchmark(
        modelos, exec_cfg.strategies, y, clusters, outer, len(classes), classes,
        factory, lambda m: registry[m].capabilities,
        load_calibration_config(CALIBRATION), exec_cfg.n_inner_folds, checkpoint_dir=ckpt)
    write_benchmark_artifacts(result, out)
    n_ok = sum(1 for r in result.status_rows if r["status"] == "ok")
    typer.echo(f"[clinical] {n_ok}/{len(result.status_rows)} pares ok -> {out}")


@app.command("temporal-report")
def temporal_report_cmd() -> None:
    """Consolida o holdout temporal (§4.1): leaderboard espacial × temporal lado a lado.

    A comparação responde ao editor: o modelo escolhido sob validação espacial se sustenta
    num ano futuro que nunca viu? A queda esperada NÃO é zero — o holdout troca 5 dobras de
    treino por um único ajuste e move a distribuição um ano à frente."""
    from tb_outcomes.robustness import leaderboard

    exec_cfg = load_executor_config(EXECUTOR)
    if exec_cfg.temporal is None:
        raise typer.BadParameter("configs/executor.yaml não tem a seção 'temporal'")
    root = Path(exec_cfg.temporal.out_root)
    p_tmp = root / "benchmark_metrics.csv"
    if not p_tmp.exists():
        raise typer.BadParameter(f"não existe {p_tmp}: rode run-temporal-holdout antes")

    cfg = load_config(ANALYSIS)
    ano_teste = int(cfg.study.temporal_test_period)
    modelos = exec_cfg.temporal.models or (
        exec_cfg.sweep.models if exec_cfg.sweep else exec_cfg.models)

    esp = leaderboard(pd.read_csv(DATA_DIR / "benchmark_metrics.csv"), models=modelos)
    tmp = leaderboard(pd.read_csv(p_tmp), models=modelos)
    tab = (esp.rename(columns={"f1_macro": "espacial_k50",
                               "best_strategy": "estrategia_espacial"})
              .merge(tmp.rename(columns={"f1_macro": f"temporal_{ano_teste}",
                                         "best_strategy": "estrategia_temporal"}),
                     on="model", how="outer"))
    tab["delta"] = tab[f"temporal_{ano_teste}"] - tab["espacial_k50"]
    tab["pct_retido"] = 100 * tab[f"temporal_{ano_teste}"] / tab["espacial_k50"]
    tab["mesma_estrategia"] = tab["estrategia_espacial"] == tab["estrategia_temporal"]
    tab = tab.sort_values("espacial_k50", ascending=False).reset_index(drop=True)
    tab["rank_espacial"] = tab["espacial_k50"].rank(ascending=False, method="min").astype("Int64")
    tab["rank_temporal"] = tab[f"temporal_{ano_teste}"].rank(
        ascending=False, method="min").astype("Int64")

    # estabilidade do ranking entre os dois eixos de validação
    comum = tab.dropna(subset=["espacial_k50", f"temporal_{ano_teste}"])
    tau = p = float("nan")
    if len(comum) > 2:
        from scipy.stats import kendalltau
        tau, p = kendalltau(comum["rank_espacial"], comum["rank_temporal"])
    top5_esp = set(comum.nsmallest(5, "rank_espacial")["model"])
    top5_tmp = set(comum.nsmallest(5, "rank_temporal")["model"])

    out = root / "temporal_vs_spatial.csv"
    tab.to_csv(out, index=False)
    typer.echo(f"[temporal-report] {len(comum)} modelos | Kendall τ = {tau:.3f} (p={p:.2g}) "
               f"| overlap top-5 = {len(top5_esp & top5_tmp) / 5:.1f} | {out}")
    typer.echo(tab[["model", "espacial_k50", f"temporal_{ano_teste}", "delta", "pct_retido",
                    "mesma_estrategia"]].round(4).to_string(index=False))


@app.command("ablation-report")
def ablation_report_cmd() -> None:
    """Consolida a ablação territorial (§3.3): leaderboard por braço (individual,
    municipal, combinado) e o contraste de completude de notificação. O braço combinado
    vem da produção k=50 em data/ — não é re-rodado."""
    import numpy as np

    from tb_outcomes.equity import subgroup_metrics
    from tb_outcomes.features import classify_feature_scope
    from tb_outcomes.robustness import leaderboard

    exec_cfg = load_executor_config(EXECUTOR)
    if exec_cfg.ablation is None:
        raise typer.BadParameter("configs/executor.yaml não tem a seção 'ablation'")
    abl = exec_cfg.ablation
    root = Path(abl.out_root)

    # --- 1. leaderboard por braço ------------------------------------------------
    fontes = {"combined": DATA_DIR / "benchmark_metrics.csv"}
    for s in abl.scopes:
        fontes[s] = root / s / "benchmark_metrics.csv"
    faltando = [k for k, p in fontes.items() if not p.exists()]
    if faltando:
        raise typer.BadParameter(f"braço(s) sem benchmark_metrics.csv: {faltando}")

    quadros = []
    for escopo, p in fontes.items():
        lb = leaderboard(pd.read_csv(p), models=abl.models)
        lb["braco"] = escopo
        quadros.append(lb)
    tab = pd.concat(quadros, ignore_index=True)
    wide = tab.pivot(index="model", columns="braco", values="f1_macro")
    for s in abl.scopes:
        if s in wide:
            wide[f"delta_{s}_vs_combined"] = wide[s] - wide["combined"]
            wide[f"pct_do_combined_{s}"] = wide[s] / wide["combined"]
    wide = wide.sort_values("combined", ascending=False).reset_index()

    # --- 2. completude de notificação × desempenho -------------------------------
    # Operacionalização declarada: completude do município = proporção média de campos
    # INDIVIDUAIS (SINAN) preenchidos nas notificações daquele município. Alta/baixa pela
    # mediana ponderada por paciente. É o contraste que separa "determinante real" de
    # "proxy da qualidade do registro": se o desempenho despenca só onde a notificação é
    # ruim, o sinal municipal é do registro, não do território.
    lb50 = leaderboard(pd.read_csv(DATA_DIR / "benchmark_metrics.csv")).sort_values(
        "f1_macro", ascending=False).iloc[0]
    model, strategy = str(lb50["model"]), str(lb50["best_strategy"])

    X, y, strata, clusters, outer, classes, specs, labels, municipio = _load_benchmark_inputs(
        exec_cfg, False, with_municipio=True)
    escopo_col = classify_feature_scope(list(X.columns), specs)
    cols_ind = [c for c, e in escopo_col.items() if e == "individual"]
    preench = X[cols_ind].notna().mean(axis=1)

    # a completude é atributo do serviço que notifica, logo MUNICIPAL — agrupar por
    # cluster (~centenas de municípios juntos) dissolveria justamente este contraste.
    completude = preench.groupby(municipio.to_numpy()).transform("mean")
    corte = float(np.median(completude))
    faixa = np.where(completude >= corte, "alta_completude", "baixa_completude")

    oof = pd.read_parquet(
        DATA_DIR / "oof_predictions.parquet",
        filters=[("model", "==", model), ("strategy", "==", strategy)],
    ).reset_index(drop=True)
    sub = pd.DataFrame({"completude": faixa}).iloc[oof["record_pos"].to_numpy()].reset_index(
        drop=True)

    linhas = []
    for j, c in enumerate(classes):
        d = pd.DataFrame({
            "y": (oof["y_true"].to_numpy() == c).astype(int),
            "pred": (oof["pred_policy"].to_numpy() == c).astype(int),
            "proba": oof[f"proba_{c}"].to_numpy(),
            "completude": sub["completude"],
        })
        linhas.append(subgroup_metrics(d, axis="completude_notificacao",
                                       group_col="completude", y_col="y", pred_col="pred",
                                       proba_col="proba",
                                       extra={"classe": labels[j], "regra": "pred_policy",
                                              "model": model, "strategy": strategy,
                                              "corte_completude": round(corte, 4)}))
    comp = pd.concat(linhas, ignore_index=True)

    p1 = root / "ablation_leaderboard.csv"
    p2 = root / "ablation_completeness.csv"
    root.mkdir(parents=True, exist_ok=True)
    wide.to_csv(p1, index=False)
    comp.to_csv(p2, index=False)
    typer.echo(f"[ablation-report] braços {list(fontes)} | corte de completude {corte:.4f} "
               f"| {p1}, {p2}")


@app.command("shap-report")
def shap_report_cmd(
    model: str = typer.Option(None, "--model", help="padrão: o 1º do leaderboard k=50"),
    strategy: str = typer.Option(None, "--strategy", help="padrão: a melhor do modelo"),
    per_cell: int = typer.Option(1500, "--per-cell",
                                 help="registros por dobra × classe em cada amostra"),
    seeds: str = typer.Option("42,1337", "--seeds",
                              help="sementes das amostras; ≥2 para medir estabilidade"),
    sanity: bool = typer.Option(False, "--sanity",
                                help="subamostra: valida o encanamento, não o resultado"),
) -> None:
    """SHAP consensual restrito (§4.2): explica APENAS o modelo campeão, out-of-fold,
    em amostra estratificada por dobra espacial × classe. Grava valores, resumo por
    classe com direção do efeito, estabilidade entre sementes e o manifesto da amostra."""
    import json
    import time

    import numpy as np
    import shap

    from tb_outcomes.executor import build_model_input
    from tb_outcomes.imbalance import apply_strategy, load_imbalance_config
    from tb_outcomes.models import build_adapter, load_models_config
    from tb_outcomes.preprocess import infer_column_types, load_preprocess_config
    from tb_outcomes.robustness import leaderboard
    from tb_outcomes.shap_analysis import (
        normalize_shap_output,
        stability,
        stratified_sample,
        summarize,
    )

    exec_cfg = load_executor_config(EXECUTOR)
    if model is None or strategy is None:
        lb = leaderboard(pd.read_csv(DATA_DIR / "benchmark_metrics.csv")).sort_values(
            "f1_macro", ascending=False)
        topo = lb.iloc[0]
        model, strategy = model or str(topo["model"]), strategy or str(topo["best_strategy"])
        logger.info("Modelo final pelo leaderboard: %s × %s (F1-macro %.4f)",
                    model, strategy, topo["f1_macro"])

    sementes = [int(s) for s in seeds.split(",") if s.strip()]
    X, y, strata, clusters, outer, classes, specs, labels = _load_benchmark_inputs(
        exec_cfg, sanity)
    if sanity:
        logger.warning("MODO SANIDADE: %d registros. Valida o encanamento, NÃO o resultado.",
                       len(X))

    mc = load_models_config(MODELS)
    entries = {**mc.baselines, **mc.models}
    pre_cfg = load_preprocess_config(PREPROCESS)
    imb_cfg = load_imbalance_config(IMBALANCE)
    familia = entries[model].family

    t0 = time.time()
    valores, resumos, amostra_meta = [], {s: [] for s in sementes}, []
    for f in sorted(pd.Series(outer).unique()):
        dev = np.where(np.asarray(outer) != f)[0]
        ev = np.where(np.asarray(outer) == f)[0]

        # refit no DEV desta dobra — o mesmo fit que gerou o OOF do EVAL (§4.2, decisão 1)
        adapter = build_adapter(model, entries[model])
        types = infer_column_types(X.iloc[dev], specs, pre_cfg.max_cardinality)
        in_dev, in_ev = build_model_input(
            X.iloc[dev].reset_index(drop=True), X.iloc[ev].reset_index(drop=True),
            types, adapter.capabilities.preprocess_profile, adapter.input_kind, pre_cfg)
        res = apply_strategy(y.iloc[dev].reset_index(drop=True),
                             strata.iloc[dev].reset_index(drop=True), strategy, imb_cfg,
                             exec_cfg.random_state,
                             supports_weight=adapter.capabilities.sample_weight)
        adapter.fit(in_dev.iloc[res.train_index], y.to_numpy()[dev][res.train_index],
                    sample_weight=res.sample_weight)

        explainer = shap.TreeExplainer(adapter._est)
        y_ev, fold_ev = y.to_numpy()[ev], np.full(len(ev), f)
        for semente in sementes:
            sel = stratified_sample(y_ev, fold_ev, per_cell=per_cell, seed=semente)
            sv = normalize_shap_output(
                explainer.shap_values(in_ev.iloc[sel]), len(sel), in_ev.shape[1], len(classes))
            nomes = list(in_ev.columns)
            resumos[semente].append(summarize(sv, nomes, labels,
                                              extra={"outer_fold": int(f), "seed": semente}))
            if semente == sementes[0]:  # valores completos só do painel principal
                for j, classe in enumerate(labels):
                    bloco = pd.DataFrame(sv[j], columns=nomes)
                    bloco.insert(0, "classe", classe)
                    bloco.insert(0, "y_true", y_ev[sel])
                    bloco.insert(0, "record_pos", ev[sel])
                    bloco.insert(0, "outer_fold", int(f))
                    valores.append(bloco)
            amostra_meta.append({"outer_fold": int(f), "seed": semente, "n": int(len(sel)),
                                 **{f"n_{labels[c]}": int((y_ev[sel] == c).sum())
                                    for c in range(len(classes))}})
        logger.info("SHAP dobra %s: DEV=%d EVAL=%d explicados=%d", f, len(dev), len(ev),
                    len(sel))

    # resumo do painel: média das dobras, para não deixar a dobra maior dominar
    resumo_final = []
    for semente, partes in resumos.items():
        d = pd.concat(partes, ignore_index=True)
        agg = (d.groupby(["classe", "feature"], as_index=False)
                 .agg(mean_abs_shap=("mean_abs_shap", "mean"),
                      mean_shap=("mean_shap", "mean"),
                      pct_positivo=("pct_positivo", "mean"),
                      n_dobras=("outer_fold", "nunique")))
        agg["rank"] = agg.groupby("classe")["mean_abs_shap"].rank(
            ascending=False, method="min").astype(int)
        agg["_run"] = f"seed{semente}"
        resumo_final.append(agg.sort_values(["classe", "rank"]).reset_index(drop=True))

    est = stability(resumo_final, top_n=15)
    resumo = pd.concat(resumo_final, ignore_index=True)

    p_val = DATA_DIR / "shap_values.parquet"
    pd.concat(valores, ignore_index=True).to_parquet(p_val, index=False)
    resumo.to_csv(DATA_DIR / "shap_summary.csv", index=False)
    est.to_csv(DATA_DIR / "shap_stability.csv", index=False)

    manifesto = {
        "model": model, "strategy": strategy, "family": familia,
        "explainer": "shap.TreeExplainer",
        "feature_perturbation": getattr(explainer, "feature_perturbation", None),
        "background_set": ("nenhum conjunto explícito: tree_path_dependent usa a cobertura "
                           "de caminho das árvores do próprio treino"),
        "n_features": int(in_ev.shape[1]),
        "classes": list(labels),
        "amostragem": {"estratificacao": "dobra espacial externa × classe",
                       "por_celula": int(per_cell), "sementes": sementes,
                       "sem_reposicao": True},
        "composicao_da_amostra": amostra_meta,
        "protocolo": ("refit por dobra no DEV, explicação de registros do EVAL "
                      "(nunca registros vistos no treino)"),
        "n_linhas_valores": int(sum(len(v) for v in valores)),
        "segundos": round(time.time() - t0, 1),
    }
    with (DATA_DIR / "shap_manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(manifesto, fh, ensure_ascii=False, indent=2)

    rho = est["spearman"].min() if len(est) else float("nan")
    typer.echo(f"[shap] {model} × {strategy} | {manifesto['n_linhas_valores']} linhas de valores "
               f"| estabilidade Spearman mín {rho:.3f} | {p_val}, shap_summary.csv, "
               f"shap_stability.csv, shap_manifest.json")


@app.command("equity-report")
def equity_report_cmd(
    model: str = typer.Option(None, "--model", help="padrão: o 1º do leaderboard k=50"),
    strategy: str = typer.Option(None, "--strategy", help="padrão: a melhor do modelo"),
) -> None:
    """Equidade e subgrupos (§3.2): recorta o OOF do k=50 por região, sexo, raça/cor,
    faixa etária, escolaridade, vulnerabilidade municipal e dobra espacial. Não treina
    nada — os números saem das MESMAS predições da tabela principal."""
    from tb_outcomes.equity import build_subgroup_frame, disparity_summary, equity_table
    from tb_outcomes.robustness import leaderboard

    exec_cfg = load_executor_config(EXECUTOR)

    if model is None or strategy is None:
        lb = leaderboard(pd.read_csv(DATA_DIR / "benchmark_metrics.csv")).sort_values(
            "f1_macro", ascending=False)
        topo = lb.iloc[0]
        model = model or str(topo["model"])
        strategy = strategy or str(topo["best_strategy"])
        logger.info("Par escolhido pelo leaderboard: %s × %s (F1-macro %.4f)",
                    model, strategy, topo["f1_macro"])

    X, y, strata, clusters, outer, classes, specs, labels = _load_benchmark_inputs(
        exec_cfg, False)
    subgroups = build_subgroup_frame(X, outer)

    # filtro empurrado para o parquet: o OOF do k=50 tem 70 pares e ~1,7 GB
    oof = pd.read_parquet(
        DATA_DIR / "oof_predictions.parquet",
        filters=[("model", "==", model), ("strategy", "==", strategy)],
    )
    if oof.empty:
        raise typer.BadParameter(f"o OOF não tem o par {model} × {strategy}")
    logger.info("OOF do par: %d linhas; %d registros no X.", len(oof), len(X))

    tabelas = [equity_table(oof, subgroups, classes, labels, rule=r)
               for r in ("pred_policy", "pred_argmax")]
    tab = pd.concat(tabelas, ignore_index=True)
    tab.insert(0, "model", model)
    tab.insert(1, "strategy", strategy)

    disp = pd.concat([disparity_summary(tab, m) for m in ("fn_rate", "ppv", "especificidade")],
                     ignore_index=True)
    disp.insert(0, "model", model)
    disp.insert(1, "strategy", strategy)

    p1 = DATA_DIR / "equity_table.csv"
    p2 = DATA_DIR / "equity_disparity.csv"
    tab.to_csv(p1, index=False)
    disp.to_csv(p2, index=False)
    n_sup = int(tab["suprimido"].sum())
    typer.echo(f"[equity] {model} × {strategy} | {len(tab)} linhas "
               f"({n_sup} suprimidas por célula <{tab['min_cell'].iloc[0]}) | {p1}, {p2}")


@app.command("run-ablation")
def run_ablation_cmd() -> None:
    """Ablação territorial (§3.3): roda o conjunto de ablação em cada braço de features,
    com checkpoint por par, gravando em data/ablation/{scope}/. O braço 'combined' NÃO
    roda aqui — é a produção k=50 já existente em data/, reaproveitada pelo relatório."""
    from tb_outcomes.calibration import load_calibration_config
    from tb_outcomes.executor import make_sklearn_fold_factory, run_benchmark
    from tb_outcomes.executor_io import write_benchmark_artifacts
    from tb_outcomes.imbalance import load_imbalance_config
    from tb_outcomes.models import build_registry, load_models_config
    from tb_outcomes.preprocess import load_preprocess_config

    exec_cfg = load_executor_config(EXECUTOR)
    if exec_cfg.ablation is None:
        raise typer.BadParameter("configs/executor.yaml não tem a seção 'ablation'")
    abl = exec_cfg.ablation
    registry = build_registry(load_models_config(MODELS))

    for scope in abl.scopes:
        out_s = Path(abl.out_root) / scope
        ckpt = out_s / "pairs"
        typer.echo(f"[ablation] braço '{scope}': {len(abl.models)} modelos × "
                   f"{len(exec_cfg.strategies)} estratégias -> {out_s} (checkpoint em {ckpt})")
        X, y, strata, clusters, outer, classes, specs, labels = _load_benchmark_inputs(
            exec_cfg, False, feature_scope=scope)
        factory = make_sklearn_fold_factory(
            X, y, strata, registry, load_imbalance_config(IMBALANCE),
            load_preprocess_config(PREPROCESS), specs, exec_cfg.random_state)
        result = run_benchmark(
            abl.models, exec_cfg.strategies, y, clusters, outer, len(classes), classes,
            factory, lambda m: registry[m].capabilities,
            load_calibration_config(CALIBRATION), exec_cfg.n_inner_folds,
            checkpoint_dir=ckpt)
        write_benchmark_artifacts(result, out_s)
        n_ok = sum(1 for r in result.status_rows if r["status"] == "ok")
        typer.echo(f"[ablation] braço '{scope}': {n_ok}/{len(result.status_rows)} pares ok "
                   f"({X.shape[1]} features) -> {out_s}")


@app.command("sweep-report")
def sweep_report_cmd() -> None:
    """Consolida a robustez do sweep: tabela k∈{27,50,75}, estabilidade de ranking
    (Kendall τ + overlap do top-N) e figuras. Lê o k=50 de data/ e os k do sweep de
    data/sweep/k{K}/. Requer o sweep já concluído."""
    from pathlib import Path

    from tb_outcomes.figures import fig_leaderboard_across_k, fig_rank_bump
    from tb_outcomes.robustness import rank_stability, robustness_table

    exec_cfg = load_executor_config(EXECUTOR)
    if exec_cfg.sweep is None:
        raise typer.BadParameter("configs/executor.yaml não tem a seção 'sweep'")
    sweep = exec_cfg.sweep
    ref_k = exec_cfg.k  # 50, braço de referência

    def _load(k):
        path = (DATA_DIR / "benchmark_metrics.csv" if k == ref_k
                else Path(sweep.out_root) / f"k{k}" / "benchmark_metrics.csv")
        if not path.exists():
            raise typer.BadParameter(f"faltando métricas de k={k}: {path}")
        return pd.read_csv(path)

    ks = sorted({ref_k, *sweep.ks})
    metrics_by_k = {k: _load(k) for k in ks}

    tab = robustness_table(metrics_by_k, models=sweep.models)
    st = rank_stability(metrics_by_k, models=sweep.models, top_n=5)

    out_root = Path(sweep.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    tab.to_csv(out_root / "robustness_table.csv", index=False)
    st.to_csv(out_root / "rank_stability.csv", index=False)
    p1 = fig_leaderboard_across_k(tab)
    p2 = fig_rank_bump(tab)
    typer.echo(f"[sweep-report] k={ks} | robustness_table.csv, rank_stability.csv | {p1}, {p2}")
    typer.echo(st.to_string(index=False))


if __name__ == "__main__":
    app()
