"""Executor da validação espacial aninhada (ESPECIFICACAO §1; SP4e).

Funções puras em camadas. A lógica de laço é separada da fiação sklearn pelo
seam `make_sklearn_fold_factory`, o que a torna testável com stubs. O executor
não tem lógica de modelo: orquestra splits/preprocess/imbalance/models/calibration.

Opção A: benchmark de defaults, sem busca. O laço interno só gera OOF e ajusta
preprocess/reamostragem/calibração/limiares. EVAL externo é tocado uma vez.
"""
from __future__ import annotations

import logging
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel

from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from tb_outcomes.calibration import (
    assert_calibrated_before_metrics,
    calibration_manifest,
    calibration_metrics,
    classify_argmax,
    classify_policy,
    fit_calibrator,
    fit_thresholds,
)
from tb_outcomes.checkpoint import (
    load_pair_checkpoint,
    pair_checkpoint_exists,
    write_pair_checkpoint,
)
from tb_outcomes.imbalance import apply_strategy
from tb_outcomes.models import build_adapter, load_models_config
from tb_outcomes.preprocess import (
    coerce_for_sklearn,
    infer_column_types,
    make_preprocessor,
)
from tb_outcomes.splits import make_inner_folds

logger = logging.getLogger("tb_outcomes")


class SweepConfig(BaseModel):
    ks: list[int]
    out_root: str
    models: list[str]


class TemporalConfig(BaseModel):
    """Holdout temporal (ESPECIFICACAO §4.1)."""

    out_root: str
    models_from_sweep: bool = True
    models: list[str] | None = None


class AblationConfig(BaseModel):
    """Ablação territorial (ESPECIFICACAO §3.3)."""

    scopes: list[str]
    out_root: str
    models: list[str]


class ExecutorConfig(BaseModel):
    outcome_schema: str
    strategies: list[str]
    n_outer_folds: int
    n_inner_folds: int
    fold_method: str
    k: int
    random_state: int
    sanity_subsample: int
    models: list[str]
    sweep: SweepConfig | None = None
    ablation: AblationConfig | None = None
    temporal: TemporalConfig | None = None


def load_executor_config(path: str | Path) -> ExecutorConfig:
    with Path(path).open("r", encoding="utf-8") as fh:
        return ExecutorConfig.model_validate(yaml.safe_load(fh))


def as_score_matrix(raw: np.ndarray, n_classes: int) -> np.ndarray:
    raw = np.asarray(raw, dtype=float)
    if raw.ndim == 1:
        if n_classes != 2:
            raise ValueError(
                f"escore 1D só é válido para binário; n_classes={n_classes}"
            )
        return np.column_stack([-raw, raw])
    if raw.ndim == 2 and raw.shape[1] == n_classes:
        return raw
    raise ValueError(
        f"forma {raw.shape} incompatível com n_classes={n_classes}"
    )


def inner_oof_scores(y_dev, clusters_dev, n_classes, n_inner, fit_predict_fn,
                     fold_method="binpack") -> np.ndarray:
    y_dev = pd.Series(y_dev).reset_index(drop=True)
    clusters_dev = pd.Series(clusters_dev).reset_index(drop=True)
    oof = np.full((len(y_dev), n_classes), np.nan, dtype=float)
    for train_idx, val_idx in make_inner_folds(clusters_dev, y_dev, n_inner, fold_method):
        scores = np.asarray(fit_predict_fn(train_idx, val_idx), dtype=float)
        if scores.shape != (len(val_idx), n_classes):
            raise ValueError(
                f"fit_predict_fn devolveu {scores.shape}, esperado {(len(val_idx), n_classes)}"
            )
        oof[val_idx] = scores
    if np.isnan(oof).any():
        raise ValueError("OOF interno incompleto: alguma linha não foi validada")
    return oof


def _safe_auc(fn, yb, p) -> float:
    # AUC/AUPRC exigem as duas classes presentes; senão é indefinido.
    if len(np.unique(yb)) < 2:
        return float("nan")
    return float(fn(yb, p))


def discrimination_metrics(y_true, probs, pred_argmax, classes) -> list[dict]:
    y_true = np.asarray(y_true)
    probs = np.asarray(probs, dtype=float)
    pred_argmax = np.asarray(pred_argmax)
    classes = list(classes)
    rows = []
    for j, c in enumerate(classes):
        yb = (y_true == c).astype(int)
        pb = (pred_argmax == c).astype(int)
        rows.append({
            "class": c, "axis": "discrimination",
            "recall": float(recall_score(yb, pb, zero_division=0)),
            "precision": float(precision_score(yb, pb, zero_division=0)),
            "auprc": _safe_auc(average_precision_score, yb, probs[:, j]),
            "roc_auc": _safe_auc(roc_auc_score, yb, probs[:, j]),
        })
    rows.append({
        "class": "__global__", "axis": "discrimination",
        "f1_macro": float(f1_score(y_true, pred_argmax, average="macro", labels=classes, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred_argmax)),
    })
    return rows


@dataclass
class PairResult:
    model: str
    strategy: str
    oof_rows: list = field(default_factory=list)
    metric_rows: list = field(default_factory=list)
    manifest_rows: list = field(default_factory=list)
    status: str = "ok"


def raw_norm(raw: np.ndarray) -> np.ndarray:
    # normaliza o escore bruto para uma pseudo-proba só para medir a calibração
    # PRÉ-calibração (a distorção vira número); clip a positivo evita soma zero.
    r = np.clip(np.asarray(raw, dtype=float), 1e-12, None)
    return r / r.sum(axis=1, keepdims=True)


def _evaluate_split(res: PairResult, model, strategy, y, record_clusters, dev_idx, eval_idx,
                    n_classes, classes, make_fold_fn, capabilities, cal_cfg, n_inner,
                    fold_id) -> None:
    """Avalia UM par dev/eval e acumula as linhas em `res`.

    Extraído de `evaluate_pair` para o holdout temporal (§4.1) usar exatamente a mesma
    disciplina — calibrador e limiares ajustados SÓ no OOF interno do DEV, EVAL tocado uma
    vez — em vez de uma segunda implementação que poderia divergir dela em silêncio.
    `fold_id` identifica a partição: o índice da dobra espacial, ou o rótulo do holdout.
    """
    y_dev = y.iloc[dev_idx].reset_index(drop=True)
    clusters_dev = record_clusters.iloc[dev_idx].reset_index(drop=True)
    inner_fit_predict, dev_full_predict_eval = make_fold_fn(dev_idx, eval_idx, strategy)

    # OOF interno em DEV -> calibrador + limiares (só no OOF)
    oof_dev = inner_oof_scores(
        y_dev, clusters_dev, n_classes, n_inner,
        fit_predict_fn=lambda tr, va: as_score_matrix(inner_fit_predict(tr, va), n_classes),
    )
    calibrator = fit_calibrator(oof_dev, y_dev.to_numpy(), cal_cfg)
    cal_oof = calibrator.transform(oof_dev)
    thresholds = fit_thresholds(cal_oof, y_dev.to_numpy(), cal_cfg)

    # EVAL tocado uma vez: refit em DEV inteiro -> escore bruto -> calibra
    raw_eval = as_score_matrix(dev_full_predict_eval(), n_classes)
    proba_eval = calibrator.transform(raw_eval)
    assert_calibrated_before_metrics(capabilities, calibrated=True)
    pred_argmax = classify_argmax(proba_eval, classes)
    pred_policy = classify_policy(proba_eval, thresholds, classes)
    y_eval = y.iloc[eval_idx].to_numpy()

    # métricas: calibração (antes/depois renorm) + discriminação
    pre = calibration_metrics(y_eval, raw_norm(raw_eval), cal_cfg, stage="pre_renorm")
    post = calibration_metrics(y_eval, proba_eval, cal_cfg, stage="post_renorm")
    disc = discrimination_metrics(y_eval, proba_eval, pred_argmax, classes)
    for row in (*pre, *post, *disc):
        res.metric_rows.append({"model": model, "strategy": strategy,
                                "outer_fold": fold_id, **row})
    res.manifest_rows.append({
        "model": model, "strategy": strategy, "outer_fold": fold_id,
        **calibration_manifest(calibrator),
    })
    for i, rec in enumerate(eval_idx):
        row = {"record_pos": int(rec), "outer_fold": fold_id, "model": model,
               "strategy": strategy, "y_true": int(y_eval[i]),
               "pred_argmax": int(pred_argmax[i]), "pred_policy": int(pred_policy[i])}
        for jc, c in enumerate(classes):
            row[f"raw_score_{c}"] = float(raw_eval[i, jc])
            row[f"proba_{c}"] = float(proba_eval[i, jc])
        res.oof_rows.append(row)


def evaluate_pair(model, strategy, y, record_clusters, outer_folds, n_classes, classes,
                  make_fold_fn, capabilities, cal_cfg, n_inner=5) -> PairResult:
    y = pd.Series(y).reset_index(drop=True)
    record_clusters = pd.Series(record_clusters).reset_index(drop=True)
    outer_folds = pd.Series(outer_folds).reset_index(drop=True)
    res = PairResult(model=model, strategy=strategy)
    for f in sorted(outer_folds.unique()):
        eval_idx = np.where(outer_folds.to_numpy() == f)[0]
        dev_idx = np.where(outer_folds.to_numpy() != f)[0]
        _evaluate_split(res, model, strategy, y, record_clusters, dev_idx, eval_idx,
                        n_classes, classes, make_fold_fn, capabilities, cal_cfg, n_inner,
                        fold_id=int(f))
    return res


def evaluate_temporal_pair(model, strategy, y, record_clusters, train_mask, n_classes,
                           classes, make_fold_fn, capabilities, cal_cfg, n_inner=5,
                           fold_id="temporal_holdout") -> PairResult:
    """Holdout temporal (ESPECIFICACAO §4.1): treina no passado, avalia no ano reservado.

    Uma partição só, no tempo, em vez das 5 dobras espaciais. A disciplina é idêntica à da
    validação espacial (mesmo `_evaluate_split`): o calibrador e os limiares saem do OOF
    interno do período de TREINO, e o ano de teste é tocado uma única vez.

    O agrupamento do OOF interno continua sendo o cluster espacial — dentro do período de
    treino a dependência entre municípios não desaparece por o corte externo ser temporal.
    """
    y = pd.Series(y).reset_index(drop=True)
    record_clusters = pd.Series(record_clusters).reset_index(drop=True)
    train_mask = np.asarray(train_mask, dtype=bool)
    dev_idx, eval_idx = np.where(train_mask)[0], np.where(~train_mask)[0]
    if len(dev_idx) == 0 or len(eval_idx) == 0:
        raise ValueError(
            f"partição temporal degenerada: treino={len(dev_idx)}, teste={len(eval_idx)}")
    res = PairResult(model=model, strategy=strategy)
    _evaluate_split(res, model, strategy, y, record_clusters, dev_idx, eval_idx,
                    n_classes, classes, make_fold_fn, capabilities, cal_cfg, n_inner,
                    fold_id=fold_id)
    return res


_ID_KEYS = {"model", "strategy", "outer_fold", "class", "stage", "axis"}


def aggregate_pair(pair_result: PairResult) -> list[dict]:
    buckets: dict = {}
    for row in pair_result.metric_rows:
        for metric, value in row.items():
            if metric in _ID_KEYS or not isinstance(value, (int, float)):
                continue
            key = (row.get("class", "__global__"), row.get("stage"), row.get("axis"), metric)
            buckets.setdefault(key, []).append(float(value))
    out = []
    for (klass, stage, axis, metric), vals in buckets.items():
        arr = np.array([v for v in vals if not np.isnan(v)], dtype=float)
        if arr.size == 0:
            continue
        out.append({
            "model": pair_result.model, "strategy": pair_result.strategy,
            "metric": metric, "class": klass, "stage": stage, "axis": axis,
            "mean": float(arr.mean()), "median": float(np.median(arr)),
            "min": float(arr.min()), "max": float(arr.max()),
            "std": float(arr.std(ddof=0)), "n_folds": int(arr.size),
        })
    return out


@dataclass
class BenchmarkResult:
    oof_rows: list = field(default_factory=list)
    metric_rows: list = field(default_factory=list)
    aggregate_rows: list = field(default_factory=list)
    manifest_rows: list = field(default_factory=list)
    status_rows: list = field(default_factory=list)


def run_benchmark(models, strategies, y, record_clusters, outer_folds, n_classes, classes,
                  make_fold_factory, capabilities_of, cal_cfg, n_inner,
                  checkpoint_dir=None) -> BenchmarkResult:
    out = BenchmarkResult()
    total = len(models) * len(strategies)
    done = 0
    for mi, model in enumerate(models):
        caps = capabilities_of(model)
        logger.info("[benchmark] modelo %d/%d: %s", mi + 1, len(models), model)
        for strategy in strategies:
            done += 1

            # resume: par já gravado -> carrega o shard e pula o refit
            if checkpoint_dir is not None and pair_checkpoint_exists(checkpoint_dir, model, strategy):
                ck = load_pair_checkpoint(checkpoint_dir, model, strategy)
                out.oof_rows.extend(ck["oof_rows"])
                out.metric_rows.extend(ck["metric_rows"])
                out.manifest_rows.extend(ck["manifest_rows"])
                out.aggregate_rows.extend(ck["aggregate_rows"])
                out.status_rows.append(ck["status_row"])
                logger.info("[benchmark] %d/%d %s × %s -> checkpoint (pulado: %s)",
                            done, total, model, strategy, ck["status_row"]["status"])
                continue

            if strategy == "local_cost_sensitive" and not getattr(caps, "sample_weight", False):
                logger.info("[benchmark] %d/%d %s × %s -> not_run_incompatible (sem sample_weight)",
                            done, total, model, strategy)
                status_row = {"model": model, "strategy": strategy,
                              "status": "not_run_incompatible"}
                out.status_rows.append(status_row)
                if checkpoint_dir is not None:
                    write_pair_checkpoint(checkpoint_dir, model, strategy,
                                          [], [], [], [], status_row)
                continue

            logger.info("[benchmark] %d/%d início %s × %s", done, total, model, strategy)
            t0 = time.time()
            make_fold_fn = make_fold_factory(model, strategy)
            try:
                pair = evaluate_pair(
                    model, strategy, y, record_clusters, outer_folds, n_classes, classes,
                    make_fold_fn, caps, cal_cfg, n_inner=n_inner,
                )
            except Exception as exc:  # noqa: BLE001 — um par que quebra não derruba o resto
                logger.warning("[benchmark] %d/%d %s × %s ERRO em %.1fs: %s",
                               done, total, model, strategy, time.time() - t0, str(exc)[:120])
                logger.error("[benchmark] traceback completo de %s × %s:\n%s",
                             model, strategy, traceback.format_exc())
                status_row = {"model": model, "strategy": strategy,
                              "status": "error", "error": str(exc)[:200]}
                out.status_rows.append(status_row)
                if checkpoint_dir is not None:
                    write_pair_checkpoint(checkpoint_dir, model, strategy,
                                          [], [], [], [], status_row)
                continue
            logger.info("[benchmark] %d/%d fim %s × %s: %s em %.1fs",
                        done, total, model, strategy, pair.status, time.time() - t0)
            agg = aggregate_pair(pair)
            out.oof_rows.extend(pair.oof_rows)
            out.metric_rows.extend(pair.metric_rows)
            out.manifest_rows.extend(pair.manifest_rows)
            out.aggregate_rows.extend(agg)
            status_row = {"model": model, "strategy": strategy, "status": pair.status}
            out.status_rows.append(status_row)
            if checkpoint_dir is not None:
                write_pair_checkpoint(checkpoint_dir, model, strategy,
                                      pair.oof_rows, pair.metric_rows, pair.manifest_rows,
                                      agg, status_row)
    return out


# ---------------------------------------------------------------------------
# Plumbing sklearn — concentrado aqui; as camadas acima não o tocam.
# ---------------------------------------------------------------------------

def build_model_input(X_tr, X_va, types, profile, input_kind, pre_cfg):
    """Monta a entrada conforme o input_kind declarado pelo adapter (SP4f).

    matrix: matriz one-hot do ColumnTransformer. categorical_frame: DataFrame com
    categóricas em dtype `category` (boosting nativo). dl_frame: DataFrame cru coerced
    (pytorch_tabular). O executor não decide por nome de modelo — só pelo input_kind.
    """
    if input_kind == "matrix":
        pre = make_preprocessor(profile, types, pre_cfg)
        Xtr = pre.fit_transform(coerce_for_sklearn(X_tr, types))
        Xva = pre.transform(coerce_for_sklearn(X_va, types))
        return Xtr, Xva
    ctr, cva = coerce_for_sklearn(X_tr, types), coerce_for_sklearn(X_va, types)
    if input_kind == "categorical_frame":
        for c in types.categorical:
            ctr[c] = ctr[c].astype("category")
            cva[c] = cva[c].astype("category")
        return ctr, cva
    if input_kind == "dl_frame":
        return ctr, cva
    raise ValueError(f"input_kind desconhecido: {input_kind}")


def _take_rows(X, idx):
    # matriz (numpy/sparse) indexa por linha; DataFrame usa iloc
    if isinstance(X, pd.DataFrame):
        return X.iloc[idx].reset_index(drop=True)
    return X[idx]


def _fit_transform_predict(X_tr, y_tr, X_va, model_name, strategy, entries, feature_specs,
                           pre_cfg, imb_cfg, strata_tr, seed, n_classes, dl_cfg=None):
    """Uma dobra de plumbing: prepara a entrada por input_kind, reamostra o treino, ajusta, prediz.

    Um adapter novo por fit (o registry guarda uma instância só; refit repetido sobre
    o mesmo estimador contaminaria as dobras). O estrato de custo vem POR FORA do X.
    """
    adapter = build_adapter(model_name, entries[model_name], dl_cfg)
    profile = adapter.capabilities.preprocess_profile
    types = infer_column_types(X_tr, feature_specs, pre_cfg.max_cardinality)
    input_tr, input_va = build_model_input(X_tr, X_va, types, profile, adapter.input_kind, pre_cfg)
    res = apply_strategy(y_tr, strata_tr, strategy, imb_cfg, seed,
                         supports_weight=adapter.capabilities.sample_weight)
    input_tr = _take_rows(input_tr, res.train_index)
    yr = y_tr.to_numpy()[res.train_index]
    adapter.fit(input_tr, yr, sample_weight=res.sample_weight)
    return as_score_matrix(adapter.predict_raw_scores(input_va), n_classes)


def make_sklearn_fold_factory(X, y, strata, registry, imb_cfg, pre_cfg, feature_specs, seed,
                              dl_cfg=None):
    X = X.reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)
    strata = pd.Series(strata).reset_index(drop=True)
    n_classes = int(y.nunique())
    entries = {**load_models_config("configs/models.yaml").baselines,
               **load_models_config("configs/models.yaml").models}

    def factory(model, strategy):
        def make_fold_fn(dev_idx, eval_idx, strategy):
            X_dev = X.iloc[dev_idx].reset_index(drop=True)
            y_dev = y.iloc[dev_idx].reset_index(drop=True)
            strata_dev = strata.iloc[dev_idx].reset_index(drop=True)
            X_eval = X.iloc[eval_idx].reset_index(drop=True)

            def inner(train_idx, val_idx):
                return _fit_transform_predict(
                    X_dev.iloc[train_idx], y_dev.iloc[train_idx], X_dev.iloc[val_idx],
                    model, strategy, entries, feature_specs, pre_cfg, imb_cfg,
                    strata_dev.iloc[train_idx], seed, n_classes, dl_cfg)

            def dev_eval():
                return _fit_transform_predict(
                    X_dev, y_dev, X_eval, model, strategy, entries, feature_specs,
                    pre_cfg, imb_cfg, strata_dev, seed, n_classes, dl_cfg)

            return inner, dev_eval
        return make_fold_fn
    return factory
