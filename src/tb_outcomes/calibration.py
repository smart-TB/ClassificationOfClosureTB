"""Componentes puros de calibração cross-fitted (briefing §15; ESPECIFICACAO §2.2).

Recebem escores OOF e não conhecem o laço — o SP4e gera os OOF e costura. Duas
correções honestas sobre os defaults do briefing:
  - guarda isotônica<->Platt por escassez de eventos OOF (isotônica superajusta na
    classe rara; Platt é robusta em amostra pequena) — a escolha vira número;
  - métricas antes E depois da renormalização — a distorção vira número.
Trava do §2.2/princípio 7: modelo sem proba nativa não gera métrica probabilística
sem calibração explícita.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import yaml
from pydantic import BaseModel
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss


class CalibrationConfig(BaseModel):
    method: Literal["auto", "isotonic", "sigmoid"]
    epsilon: float
    sum_tolerance: float
    isotonic_min_class_events: int
    ece_bins: int
    objective: str


def load_calibration_config(path: str | Path) -> CalibrationConfig:
    with Path(path).open("r", encoding="utf-8") as fh:
        return CalibrationConfig.model_validate(yaml.safe_load(fh))


def postprocess_probabilities(probs: np.ndarray, epsilon: float, tol: float) -> np.ndarray:
    probs = np.asarray(probs, dtype=float)
    if np.isnan(probs).any():
        raise ValueError("probabilidades com NaN antes do pós-processamento")
    if (probs < 0).any():
        raise ValueError("probabilidades negativas antes do pós-processamento")
    # Checa a soma da entrada CRUA, antes do clip: uma linha toda-zero é degenerada
    # (o modelo não produziu nada) e deve falhar, não ser resgatada pelo epsilon e
    # virada silenciosamente em uniforme.
    if (probs.sum(axis=1) <= 0).any():
        raise ValueError("linha com soma não-positiva: impossível renormalizar")
    clipped = np.clip(probs, epsilon, 1 - epsilon)
    row_sums = clipped.sum(axis=1, keepdims=True)
    out = clipped / row_sums
    if not np.allclose(out.sum(axis=1), 1.0, atol=max(tol, 1e-9)):
        raise ValueError("soma das probabilidades fora da tolerância após renormalizar")
    return out


@dataclass
class Calibrator:
    classes: list
    method_per_class: dict
    models_per_class: dict
    epsilon: float
    sum_tolerance: float

    def transform(self, raw_scores: np.ndarray) -> np.ndarray:
        raw_scores = np.asarray(raw_scores, dtype=float)
        cols = []
        for j, c in enumerate(self.classes):
            model = self.models_per_class[c]
            s = raw_scores[:, j]
            if self.method_per_class[c] == "isotonic":
                cols.append(model.predict(s))
            else:  # sigmoid/Platt
                cols.append(model.predict_proba(s.reshape(-1, 1))[:, 1])
        stacked = np.clip(np.column_stack(cols), 0, 1)
        return postprocess_probabilities(stacked, self.epsilon, self.sum_tolerance)


def _choose_method(n_events: int, cfg: CalibrationConfig, method: str) -> str:
    if method == "isotonic":
        return "isotonic"
    if method == "sigmoid":
        return "sigmoid"
    # auto: a guarda por escassez de eventos
    return "isotonic" if n_events >= cfg.isotonic_min_class_events else "sigmoid"


def fit_calibrator(oof_scores, y, cfg: CalibrationConfig, method: str | None = None) -> Calibrator:
    method = method or cfg.method
    oof_scores = np.asarray(oof_scores, dtype=float)
    y = np.asarray(y)
    classes = sorted(np.unique(y).tolist())
    method_per_class, models_per_class = {}, {}
    for j, c in enumerate(classes):
        target = (y == c).astype(int)
        chosen = _choose_method(int(target.sum()), cfg, method)
        s = oof_scores[:, j]
        if chosen == "isotonic":
            m = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            m.fit(s, target)
        else:
            m = LogisticRegression()
            m.fit(s.reshape(-1, 1), target)
        method_per_class[c] = chosen
        models_per_class[c] = m
    return Calibrator(classes, method_per_class, models_per_class, cfg.epsilon, cfg.sum_tolerance)


def expected_calibration_error(y_true_bin, p, n_bins: int, adaptive: bool = False) -> float:
    y_true_bin = np.asarray(y_true_bin, dtype=float)
    p = np.asarray(p, dtype=float)
    if adaptive:  # bins por quantil (amostra igual por bin) — melhor p/ classe rara
        edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
        edges[0], edges[-1] = 0.0, 1.0
        edges = np.unique(edges)
    else:
        edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(p)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p > lo) & (p <= hi) if lo > 0 else (p >= lo) & (p <= hi)
        if mask.sum() == 0:
            continue
        conf = p[mask].mean()
        acc = y_true_bin[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


def _cal_slope_intercept(y_true_bin, p) -> tuple[float, float]:
    eps = 1e-12
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    lr = LogisticRegression()
    try:
        lr.fit(logit, y_true_bin)
        return float(lr.coef_[0, 0]), float(lr.intercept_[0])
    except ValueError:  # uma só classe presente
        return float("nan"), float("nan")


def _onehot(y, classes) -> np.ndarray:
    idx = {c: j for j, c in enumerate(classes)}
    oh = np.zeros((len(y), len(classes)))
    for i, v in enumerate(y):
        oh[i, idx[v]] = 1.0
    return oh


def calibration_metrics(y_true, probs, cfg: CalibrationConfig, *, stage: str) -> list[dict]:
    y_true = np.asarray(y_true)
    probs = np.asarray(probs, dtype=float)
    classes = sorted(np.unique(y_true).tolist())
    rows = []
    for j, c in enumerate(classes):
        yb = (y_true == c).astype(int)
        p = probs[:, j]
        slope, intercept = _cal_slope_intercept(yb, p)
        rows.append({
            "class": c, "stage": stage,
            "log_loss": float(log_loss(yb, np.c_[1 - p, p], labels=[0, 1])),
            "brier_ovr": float(brier_score_loss(yb, p)),
            "ece_fixed": expected_calibration_error(yb, p, cfg.ece_bins, adaptive=False),
            "ece_adaptive": expected_calibration_error(yb, p, cfg.ece_bins, adaptive=True),
            "cal_slope": slope,
            "cal_intercept": intercept,
        })
    rows.append({
        "class": "__global__", "stage": stage,
        "log_loss": float(log_loss(y_true, probs, labels=classes)),
        "brier_multiclass": float(np.mean(np.sum((probs - _onehot(y_true, classes)) ** 2, axis=1))),
    })
    return rows


def classify_argmax(probs, classes) -> np.ndarray:
    probs = np.asarray(probs, dtype=float)
    classes = list(classes)
    return np.array([classes[i] for i in probs.argmax(axis=1)])


def classify_policy(probs, thresholds, classes) -> np.ndarray:
    probs = np.asarray(probs, dtype=float)
    classes = list(classes)
    thr = np.array([thresholds[c] for c in classes], dtype=float)
    scores = probs / thr  # score_c = p_c / threshold_c
    # desempate determinístico: maior score; np.argmax devolve o primeiro em empate,
    # e como as colunas estão na ordem canônica das classes, o desempate é estável
    idx = scores.argmax(axis=1)
    return np.array([classes[i] for i in idx])


def fit_thresholds(oof_probs, y, cfg: CalibrationConfig) -> dict:
    # busca em grade por classe: o limiar que maximiza F1 nas predições OOF. Ajustado
    # SÓ no OOF fornecido (nunca no teste externo). Vetorizado — TP/FP/FN por limiar de
    # uma vez, sem o overhead de chamar f1_score do sklearn 19 vezes por classe; o
    # argmax escolhe a MENOR threshold no empate, idêntico à busca estrita original.
    oof_probs = np.asarray(oof_probs, dtype=float)
    y = np.asarray(y)
    classes = sorted(np.unique(y).tolist())
    grid = np.linspace(0.05, 0.95, 19)
    best = {}
    for j, c in enumerate(classes):
        pos = y == c
        p = oof_probs[:, j]
        ge = p[:, None] >= grid[None, :]          # (n, 19): pred=1 por limiar
        tp = (ge & pos[:, None]).sum(axis=0).astype(float)
        fp = (ge & ~pos[:, None]).sum(axis=0).astype(float)
        fn = float(pos.sum()) - tp
        denom = 2 * tp + fp + fn
        f1 = np.where(denom > 0, 2 * tp / denom, 0.0)
        best[c] = float(grid[int(np.argmax(f1))])
    return best


def assert_calibrated_before_metrics(capabilities, calibrated: bool) -> None:
    if getattr(capabilities, "calibrator_required", False) and not calibrated:
        raise ValueError(
            f"{capabilities.name}: modelo sem probabilidade nativa não pode gerar métrica "
            f"probabilística sem calibração explícita (§2.2, princípio 7)"
        )


def calibration_manifest(calibrator: Calibrator, n_folds: int = 1) -> dict:
    methods = list(calibrator.method_per_class.values())
    n = len(methods)
    n_iso = sum(m == "isotonic" for m in methods)
    n_sig = n - n_iso
    return {
        "n_folds": n_folds,
        "n_classes": n,
        "n_isotonic": n_iso,
        "n_sigmoid": n_sig,
        "frac_isotonic": (n_iso / n) if n else 0.0,
        "method_per_class": dict(calibrator.method_per_class),
    }
