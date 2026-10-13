"""Candidato experimental: otimizar a decisão multiclasse real no desenvolvimento.

Não é ativado no benchmark histórico. Não ajustar em OOF externo/holdout final.
Busca coordenada limitada: não promete ótimo global.
"""
from __future__ import annotations

import numpy as np


def fit_joint_macro_policy(probs, y, classes, initial_thresholds=None):
    p = np.asarray(probs, dtype=float)
    y = np.asarray(y)
    classes = list(classes)
    if p.shape != (len(y), len(classes)) or not np.isfinite(p).all() or (p < 0).any():
        raise ValueError("Probabilidades inválidas para otimização da política")
    lookup = {c: i for i, c in enumerate(classes)}
    target = np.array([lookup[c] for c in y])
    k = len(classes)

    def score(weights):
        pred = (p * weights).argmax(axis=1)
        cm = np.bincount(k * target + pred, minlength=k * k).reshape(k, k)
        den = cm.sum(axis=0) + cm.sum(axis=1)
        return np.divide(2 * cm.diagonal(), den, out=np.zeros(k), where=den > 0).mean()

    starts = [np.ones(k)]
    if initial_thresholds is not None:
        t = np.array([initial_thresholds[c] for c in classes], dtype=float)
        if not np.isfinite(t).all() or (t <= 0).any():
            raise ValueError("Limiares iniciais inválidos")
        starts.append((1 / t) / (1 / t[0]))
    best, best_score = starts[0].copy(), score(starts[0])
    grid = np.logspace(-2, 2, 33)
    for weights in starts:
        weights = weights.copy()
        current = score(weights)
        for _ in range(4):
            changed = False
            for j in range(1, k):  # classe 0 ancora a escala, sem perda por fator comum
                next_weights, next_score = weights, current
                for v in grid:
                    candidate = weights.copy()
                    candidate[j] = v
                    value = score(candidate)
                    if value > next_score + 1e-12:
                        next_weights, next_score = candidate, value
                if next_score > current + 1e-12:
                    weights, current, changed = next_weights, next_score, True
            if not changed:
                break
        if current > best_score + 1e-12:
            best, best_score = weights.copy(), current
    thresholds = 1 / best
    thresholds /= thresholds.max()  # mesma decisão, valores em (0,1]
    return dict(zip(classes, thresholds.tolist()))
