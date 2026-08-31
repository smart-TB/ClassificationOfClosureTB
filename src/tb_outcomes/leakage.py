"""Testes de vazamento do protocolo aninhado (ESPECIFICACAO §7).

Funções que FALHAM. Cada uma teria pego um defeito real da análise original:
vazamento de grupo, estratos no X, incoerência de contagem, não-reprodutibilidade,
transformador ajustado sobre avaliação. A não-identidade entre estratégias reusa
imbalance.assert_strategies_differ.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

STRATA_COLUMNS = {"NU_ANO", "ID_MN_RESI", "ID_MUNIC_ANALISE", "ID_MUNICIP"}


def assert_no_group_overlap(train_clusters, eval_clusters) -> None:
    shared = set(np.asarray(train_clusters).tolist()) & set(np.asarray(eval_clusters).tolist())
    if shared:
        raise ValueError(f"vazamento de grupo: cluster(s) {sorted(shared)} em treino e avaliação")


def assert_strata_out_of_features(columns) -> None:
    hit = STRATA_COLUMNS & set(columns)
    if hit:
        raise ValueError(f"estrato(s) {sorted(hit)} não podem entrar no X (ex.: NU_ANO/ID_MN_RESI)")


def assert_outcome_count_coherence(y, expected_n: int) -> None:
    n = len(pd.Series(y).dropna())
    if n != expected_n:
        raise ValueError(f"incoerência de contagem: soma dos desfechos {n} != n da coorte {expected_n}")


def assert_reproducible(rows_a, rows_b, tol: float = 1e-9) -> None:
    if len(rows_a) != len(rows_b):
        raise ValueError("não-reprodutível: número de linhas difere entre execuções")
    for ra, rb in zip(rows_a, rows_b):
        for k, va in ra.items():
            vb = rb.get(k)
            if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                if not np.isclose(va, vb, atol=tol, equal_nan=True):
                    raise ValueError(f"não-reprodutível: {k} {va} != {vb} sob mesma semente")
            elif va != vb:
                raise ValueError(f"não-reprodutível: {k} {va} != {vb}")


def assert_transformers_fit_on_train_only(fit_indices, eval_indices) -> None:
    shared = set(np.asarray(fit_indices).tolist()) & set(np.asarray(eval_indices).tolist())
    if shared:
        raise ValueError(f"transformador ajustado sobre índice de avaliação: {sorted(shared)}")
