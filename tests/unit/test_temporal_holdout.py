"""Holdout temporal (ESPECIFICACAO §4.1).

O que estes testes protegem: que o ano de teste NUNCA participe do ajuste — nem do treino,
nem do calibrador, nem dos limiares — e que a partição temporal seja recusada quando
degenerada. É a mesma disciplina da validação espacial; a diferença é só o eixo do corte.
"""
import numpy as np
import pandas as pd
import pytest

from tb_outcomes.calibration import load_calibration_config
from tb_outcomes.executor import evaluate_pair, evaluate_temporal_pair


class _Caps:
    sample_weight = True
    predict_proba = True


def _cal_cfg():
    return load_calibration_config("configs/calibration.yaml")


def _setup(n=600, n_classes=3, seed=0):
    rng = np.random.default_rng(seed)
    y = pd.Series(rng.integers(0, n_classes, n))
    clusters = pd.Series(rng.integers(0, 12, n))
    ano = pd.Series(rng.choice([2021, 2022, 2023, 2024], n))
    return y, clusters, ano


def _factory(vistos: list, n_classes=3):
    """Fábrica que registra QUAIS índices cada ajuste enxergou."""
    def make_fold_fn(dev_idx, eval_idx, strategy):
        vistos.append({"dev": np.asarray(dev_idx), "eval": np.asarray(eval_idx)})

        def inner(train_idx, val_idx):
            vistos.append({"inner_train_global": np.asarray(dev_idx)[np.asarray(train_idx)]})
            rng = np.random.default_rng(1)
            return rng.random((len(val_idx), n_classes))

        def dev_eval():
            rng = np.random.default_rng(2)
            return rng.random((len(eval_idx), n_classes))

        return inner, dev_eval
    return make_fold_fn


def test_ano_de_teste_nao_entra_em_nenhum_ajuste():
    y, clusters, ano = _setup()
    treino = (ano < 2024).to_numpy()
    idx_teste = set(np.where(~treino)[0].tolist())

    vistos = []
    evaluate_temporal_pair("m", "random_undersampling", y, clusters, treino, 3, [0, 1, 2],
                           _factory(vistos), _Caps(), _cal_cfg(), n_inner=3)

    # nenhum ajuste — nem o interno da calibração, nem o refit — viu o ano de teste
    for v in vistos:
        if "dev" in v:
            assert not (set(v["dev"].tolist()) & idx_teste)
        if "inner_train_global" in v:
            assert not (set(v["inner_train_global"].tolist()) & idx_teste)


def test_avalia_exatamente_o_ano_reservado():
    y, clusters, ano = _setup()
    treino = (ano < 2024).to_numpy()
    res = evaluate_temporal_pair("m", "random_undersampling", y, clusters, treino, 3,
                                 [0, 1, 2], _factory([]), _Caps(), _cal_cfg(), n_inner=3)
    posicoes = {r["record_pos"] for r in res.oof_rows}
    assert posicoes == set(np.where(~treino)[0].tolist())
    assert len(res.oof_rows) == int((~treino).sum())


def test_particao_degenerada_falha_alto():
    y, clusters, _ = _setup()
    todos = np.ones(len(y), dtype=bool)
    with pytest.raises(ValueError, match="degenerada"):
        evaluate_temporal_pair("m", "s", y, clusters, todos, 3, [0, 1, 2],
                               _factory([]), _Caps(), _cal_cfg(), n_inner=3)
    with pytest.raises(ValueError, match="degenerada"):
        evaluate_temporal_pair("m", "s", y, clusters, ~todos, 3, [0, 1, 2],
                               _factory([]), _Caps(), _cal_cfg(), n_inner=3)


def test_rotulo_da_particao_vai_para_as_linhas():
    y, clusters, ano = _setup()
    res = evaluate_temporal_pair("m", "s", y, clusters, (ano < 2024).to_numpy(), 3,
                                 [0, 1, 2], _factory([]), _Caps(), _cal_cfg(), n_inner=3,
                                 fold_id="holdout_2024")
    assert {r["outer_fold"] for r in res.oof_rows} == {"holdout_2024"}
    assert {r["outer_fold"] for r in res.metric_rows} == {"holdout_2024"}


def test_mascara_anulavel_nao_pode_virar_object():
    """Regressão do bug que foi a produção em 2026-08-03.

    `ano` vem em dtype anulável; `(ano < 2024).to_numpy()` devolve dtype **object**, e `~`
    sobre object faz complemento de dois (True -> -2), não negação — o contador de teste
    saiu negativo (-1.539.974). O `dtype=bool` é o que impede isso.
    """
    ano = pd.Series([2022, 2023, 2024, 2025], dtype="Int64")

    ingenuo = (ano < 2024).to_numpy()
    assert ingenuo.dtype == object
    assert (~ingenuo).sum() != 2  # o valor errado que o bug produzia

    correto = (ano < 2024).to_numpy(dtype=bool)
    assert correto.dtype == bool
    assert int((~correto).sum()) == 2


def test_a_refatoracao_preservou_a_validacao_espacial():
    # `evaluate_pair` passou a delegar a `_evaluate_split`; continua produzindo uma
    # avaliação por dobra, cobrindo todos os registros exatamente uma vez.
    y, clusters, _ = _setup()
    outer = pd.Series(np.resize([0, 1, 2, 3, 4], len(y)))
    res = evaluate_pair("m", "s", y, clusters, outer, 3, [0, 1, 2], _factory([]),
                        _Caps(), _cal_cfg(), n_inner=3)
    assert len(res.oof_rows) == len(y)
    assert {r["outer_fold"] for r in res.oof_rows} == {0, 1, 2, 3, 4}
