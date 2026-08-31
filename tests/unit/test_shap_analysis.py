"""SHAP consensual restrito (ESPECIFICACAO §4.2)."""
import numpy as np
import pandas as pd
import pytest

from tb_outcomes.shap_analysis import (
    normalize_shap_output,
    stability,
    stratified_sample,
    summarize,
)


def test_normaliza_as_duas_formas_do_shap():
    n, f, c = 7, 4, 3
    lista = [np.random.default_rng(i).normal(size=(n, f)) for i in range(c)]
    empilhado = np.transpose(np.array(lista), (1, 2, 0))  # (n, f, c)
    a = normalize_shap_output(lista, n, f, c)
    b = normalize_shap_output(empilhado, n, f, c)
    assert a.shape == b.shape == (c, n, f)
    assert np.allclose(a, b)


def test_forma_desconhecida_falha_alto():
    with pytest.raises(ValueError, match="forma de SHAP inesperada"):
        normalize_shap_output(np.zeros((2, 2)), 7, 4, 3)


def test_amostra_cobre_toda_celula_dobra_x_classe():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 5000)
    folds = rng.integers(0, 5, 5000)
    idx = stratified_sample(y, folds, per_cell=40, seed=1)
    d = pd.DataFrame({"y": y[idx], "f": folds[idx]})
    celulas = d.groupby(["f", "y"]).size()
    assert len(celulas) == 15  # 5 dobras × 3 classes, nenhuma ausente
    assert celulas.max() <= 40


def test_classe_rara_entra_inteira_sem_reposicao():
    y = np.array([0] * 1000 + [1] * 7)
    folds = np.zeros(1007, dtype=int)
    idx = stratified_sample(y, folds, per_cell=100, seed=3)
    assert len(np.unique(idx)) == len(idx)  # sem repetição
    assert (y[idx] == 1).sum() == 7  # a classe rara entrou toda


def test_amostra_muda_com_a_semente():
    y = np.random.default_rng(0).integers(0, 3, 3000)
    folds = np.zeros(3000, dtype=int)
    a = stratified_sample(y, folds, per_cell=50, seed=1)
    b = stratified_sample(y, folds, per_cell=50, seed=2)
    assert not np.array_equal(a, b)


def test_summarize_reporta_magnitude_e_direcao():
    # feature 0 empurra sempre a favor, feature 1 sempre contra, feature 2 é ruído
    n = 200
    v = np.zeros((1, n, 3))
    v[0, :, 0] = 0.5
    v[0, :, 1] = -0.9
    v[0, :, 2] = np.linspace(-0.01, 0.01, n)
    d = summarize(v, ["a", "b", "c"], ["cure"])
    linha = d.set_index("feature")
    assert linha.loc["b", "rank"] == 1  # maior magnitude
    assert linha.loc["b", "mean_shap"] < 0 and linha.loc["a", "mean_shap"] > 0
    assert linha.loc["a", "pct_positivo"] == pytest.approx(1.0)
    assert linha.loc["c", "rank"] == 3


def test_stability_detecta_concordancia_e_discordancia():
    feats = [f"f{i}" for i in range(20)]
    base = pd.DataFrame({"classe": "cure", "feature": feats,
                         "mean_abs_shap": np.linspace(1, 0.05, 20), "_run": "a"})
    igual = base.assign(_run="b")
    invertido = base.assign(mean_abs_shap=base["mean_abs_shap"].to_numpy()[::-1], _run="c")

    concorda = stability([base, igual], top_n=5).iloc[0]
    discorda = stability([base, invertido], top_n=5).iloc[0]
    assert concorda["spearman"] == pytest.approx(1.0)
    assert concorda["top5_overlap"] == pytest.approx(1.0)
    assert discorda["spearman"] == pytest.approx(-1.0)
    assert discorda["top5_overlap"] == pytest.approx(0.0)
