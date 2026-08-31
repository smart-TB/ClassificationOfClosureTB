"""Dados numéricos por figura (ESPECIFICACAO §6).

O contrato: o CSV descreve o que foi DESENHADO. Se um valor está no gráfico, está no
arquivo — e com o mesmo número, não uma rederivação.
"""
import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tb_outcomes.figure_data import extract_figure  # noqa: E402


def test_extrai_linha_com_os_valores_exatos():
    fig, ax = plt.subplots()
    ax.plot([27, 50, 75], [0.4919, 0.4955, 0.4953], label="lightgbm")
    d = extract_figure(fig, "teste")
    plt.close(fig)

    linha = d[d.tipo == "linha"]
    assert list(linha["serie"].unique()) == ["lightgbm"]
    assert linha["x"].tolist() == [27.0, 50.0, 75.0]
    assert linha["y"].tolist() == pytest.approx([0.4919, 0.4955, 0.4953])


def test_extrai_barras_com_altura():
    fig, ax = plt.subplots()
    ax.bar(["a", "b", "c"], [10, 25, 7])
    d = extract_figure(fig, "teste")
    plt.close(fig)

    barras = d[d.tipo == "barra"].sort_values("ponto")
    assert barras["y"].tolist() == pytest.approx([10.0, 25.0, 7.0])
    # os rótulos categóricos do eixo x também são preservados
    ticks = d[d.tipo == "rotulo_tick_x"]["rotulo"].tolist()
    assert {"a", "b", "c"} <= set(ticks)


def test_extrai_heatmap_celula_a_celula():
    m = np.array([[0.8, 0.1], [0.3, 0.9]])
    fig, ax = plt.subplots()
    ax.imshow(m)
    d = extract_figure(fig, "teste")
    plt.close(fig)

    cel = d[d.tipo == "celula"]
    assert len(cel) == 4
    v = cel.set_index(["y", "x"])["valor"]
    assert v.loc[(0, 0)] == pytest.approx(0.8)
    assert v.loc[(1, 1)] == pytest.approx(0.9)


def test_extrai_dispersao():
    fig, ax = plt.subplots()
    ax.scatter([1, 2, 3], [4, 5, 6])
    d = extract_figure(fig, "teste")
    plt.close(fig)
    disp = d[d.tipo == "dispersao"]
    assert disp["x"].tolist() == pytest.approx([1.0, 2.0, 3.0])
    assert disp["y"].tolist() == pytest.approx([4.0, 5.0, 6.0])


def test_varios_eixos_ficam_distinguiveis():
    fig, (a1, a2) = plt.subplots(1, 2)
    a1.plot([0, 1], [10, 11])
    a2.plot([0, 1], [20, 21])
    d = extract_figure(fig, "teste")
    plt.close(fig)
    assert set(d["eixo"].unique()) == {0, 1}
    assert d[d.eixo == 0]["y"].max() == 11.0
    assert d[d.eixo == 1]["y"].max() == 21.0


def test_histograma_stepfilled_e_capturado():
    """`hist(histtype="stepfilled")` produz Polygon, não Rectangle — a fig 19 saía vazia."""
    fig, ax = plt.subplots()
    ax.hist(np.random.default_rng(0).normal(size=500), bins=20, histtype="stepfilled",
            label="classe A")
    d = extract_figure(fig, "teste")
    plt.close(fig)
    assert not d.empty
    assert "contorno" in d["tipo"].unique()
    assert "classe A" in d["serie"].unique()


def test_serie_longa_e_subamostrada_e_declarada():
    """Uma curva PR/ROC sobre a coorte tem ~617 mil vértices; o teto evita que o
    acompanhamento vire um segundo canal de microdado. O corte tem de ser DECLARADO."""
    from tb_outcomes.figure_data import MAX_PONTOS_POR_SERIE

    n = 50_000
    fig, ax = plt.subplots()
    ax.plot(np.linspace(0, 1, n), np.linspace(0, 1, n))
    d = extract_figure(fig, "teste")
    plt.close(fig)

    linha = d[d.tipo == "linha"]
    assert len(linha) <= MAX_PONTOS_POR_SERIE + 1
    assert bool(linha["subamostrada"].iloc[0]) is True
    assert int(linha["n_pontos_originais"].iloc[0]) == n
    # as pontas são preservadas: o leitor ainda vê onde a curva começa e termina
    assert linha["x"].iloc[0] == pytest.approx(0.0)
    assert linha["x"].iloc[-1] == pytest.approx(1.0)


def test_serie_curta_nao_e_marcada_como_subamostrada():
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [4, 5, 6])
    d = extract_figure(fig, "teste")
    plt.close(fig)
    linha = d[d.tipo == "linha"]
    assert len(linha) == 3
    assert not linha["subamostrada"].any()


def test_figura_vazia_devolve_quadro_vazio():
    fig, _ = plt.subplots()
    d = extract_figure(fig, "vazia")
    plt.close(fig)
    assert d.empty


def test_coluna_figura_identifica_o_arquivo():
    fig, ax = plt.subplots()
    ax.plot([1, 2], [3, 4])
    d = extract_figure(fig, "25_leaderboard_across_k")
    plt.close(fig)
    assert (d["figura"] == "25_leaderboard_across_k").all()


def test_save_grava_o_csv_ao_lado_da_figura(tmp_path, monkeypatch):
    """O gancho no _save é o que torna o acompanhamento automático para as 26 figuras."""
    import tb_outcomes.figures as figs

    monkeypatch.setattr(figs, "FIG_DIR", tmp_path)
    monkeypatch.setattr(figs, "FIG_DATA_DIR", tmp_path / "data")
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [0.1, 0.2, 0.3])
    p = figs._save(fig, "fig_teste")

    assert p.exists()
    csv = tmp_path / "data" / "fig_teste.csv"
    assert csv.exists()
    import pandas as pd
    d = pd.read_csv(csv)
    assert d["y"].tolist() == pytest.approx([0.1, 0.2, 0.3])


def test_falha_na_extracao_nao_impede_a_figura(tmp_path, monkeypatch):
    import tb_outcomes.figures as figs

    monkeypatch.setattr(figs, "FIG_DIR", tmp_path)
    monkeypatch.setattr(figs, "FIG_DATA_DIR", tmp_path / "data")

    def explode(*a, **k):
        raise RuntimeError("falha simulada na extração")

    monkeypatch.setattr("tb_outcomes.figure_data.extract_figure", explode)
    fig, ax = plt.subplots()
    ax.plot([1, 2], [3, 4])
    p = figs._save(fig, "fig_resiliente")
    assert p.exists()  # a figura saiu mesmo com a extração quebrada
