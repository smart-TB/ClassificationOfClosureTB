"""Utilidade programática (ESPECIFICACAO §3.8).

O que estes testes protegem é a aritmética que sustenta a decisão do programa: quantos
alertas a equipe recebe, quanto do que ela visita é evento de verdade, quanto dos eventos
ela alcança, e se tudo isso bate "intervir em todos" / "não intervir em ninguém".
"""
import numpy as np
import pytest

from tb_outcomes.utility import (
    alerts_per_1000,
    decision_curve,
    net_benefit,
    topk_table,
)


def test_alertas_por_mil_e_so_uma_taxa():
    pred = np.array([1] * 120 + [0] * 880)
    assert alerts_per_1000(pred) == pytest.approx(120.0)
    # meia coorte do mesmo tamanho relativo dá a mesma taxa
    assert alerts_per_1000(np.array([1] * 60 + [0] * 440)) == pytest.approx(120.0)


def test_topk_ppv_e_captura_com_ordenacao_perfeita():
    # 100 eventos no topo de 1000: no top 10% o PPV é 1,0 e a captura é 1,0
    proba = np.linspace(1, 0, 1000)
    y = np.zeros(1000, dtype=int)
    y[:100] = 1
    t = topk_table(y, proba, ks=(0.05, 0.10, 0.20)).set_index("k")

    assert t.loc[0.10, "ppv"] == pytest.approx(1.0)
    assert t.loc[0.10, "captura"] == pytest.approx(1.0)
    assert t.loc[0.10, "nns"] == pytest.approx(1.0)
    # no top 5% só cabe metade dos eventos: PPV segue 1,0, captura cai a 0,5
    assert t.loc[0.05, "ppv"] == pytest.approx(1.0)
    assert t.loc[0.05, "captura"] == pytest.approx(0.5)
    # no top 20% metade dos sinalizados é evento
    assert t.loc[0.20, "ppv"] == pytest.approx(0.5)
    assert t.loc[0.20, "captura"] == pytest.approx(1.0)


def test_topk_com_ordenacao_aleatoria_reproduz_a_prevalencia():
    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.10, 20000)
    proba = rng.random(20000)  # sem informação nenhuma
    t = topk_table(y, proba, ks=(0.10,)).iloc[0]
    assert t["ppv"] == pytest.approx(0.10, abs=0.02)
    assert t["captura"] == pytest.approx(0.10, abs=0.02)
    assert t["nns"] == pytest.approx(10.0, rel=0.25)


def test_nns_e_o_inverso_do_ppv():
    proba = np.linspace(1, 0, 500)
    y = (np.arange(500) % 4 == 0).astype(int)
    t = topk_table(y, proba, ks=(0.2,)).iloc[0]
    assert t["nns"] == pytest.approx(1.0 / t["ppv"])


def test_net_benefit_de_intervir_em_todos_no_limiar_da_prevalencia():
    """Em pt = prevalência, 'intervir em todos' tem benefício líquido ZERO — é o ponto em
    que o ganho dos verdadeiros positivos empata com o custo dos falsos positivos."""
    y = np.array([1] * 100 + [0] * 900)
    todos = np.ones(1000)
    assert net_benefit(y, todos, 0.10) == pytest.approx(0.0, abs=1e-9)
    # abaixo da prevalência, intervir em todos compensa; acima, não
    assert net_benefit(y, todos, 0.05) > 0
    assert net_benefit(y, todos, 0.20) < 0


def test_net_benefit_de_nao_intervir_e_sempre_zero():
    y = np.array([1] * 100 + [0] * 900)
    nenhum = np.zeros(1000)
    for pt in (0.02, 0.1, 0.3, 0.6):
        assert net_benefit(y, nenhum, pt) == pytest.approx(0.0)


def test_modelo_perfeito_domina_as_duas_referencias():
    y = np.array([1] * 100 + [0] * 900)
    perfeito = y.astype(float)
    d = decision_curve(y, perfeito, thresholds=np.array([0.05, 0.1, 0.2, 0.4]))
    assert (d["nb_modelo"] >= d["nb_intervir_em_todos"] - 1e-12).all()
    assert (d["nb_modelo"] >= d["nb_nao_intervir"] - 1e-12).all()
    assert (d["nb_modelo"] > 0).all()


def test_modelo_sem_informacao_nao_supera_as_referencias():
    rng = np.random.default_rng(3)
    y = rng.binomial(1, 0.1, 20000)
    ruido = rng.random(20000)
    d = decision_curve(y, ruido, thresholds=np.array([0.15, 0.25, 0.40]))
    # acima da prevalência, um escore sem informação não deve ganhar de não intervir
    assert (d["nb_modelo"] <= 0.005).all()


def test_decision_curve_traz_as_tres_curvas_e_o_limiar():
    y = np.array([1] * 50 + [0] * 450)
    d = decision_curve(y, np.linspace(1, 0, 500), thresholds=np.array([0.1, 0.2]))
    assert list(d.columns) == ["threshold", "nb_modelo", "nb_intervir_em_todos",
                              "nb_nao_intervir", "n_alertas", "alertas_por_1000"]
    assert len(d) == 2
