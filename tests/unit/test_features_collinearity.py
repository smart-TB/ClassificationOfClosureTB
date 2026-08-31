import numpy as np
import pandas as pd

from tb_outcomes.features import correlation_report, vif_report


def test_vif_is_na_for_constant_column_never_zero():
    # A origem dos 'VIF 0,000' do manuscrito: coluna constante.
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "const": [1.0, 1.0, 1.0, 1.0]})
    r = vif_report(X).set_index("column")
    assert pd.isna(r.loc["const", "vif"]), "constante deve ser NA"
    assert r.loc["const", "reason"] == "constant"
    assert not pd.isna(r.loc["a", "vif"])


def test_vif_detects_perfect_collinearity():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [2.0, 4.0, 6.0, 8.0]})
    r = vif_report(X).set_index("column")
    assert pd.isna(r.loc["a", "vif"]) or r.loc["a", "vif"] > 100


def test_vif_is_low_for_independent_columns():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=500), "b": rng.normal(size=500)})
    r = vif_report(X).set_index("column")
    assert r.loc["a", "vif"] < 2


def test_vif_never_returns_zero():
    # VIF mínimo teórico é 1. Zero é sinal de degeneração, nunca resultado.
    rng = np.random.default_rng(1)
    X = pd.DataFrame(
        {
            "a": rng.normal(size=200),
            "b": rng.normal(size=200),
            "const": np.ones(200),
        }
    )
    r = vif_report(X)
    assert not (r.vif == 0).any()


def test_vif_actually_computes_something():
    # ESTE é o teste que faltava. 'Sem zeros' passa trivialmente num relatório
    # inteiramente NA — foi o que aconteceu no dado real: o dropna de caso
    # completo sobre 94 colunas zerou 1,3 milhão de linhas e o relatório saiu
    # vazio, mas verde.
    rng = np.random.default_rng(2)
    X = pd.DataFrame({"a": rng.normal(size=300), "b": rng.normal(size=300)})
    r = vif_report(X)
    assert r.vif.notna().any(), "relatório inteiramente NA não é relatório"
    assert (r.reason == "ok").any()


def test_vif_excludes_columns_with_too_much_missing_instead_of_killing_the_matrix():
    # Uma coluna quase toda ausente derrubaria todas as linhas no caso completo.
    # Ela sai com motivo declarado; as demais continuam computáveis.
    rng = np.random.default_rng(3)
    n = 300
    quase_vazia = np.full(n, np.nan)
    quase_vazia[:5] = rng.normal(size=5)
    X = pd.DataFrame(
        {"a": rng.normal(size=n), "b": rng.normal(size=n), "rara": quase_vazia}
    )
    r = vif_report(X, max_missing_fraction=0.30).set_index("column")
    assert pd.isna(r.loc["rara", "vif"])
    assert r.loc["rara", "reason"].startswith("too_much_missing")
    assert not pd.isna(r.loc["a", "vif"]), "'a' e 'b' continuam computáveis"
    assert r.loc["a", "n_rows_used"] == n


def test_vif_reports_how_many_rows_supported_each_estimate():
    # Sem n_rows_used, um resultado vazio é indistinguível de um resultado.
    rng = np.random.default_rng(4)
    X = pd.DataFrame({"a": rng.normal(size=150), "b": rng.normal(size=150)})
    r = vif_report(X)
    assert "n_rows_used" in r.columns
    assert (r.n_rows_used == 150).all()


def test_correlation_report_is_long_format_without_self_pairs():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [3.0, 2.0, 1.0]})
    r = correlation_report(X)
    assert len(r) == 1, "um par, sem duplicar nem incluir a diagonal"
    assert abs(r.iloc[0].correlation + 1.0) < 1e-9
