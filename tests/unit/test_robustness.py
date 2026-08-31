import pandas as pd

from tb_outcomes.robustness import leaderboard, rank_stability, robustness_table


def _metrics(order, metric="f1_macro"):
    """metrics_df sintético: linhas agregadas globais, 1 estratégia por modelo.
    `order` = lista de (model, mean) já como queremos o ranking."""
    rows = []
    for model, mean in order:
        rows.append({"kind": "aggregate", "model": model, "strategy": "random_oversampling",
                     "metric": metric, "class": "__global__", "stage": None, "axis": None,
                     "mean": mean})
        # uma estratégia pior, para testar o "melhor por modelo"
        rows.append({"kind": "aggregate", "model": model, "strategy": "random_undersampling",
                     "metric": metric, "class": "__global__", "stage": None, "axis": None,
                     "mean": mean - 0.05})
        # ruído: linha per_fold e outra métrica não devem entrar
        rows.append({"kind": "per_fold", "model": model, "strategy": "random_oversampling",
                     "metric": metric, "class": "__global__", "stage": None, "axis": None,
                     "mean": 0.0})
    return pd.DataFrame(rows)


def test_leaderboard_picks_best_strategy_global_aggregate():
    df = _metrics([("lightgbm", 0.55), ("catboost", 0.50)])
    lb = leaderboard(df).set_index("model")
    assert lb.loc["lightgbm", "best_strategy"] == "random_oversampling"
    assert abs(lb.loc["lightgbm", "f1_macro"] - 0.55) < 1e-9
    assert list(lb.sort_values("f1_macro", ascending=False).index) == ["lightgbm", "catboost"]


def test_robustness_table_long_with_rank_per_k():
    by_k = {
        27: _metrics([("lightgbm", 0.54), ("catboost", 0.51), ("majority_class", 0.29)]),
        50: _metrics([("lightgbm", 0.55), ("catboost", 0.50), ("majority_class", 0.29)]),
        75: _metrics([("lightgbm", 0.53), ("catboost", 0.52), ("majority_class", 0.29)]),
    }
    tab = robustness_table(by_k)
    lgb = tab[(tab["model"] == "lightgbm")].set_index("k")
    assert set(tab["k"]) == {27, 50, 75}
    assert (lgb["rank"] == 1).all()  # lightgbm é 1º em todo k


def test_rank_stability_perfect_when_order_identical():
    same = [("lightgbm", 0.55), ("catboost", 0.50), ("random_forest", 0.45),
            ("adaboost", 0.40), ("majority_class", 0.29)]
    by_k = {27: _metrics(same), 50: _metrics(same), 75: _metrics(same)}
    st = rank_stability(by_k, top_n=3)
    assert len(st) == 3  # pares (27,50),(27,75),(50,75)
    assert (st["kendall_tau"] > 0.999).all()
    assert (st["top3_overlap"] == 1.0).all()


def test_rank_stability_detects_swap():
    a = [("lightgbm", 0.55), ("catboost", 0.50)]
    b = [("catboost", 0.55), ("lightgbm", 0.50)]  # troca o topo
    st = rank_stability({27: _metrics(a), 75: _metrics(b)}, top_n=2)
    assert st.iloc[0]["kendall_tau"] < 0  # inversão perfeita -> tau negativo
