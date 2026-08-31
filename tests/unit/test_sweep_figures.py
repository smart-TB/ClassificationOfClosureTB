import pandas as pd

from tb_outcomes.figures import fig_leaderboard_across_k, fig_rank_bump


def _robust_long():
    rows = []
    data = {
        "lightgbm": {27: 0.54, 50: 0.55, 75: 0.53},
        "catboost": {27: 0.51, 50: 0.50, 75: 0.52},
        "majority_class": {27: 0.29, 50: 0.29, 75: 0.29},
    }
    for model, byk in data.items():
        for k, v in byk.items():
            rank = 1 if model == "lightgbm" else (2 if model == "catboost" else 3)
            rows.append({"model": model, "k": k, "f1_macro": v, "rank": rank})
    return pd.DataFrame(rows)


def test_leaderboard_across_k_writes_png(tmp_path, monkeypatch):
    import tb_outcomes.figures as figs
    monkeypatch.setattr(figs, "FIG_DIR", tmp_path)
    p = fig_leaderboard_across_k(_robust_long())
    assert p.exists() and p.suffix == ".png"


def test_rank_bump_writes_png(tmp_path, monkeypatch):
    import tb_outcomes.figures as figs
    monkeypatch.setattr(figs, "FIG_DIR", tmp_path)
    p = fig_rank_bump(_robust_long())
    assert p.exists() and p.suffix == ".png"


def _robust_long_irregular():
    """Cobertura irregular de k: um par que falhou num k desaparece do robustness_table,
    então nem todo modelo tem linha em TODOS os k (ft_transformer só em 27 e 50; tabnet
    entra só no 75). As figuras não podem quebrar nem contar ranks a menos."""
    df = _robust_long()
    extra = pd.DataFrame([
        {"model": "ft_transformer", "k": 27, "f1_macro": 0.50, "rank": 4},
        {"model": "ft_transformer", "k": 50, "f1_macro": 0.49, "rank": 4},
        {"model": "tabnet", "k": 75, "f1_macro": 0.48, "rank": 4},
    ])
    return pd.concat([df, extra], ignore_index=True)


def test_leaderboard_across_k_tolerates_irregular_k_coverage(tmp_path, monkeypatch):
    import tb_outcomes.figures as figs
    monkeypatch.setattr(figs, "FIG_DIR", tmp_path)
    p = fig_leaderboard_across_k(_robust_long_irregular())
    assert p.exists() and p.suffix == ".png"


def test_rank_bump_tolerates_irregular_k_coverage(tmp_path, monkeypatch):
    import tb_outcomes.figures as figs
    monkeypatch.setattr(figs, "FIG_DIR", tmp_path)
    p = fig_rank_bump(_robust_long_irregular())
    assert p.exists() and p.suffix == ".png"
