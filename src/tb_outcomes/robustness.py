"""Robustez do leaderboard à granularidade espacial k (SP4g).

O valor de leaderboard de um modelo num k é a MELHOR estratégia (maior média entre
as dobras externas) na métrica global escolhida — linhas `kind=='aggregate'`,
`class=='__global__'`. Kendall τ e o overlap do top-N medem a estabilidade do
ranking entre granularidades.
"""
from __future__ import annotations

import pandas as pd
from scipy.stats import kendalltau


def leaderboard(metrics_df: pd.DataFrame, metric: str = "f1_macro",
                models=None) -> pd.DataFrame:
    df = metrics_df[
        (metrics_df["kind"] == "aggregate")
        & (metrics_df["metric"] == metric)
        & (metrics_df["class"] == "__global__")
    ].copy()
    if models is not None:
        df = df[df["model"].isin(models)]
    best = (df.sort_values("mean", ascending=False)
              .groupby("model", as_index=False).first())
    return (best[["model", "strategy", "mean"]]
            .rename(columns={"strategy": "best_strategy", "mean": metric})
            .reset_index(drop=True))


def robustness_table(metrics_by_k: dict, metric: str = "f1_macro",
                     models=None) -> pd.DataFrame:
    frames = []
    for k in sorted(metrics_by_k):
        lb = leaderboard(metrics_by_k[k], metric, models)
        lb["k"] = k
        lb["rank"] = lb[metric].rank(ascending=False, method="min").astype(int)
        frames.append(lb)
    return pd.concat(frames, ignore_index=True)


def rank_stability(metrics_by_k: dict, metric: str = "f1_macro", models=None,
                   top_n: int = 5) -> pd.DataFrame:
    order = {}
    for k in sorted(metrics_by_k):
        lb = leaderboard(metrics_by_k[k], metric, models).sort_values(
            metric, ascending=False)
        order[k] = lb["model"].tolist()
    ks = sorted(order)
    rows = []
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            a, b = ks[i], ks[j]
            common = [m for m in order[a] if m in order[b]]
            ra = [order[a].index(m) for m in common]
            rb = [order[b].index(m) for m in common]
            tau, p = kendalltau(ra, rb)
            top_a, top_b = set(order[a][:top_n]), set(order[b][:top_n])
            rows.append({
                "k_a": a, "k_b": b, "n_models": len(common),
                "kendall_tau": float(tau), "p_value": float(p),
                f"top{top_n}_overlap": len(top_a & top_b) / float(top_n),
            })
    return pd.DataFrame(rows)
