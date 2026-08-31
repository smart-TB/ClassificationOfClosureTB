"""SHAP consensual restrito (ESPECIFICACAO §4.2).

O painel principal explica **um** modelo — o campeão do leaderboard — e não o consenso
dos 25, que a §4.2 rebaixa a análise de sensibilidade. O motivo é que uma média de
importâncias entre famílias com escalas e geometrias diferentes não tem interpretação
definida; o consenso mede concordância, não efeito.

Duas decisões de protocolo:

1. **Explicar out-of-fold.** Para cada dobra espacial externa, o modelo é reajustado no
   DEV daquela dobra e explica registros do EVAL — nunca registros que ele viu. É a mesma
   disciplina da tabela principal; um refit global explicaria o próprio treino e inflaria
   a importância das variáveis memorizadas.

2. **Amostra estratificada por classe E por território.** A classe minoritária (`tb_death`)
   some numa amostra aleatória simples, e o território é justamente a dimensão sob suspeita
   (§3.3). Amostrar por dobra × classe garante as duas.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# LightGBM/CatBoost recebem o frame com categóricas nativas, então os nomes das colunas
# do SHAP JÁ são os nomes das features — sem expansão one-hot para desfazer depois.
EXPLAINER_BY_FAMILY = {
    "boosting": "TreeExplainer",
    "tree": "TreeExplainer",
}


def normalize_shap_output(sv, n_rows: int, n_features: int, n_classes: int) -> np.ndarray:
    """Devolve sempre (n_classes, n_rows, n_features).

    O shap mudou a forma de saída multiclasse entre versões: lista de arrays (n,f) por
    classe nas antigas, array (n,f,c) nas novas. Normalizar aqui evita que a diferença
    vaze para a agregação — e o teste de forma abaixo falha alto se aparecer uma terceira.
    """
    a = np.array(sv) if isinstance(sv, list) else np.asarray(sv)
    if a.shape == (n_classes, n_rows, n_features):
        return a
    if a.shape == (n_rows, n_features, n_classes):
        return np.transpose(a, (2, 0, 1))
    raise ValueError(
        f"forma de SHAP inesperada {a.shape}; esperado "
        f"({n_classes},{n_rows},{n_features}) ou ({n_rows},{n_features},{n_classes})"
    )


def stratified_sample(y, folds, per_cell: int, seed: int) -> np.ndarray:
    """Índices amostrados por (dobra externa × classe), até `per_cell` de cada célula.

    Célula com menos que `per_cell` entra inteira — não se reamostra com reposição, que
    fabricaria estabilidade artificial entre sementes.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    folds = np.asarray(folds)
    escolhidos = []
    for f in np.unique(folds):
        for c in np.unique(y):
            idx = np.where((folds == f) & (y == c))[0]
            if len(idx) == 0:
                continue
            take = idx if len(idx) <= per_cell else rng.choice(idx, per_cell, replace=False)
            escolhidos.append(take)
    return np.sort(np.concatenate(escolhidos)) if escolhidos else np.array([], dtype=int)


def summarize(values: np.ndarray, feature_names, class_names, extra: dict | None = None
              ) -> pd.DataFrame:
    """Importância e DIREÇÃO por classe × feature.

    `mean_abs_shap` ordena (magnitude do efeito); `mean_shap` dá o sinal médio, que é o que
    a §4.2 chama de direção. Reportar só a magnitude esconde se a variável empurra para a
    classe ou contra ela.
    """
    linhas = []
    for j, classe in enumerate(class_names):
        v = values[j]
        for i, feat in enumerate(feature_names):
            col = v[:, i]
            linhas.append({
                "classe": classe, "feature": feat,
                "mean_abs_shap": float(np.abs(col).mean()),
                "mean_shap": float(col.mean()),
                "std_shap": float(col.std(ddof=0)),
                "pct_positivo": float((col > 0).mean()),
                **(extra or {}),
            })
    d = pd.DataFrame(linhas)
    d["rank"] = d.groupby("classe")["mean_abs_shap"].rank(ascending=False, method="min").astype(int)
    return d.sort_values(["classe", "rank"]).reset_index(drop=True)


def stability(summaries: list[pd.DataFrame], top_n: int = 15) -> pd.DataFrame:
    """Concordância do ranking de importância entre amostras/sementes (§4.2).

    Spearman sobre o ranking completo + overlap do top-N. Se isto for baixo, o painel
    principal não é reportável: a "importância" seria ruído de amostragem.
    """
    from scipy.stats import spearmanr

    linhas = []
    rotulos = [s["_run"].iloc[0] for s in summaries]
    for i in range(len(summaries)):
        for j in range(i + 1, len(summaries)):
            a, b = summaries[i], summaries[j]
            for classe in sorted(set(a["classe"]) & set(b["classe"])):
                ga = a[a["classe"] == classe].set_index("feature")["mean_abs_shap"]
                gb = b[b["classe"] == classe].set_index("feature")["mean_abs_shap"]
                comuns = ga.index.intersection(gb.index)
                rho, p = spearmanr(ga[comuns], gb[comuns])
                ta = set(ga[comuns].sort_values(ascending=False).head(top_n).index)
                tb = set(gb[comuns].sort_values(ascending=False).head(top_n).index)
                linhas.append({
                    "run_a": rotulos[i], "run_b": rotulos[j], "classe": classe,
                    "n_features": int(len(comuns)),
                    "spearman": float(rho), "p_value": float(p),
                    f"top{top_n}_overlap": len(ta & tb) / float(top_n),
                })
    return pd.DataFrame(linhas)
