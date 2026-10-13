"""Confronta a massa de atribuição SHAP do bloco municipal com o seu custo de remoção.

O manuscrito já afirma que atribuição dentro de um modelo ajustado não estabelece sinal
transferível. Esta tabela põe número na afirmação: quanto da |SHAP| total cai sobre as 43
variáveis municipais, contra quanto do F1 macro se perde ao removê-las todas.

As duas medidas respondem a perguntas diferentes por construção — o SHAP atribui com as
demais variáveis fixas, a ablação deixa o modelo reaprender com substitutos correlacionados —
e por isso divergir é esperado. O que a tabela mostra é a MAGNITUDE da divergência, e o braço
só-municipal parado no piso da classe majoritária, que é o que separa redundância de ausência
de sinal próprio.

Escreve `data/attribution_vs_ablation.csv`.

Uso:
    poetry run python scripts/build_attribution_vs_ablation.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from tb_outcomes.features import classify_feature_scope, load_feature_config

DATA = Path("data")
SAIDA = DATA / "attribution_vs_ablation.csv"


def massa_shap() -> pd.DataFrame:
    """Fração da |SHAP| total que cai sobre cada escopo, por classe."""
    d = pd.read_csv(DATA / "shap_summary.csv")
    d = d[d["_run"] == d["_run"].iloc[0]].copy()
    specs = load_feature_config(Path("configs/features.yaml"))
    d["escopo"] = d["feature"].map(
        classify_feature_scope(sorted(d["feature"].unique()), specs))

    total = d.groupby("classe")["mean_abs_shap"].sum()
    municipal = d[d["escopo"] == "municipal"].groupby("classe")["mean_abs_shap"].sum()
    top15 = (d[(d["rank"] <= 15) & (d["escopo"] == "municipal")]
             .groupby("classe").size())

    return pd.DataFrame({
        "shap_municipal_pct": (municipal / total * 100).round(2),
        "municipais_no_top15": top15.reindex(total.index).fillna(0).astype(int),
    }).reset_index()


def custo_ablacao() -> pd.DataFrame:
    """Quanto do F1 macro combinado sobrevive sem as variáveis municipais."""
    ab = pd.read_csv(DATA / "ablation" / "ablation_leaderboard.csv")
    return pd.DataFrame({
        "model": ab["model"],
        "f1_combinado": ab["combined"].round(4),
        "f1_so_individual": ab["individual"].round(4),
        "f1_so_municipal": ab["municipal"].round(4),
        "retido_sem_municipais_pct": (ab["pct_do_combined_individual"] * 100).round(2),
    })


def main() -> None:
    import sys as _sys
    _sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent / "editorial"))
    from guarda_historico import exigir_historico
    exigir_historico("build_attribution_vs_ablation.py", "python scripts/editorial/figuras_c7.py")
    shap = massa_shap()
    abl = custo_ablacao()

    # produto cartesiano deliberado: a atribuição é por CLASSE e a ablação é por MODELO.
    # Cruzá-las numa tabela só é o ponto — cada linha confronta uma medida com a outra.
    out = shap.merge(abl, how="cross")
    out.to_csv(SAIDA, index=False)

    print(f"{SAIDA} — {len(out)} linhas\n")
    print("massa de atribuição SHAP no bloco municipal, por classe:")
    print(shap.to_string(index=False))
    print("\ncusto de remover as 43 municipais, por modelo:")
    print(abl.to_string(index=False))


if __name__ == "__main__":
    main()
