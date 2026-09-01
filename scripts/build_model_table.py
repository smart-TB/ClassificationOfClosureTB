"""Tabela descritiva dos modelos (§2.10): família, implementação e hiperparâmetros.

Os hiperparâmetros são lidos dos estimadores REALMENTE instanciados pelo registry, e não
transcritos da documentação das bibliotecas. Como o protocolo roda tudo com o padrão de
biblioteca, a tabela documenta exatamente o que foi executado; se uma versão de biblioteca
mudar um padrão, a tabela acompanha.

Os modelos de aprendizado profundo não expõem um estimador sklearn: o orçamento de treino
deles é pré-registrado em configs/deep_learning.yaml e é isso que a tabela reporta.

Saída: data/model_description.csv
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import yaml

from tb_outcomes.labels import algoritmo
from tb_outcomes.executor import load_executor_config
from tb_outcomes.models import build_registry, load_models_config

# Parâmetros que definem a capacidade do modelo, por família. Mostrar get_params() inteiro
# daria dezenas de colunas irrelevantes; mostrar nenhum deixaria a tabela sem informação.
CHAVES = {
    "boosting": ["n_estimators", "iterations", "learning_rate", "max_depth", "num_leaves",
                 "depth", "max_iter", "subsample"],
    "tree": ["n_estimators", "criterion", "max_depth", "min_samples_split",
             "min_samples_leaf", "max_features"],
    "linear": ["C", "penalty", "solver", "max_iter", "alpha"],
    "discriminant": ["solver", "shrinkage", "reg_param"],
    "nb": ["alpha", "var_smoothing", "binarize", "fit_prior"],
    "neighbors": ["n_neighbors", "weights", "metric", "p"],
    "neural net": ["hidden_layer_sizes", "activation", "alpha", "max_iter", "solver"],
    "baseline": ["strategy"],
}
FAMILIA_PT = {
    "boosting": "Boosting", "tree": "Árvore", "linear": "Linear",
    "discriminant": "Discriminante", "nb": "Naive Bayes", "neighbors": "Vizinhos",
    "neural net": "Rede neural", "deep learning": "Aprendizado profundo",
    "baseline": "Referência",
}
FAMILIA_EN = {
    "boosting": "Boosting", "tree": "Tree ensemble", "linear": "Linear",
    "discriminant": "Discriminant", "nb": "Naive Bayes", "neighbors": "Neighbours",
    "neural net": "Neural network", "deep learning": "Deep learning",
    "baseline": "Baseline",
}
PERFIL = {
    "onehot_scaled": "One-hot, standardised",
    "onehot_unscaled": "One-hot",
    "native_categorical": "Native categorical",
    "nonnegative": "One-hot, non-negative",
    "binary": "Binarised",
    "dl_frame": "Raw frame (deep learning)",
}


def _fmt(v) -> str:
    if v is None:
        return "None"
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def main() -> None:
    cfg = load_models_config("configs/models.yaml")
    entries = {**cfg.baselines, **cfg.models}
    registry = build_registry(cfg)
    # os dois SVC estão no registry mas foram excluídos da grade antes do benchmark;
    # a tabela precisa dizê-lo, senão contradiz o texto, que fala em 26 avaliados.
    avaliados = set(load_executor_config("configs/executor.yaml").models)
    dl = yaml.safe_load(Path("configs/deep_learning.yaml").read_text(encoding="utf-8"))

    linhas = []
    for nome, entry in entries.items():
        adapter = registry.get(nome)
        fam = entry.family
        est = getattr(adapter, "_est", None)

        if est is not None:
            lib = type(est).__module__.split(".")[0]
            # "Classifier" é sufixo de toda classe sklearn e só consome largura
            cls = re.sub(r"Classifier$", "", type(est).__name__)
            params = est.get_params()
            chaves = CHAVES.get(fam, [])
            partes = [f"{k}={_fmt(params[k])}" for k in chaves if k in params]
            hp = "; ".join(partes) if partes else "library defaults"
        else:
            # DL: o orçamento é pré-registrado, não vem de get_params()
            lib, cls = "pytorch-tabular", entry.family
            hp = (f"max\\_epochs={dl['max_epochs']}; "
                  f"early\\_stopping\\_patience={dl['early_stopping_patience']}; "
                  f"batch\\_size={dl['batch_size']}; "
                  f"validation\\_split={dl['validation_split']}")

        caps = adapter.capabilities
        linhas.append({
            "model": nome,
            "Algorithm": algoritmo(nome),
            "Family": FAMILIA_EN.get(fam, fam),
            "Familia": FAMILIA_PT.get(fam, fam),
            "Library": lib,
            "Class": cls,
            "Input": PERFIL.get(caps.preprocess_profile, caps.preprocess_profile),
            "Native probability": "yes" if caps.native_probability else "no",
            "Sample weight": "yes" if caps.sample_weight else "no",
            "Hyperparameters": hp,
            "Evaluated": "yes" if nome in avaliados else "excluded a priori",
        })

    d = pd.DataFrame(linhas)
    ordem = {"Baseline": 0, "Tree ensemble": 1, "Boosting": 2, "Linear": 3,
             "Discriminant": 4, "Naive Bayes": 5, "Neighbours": 6, "Neural network": 7,
             "Deep learning": 8}
    d = d.sort_values(["Family", "Algorithm"],
                      key=lambda s: s.map(ordem) if s.name == "Family" else s)
    out = Path("data/model_description.csv")
    d.to_csv(out, index=False)
    print(f"gravado: {out} ({len(d)} modelos)")
    print(d[["Algorithm", "Family", "Library", "Class"]].head(8).to_string(index=False))


if __name__ == "__main__":
    main()
