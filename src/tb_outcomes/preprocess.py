"""Fábrica de pré-processadores por família de modelo.

Cada perfil é um sklearn Pipeline ajustado SOMENTE no treino da dobra (§3
princípio 2). O contrato fit/transform do sklearn é a barreira de vazamento: fit
só enxerga o argumento que recebe. A proteção de que o executor passa o split
CERTO é do SP4e — aqui garante-se apenas que o objeto é bem-comportado.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class ContextualPCAConfig(BaseModel):
    enabled: bool
    n_components: int
    sign_reference: str


class PreprocessConfig(BaseModel):
    missing_token: str
    unknown_token: str
    max_cardinality: int
    contextual_pca: ContextualPCAConfig


def load_preprocess_config(path: Path) -> PreprocessConfig:
    with Path(path).open("r", encoding="utf-8") as fh:
        return PreprocessConfig.model_validate(yaml.safe_load(fh))


@dataclass
class ColumnTypes:
    categorical: list[str]
    numeric: list[str]


def infer_column_types(X: pd.DataFrame, specs, max_cardinality: int) -> ColumnTypes:
    """Tipa cada coluna do X a partir do features.yaml + regra de cardinalidade.

    Nada de heurística silenciosa: o 'type' do FeatureSpec manda; a cardinalidade
    só desempata as derivadas (IDADE, *_BIN) que não têm spec.
    """
    por_nome = {s.raw_name: s for s in specs}
    categorical, numeric = [], []
    for c in X.columns:
        spec = por_nome.get(c)
        eh_num = pd.api.types.is_float_dtype(X[c]) or (
            pd.api.types.is_integer_dtype(X[c]) and X[c].nunique(dropna=True) > 20
        )
        if spec is not None:
            (numeric if spec.type == "numeric" else categorical).append(c)
        elif eh_num:
            numeric.append(c)
        else:
            categorical.append(c)
    return ColumnTypes(categorical=categorical, numeric=numeric)


def coerce_for_sklearn(
    X: pd.DataFrame, types: ColumnTypes, missing_token: str = "MISSING"
) -> pd.DataFrame:
    """Converte os dtypes anuláveis para o que o sklearn aceita.

    OneHotEncoder e StandardScaler quebram com Int64/string anuláveis
    ('boolean value of NA is ambiguous'). Categórica -> object com o token no
    lugar do <NA>; numérica -> float64 com np.nan.
    """
    X = X.copy()
    for c in types.categorical:
        col = X[c].astype("object")
        X[c] = col.where(col.notna(), missing_token).astype(str)
    for c in types.numeric:
        X[c] = pd.to_numeric(X[c], errors="coerce").astype("float64")
    return X


class PreprocessError(Exception):
    """Uma coluna chega ao preprocess num estado que produziria vazamento silencioso."""


def assert_sane_columns(X_train: pd.DataFrame, types: ColumnTypes, max_cardinality: int) -> None:
    """Falha se alguma coluna do treino da dobra for degenerada.

    Constante ou toda-<NA> viraria uma coluna de zeros no one-hot / NaN no scaler,
    que o modelo aprende como ruído em silêncio. Cardinalidade explosiva denuncia
    texto livre que escapou da classificação (o caso AGRAVOUTRA do SP2). Roda por
    dobra: uma coluna constante no treino de UMA dobra (mas não na base inteira)
    é onde o vazamento silencioso se esconde.
    """
    problemas = []
    for c in types.numeric + types.categorical:
        col = X_train[c]
        n_unicos = col.nunique(dropna=True)
        if col.isna().all():
            problemas.append(f"{c}: toda-<NA> no treino desta dobra")
        elif n_unicos <= 1:
            problemas.append(f"{c}: constante no treino desta dobra ({n_unicos} valor)")
        elif c in types.categorical and n_unicos > max_cardinality:
            problemas.append(
                f"{c}: cardinalidade {n_unicos} > {max_cardinality} (texto livre?)"
            )
    if problemas:
        raise PreprocessError(
            "colunas degeneradas no treino da dobra: " + "; ".join(problemas)
        )


# ---------------------------------------------------------------------------
# Blocos de transformação (numérico e categórico)
# ---------------------------------------------------------------------------

from sklearn.impute import MissingIndicator, SimpleImputer  # noqa: E402
from sklearn.pipeline import FeatureUnion, Pipeline  # noqa: E402
from sklearn.preprocessing import (  # noqa: E402
    MinMaxScaler,
    OneHotEncoder,
    StandardScaler,
)


def numeric_pipeline(scale: bool, nonnegative: bool) -> Pipeline:
    """Imputação por mediana + indicador de ausência + escala opcional.

    O indicador de ausência é feature (§11.3). Usamos MissingIndicator(features='all')
    em FeatureUnion, não SimpleImputer(add_indicator=True): este último só cria o
    indicador para colunas com <NA> NO TREINO, e o shape de saída variaria entre
    dobras. features='all' dá um indicador por coluna sempre — shape fixo, exigência
    do SP4e. A escala vem depois da imputação, sobre os valores imputados; o
    indicador não é escalado. nonnegative usa MinMax [0,1] (clip=True) para MNB/CNB.
    """
    imputa_e_escala = [("imp", SimpleImputer(strategy="median"))]
    if nonnegative:
        imputa_e_escala.append(("scale", MinMaxScaler(clip=True)))
    elif scale:
        imputa_e_escala.append(("scale", StandardScaler()))

    return Pipeline(
        [
            (
                "union",
                FeatureUnion(
                    [
                        ("valores", Pipeline(imputa_e_escala)),
                        ("ausencia", MissingIndicator(features="all")),
                    ]
                ),
            )
        ]
    )


def categorical_encoder(kind: str):
    """Codificação categórica. 'onehot' binariza; 'passthrough' mantém tokens (nativo)."""
    if kind == "onehot":
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    if kind == "passthrough":
        return "passthrough"  # CatBoost/DL recebem os tokens como estão
    raise ValueError(f"codificação categórica desconhecida: {kind}")


# ---------------------------------------------------------------------------
# Os cinco perfis (briefing §11.2)
# ---------------------------------------------------------------------------

from sklearn.compose import ColumnTransformer  # noqa: E402

PROFILES = ["onehot_scaled", "onehot_unscaled", "native_categorical", "nonnegative", "binary"]

# (escala numérica, nonnegative, codificação categórica)
_PROFILE_SPEC = {
    "onehot_scaled": (True, False, "onehot"),
    "onehot_unscaled": (False, False, "onehot"),
    "native_categorical": (False, False, "passthrough"),
    "nonnegative": (False, True, "onehot"),
    "binary": (False, False, "onehot"),  # numéricas binarizadas abaixo
}


def make_preprocessor(profile: str, types: ColumnTypes, cfg) -> ColumnTransformer:
    """Monta o pré-processador do perfil como ColumnTransformer.

    fit no treino, transform no resto — o contrato do sklearn é a barreira de
    vazamento. O binary do BernoulliNB binariza as numéricas pela mediana da
    dobra (KBinsDiscretizer, 2 bins), não reutiliza o perfil do MNB.
    """
    if profile not in _PROFILE_SPEC:
        raise ValueError(f"perfil desconhecido: {profile}")
    scale, nonneg, cat_kind = _PROFILE_SPEC[profile]

    if profile == "binary":
        from sklearn.preprocessing import KBinsDiscretizer

        num_tf = Pipeline(
            [
                ("imp", SimpleImputer(strategy="median")),
                (
                    "bin",
                    KBinsDiscretizer(n_bins=2, encode="ordinal", strategy="quantile"),
                ),
            ]
        )
    else:
        num_tf = numeric_pipeline(scale=scale, nonnegative=nonneg)

    transformers = []
    if types.numeric:
        transformers.append(("num", num_tf, types.numeric))
    if types.categorical:
        transformers.append(("cat", categorical_encoder(cat_kind), types.categorical))

    return ColumnTransformer(
        transformers, remainder="drop", verbose_feature_names_out=False
    )


# ---------------------------------------------------------------------------
# Índice de vulnerabilidade socioeconômica (spec §6)
# ---------------------------------------------------------------------------

from sklearn.base import BaseEstimator, TransformerMixin  # noqa: E402


class VulnerabilityIndex(BaseEstimator, TransformerMixin):
    """PCA(1) sobre indicadores contextuais, ajustado na dobra (spec §6).

    Imputação, padronização e PCA aprendidos só no treino. O sinal é fixado pela
    correlação com o indicador de referência (IDHM), de modo que valores maiores
    indiquem maior vulnerabilidade. Transformador sklearn próprio, serializável
    por dobra (§25.4). Corrige a versão do SP2 que ajustava sobre a base inteira.
    """

    def __init__(self, sign_reference: str = "IPEA_IDHM"):
        self.sign_reference = sign_reference

    def fit(self, X, y=None):
        import numpy as np
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        self.columns_ = list(X.columns)
        self.imputer_ = SimpleImputer(strategy="median").fit(X)
        Xi = self.imputer_.transform(X)
        self.scaler_ = StandardScaler().fit(Xi)
        Xs = self.scaler_.transform(Xi)
        self.pca_ = PCA(n_components=1).fit(Xs)

        pc1 = self.pca_.transform(Xs)[:, 0]
        ref = Xi[:, self.columns_.index(self.sign_reference)]
        # maior IDHM = menos vulnerável: se pc1 correlaciona POSITIVO com IDHM, inverte
        self.sign_ = -1.0 if np.corrcoef(pc1, ref)[0, 1] > 0 else 1.0
        return self

    def transform(self, X):
        Xs = self.scaler_.transform(self.imputer_.transform(X))
        return (self.sign_ * self.pca_.transform(Xs)[:, [0]]).astype("float64")


def vulnerability_loadings(idx: VulnerabilityIndex, columns: list[str]) -> pd.DataFrame:
    """Peso de cada indicador no PC1 (com o sinal aplicado), para o manuscrito."""
    return pd.DataFrame(
        {"indicator": columns, "loading": idx.sign_ * idx.pca_.components_[0]}
    ).sort_values("loading", key=abs, ascending=False)


# ---------------------------------------------------------------------------
# Manifesto
# ---------------------------------------------------------------------------


def preprocessing_manifest(types: ColumnTypes, cfg) -> dict:
    """Perfis, colunas por tipo e opções — o registro do que o preprocess faz (§11.2)."""
    return {
        "profiles": PROFILES,
        "n_categorical": len(types.categorical),
        "n_numeric": len(types.numeric),
        "categorical": types.categorical,
        "numeric": types.numeric,
        "missing_token": cfg.missing_token,
        "unknown_token": cfg.unknown_token,
        "contextual_pca_enabled": cfg.contextual_pca.enabled,
    }
