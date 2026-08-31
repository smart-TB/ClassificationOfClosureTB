"""Particionamento espacial: clusters municipais e dobras aninhadas.

A partição em que toda a validação da reanálise se apoia. Corrige o defeito
central que a ESPECIFICACAO §0 aponta — seleção no hold-out — tornando-o
impossível: nenhum cluster aparece em treino e avaliação ao mesmo tempo.

Não treina modelo. Define e congela as partições; o SP4 as consome.
"""
from __future__ import annotations

import hashlib
import logging
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pyproj
import yaml
from pydantic import BaseModel
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedGroupKFold

logger = logging.getLogger(__name__)

EPSG_LATLON = 4326

# Códigos de desfecho para os resumos (SINAN NET 5.0).
_CURA, _INTERRUP, _OBITO = 1, 2, 3


class SplitsConfig(BaseModel):
    epsg_metric: int
    cluster_level: str
    k_primary: int
    k_sensitivity: list[int]
    n_outer_folds: int
    n_inner_folds: int
    fold_assignment: str
    stratify_by: str
    min_classes_per_fold: int
    max_fold_size_ratio: float


def load_splits_config(path: Path) -> SplitsConfig:
    with Path(path).open("r", encoding="utf-8") as fh:
        return SplitsConfig.model_validate(yaml.safe_load(fh))


def project_to_metric(lat: np.ndarray, lon: np.ndarray, epsg: int) -> np.ndarray:
    """Projeta lat/long (EPSG 4326) para um sistema métrico, saída em km.

    EPSG 5880 (Brazil Polyconic) dá distâncias planas adequadas ao território
    brasileiro. Graus não servem: 1° de longitude vale ~110 km no equador e ~85
    km no Sul, o que distorceria o KMeans.
    """
    transformer = pyproj.Transformer.from_crs(EPSG_LATLON, epsg, always_xy=True)
    x, y = transformer.transform(lon, lat)  # always_xy: (lon, lat) -> (x, y)
    return np.column_stack([x, y]) / 1000.0


def build_municipality_table(df: pd.DataFrame, key: str = "ID_MUNIC_ANALISE") -> pd.DataFrame:
    """Uma linha por município, com lat/long e contagem de notificações.

    Exige coordenada única por município (§8.5): se um município tiver mais de
    uma lat/long, é defeito de dado, não algo a resolver por média.
    """
    sub = df[[key, "LAT_MUNIC", "LONG_MUNIC"]].dropna(subset=[key, "LAT_MUNIC", "LONG_MUNIC"])

    n_coord = sub.groupby(key)[["LAT_MUNIC", "LONG_MUNIC"]].nunique()
    multiplos = n_coord[(n_coord > 1).any(axis=1)]
    if len(multiplos):
        raise ValueError(
            f"{len(multiplos)} municípios com mais de uma coordenada (§8.5): "
            f"{list(multiplos.index[:5])}. Coordenada deve ser única por município."
        )

    tabela = sub.groupby(key).agg(
        lat=("LAT_MUNIC", "first"),
        lon=("LONG_MUNIC", "first"),
        n=("LAT_MUNIC", "size"),
    )
    logger.info("Tabela de municípios: %d municípios com coordenada.", len(tabela))
    return tabela


def fit_spatial_clusters(
    mun_table: pd.DataFrame, k: int, epsg: int, random_state: int
) -> pd.DataFrame:
    """KMeans nos municípios projetados. Peso igual por município (§12.1).

    O peso igual é a regra: ponderar por número de notificações faria São Paulo
    dominar a geometria, e o cluster deixaria de ser geográfico para virar
    'onde há mais casos'. n_init alto para estabilidade da atribuição.
    """
    xy = project_to_metric(mun_table["lat"].to_numpy(), mun_table["lon"].to_numpy(), epsg)
    km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = km.fit_predict(xy)
    out = pd.DataFrame({"cluster": labels}, index=mun_table.index)
    logger.info("Clusters k=%d: %d municípios, random_state=%d.", k, len(out), random_state)
    return out


def assignment_hash(clusters: pd.DataFrame) -> str:
    """Hash da atribuição município->cluster (§12.1). Ordena por índice antes."""
    ordenado = clusters.sort_index()
    return hashlib.sha256(ordenado["cluster"].to_json().encode("utf-8")).hexdigest()[:16]


def assign_clusters_to_records(
    df: pd.DataFrame, clusters: pd.DataFrame, key: str = "ID_MUNIC_ANALISE"
) -> pd.Series:
    """Associa o cluster de cada município de volta às notificações."""
    return df[key].map(clusters["cluster"]).rename("cluster")


def assign_clusters_binpack(cluster_sizes: pd.Series, n_folds: int) -> dict[int, int]:
    """Distribui clusters inteiros em dobras: maior cluster -> dobra mais vazia.

    Determinístico. A ordenação estável desempata pelo índice do cluster, então
    o hash é reproduzível. Isto substitui a agregação aleatória do
    StratifiedGroupKFold, que era a origem do desequilíbrio 18,1x (spec §2.1).
    """
    carga = [0] * n_folds
    atribuicao: dict[int, int] = {}
    for cluster, tamanho in cluster_sizes.sort_values(ascending=False, kind="stable").items():
        f = int(np.argmin(carga))
        atribuicao[int(cluster)] = f
        carga[f] += int(tamanho)
    return atribuicao


def make_outer_folds(
    df: pd.DataFrame,
    record_clusters: pd.Series,
    y: pd.Series,
    n_folds: int,
    method: str = "binpack",
) -> pd.Series:
    """Registro -> dobra externa, mantendo cada cluster inteiro numa dobra.

    method='binpack' (principal): dobras de tamanho igual (spec §2.1).
    method='random_group': StratifiedGroupKFold, para a sensibilidade que mostra
    que a ordenação dos modelos não depende do método de atribuição.
    """
    if method == "binpack":
        tamanhos = record_clusters.groupby(record_clusters).size()
        mapa = assign_clusters_binpack(tamanhos, n_folds)
        folds = record_clusters.map(mapa).rename("outer_fold")
    elif method == "random_group":
        y_int = y.astype(int).to_numpy()
        sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=42)
        folds = pd.Series(-1, index=df.index, name="outer_fold")
        for i, (_, te) in enumerate(sgkf.split(df, y_int, groups=record_clusters.to_numpy())):
            folds.iloc[te] = i
    else:
        raise ValueError(f"método de atribuição desconhecido: {method}")

    # Verificação inline do bloqueio espacial — barata e sempre vale a pena.
    por_cluster = pd.DataFrame({"cluster": record_clusters.to_numpy(), "fold": folds.to_numpy()})
    vazando = por_cluster.groupby("cluster").fold.nunique()
    if (vazando > 1).any():
        raise ValueError(
            f"vazamento: {int((vazando > 1).sum())} clusters em mais de uma dobra."
        )
    return folds


def make_inner_folds(
    outer_train_clusters: pd.Series, y: pd.Series, n_folds: int, method: str = "binpack"
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Dobras internas dentro de um outer_train. Geradas sob demanda, não vão a disco.

    Materializar seria 5 externas x 5 internas x 3 valores de k. Cada dobra
    interna respeita os mesmos clusters — nenhum cluster do outer_train aparece
    em treino e validação internos ao mesmo tempo.
    """
    idx = np.arange(len(outer_train_clusters))
    if method == "binpack":
        tamanhos = outer_train_clusters.groupby(outer_train_clusters).size()
        mapa = assign_clusters_binpack(tamanhos, n_folds)
        fold_de = outer_train_clusters.map(mapa).to_numpy()
        for f in range(n_folds):
            va = idx[fold_de == f]
            tr = idx[fold_de != f]
            if len(va):
                yield tr, va
    elif method == "random_group":
        sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=42)
        yield from sgkf.split(idx, y.astype(int).to_numpy(), groups=outer_train_clusters.to_numpy())
    else:
        raise ValueError(f"método desconhecido: {method}")


def cluster_summary(
    record_clusters: pd.Series, y: pd.Series, mun_table: pd.DataFrame
) -> pd.DataFrame:
    """Municípios, registros e eventos por cluster (§12.1)."""
    d = pd.DataFrame({"cluster": record_clusters.to_numpy(), "y": y.to_numpy()})
    mun_por_cluster = mun_table.groupby("cluster").size()
    g = d.groupby("cluster").agg(
        n_records=("y", "size"),
        events_cure=("y", lambda v: int((v == _CURA).sum())),
        events_interruption=("y", lambda v: int((v == _INTERRUP).sum())),
        events_death=("y", lambda v: int((v == _OBITO).sum())),
    )
    g["n_municipalities"] = mun_por_cluster
    return g.reset_index()


def fold_summary(folds: pd.Series, record_clusters: pd.Series, y: pd.Series) -> pd.DataFrame:
    """n, eventos por classe e nº de clusters, por dobra (§23.2, TRIPOD+AI 21)."""
    d = pd.DataFrame(
        {"fold": folds.to_numpy(), "cluster": record_clusters.to_numpy(), "y": y.to_numpy()}
    )
    return (
        d.groupby("fold")
        .agg(
            n=("y", "size"),
            n_clusters=("cluster", "nunique"),
            death_pct=("y", lambda v: round(float((v == _OBITO).mean() * 100), 2)),
            events_death=("y", lambda v: int((v == _OBITO).sum())),
        )
        .reset_index()
    )


def clusters_to_geojson(mun_table: pd.DataFrame, clusters: pd.DataFrame, path: Path) -> None:
    """Grava um ponto por município, com cluster, tamanho e código (§23.3)."""
    import geopandas as gpd
    from shapely.geometry import Point

    dados = mun_table.join(clusters)
    gdf = gpd.GeoDataFrame(
        {
            "municipio": dados.index.astype(str),
            "cluster": dados["cluster"].astype(int),
            "n_records": dados["n"].astype(int),
        },
        geometry=[Point(lon, lat) for lon, lat in zip(dados["lon"], dados["lat"])],
        crs="EPSG:4326",
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(path, driver="GeoJSON")
    logger.info("GeoJSON de clusters gravado: %s (%d municípios).", path, len(gdf))
