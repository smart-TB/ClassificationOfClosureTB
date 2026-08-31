import numpy as np
import pandas as pd
import pytest

from tb_outcomes.splits import (
    build_municipality_table,
    load_splits_config,
    project_to_metric,
)


def test_project_to_metric_returns_km_for_a_known_point():
    # São Paulo (-23.55, -46.63) e Brasília (-15.79, -47.88): ~870 km reais.
    xy = project_to_metric(
        np.array([-23.55, -15.79]), np.array([-46.63, -47.88]), epsg=5880
    )
    dist = np.hypot(*(xy[0] - xy[1]))
    assert 800 < dist < 950, f"distância implausível: {dist:.0f} km"


def test_municipality_table_is_one_row_per_municipality():
    df = pd.DataFrame(
        {
            "ID_MUNIC_ANALISE": pd.array([1, 1, 2], dtype="Int64"),
            "LAT_MUNIC": [-23.5, -23.5, -3.7],
            "LONG_MUNIC": [-46.6, -46.6, -38.5],
        }
    )
    t = build_municipality_table(df)
    assert len(t) == 2
    assert set(t.columns) >= {"lat", "lon", "n"}
    assert t.loc[1, "n"] == 2


def test_municipality_table_rejects_multiple_coordinates_per_municipality():
    # §8.5: coordenada única por município. Se não for, é defeito de dado.
    df = pd.DataFrame(
        {
            "ID_MUNIC_ANALISE": pd.array([1, 1], dtype="Int64"),
            "LAT_MUNIC": [-23.5, -20.0],
            "LONG_MUNIC": [-46.6, -46.6],
        }
    )
    with pytest.raises(ValueError, match="coordenada"):
        build_municipality_table(df)


def test_load_splits_config_reads_real_file():
    cfg = load_splits_config("configs/splits.yaml")
    assert cfg.epsg_metric == 5880
    assert cfg.k_primary == 50
    assert cfg.fold_assignment == "binpack"
