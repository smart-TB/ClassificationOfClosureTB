import json

import pandas as pd

from tb_outcomes.splits import clusters_to_geojson


def test_geojson_has_one_point_per_municipality(tmp_path):
    mun = pd.DataFrame(
        {"lat": [-23.5, -3.7, -15.8], "lon": [-46.6, -38.5, -47.9], "n": [100, 50, 20]},
        index=pd.Index([1, 2, 3], name="ID_MUNIC_ANALISE"),
    )
    clusters = pd.DataFrame({"cluster": [0, 1, 0]}, index=mun.index)
    destino = tmp_path / "clusters.geojson"
    clusters_to_geojson(mun, clusters, destino)

    gj = json.loads(destino.read_text(encoding="utf-8"))
    assert gj["type"] == "FeatureCollection"
    assert len(gj["features"]) == 3
    f0 = gj["features"][0]
    assert f0["geometry"]["type"] == "Point"
    assert "cluster" in f0["properties"]
