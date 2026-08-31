"""Manifesto e serialização do preprocess sobre a fixture."""
import json

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from tb_outcomes.cli import app
from tb_outcomes.preprocess import (
    ColumnTypes,
    coerce_for_sklearn,
    load_preprocess_config,
    make_preprocessor,
)

pytestmark = pytest.mark.integration
runner = CliRunner()


def test_preprocess_manifest_is_generated(projeto_features):
    r = runner.invoke(app, ["preprocess-manifest", "--config", "configs/analysis_decisions.yaml"])
    assert r.exit_code == 0, r.output
    m = json.loads((projeto_features / "data" / "preprocessing_manifest.json").read_text())
    assert "profiles" in m and "n_categorical" in m and "n_numeric" in m


def test_fitted_preprocessor_survives_serialization(tmp_path):
    import joblib

    cfg = load_preprocess_config("configs/preprocess.yaml")
    types = ColumnTypes(categorical=["c"], numeric=["n"])
    X = coerce_for_sklearn(
        pd.DataFrame({"c": ["x", "y", "x"], "n": [1.0, 2.0, 3.0]}), types, cfg.missing_token
    )
    p = make_preprocessor("onehot_scaled", types, cfg).fit(X)
    antes = p.transform(X)

    caminho = tmp_path / "prep.joblib"
    joblib.dump(p, caminho)
    depois = joblib.load(caminho).transform(X)
    np.testing.assert_allclose(
        np.asarray(antes, dtype=float), np.asarray(depois, dtype=float)
    )
