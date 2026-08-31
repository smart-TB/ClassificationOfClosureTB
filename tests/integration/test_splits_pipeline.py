"""build-splits ponta a ponta sobre a fixture sintética."""
import json

import pandas as pd
import pytest
from typer.testing import CliRunner

from tb_outcomes.cli import app

pytestmark = pytest.mark.integration
runner = CliRunner()


def test_build_splits_produces_artifacts(projeto_features):
    # k reduzido: a fixture tem 5 municípios, não dá para k=50.
    r = runner.invoke(
        app, ["build-splits", "--config", "configs/analysis_decisions.yaml", "--k", "3"]
    )
    assert r.exit_code == 0, r.output
    for nome in [
        "municipality_clusters_k3.parquet",
        "outer_folds.parquet",
        "fold_summary.csv",
        "cluster_summary.csv",
        "leakage_checks.json",
    ]:
        assert (projeto_features / "data" / nome).exists(), f"faltou {nome}"


def test_leakage_checks_all_pass(projeto_features):
    runner.invoke(app, ["build-splits", "--config", "configs/analysis_decisions.yaml", "--k", "3"])
    checks = json.loads((projeto_features / "data" / "leakage_checks.json").read_text())
    assert all(checks.values()), f"vazamento detectado: {checks}"


def test_no_cluster_spans_two_folds(projeto_features):
    runner.invoke(app, ["build-splits", "--config", "configs/analysis_decisions.yaml", "--k", "3"])
    folds = pd.read_parquet(projeto_features / "data" / "outer_folds.parquet")
    assert (folds.groupby("cluster").outer_fold.nunique() == 1).all()
