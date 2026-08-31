"""Auditoria de desbalanceamento sobre a fixture."""
import json

import pandas as pd
import pytest
from typer.testing import CliRunner

from tb_outcomes.cli import app
from tb_outcomes.imbalance import build_strata

pytestmark = pytest.mark.integration
runner = CliRunner()


def test_build_strata_combines_year_and_region():
    df = pd.DataFrame(
        {"NU_ANO": pd.array([2019, 2020], dtype="Int64"), "REGIAO": ["Sudeste", "Norte"]}
    )
    s = build_strata(df, "year_region")
    assert s.iloc[0] == "2019_Sudeste"
    assert s.nunique() == 2


def test_imbalance_audit_artifact_is_generated(projeto_features):
    r = runner.invoke(app, ["imbalance-audit", "--config", "configs/analysis_decisions.yaml"])
    assert r.exit_code == 0, r.output
    a = json.loads((projeto_features / "data" / "imbalance_audit.json").read_text())
    for st in (
        "none",
        "random_oversampling",
        "random_undersampling",
        "local_cost_sensitive",
    ):
        assert st in a
    assert a["random_oversampling"]["n_after"] > a["none"]["n_after"]
