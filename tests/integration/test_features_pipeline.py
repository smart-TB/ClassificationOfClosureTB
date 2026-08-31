"""build-features ponta a ponta sobre a fixture sintética."""
import json
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from tb_outcomes.cli import app

pytestmark = pytest.mark.integration
runner = CliRunner()

ARTEFATOS = [
    "feature_availability.csv",
    "feature_dictionary.csv",
    "hiv_rule_audit.csv",
    "outcome_contamination_audit.csv",
    "closure_timing_audit.csv",
    "hiv_availability_by_duration.csv",
    "missingness_overall.csv",
    "missingness_by_outcome.csv",
    "early_death_missingness.csv",
    "collinearity.csv",
    "vif.csv",
]


def test_build_features_produces_every_artifact(projeto_features):
    r = runner.invoke(app, ["build-features", "--config", "configs/analysis_decisions.yaml"])
    assert r.exit_code == 0, r.output
    for nome in ARTEFATOS:
        assert (projeto_features / "data" / nome).exists(), f"faltou {nome}"


def test_no_outcome_or_post_baseline_column_in_the_published_dictionary(projeto_features):
    runner.invoke(app, ["build-features", "--config", "configs/analysis_decisions.yaml"])
    d = pd.read_csv(projeto_features / "data" / "feature_dictionary.csv")
    no_set = d[d.in_notification_set]
    assert set(no_set.availability) == {"at_notification"}


def test_no_feature_in_the_set_derives_from_the_outcome(projeto_features):
    # O defeito da análise original, agora impossível: TEMPO_TRATAMENTO e
    # FLAG_ENCERRA derivavam de DT_ENCERRA e chegaram ao X.
    runner.invoke(app, ["build-features", "--config", "configs/analysis_decisions.yaml"])
    d = pd.read_csv(projeto_features / "data" / "feature_dictionary.csv")
    no_set = d[d.in_notification_set]
    derivacoes = no_set.derived_from.fillna("")
    assert not derivacoes.str.contains("DT_ENCERRA").any()
    assert not derivacoes.str.contains("SITUA_ENCE").any()


def test_vif_never_reports_zero(projeto_features):
    runner.invoke(app, ["build-features", "--config", "configs/analysis_decisions.yaml"])
    v = pd.read_csv(projeto_features / "data" / "vif.csv")
    assert not (v.vif == 0).any(), "VIF 0 é impossível; constante deve ser NA"


def test_artifacts_are_recorded_in_the_run_manifest(projeto_features):
    runner.invoke(app, ["build-features", "--config", "configs/analysis_decisions.yaml"])
    m = json.loads(next((projeto_features / "artifacts").rglob("run_manifest.json")).read_text())
    caminhos = {Path(a["path"]).name for a in m["artifacts"]}
    assert "feature_availability.csv" in caminhos
