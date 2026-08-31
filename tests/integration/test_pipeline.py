"""Pipeline ponta a ponta sobre a fixture sintética.

Nenhum dado real: é o que permite ao revisor rodar tudo sem microdado.
"""
import json

import pandas as pd
import pytest
from typer.testing import CliRunner

from tb_outcomes.cli import app

pytestmark = pytest.mark.integration
runner = CliRunner()

def test_validate_config_passes_on_fields_not_marked_tbd(projeto):
    r = runner.invoke(app, ["validate-config", "--config", "configs/analysis_decisions.yaml"])
    assert r.exit_code == 0, r.output
    assert "config_hash" in r.output


def test_build_cohort_produces_every_required_artifact(projeto):
    r = runner.invoke(app, ["build-cohort", "--config", "configs/analysis_decisions.yaml"])
    assert r.exit_code == 0, r.output

    for nome in [
        "raw_manifest.csv",
        "data_quality_report.json",
        "data_quality_report.md",
        "auditoria_municipio_analise.csv",
        "duplicate_audit.csv",
        "followup_audit.csv",
    ]:
        assert (projeto / "data" / nome).exists(), f"faltou {nome}"

    for schema in ["legacy_3class", "revised_4class", "actionable_binary"]:
        assert (projeto / "data" / f"cohort_flow_{schema}.csv").exists()
        assert (projeto / "data" / f"outcome_distribution_{schema}.csv").exists()


def test_run_manifest_lists_artifacts_with_hashes(projeto):
    runner.invoke(app, ["build-cohort", "--config", "configs/analysis_decisions.yaml"])
    manifestos = list((projeto / "artifacts").rglob("run_manifest.json"))
    assert len(manifestos) == 1
    conteudo = json.loads(manifestos[0].read_text(encoding="utf-8"))
    assert conteudo["artifacts"], "manifesto vazio"
    assert all(len(a["sha256"]) == 64 for a in conteudo["artifacts"])


def test_git_absence_is_recorded_not_hidden(projeto):
    runner.invoke(app, ["build-cohort", "--config", "configs/analysis_decisions.yaml"])
    estado = json.loads(next((projeto / "artifacts").rglob("git_state.json")).read_text())
    assert estado["available"] is False


def test_same_seed_gives_identical_cohort(projeto):
    runner.invoke(app, ["build-cohort", "--config", "configs/analysis_decisions.yaml"])
    a = pd.read_csv(projeto / "data" / "cohort_flow_legacy_3class.csv")
    runner.invoke(app, ["build-cohort", "--config", "configs/analysis_decisions.yaml"])
    b = pd.read_csv(projeto / "data" / "cohort_flow_legacy_3class.csv")
    pd.testing.assert_frame_equal(a, b)


def test_missing_config_fails_explicitly(projeto):
    r = runner.invoke(app, ["build-cohort", "--config", "configs/inexistente.yaml"])
    assert r.exit_code != 0
