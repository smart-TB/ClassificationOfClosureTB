import pandas as pd
import pytest
from typer.testing import CliRunner

from tb_outcomes.cli import app

pytestmark = pytest.mark.integration
runner = CliRunner()


def test_capabilities_matrix_artifact(projeto_features):
    r = runner.invoke(app, ["capabilities-matrix"])
    assert r.exit_code == 0, r.output
    df = pd.read_csv(projeto_features / "data" / "model_capabilities.csv")
    assert len(df) == 28
    # os três defeitos do §2.2/§4b visíveis na matriz gravada
    need_cal = set(df.loc[df["calibrator_required"], "name"])
    assert need_cal == {"ridge_classifier", "linear_svc", "rbf_svm"}
    no_cost = set(df.loc[df["cost_sensitive_status"] == "not_run_incompatible", "name"])
    assert {"lda", "qda", "knn", "mlp"} <= no_cost
