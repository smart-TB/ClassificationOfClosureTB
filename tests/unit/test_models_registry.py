from tb_outcomes.models import build_registry, capabilities_matrix, load_models_config

CFG = load_models_config("configs/models.yaml")
REG = build_registry(CFG)
MAT = capabilities_matrix(REG)


def test_registry_has_all_28():
    assert len(REG) == 28


def test_matrix_one_row_per_model():
    assert len(MAT) == 28
    assert set(MAT["name"]) == set(REG)


def test_matrix_marks_cost_incompatible_explicitly():
    # LDA/QDA/kNN/MLP: cost aparece como not_run_incompatible, nunca ausência muda
    for name in ("lda", "qda", "knn", "mlp"):
        row = MAT.loc[MAT["name"] == name].iloc[0]
        assert "not_run_incompatible" in row["cost_sensitive_status"]


def test_matrix_has_no_silent_empty_cells():
    assert not MAT.isna().any().any()


def test_calibrator_required_only_for_the_three():
    need = set(MAT.loc[MAT["calibrator_required"], "name"])
    assert need == {"ridge_classifier", "linear_svc", "rbf_svm"}
