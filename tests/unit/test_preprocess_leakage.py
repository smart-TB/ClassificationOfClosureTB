import numpy as np
import pandas as pd
import pytest

from tb_outcomes.preprocess import (
    PROFILES,
    ColumnTypes,
    coerce_for_sklearn,
    load_preprocess_config,
    make_preprocessor,
)

CFG = load_preprocess_config("configs/preprocess.yaml")
TYPES = ColumnTypes(categorical=["c"], numeric=["n"])


def _coerce(df):
    return coerce_for_sklearn(df, TYPES, CFG.missing_token)


@pytest.mark.parametrize("profile", PROFILES)
def test_transform_of_a_probe_row_is_independent_of_the_batch(profile):
    treino = _coerce(pd.DataFrame({"c": ["x", "y", "x", "y"], "n": [1.0, 2.0, 3.0, 4.0]}))
    p = make_preprocessor(profile, TYPES, CFG).fit(treino)

    sonda = _coerce(pd.DataFrame({"c": ["x"], "n": [2.5]}))
    sozinha = p.transform(sonda)

    # a MESMA sonda dentro de um lote com estatística deslocada de propósito:
    # média e categorias absurdas fariam qualquer refit-no-transform mudar o resultado.
    lote = _coerce(pd.DataFrame({"c": ["x", "z", "z", "z"], "n": [2.5, 1e9, 1e9, -1e9]}))
    dentro = p.transform(lote)[0]

    # a linha-sonda deve sair idêntica sozinha ou no lote — o transform é
    # independente de linha (vale para saída numérica e para tokens string).
    np.testing.assert_array_equal(
        np.asarray(sozinha[0], dtype=object),
        np.asarray(dentro, dtype=object),
        err_msg=f"{profile}: transform da sonda mudou com o lote — vazamento no transform",
    )


def test_scaler_uses_train_statistics_not_eval():
    p = make_preprocessor("onehot_scaled", TYPES, CFG)
    treino = _coerce(pd.DataFrame({"c": ["x", "y"], "n": [10.0, 20.0]}))
    p.fit(treino)
    num = p.named_transformers_["num"]
    scaler = num.named_steps["union"].transformer_list[0][1].named_steps["scale"]
    assert abs(scaler.mean_[0] - 15.0) < 1e-9, "média do scaler é a do treino (15)"
