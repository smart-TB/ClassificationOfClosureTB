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


def _xy():
    X = pd.DataFrame({"c": ["x", "y", "x", "y"], "n": [1.0, 2.0, 3.0, np.nan]})
    return coerce_for_sklearn(X, TYPES, CFG.missing_token)


@pytest.mark.parametrize("profile", PROFILES)
def test_every_profile_fits_and_transforms(profile):
    p = make_preprocessor(profile, TYPES, CFG)
    X = _xy()
    p.fit(X)
    out = p.transform(X)
    assert out.shape[0] == len(X)
    if profile == "native_categorical":
        # nativo mantém as categóricas como tokens (string) para CatBoost/DL;
        # a matriz é mista, mas nenhuma célula pode ser NaN/None.
        assert not pd.DataFrame(out).isna().any().any(), "sem ausência após transform"
    else:
        assert not np.isnan(np.asarray(out, dtype=float)).any(), "sem NaN após transform"


def test_nonnegative_profile_never_negative():
    p = make_preprocessor("nonnegative", TYPES, CFG).fit(_xy())
    out = p.transform(_xy())
    assert (np.asarray(out, dtype=float) >= 0).all()


def test_onehot_scaled_has_more_columns_than_native():
    scaled = make_preprocessor("onehot_scaled", TYPES, CFG).fit_transform(_xy())
    native = make_preprocessor("native_categorical", TYPES, CFG).fit_transform(_xy())
    assert np.asarray(scaled).shape[1] > np.asarray(native).shape[1]
