import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location('c7_shap', Path('scripts/c7_selected_attribution.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize('name', ['lightgbm', 'catboost'])
def test_native_multiclass_categorical_additivity(name):
    rng = np.random.default_rng(42)
    frame = pd.DataFrame({'numeric': rng.normal(size=120),
                          'category': pd.Categorical(np.tile(['a', 'b', 'c'], 40))})
    y = np.tile([0, 1, 2], 40)
    if name == 'lightgbm':
        from lightgbm import LGBMClassifier
        model = LGBMClassifier(n_estimators=10, n_jobs=2, verbosity=-1, random_state=42)
        model.fit(frame, y)
    else:
        from catboost import CatBoostClassifier
        model = CatBoostClassifier(iterations=10, depth=3, thread_count=2, verbose=False,
                                   allow_writing_files=False, random_seed=42)
        model.fit(frame, y, cat_features=['category'])
    values, error = module.native_contributions(model, name, frame.iloc[:12], 3)
    assert values.shape == (12, 3, 3)
    assert np.isfinite(values).all()
    assert error < 1e-7
