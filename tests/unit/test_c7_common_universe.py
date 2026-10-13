import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('c7_compare', Path('scripts/compare_c7_arms.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def records():
    return [dict(model=m, strategy='none', outer_fold=f, candidate='0', calibration='sigmoid',
                 policy='argmax', argmax_f1_macro=score, policy_f1_macro=score,
                 external_score=1-score)
            for f in range(2) for m, score in [('a', .4), ('b', .6), ('outside', .9)]]


def test_common_universe_ignores_better_undeclared_and_external_scores():
    pairs = {('a', 'none'), ('b', 'none')}
    rows = module.restrict_records(records(), pairs)
    winners = module.choose_inner_winners(rows)
    assert winners.model.tolist() == ['b', 'b']
    for row in rows:
        row['external_score'] = 0 if row['model'] == 'b' else 1
    assert module.choose_inner_winners(rows).model.tolist() == ['b', 'b']


def test_missing_common_candidate_fold_fails():
    rows = records()
    rows = [r for r in rows if not (r['model'] == 'b' and r['outer_fold'] == 1)]
    with pytest.raises(ValueError, match='Incomplete'):
        module.restrict_records(rows, {('a', 'none'), ('b', 'none')})
