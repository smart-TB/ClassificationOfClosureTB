from pathlib import Path

import pandas as pd
import pytest

from tb_outcomes.cohort import TargetSchema, load_outcome_rules, map_outcome

RULES = load_outcome_rules(Path("configs/outcomes.yaml"))


def _df(codigos):
    return pd.DataFrame({"SITUA_ENCE": pd.array(codigos, dtype="Int64")})


def test_zero_and_na_both_mean_unresolved():
    # Mudança de codificação do SINAN em 2018: '0' até 2017, vazio depois.
    # Os dois são o mesmo estado; tratá-los diferente inventaria um degrau em 2018.
    r = map_outcome(_df([0, None]), TargetSchema.LEGACY_3CLASS, RULES)
    assert list(r) == ["unresolved_or_censored", "unresolved_or_censored"]


def test_legacy_3class_mapping():
    r = map_outcome(_df([1, 2, 3, 4, 5, 10]), TargetSchema.LEGACY_3CLASS, RULES)
    assert list(r) == [
        "cure",
        "treatment_interruption",
        "tb_attributed_death",
        "excluded",
        "excluded",
        "excluded",
    ]


def test_revised_4class_merges_both_deaths():
    r = map_outcome(_df([3, 4]), TargetSchema.REVISED_4CLASS, RULES)
    assert list(r) == ["death_all_cause", "death_all_cause"]


def test_revised_4class_groups_other_closures():
    r = map_outcome(_df([5, 6, 7, 8, 9, 10]), TargetSchema.REVISED_4CLASS, RULES)
    assert set(r) == {"other_closure"}


def test_actionable_binary_mapping():
    r = map_outcome(_df([1, 2, 3, 4, 9, 10, 5]), TargetSchema.ACTIONABLE_BINARY, RULES)
    assert list(r) == ["no_additional_risk_flag"] + ["enhanced_followup_candidate"] * 5 + [
        "excluded"
    ]


def test_code_6_is_mapped_even_though_absent_from_real_data():
    # Código 6 é válido no dicionário SINAN NET 5.0 e tem zero registros no
    # snapshot. A regra existe; a ausência é fato do dado, não da regra.
    r = map_outcome(_df([6]), TargetSchema.REVISED_4CLASS, RULES)
    assert list(r) == ["other_closure"]


def test_unknown_code_raises():
    with pytest.raises(ValueError, match="99"):
        map_outcome(_df([99]), TargetSchema.LEGACY_3CLASS, RULES)
