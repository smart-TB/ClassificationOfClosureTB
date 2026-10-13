import pandas as pd
import pytest

from tb_outcomes.features import build_notification_feature_set, load_feature_config


def test_sensitivity_profiles_do_not_change_primary_or_reintroduce_other_postbaseline_fields():
    specs = load_feature_config("configs/features.yaml")
    df = pd.DataFrame({"HIV": [1, 3], "AGRAVAIDS": [2, 1], "BACILOSC_E": [1, 2],
                       "BACILOSC_O": [1, 2], "TEST_MOLEC": [1, 2], "CULTURA_ES": [1, 2],
                       "SITUA_ENCE": [1, 3], "ANT_RETRO": [1, 2],
                       "NU_ANO": [2019, 2020], "ANO_NASC": [1980, 1990]}, index=[17, 83])
    # No IPEA feature in this fixture; their vintage gate is tested separately.
    specs = [s for s in specs if not s.raw_name.startswith("IPEA_")]
    primary = build_notification_feature_set(df, specs)
    hiv = build_notification_feature_set(df, specs, availability_profile="hiv_uncertain")
    strict = build_notification_feature_set(df, specs, availability_profile="without_diagnostic_tests")
    assert "HIV" not in primary and "hiv_pos_model" not in primary
    assert {"HIV", "hiv_pos_model"} <= set(hiv)
    assert hiv.HIV.tolist() == [1, 4]
    assert hiv.hiv_pos_model.tolist() == [1, 1]
    assert set(primary) - set(strict) == {"BACILOSC_E", "BACILOSC_O", "TEST_MOLEC"}
    for frame in [primary, hiv, strict]:
        assert frame.index.equals(df.index)
        assert not {"SITUA_ENCE", "CULTURA_ES", "ANT_RETRO"} & set(frame)
    pd.testing.assert_frame_equal(primary, build_notification_feature_set(df, specs))


def test_unrecognized_availability_profile_fails():
    with pytest.raises(ValueError, match="availability profile"):
        build_notification_feature_set(pd.DataFrame(), [], availability_profile="include_everything")
