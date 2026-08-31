from tb_outcomes.models import load_models_config

CFG = load_models_config("configs/models.yaml")


def test_registry_has_25_models_and_3_baselines():
    assert len(CFG.baselines) == 3
    assert len(CFG.models) == 25


def test_no_class_weight_anywhere():
    # desbalanceamento é externo (SP4b); class_weight não pode vazar pro modelo
    import yaml

    raw = yaml.safe_load(open("configs/models.yaml"))
    assert "class_weight" not in str(raw)


def test_every_profile_is_a_valid_sp4a_profile():
    valid = {"onehot_scaled", "onehot_unscaled", "native_categorical", "nonnegative", "binary"}
    for entry in {**CFG.baselines, **CFG.models}.values():
        assert entry.preprocess_profile in valid


def test_nonnegative_models_are_the_two_nb():
    nonneg = {n for n, e in CFG.models.items() if e.preprocess_profile == "nonnegative"}
    assert nonneg == {"multinomial_nb", "complement_nb"}
