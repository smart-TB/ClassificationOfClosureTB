from tb_outcomes.executor import load_executor_config


def test_executor_config_has_sweep():
    cfg = load_executor_config("configs/executor.yaml")
    assert cfg.sweep is not None
    assert cfg.sweep.ks == [27, 75]
    assert cfg.sweep.out_root == "data/sweep"
    assert "lightgbm" in cfg.sweep.models
    assert "majority_class" in cfg.sweep.models
    # 11 contendores: gradient_boosting cortado (2026-07-27) — redundante com
    # hist_gradient_boost, que o supera no k=50; hist permanece no sweep.
    assert len(cfg.sweep.models) == 11
    assert "gradient_boosting" not in cfg.sweep.models
    assert "hist_gradient_boost" in cfg.sweep.models
    # o k=50 NÃO está no sweep (é o braço de referência)
    assert 50 not in cfg.sweep.ks
