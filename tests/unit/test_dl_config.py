import pytest

from tb_outcomes.dl_config import DLConfig, load_dl_config


def test_loads_pre_registered_budget():
    cfg = load_dl_config("configs/deep_learning.yaml")
    assert cfg.max_epochs == 50
    assert 0.0 < cfg.validation_split < 0.5
    assert cfg.early_stopping_patience > 0


def test_rejects_bad_type():
    with pytest.raises(Exception):
        DLConfig(max_epochs="cinquenta", early_stopping_patience=8,
                 validation_split=0.15, batch_size=1024, seed=42)


def test_dlconfig_num_workers_default_zero():
    cfg = DLConfig(max_epochs=2, early_stopping_patience=2, validation_split=0.2,
                   batch_size=256, seed=0)
    assert cfg.num_workers == 0


def test_load_dl_config_reads_num_workers(tmp_path):
    p = tmp_path / "dl.yaml"
    p.write_text(
        "max_epochs: 50\nearly_stopping_patience: 8\nvalidation_split: 0.15\n"
        "batch_size: 1024\nseed: 42\nnum_workers: 4\n",
        encoding="utf-8",
    )
    cfg = load_dl_config(p)
    assert cfg.num_workers == 4


def test_production_dl_yaml_has_num_workers():
    cfg = load_dl_config("configs/deep_learning.yaml")
    # campo presente e não-negativo. Fica em 0: num_workers>0 deadlocka o DataLoader
    # sob CUDA neste executor (ver deep_learning.yaml); 0 é o regime seguro do k=50.
    assert cfg.num_workers >= 0
    assert isinstance(cfg.num_workers, int)
