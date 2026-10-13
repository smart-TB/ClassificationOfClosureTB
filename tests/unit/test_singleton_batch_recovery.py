"""Focused recovery tests; synthetic data never enters scientific artifacts."""
import os
import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, SequentialSampler

from singleton_batch_recovery import safe_loader, install


@pytest.mark.parametrize('n,batch', [(2,1024),(1023,1024),(1024,1024),(1025,1024),
                                     (2049,1024),(3073,1024),(12,4),(13,4)])
def test_records_order_and_batch_sizes(n,batch):
    dataset = list(range(n))
    original = DataLoader(dataset, batch_size=batch, shuffle=False)
    events = []
    patched = safe_loader(original, events.append)
    batches = list(patched)
    assert torch.cat(batches).tolist() == dataset
    assert min(map(len,batches)) >= 2
    assert len(patched) == len(batches)
    if n % batch == 1:
        assert len(events) == 1 and len(batches[-1]) == batch + 1
        assert len(batches) == len(original) - 1
    else:
        assert patched is original and not events


def test_shuffle_and_rng_preserved():
    def get(patch):
        torch.manual_seed(42)
        loader = DataLoader(list(range(2049)), batch_size=1024, shuffle=True)
        if patch: loader = safe_loader(loader)
        order = torch.cat(list(loader)).tolist()
        return order, torch.get_rng_state()
    before, rng1 = get(False)
    after, rng2 = get(True)
    assert before == after and len(set(after)) == 2049
    assert torch.equal(rng1, rng2)


def test_loader_settings_preserved():
    loader = DataLoader(list(range(1025)),batch_size=1024,pin_memory=True)
    patched = safe_loader(loader)
    assert patched.batch_sampler.sampler is loader.batch_sampler.sampler
    for key in ['dataset','generator','collate_fn','pin_memory','num_workers']:
        assert getattr(loader,key) is getattr(patched,key)


@pytest.mark.parametrize('n,drop_last',[(1,False),(1025,True)])
def test_unsupported_cases_fail(n,drop_last):
    with pytest.raises(ValueError):
        safe_loader(DataLoader(list(range(n)),batch_size=1024,drop_last=drop_last))


def test_reproduces_and_removes_batchnorm_singleton_error():
    norm = torch.nn.BatchNorm1d(25).train()
    with pytest.raises(ValueError,match='more than 1 value'):
        norm(torch.randn(1,25))
    for batch in safe_loader(DataLoader(torch.randn(1025,25),batch_size=1024)):
        assert torch.isfinite(norm(batch)).all()


@pytest.mark.skipif(os.environ.get('SMARTTB_RECOVERY_GPU_TEST') != '1', reason='explicit GPU smoke only')
def test_real_adapters_gpu(tmp_path):
    from pytorch_tabular.tabular_datamodule import TabularDatamodule
    from tb_outcomes.dl_config import load_dl_config
    from tb_outcomes.models import load_models_config, build_adapter
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    rng = np.random.default_rng(42)
    # Installed library floors validation size: 1205-int(1205*.15)=1025.
    frame = pd.DataFrame(rng.normal(size=(1205,25)),columns=[f'x{i}' for i in range(25)])
    frame['category'] = np.resize(['a','b','c'],1205)
    target = np.resize(np.array([0,1,2]),1205)
    cfg = load_dl_config('configs/deep_learning.yaml').model_copy(update={
        'max_epochs':1,'early_stopping_patience':1,'torch_num_threads':2})
    model_cfg = load_models_config('configs/models.yaml')
    entries = {**model_cfg.baselines,**model_cfg.models}
    validation = TabularDatamodule.val_dataloader
    events = []
    original = install(events.append)
    try:
        # GPUFanLease is owned by the launcher process; the CUDA child must exit
        # before the launcher can safely restore fan targets to 30%.
        for family in ['ft_transformer','tabnet']:
            count = len(events)
            model = build_adapter(family,entries[family],cfg)
            try:
                model.fit(frame,target)
                probs = model.predict_proba_if_native(frame.iloc[:17])
                assert probs.shape == (17,3) and np.isfinite(probs).all()
                np.testing.assert_allclose(probs.sum(axis=1),1,atol=1e-5)
                assert len(events)>count
                assert events[-1]['n_train']==1025
                assert events[-1]['final_batch_size']==1025
            finally: model.close()
        assert TabularDatamodule.val_dataloader is validation
    finally:
        TabularDatamodule.train_dataloader = original
