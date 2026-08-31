from tb_outcomes.checkpoint import (
    load_pair_checkpoint,
    pair_checkpoint_exists,
    write_pair_checkpoint,
)


def _rows():
    oof = [{"record_pos": 0, "outer_fold": 0, "model": "m", "strategy": "s",
            "y_true": 1, "pred_argmax": 1, "pred_policy": 1,
            "raw_score_0": 0.1, "proba_0": 0.2}]
    metric = [{"model": "m", "strategy": "s", "outer_fold": 0, "metric": "f1_macro",
               "f1_macro": 0.5}]
    manifest = [{"model": "m", "strategy": "s", "outer_fold": 0, "method": "isotonic"}]
    agg = [{"model": "m", "strategy": "s", "metric": "f1_macro", "class": "__global__",
            "stage": None, "axis": None, "mean": 0.5, "n_folds": 5}]
    status = {"model": "m", "strategy": "s", "status": "ok"}
    return oof, metric, manifest, agg, status


def test_exists_false_before_write(tmp_path):
    assert pair_checkpoint_exists(tmp_path, "m", "s") is False


def test_write_then_load_roundtrip(tmp_path):
    oof, metric, manifest, agg, status = _rows()
    write_pair_checkpoint(tmp_path, "m", "s", oof, metric, manifest, agg, status)
    assert pair_checkpoint_exists(tmp_path, "m", "s") is True
    got = load_pair_checkpoint(tmp_path, "m", "s")
    assert got["status_row"] == status
    assert got["metric_rows"] == metric
    assert got["manifest_rows"] == manifest
    assert got["aggregate_rows"] == agg
    assert len(got["oof_rows"]) == 1
    assert got["oof_rows"][0]["record_pos"] == 0
    assert got["oof_rows"][0]["proba_0"] == 0.2


def test_empty_oof_pair_roundtrips(tmp_path):
    # pares not_run_incompatible / error não têm OOF, mas o shard existe
    status = {"model": "m", "strategy": "local_cost_sensitive",
              "status": "not_run_incompatible"}
    write_pair_checkpoint(tmp_path, "m", "local_cost_sensitive", [], [], [], [], status)
    assert pair_checkpoint_exists(tmp_path, "m", "local_cost_sensitive") is True
    got = load_pair_checkpoint(tmp_path, "m", "local_cost_sensitive")
    assert got["oof_rows"] == []
    assert got["status_row"]["status"] == "not_run_incompatible"


def test_meta_written_atomically_last(tmp_path, monkeypatch):
    # se a escrita do meta falhar, exists() deve continuar False (nada de shard meia-boca)
    import tb_outcomes.checkpoint as ck
    oof, metric, manifest, agg, status = _rows()
    orig_replace = ck.Path.replace

    def boom(self, target):
        raise RuntimeError("kill no meio do rename")

    monkeypatch.setattr(ck.Path, "replace", boom)
    try:
        ck.write_pair_checkpoint(tmp_path, "m", "s", oof, metric, manifest, agg, status)
    except RuntimeError:
        pass
    monkeypatch.setattr(ck.Path, "replace", orig_replace)
    assert pair_checkpoint_exists(tmp_path, "m", "s") is False
