import json

import pytest

from tb_outcomes.checkpoint import load_pair_checkpoint, pair_checkpoint_exists, write_pair_checkpoint
from tb_outcomes.run_identity import bind_identity, identity_hash


def test_resumption_requires_identical_source_config_and_partitions(tmp_path):
    manifest = {"source": "abc", "features": ["age"], "folds": "123"}
    bind_identity(tmp_path, manifest)
    bind_identity(tmp_path, dict(reversed(list(manifest.items()))))
    with pytest.raises(ValueError, match="fingerprint changed"):
        bind_identity(tmp_path, {**manifest, "folds": "124"})
    assert identity_hash(manifest) != identity_hash({**manifest, "features": ["age", "HIV"]})


def test_unidentified_old_outputs_are_rejected(tmp_path):
    (tmp_path / "old.parquet").write_text("old")
    with pytest.raises(ValueError, match="unidentified"):
        bind_identity(tmp_path, {})


def test_missing_or_changed_prediction_file_cannot_resume(tmp_path):
    write_pair_checkpoint(tmp_path, "m", "s", [{"record_pos": 0}], [], [], [], {"status": "ok"})
    p = tmp_path / "m__s.parquet"
    p.write_bytes(b"corrupted")
    assert not pair_checkpoint_exists(tmp_path, "m", "s")
    with pytest.raises(ValueError, match="corrupted"):
        load_pair_checkpoint(tmp_path, "m", "s")
    p.unlink()
    assert not pair_checkpoint_exists(tmp_path, "m", "s")


def test_historical_metadata_and_failed_pairs_are_not_successes(tmp_path):
    (tmp_path / "m__s.meta.json").write_text(json.dumps({"status_row": {"status": "ok"}}))
    assert not pair_checkpoint_exists(tmp_path, "m", "s")
    write_pair_checkpoint(tmp_path, "m", "s", [], [], [], [], {"status": "error"})
    assert not pair_checkpoint_exists(tmp_path, "m", "s")
