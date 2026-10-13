import hashlib

import pytest

from tb_outcomes.dl_parallel import freeze_runtime


def test_runtime_remains_fixed_when_working_repository_changes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "src/tb_outcomes/example.py"
    source.parent.mkdir(parents=True)
    source.write_text("x = 1\n")
    signature = {"src/tb_outcomes/example.py": hashlib.sha256(source.read_bytes()).hexdigest()}
    runtime = freeze_runtime(tmp_path / "run", signature)
    source.write_text("x = 2\n")
    assert (runtime / "src/tb_outcomes/example.py").read_text() == "x = 1\n"
    assert freeze_runtime(tmp_path / "run", signature) == runtime


def test_source_race_and_frozen_corruption_are_rejected(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "model.py"
    source.write_text("a")
    signature = {"model.py": hashlib.sha256(b"a").hexdigest()}
    with pytest.raises(RuntimeError, match="Source changed"):
        freeze_runtime(tmp_path / "run", {"model.py": "wrong"})
    runtime = freeze_runtime(tmp_path / "run", signature)
    (runtime / "model.py").write_text("corrupt")
    with pytest.raises(RuntimeError, match="Frozen runtime changed"):
        freeze_runtime(tmp_path / "run", signature)
