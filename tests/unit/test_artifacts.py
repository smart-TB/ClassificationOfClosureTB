import json

import pandas as pd
import pytest

from tb_outcomes.artifacts import RunManifest, atomic_write_text, write_csv, write_json
from tb_outcomes.provenance import sha256_file


def test_atomic_write_creates_file_and_leaves_no_tmp(tmp_path):
    destino = tmp_path / "sub" / "a.txt"
    atomic_write_text(destino, "olá")
    assert destino.read_text(encoding="utf-8") == "olá"
    assert list(tmp_path.rglob("*.tmp")) == []


def test_atomic_write_does_not_clobber_on_failure(tmp_path):
    destino = tmp_path / "a.txt"
    atomic_write_text(destino, "original")
    with pytest.raises(TypeError):
        atomic_write_text(destino, None)  # type: ignore[arg-type]
    assert destino.read_text(encoding="utf-8") == "original"


def test_write_csv_roundtrip(tmp_path):
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    destino = tmp_path / "t.csv"
    write_csv(df, destino)
    pd.testing.assert_frame_equal(pd.read_csv(destino), df)


def test_run_manifest_records_hash_size_and_step(tmp_path):
    alvo = tmp_path / "art.json"
    write_json({"k": 1}, alvo)
    m = RunManifest(run_id="teste_run")
    m.record(alvo, step="build-cohort")
    destino = tmp_path / "run_manifest.json"
    m.save(destino)
    conteudo = json.loads(destino.read_text(encoding="utf-8"))
    assert conteudo["run_id"] == "teste_run"
    entrada = conteudo["artifacts"][0]
    assert entrada["sha256"] == sha256_file(alvo)
    assert entrada["step"] == "build-cohort"
    assert entrada["size_bytes"] == alvo.stat().st_size


def test_run_manifest_refuses_missing_artifact(tmp_path):
    m = RunManifest(run_id="teste_run")
    with pytest.raises(FileNotFoundError):
        m.record(tmp_path / "inexistente.csv", step="build-cohort")
