import hashlib

from tb_outcomes.provenance import (
    environment,
    git_state,
    hardware,
    make_run_id,
    sha256_file,
)


def test_sha256_file_matches_hashlib(tmp_path):
    f = tmp_path / "x.bin"
    f.write_bytes(b"conteudo de teste")
    assert sha256_file(f) == hashlib.sha256(b"conteudo de teste").hexdigest()


def test_git_state_degrades_without_repository(tmp_path, monkeypatch):
    # O projeto roda sem git por decisão do PI: registrar a ausência, nunca falhar.
    monkeypatch.chdir(tmp_path)
    estado = git_state()
    assert estado["available"] is False
    assert "reason" in estado


def test_environment_reports_key_libraries():
    env = environment()
    assert "python" in env
    assert "pandas" in env


def test_hardware_reports_cpu_and_ram():
    hw = hardware()
    assert hw["cpu_count"] >= 1
    assert hw["ram_total_gb"] > 0


def test_run_id_marks_absence_of_git(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    rid = make_run_id("revised_primary", "c0ffee123456", "da7a00112233")
    assert rid.startswith("revised_primary_")
    assert "nogit" in rid
    assert "c0ffee123456" in rid
    assert "da7a00112233" in rid
