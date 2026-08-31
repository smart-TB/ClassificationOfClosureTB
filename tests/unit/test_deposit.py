"""Montagem do depósito com DOI (ESPECIFICACAO §8, passo 7).

O que estes testes protegem: que material por paciente NUNCA caia no registro aberto, que
a ausência de um artefato obrigatório seja reportada em vez de silenciada, e que o
manifesto permita verificar o que foi depositado.
"""
import json

import pandas as pd
import pytest

from tb_outcomes.deposit import (
    OPEN_LAYOUT,
    OPEN_TREES,
    RESTRICTED_LAYOUT,
    build,
    zenodo_metadata,
)

PADROES_PACIENTE = ("oof_predictions", "shap_values.parquet", "outer_folds",
                    "harmonized", "sinnan")


def test_registro_aberto_nao_referencia_material_por_paciente():
    origens = [o for _, o, _ in OPEN_LAYOUT] + [o for _, o, _, _ in OPEN_TREES]
    for origem in origens:
        for p in PADROES_PACIENTE:
            assert p not in origem, f"{origem} referencia material por paciente"


def test_o_restrito_contem_exatamente_o_que_e_por_paciente():
    origens = [o for _, o, _ in RESTRICTED_LAYOUT]
    assert any("shap_values" in o for o in origens)
    assert any("outer_folds" in o for o in origens)
    # o microdado harmonizado não entra em NENHUM registro: a fonte é pública
    assert not any("harmonized" in o or "sinnan" in o for o in origens)
    # o OOF NÃO é copiado inteiro: é recortado ao par final (ver extract_final_oof)
    assert not any("oof_predictions" in o for o in origens)


def _projeto_falso(tmp_path):
    root = tmp_path / "proj"
    for rel in ["data/shap_values.parquet", "data/outer_folds.parquet"]:
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x,y\n1,2\n", encoding="utf-8")

    # leaderboard mínimo: 'campeao' vence 'perdedor' na métrica global agregada
    linhas = []
    for modelo, valor in (("campeao", 0.60), ("perdedor", 0.30)):
        linhas.append({"kind": "aggregate", "model": modelo, "strategy": "cost",
                       "metric": "f1_macro", "class": "__global__", "mean": valor})
    (root / "data").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(linhas).to_csv(root / "data/benchmark_metrics.csv", index=False)

    # OOF com DOIS pares: só o do campeão pode ser depositado
    oof = pd.concat([
        pd.DataFrame({"record_pos": range(5), "model": m, "strategy": "cost",
                      "y_true": 0, "proba_0": 0.5})
        for m in ("campeao", "perdedor")
    ], ignore_index=True)
    oof.to_parquet(root / "data/oof_predictions.parquet", index=False)

    (root / "artifacts/figures/data").mkdir(parents=True, exist_ok=True)
    (root / "artifacts/figures/01_teste.png").write_bytes(b"\x89PNG fake")
    (root / "artifacts/figures/data/01_teste.csv").write_text("a\n1\n", encoding="utf-8")
    (root / "configs").mkdir(parents=True, exist_ok=True)
    (root / "configs/executor.yaml").write_text("k: 50\n", encoding="utf-8")
    return root


def test_deposita_apenas_o_par_do_modelo_final(tmp_path):
    """O arquivo de origem tem os 70 pares do benchmark; só o campeão vai ao depósito."""
    root = _projeto_falso(tmp_path)
    out = tmp_path / "deposito"
    r = build(root, out, "2027-06-01", {})

    d = pd.read_parquet(out / "restricted" / "oof" / "final_model.parquet")
    assert set(d["model"].unique()) == {"campeao"}
    assert "perdedor" not in set(d["model"])
    assert r["restricted"]["modelo_final"]["model"] == "campeao"
    # e o arquivo com os 70 pares NÃO é copiado inteiro
    assert not (out / "restricted" / "oof" / "main_k50.parquet").exists()


def test_o_modelo_final_vem_do_leaderboard_nao_do_codigo(tmp_path):
    """Trocar quem vence no artefato tem de trocar o que é depositado."""
    from tb_outcomes.deposit import final_model_pair

    root = _projeto_falso(tmp_path)
    assert final_model_pair(root)[0] == "campeao"

    invertido = pd.read_csv(root / "data/benchmark_metrics.csv")
    invertido.loc[invertido.model == "perdedor", "mean"] = 0.99
    invertido.to_csv(root / "data/benchmark_metrics.csv", index=False)
    assert final_model_pair(root)[0] == "perdedor"


def test_build_separa_os_dois_registros(tmp_path):
    root = _projeto_falso(tmp_path)
    out = tmp_path / "deposito"
    r = build(root, out, "2027-06-01", {"available": True, "commit": "abc123"})

    assert (out / "open" / "results/main/benchmark_metrics.csv").exists()
    assert (out / "restricted" / "oof/final_model.parquet").exists()
    # nada por paciente do lado aberto
    abertos = [p.name for p in (out / "open").rglob("*") if p.is_file()]
    assert not any(any(x in n for x in PADROES_PACIENTE) for n in abertos)
    assert r["open"]["n_arquivos"] > 0 and r["restricted"]["n_arquivos"] > 0


def test_manifesto_traz_hash_de_cada_arquivo(tmp_path):
    root = _projeto_falso(tmp_path)
    out = tmp_path / "deposito"
    build(root, out, "2027-06-01", {"available": True})
    m = pd.read_csv(out / "open" / "MANIFEST.csv")
    assert {"arquivo", "origem", "bytes", "sha256"} <= set(m.columns)
    assert m["sha256"].str.len().eq(64).all()
    assert m["arquivo"].is_unique


def test_artefato_obrigatorio_ausente_e_reportado(tmp_path):
    root = _projeto_falso(tmp_path)
    (root / "data/benchmark_metrics.csv").unlink()
    out = tmp_path / "deposito"
    r = build(root, out, "2027-06-01", {"available": False})
    assert "data/benchmark_metrics.csv" in r["open"]["faltando"]


def test_pode_montar_so_o_aberto(tmp_path):
    root = _projeto_falso(tmp_path)
    out = tmp_path / "deposito"
    r = build(root, out, "2027-06-01", {}, include_restricted=False)
    assert not (out / "restricted").exists()
    assert r["restricted"]["n_arquivos"] == 0


def test_build_e_idempotente(tmp_path):
    root = _projeto_falso(tmp_path)
    out = tmp_path / "deposito"
    a = build(root, out, "2027-06-01", {})
    b = build(root, out, "2027-06-01", {})
    assert a["open"]["n_arquivos"] == b["open"]["n_arquivos"]


def test_metadados_do_aberto_tem_embargo_com_data():
    m = zenodo_metadata("2027-06-01", "1.0.0", "https://github.com/x/y", ["Sobrenome, Nome"])
    assert m["access_right"] == "embargoed"
    assert m["embargo_date"] == "2027-06-01"
    assert m["license"] == "cc-by-4.0"   # dados, não MIT (que é licença de software)
    assert m["upload_type"] == "dataset"
    assert m["creators"] == [{"name": "Sobrenome, Nome"}]
    # o CNPq não entra em `grants` (vocabulário OpenAIRE não o cobre) — vai em notes
    assert "grants" not in m
    assert "CNPq" in m["notes"]


def test_metadados_do_restrito_nao_tem_data_de_abertura():
    m = zenodo_metadata("2027-06-01", "1.0.0", "https://github.com/x/y",
                        ["Sobrenome, Nome"], restricted=True)
    assert m["access_right"] == "restricted"
    assert "embargo_date" not in m   # restrito não abre sozinho
    assert "access_conditions" in m


def test_metadados_apontam_para_o_codigo():
    m = zenodo_metadata("2027-06-01", "1.0.0", "https://github.com/smart-TB/x", ["A, B"])
    rel = m["related_identifiers"][0]
    assert rel["relation"] == "isSupplementTo"
    assert "github.com" in rel["identifier"]


def test_readme_do_registro_e_regenerado_pelo_build(tmp_path):
    """O README tem de vir de template: `build` apaga a árvore, então um arquivo escrito
    à mão dentro dela some na montagem seguinte."""
    root = _projeto_falso(tmp_path)
    out = tmp_path / "deposito"
    build(root, out, "2027-06-01", {})
    r1 = (out / "open" / "README.md").read_text(encoding="utf-8")
    assert "no patient-level data" in r1
    build(root, out, "2027-06-01", {})   # segunda montagem
    assert (out / "open" / "README.md").read_text(encoding="utf-8") == r1
    assert "restricted" in (out / "restricted" / "README.md").read_text(encoding="utf-8")


def test_resumo_do_deposito_e_serializavel(tmp_path):
    root = _projeto_falso(tmp_path)
    out = tmp_path / "deposito"
    build(root, out, "2027-06-01", {"available": True, "commit": "abc"})
    with (out / "DEPOSIT_SUMMARY.json").open(encoding="utf-8") as fh:
        d = json.load(fh)
    assert d["embargo_ate"] == "2027-06-01"
    assert d["git"]["commit"] == "abc"
