"""Inventário e hashes dos arquivos editoriais sob responsabilidade do Claude (Parte B).

O contrato de execução conjunta exige hash inicial, backup do que for editado e, na
entrega, o diff contra o backup — e não contra o HEAD do Git, porque há alterações prévias
não commitadas que precisam ser preservadas.

Este utilitário serve às duas pontas: gera o inventário inicial (`--backup`) e, ao final,
o inventário de conferência para comparar hashes e listar o que mudou.

Uso:
    python scripts/editorial/editorial_inventory.py --out docs/.../inventario_inicial.json --backup
    python scripts/editorial/editorial_inventory.py --out docs/.../inventario_final.json \
        --compare docs/.../inventario_inicial.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
BACKUP_DIR = RAIZ / "docs/manuscrito/colaboracao/claude/backups"

# Escopo exato do contrato, Parte B. Nada fora daqui é de propriedade do Claude.
GLOBS_EDITORIAIS = [
    "manuscrito/PLOS_DigitalHealth/*.tex",
    "manuscrito/PLOS_DigitalHealth/*.bib",
    "manuscrito/PLOS_DigitalHealth/*.bst",
]
ARQUIVOS_EDITORIAIS = [
    "src/tb_outcomes/figures.py",
    "src/tb_outcomes/labels.py",
    "scripts/inject_manuscript_tables.py",
    "scripts/inject_ptbr_tables.py",
    "scripts/inject_manuscript_figures.py",
    "scripts/build_supplementary.py",
    "scripts/build_baseline_table.py",
    "scripts/build_model_table.py",
    "scripts/build_attribution_vs_ablation.py",
    "scripts/export_plos_figures.py",
    "scripts/check_checklist_pages.py",
    "scripts/build_results_dossier.py",
]


def sha256(caminho: Path) -> str:
    h = hashlib.sha256()
    with caminho.open("rb") as fh:
        for bloco in iter(lambda: fh.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def alvos() -> list[Path]:
    encontrados: list[Path] = []
    for rel in ARQUIVOS_EDITORIAIS:
        p = RAIZ / rel
        if p.is_file():
            encontrados.append(p)
    for padrao in GLOBS_EDITORIAIS:
        encontrados.extend(sorted(RAIZ.glob(padrao)))
    # dedup preservando ordem
    vistos, saida = set(), []
    for p in encontrados:
        if p not in vistos:
            vistos.add(p)
            saida.append(p)
    return saida


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--backup", action="store_true",
                    help="copia cada arquivo para claude/backups/ preservando o caminho relativo")
    ap.add_argument("--compare", type=Path, default=None,
                    help="inventário anterior; lista o que mudou")
    args = ap.parse_args()

    registros = []
    for p in alvos():
        rel = p.relative_to(RAIZ).as_posix()
        h = sha256(p)
        registros.append({"path": rel, "sha256": h, "bytes": p.stat().st_size})
        if args.backup:
            destino = BACKUP_DIR / rel
            destino.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, destino)

    doc = {
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "root": str(RAIZ),
        "n_files": len(registros),
        "files": registros,
    }

    if args.compare and args.compare.is_file():
        antes = {r["path"]: r["sha256"] for r in json.loads(
            args.compare.read_text(encoding="utf-8"))["files"]}
        mudou, novos = [], []
        for r in registros:
            if r["path"] not in antes:
                novos.append(r["path"])
            elif antes[r["path"]] != r["sha256"]:
                mudou.append(r["path"])
        sumidos = sorted(set(antes) - {r["path"] for r in registros})
        doc["diff_vs_baseline"] = {"changed": mudou, "added": novos, "removed": sumidos}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"{len(registros)} arquivos editoriais inventariados -> {args.out}")
    if args.backup:
        print(f"backup em {BACKUP_DIR.relative_to(RAIZ)}")
    if "diff_vs_baseline" in doc:
        d = doc["diff_vs_baseline"]
        print(f"alterados: {len(d['changed'])} | novos: {len(d['added'])} | removidos: {len(d['removed'])}")
        for c in d["changed"]:
            print(f"  M {c}")
        for c in d["added"]:
            print(f"  + {c}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
