"""Valida o `release_manifest.json` antes de qualquer número entrar no manuscrito.

O contrato de execução conjunta exige que cada artefato liberado declare caminho, SHA-256,
braço, população, regra de decisão e agregação. Essa exigência existe porque boa parte das
divergências apuradas na revisão veio exatamente daí: a Tabela 1 usava `pred_argmax` e os
estratos de completude usavam `pred_policy`, sem que o texto dissesse, e as horas de
computação se referiam a braços auxiliares e não ao benchmark principal.

Este verificador roda ANTES do L5. Enquanto o status for `not_ready`, ele reporta e sai com
zero — não é erro, é o estado esperado. Quando estiver liberado, ele falha alto se faltar
qualquer campo, se um arquivo não existir, se o hash divergir ou se os artefatos não
compartilharem o mesmo `run_id`.

Uso:
    python scripts/editorial/check_release_manifest.py
    python scripts/editorial/check_release_manifest.py --json   # saída para outro programa
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
MANIFESTO = RAIZ / "docs/manuscrito/colaboracao/release_manifest.json"

# Campos que o contrato exige no topo do manifesto.
CAMPOS_TOPO = ("schema_version", "status", "owner", "run_id", "artifacts", "class_labels")
# Campos que o contrato exige em CADA artefato. A ausência de qualquer um deles impede
# rastrear um número até a sua origem, que é o objetivo do L5.
CAMPOS_ARTEFATO = ("path", "sha256", "arm", "population", "decision_rule", "aggregation")

PRONTO = "ready_for_reporting"


def sha256(caminho: Path) -> str:
    h = hashlib.sha256()
    with caminho.open("rb") as fh:
        for bloco in iter(lambda: fh.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def validar(doc: dict) -> tuple[list[str], list[str]]:
    """Devolve (erros, avisos). Erros impedem o L5; avisos apenas registram."""
    erros: list[str] = []
    avisos: list[str] = []

    faltando = [c for c in CAMPOS_TOPO if c not in doc]
    if faltando:
        erros.append(f"campos ausentes no topo: {faltando}")

    if doc.get("status") != PRONTO:
        return erros, avisos  # nada mais a exigir enquanto não liberado

    if not doc.get("run_id"):
        erros.append("status é ready_for_reporting mas run_id está vazio")

    artefatos = doc.get("artifacts") or []
    if not artefatos:
        erros.append("status é ready_for_reporting mas a lista de artefatos está vazia")

    run_ids = set()
    for i, a in enumerate(artefatos):
        rotulo = a.get("path", f"artefato[{i}]")
        ausentes = [c for c in CAMPOS_ARTEFATO if not a.get(c)]
        if ausentes:
            erros.append(f"{rotulo}: campos ausentes {ausentes}")
            continue

        if "run_id" in a:
            run_ids.add(a["run_id"])

        caminho = RAIZ / a["path"]
        if not caminho.is_file():
            erros.append(f"{rotulo}: arquivo não existe")
            continue

        real = sha256(caminho)
        if real != a["sha256"]:
            erros.append(f"{rotulo}: SHA-256 diverge\n"
                         f"      manifesto {a['sha256']}\n"
                         f"      arquivo   {real}")

    if len(run_ids) > 1:
        erros.append(f"artefatos de run_id diferentes no mesmo manifesto: {sorted(run_ids)}")
    if run_ids and doc.get("run_id") and run_ids != {doc["run_id"]}:
        erros.append(f"run_id do topo ({doc['run_id']}) não bate com o dos artefatos {sorted(run_ids)}")

    if not doc.get("limitations"):
        avisos.append("manifesto sem campo `limitations`; o contrato prevê declará-las")

    return erros, avisos


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true", help="saída estruturada")
    args = ap.parse_args()

    if not MANIFESTO.is_file():
        print(f"manifesto não encontrado: {MANIFESTO}")
        return 2

    doc = json.loads(MANIFESTO.read_text(encoding="utf-8"))
    erros, avisos = validar(doc)
    status = doc.get("status")
    liberado = status == PRONTO and not erros

    if args.json:
        print(json.dumps({"status": status, "liberado": liberado,
                          "run_id": doc.get("run_id"),
                          "n_artefatos": len(doc.get("artifacts") or []),
                          "erros": erros, "avisos": avisos}, ensure_ascii=False, indent=2))
        return 0 if liberado or status != PRONTO else 1

    print(f"status: {status}")
    print(f"run_id: {doc.get('run_id')}")
    print(f"artefatos: {len(doc.get('artifacts') or [])}")

    if status != PRONTO:
        print("\nAINDA NÃO LIBERADO — L5 e L6 permanecem bloqueados.")
        print("Estado esperado enquanto o Codex não conclui C6/C7.")
        for a in avisos:
            print(f"  aviso: {a}")
        return 0

    if erros:
        print(f"\n{len(erros)} PROBLEMA(S) — não usar estes números:")
        for e in erros:
            print(f"  - {e}")
        return 1

    print("\nliberado e íntegro: caminhos existem, hashes conferem, run_id único,")
    print("e cada artefato declara braço, população, regra de decisão e agregação.")
    for a in avisos:
        print(f"  aviso: {a}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
